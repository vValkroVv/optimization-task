from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .config import S3DatabaseConfig, silence_insecure_request_warnings
from .query import SQLiteError, make_requests_session, sqlite_s3_query
from .schema import DatabaseType, ensure_schema
from .sqlite_file import normalize_sqlite_file

logger = logging.getLogger(__name__)


def _build_s3_client(config: S3DatabaseConfig):
    import boto3
    from botocore.config import Config as BotoConfig

    silence_insecure_request_warnings(config.verify_ssl)

    access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    session_token = os.environ.get("AWS_SESSION_TOKEN")

    config_kwargs: dict[str, Any] = {
        "signature_version": "s3v4",
        "connect_timeout": 5,
        "read_timeout": 30,
        "retries": {"max_attempts": 2, "mode": "standard"},
        "s3": {
            "addressing_style": "path" if config.force_path_style else "virtual",
            "payload_signing_enabled": False,
        },
    }
    optional_config_kwargs: dict[str, Any] = {
        "request_checksum_calculation": "when_required",
        "response_checksum_validation": "when_required",
    }
    try:
        boto_config = BotoConfig(**config_kwargs, **optional_config_kwargs)
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
        # Older botocore versions do not support checksum config fields.
        boto_config = BotoConfig(**config_kwargs)
        logger.warning(
            "botocore Config checksum options are unsupported; "
            "falling back to legacy S3 client config: %s",
            exc,
        )

    kwargs: dict[str, Any] = {
        "service_name": "s3",
        "region_name": config.region,
        "endpoint_url": config.endpoint_url,
        "verify": config.verify_ssl,
        "config": boto_config,
    }
    if access_key and secret_key:
        kwargs["aws_access_key_id"] = access_key
        kwargs["aws_secret_access_key"] = secret_key
        if session_token:
            kwargs["aws_session_token"] = session_token

    return boto3.client(**kwargs)


def _configure_journal_mode(conn: sqlite3.Connection) -> None:
    """Enable WAL, falling back to an NFS-safe rollback journal.

    WAL needs a shared-memory index (``-shm``) that some network filesystems
    (notably NFS) cannot lock; that surfaces as
    ``OperationalError: locking protocol``. On such a filesystem switch to
    EXCLUSIVE locking (wal-index kept in heap, no ``-shm``) and a plain DELETE
    journal, which also converts a file already left in WAL mode back to
    rollback journalling without needing ``-shm``.
    """
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.OperationalError as exc:
        logger.warning(
            "WAL journal unavailable (%s); using EXCLUSIVE/DELETE journal "
            "(filesystem without shared-memory locking, e.g. NFS)",
            exc,
        )
        conn.execute("PRAGMA locking_mode=EXCLUSIVE")
        conn.execute("PRAGMA journal_mode=DELETE")


def _is_filesystem_locking_error(exc: sqlite3.Error) -> bool:
    """True for SQLite errors caused by a filesystem without working locks."""
    text = str(exc).lower()
    return "locking protocol" in text or "disk i/o error" in text


def _prepare_for_nolock(path: Path) -> None:
    """Make a local database file openable with ``nolock=1``.

    A WAL-mode database always needs shared memory, which is unavailable under
    nolock, so drop stale WAL sidecars and downgrade the file header (bytes
    18-19: 2 -> 1) to a rollback journal. Safe here: the file is a freshly
    downloaded, self-contained copy (S3 is the source of truth), so there are no
    pending WAL frames to lose.
    """
    for suffix in ("-wal", "-shm"):
        path.with_name(path.name + suffix).unlink(missing_ok=True)
    try:
        with open(path, "rb") as handle:
            header = handle.read(20)
    except OSError:
        return
    if len(header) >= 20 and header[18:20] == b"\x02\x02":
        with open(path, "r+b") as handle:
            handle.seek(18)
            handle.write(b"\x01\x01")


class SqliteS3Database:
    """
    SQLite database synced with S3.

    Workflow: download from S3 -> work locally -> upload back to S3.
    Each instance manages a single S3 object / local file pair.
    """

    def __init__(
        self,
        config: S3DatabaseConfig,
        s3_key: str,
        db_type: DatabaseType,
        *,
        local_dir: Path | None = None,
    ) -> None:
        self._config = config
        self._s3_key = s3_key
        self._db_type = db_type
        self._local_dir = local_dir or Path(tempfile.mkdtemp(prefix="tracker-sqlite-"))
        self._local_dir.mkdir(parents=True, exist_ok=True)
        safe_name = s3_key.replace("/", "_").replace("\\", "_")
        self._local_path = self._local_dir / safe_name
        self._conn: sqlite3.Connection | None = None
        self._conn_stat: tuple[int, int] | None = None
        self._s3_client = None

    @property
    def local_path(self) -> Path:
        return self._local_path

    @property
    def connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        self._heal_if_moved()
        return self._conn

    def _heal_if_moved(self) -> None:
        """Reconnect when the local file was replaced or deleted under us.

        SQLite marks a connection permanently readonly once its database file
        is renamed away or unlinked (SQLITE_READONLY_DBMOVED) — every write
        then fails with "attempt to write a readonly database" even though the
        path is writable. Compare the inode captured at connect time with the
        path's current inode and transparently reopen on mismatch.
        """
        if self._conn is None or self._conn_stat is None:
            return
        try:
            stat = os.stat(self._local_path)
            current = (stat.st_dev, stat.st_ino)
        except OSError:
            current = None
        if current == self._conn_stat:
            return
        logger.warning(
            "Local SQLite file for %s was replaced or deleted under a live "
            "connection (inode %s -> %s); reopening",
            self._s3_key,
            self._conn_stat,
            current,
        )
        self.close()
        self.connect()

    def _get_s3_client(self):
        if self._s3_client is None:
            self._s3_client = _build_s3_client(self._config)
        return self._s3_client

    def download(self) -> bool:
        """Download SQLite file from S3. Returns False if object doesn't exist."""
        if self._conn is not None:
            # The download replaces the local file; a connection opened on the
            # old inode would silently turn readonly (see _heal_if_moved).
            logger.debug(
                "Closing live connection to %s before S3 download", self._s3_key
            )
            self.close()
        try:
            client = self._get_s3_client()
            full_key = self._config.full_key(self._s3_key)
            client.download_file(self._config.bucket, full_key, str(self._local_path))
            try:
                _, repaired = normalize_sqlite_file(self._local_path)
            except ValueError as exc:
                raise sqlite3.DatabaseError(str(exc)) from exc
            if repaired:
                logger.warning(
                    "Recovered chunked-encoded SQLite file after download: %s",
                    self._s3_key,
                )
            logger.debug("Downloaded %s from S3", self._s3_key)
            return True
        except Exception as exc:
            error_code = getattr(getattr(exc, "response", None), "get", lambda *_: {})(
                "Error", {}
            ).get("Code", "")
            if error_code in ("404", "NoSuchKey", "NotFound"):
                logger.debug("S3 object %s not found, will create new", self._s3_key)
                return False
            try:
                import botocore.exceptions

                if isinstance(exc, botocore.exceptions.ClientError):
                    code = exc.response.get("Error", {}).get("Code", "")
                    if code in ("404", "NoSuchKey", "NotFound"):
                        logger.debug(
                            "S3 object %s not found, will create new", self._s3_key
                        )
                        return False
            except ImportError:
                pass
            logger.error("Failed to download %s: %s", self._s3_key, exc)
            raise

    def upload(self) -> None:
        """Upload a consistent snapshot of the local SQLite file to S3.

        The upload always streams an immutable snapshot copy, never the live
        file: concurrent writers mutating the file mid-upload make S3 reject
        the object (BadDigest, Content-MD5 mismatch). With a live connection
        the snapshot goes through the SQLite backup API (transactionally
        consistent, WAL included); otherwise it is a plain file copy.
        """
        if self._conn is None and not self._local_path.exists():
            raise FileNotFoundError(f"Local DB not found: {self._local_path}")
        snapshot_path = self._local_path.with_name(self._local_path.name + ".upload")
        try:
            if self._conn is not None:
                self._conn.commit()
                snapshot = sqlite3.connect(str(snapshot_path))
                try:
                    self._conn.backup(snapshot)
                finally:
                    snapshot.close()
            else:
                shutil.copyfile(self._local_path, snapshot_path)
            try:
                _, repaired = normalize_sqlite_file(snapshot_path)
            except ValueError as exc:
                raise sqlite3.DatabaseError(str(exc)) from exc
            if repaired:
                logger.warning(
                    "Recovered chunked-encoded SQLite file before upload: %s",
                    self._s3_key,
                )
            client = self._get_s3_client()
            full_key = self._config.full_key(self._s3_key)
            client.upload_file(
                str(snapshot_path),
                self._config.bucket,
                full_key,
            )
        finally:
            snapshot_path.unlink(missing_ok=True)
        logger.debug("Uploaded %s to S3", self._s3_key)

    def connect(self) -> sqlite3.Connection:
        """Open local SQLite connection, creating schema if needed."""
        if self._conn is not None:
            return self._conn
        if self._local_path.exists() and self._local_path.stat().st_size > 0:
            try:
                _, repaired = normalize_sqlite_file(self._local_path)
            except ValueError as exc:
                raise sqlite3.DatabaseError(str(exc)) from exc
            if repaired:
                logger.warning(
                    "Recovered chunked-encoded SQLite file before connect: %s",
                    self._s3_key,
                )
        try:
            self._conn = self._open_configured(nolock=False)
        except sqlite3.OperationalError as exc:
            if not _is_filesystem_locking_error(exc):
                raise
            logger.warning(
                "SQLite locking unavailable (%s) for %s; reopening with "
                "nolock=1 (single-writer copy on a filesystem without POSIX "
                "file locks, e.g. NFS)",
                exc,
                self._s3_key,
            )
            self._conn = self._open_configured(nolock=True)
        try:
            stat = os.stat(self._local_path)
            self._conn_stat = (stat.st_dev, stat.st_ino)
        except OSError:
            self._conn_stat = None
        return self._conn

    def _open_configured(self, *, nolock: bool) -> sqlite3.Connection:
        # check_same_thread=False: callers serialize access with their own
        # locks and may run blocking DB work in worker threads (asyncio
        # to_thread) while the connection was created elsewhere.
        if nolock:
            _prepare_for_nolock(self._local_path)
            conn = sqlite3.connect(
                self._local_path.as_uri() + "?nolock=1",
                uri=True,
                check_same_thread=False,
            )
        else:
            conn = sqlite3.connect(str(self._local_path), check_same_thread=False)
        try:
            conn.row_factory = sqlite3.Row
            if nolock:
                # WAL needs shared memory and file locks; use a rollback journal.
                conn.execute("PRAGMA journal_mode=DELETE")
            else:
                _configure_journal_mode(conn)
            conn.execute("PRAGMA foreign_keys=ON")
            ensure_schema(conn, self._db_type)
        except Exception:
            conn.close()
            raise
        return conn

    def close(self) -> None:
        """Close local connection."""
        if self._conn is not None:
            try:
                self._conn.commit()
                self._conn.close()
            except Exception as exc:
                logger.warning("Error closing database: %s", exc)
            finally:
                self._conn = None
                self._conn_stat = None

    def sync_to_s3(self) -> None:
        """Commit and upload current state to S3."""
        if self._conn is not None:
            self._conn.commit()
        self.upload()

    def __enter__(self) -> SqliteS3Database:
        self.download()
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @staticmethod
    @contextmanager
    def read_only(
        config: S3DatabaseConfig,
        s3_key: str,
    ) -> Iterator[sqlite3.Connection]:
        """
        One-shot read-only access via sqlite_s3_query.

        Downloads from S3, opens read-only, yields connection, cleans up.
        Falls back to empty in-memory DB if the S3 object doesn't exist.
        """
        url = config.object_url(s3_key)

        silence_insecure_request_warnings(config.verify_ssl)

        def _http_client():
            session = make_requests_session()
            session.verify = config.verify_ssl
            return session

        def _credentials(_now):
            return (
                config.region,
                os.environ["AWS_ACCESS_KEY_ID"],
                os.environ["AWS_SECRET_ACCESS_KEY"],
                os.environ.get("AWS_SESSION_TOKEN"),
            )

        try:
            with sqlite_s3_query(
                url=url,
                get_credentials=_credentials,
                get_http_client=_http_client,
            ) as query_fn:
                yield query_fn
        except SQLiteError as exc:
            if "does not exist" in str(exc):
                logger.debug(
                    "S3 object %s not found for read-only access, yielding empty",
                    s3_key,
                )

                @contextmanager
                def _empty_query(sql, params=(), named_params=()):
                    yield (), iter([])

                yield _empty_query
            else:
                raise
