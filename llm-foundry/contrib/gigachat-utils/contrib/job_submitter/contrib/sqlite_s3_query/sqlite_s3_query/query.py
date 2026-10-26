from __future__ import annotations

import hmac
import os
import sqlite3
import tempfile
import threading
from contextlib import contextmanager
from ctypes import (
    CFUNCTYPE,
    POINTER,
    Structure,
    addressof,
    byref,
    c_char,
    c_char_p,
    c_double,
    c_int,
    c_int64,
    c_void_p,
    cast,
    cdll,
    create_string_buffer,
    memmove,
    pointer,
    sizeof,
    string_at,
)
from ctypes.util import find_library
from datetime import datetime
from functools import partial
from hashlib import sha256
from itertools import zip_longest
from pathlib import Path
from time import time
from typing import Any, Callable, Iterable, Iterator
from urllib.parse import quote, urlsplit
from uuid import uuid4

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .sqlite_file import normalize_sqlite_file

_BACKEND_ENV_NAME = "SQLITE_S3_QUERY_READ_BACKEND"
_BACKEND_CTYPES = "ctypes"
_BACKEND_DOWNLOAD = "download"


class SQLiteS3QueryError(Exception):
    pass


class VersioningNotEnabledError(SQLiteS3QueryError):
    pass


class SQLiteError(SQLiteS3QueryError):
    pass


class QueryContextClosedError(SQLiteS3QueryError):
    pass


def make_requests_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.3,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=("HEAD", "GET"),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


@contextmanager
def sqlite_s3_query_multi(
    url: str,
    get_credentials: Callable[[datetime], tuple[str, str, str, str | None]] | None = (
        lambda now: (
            os.environ["AWS_REGION"],
            os.environ["AWS_ACCESS_KEY_ID"],
            os.environ["AWS_SECRET_ACCESS_KEY"],
            os.environ.get("AWS_SESSION_TOKEN"),
        )
    ),
    get_http_client: Callable[[], requests.Session] = make_requests_session,
    get_libsqlite3: Callable[[], Any] | None = None,
):
    backend = _resolve_backend()
    if backend == _BACKEND_CTYPES:
        with _sqlite_s3_query_multi_ctypes(
            url=url,
            get_credentials=get_credentials,
            get_http_client=get_http_client,
            get_libsqlite3=get_libsqlite3,
        ) as query_fn:

            def query_with_fallback(
                sql: str,
                params: tuple[tuple[Any, ...], ...] = (),
                named_params: tuple[tuple[tuple[str, Any], ...], ...] = (),
            ):
                try:
                    yield from query_fn(sql, params, named_params)
                    return
                except SQLiteError as exc:
                    if not _is_non_sqlite_database_error(exc):
                        raise

                # The object may contain chunked framing bytes and cannot be read
                # through the VFS/range backend. Retry with full download path that
                # applies payload normalization/repair.
                with _sqlite_s3_query_multi_download(
                    url=url,
                    get_credentials=get_credentials,
                    get_http_client=get_http_client,
                ) as download_query_fn:
                    yield from download_query_fn(sql, params, named_params)

            yield query_with_fallback
        return

    with _sqlite_s3_query_multi_download(
        url=url,
        get_credentials=get_credentials,
        get_http_client=get_http_client,
    ) as query_fn:
        yield query_fn


@contextmanager
def sqlite_s3_query(
    url: str,
    get_credentials: Callable[[datetime], tuple[str, str, str, str | None]] | None = (
        lambda now: (
            os.environ["AWS_REGION"],
            os.environ["AWS_ACCESS_KEY_ID"],
            os.environ["AWS_SECRET_ACCESS_KEY"],
            os.environ.get("AWS_SESSION_TOKEN"),
        )
    ),
    get_http_client: Callable[[], requests.Session] = make_requests_session,
    get_libsqlite3: Callable[[], Any] | None = None,
):
    @contextmanager
    def query(
        query_base: Callable[
            ..., Iterable[tuple[tuple[str, ...], Iterator[tuple[Any, ...]]]]
        ],
        sql: str,
        params: tuple[Any, ...] = (),
        named_params: tuple[tuple[str, Any], ...] = (),
    ):
        for columns, rows in query_base(sql, (params,), (named_params,)):
            yield columns, rows
            break

    with sqlite_s3_query_multi(
        url,
        get_credentials=get_credentials,
        get_http_client=get_http_client,
        get_libsqlite3=get_libsqlite3,
    ) as query_base:
        yield partial(query, query_base)


def _resolve_backend() -> str:
    backend = os.environ.get(_BACKEND_ENV_NAME, _BACKEND_CTYPES).strip().lower()
    if backend in (_BACKEND_CTYPES, _BACKEND_DOWNLOAD):
        return backend
    raise SQLiteError(
        f"Unsupported {_BACKEND_ENV_NAME}={backend!r}. "
        f"Expected '{_BACKEND_CTYPES}' or '{_BACKEND_DOWNLOAD}'."
    )


def _is_non_sqlite_database_error(exc: SQLiteError) -> bool:
    message = str(exc).lower()
    return "file is not a database" in message or "not a sqlite database" in message


@contextmanager
def _sqlite_s3_query_multi_download(
    url: str,
    get_credentials: Callable[[datetime], tuple[str, str, str, str | None]] | None = (
        lambda now: (
            os.environ["AWS_REGION"],
            os.environ["AWS_ACCESS_KEY_ID"],
            os.environ["AWS_SECRET_ACCESS_KEY"],
            os.environ.get("AWS_SESSION_TOKEN"),
        )
    ),
    get_http_client: Callable[[], requests.Session] = make_requests_session,
):
    with get_http_client() as http_client:
        local_path = _download_sqlite(url, http_client, get_credentials)
        conn = sqlite3.connect(f"file:{local_path}?mode=ro", uri=True)
        conn.row_factory = None
        try:
            yield partial(_query_many, conn)
        finally:
            conn.close()
            local_path.unlink(missing_ok=True)


@contextmanager
def _sqlite_s3_query_multi_ctypes(
    url: str,
    get_credentials: Callable[[datetime], tuple[str, str, str, str | None]] | None = (
        lambda now: (
            os.environ["AWS_REGION"],
            os.environ["AWS_ACCESS_KEY_ID"],
            os.environ["AWS_SECRET_ACCESS_KEY"],
            os.environ.get("AWS_SESSION_TOKEN"),
        )
    ),
    get_http_client: Callable[[], requests.Session] = make_requests_session,
    get_libsqlite3: Callable[[], Any] | None = None,
):
    lib_loader = get_libsqlite3 or (lambda: cdll.LoadLibrary(find_library("sqlite3")))
    libsqlite3 = lib_loader()

    libsqlite3.sqlite3_errstr.restype = c_char_p
    libsqlite3.sqlite3_errmsg.restype = c_char_p
    libsqlite3.sqlite3_column_name.restype = c_char_p
    libsqlite3.sqlite3_column_double.restype = c_double
    libsqlite3.sqlite3_column_int64.restype = c_int64
    libsqlite3.sqlite3_column_blob.restype = c_void_p
    libsqlite3.sqlite3_column_bytes.restype = c_int64

    sqlite_ok = 0
    sqlite_ioerr = 10
    sqlite_notfound = 12
    sqlite_row = 100
    sqlite_done = 101
    sqlite_ioerr_short_read = 522
    sqlite_transient = c_void_p(-1)
    sqlite_open_readonly = 0x00000001
    sqlite_open_nomutex = 0x00008000
    sqlite_iocap_immutable = 0x00002000

    def bind_text(pp_stmt, index, value: str):
        encoded = value.encode("utf-8")
        return libsqlite3.sqlite3_bind_text(
            pp_stmt,
            index,
            encoded,
            len(encoded),
            sqlite_transient,
        )

    def bind_blob(pp_stmt, index, value: bytes):
        return libsqlite3.sqlite3_bind_blob(
            pp_stmt,
            index,
            value,
            len(value),
            sqlite_transient,
        )

    extract = {
        1: libsqlite3.sqlite3_column_int64,
        2: libsqlite3.sqlite3_column_double,
        3: lambda pp_stmt, i: string_at(
            libsqlite3.sqlite3_column_blob(pp_stmt, i),
            libsqlite3.sqlite3_column_bytes(pp_stmt, i),
        ).decode(),
        4: lambda pp_stmt, i: string_at(
            libsqlite3.sqlite3_column_blob(pp_stmt, i),
            libsqlite3.sqlite3_column_bytes(pp_stmt, i),
        ),
        5: lambda pp_stmt, i: None,
    }

    local = threading.local()
    local.pending_exception = None

    def set_pending_exception(exception: Exception) -> None:
        local.pending_exception = exception

    def raise_any_pending_exception() -> None:
        to_raise = local.pending_exception
        if to_raise is not None:
            local.pending_exception = None
            raise to_raise

    def run(func, *args) -> None:
        result = func(*args)
        raise_any_pending_exception()
        if result != sqlite_ok:
            raise SQLiteError(libsqlite3.sqlite3_errstr(result).decode())

    def run_with_db(db, func, *args) -> None:
        result = func(*args)
        raise_any_pending_exception()
        if result != sqlite_ok:
            raise SQLiteError(libsqlite3.sqlite3_errmsg(db).decode())

    def bind_value(db, pp_stmt, index: int, value: Any) -> None:
        if value is None:
            run_with_db(db, libsqlite3.sqlite3_bind_null, pp_stmt, index)
            return
        if isinstance(value, bool) or isinstance(value, int):
            run_with_db(db, libsqlite3.sqlite3_bind_int64, pp_stmt, index, int(value))
            return
        if isinstance(value, float):
            run_with_db(db, libsqlite3.sqlite3_bind_double, pp_stmt, index, value)
            return
        if isinstance(value, str):
            run_with_db(db, bind_text, pp_stmt, index, value)
            return
        if isinstance(value, memoryview):
            value = value.tobytes()
        if isinstance(value, bytearray):
            value = bytes(value)
        if isinstance(value, bytes):
            run_with_db(db, bind_blob, pp_stmt, index, value)
            return
        raise SQLiteError(f"Unsupported SQLite bind parameter type: {type(value)!r}")

    @contextmanager
    def make_auth_request(http_client, method, params, headers):
        now = datetime.utcnow()
        request_headers = dict(get_request_headers(method, params, headers, now))
        request_url = f"{scheme}://{netloc}{path}"

        stream_method = getattr(http_client, "stream", None)
        if callable(stream_method):
            with stream_method(
                method,
                request_url,
                params=params,
                headers=request_headers,
            ) as response:
                response.raise_for_status()
                yield response
            return

        response = http_client.request(
            method,
            request_url,
            params=params,
            headers=request_headers,
            stream=True,
            timeout=60,
        )
        try:
            response.raise_for_status()
            yield response
        finally:
            response.close()

    def get_request_headers_for_private_buckets(method, params, headers, now):
        region, access_key_id, secret_access_key, session_token = get_credentials(now)
        return _aws_sigv4_headers(
            now=now,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            region=region,
            method=method,
            url=f"{scheme}://{netloc}{path}",
            headers_to_sign=(("host", netloc),) + tuple(headers),
            query_params=tuple(params),
            session_token=session_token,
        )

    def get_request_headers_for_public_buckets(_, __, headers, ___):
        return headers

    get_request_headers = (
        get_request_headers_for_private_buckets
        if get_credentials is not None
        else get_request_headers_for_public_buckets
    )

    def iter_response_bytes(response, chunk_size: int = 65536):
        if hasattr(response, "iter_content"):
            yield from response.iter_content(chunk_size=chunk_size)
            return
        if hasattr(response, "iter_bytes"):
            yield from response.iter_bytes()
            return
        data = getattr(response, "content", b"")
        if data:
            yield data

    def make_struct(fields):
        class Struct(Structure):
            _fields_ = [
                (field_name, field_type) for (field_name, field_type, _) in fields
            ]

        return Struct(*tuple(value for (_, _, value) in fields))

    run(libsqlite3.sqlite3_initialize)

    vfs_name = b"s3-" + str(uuid4()).encode("ascii")
    file_name = b"s3-" + str(uuid4()).encode("ascii")
    split = urlsplit(url)
    scheme, netloc, path = split.scheme, split.netloc, split.path
    if not scheme or not netloc:
        raise SQLiteError(f"Invalid URL: {url!r}")

    @contextmanager
    def get_vfs(http_client):
        with make_auth_request(http_client, "HEAD", (), ()) as response:
            head_headers = response.headers
            next(iter_response_bytes(response), b"")

        version_id = head_headers.get("x-amz-version-id")
        if not version_id:
            raise VersioningNotEnabledError("The bucket must have versioning enabled")

        content_length = head_headers.get("content-length")
        if content_length is None:
            raise SQLiteError("Missing content-length header from S3 response")
        size = int(content_length)

        x_open_type = CFUNCTYPE(
            c_int, c_void_p, c_char_p, c_void_p, c_int, POINTER(c_int)
        )

        def x_open(p_vfs, z_name, p_file, flags, p_out_flags):
            memmove(p_file, addressof(file), sizeof(file))
            p_out_flags[0] = flags
            return sqlite_ok

        x_close_type = CFUNCTYPE(c_int, c_void_p)

        def x_close(p_file):
            return sqlite_ok

        x_read_type = CFUNCTYPE(c_int, c_void_p, c_void_p, c_int, c_int64)

        def x_read(p_file, p_out, i_amt, i_ofst):
            offset = 0
            base_address = (
                p_out if isinstance(p_out, int) else cast(p_out, c_void_p).value
            )
            if base_address is None:
                set_pending_exception(SQLiteError("Invalid SQLite read buffer pointer"))
                return sqlite_ioerr

            try:
                with make_auth_request(
                    http_client,
                    "GET",
                    (("versionId", version_id),),
                    (("range", f"bytes={i_ofst}-{i_ofst + i_amt - 1}"),),
                ) as response:
                    for chunk in iter_response_bytes(response):
                        if not chunk:
                            continue
                        write_size = min(i_amt - offset, len(chunk))
                        if write_size > 0:
                            memmove(base_address + offset, chunk, write_size)
                        offset += len(chunk)
                        if offset > i_amt:
                            break
            except Exception as exception:  # pragma: no cover - callback path
                set_pending_exception(exception)
                return sqlite_ioerr

            if offset < i_amt:
                remainder = i_amt - offset
                memmove(base_address + offset, b"\0" * remainder, remainder)
                return sqlite_ioerr_short_read

            if offset > i_amt:
                return sqlite_ioerr

            return sqlite_ok

        x_file_size_type = CFUNCTYPE(c_int, c_void_p, POINTER(c_int64))

        def x_file_size(p_file, p_size):
            p_size[0] = size
            return sqlite_ok

        x_lock_type = CFUNCTYPE(c_int, c_void_p, c_int)

        def x_lock(p_file, e_lock):
            return sqlite_ok

        x_unlock_type = CFUNCTYPE(c_int, c_void_p, c_int)

        def x_unlock(p_file, e_lock):
            return sqlite_ok

        x_file_control_type = CFUNCTYPE(c_int, c_void_p, c_int, c_void_p)

        def x_file_control(p_file, op, p_arg):
            return sqlite_notfound

        x_device_characteristics_type = CFUNCTYPE(c_int, c_void_p)

        def x_device_characteristics(p_file):
            return sqlite_iocap_immutable

        x_access_type = CFUNCTYPE(c_int, c_void_p, c_char_p, c_int, POINTER(c_int))

        def x_access(p_vfs, z_name, flags, z_out):
            z_out[0] = 0
            return sqlite_ok

        x_full_pathname_type = CFUNCTYPE(
            c_int, c_void_p, c_char_p, c_int, POINTER(c_char)
        )

        def x_full_pathname(p_vfs, z_name, n_out, z_out):
            name_bytes = (
                z_name if isinstance(z_name, (bytes, bytearray)) else string_at(z_name)
            )
            copy_size = min(len(name_bytes) + 1, n_out)
            memmove(z_out, name_bytes, copy_size)
            return sqlite_ok

        x_current_time_type = CFUNCTYPE(c_int, c_void_p, POINTER(c_double))

        def x_current_time(p_vfs, c_double_p):
            c_double_p[0] = time() / 86400.0 + 2440587.5
            return sqlite_ok

        io_methods = make_struct(
            (
                ("i_version", c_int, 1),
                ("x_close", x_close_type, x_close_type(x_close)),
                ("x_read", x_read_type, x_read_type(x_read)),
                ("x_write", c_void_p, None),
                ("x_truncate", c_void_p, None),
                ("x_sync", c_void_p, None),
                ("x_file_size", x_file_size_type, x_file_size_type(x_file_size)),
                ("x_lock", x_lock_type, x_lock_type(x_lock)),
                ("x_unlock", x_unlock_type, x_unlock_type(x_unlock)),
                ("x_check_reserved_lock", c_void_p, None),
                (
                    "x_file_control",
                    x_file_control_type,
                    x_file_control_type(x_file_control),
                ),
                ("x_sector_size", c_void_p, None),
                (
                    "x_device_characteristics",
                    x_device_characteristics_type,
                    x_device_characteristics_type(x_device_characteristics),
                ),
            )
        )

        file = make_struct(
            (("p_methods", POINTER(type(io_methods)), pointer(io_methods)),)
        )

        vfs = make_struct(
            (
                ("i_version", c_int, 1),
                ("sz_os_file", c_int, sizeof(file)),
                ("mx_pathname", c_int, 1024),
                ("p_next", c_void_p, None),
                ("z_name", c_char_p, vfs_name),
                ("p_app_data", c_char_p, None),
                ("x_open", x_open_type, x_open_type(x_open)),
                ("x_delete", c_void_p, None),
                ("x_access", x_access_type, x_access_type(x_access)),
                (
                    "x_full_pathname",
                    x_full_pathname_type,
                    x_full_pathname_type(x_full_pathname),
                ),
                ("x_dl_open", c_void_p, None),
                ("x_dl_error", c_void_p, None),
                ("x_dl_sym", c_void_p, None),
                ("x_dl_close", c_void_p, None),
                ("x_randomness", c_void_p, None),
                ("x_sleep", c_void_p, None),
                (
                    "x_current_time",
                    x_current_time_type,
                    x_current_time_type(x_current_time),
                ),
                ("x_get_last_error", c_void_p, None),
            )
        )

        run(libsqlite3.sqlite3_vfs_register, byref(vfs), 0)
        try:
            yield vfs
        finally:
            run(libsqlite3.sqlite3_vfs_unregister, byref(vfs))

    @contextmanager
    def get_db(vfs):
        db = c_void_p()
        run(
            libsqlite3.sqlite3_open_v2,
            file_name,
            byref(db),
            sqlite_open_readonly | sqlite_open_nomutex,
            vfs_name,
        )
        try:
            yield db
        finally:
            run_with_db(db, libsqlite3.sqlite3_close, db)

    @contextmanager
    def get_pp_stmt_getter(db):
        statements: dict[object, c_void_p] = {}

        def get_pp_stmt(statement):
            try:
                return statements[statement]
            except KeyError:
                raise QueryContextClosedError(
                    "Attempting to use finalized statement"
                ) from None

        def finalize(statement):
            pp_stmt = statements.pop(statement)
            try:
                run_with_db(db, libsqlite3.sqlite3_finalize, pp_stmt)
            except Exception:
                # If the statement is already invalid because of malformed remote input,
                # finalize may fail. Preserve original error path.
                pass

        def get_pp_stmts(sql):
            p_encoded = POINTER(c_char)(create_string_buffer(sql.encode("utf-8")))
            while True:
                pp_stmt = c_void_p()
                run_with_db(
                    db,
                    libsqlite3.sqlite3_prepare_v2,
                    db,
                    p_encoded,
                    -1,
                    byref(pp_stmt),
                    byref(p_encoded),
                )
                if not pp_stmt:
                    break

                statement = object()
                statements[statement] = pp_stmt
                yield partial(get_pp_stmt, statement), partial(finalize, statement)

        yield get_pp_stmts

    def rows(get_pp_stmt, columns):
        while True:
            pp_stmt = get_pp_stmt()
            result = libsqlite3.sqlite3_step(pp_stmt)
            if result == sqlite_done:
                break
            if result != sqlite_row:
                raise_any_pending_exception()
                raise SQLiteError(libsqlite3.sqlite3_errstr(result).decode())

            yield tuple(
                extract[libsqlite3.sqlite3_column_type(pp_stmt, i)](pp_stmt, i)
                for i in range(0, len(columns))
            )

    def query(vfs, sql, params=(), named_params=()):
        def zip_first(first_iterable, *iterables, default=()):
            iterators = tuple(iter(iterable) for iterable in iterables)
            for value in first_iterable:
                yield (value,) + tuple(
                    next(iterator, default) for iterator in iterators
                )

        with get_db(vfs) as db, get_pp_stmt_getter(db) as get_pp_stmts:
            for (
                get_pp_stmt,
                finalize_stmt,
            ), statement_params, statement_named_params in zip_first(
                get_pp_stmts(sql),
                params,
                named_params,
            ):
                try:
                    pp_stmt = get_pp_stmt()
                    for i, param in enumerate(statement_params):
                        bind_value(db, pp_stmt, i + 1, param)

                    for param_name, param_value in statement_named_params:
                        index = libsqlite3.sqlite3_bind_parameter_index(
                            pp_stmt, param_name.encode("utf-8")
                        )
                        bind_value(db, pp_stmt, index, param_value)

                    columns = tuple(
                        libsqlite3.sqlite3_column_name(pp_stmt, i).decode()
                        for i in range(0, libsqlite3.sqlite3_column_count(pp_stmt))
                    )
                    yield columns, rows(get_pp_stmt, columns)
                finally:
                    finalize_stmt()

    with get_http_client() as http_client, get_vfs(http_client) as vfs:
        yield partial(query, vfs)


def _download_sqlite(
    url: str,
    http_client: requests.Session,
    get_credentials: Callable[[datetime], tuple[str, str, str, str | None]] | None,
) -> Path:
    split = urlsplit(url)
    if not split.scheme or not split.netloc:
        raise SQLiteError(f"Invalid URL: {url!r}")

    headers: dict[str, str] = {}
    if get_credentials is not None:
        now = datetime.utcnow()
        region, access_key, secret_key, token = get_credentials(now)
        signed = _aws_sigv4_headers(
            now=now,
            access_key_id=access_key,
            secret_access_key=secret_key,
            region=region,
            method="GET",
            url=url,
            headers_to_sign=(("host", split.netloc),),
            query_params=(),
            session_token=token,
        )
        headers = dict(signed)

    response = http_client.get(url, headers=headers, stream=True, timeout=60)
    try:
        response.raise_for_status()
        local_path: Path | None = None
        with tempfile.NamedTemporaryFile(
            prefix="sqlite-s3-", suffix=".sqlite", delete=False
        ) as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    temp_file.write(chunk)
            local_path = Path(temp_file.name)
        try:
            normalize_sqlite_file(local_path)
        except ValueError as exc:
            local_path.unlink(missing_ok=True)
            raise SQLiteError(str(exc)) from exc
        return local_path
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            raise SQLiteError("SQLite object does not exist in S3") from exc
        raise
    finally:
        response.close()


def _query_many(
    conn: sqlite3.Connection,
    sql: str,
    params: tuple[tuple[Any, ...], ...] = (),
    named_params: tuple[tuple[tuple[str, Any], ...], ...] = (),
):
    param_sets = params or ((),)
    named_param_sets = named_params or ((),)

    for statement_params, statement_named_params in zip_longest(
        param_sets,
        named_param_sets,
        fillvalue=(),
    ):
        try:
            if statement_named_params:
                if statement_params:
                    raise SQLiteError(
                        "Cannot bind positional and named parameters together"
                    )
                cursor = conn.execute(sql, dict(statement_named_params))
            else:
                cursor = conn.execute(sql, statement_params)
        except sqlite3.Error as exc:
            raise SQLiteError(str(exc)) from exc

        columns = tuple(desc[0] for desc in (cursor.description or ()))
        yield columns, _iter_rows(cursor)


def _iter_rows(cursor: sqlite3.Cursor) -> Iterator[tuple[Any, ...]]:
    while True:
        row = cursor.fetchone()
        if row is None:
            return
        yield tuple(row)


def _aws_sigv4_headers(
    *,
    now: datetime,
    access_key_id: str,
    secret_access_key: str,
    region: str,
    method: str,
    url: str,
    headers_to_sign: tuple[tuple[str, str], ...],
    query_params: tuple[tuple[str, str], ...],
    session_token: str | None,
) -> tuple[tuple[str, str], ...]:
    split = urlsplit(url)
    algorithm = "AWS4-HMAC-SHA256"
    body_hash = sha256(b"").hexdigest()

    amzdate = now.strftime("%Y%m%dT%H%M%SZ")
    datestamp = amzdate[:8]
    credential_scope = f"{datestamp}/{region}/s3/aws4_request"

    headers = tuple(
        sorted(
            headers_to_sign
            + (
                ("x-amz-content-sha256", body_hash),
                ("x-amz-date", amzdate),
            )
            + ((("x-amz-security-token", session_token),) if session_token else ())
        )
    )
    signed_headers = ";".join(key for key, _ in headers)

    canonical_uri = quote(split.path or "/", safe="/~")
    quoted_params = sorted(
        (quote(key, safe="~"), quote(value, safe="~")) for key, value in query_params
    )
    canonical_querystring = "&".join(f"{key}={value}" for key, value in quoted_params)
    canonical_headers = "".join(f"{key}:{value}\n" for key, value in headers)
    canonical_request = (
        f"{method}\n{canonical_uri}\n{canonical_querystring}\n"
        f"{canonical_headers}\n{signed_headers}\n{body_hash}"
    )

    string_to_sign = (
        f"{algorithm}\n{amzdate}\n{credential_scope}\n"
        f"{sha256(canonical_request.encode('ascii')).hexdigest()}"
    )

    signing_key = _sign(
        _sign(
            _sign(
                _sign(("AWS4" + secret_access_key).encode("ascii"), datestamp),
                region,
            ),
            "s3",
        ),
        "aws4_request",
    )
    signature = hmac.new(
        signing_key, string_to_sign.encode("ascii"), sha256
    ).hexdigest()

    auth_header = (
        "Authorization",
        (
            f"{algorithm} Credential={access_key_id}/{credential_scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        ),
    )
    return (auth_header,) + headers


def _sign(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("ascii"), sha256).digest()
