import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Sequence

from ..logger import Logger
from ..subprocess_manager import SubprocessManager

# Hardcoded bucket and root prefix for pretrain-auto-metrics artifacts on S3.
# Example resulting key:
#   pretrain-auto-metrics-artifacts/hf_checkpoint/artifacts/metrics/obm/<obm_commit_hash>/all_metrics.jsonl
ARTIFACTS_BUCKET = "gigaeye-pretrain-checkpoints"
ARTIFACTS_PREFIX = "pretrain-auto-metrics-artifacts"

# rclone configuration. The named remote (``RCLONE_REMOTE``) is defined in the
# rclone config file and holds the S3 endpoint + credentials, so no secrets live
# in the repo or the environment. ``RCLONE_REMOTE`` defaults to ``gigaeye-s3`` on
# purpose: a path ``gigaeye-s3:<bucket>/<key>`` is then simultaneously a valid
# rclone path *and* the ``gigaeye-s3:`` URL scheme job-dispatcher stores and reads.
#
# The binary and config default to the copy bundled with job-dispatcher-utils, but
# a user can point ``RCLONE_BINARY`` / ``RCLONE_CONFIG`` at their own install.
_RCLONE_DEFAULT_DIR = "/home/jovyan/job-dispatcher-utils/s3-interaction"
RCLONE_BINARY = os.getenv("RCLONE_BINARY", f"{_RCLONE_DEFAULT_DIR}/rclone")
RCLONE_CONFIG = os.getenv("RCLONE_CONFIG", f"{_RCLONE_DEFAULT_DIR}/rclone.conf")
RCLONE_REMOTE = os.getenv("RCLONE_REMOTE", "gigaeye-s3")
RCLONE_TIMEOUT_SECONDS = 5 * 60
RCLONE_ATTEMPTS = 3

# Mirror the copy tuning used by the gigachat s3-synchronize script (see main.py).
_RCLONE_COPY_FLAGS = [
    "--s3-no-check-bucket",
    "--transfers",
    "16",
    "--buffer-size",
    "256Mi",
    "--multi-thread-streams",
    "16",
    "--s3-disable-checksum",
    "--ignore-checksum",
    "--fast-list",
]


class S3ArtifactUploader:
    """Uploads local artifact directories to S3 via the ``rclone`` CLI, merging
    with existing keys.

    The bucket and prefix are hardcoded (``ARTIFACTS_BUCKET`` / ``ARTIFACTS_PREFIX``).
    Endpoint and credentials are not read here at all: they live in the rclone
    config under the ``RCLONE_REMOTE`` remote, so nothing secret is stored in the
    repo. Overridable via environment (defaults under ``_RCLONE_DEFAULT_DIR``):
        - RCLONE_BINARY   (path to the rclone executable)
        - RCLONE_CONFIG   (path to rclone.conf)
        - RCLONE_REMOTE   (remote name; default ``gigaeye-s3``)
    """

    def __init__(self) -> None:
        self.bucket = ARTIFACTS_BUCKET
        self.rclone_binary = RCLONE_BINARY
        self.config_path = RCLONE_CONFIG
        self.remote = RCLONE_REMOTE

    def is_configured(self) -> bool:
        if shutil.which(self.rclone_binary) is None:
            Logger.warning(
                f"rclone executable not found ({self.rclone_binary}); skipping S3 upload"
            )
            return False
        if self.config_path and not Path(self.config_path).is_file():
            Logger.warning(
                f"rclone config not found ({self.config_path}); skipping S3 upload"
            )
            return False
        return True

    def _base_cmd(self) -> list[str]:
        """``[rclone, --config <cfg>]`` prefix shared by every rclone invocation."""
        cmd = [self.rclone_binary]
        if self.config_path:
            cmd += ["--config", self.config_path]
        return cmd

    def _remote_path(self, key: str) -> str:
        """``<remote>:<bucket>/<key>`` — the rclone path for an artifact key."""
        return f"{self.remote}:{self.bucket}/{key.strip('/')}"

    def _run(self, args: list[str], stdout_path: Optional[Path] = None) -> bool:
        run_name = f"rclone {args[0]} {args[-1]}"
        cmd = self._base_cmd() + args
        for _ in range(RCLONE_ATTEMPTS):
            if stdout_path:
                stdout_path.unlink(missing_ok=True)
            sm = SubprocessManager(cmd, stdout_path, None, run_name)
            sm.start()
            try:
                sm.process.wait(timeout=RCLONE_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                Logger.error(f"{run_name} timed out after {RCLONE_TIMEOUT_SECONDS}s")
                sm.graceful_stop()
            else:
                sm.wait_for_completion()
            if sm.is_successful():
                return True
            sm.log_error()
        return False

    def get_object_bytes(self, s3_key: str) -> Optional[bytes]:
        if not self.is_configured():
            return None

        remote_path = self._remote_path(s3_key)
        # rclone cat of a non-existent key prints nothing and still exits 0;
        # -q keeps rclone's own messages out of the captured object bytes.
        with tempfile.TemporaryDirectory() as tmp_dir:
            stdout_path = Path(tmp_dir) / "object"
            if not self._run(
                ["cat", "-q", "--s3-no-check-bucket", remote_path], stdout_path
            ):
                return None
            return stdout_path.read_bytes()

    def _copy_merge(self, local_dir: Path, s3_key_prefix: str) -> bool:
        if not self.is_configured():
            return False

        dest = self._remote_path(s3_key_prefix)
        Logger.info(f"Uploading {local_dir} to {dest} (merge)")
        if not self._run(["copy", *_RCLONE_COPY_FLAGS, str(local_dir), dest]):
            return False

        Logger.info(f"Uploaded {local_dir} to {dest}")
        return True

    def upload_file(self, local_path: Path, s3_key_prefix: str) -> bool:
        local_path = Path(local_path)
        if not local_path.is_file():
            Logger.warning(f"S3 upload source is not a file: {local_path}")
            return False
        return self._copy_merge(local_path, s3_key_prefix)

    def upload_dir_merge(self, local_dir: Path, s3_key_prefix: str) -> bool:
        local_dir = Path(local_dir)
        if not local_dir.is_dir():
            Logger.warning(f"S3 upload source is not a directory: {local_dir}")
            return False
        return self._copy_merge(local_dir, s3_key_prefix)

    def upload_files_merge(
        self, local_dir: Path, rel_paths: Sequence[Path], s3_key_prefix: str
    ) -> bool:
        local_dir = Path(local_dir)
        if not local_dir.is_dir():
            Logger.warning(f"S3 upload source is not a directory: {local_dir}")
            return False
        if not self.is_configured():
            return False

        return self._copy_files_merge(
            str(local_dir), self._remote_path(s3_key_prefix), rel_paths
        )

    def copy_files_merge(
        self, src_key_prefix: str, rel_paths: Sequence[Path], dst_key_prefix: str
    ) -> bool:
        if not self.is_configured():
            return False

        return self._copy_files_merge(
            self._remote_path(src_key_prefix),
            self._remote_path(dst_key_prefix),
            rel_paths,
        )

    def _copy_files_merge(
        self, source: str, dest: str, rel_paths: Sequence[Path]
    ) -> bool:
        with tempfile.TemporaryDirectory() as tmp_dir:
            listing = Path(tmp_dir) / "files-from.txt"
            listing.write_text(
                "".join(f"{path}\n" for path in rel_paths), encoding="utf-8"
            )
            Logger.info(
                f"Copying {len(rel_paths)} file(s) from {source} to {dest} (merge)"
            )
            if not self._run(
                [
                    "copy",
                    *_RCLONE_COPY_FLAGS,
                    "--files-from",
                    str(listing),
                    source,
                    dest,
                ]
            ):
                return False

        Logger.info(f"Copied {len(rel_paths)} file(s) from {source} to {dest}")
        return True
