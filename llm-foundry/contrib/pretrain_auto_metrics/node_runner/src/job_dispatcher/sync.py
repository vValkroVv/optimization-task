import os
import socket
from pathlib import Path
from typing import Optional

from ..logger import Logger
from ..subprocess_manager import SubprocessManager


class JDSync:
    @classmethod
    def internet_is_available(cls, timeout: int = 3) -> bool:
        try:
            socket.setdefaulttimeout(timeout)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("1.1.1.1", 53))
            return True
        except OSError:
            return False

    @classmethod
    def get_jd_base_url(cls) -> str:
        if cls.internet_is_available():
            return "https://jdmetrics.sberdevices.ru/api"
        return "https://dj.static.space/api"

    @classmethod
    def get_jd_sync_dir(cls) -> Optional[Path]:
        jd_sync_dir = os.getenv("JD_SYNC_DIR")
        return Path(jd_sync_dir) if jd_sync_dir else None

    @classmethod
    def tar_metrics_dir(
        cls,
        metrics_dir: Path,
        original_artifacts_path: Path,
        evaluator_name: str,
    ) -> Optional[Path]:
        if cls.internet_is_available():
            return None
        if cls.get_jd_sync_dir() is None:
            Logger.info(
                f"JD_SYNC_DIR is not set, skipping {evaluator_name} artifacts archive"
            )
            return None
        tar_archive_path = (
            original_artifacts_path / f"{metrics_dir.name}_artifacts.tar.zst"
        )
        if tar_archive_path.exists():
            tar_archive_path.unlink()
        tar_archive_path = tar_archive_path.absolute().as_posix()
        tar_cmd = [
            "tar",
            "--zstd",
            "--ignore-failed-read",
            "-cf",
            tar_archive_path,
            metrics_dir.name,
        ]
        sm = SubprocessManager(
            tar_cmd, None, metrics_dir.parent, f"Tar {metrics_dir.name}"
        )
        sm.start()
        sm.wait_for_completion()
        sm.raise_for_exit_code()
        Logger.info(f"Tarred {metrics_dir.name} to {tar_archive_path}")
        return Path(tar_archive_path)

    @classmethod
    def symlink_metrics_for_transfer(
        cls,
        tar_archive_path: Optional[Path],
        hf_checkpoint_path: Path,
        evaluator_name: str,
    ) -> None:
        if cls.internet_is_available():
            return
        jd_sync_path = cls.get_jd_sync_dir()
        if jd_sync_path is None:
            Logger.warning(
                f"JD_SYNC_DIR is not set, skipping symlinking {evaluator_name} artifacts"
            )
            return

        jd_sync_path.mkdir(parents=True, exist_ok=True)

        transfer_directory = jd_sync_path / cls.parse_transfer_directory(
            hf_checkpoint_path
        )
        transfer_directory.mkdir(parents=True, exist_ok=True)

        if tar_archive_path is None or not tar_archive_path.exists():
            Logger.warning(
                f"{evaluator_name} artifacts archive not found: {tar_archive_path}"
            )
            return

        link_path = transfer_directory / tar_archive_path.name
        if link_path.exists() or link_path.is_symlink():
            Logger.info("Link exists, unlinking...")
            link_path.unlink()

        link_path.symlink_to(tar_archive_path.resolve())
        Logger.info(f"Created symlink: {link_path} -> {tar_archive_path}")

        cls.transfer_to_cloud(link_path, f"pam-{evaluator_name.lower()}-transfer")

    @classmethod
    def transfer_to_cloud(cls, source_path: Path, exp_name: str) -> bool:
        try:
            # импорт здесь, а не наверху т.к. на NFS
            # хватает чекаутов, где job submitter криво стоит, не
            # чекаутится или где ничего не настроено, в свою очередь
            # отсюда это не рарезолвить даже pythonpath'ом, а
            # наверху расположенный такой чекаут у пользователя кривой
            # положит вообще весь node_runner, а тут просто варнинг
            from contrib.job_submitter.job_submitter.jobs.transfers.transfer2cloud import (
                Transfer2CloudJob,
            )

            Transfer2CloudJob(
                source_path=source_path,
                staging_path=cls.get_jd_sync_dir(),
                exp_name=exp_name,
            ).submit()
        except Exception as error:
            Logger.warning(f"{source_path} stays in JD_SYNC_DIR: {error}")
            return False
        return True
