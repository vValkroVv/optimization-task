import hashlib
import shutil
from pathlib import Path
from typing import Optional, Sequence

from ..logger import Logger
from .s3_uploader import ARTIFACTS_BUCKET, ARTIFACTS_PREFIX, S3ArtifactUploader

PAPAYA_TRANSFER_URL = "papayaeye-papaya:transfer2cloud"


class JDArtifacts:
    @classmethod
    def _checkpoint_metrics_key(
        cls,
        hf_checkpoint_path: Path,
        checkpoint_type: Optional[str],
        checkpoint_id: Optional[str],
    ) -> str:
        _, model_name, epoch, step = cls.parse_checkpoint_path(hf_checkpoint_path)
        checkpoint_dir = cls.find_checkpoint_dir(hf_checkpoint_path)
        checkpoint_name = (
            checkpoint_dir.name if checkpoint_dir else f"ep{epoch}_ba{step}"
        )
        key = f"{ARTIFACTS_PREFIX}/{model_name}/{checkpoint_name}/artifacts/metrics"
        if checkpoint_type:
            return f"{key}/{checkpoint_type}/{checkpoint_id}"
        path_hash = hashlib.sha1(str(Path(hf_checkpoint_path).resolve()).encode())
        return f"{key}/unregistered/{path_hash.hexdigest()[:12]}"

    @classmethod
    def artifact_key_prefix(
        cls,
        hf_checkpoint_path: Path,
        checkpoint_type: Optional[str],
        checkpoint_id: Optional[str],
        shooter_name: str,
        shooter_version: str,
    ) -> str:
        """Deterministic S3 key prefix for this checkpoint's metrics of one shooter.

        Uses the checkpoint parser shared by all evaluators and keys the metrics by
        the JD checkpoint type and id, so the axes stay separated:
            <prefix>/<model>/<step>/artifacts/metrics/<checkpoint_type>/<checkpoint_id>/<shooter>/<version>
            └ many checkpoints per model ┘            └────────── checkpoint ─────────┘ └ shooter ┘└ version ┘
        """
        return (
            f"{cls._checkpoint_metrics_key(hf_checkpoint_path, checkpoint_type, checkpoint_id)}"
            f"/{shooter_name}/{shooter_version}"
        )

    @classmethod
    def record_key_prefix(
        cls,
        hf_checkpoint_path: Path,
        checkpoint_type: str,
        checkpoint_id: str,
        record_id: str,
    ) -> str:
        return (
            f"{cls._checkpoint_metrics_key(hf_checkpoint_path, checkpoint_type, checkpoint_id)}"
            f"/{record_id}"
        )

    @classmethod
    def upload_pam_config(
        cls,
        config_path: Path,
        hf_checkpoint_path: Path,
        checkpoint_type: str,
        checkpoint_id: str,
        record_id: str,
    ) -> Optional[str]:
        key_prefix = cls.record_key_prefix(
            hf_checkpoint_path, checkpoint_type, checkpoint_id, record_id
        )
        uploaded = cls.upload_artifact_file(config_path, key_prefix)
        return cls.gigaeye_url(key_prefix, Path(config_path).name) if uploaded else None

    @classmethod
    def upload_offline(
        cls,
        local_dir: Path,
        rel_paths: Sequence[Path],
        key_prefix: str,
        extra_files: Sequence[Path] = (),
    ) -> bool:
        jd_sync_dir = cls.get_jd_sync_dir()
        if jd_sync_dir is None:
            Logger.warning("No internet available; skipping S3 upload")
            return False
        staging = jd_sync_dir / key_prefix
        sources = [(Path(local_dir) / rel_path, rel_path) for rel_path in rel_paths]
        sources += [(Path(path), Path(Path(path).name)) for path in extra_files]
        try:
            for source, rel_path in sources:
                (staging / rel_path).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, staging / rel_path)
        except OSError as error:
            Logger.error(f"Failed to stage {key_prefix} in JD_SYNC_DIR: {error}")
            return False
        return cls.transfer_to_cloud(staging, "pam-s3-transfer")

    @classmethod
    def upload_artifact_file(cls, local_path: Path, key_prefix: str) -> bool:
        if not cls.internet_is_available():
            return cls.upload_offline(
                Path(local_path).parent, [Path(Path(local_path).name)], key_prefix
            )
        return S3ArtifactUploader().upload_file(local_path, key_prefix)

    @classmethod
    def gigaeye_url(cls, key_prefix: str, suffix: str) -> str:
        return f"{cls.gigaeye_dir_url(key_prefix)}/{suffix}"

    @classmethod
    def gigaeye_dir_url(cls, key_prefix: str) -> str:
        jd_sync_dir = cls.get_jd_sync_dir()
        if jd_sync_dir is not None and not cls.internet_is_available():
            return (
                f"{PAPAYA_TRANSFER_URL}/{jd_sync_dir.parent.name}/"
                f"{jd_sync_dir.name}/{key_prefix}"
            )
        return f"gigaeye-s3:{ARTIFACTS_BUCKET}/{key_prefix}"

    @classmethod
    def upload_artifacts(
        cls, local_dir: Path, key_prefix: str, extra_files: Sequence[Path] = ()
    ) -> bool:
        """Upload ``local_dir`` to S3 under ``key_prefix``, merging
        with anything already present at the destination prefix.

        Destination layout (the prefix is built by :meth:`artifact_key_prefix`):
            s3://<bucket>/<ARTIFACTS_PREFIX>/<model>/<checkpoint>/artifacts/metrics/<shooter>/<version>/...
        """
        if not cls.internet_is_available():
            return cls.upload_offline(
                local_dir,
                [
                    path.relative_to(local_dir)
                    for path in Path(local_dir).rglob("*")
                    if path.is_file()
                ],
                key_prefix,
                extra_files,
            )
        uploader = S3ArtifactUploader()
        return uploader.upload_dir_merge(local_dir, key_prefix) and all(
            uploader.upload_file(path, key_prefix) for path in extra_files
        )

    @classmethod
    def upload_artifact_files(
        cls,
        local_dir: Path,
        rel_paths: Sequence[Path],
        key_prefix: str,
        extra_files: Sequence[Path] = (),
    ) -> bool:
        if not cls.internet_is_available():
            return cls.upload_offline(local_dir, rel_paths, key_prefix, extra_files)
        uploader = S3ArtifactUploader()
        return (
            not rel_paths
            or uploader.upload_files_merge(local_dir, rel_paths, key_prefix)
        ) and all(uploader.upload_file(path, key_prefix) for path in extra_files)

    @classmethod
    def copy_artifact_files(
        cls, src_key_prefix: str, rel_paths: Sequence[Path], dst_key_prefix: str
    ) -> bool:
        if not cls.internet_is_available():
            Logger.warning("No internet available; skipping S3 copy")
            return False
        return S3ArtifactUploader().copy_files_merge(
            src_key_prefix, rel_paths, dst_key_prefix
        )

    @classmethod
    def read_artifact(cls, key: str) -> Optional[bytes]:
        return S3ArtifactUploader().get_object_bytes(key)
