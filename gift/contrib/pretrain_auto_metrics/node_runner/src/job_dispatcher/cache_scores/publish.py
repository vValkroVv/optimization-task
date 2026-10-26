import tempfile
from pathlib import Path
from typing import Callable, Optional

from ...config_model import ModelInfoConfig
from ...logger import Logger
from ..measurement import MeasurementFiles, MeasurementUpload
from ..submetrics import SubmetricGroup
from .base import (
    CacheScoresBuilder,
    CacheScoresSpec,
    iter_compressed_jsonl_bytes,
    merge_key,
)
from .scalar import ScalarCacheScoresBuilder


class JDCacheScores:
    CI_TMP_DIR = Path("/home/user/tmp")

    @classmethod
    def _ci_tmp_dir(cls) -> tempfile.TemporaryDirectory:
        cls.CI_TMP_DIR.mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(dir=cls.CI_TMP_DIR)

    @classmethod
    def send_evaluator_metrics_with_scores(
        cls,
        *,
        scores: ScalarCacheScoresBuilder,
        metrics_dir: Path,
        hf_checkpoint_path: Path,
        shooter_name: str,
        shooter_version: str,
        model_info: ModelInfoConfig,
        extra_body: Optional[dict] = None,
        submetric_groups: list[SubmetricGroup] | None = None,
    ) -> None:
        if not shooter_version or shooter_version == "PAM_FILE":
            Logger.error(f"{shooter_name}: no producer version; withholding JD update")
            return
        for rows in scores.records.values():
            for row in rows:
                row["uid"] = f"{shooter_version}:{row['uid']}"
        paths = {}
        try:
            scores.iter_source_files()
            version_file = Path(metrics_dir) / "shooter_version.txt"
            version_file.write_text(shooter_version + "\n", encoding="utf-8")
            paths = cls.upload_metrics_and_cache_scores(
                make_builder=lambda: scores,
                metrics_dir=metrics_dir,
                hf_checkpoint_path=hf_checkpoint_path,
                shooter_name=shooter_name,
                version_file=str(version_file),
            )
        except OSError as error:
            Logger.error(f"{shooter_name}: failed to prepare CI artifacts: {error}")
        if "cache_s3_path" not in paths:
            Logger.warning(
                f"{shooter_name}: CI upload failed; sending metrics without CI"
            )
        # JD keeps an old pointer for None; an empty string clears it.
        cache_path = paths.get("cache_s3_path", "")
        cls.send_evaluator_metrics(
            hf_checkpoint_path=hf_checkpoint_path,
            metrics_dir=metrics_dir,
            jd_metrics=scores.metrics,
            shooter_name=shooter_name,
            shooter_version=shooter_version,
            model_info=model_info,
            extra_body={**paths, **(extra_body or {}), "cache_s3_path": cache_path},
            submetric_groups=submetric_groups,
        )

    @classmethod
    def upload_metrics_and_cache_scores(
        cls,
        make_builder: Callable[[], CacheScoresBuilder],
        metrics_dir: Path,
        hf_checkpoint_path: Path,
        shooter_name: str,
        version_file: str,
        uploaded: Optional[MeasurementUpload] = None,
    ) -> dict[str, str]:
        metrics_dir = Path(metrics_dir)
        key_prefix = cls._cache_scores_key_prefix(
            hf_checkpoint_path, shooter_name, version_file
        )
        if key_prefix is None:
            Logger.warning("S3 key prefix is unavailable; skipping S3 upload")
            return {}
        (metrics_dir / CacheScoresSpec.output_filename).unlink(missing_ok=True)
        with cls._ci_tmp_dir() as scores_dir:
            builder, scores_path, remote_scores_stale = cls._build_cache_scores(
                make_builder, Path(scores_dir), key_prefix
            )
            extra_files = [scores_path] if scores_path else []
            if not cls._upload_metrics_dir(
                metrics_dir, key_prefix, uploaded, extra_files
            ):
                return cls._pointer_without_scores(remote_scores_stale)
        paths = cls._cache_s3_paths(metrics_dir, key_prefix, scores_path, builder)
        if "cache_s3_path" not in paths:
            paths.update(cls._pointer_without_scores(remote_scores_stale))
        return paths

    @staticmethod
    def _pointer_without_scores(remote_scores_stale: bool) -> dict[str, str]:
        # JD keeps its cache_s3_path when the field is absent and clears it on "".
        # Keeping it is only safe when the remote file has no rows of what this run
        # re-measured: otherwise JD would pair new metrics with old answers.
        if remote_scores_stale:
            Logger.warning("cache_scores not published; clearing the JD pointer")
            return {"cache_s3_path": ""}
        return {}

    @classmethod
    def _upload_metrics_dir(
        cls,
        metrics_dir: Path,
        key_prefix: str,
        uploaded: Optional[MeasurementUpload],
        extra_files: list[Path],
    ) -> bool:
        if uploaded is None or not cls.copy_artifact_files(
            uploaded.key_prefix, uploaded.rel_paths, key_prefix
        ):
            return cls.upload_artifacts(metrics_dir, key_prefix, extra_files)

        copied = set(uploaded.rel_paths)
        rest = [
            path.relative_to(metrics_dir)
            for path in metrics_dir.rglob("*")
            if path.is_file() and path.relative_to(metrics_dir) not in copied
        ]
        return not (rest or extra_files) or cls.upload_artifact_files(
            metrics_dir, rest, key_prefix, extra_files
        )

    @classmethod
    def publish_metrics_cache_per_measurement(
        cls,
        make_builder: Callable[[set[Path]], Optional[CacheScoresBuilder]],
        measurement: MeasurementFiles,
        metrics_dir: Path,
        hf_checkpoint_path: Path,
        shooter_name: str,
        run_uuid: str,
    ) -> Optional[MeasurementUpload]:
        try:
            artifacts_path = cls.checkpoint_artifacts_path(hf_checkpoint_path)
            key_prefix = cls._measurement_key_prefix(
                hf_checkpoint_path, artifacts_path, shooter_name, run_uuid
            )
            if key_prefix is None:
                Logger.warning(
                    f"No checkpoint info or no record for run {run_uuid} in "
                    f"{artifacts_path / 'jd_data.json'}; skipping "
                    f"'{shooter_name}' metrics cache per measurement"
                )
                return None

            metrics_dir = Path(metrics_dir)
            changed = measurement.changed()
            if not changed:
                Logger.warning(
                    f"Run {run_uuid} produced no files in {metrics_dir}; "
                    f"no '{shooter_name}' metrics cache per measurement"
                )
                return None

            builder = make_builder({metrics_dir / rel_path for rel_path in changed})
            if builder is None:
                return None

            cache_files = cls._files_under(changed, builder.cache_subdir)
            with cls._ci_tmp_dir() as scores_dir:
                scores_path = builder.build(Path(scores_dir))
                extra_files = [scores_path] if scores_path else []
                if not (cache_files or extra_files) or not cls.upload_artifact_files(
                    metrics_dir, cache_files, key_prefix, extra_files
                ):
                    Logger.error(
                        f"Nothing uploaded to {key_prefix}; "
                        f"no '{shooter_name}' metrics cache for run {run_uuid}"
                    )
                    return None

            paths = {}
            if cache_files:
                paths["cache_dir_s3_path"] = cls._cache_dir_s3_path(
                    key_prefix, builder.cache_subdir
                )
            if scores_path is not None:
                paths["cache_s3_path"] = cls.gigaeye_url(key_prefix, scores_path.name)
            cls.add_metrics_cache_per_measurement(
                artifacts_path,
                run_uuid,
                shooter_name,
                paths,
            )
            return MeasurementUpload(key_prefix, cache_files) if cache_files else None
        except Exception as e:
            Logger.error(f"Failed to publish '{shooter_name}' metrics cache: {e}")
            return None

    @staticmethod
    def _files_under(rel_paths: list[Path], cache_subdir: Optional[str]) -> list[Path]:
        if cache_subdir is None:
            return rel_paths
        return [
            rel_path for rel_path in rel_paths if rel_path.is_relative_to(cache_subdir)
        ]

    @classmethod
    def _measurement_key_prefix(
        cls,
        hf_checkpoint_path: Path,
        artifacts_path: Path,
        shooter_name: str,
        run_uuid: str,
    ) -> Optional[str]:
        keys = cls.read_measurement_keys(artifacts_path, run_uuid)
        if keys is None:
            return None

        checkpoint_type, checkpoint_id, record_id = keys
        record_prefix = cls.record_key_prefix(
            hf_checkpoint_path, checkpoint_type, checkpoint_id, record_id
        )
        return f"{record_prefix}/{shooter_name}"

    @classmethod
    def _cache_scores_key_prefix(
        cls, hf_checkpoint_path: Path, shooter_name: str, version_file: str
    ) -> Optional[str]:
        shooter_version = cls.get_shooter_version(version_file)
        if not shooter_version or shooter_version == "PAM_FILE":
            return None
        return cls.ci_key_prefix(hf_checkpoint_path, shooter_name, shooter_version)

    @classmethod
    def ci_key_prefix(
        cls, hf_checkpoint_path: Path, shooter_name: str, shooter_version: str
    ) -> Optional[str]:
        if not cls.reports_to_jd:
            Logger.info("Answers come from a recording; skip uploading CI")
            return None
        checkpoint_type, checkpoint_id = cls.read_checkpoint_type_and_id(
            Path(hf_checkpoint_path, "artifacts")
        ) or (None, None)
        return cls.artifact_key_prefix(
            hf_checkpoint_path,
            checkpoint_type,
            checkpoint_id,
            shooter_name,
            shooter_version,
        )

    @classmethod
    def _build_cache_scores(
        cls,
        make_builder: Callable[[], CacheScoresBuilder],
        out_dir: Path,
        key_prefix: str,
    ) -> tuple[Optional[CacheScoresBuilder], Optional[Path], bool]:
        """Build the scores file; return the builder, the file and whether the remote
        file may now be stale. The builder is ``None`` when it could not be created,
        the file when it was not built.

        Stale means it has rows this run re-measured, or it could not be checked.
        """
        try:
            builder = make_builder()
        except Exception as e:
            Logger.error(f"Failed to create cache_scores builder: {e}")
            return None, None, True
        try:
            filename = builder.spec.output_filename
            if not (builder.spec.merge_keys and cls.internet_is_available()):
                scores_path = builder.build(out_dir)
                return builder, scores_path, scores_path is None
            key = f"{key_prefix}/{filename}"
            existing = cls.read_artifact(key)
            if existing is None:
                Logger.error(f"Could not read {key}; skipping cache_scores")
                return builder, None, True
            if not existing:
                return builder, builder.build(out_dir), False
            builder.existing_rows_loader = lambda: iter_compressed_jsonl_bytes(existing)
            scores_path = builder.build(out_dir)
            remote_scores_stale = any(
                merge_key(builder.spec.merge_keys, row) in builder.rebuilt_keys
                for row in builder.existing_rows()
            )
            return builder, scores_path, remote_scores_stale
        except Exception as e:
            Logger.error(f"Failed to build cache_scores file: {e}")
            return builder, None, True

    @classmethod
    def _cache_s3_paths(
        cls,
        metrics_dir: Path,
        key_prefix: str,
        scores_path: Optional[Path],
        builder: Optional[CacheScoresBuilder],
    ) -> dict[str, str]:
        paths = {}
        if scores_path is not None:
            paths["cache_s3_path"] = cls.gigaeye_url(key_prefix, scores_path.name)
        if builder is None:
            return paths
        cache_dir_url = cls._cache_dir_url(
            metrics_dir, key_prefix, builder.cache_subdir
        )
        if cache_dir_url is not None:
            paths["cache_dir_s3_path"] = cache_dir_url
        return paths

    @classmethod
    def _cache_dir_url(
        cls, metrics_dir: Path, key_prefix: str, cache_subdir: Optional[str]
    ) -> Optional[str]:
        if cache_subdir is not None and not any(
            path.is_file() for path in (metrics_dir / cache_subdir).rglob("*")
        ):
            return None
        return cls._cache_dir_s3_path(key_prefix, cache_subdir)

    @classmethod
    def _cache_dir_s3_path(cls, key_prefix: str, cache_subdir: Optional[str]) -> str:
        if cache_subdir is None:
            return cls.gigaeye_dir_url(key_prefix)
        return cls.gigaeye_url(key_prefix, cache_subdir)
