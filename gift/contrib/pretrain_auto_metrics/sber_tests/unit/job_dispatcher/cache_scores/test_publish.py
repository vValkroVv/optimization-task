import hashlib
import os
from pathlib import Path

import pytest
from node_runner.src.job_dispatcher import JD
from node_runner.src.job_dispatcher.cache_scores import ObmCacheScoresBuilder
from node_runner.src.job_dispatcher.cache_scores.base import CacheScoresSpec
from node_runner.src.job_dispatcher.measurement import (
    MeasurementFiles,
    MeasurementUpload,
)

pytestmark = pytest.mark.unit

PREFIX = "prefix"
SCORES_FILENAME = "cache_scores.jsonl.zst"
RECORD_ID = "record-uuid"
MEASUREMENT_PREFIX = f"{PREFIX}/{RECORD_ID}/obm"
RECORDED_MTIME_NS = 1_700_000_000 * 1_000_000_000


@pytest.fixture(autouse=True)
def ci_tmp_dir(monkeypatch: pytest.MonkeyPatch, tmp_path_factory) -> None:
    monkeypatch.setattr(JD, "CI_TMP_DIR", tmp_path_factory.mktemp("ci_tmp"))


def make_builder_over(metrics_dir: Path):
    return lambda: ObmCacheScoresBuilder(metrics_dir)


def write_cache_file(metrics_dir: Path, benchmark: str = "alpha") -> None:
    cache_dir = metrics_dir / "cache"
    cache_dir.mkdir(exist_ok=True)
    (cache_dir / f"{benchmark}.jsonl").write_text(
        '{"uid": "local"}\n', encoding="utf-8"
    )


def write_recorded_cache_file(metrics_dir: Path, benchmark: str = "alpha") -> None:
    write_cache_file(metrics_dir, benchmark)
    path = metrics_dir / "cache" / f"{benchmark}.jsonl"
    os.utime(path, ns=(RECORDED_MTIME_NS, RECORDED_MTIME_NS))


def stub_measurement_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the per-measurement upload at ``<PREFIX>/<RECORD_ID>/<shooter>``."""
    monkeypatch.setattr(
        JD, "read_measurement_keys", lambda *args: ("hf", "ckpt-uuid", RECORD_ID)
    )
    monkeypatch.setattr(JD, "record_key_prefix", lambda *args: f"{PREFIX}/{RECORD_ID}")


@pytest.mark.parametrize("shooter_version", [None, "PAM_FILE"])
def test_key_prefix_is_none_without_version(
    monkeypatch: pytest.MonkeyPatch, shooter_version: str | None
) -> None:
    monkeypatch.setattr(JD, "get_shooter_version", lambda path: shooter_version)
    monkeypatch.setattr(
        JD, "read_checkpoint_type_and_id", lambda _path: ("hf", "ckpt-uuid")
    )

    assert JD._cache_scores_key_prefix(Path("/hf"), "obm", "version.txt") is None


@pytest.mark.parametrize(
    ("checkpoint_suffix", "expected_checkpoint_name"),
    [
        (("ep3_ba42", "artifacts", "hf_checkpoint"), "ep3_ba42"),
        (("global_step_42", "actor", "huggingface"), "global_step_42"),
    ],
)
def test_key_prefix_separates_checkpoint_shooter_and_version(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    checkpoint_suffix: tuple[str, ...],
    expected_checkpoint_name: str,
) -> None:
    hf_checkpoint_path = tmp_path / "model"
    for path_part in checkpoint_suffix:
        hf_checkpoint_path /= path_part
    monkeypatch.setattr(JD, "get_shooter_version", lambda path: "abc123")
    monkeypatch.setattr(
        JD, "read_checkpoint_type_and_id", lambda _path: ("hf", "ckpt-uuid")
    )

    prefix = JD._cache_scores_key_prefix(hf_checkpoint_path, "obm", "version.txt")

    assert prefix == (
        f"pretrain-auto-metrics-artifacts/model/{expected_checkpoint_name}/"
        "artifacts/metrics/hf/ckpt-uuid/obm/abc123"
    )


def test_key_prefix_without_jd_checkpoint_info_is_unique_per_checkpoint_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(JD, "get_shooter_version", lambda path: "abc123")
    monkeypatch.setattr(JD, "read_checkpoint_type_and_id", lambda _path: None)
    hf_a = tmp_path / "a" / "model" / "ep3_ba42" / "artifacts" / "hf_checkpoint"
    hf_b = tmp_path / "b" / "model" / "ep3_ba42" / "artifacts" / "hf_checkpoint"

    prefix_a = JD._cache_scores_key_prefix(hf_a, "obm", "version.txt")
    prefix_b = JD._cache_scores_key_prefix(hf_b, "obm", "version.txt")

    path_hash = hashlib.sha1(str(hf_a.resolve()).encode()).hexdigest()[:12]
    assert prefix_a == (
        "pretrain-auto-metrics-artifacts/model/ep3_ba42/artifacts/metrics"
        f"/unregistered/{path_hash}/obm/abc123"
    )
    assert prefix_b != prefix_a


def test_replay_uploads_no_ci(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(JD, "reports_to_jd", False)
    monkeypatch.setattr(
        JD, "read_checkpoint_type_and_id", lambda _path: ("hf", "ckpt-uuid")
    )

    assert JD.ci_key_prefix(Path("/hf"), "obm", "abc123") is None


class FakeBuilder:
    def __init__(
        self,
        merge_keys: tuple[str, ...] = ("benchmark",),
        cache_subdir: str | None = None,
    ) -> None:
        self.spec = CacheScoresSpec(
            output_filename=SCORES_FILENAME, merge_keys=merge_keys
        )
        self.cache_subdir = cache_subdir
        self.existing_rows_loader = None
        self.rebuilt_keys = {("alpha",)}

    def build(self, out_dir: Path) -> Path:
        return out_dir / SCORES_FILENAME

    def existing_rows(self):
        return self.existing_rows_loader() if self.existing_rows_loader else ()


class FailingBuilder(FakeBuilder):
    def build(self, out_dir: Path) -> Path:
        raise RuntimeError("build failed")


def test_build_feeds_the_remote_scores_file_to_the_builder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    requested = []

    def read_artifact(key: str) -> bytes:
        requested.append(key)
        return b"compressed"

    monkeypatch.setattr(JD, "internet_is_available", lambda: True)
    monkeypatch.setattr(JD, "read_artifact", read_artifact)
    monkeypatch.setattr(
        "node_runner.src.job_dispatcher.cache_scores.publish.iter_compressed_jsonl_bytes",
        lambda data: iter([{"benchmark": "alpha", "data": data.decode()}]),
    )
    builder = FakeBuilder()

    assert JD._build_cache_scores(lambda: builder, tmp_path, PREFIX) == (
        builder,
        tmp_path / SCORES_FILENAME,
        True,
    )
    assert requested == [f"{PREFIX}/{SCORES_FILENAME}"]
    assert list(builder.existing_rows_loader()) == [
        {"benchmark": "alpha", "data": "compressed"}
    ]


@pytest.mark.parametrize(
    ("internet", "existing", "merge_keys"),
    [
        (False, b"compressed", ("benchmark",)),
        (True, b"", ("benchmark",)),
        (True, b"compressed", ()),
    ],
)
def test_build_leaves_the_builder_unmerged_without_remote_scores(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    internet: bool,
    existing: bytes,
    merge_keys: tuple[str, ...],
) -> None:
    monkeypatch.setattr(JD, "internet_is_available", lambda: internet)
    monkeypatch.setattr(JD, "read_artifact", lambda key: existing)
    builder = FakeBuilder(merge_keys)

    assert JD._build_cache_scores(lambda: builder, tmp_path, PREFIX) == (
        builder,
        tmp_path / SCORES_FILENAME,
        False,
    )
    assert builder.existing_rows_loader is None


def test_build_logs_a_failing_builder_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    errors = []

    def failing_builder():
        raise RuntimeError("no metrics file")

    monkeypatch.setattr(
        "node_runner.src.job_dispatcher.cache_scores.publish.Logger.error",
        errors.append,
    )

    assert JD._build_cache_scores(failing_builder, tmp_path, PREFIX) == (
        None,
        None,
        True,
    )
    assert errors == ["Failed to create cache_scores builder: no metrics file"]


def test_a_failed_build_still_names_the_cache_folder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    write_cache_file(tmp_path)
    errors = []
    monkeypatch.setattr(
        "node_runner.src.job_dispatcher.cache_scores.publish.Logger.error",
        errors.append,
    )

    builder, scores_path, _ = JD._build_cache_scores(
        lambda: FailingBuilder(merge_keys=(), cache_subdir="cache"),
        tmp_path,
        PREFIX,
    )

    assert scores_path is None
    assert errors == ["Failed to build cache_scores file: build failed"]
    assert JD._cache_s3_paths(tmp_path, PREFIX, scores_path, builder) == {
        "cache_dir_s3_path": f"gigaeye-s3:gigaeye-pretrain-checkpoints/{PREFIX}/cache"
    }


def test_publish_does_not_upload_a_previous_file_when_the_s3_read_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    write_cache_file(tmp_path)
    scores_path = tmp_path / SCORES_FILENAME
    scores_path.write_bytes(b"previous")
    uploaded = []

    monkeypatch.setattr(JD, "_cache_scores_key_prefix", lambda *args: PREFIX)
    monkeypatch.setattr(JD, "internet_is_available", lambda: True)
    monkeypatch.setattr(JD, "read_artifact", lambda key: None)
    monkeypatch.setattr(
        JD,
        "upload_artifacts",
        lambda local_dir, key_prefix, extra_files: (
            uploaded.extend([*local_dir.rglob("*"), *extra_files]) or True
        ),
    )

    paths = JD.upload_metrics_and_cache_scores(
        make_builder=make_builder_over(tmp_path),
        metrics_dir=tmp_path,
        hf_checkpoint_path=tmp_path / "hf",
        shooter_name="obm",
        version_file="version.txt",
    )

    assert paths["cache_s3_path"] == ""
    assert scores_path not in uploaded


@pytest.mark.parametrize(
    ("cache_subdir", "expected_dir_url"),
    [
        ("cache", f"gigaeye-s3:gigaeye-pretrain-checkpoints/{PREFIX}/cache"),
        (None, f"gigaeye-s3:gigaeye-pretrain-checkpoints/{PREFIX}"),
    ],
)
def test_cache_paths_name_the_scores_file_and_the_cache_folder(
    tmp_path: Path, cache_subdir: str | None, expected_dir_url: str
) -> None:
    write_cache_file(tmp_path)
    scores_path = tmp_path / SCORES_FILENAME
    scores_path.write_bytes(b"scores")

    assert JD._cache_s3_paths(
        tmp_path, PREFIX, scores_path, FakeBuilder(cache_subdir=cache_subdir)
    ) == {
        "cache_s3_path": (
            f"gigaeye-s3:gigaeye-pretrain-checkpoints/{PREFIX}/{SCORES_FILENAME}"
        ),
        "cache_dir_s3_path": expected_dir_url,
    }


def test_cache_paths_omit_what_was_not_produced(tmp_path: Path) -> None:
    (tmp_path / "cache").mkdir()

    builder = FakeBuilder(cache_subdir="cache")

    assert JD._cache_s3_paths(tmp_path, PREFIX, None, builder) == {}


def test_cache_paths_are_empty_without_a_builder(tmp_path: Path) -> None:
    assert JD._cache_s3_paths(tmp_path, PREFIX, None, None) == {}


def test_publish_uploads_the_metrics_dir_and_returns_its_cache_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    write_cache_file(tmp_path)
    uploaded = {}

    monkeypatch.setattr(JD, "_cache_scores_key_prefix", lambda *args: PREFIX)
    monkeypatch.setattr(JD, "internet_is_available", lambda: False)
    monkeypatch.setattr(
        JD,
        "upload_artifacts",
        lambda local_dir, key_prefix, extra_files: (
            uploaded.update(
                {
                    "local_dir": local_dir,
                    "key_prefix": key_prefix,
                    "extra_files": [path.name for path in extra_files],
                    "scores_off_metrics_dir": tmp_path not in extra_files[0].parents,
                }
            )
            or True
        ),
    )

    paths = JD.upload_metrics_and_cache_scores(
        make_builder=make_builder_over(tmp_path),
        metrics_dir=tmp_path,
        hf_checkpoint_path=tmp_path / "hf",
        shooter_name="obm",
        version_file="version.txt",
    )

    assert uploaded == {
        "local_dir": tmp_path,
        "key_prefix": PREFIX,
        "extra_files": [SCORES_FILENAME],
        "scores_off_metrics_dir": True,
    }
    assert not (tmp_path / SCORES_FILENAME).exists()
    assert paths == {
        "cache_s3_path": (
            f"gigaeye-s3:gigaeye-pretrain-checkpoints/{PREFIX}/{SCORES_FILENAME}"
        ),
        "cache_dir_s3_path": f"gigaeye-s3:gigaeye-pretrain-checkpoints/{PREFIX}/cache",
    }


def test_publish_without_a_destination_skips_the_upload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    write_cache_file(tmp_path)
    warnings = []

    monkeypatch.setattr(JD, "_cache_scores_key_prefix", lambda *args: None)
    monkeypatch.setattr(
        JD,
        "upload_artifacts",
        lambda local_dir, key_prefix: pytest.fail("must not upload without a prefix"),
    )
    monkeypatch.setattr(
        "node_runner.src.job_dispatcher.cache_scores.publish.Logger.warning",
        warnings.append,
    )

    paths = JD.upload_metrics_and_cache_scores(
        make_builder=make_builder_over(tmp_path),
        metrics_dir=tmp_path,
        hf_checkpoint_path=tmp_path / "hf",
        shooter_name="obm",
        version_file="version.txt",
    )

    assert paths == {}
    assert warnings == ["S3 key prefix is unavailable; skipping S3 upload"]
    assert not (tmp_path / SCORES_FILENAME).exists()


def test_publish_returns_nothing_when_the_upload_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    write_cache_file(tmp_path)

    monkeypatch.setattr(JD, "_cache_scores_key_prefix", lambda *args: PREFIX)
    monkeypatch.setattr(JD, "internet_is_available", lambda: False)
    monkeypatch.setattr(JD, "upload_artifacts", lambda *args: False)

    paths = JD.upload_metrics_and_cache_scores(
        make_builder=make_builder_over(tmp_path),
        metrics_dir=tmp_path,
        hf_checkpoint_path=tmp_path / "hf",
        shooter_name="obm",
        version_file="version.txt",
    )

    assert paths == {}


@pytest.mark.parametrize(
    ("remote_benchmark", "expected"),
    [("alpha", {"cache_s3_path": ""}), ("beta", {})],
)
def test_publish_only_clears_a_stale_remote_pointer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    remote_benchmark: str,
    expected: dict[str, str],
) -> None:
    monkeypatch.setattr(JD, "_cache_scores_key_prefix", lambda *args: PREFIX)
    monkeypatch.setattr(JD, "internet_is_available", lambda: True)
    monkeypatch.setattr(JD, "read_artifact", lambda key: b"remote")
    monkeypatch.setattr(
        "node_runner.src.job_dispatcher.cache_scores.publish.iter_compressed_jsonl_bytes",
        lambda data: iter([{"benchmark": remote_benchmark, "uid": "old"}]),
    )
    monkeypatch.setattr(JD, "upload_artifacts", lambda *args: False)

    paths = JD.upload_metrics_and_cache_scores(
        make_builder=FakeBuilder,
        metrics_dir=tmp_path,
        hf_checkpoint_path=tmp_path / "hf",
        shooter_name="obm",
        version_file="version.txt",
    )

    assert paths == expected


def test_publish_clears_stale_pointer_when_scores_build_fails_after_upload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(JD, "_cache_scores_key_prefix", lambda *args: PREFIX)
    monkeypatch.setattr(JD, "_build_cache_scores", lambda *args: (None, None, True))
    monkeypatch.setattr(JD, "upload_artifacts", lambda *args: True)

    paths = JD.upload_metrics_and_cache_scores(
        make_builder=FakeBuilder,
        metrics_dir=tmp_path,
        hf_checkpoint_path=tmp_path / "hf",
        shooter_name="obm",
        version_file="version.txt",
    )

    assert paths["cache_s3_path"] == ""


@pytest.mark.parametrize("version", ["", "PAM_FILE"])
def test_scalar_publication_requires_producer_version(monkeypatch, tmp_path, version):
    from unittest.mock import Mock
    from node_runner.src.job_dispatcher.cache_scores.scalar import (
        ScalarCacheScoresBuilder,
    )

    upload = Mock()
    send = Mock()
    monkeypatch.setattr(JD, "upload_metrics_and_cache_scores", upload)
    monkeypatch.setattr(JD, "send_evaluator_metrics", send)
    JD.send_evaluator_metrics_with_scores(
        scores=ScalarCacheScoresBuilder(tmp_path, {"quality": 0.5}),
        metrics_dir=tmp_path,
        hf_checkpoint_path=tmp_path,
        shooter_name="tau2",
        shooter_version=version,
        model_info=None,
    )
    upload.assert_not_called()
    send.assert_not_called()


def test_scalar_publication_clears_stale_cache_pointer_when_upload_fails(
    monkeypatch, tmp_path
):
    from node_runner.src.job_dispatcher.cache_scores.scalar import (
        ScalarCacheScoresBuilder,
    )

    sent = []
    monkeypatch.setattr(JD, "upload_metrics_and_cache_scores", lambda **kwargs: {})
    monkeypatch.setattr(
        JD, "send_evaluator_metrics", lambda **kwargs: sent.append(kwargs)
    )
    JD.send_evaluator_metrics_with_scores(
        scores=ScalarCacheScoresBuilder(tmp_path, {"quality": 0.5}),
        metrics_dir=tmp_path,
        hf_checkpoint_path=tmp_path,
        shooter_name="tau2",
        shooter_version="abc123",
        model_info=None,
    )
    assert sent[0]["extra_body"] == {"cache_s3_path": ""}


def test_scalar_publication_passes_uploaded_cache_pointer(monkeypatch, tmp_path):
    from node_runner.src.job_dispatcher.cache_scores.scalar import (
        ScalarCacheScoresBuilder,
    )

    scores = ScalarCacheScoresBuilder(tmp_path, {"quality": 0.5})
    scores.add_metric("quality", [("a", 0.0), ("b", 1.0)], {"task": "quality"})
    sent = []
    monkeypatch.setattr(
        JD,
        "upload_metrics_and_cache_scores",
        lambda **kwargs: {"cache_s3_path": "gigaeye-s3:bucket/scores"},
    )
    monkeypatch.setattr(
        JD, "send_evaluator_metrics", lambda **kwargs: sent.append(kwargs)
    )

    JD.send_evaluator_metrics_with_scores(
        scores=scores,
        metrics_dir=tmp_path,
        hf_checkpoint_path=tmp_path,
        shooter_name="tau2",
        shooter_version="abc123",
        model_info=None,
    )

    assert sent[0]["extra_body"] == {"cache_s3_path": "gigaeye-s3:bucket/scores"}
    assert all(row["uid"].startswith("abc123:") for row in scores.records["quality"])


@pytest.mark.parametrize("filename", ["ci_samples.jsonl", "shooter_version.txt"])
def test_scalar_publication_keeps_metrics_when_ci_file_cannot_be_written(
    monkeypatch, tmp_path, filename
):
    from unittest.mock import Mock
    from node_runner.src.job_dispatcher.cache_scores.scalar import (
        ScalarCacheScoresBuilder,
    )

    (tmp_path / filename).mkdir()
    upload, send = Mock(), Mock()
    monkeypatch.setattr(JD, "upload_metrics_and_cache_scores", upload)
    monkeypatch.setattr(JD, "send_evaluator_metrics", send)
    JD.send_evaluator_metrics_with_scores(
        scores=ScalarCacheScoresBuilder(tmp_path, {"quality": 0.5}),
        metrics_dir=tmp_path,
        hf_checkpoint_path=tmp_path,
        shooter_name="tau2",
        shooter_version="abc123",
        model_info=None,
    )
    upload.assert_not_called()
    assert send.call_args.kwargs["jd_metrics"] == {"quality": 0.5}
    assert send.call_args.kwargs["extra_body"]["cache_s3_path"] == ""


def test_publish_per_measurement_uploads_only_what_this_run_produced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    write_recorded_cache_file(tmp_path, "from_a_previous_run")
    measurement = MeasurementFiles(tmp_path, RECORDED_MTIME_NS)
    write_cache_file(tmp_path, "from_this_run")
    record_keys: list[tuple[str, str, str]] = []
    uploaded_files: list[tuple] = []
    recorded: list[tuple] = []

    monkeypatch.setattr(
        JD, "read_measurement_keys", lambda *args: ("hf", "ckpt-uuid", RECORD_ID)
    )
    monkeypatch.setattr(
        JD,
        "record_key_prefix",
        lambda hf_path, checkpoint_type, checkpoint_id, record_id: (
            record_keys.append((checkpoint_type, checkpoint_id, record_id))
            or f"{PREFIX}/{record_id}"
        ),
    )
    monkeypatch.setattr(
        JD,
        "upload_artifact_files",
        lambda local_dir, rel_paths, key_prefix, extra_files: (
            uploaded_files.append(
                (
                    local_dir,
                    list(rel_paths),
                    key_prefix,
                    [path.name for path in extra_files],
                )
            )
            or True
        ),
    )
    monkeypatch.setattr(
        JD,
        "add_metrics_cache_per_measurement",
        lambda artifacts_path, run_uuid, shooter_name, paths: recorded.append(
            (artifacts_path, run_uuid, shooter_name, paths)
        ),
    )

    upload = JD.publish_metrics_cache_per_measurement(
        make_builder=lambda files: FakeBuilder(cache_subdir="cache"),
        measurement=measurement,
        metrics_dir=tmp_path,
        hf_checkpoint_path=tmp_path / "hf",
        shooter_name="obm",
        run_uuid="run-1",
    )

    assert record_keys == [("hf", "ckpt-uuid", RECORD_ID)]
    assert uploaded_files == [
        (
            tmp_path,
            [Path("cache/from_this_run.jsonl")],
            MEASUREMENT_PREFIX,
            [SCORES_FILENAME],
        )
    ]
    assert recorded == [
        (
            tmp_path / "hf" / "artifacts",
            "run-1",
            "obm",
            {
                "cache_dir_s3_path": (
                    f"gigaeye-s3:gigaeye-pretrain-checkpoints/{MEASUREMENT_PREFIX}/cache"
                ),
                "cache_s3_path": (
                    "gigaeye-s3:gigaeye-pretrain-checkpoints/"
                    f"{MEASUREMENT_PREFIX}/{SCORES_FILENAME}"
                ),
            },
        )
    ]
    assert upload == MeasurementUpload(
        MEASUREMENT_PREFIX, [Path("cache/from_this_run.jsonl")]
    )


def test_publish_per_measurement_skips_a_run_that_reused_existing_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    write_recorded_cache_file(tmp_path)
    measurement = MeasurementFiles(tmp_path, RECORDED_MTIME_NS)
    warnings = []

    monkeypatch.setattr(
        JD, "read_measurement_keys", lambda *args: ("hf", "ckpt-uuid", RECORD_ID)
    )
    monkeypatch.setattr(JD, "record_key_prefix", lambda *args: f"{PREFIX}/{RECORD_ID}")
    monkeypatch.setattr(
        JD,
        "upload_artifact_files",
        lambda *args: pytest.fail("must not upload files of an earlier run"),
    )
    monkeypatch.setattr(
        JD,
        "add_metrics_cache_per_measurement",
        lambda *args: pytest.fail("must not record cache paths without new files"),
    )
    monkeypatch.setattr(
        "node_runner.src.job_dispatcher.cache_scores.publish.Logger.warning",
        warnings.append,
    )

    upload = JD.publish_metrics_cache_per_measurement(
        make_builder=lambda files: pytest.fail("must not build without new files"),
        measurement=measurement,
        metrics_dir=tmp_path,
        hf_checkpoint_path=tmp_path / "hf",
        shooter_name="obm",
        run_uuid="run-1",
    )

    assert upload is None
    assert warnings == [
        f"Run run-1 produced no files in {tmp_path}; "
        "no 'obm' metrics cache per measurement"
    ]


def test_publish_per_measurement_skips_a_run_without_a_jd_data_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    write_recorded_cache_file(tmp_path, "from_a_previous_run")
    measurement = MeasurementFiles(tmp_path, RECORDED_MTIME_NS)
    write_cache_file(tmp_path, "from_this_run")
    warnings = []

    monkeypatch.setattr(JD, "read_measurement_keys", lambda *args: None)
    monkeypatch.setattr(
        JD,
        "upload_artifact_files",
        lambda *args: pytest.fail("must not upload without a measurement prefix"),
    )
    monkeypatch.setattr(
        "node_runner.src.job_dispatcher.cache_scores.publish.Logger.warning",
        warnings.append,
    )

    upload = JD.publish_metrics_cache_per_measurement(
        make_builder=lambda files: FakeBuilder(cache_subdir="cache"),
        measurement=measurement,
        metrics_dir=tmp_path,
        hf_checkpoint_path=tmp_path / "hf",
        shooter_name="obm",
        run_uuid="run-1",
    )

    assert upload is None
    assert warnings == [
        "No checkpoint info or no record for run run-1 in "
        f"{tmp_path / 'hf' / 'artifacts' / 'jd_data.json'}; "
        "skipping 'obm' metrics cache per measurement"
    ]


def test_upload_metrics_dir_copies_the_measurement_files_and_uploads_the_rest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    write_cache_file(tmp_path)
    (tmp_path / "results.json").write_text("{}", encoding="utf-8")
    scores_path = tmp_path / "ci" / SCORES_FILENAME
    copied: list[tuple] = []
    uploaded: list[tuple] = []

    monkeypatch.setattr(
        JD,
        "copy_artifact_files",
        lambda src_key_prefix, rel_paths, dst_key_prefix: (
            copied.append((src_key_prefix, list(rel_paths), dst_key_prefix)) or True
        ),
    )
    monkeypatch.setattr(
        JD,
        "upload_artifact_files",
        lambda local_dir, rel_paths, key_prefix, extra_files: (
            uploaded.append((local_dir, sorted(rel_paths), key_prefix, extra_files))
            or True
        ),
    )
    monkeypatch.setattr(
        JD,
        "upload_artifacts",
        lambda *args: pytest.fail("must not re-upload the whole metrics dir"),
    )

    assert (
        JD._upload_metrics_dir(
            tmp_path,
            PREFIX,
            MeasurementUpload(MEASUREMENT_PREFIX, [Path("cache/alpha.jsonl")]),
            [scores_path],
        )
        is True
    )
    assert copied == [(MEASUREMENT_PREFIX, [Path("cache/alpha.jsonl")], PREFIX)]
    assert uploaded == [(tmp_path, [Path("results.json")], PREFIX, [scores_path])]


@pytest.mark.parametrize(
    ("uploaded_measurement", "copy_succeeds"),
    [
        pytest.param(None, True, id="nothing-uploaded-per-measurement"),
        pytest.param(
            MeasurementUpload(MEASUREMENT_PREFIX, [Path("cache/alpha.jsonl")]),
            False,
            id="server-side-copy-failed",
        ),
    ],
)
def test_upload_metrics_dir_falls_back_to_uploading_everything(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    uploaded_measurement: MeasurementUpload | None,
    copy_succeeds: bool,
) -> None:
    write_cache_file(tmp_path)
    uploaded: list[tuple] = []

    monkeypatch.setattr(JD, "copy_artifact_files", lambda *args: copy_succeeds)
    monkeypatch.setattr(
        JD,
        "upload_artifact_files",
        lambda *args: pytest.fail("must not upload a partial file list"),
    )
    monkeypatch.setattr(
        JD,
        "upload_artifacts",
        lambda local_dir, key_prefix, extra_files: (
            uploaded.append((local_dir, key_prefix)) or True
        ),
    )

    assert JD._upload_metrics_dir(tmp_path, PREFIX, uploaded_measurement, []) is True
    assert uploaded == [(tmp_path, PREFIX)]


def test_publish_per_measurement_skips_a_shooter_without_a_cache_builder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    measurement = MeasurementFiles(tmp_path)
    write_cache_file(tmp_path, "from_this_run")

    stub_measurement_prefix(monkeypatch)
    monkeypatch.setattr(
        JD,
        "upload_artifact_files",
        lambda *args: pytest.fail("must not upload without a builder"),
    )

    upload = JD.publish_metrics_cache_per_measurement(
        make_builder=lambda files: None,
        measurement=measurement,
        metrics_dir=tmp_path,
        hf_checkpoint_path=tmp_path / "hf",
        shooter_name="token_nll",
        run_uuid="run-1",
    )

    assert upload is None


def test_publish_per_measurement_uploads_every_new_file_without_a_cache_subdir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    measurement = MeasurementFiles(tmp_path)
    write_cache_file(tmp_path, "from_this_run")
    (tmp_path / "results.json").write_text("{}", encoding="utf-8")
    key_prefix = f"{PREFIX}/{RECORD_ID}/lm_eval"
    new_files = [Path("cache/from_this_run.jsonl"), Path("results.json")]
    uploaded_files: list[tuple] = []
    recorded: list[dict] = []

    stub_measurement_prefix(monkeypatch)
    monkeypatch.setattr(
        JD,
        "upload_artifact_files",
        lambda local_dir, rel_paths, prefix, extra_files: (
            uploaded_files.append((local_dir, list(rel_paths), prefix)) or True
        ),
    )
    monkeypatch.setattr(
        JD,
        "add_metrics_cache_per_measurement",
        lambda artifacts_path, run_uuid, shooter_name, paths: recorded.append(paths),
    )

    upload = JD.publish_metrics_cache_per_measurement(
        make_builder=lambda files: FakeBuilder(),
        measurement=measurement,
        metrics_dir=tmp_path,
        hf_checkpoint_path=tmp_path / "hf",
        shooter_name="lm_eval",
        run_uuid="run-1",
    )

    assert uploaded_files == [(tmp_path, new_files, key_prefix)]
    assert recorded == [
        {
            "cache_dir_s3_path": (
                f"gigaeye-s3:gigaeye-pretrain-checkpoints/{key_prefix}"
            ),
            "cache_s3_path": (
                f"gigaeye-s3:gigaeye-pretrain-checkpoints/{key_prefix}/"
                f"{SCORES_FILENAME}"
            ),
        }
    ]
    assert upload == MeasurementUpload(key_prefix, new_files)


def test_publish_per_measurement_records_nothing_when_the_upload_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    measurement = MeasurementFiles(tmp_path)
    write_cache_file(tmp_path, "from_this_run")
    recorded: list[dict] = []
    errors: list[str] = []

    stub_measurement_prefix(monkeypatch)
    monkeypatch.setattr(JD, "upload_artifact_files", lambda *args: False)
    monkeypatch.setattr(
        JD,
        "add_metrics_cache_per_measurement",
        lambda artifacts_path, run_uuid, shooter_name, paths: recorded.append(paths),
    )
    monkeypatch.setattr(
        "node_runner.src.job_dispatcher.cache_scores.publish.Logger.error",
        errors.append,
    )

    upload = JD.publish_metrics_cache_per_measurement(
        make_builder=lambda files: FakeBuilder(cache_subdir="cache"),
        measurement=measurement,
        metrics_dir=tmp_path,
        hf_checkpoint_path=tmp_path / "hf",
        shooter_name="obm",
        run_uuid="run-1",
    )

    assert upload is None
    assert recorded == []
    assert errors == [
        f"Nothing uploaded to {MEASUREMENT_PREFIX}; no 'obm' metrics cache for run run-1"
    ]


def test_publish_per_measurement_logs_a_failing_builder_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    measurement = MeasurementFiles(tmp_path)
    write_cache_file(tmp_path, "from_this_run")
    errors: list[str] = []

    stub_measurement_prefix(monkeypatch)
    monkeypatch.setattr(JD, "upload_artifact_files", lambda *args: True)
    monkeypatch.setattr(
        JD,
        "add_metrics_cache_per_measurement",
        lambda *args: pytest.fail("must not record cache paths for a failed build"),
    )
    monkeypatch.setattr(
        "node_runner.src.job_dispatcher.cache_scores.publish.Logger.error",
        errors.append,
    )

    upload = JD.publish_metrics_cache_per_measurement(
        make_builder=lambda files: FailingBuilder(cache_subdir="cache"),
        measurement=measurement,
        metrics_dir=tmp_path,
        hf_checkpoint_path=tmp_path / "hf",
        shooter_name="obm",
        run_uuid="run-1",
    )

    assert upload is None
    assert errors == ["Failed to publish 'obm' metrics cache: build failed"]


@pytest.mark.parametrize("scores_only", [True, False])
def test_publish_per_measurement_records_only_available_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, scores_only: bool
) -> None:
    measurement = MeasurementFiles(tmp_path, RECORDED_MTIME_NS)
    if scores_only:
        (tmp_path / "results.json").write_text("{}", encoding="utf-8")
    else:
        write_cache_file(tmp_path)
    builder = FakeBuilder(cache_subdir="cache")
    if not scores_only:
        monkeypatch.setattr(builder, "build", lambda out_dir: None)
    stub_measurement_prefix(monkeypatch)
    monkeypatch.setattr(JD, "upload_artifact_files", lambda *args: True)
    recorded = []
    monkeypatch.setattr(
        JD,
        "add_metrics_cache_per_measurement",
        lambda artifacts_path, run_uuid, shooter_name, paths: recorded.append(paths),
    )

    upload = JD.publish_metrics_cache_per_measurement(
        make_builder=lambda files: builder,
        measurement=measurement,
        metrics_dir=tmp_path,
        hf_checkpoint_path=tmp_path / "hf",
        shooter_name="obm",
        run_uuid="run-1",
    )

    if scores_only:
        assert upload is None
        assert recorded == [
            {"cache_s3_path": JD.gigaeye_url(MEASUREMENT_PREFIX, SCORES_FILENAME)}
        ]
    else:
        assert upload == MeasurementUpload(
            MEASUREMENT_PREFIX, [Path("cache/alpha.jsonl")]
        )
        assert recorded == [
            {"cache_dir_s3_path": JD.gigaeye_url(MEASUREMENT_PREFIX, "cache")}
        ]


def test_offline_rerun_clears_ci_pointer_when_cache_cannot_be_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    write_cache_file(metrics_dir)
    staging = tmp_path / "transfer"
    monkeypatch.setattr(JD, "_cache_scores_key_prefix", lambda *args: PREFIX)
    monkeypatch.setattr(JD, "internet_is_available", lambda: False)
    monkeypatch.setattr(JD, "get_jd_sync_dir", lambda: staging)
    monkeypatch.setattr(JD, "transfer_to_cloud", lambda *args: True)

    def publish():
        return JD.upload_metrics_and_cache_scores(
            make_builder=make_builder_over(metrics_dir),
            metrics_dir=metrics_dir,
            hf_checkpoint_path=tmp_path / "hf",
            shooter_name="obm",
            version_file="version.txt",
        )

    first = publish()
    assert first["cache_s3_path"].endswith(SCORES_FILENAME)
    previous = (staging / PREFIX / SCORES_FILENAME).read_bytes()
    (metrics_dir / "cache" / "alpha.jsonl").write_bytes(b"\xff")

    second = publish()

    assert (staging / PREFIX / SCORES_FILENAME).read_bytes() == previous
    assert second["cache_s3_path"] == ""
