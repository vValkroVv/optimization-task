import json
from pathlib import Path

import pytest

from node_runner.src.job_dispatcher import JD, jd_data


pytestmark = pytest.mark.unit
CHECKPOINT_ID = "12345678-1234-5678-1234-567812345678"
RECORD_ID = "87654321-4321-8765-4321-876543218765"


def _autometrics_record(run_uuid: str, record_id: str = RECORD_ID) -> dict:
    return {
        "id": record_id,
        "action": "autometrics",
        "checkpoint_id": CHECKPOINT_ID,
        "data": {"run_uuid": run_uuid, "updated_at": "2026-01-01T00:00:00Z"},
    }


def _write_jd_data(path: Path, history: list | None = None) -> None:
    path.write_text(
        json.dumps(
            {
                "checkpoint_info": {
                    "id": CHECKPOINT_ID,
                    "nfs_path": "/checkpoints/model",
                    "created_at": "2026-01-01T00:00:00Z",
                    "step": 1,
                    "checkpoint_type": "hf",
                    "checkpoint_creation_type": "sft",
                },
                "history": history or [],
            }
        )
    )


class TestJdData:
    def test_collect_checkpoint_jd_data_returns_none_for_missing_or_invalid_data(
        self, tmp_path: Path
    ):
        assert JD.collect_checkpoint_jd_data(tmp_path) is None

        jd_data_path = tmp_path / "jd_data.json"
        jd_data_path.write_text("invalid json")
        assert JD.collect_checkpoint_jd_data(tmp_path) is None

        jd_data_path.write_text("{}")
        assert JD.collect_checkpoint_jd_data(tmp_path) is None

    def test_collect_checkpoint_jd_data_returns_validated_record(self, tmp_path: Path):
        _write_jd_data(tmp_path / "jd_data.json")

        jd_data_dict = JD.collect_checkpoint_jd_data(tmp_path)

        assert jd_data_dict is not None
        assert jd_data_dict[0]["checkpoint_info"]["id"] == CHECKPOINT_ID
        assert jd_data_dict[0]["history"] == []

    def test_ensure_autometrics_record_skips_missing_jd_data(self, tmp_path: Path):
        JD.ensure_autometrics_record(tmp_path, "run-1", None, None, None)
        JD.add_evaluator_metrics(tmp_path, "run-1", "bfcl", {"score": 0.5})

        assert not (tmp_path / "jd_data.json").exists()

    def test_record_is_created_once_and_collects_the_metrics_of_every_evaluator(
        self, tmp_path: Path
    ):
        jd_data_path = tmp_path / "jd_data.json"
        _write_jd_data(jd_data_path, [{"action": "convertation", "data": {}}])

        JD.ensure_autometrics_record(
            tmp_path,
            "run-1",
            Path("/checkpoints/source"),
            Path("/checkpoints/hf"),
            "test-run",
        )
        JD.ensure_autometrics_record(tmp_path, "run-1", None, None, None)
        JD.add_evaluator_metrics(tmp_path, "run-1", "bfcl", {"score": 0.5})
        JD.add_evaluator_metrics(tmp_path, "run-1", "bfcl", {"accuracy": 0.8})
        JD.add_evaluator_metrics(tmp_path, "run-1", "harness_bench", {"score": 0.9})

        saved = json.loads(jd_data_path.read_text())
        assert len(saved["history"]) == 2
        record = saved["history"][1]
        assert record["action"] == "autometrics"
        assert record["checkpoint_id"] == CHECKPOINT_ID
        assert record["data"]["run_uuid"] == "run-1"
        assert record["data"]["checkpoint_path"] == "/checkpoints/source"
        assert record["data"]["hf_checkpoint_path"] == "/checkpoints/hf"
        assert record["data"]["run_name"] == "test-run"
        assert record["data"]["evaluators"] == ["bfcl", "harness_bench"]
        assert record["data"]["metrics"] == {
            "bfcl": {"score": 0.5, "accuracy": 0.8},
            "harness_bench": {"score": 0.9},
        }

    def test_ensure_autometrics_record_creates_run_without_optional_paths(
        self, tmp_path: Path
    ):
        jd_data_path = tmp_path / "jd_data.json"
        _write_jd_data(jd_data_path)

        JD.ensure_autometrics_record(tmp_path, "run-1", None, None, None)

        record = json.loads(jd_data_path.read_text())["history"][0]
        assert record["data"]["checkpoint_path"] is None
        assert record["data"]["hf_checkpoint_path"] is None
        assert record["data"]["run_name"] is None

    @pytest.mark.parametrize(
        ("env", "expected_workspace"),
        [
            pytest.param(
                {
                    "MLS_JOB_WORKSPACE_NAME": "job-workspace",
                    "MLS_NTB_WORKSPACE_NAME": "notebook-workspace",
                },
                "job-workspace",
                id="job-workspace-first",
            ),
            pytest.param(
                {"MLS_NTB_WORKSPACE_NAME": "notebook-workspace"},
                "notebook-workspace",
                id="notebook-workspace-fallback",
            ),
            pytest.param({}, None, id="no-workspace"),
        ],
    )
    def test_ensure_autometrics_record_stores_job_with_env_workspace(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        env: dict[str, str],
        expected_workspace: str | None,
    ):
        jd_data_path = tmp_path / "jd_data.json"
        _write_jd_data(jd_data_path)
        monkeypatch.delenv("MLS_JOB_WORKSPACE_NAME", raising=False)
        monkeypatch.delenv("MLS_NTB_WORKSPACE_NAME", raising=False)
        for name, value in env.items():
            monkeypatch.setenv(name, value)

        JD.ensure_autometrics_record(
            tmp_path, "run-1", None, None, None, job_id="job-1"
        )

        record = json.loads(jd_data_path.read_text())["history"][0]
        assert record["data"]["job"] == {
            "name": "job-1",
            "workspace": expected_workspace,
        }

    def test_ensure_autometrics_record_logs_invalid_checkpoint_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        jd_data_path = tmp_path / "jd_data.json"
        jd_data_path.write_text(json.dumps({"checkpoint_info": {}, "history": []}))
        errors = []
        monkeypatch.setattr(jd_data.Logger, "error", errors.append)

        JD.ensure_autometrics_record(tmp_path, "run-1", None, None, None)

        assert len(errors) == 1
        assert "Error while adding an autometrics record" in errors[0]


class TestEnsureAutometricsRecordWithoutJdData:
    @pytest.mark.parametrize(
        ("config", "expected_checkpoint_type"),
        [
            pytest.param(
                {"quantization_config": {"quant_method": "fp8"}}, "hf_fp8", id="fp8"
            ),
            pytest.param(
                {"quantization_config": {"quant_method": "fbgemm_fp8"}},
                "hf_fp8",
                id="fp8-variant",
            ),
            pytest.param({}, "hf_bf16", id="not-quantized"),
        ],
    )
    def test_creates_jd_data_for_the_measured_checkpoint(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        config: dict,
        expected_checkpoint_type: str,
    ):
        checkpoint_path = tmp_path / "model" / "ep0_ba1500"
        hf_checkpoint_path = checkpoint_path / "artifacts" / "hf_checkpoint"
        hf_checkpoint_path.mkdir(parents=True)
        (hf_checkpoint_path / "config.json").write_text(json.dumps(config))
        artifacts_path = JD.checkpoint_artifacts_path(hf_checkpoint_path)
        monkeypatch.setenv("MLS_JOB_WORKSPACE_NAME", "wm-train")

        JD.ensure_autometrics_record(
            artifacts_path, "run-1", checkpoint_path, hf_checkpoint_path, "test-run"
        )
        JD.add_evaluator_metrics(artifacts_path, "run-1", "bfcl", {"score": 0.5})

        saved = json.loads((artifacts_path / "jd_data.json").read_text())
        checkpoint_info = saved["checkpoint_info"]
        checkpoint_id = checkpoint_info.pop("id")
        checkpoint_info.pop("created_at")
        assert checkpoint_info == {
            "nfs_path": str(hf_checkpoint_path),
            "step": 1500,
            "checkpoint_type": expected_checkpoint_type,
            "checkpoint_creation_type": "convertation",
            "workspace": "wm-train",
            "parent_id": None,
        }
        [record] = saved["history"]
        assert record["checkpoint_id"] == checkpoint_id
        assert record["data"]["metrics"] == {"bfcl": {"score": 0.5}}
        assert JD.collect_checkpoint_jd_data(artifacts_path) is not None

    def test_writes_nothing_without_config_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        hf_checkpoint_path = tmp_path / "hf_checkpoint"
        hf_checkpoint_path.mkdir()
        errors = []
        monkeypatch.setattr(jd_data.Logger, "error", errors.append)

        JD.ensure_autometrics_record(tmp_path, "run-1", None, hf_checkpoint_path, None)

        assert not (tmp_path / "jd_data.json").exists()
        assert len(errors) == 1


class TestAddEvaluatorMetricsWithoutRecord:
    def test_warns_and_writes_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        jd_data_path = tmp_path / "jd_data.json"
        _write_jd_data(jd_data_path)
        warnings = []
        monkeypatch.setattr(jd_data.Logger, "warning", warnings.append)

        JD.add_evaluator_metrics(tmp_path, "run-1", "bfcl", {"score": 0.5})

        assert json.loads(jd_data_path.read_text())["history"] == []
        assert warnings == [
            f"No autometrics record for run run-1 in {jd_data_path}; "
            "skip recording 'bfcl' metrics"
        ]


class TestMetricsFilesMtimeNs:
    def test_reads_the_newest_mtime_recorded_for_the_evaluator(self, tmp_path: Path):
        first = _autometrics_record("run-1")
        first["data"]["metrics_files_mtime_ns"] = {"lm_eval": 100, "obm": 700}
        second = _autometrics_record("run-2")
        second["data"]["metrics_files_mtime_ns"] = {"lm_eval": 300}
        _write_jd_data(tmp_path / "jd_data.json", [first, second])

        assert JD.read_metrics_files_mtime_ns(tmp_path, "lm_eval") == 300
        assert JD.read_metrics_files_mtime_ns(tmp_path, "obm") == 700
        assert JD.read_metrics_files_mtime_ns(tmp_path, "mera") == 0

    @pytest.mark.parametrize(
        "history",
        [
            pytest.param(None, id="no-jd-data-file"),
            pytest.param([_autometrics_record("run-1")], id="record-without-the-mtime"),
            pytest.param(
                [
                    {
                        "action": "convertation",
                        "data": {"metrics_files_mtime_ns": {"lm_eval": 300}},
                    }
                ],
                id="mtime-of-another-action",
            ),
        ],
    )
    def test_reads_zero_without_a_recorded_measurement(
        self, tmp_path: Path, history: list | None
    ):
        if history is not None:
            _write_jd_data(tmp_path / "jd_data.json", history)

        assert JD.read_metrics_files_mtime_ns(tmp_path, "lm_eval") == 0

    def test_add_evaluator_metrics_records_the_mtime_per_evaluator(
        self, tmp_path: Path
    ):
        jd_data_path = tmp_path / "jd_data.json"
        _write_jd_data(jd_data_path)

        JD.ensure_autometrics_record(tmp_path, "run-1", None, None, None)
        JD.add_evaluator_metrics(
            tmp_path, "run-1", "lm_eval", {"score": 0.5}, metrics_files_mtime_ns=100
        )
        JD.add_evaluator_metrics(tmp_path, "run-1", "obm", {"score": 0.9})

        record = json.loads(jd_data_path.read_text())["history"][0]
        assert record["data"]["metrics_files_mtime_ns"] == {"lm_eval": 100}

    def test_add_evaluator_metrics_logs_a_broken_record(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        jd_data_path = tmp_path / "jd_data.json"
        record = _autometrics_record("run-1")
        record["data"]["metrics"] = "broken"
        _write_jd_data(jd_data_path, [record])
        before = jd_data_path.read_text()
        errors = []
        monkeypatch.setattr(jd_data.Logger, "error", errors.append)

        JD.add_evaluator_metrics(tmp_path, "run-1", "lm_eval", {"score": 0.5})

        assert len(errors) == 1
        assert "Error while updating jd_data.json history" in errors[0]
        assert jd_data_path.read_text() == before


class TestReadCheckpointTypeAndId:
    def test_returns_type_and_id_from_jd_data(self, tmp_path: Path):
        _write_jd_data(tmp_path / "jd_data.json")

        assert JD.read_checkpoint_type_and_id(tmp_path) == ("hf", CHECKPOINT_ID)

    @pytest.mark.parametrize(
        "checkpoint_info",
        [
            pytest.param(None, id="no-jd-data-file"),
            pytest.param({}, id="empty-checkpoint-info"),
            pytest.param({"id": CHECKPOINT_ID}, id="missing-type"),
            pytest.param({"checkpoint_type": "hf"}, id="missing-id"),
        ],
    )
    def test_returns_none_when_type_or_id_is_unknown(
        self, tmp_path: Path, checkpoint_info: dict | None
    ):
        if checkpoint_info is not None:
            (tmp_path / "jd_data.json").write_text(
                json.dumps({"checkpoint_info": checkpoint_info})
            )

        assert JD.read_checkpoint_type_and_id(tmp_path) is None


class TestEnsureAutometricsRecordPamConfig:
    def test_new_run_uploads_pam_config_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        jd_data_path = tmp_path / "jd_data.json"
        _write_jd_data(jd_data_path)
        uploads = []

        def fake_upload_pam_config(
            config_path, hf_checkpoint_path, checkpoint_type, checkpoint_id, record_id
        ):
            uploads.append(
                (
                    config_path,
                    hf_checkpoint_path,
                    checkpoint_type,
                    checkpoint_id,
                    record_id,
                )
            )
            return "gigaeye-s3:bucket/prefix/pam_config.yaml"

        monkeypatch.setattr(JD, "upload_pam_config", fake_upload_pam_config)

        for _ in range(2):
            JD.ensure_autometrics_record(
                tmp_path,
                "run-1",
                None,
                Path("/checkpoints/hf"),
                None,
                pam_config_path=Path("/checkpoints/hf/artifacts/pam_config.yaml"),
            )

        record = json.loads(jd_data_path.read_text())["history"][0]
        assert uploads == [
            (
                Path("/checkpoints/hf/artifacts/pam_config.yaml"),
                Path("/checkpoints/hf"),
                "hf",
                CHECKPOINT_ID,
                record["id"],
            )
        ]
        assert record["data"]["pam_config_s3_path"] == (
            "gigaeye-s3:bucket/prefix/pam_config.yaml"
        )

    def test_new_run_keeps_metrics_when_pam_config_upload_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        jd_data_path = tmp_path / "jd_data.json"
        _write_jd_data(jd_data_path)
        uploads = []

        def failed_upload_pam_config(*args):
            uploads.append(args)
            return None

        monkeypatch.setattr(JD, "upload_pam_config", failed_upload_pam_config)

        JD.ensure_autometrics_record(
            tmp_path,
            "run-1",
            None,
            Path("/checkpoints/hf"),
            None,
            pam_config_path=Path("/checkpoints/hf/artifacts/pam_config.yaml"),
        )
        JD.add_evaluator_metrics(tmp_path, "run-1", "bfcl", {"score": 0.5})

        assert len(uploads) == 1
        record = json.loads(jd_data_path.read_text())["history"][0]
        assert record["data"]["pam_config_s3_path"] is None
        assert record["data"]["metrics"] == {"bfcl": {"score": 0.5}}

    @pytest.mark.parametrize(
        ("checkpoint_type", "hf_checkpoint_path", "pam_config_path"),
        [
            pytest.param("hf", Path("/checkpoints/hf"), None, id="no-pam-config-path"),
            pytest.param(
                "hf",
                None,
                Path("/checkpoints/hf/artifacts/pam_config.yaml"),
                id="no-hf-checkpoint-path",
            ),
            pytest.param(
                None,
                Path("/checkpoints/hf"),
                Path("/checkpoints/hf/artifacts/pam_config.yaml"),
                id="no-checkpoint-type",
            ),
        ],
    )
    def test_new_run_skips_pam_config_upload_without_info_about_checkpoint(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        checkpoint_type: str | None,
        hf_checkpoint_path: Path | None,
        pam_config_path: Path | None,
    ):
        jd_data_path = tmp_path / "jd_data.json"
        jd_data_path.write_text(
            json.dumps(
                {
                    "checkpoint_info": {
                        "id": CHECKPOINT_ID,
                        "checkpoint_type": checkpoint_type,
                    }
                }
            )
        )
        uploads = []
        monkeypatch.setattr(JD, "upload_pam_config", lambda *args: uploads.append(args))

        JD.ensure_autometrics_record(
            tmp_path,
            "run-1",
            None,
            hf_checkpoint_path,
            None,
            pam_config_path=pam_config_path,
        )
        JD.add_evaluator_metrics(tmp_path, "run-1", "bfcl", {"score": 0.5})

        assert uploads == []
        record = json.loads(jd_data_path.read_text())["history"][0]
        assert record["data"]["pam_config_s3_path"] is None
        assert record["data"]["metrics"] == {"bfcl": {"score": 0.5}}


class TestCopyJdDataForQuantizedCheckpoint:
    def test_creates_jd_data_for_new_checkpoint_from_source(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        source_artifacts_path = tmp_path / "checkpoint" / "artifacts"
        source_artifacts_path.mkdir(parents=True)
        (source_artifacts_path / "jd_data.json").write_text(
            json.dumps(
                {
                    "checkpoint_info": {
                        "id": CHECKPOINT_ID,
                        "nfs_path": "/checkpoints/model",
                        "created_at": "2026-01-01T00:00:00Z",
                        "step": 1,
                        "checkpoint_type": "hf",
                        "checkpoint_creation_type": "sft",
                    },
                    "model_info": {"name": "model"},
                    "history": [{"action": "convertation", "data": {}}],
                }
            )
        )
        quantized_checkpoint_path = tmp_path / "hf_checkpoint_fp8"
        monkeypatch.setattr(JD, "_new_uuid", lambda: "quantized-checkpoint-id")
        monkeypatch.setattr(JD, "_utc_now", lambda: "2026-09-16T00:00:00Z")

        JD.copy_jd_data_for_quantized_checkpoint(
            source_artifacts_path, quantized_checkpoint_path
        )

        jd_data_path = quantized_checkpoint_path / "artifacts" / "jd_data.json"
        assert json.loads(jd_data_path.read_text()) == {
            "checkpoint_info": {
                "id": "quantized-checkpoint-id",
                "nfs_path": str(quantized_checkpoint_path),
                "created_at": "2026-09-16T00:00:00Z",
                "step": 1,
                "checkpoint_type": "hf_fp8",
                "checkpoint_creation_type": "convertation",
                "parent_id": CHECKPOINT_ID,
            },
            "model_info": {"name": "model"},
            "history": [],
        }

    def test_keeps_existing_jd_data(self, tmp_path: Path):
        source_artifacts_path = tmp_path / "checkpoint" / "artifacts"
        source_artifacts_path.mkdir(parents=True)
        _write_jd_data(source_artifacts_path / "jd_data.json")
        quantized_checkpoint_path = tmp_path / "hf_checkpoint_fp8"
        jd_data_path = quantized_checkpoint_path / "artifacts" / "jd_data.json"
        jd_data_path.parent.mkdir(parents=True)
        existing = {"checkpoint_info": {"id": "fp8-checkpoint-id"}, "history": []}
        jd_data_path.write_text(json.dumps(existing))

        JD.copy_jd_data_for_quantized_checkpoint(
            source_artifacts_path, quantized_checkpoint_path
        )

        assert json.loads(jd_data_path.read_text()) == existing

    def test_skips_when_source_has_no_jd_data(self, tmp_path: Path):
        quantized_checkpoint_path = tmp_path / "hf_checkpoint_fp8"

        JD.copy_jd_data_for_quantized_checkpoint(
            tmp_path / "checkpoint" / "artifacts", quantized_checkpoint_path
        )

        assert not (quantized_checkpoint_path / "artifacts" / "jd_data.json").exists()


class TestMeasurementCachePaths:
    def test_read_measurement_keys_returns_the_checkpoint_and_the_record(
        self, tmp_path: Path
    ):
        _write_jd_data(tmp_path / "jd_data.json", [_autometrics_record("run-1")])

        assert JD.read_measurement_keys(tmp_path, "run-1") == (
            "hf",
            CHECKPOINT_ID,
            RECORD_ID,
        )

    @pytest.mark.parametrize(
        "history",
        [
            pytest.param([], id="no-history"),
            pytest.param([_autometrics_record("run-2")], id="another-run"),
            pytest.param(
                [{"action": "autometrics", "data": {"run_uuid": "run-1"}}],
                id="record-without-id",
            ),
        ],
    )
    def test_read_measurement_keys_returns_none_without_a_record(
        self, tmp_path: Path, history: list
    ):
        _write_jd_data(tmp_path / "jd_data.json", history)

        assert JD.read_measurement_keys(tmp_path, "run-1") is None

    def test_add_metrics_cache_per_measurement_stores_paths_per_evaluator(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        jd_data_path = tmp_path / "jd_data.json"
        _write_jd_data(jd_data_path, [_autometrics_record("run-1")])
        monkeypatch.setattr(JD, "_utc_now", lambda: "2026-09-16T00:00:00Z")

        JD.add_metrics_cache_per_measurement(
            tmp_path, "run-1", "obm", {"cache_s3_path": "gigaeye-s3:bucket/obm/scores"}
        )
        JD.add_metrics_cache_per_measurement(
            tmp_path,
            "run-1",
            "lm_eval",
            {"cache_dir_s3_path": "gigaeye-s3:bucket/lm_eval"},
        )

        assert json.loads(jd_data_path.read_text())["history"][0]["data"] == {
            "run_uuid": "run-1",
            "updated_at": "2026-09-16T00:00:00Z",
            "metrics_cache_per_measurement": {
                "obm": {"cache_s3_path": "gigaeye-s3:bucket/obm/scores"},
                "lm_eval": {"cache_dir_s3_path": "gigaeye-s3:bucket/lm_eval"},
            },
        }

    def test_add_metrics_cache_per_measurement_keeps_paths_of_an_earlier_attempt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        jd_data_path = tmp_path / "jd_data.json"
        _write_jd_data(jd_data_path, [_autometrics_record("run-1")])
        monkeypatch.setattr(JD, "_utc_now", lambda: "2026-09-16T00:00:00Z")

        JD.add_metrics_cache_per_measurement(
            tmp_path,
            "run-1",
            "obm",
            {
                "cache_dir_s3_path": "gigaeye-s3:bucket/obm/cache",
                "cache_s3_path": "gigaeye-s3:bucket/obm/scores",
            },
        )
        JD.add_metrics_cache_per_measurement(
            tmp_path, "run-1", "obm", {"cache_s3_path": "gigaeye-s3:bucket/obm/scores"}
        )

        record = json.loads(jd_data_path.read_text())["history"][0]
        assert record["data"]["metrics_cache_per_measurement"] == {
            "obm": {
                "cache_dir_s3_path": "gigaeye-s3:bucket/obm/cache",
                "cache_s3_path": "gigaeye-s3:bucket/obm/scores",
            }
        }

    def test_add_metrics_cache_per_measurement_keeps_jd_data_for_an_unknown_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        jd_data_path = tmp_path / "jd_data.json"
        _write_jd_data(jd_data_path, [_autometrics_record("run-1")])
        before = jd_data_path.read_text()
        warnings = []
        monkeypatch.setattr(jd_data.Logger, "warning", warnings.append)

        JD.add_metrics_cache_per_measurement(
            tmp_path, "run-2", "obm", {"cache_s3_path": "gigaeye-s3:bucket/obm/scores"}
        )

        assert jd_data_path.read_text() == before
        assert warnings == [
            f"No autometrics record for run run-2 in {jd_data_path}; "
            "skip recording 'obm' cache paths"
        ]

    def test_add_metrics_cache_per_measurement_writes_nothing_without_jd_data(
        self, tmp_path: Path
    ):
        JD.add_metrics_cache_per_measurement(
            tmp_path, "run-1", "obm", {"cache_s3_path": "gigaeye-s3:bucket/obm/scores"}
        )

        assert not (tmp_path / "jd_data.json").exists()
