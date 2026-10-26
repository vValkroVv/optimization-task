import json
from pathlib import Path

import pytest
from node_runner.src.job_dispatcher import JD
from node_runner.src.config_model import (
    MeraEvaluatorConfig,
    MeraTaskConfig,
    ModelInfoConfig as ModelInfo,
)
from node_runner.src.metrics.evaluators.mera import MeraEvaluator

pytestmark = pytest.mark.unit


MOCK_MERA_METRICS_DIR = (
    Path(__file__).resolve().parents[3] / "test_data" / "mock_mera_metrics"
)


def return_mock_mera_metrics() -> dict[str, float]:
    return {
        "mera/ruhhh_gen/0-shot/em.scoring": 0.42,
        "mera/simplear/2-shot/exact_match.scoring": 0.75,
        "mera/strucom/0-shot/chrf": 0.65,
        "mera/rummlu_gen/1-shot/em.scoring": 0.61,
    }


class TestMeraEvaluatorSetup:
    def test_mera_evaluator_setup_builds_cmds_and_skips_existing(self, tmp_path: Path):
        artifacts_path = tmp_path / "artifacts"
        metrics_dir = artifacts_path / "metrics" / "mera"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / "simplear_2026-01-01.json").write_text("{}")

        evaluator = MeraEvaluator()
        evaluator.setup(
            autometric_runs_config=MeraEvaluatorConfig(
                tasks={
                    "simplear": MeraTaskConfig(),
                    "bps_gen": MeraTaskConfig(),
                    "rummlu_gen": MeraTaskConfig(batch_size=8),
                }
            ),
            artifacts_path=artifacts_path,
            hf_checkpoint_path=tmp_path / "hf_ckpt",
            model_info=ModelInfo(model_class="sft", size="10b", username="test"),
        )

        assert evaluator.metrics_dir == metrics_dir
        assert "simplear" not in evaluator.cmds_to_run
        assert "bps_gen" in evaluator.cmds_to_run
        assert "rummlu_gen" in evaluator.cmds_to_run
        assert evaluator.cmds_to_run["bps_gen"][0] == MeraEvaluator.MERA_VENV_PYTHON
        assert evaluator.cmds_to_run["bps_gen"][2] == "lm_eval"

    def test_mera_evaluator_setup_default_batch_size(self, tmp_path: Path):
        artifacts_path = tmp_path / "artifacts"
        (artifacts_path / "metrics" / "mera").mkdir(parents=True, exist_ok=True)

        evaluator = MeraEvaluator()
        evaluator.setup(
            autometric_runs_config=MeraEvaluatorConfig(
                tasks={
                    "simplear": MeraTaskConfig(),
                }
            ),
            artifacts_path=artifacts_path,
            hf_checkpoint_path=tmp_path / "hf_ckpt",
            model_info=ModelInfo(model_class="sft", size="10b", username="test"),
        )

        cmd = evaluator.cmds_to_run["simplear"]
        assert "--batch_size" in cmd
        bs_idx = cmd.index("--batch_size")
        assert cmd[bs_idx + 1] == "1"

    def test_mera_evaluator_setup_custom_batch_size(self, tmp_path: Path):
        artifacts_path = tmp_path / "artifacts"
        (artifacts_path / "metrics" / "mera").mkdir(parents=True, exist_ok=True)

        evaluator = MeraEvaluator()
        evaluator.setup(
            autometric_runs_config=MeraEvaluatorConfig(
                tasks={
                    "simplear": MeraTaskConfig(batch_size=8),
                }
            ),
            artifacts_path=artifacts_path,
            hf_checkpoint_path=tmp_path / "hf_ckpt",
            model_info=ModelInfo(model_class="sft", size="10b", username="test"),
        )

        cmd = evaluator.cmds_to_run["simplear"]
        bs_idx = cmd.index("--batch_size")
        assert cmd[bs_idx + 1] == "8"

    def test_mera_evaluator_setup_cmd_contains_required_flags(self, tmp_path: Path):
        artifacts_path = tmp_path / "artifacts"
        (artifacts_path / "metrics" / "mera").mkdir(parents=True, exist_ok=True)

        evaluator = MeraEvaluator()
        evaluator.setup(
            autometric_runs_config=MeraEvaluatorConfig(
                tasks={
                    "simplear": MeraTaskConfig(),
                }
            ),
            artifacts_path=artifacts_path,
            hf_checkpoint_path=tmp_path / "hf_ckpt",
            model_info=ModelInfo(model_class="sft", size="10b", username="test"),
        )

        cmd = evaluator.cmds_to_run["simplear"]
        assert "--model" in cmd
        assert "local-chat-completions" in cmd
        assert "--log_samples" in cmd
        assert "--fewshot_as_multiturn" in cmd
        assert "--batch_size" in cmd
        assert "--process_results_workers" not in cmd
        assert "--filter_workers" not in cmd

    def test_mera_evaluator_metadata_batch_size(self, tmp_path: Path):
        artifacts_path = tmp_path / "artifacts"
        (artifacts_path / "metrics" / "mera").mkdir(parents=True, exist_ok=True)

        evaluator = MeraEvaluator()
        evaluator.setup(
            autometric_runs_config=MeraEvaluatorConfig(
                tasks={
                    "simplear": MeraTaskConfig(batch_size=4),
                    "rummlu_gen": MeraTaskConfig(batch_size=16),
                }
            ),
            artifacts_path=artifacts_path,
            hf_checkpoint_path=tmp_path / "hf_ckpt",
            model_info=ModelInfo(model_class="sft", size="10b", username="test"),
        )

        evaluator._save_submission_metadata()

        submissions_dir = evaluator.metrics_dir / "submissions"
        meta = json.loads((submissions_dir / "metadata.json").read_text())
        assert meta["batch_size"] == 16

    def test_metadata_saved_before_generate_submissions(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        artifacts_path = tmp_path / "artifacts"
        (artifacts_path / "metrics" / "mera").mkdir(parents=True, exist_ok=True)

        evaluator = MeraEvaluator()
        evaluator.setup(
            autometric_runs_config=MeraEvaluatorConfig(
                tasks={
                    "simplear": MeraTaskConfig(batch_size=4),
                }
            ),
            artifacts_path=artifacts_path,
            hf_checkpoint_path=tmp_path / "hf_ckpt",
            model_info=ModelInfo(model_class="sft", size="10b", username="test"),
        )
        monkeypatch.setattr(evaluator, "_read_file_stripped", lambda *_: "test-ver")
        monkeypatch.setattr(evaluator, "_query_mera_venv", lambda *_: "test-env")
        monkeypatch.setattr(evaluator, "_send_metrics_to_jd", lambda *_: None)
        monkeypatch.setattr(
            JD,
            "tar_metrics_dir",
            lambda *_a, **_kw: artifacts_path / "metrics" / "mera.tar.gz",
        )
        monkeypatch.setattr(
            JD,
            "symlink_metrics_for_transfer",
            lambda *_a, **_kw: None,
        )

        def fake_generate_submissions():
            meta_path = evaluator.metrics_dir / "submissions" / "metadata.json"
            assert meta_path.exists(), "metadata.json must exist before submissions"
            meta = json.loads(meta_path.read_text())
            assert meta["mera_version"] == "test-ver"

        monkeypatch.setattr(
            evaluator,
            "_generate_submissions",
            fake_generate_submissions,
        )
        evaluator.after_run({})


class TestMeraEvaluatorMetrics:
    def test_mera_evaluator_get_metrics_conversion(self):
        evaluator = MeraEvaluator()
        evaluator.metrics_dir = MOCK_MERA_METRICS_DIR
        converted = evaluator.get_metrics()
        assert converted == return_mock_mera_metrics()

    def test_mera_evaluator_get_metrics_normalizes_chrf(self):
        evaluator = MeraEvaluator()
        evaluator.metrics_dir = MOCK_MERA_METRICS_DIR
        converted = evaluator.get_metrics()
        assert converted["mera/strucom/0-shot/chrf"] == 0.65

    def test_mera_already_calculated(self):
        evaluator = MeraEvaluator()
        assert evaluator._is_already_calculated(
            "results_ruhhh_gen", MOCK_MERA_METRICS_DIR
        )
        assert evaluator._is_already_calculated(
            "results_simplear", MOCK_MERA_METRICS_DIR
        )
        assert not evaluator._is_already_calculated(
            "non_existing_task", MOCK_MERA_METRICS_DIR
        )

    def test_get_metrics_without_metrics_dir(self, monkeypatch: pytest.MonkeyPatch):
        warnings = []
        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.mera.Logger.warning", warnings.append
        )
        evaluator = MeraEvaluator()
        evaluator.metrics_dir = None
        assert evaluator.get_metrics() == {}
        assert warnings

    def test_collect_and_aggregate_single_run(self, tmp_path: Path):
        evaluator = MeraEvaluator()
        metrics_dir = tmp_path / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        run = {
            "n-shot": {"task": 2},
            "results": {
                "simplear": {
                    "exact_match.scoring": 0.7,
                    "exact_match.scoring_stderr": 0.1,
                    "non_numeric": "x",
                }
            },
        }
        (metrics_dir / "a.json").write_text("{}", encoding="utf-8")
        (metrics_dir / "metrics_step_1.json").write_text("{}", encoding="utf-8")
        evaluator._load_json_file = lambda _: run
        out = evaluator._collect_metrics_from_dir(metrics_dir)
        assert out == {"mera/simplear/2-shot/exact_match.scoring": 0.7}

        assert evaluator._aggregate_single_run({"results": None}) == {}


class TestMeraEvaluatorHelpers:
    def test_normalize_metrics_skips_absent_scale_down_keys(self):
        metrics = {"mera/simplear/2-shot/exact_match.scoring": 0.5}
        MeraEvaluator._normalize_metrics(metrics)
        assert metrics["mera/simplear/2-shot/exact_match.scoring"] == 0.5

    def test_mera_helpers_task_parsing_and_matching(self):
        assert (
            MeraEvaluator._extract_task_name(
                "samples_bps_gen_2026-02-17T10-50-09.123456.jsonl"
            )
            == "bps_gen"
        )
        assert MeraEvaluator._extract_task_name("samples_no_date.jsonl") is None
        assert MeraEvaluator._task_matches_category("bps_gen", {"bps"}) is True
        assert MeraEvaluator._task_matches_category("simplear", {"simplear"}) is True
        assert MeraEvaluator._task_matches_category("unknown", {"simplear"}) is False

    def test_read_file_stripped(self, tmp_path: Path):
        file_path = tmp_path / "v.txt"
        file_path.write_text("  value  ", encoding="utf-8")
        assert MeraEvaluator._read_file_stripped(str(file_path)) == "value"
        assert MeraEvaluator._read_file_stripped(str(tmp_path / "missing.txt")) is None

    def test_query_mera_venv_and_model_metadata(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        import subprocess

        evaluator = MeraEvaluator()

        class _Res:
            def __init__(self, returncode=0, stdout="ok\n"):
                self.returncode = returncode
                self.stdout = stdout

        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *args, **kwargs: _Res(returncode=0, stdout="2.7.0\n"),
        )
        assert evaluator._query_mera_venv("print(1)") == "2.7.0"

        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *args, **kwargs: _Res(returncode=1, stdout=""),
        )
        assert evaluator._query_mera_venv("print(1)") is None

        cfg_dir = tmp_path / "model-10b"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        (cfg_dir / "config.json").write_text(
            '{"architectures":["LlamaForCausalLM"],"quantization_config":{"quant_method":"fp8"}}',
            encoding="utf-8",
        )
        evaluator.hf_checkpoint_path = cfg_dir
        evaluator.model_info = ModelInfo(
            model_class="sft", size="fallback", username="u"
        )
        assert evaluator._get_model_architecture() == "LlamaForCausalLM"
        assert evaluator._get_model_precision() == "fp8"
        assert evaluator._parse_model_size() == "10b"

    def test_create_filtered_outputs_dir_and_pack(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = MeraEvaluator()
        evaluator.metrics_dir = tmp_path / "metrics"
        evaluator.metrics_dir.mkdir(parents=True, exist_ok=True)
        (evaluator.metrics_dir / "results_simplear.json").write_text(
            "{}", encoding="utf-8"
        )
        (
            evaluator.metrics_dir / "samples_simplear_2026-02-17T10-50-09.123456.jsonl"
        ).write_text("{}", encoding="utf-8")
        (
            evaluator.metrics_dir / "samples_unknown_2026-02-17T10-50-09.123456.jsonl"
        ).write_text("{}", encoding="utf-8")

        filtered = evaluator._create_filtered_outputs_dir("text")
        assert filtered is not None
        assert (Path(filtered) / "results_simplear.json").exists()
        assert (
            Path(filtered) / "samples_simplear_2026-02-17T10-50-09.123456.jsonl"
        ).exists()
        shutil_rmtree_calls = []
        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.mera.shutil.rmtree",
            lambda p, **kwargs: shutil_rmtree_calls.append(Path(p)),
        )
        assert evaluator._create_filtered_outputs_dir("unknown") is None

        submissions_dir = tmp_path / "submissions"
        submissions_dir.mkdir(parents=True, exist_ok=True)
        (submissions_dir / "text.zip").write_text("x", encoding="utf-8")
        (submissions_dir / "metadata.json").write_text("{}", encoding="utf-8")
        evaluator._pack_all_submissions(submissions_dir)
        assert (submissions_dir / "mera_submissions_all.zip").exists()

    def test_pack_all_submissions_branches_no_zips_and_no_metadata(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        warnings = []
        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.mera.Logger.warning", warnings.append
        )
        evaluator = MeraEvaluator()
        submissions_dir = tmp_path / "submissions_empty"
        submissions_dir.mkdir(parents=True, exist_ok=True)
        evaluator._pack_all_submissions(submissions_dir)
        assert warnings

        submissions_dir2 = tmp_path / "submissions_no_meta"
        submissions_dir2.mkdir(parents=True, exist_ok=True)
        (submissions_dir2 / "text.zip").write_text("x", encoding="utf-8")
        evaluator._pack_all_submissions(submissions_dir2)
        assert (submissions_dir2 / "mera_submissions_all.zip").exists()

    def test_cleanup_stale_submission_archives_logs_warning_on_unlink_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = MeraEvaluator()
        submissions_dir = tmp_path / "submissions_stale"
        submissions_dir.mkdir(parents=True, exist_ok=True)
        (submissions_dir / "text.zip").write_text("x", encoding="utf-8")

        def _boom(_self):
            raise OSError("cannot unlink")

        warnings = []
        monkeypatch.setattr(Path, "unlink", _boom)
        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.mera.Logger.warning", warnings.append
        )

        evaluator._cleanup_stale_submission_archives(submissions_dir)
        assert warnings

    def test_save_submission_metadata_creates_dir_when_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        evaluator = MeraEvaluator()
        evaluator.metrics_dir = tmp_path / "metrics_missing_submissions"
        evaluator.metrics_dir.mkdir(parents=True, exist_ok=True)
        evaluator._used_batch_sizes = set()
        evaluator.model_info = ModelInfo(model_class="sft", size="7b", username="u")
        evaluator.hf_checkpoint_path = None
        monkeypatch.setattr(evaluator, "_read_file_stripped", lambda *_: None)
        monkeypatch.setattr(evaluator, "_query_mera_venv", lambda *_: None)
        evaluator._save_submission_metadata()
        assert (evaluator.metrics_dir / "submissions" / "metadata.json").exists()

    def test_model_metadata_fallback_branches(self, tmp_path: Path):
        evaluator = MeraEvaluator()
        evaluator.model_info = ModelInfo(
            model_class="sft", size="fallback", username="u"
        )

        evaluator.hf_checkpoint_path = None
        assert evaluator._get_model_architecture() is None
        assert evaluator._get_model_precision() is None
        assert evaluator._parse_model_size() == "fallback"

        cfg_dir = tmp_path / "model_dir"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        (cfg_dir / "config.json").write_text(
            '{"model_type":"llama","torch_dtype":"bfloat16"}'
        )
        evaluator.hf_checkpoint_path = cfg_dir
        assert evaluator._get_model_architecture() == "llama"
        assert evaluator._get_model_precision() == "bfloat16"

        (cfg_dir / "config.json").write_text("{bad", encoding="utf-8")
        assert evaluator._get_model_architecture() is None
        assert evaluator._get_model_precision() is None

    def test_create_filtered_outputs_dir_branch_coverage(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = MeraEvaluator()
        evaluator.metrics_dir = tmp_path / "metrics2"
        evaluator.metrics_dir.mkdir(parents=True, exist_ok=True)
        # non-file entry should hit `if not fpath.is_file(): continue`
        (evaluator.metrics_dir / "subdir").mkdir(parents=True, exist_ok=True)
        # regular file that is neither results_* nor samples_* -> 405->398 branch
        (evaluator.metrics_dir / "notes.txt").write_text("x", encoding="utf-8")
        # samples_ file with no date -> _extract_task_name returns None
        (evaluator.metrics_dir / "samples_no_date.jsonl").write_text(
            "{}", encoding="utf-8"
        )

        removed = []
        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.mera.shutil.rmtree",
            lambda p, **kwargs: removed.append(Path(p)),
        )
        out = evaluator._create_filtered_outputs_dir("text")
        assert out is None
        assert removed

    def test_mera_runs_to_bash_list_uses_default_instruction(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = MeraEvaluator()
        monkeypatch.setattr(evaluator, "_read_file_stripped", lambda _p: "sys-msg")
        cfg = MeraEvaluatorConfig(
            tasks={
                "simplear": MeraTaskConfig(model_args={"x": 1}),
            }
        )
        cmds = evaluator._mera_runs_to_bash_list(cfg, tmp_path)
        cmd = cmds["simplear"]
        assert "--system_instruction" in cmd
        idx = cmd.index("--system_instruction")
        assert cmd[idx + 1] == "sys-msg"

    def test_mera_runs_to_bash_list_uses_config_instruction(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = MeraEvaluator()
        # File fallback would return "file-msg"; the config value must win and
        # the file must not be read at all.
        monkeypatch.setattr(
            evaluator,
            "_read_file_stripped",
            lambda _p: pytest.fail("file should not be read when config sets it"),
        )
        cfg = MeraEvaluatorConfig(
            system_instruction="config-msg",
            tasks={
                "simplear": MeraTaskConfig(model_args={"x": 1}),
            },
        )
        cmds = evaluator._mera_runs_to_bash_list(cfg, tmp_path)
        cmd = cmds["simplear"]
        assert "--system_instruction" in cmd
        idx = cmd.index("--system_instruction")
        assert cmd[idx + 1] == "config-msg"

    def test_mera_runs_to_bash_list_per_task_instruction_overrides_default(
        self, tmp_path: Path
    ):
        evaluator = MeraEvaluator()
        cfg = MeraEvaluatorConfig(
            system_instruction="default-msg",
            tasks={
                "simplear": MeraTaskConfig(),
                "rucodeeval": MeraTaskConfig(system_instruction="task-msg"),
            },
        )
        cmds = evaluator._mera_runs_to_bash_list(cfg, tmp_path)

        simplear = cmds["simplear"]
        assert simplear[simplear.index("--system_instruction") + 1] == "default-msg"

        rucodeeval = cmds["rucodeeval"]
        assert rucodeeval[rucodeeval.index("--system_instruction") + 1] == "task-msg"

    def test_generate_submissions_and_after_run(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        import subprocess

        evaluator = MeraEvaluator()
        evaluator.metrics_dir = tmp_path / "metrics"
        evaluator.metrics_dir.mkdir(parents=True, exist_ok=True)
        evaluator.hf_checkpoint_path = tmp_path / "model"
        evaluator.artifacts_path = tmp_path / "artifacts"

        cleanup_dirs = []
        monkeypatch.setattr(
            evaluator,
            "_create_filtered_outputs_dir",
            lambda category: str(tmp_path / f"{category}_filtered"),
        )
        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.mera.shutil.rmtree",
            lambda p, **kwargs: cleanup_dirs.append(Path(p)),
        )

        class _Res:
            def __init__(self, returncode):
                self.returncode = returncode
                self.stderr = "err"

        runs = [_Res(0), _Res(1), _Res(0)]
        monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: runs.pop(0))
        monkeypatch.setattr(evaluator, "_pack_all_submissions", lambda d: None)
        evaluator._generate_submissions()
        assert any("text_filtered" in p.as_posix() for p in cleanup_dirs)

        calls = {}
        monkeypatch.setattr(
            evaluator, "_send_metrics_to_jd", lambda m: calls.update({"send": m})
        )
        monkeypatch.setattr(
            evaluator, "_save_submission_metadata", lambda: calls.update({"meta": True})
        )
        monkeypatch.setattr(
            evaluator, "_generate_submissions", lambda: calls.update({"subs": True})
        )
        monkeypatch.setattr(
            JD,
            "tar_metrics_dir",
            lambda m, a, _name: tmp_path / "m.tar.zst",
        )
        monkeypatch.setattr(
            JD,
            "symlink_metrics_for_transfer",
            lambda tar_path, hf_checkpoint_path, _name: calls.update(
                {"symlink": tar_path}
            ),
        )
        evaluator.after_run({"acc": 1.0})
        assert calls["send"] == {"acc": 1.0}
        assert calls["meta"] is True
        assert calls["subs"] is True


class TestMeraEvaluatorRunAndJd:
    def test_run_executes_pool(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        evaluator = MeraEvaluator()
        evaluator.autometric_runs_config = MeraEvaluatorConfig(pool_max_parallel=3)
        evaluator.metrics_dir = tmp_path
        evaluator.cmds_to_run = {"a": ["cmd"]}

        called = {"run_all": 0}

        class _Pool:
            def __init__(
                self, cmds_to_run, max_parallel, autometrics_log_file, cwd, env=None
            ):
                assert cmds_to_run == {"a": ["cmd"]}
                assert max_parallel == 3
                assert cwd is None

            def run_all(self):
                called["run_all"] += 1

        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.mera.SubprocessPoolRunner", _Pool
        )
        evaluator.run()
        assert called["run_all"] == 1

    def test_run_passes_clean_env_to_pool(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        evaluator = MeraEvaluator()
        evaluator.autometric_runs_config = MeraEvaluatorConfig()
        evaluator.metrics_dir = tmp_path
        evaluator.cmds_to_run = {"a": ["cmd"]}
        captured_env = {}

        class _Pool:
            def __init__(
                self, cmds_to_run, max_parallel, autometrics_log_file, cwd, env=None
            ):
                captured_env["env"] = env

            def run_all(self):
                pass

        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.mera.SubprocessPoolRunner", _Pool
        )
        monkeypatch.setenv(
            "PYTHONPATH",
            "/some/path:/workspace/pretrain-auto-metrics/contrib/lm_evaluation_harness:/other/path",
        )
        evaluator.run()

        assert captured_env["env"] is not None
        assert "contrib/lm_evaluation_harness" not in captured_env["env"]["PYTHONPATH"]
        assert "/some/path" in captured_env["env"]["PYTHONPATH"]
        assert "/other/path" in captured_env["env"]["PYTHONPATH"]

    def test_build_mera_env_removes_lm_eval_harness_from_pythonpath(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv(
            "PYTHONPATH",
            "/a:/workspace/pretrain-auto-metrics/contrib/lm_evaluation_harness:/b",
        )
        env = MeraEvaluator._build_mera_env()
        paths = env["PYTHONPATH"].split(":")
        assert all("contrib/lm_evaluation_harness" not in p for p in paths)
        assert "/a" in paths
        assert "/b" in paths

    def test_build_mera_env_handles_empty_pythonpath(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("PYTHONPATH", raising=False)
        env = MeraEvaluator._build_mera_env()
        assert env["PYTHONPATH"] == ""

    def test_build_mera_env_handles_only_lm_eval_harness_in_pythonpath(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv(
            "PYTHONPATH",
            "/workspace/pretrain-auto-metrics/contrib/lm_evaluation_harness",
        )
        env = MeraEvaluator._build_mera_env()
        assert env["PYTHONPATH"] == ""

    def test_build_mera_env_preserves_other_env_vars(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("PYTHONPATH", "/a")
        monkeypatch.setenv("MY_VAR", "my_value")
        env = MeraEvaluator._build_mera_env()
        assert env["MY_VAR"] == "my_value"

    def test_drop_zero_metrics_drops_tasks_that_did_not_run(self):
        metrics = {"zero": 0.0, "value": 0.5}

        assert MeraEvaluator._drop_zero_metrics(metrics) == {"value": 0.5}

    def test_send_metrics_to_jd_skips_without_checkpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        warnings = []
        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.mera.Logger.warning", warnings.append
        )
        monkeypatch.setattr(
            JD,
            "send_metrics",
            lambda body, _artifacts: pytest.fail("must not send without a checkpoint"),
        )
        evaluator = MeraEvaluator()
        evaluator.hf_checkpoint_path = None
        evaluator.model_info = None

        evaluator._send_metrics_to_jd({"x": 1.0})

        assert warnings == ["hf_checkpoint_path is not set, skipping JD metrics send"]

    def test_send_metrics_to_jd_drops_zero_scored_tasks(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        sent: dict[str, object] = {}
        evaluator = MeraEvaluator()
        evaluator.hf_checkpoint_path = (
            tmp_path / "model-7b" / "ep0_ba10" / "artifacts" / "hf_checkpoint"
        )
        evaluator.model_info = ModelInfo(model_class="sft", size="7b", username="u")
        evaluator.metrics_dir = tmp_path / "metrics"
        evaluator.metrics_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(JD, "get_shooter_version", lambda _: "1.0")
        monkeypatch.setattr(
            JD, "save_metrics_json", lambda body, step, metrics_dir: None
        )
        monkeypatch.setattr(
            JD,
            "send_metrics",
            lambda body, _artifacts: sent.update(body=body),
        )

        evaluator._send_metrics_to_jd({"acc": 0.9, "not_run": 0.0})

        assert sent["body"]["metrics"] == {10: {"acc": 0.9}}
        assert sent["body"]["shooter_name"] == "mera"

    def test_generate_submissions_exception_and_filtered_none_branch(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        import subprocess

        evaluator = MeraEvaluator()
        evaluator.metrics_dir = tmp_path / "metrics3"
        evaluator.metrics_dir.mkdir(parents=True, exist_ok=True)

        # First category returns None -> exercises finally branch where filtered_dir is falsy.
        # Next categories return dirs.
        dirs = [
            None,
            str(tmp_path / "industrial_filtered"),
            str(tmp_path / "code_filtered"),
        ]
        monkeypatch.setattr(
            evaluator, "_create_filtered_outputs_dir", lambda _cat: dirs.pop(0)
        )
        monkeypatch.setattr(evaluator, "_pack_all_submissions", lambda _d: None)

        class _Res:
            def __init__(self):
                self.returncode = 0
                self.stderr = ""

        calls = {"n": 0}

        def _run(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("subprocess fail")
            return _Res()

        monkeypatch.setattr(subprocess, "run", _run)
        evaluator._generate_submissions()

    def test_generate_submissions_removes_existing_dst_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        import subprocess

        evaluator = MeraEvaluator()
        evaluator.metrics_dir = tmp_path / "metrics4"
        evaluator.metrics_dir.mkdir(parents=True, exist_ok=True)

        submissions_dir = evaluator.metrics_dir / "submissions"
        for category in ("text", "industrial", "code"):
            (submissions_dir / category).mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(
            evaluator,
            "_create_filtered_outputs_dir",
            lambda _category: None,
        )
        monkeypatch.setattr(evaluator, "_pack_all_submissions", lambda _d: None)

        removed = []
        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.mera.shutil.rmtree",
            lambda p, **kwargs: removed.append(Path(p)),
        )

        class _Res:
            returncode = 0
            stderr = ""

        monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: _Res())
        evaluator._generate_submissions()
        assert any(p.name == "text" for p in removed)
        assert any(p.name == "industrial" for p in removed)
        assert any(p.name == "code" for p in removed)

    def test_generate_submissions_removes_stale_root_archives(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        import subprocess

        evaluator = MeraEvaluator()
        evaluator.metrics_dir = tmp_path / "metrics5"
        evaluator.metrics_dir.mkdir(parents=True, exist_ok=True)

        submissions_dir = evaluator.metrics_dir / "submissions"
        submissions_dir.mkdir(parents=True, exist_ok=True)
        (submissions_dir / "text.zip").write_text("stale", encoding="utf-8")
        (submissions_dir / "industrial.zip").write_text("stale", encoding="utf-8")
        (submissions_dir / "code.zip").write_text("stale", encoding="utf-8")
        (submissions_dir / "mera_submissions_all.zip").write_text(
            "stale", encoding="utf-8"
        )

        monkeypatch.setattr(
            evaluator,
            "_create_filtered_outputs_dir",
            lambda _category: None,
        )
        monkeypatch.setattr(evaluator, "_pack_all_submissions", lambda _d: None)

        class _Res:
            returncode = 0
            stderr = ""

        monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: _Res())
        evaluator._generate_submissions()

        assert not (submissions_dir / "text.zip").exists()
        assert not (submissions_dir / "industrial.zip").exists()
        assert not (submissions_dir / "code.zip").exists()
        assert not (submissions_dir / "mera_submissions_all.zip").exists()
