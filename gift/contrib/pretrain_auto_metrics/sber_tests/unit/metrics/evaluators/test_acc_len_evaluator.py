import json
from pathlib import Path
from typing import Any

import pytest
import requests
from node_runner.src.config_model import (
    AccLenDatasetConfig,
    AccLenEvaluatorConfig,
)
from node_runner.src.config_model import ModelInfoConfig as ModelInfo
from node_runner.src.job_dispatcher import JD
from node_runner.src.metrics.evaluators import acc_len as acc_len_module
from node_runner.src.metrics.evaluators.acc_len import (
    ACC_LEN_VERSION_FILE,
    REQUEST_TIMEOUT_S,
    AccLenEvaluator,
)

pytestmark = pytest.mark.unit

WORKER = "http://10.0.0.1:30001"


def _dataset(name: str = "d1", **overrides) -> dict[str, Any]:
    dataset: dict[str, Any] = {
        "name": name,
        "path": "/data/d1.jsonl",
        "max_new_tokens": 64,
    }
    dataset.update(overrides)
    return dataset


def _config(**overrides) -> AccLenEvaluatorConfig:
    kwargs: dict[str, Any] = {"datasets": [_dataset()]}
    kwargs.update(overrides)
    cfg = AccLenEvaluatorConfig(**kwargs)
    cfg.base_url = "http://127.0.0.1:29999"
    cfg.worker_urls = [WORKER]
    cfg.max_concurrency = 4
    return cfg


def _evaluator(tmp_path: Path, cfg: AccLenEvaluatorConfig | None = None):
    evaluator = AccLenEvaluator()
    evaluator.setup(
        autometric_runs_config=cfg or _config(),
        artifacts_path=tmp_path / "artifacts",
        hf_checkpoint_path=tmp_path / "model" / "ep1_ba2" / "hf_checkpoint",
        model_info=ModelInfo(model_class="sft", size="lite", username="u"),
    )
    return evaluator


class _FakeResponse:
    def __init__(self, status_code: int = 200, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code != 200:
            raise requests.RequestException(f"status {self.status_code}")


@pytest.fixture
def fake_bench(monkeypatch: pytest.MonkeyPatch):
    """Patch SubprocessManager in acc_len and record how it was constructed."""

    def _patch(*, successful: bool = True, writes: str | None = None) -> dict:
        called: dict = {}

        class FakeSP:
            def __init__(self, cmd, log, cwd, run_name, env=None):
                called.update(cmd=cmd, log=log, cwd=cwd, run_name=run_name, env=env)

            def start(self):
                called["started"] = True
                if writes is not None:
                    output = Path(
                        called["cmd"][called["cmd"].index("--output-file") + 1]
                    )
                    output.write_text(writes)

            def wait_for_completion(self):
                called["waited"] = True

            def log_error(self):
                called["logged"] = True

            def is_successful(self):
                return successful

        monkeypatch.setattr(acc_len_module, "SubprocessManager", FakeSP)
        return called

    return _patch


@pytest.fixture
def fake_worker(monkeypatch: pytest.MonkeyPatch):
    """Patch the worker HTTP calls acc_len makes, recording requests."""

    def _patch(
        *,
        internal_states: list[dict] | None = None,
        flush: _FakeResponse | Exception | None = None,
        info: Exception | None = None,
    ) -> dict:
        calls: dict = {"flush": [], "info": []}

        def _post(url, params=None, timeout=None):
            calls["flush"].append({"url": url, "params": params, "timeout": timeout})
            if isinstance(flush, Exception):
                raise flush
            return flush or _FakeResponse()

        def _get(url, timeout=None):
            calls["info"].append({"url": url, "timeout": timeout})
            if info is not None:
                raise info
            return _FakeResponse(
                payload={"internal_states": internal_states or []},
            )

        monkeypatch.setattr(acc_len_module.requests, "post", _post)
        monkeypatch.setattr(acc_len_module.requests, "get", _get)
        return calls

    return _patch


class TestHistogramMath:
    @pytest.mark.parametrize(
        "histograms, min_length, expected",
        [
            ([[1, 2], [3, 4]], 0, [4, 6]),
            ([[1], [2, 3, 4]], 0, [3, 3, 4]),
            ([[1, 2]], 4, [1, 2, 0, 0]),
            ([], 3, [0, 0, 0]),
        ],
    )
    def test_merge_sums_ranks_and_pads_to_draft_width(
        self, histograms, min_length, expected
    ):
        assert (
            AccLenEvaluator._merge_correct_drafts_histograms(histograms, min_length)
            == expected
        )

    def test_accept_length_adds_survival_share_per_depth(self):
        # 10 steps accepted 0 drafts, 6 accepted 1, 4 accepted 2.
        histogram = [10, 6, 4]

        assert AccLenEvaluator._accept_length_by_num_steps(histogram) == [
            pytest.approx(1 + 10 / 20),
            pytest.approx(1 + 10 / 20 + 4 / 20),
        ]

    def test_accept_length_is_empty_without_verify_steps(self):
        assert AccLenEvaluator._accept_length_by_num_steps([0, 0, 0]) == []

    def test_accept_rate_is_conditional_on_the_previous_position(self):
        histogram = [10, 6, 4]

        assert AccLenEvaluator._accept_rate_by_position(histogram) == [
            pytest.approx(10 / 20),
            pytest.approx(4 / 10),
        ]

    def test_accept_rate_stops_at_a_position_no_step_reached(self):
        # Nothing ever accepted a second draft, so position 2 has no denominator.
        assert AccLenEvaluator._accept_rate_by_position([5, 0, 0]) == [
            pytest.approx(0.0)
        ]

    def test_accept_rate_is_empty_without_verify_steps(self):
        assert AccLenEvaluator._accept_rate_by_position([0, 0]) == []


class TestAccLenSetup:
    def test_setup_creates_metrics_dir_and_keeps_worker_urls(self, tmp_path: Path):
        evaluator = _evaluator(tmp_path)

        assert evaluator.metrics_dir == tmp_path / "artifacts" / "metrics" / "acc_len"
        assert evaluator.metrics_dir.exists()
        assert evaluator.results_json_path == evaluator.metrics_dir / "results.json"
        assert evaluator.worker_urls == [WORKER]

    def test_build_command_runs_bench_serving_in_the_configured_venv(
        self, tmp_path: Path
    ):
        evaluator = _evaluator(tmp_path)

        cmd = evaluator._build_command(
            evaluator.autometric_runs_config.datasets[0], tmp_path / "out.jsonl"
        )

        assert cmd[:3] == [
            evaluator.autometric_runs_config.venv_python,
            "-m",
            "sglang.bench_serving",
        ]
        assert cmd[cmd.index("--output-file") + 1] == str(tmp_path / "out.jsonl")

    def test_build_env_carries_the_request_timeout(self, tmp_path: Path):
        evaluator = _evaluator(tmp_path, _config(request_timeout=900))

        env = evaluator._build_env()

        assert env["SGLANG_BENCH_REQUEST_TIMEOUT"] == "900"
        assert "PATH" in env


class TestAccLenRun:
    def test_run_without_worker_urls_does_not_bench(
        self, tmp_path: Path, fake_bench, monkeypatch: pytest.MonkeyPatch
    ):
        cfg = _config()
        cfg.worker_urls = []
        evaluator = _evaluator(tmp_path, cfg)
        called = fake_bench()

        evaluator.run()

        assert called == {}
        assert not evaluator.results_json_path.exists()

    def test_run_skips_a_dataset_already_in_results(self, tmp_path: Path, fake_bench):
        evaluator = _evaluator(tmp_path)
        evaluator.results_json_path.write_text(json.dumps({"d1": {"accept_length": 2}}))
        called = fake_bench()

        evaluator.run()

        assert called == {}

    def test_run_records_each_dataset_as_it_completes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        cfg = _config(datasets=[_dataset("d1"), _dataset("d2")])
        evaluator = _evaluator(tmp_path, cfg)
        monkeypatch.setattr(
            evaluator,
            "_run_dataset",
            lambda dataset: {"accept_length": 2.0} if dataset.name == "d1" else None,
        )

        evaluator.run()

        assert json.loads(evaluator.results_json_path.read_text()) == {
            "d1": {"accept_length": 2.0}
        }

    def test_run_writes_nothing_when_every_dataset_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        evaluator = _evaluator(tmp_path)
        monkeypatch.setattr(evaluator, "_run_dataset", lambda dataset: None)

        evaluator.run()

        assert not evaluator.results_json_path.exists()


class TestAccLenRunDataset:
    def test_failed_bench_yields_no_result(
        self, tmp_path: Path, fake_bench, fake_worker
    ):
        evaluator = _evaluator(tmp_path)
        fake_worker()
        fake_bench(successful=False)

        assert (
            evaluator._run_dataset(evaluator.autometric_runs_config.datasets[0]) is None
        )

    def test_bench_that_completed_no_requests_yields_no_result(
        self, tmp_path: Path, fake_bench, fake_worker
    ):
        evaluator = _evaluator(tmp_path)
        fake_worker()
        fake_bench(writes=json.dumps({"completed": 0}) + "\n")

        assert (
            evaluator._run_dataset(evaluator.autometric_runs_config.datasets[0]) is None
        )

    def test_stale_output_file_is_removed_before_the_run(
        self, tmp_path: Path, fake_bench, fake_worker
    ):
        evaluator = _evaluator(tmp_path)
        stale = evaluator.metrics_dir / "d1.jsonl"
        stale.write_text(json.dumps({"completed": 999}) + "\n")
        fake_worker()
        fake_bench(writes=json.dumps({"completed": 0}) + "\n")

        evaluator._run_dataset(evaluator.autometric_runs_config.datasets[0])

        assert json.loads(stale.read_text())["completed"] == 0

    def test_short_run_still_reports_the_metrics_it_measured(
        self, tmp_path: Path, fake_bench, fake_worker
    ):
        evaluator = _evaluator(tmp_path, _config(num_prompts=512))
        fake_worker(internal_states=[{"avg_spec_accept_length": 2.5}])
        fake_bench(
            writes=json.dumps({"completed": 3, "output_throughput": 100.0}) + "\n"
        )

        result = evaluator._run_dataset(evaluator.autometric_runs_config.datasets[0])

        assert result == {
            "accept_length": pytest.approx(2.5),
            "output_throughput": pytest.approx(100.0),
        }

    def test_a_full_run_reports_the_metrics_without_a_short_count(
        self, tmp_path: Path, fake_bench, fake_worker
    ):
        evaluator = _evaluator(tmp_path, _config(num_prompts=5))
        fake_worker(internal_states=[{"avg_spec_accept_length": 2.5}])
        fake_bench(writes=json.dumps({"completed": 5}) + "\n")

        result = evaluator._run_dataset(evaluator.autometric_runs_config.datasets[0])

        assert result == {"accept_length": pytest.approx(2.5)}

    def test_dataset_without_acceptance_length_yields_no_result(
        self, tmp_path: Path, fake_bench, fake_worker
    ):
        evaluator = _evaluator(tmp_path)
        fake_worker(internal_states=[{}])
        fake_bench(writes=json.dumps({"completed": 5}) + "\n")

        assert (
            evaluator._run_dataset(evaluator.autometric_runs_config.datasets[0]) is None
        )

    def test_bench_subprocess_gets_the_env_and_run_name(
        self, tmp_path: Path, fake_bench, fake_worker
    ):
        evaluator = _evaluator(tmp_path, _config(request_timeout=1234))
        fake_worker(internal_states=[{"avg_spec_accept_length": 2.0}])
        called = fake_bench(writes=json.dumps({"completed": 5}) + "\n")

        evaluator._run_dataset(evaluator.autometric_runs_config.datasets[0])

        assert called["run_name"] == "acc_len_d1"
        assert called["env"]["SGLANG_BENCH_REQUEST_TIMEOUT"] == "1234"


class TestAccLenWorkerCalls:
    def test_flush_defers_until_requests_drain(self, tmp_path: Path, fake_worker):
        evaluator = _evaluator(tmp_path)
        calls = fake_worker()

        evaluator._flush_worker_counters()

        assert calls["flush"] == [
            {
                "url": f"{WORKER}/flush_cache",
                "params": {"timeout": acc_len_module.FLUSH_TIMEOUT_S},
                "timeout": acc_len_module.FLUSH_TIMEOUT_S + REQUEST_TIMEOUT_S,
            }
        ]

    @pytest.mark.parametrize(
        "flush",
        [_FakeResponse(status_code=400), requests.RequestException("boom")],
    )
    def test_a_worker_that_refuses_the_flush_does_not_stop_the_run(
        self, tmp_path: Path, fake_worker, flush
    ):
        evaluator = _evaluator(tmp_path)
        fake_worker(flush=flush)

        evaluator._flush_worker_counters()

    def test_internal_states_are_pooled_across_workers(
        self, tmp_path: Path, fake_worker
    ):
        cfg = _config()
        cfg.worker_urls = [WORKER, "http://10.0.0.2:30001"]
        evaluator = _evaluator(tmp_path, cfg)
        fake_worker(internal_states=[{"avg_spec_accept_length": 2.0}])

        states = evaluator._read_worker_internal_states()

        assert states == [
            {"avg_spec_accept_length": 2.0},
            {"avg_spec_accept_length": 2.0},
        ]

    def test_a_worker_that_stops_answering_is_skipped(
        self, tmp_path: Path, fake_worker
    ):
        evaluator = _evaluator(tmp_path)
        fake_worker(info=requests.RequestException("gone"))

        assert evaluator._read_worker_internal_states() == []


class TestAccLenBenchResult:
    def test_missing_output_file_reads_as_empty(self, tmp_path: Path):
        assert AccLenEvaluator._read_bench_result(tmp_path / "absent.jsonl") == {}

    def test_output_file_without_a_summary_reads_as_empty(self, tmp_path: Path):
        path = tmp_path / "empty.jsonl"
        path.write_text("\n  \n")

        assert AccLenEvaluator._read_bench_result(path) == {}

    def test_the_last_summary_line_wins(self, tmp_path: Path):
        path = tmp_path / "bench.jsonl"
        path.write_text(
            json.dumps({"completed": 1}) + "\n" + json.dumps({"completed": 7}) + "\n"
        )

        assert AccLenEvaluator._read_bench_result(path) == {"completed": 7}

    def test_serving_metrics_are_renamed_from_the_bench_summary(self):
        metrics = AccLenEvaluator._serving_metrics(
            {
                "output_throughput": 120.5,
                "max_output_tokens_per_s": 300.0,
                "median_itl_ms": 8.25,
            }
        )

        assert metrics == {
            "output_throughput": pytest.approx(120.5),
            "peak_output_throughput": pytest.approx(300.0),
            "median_itl_ms": pytest.approx(8.25),
        }

    def test_a_bench_that_reported_no_serving_metrics_publishes_none(self):
        assert AccLenEvaluator._serving_metrics({"max_output_tokens_per_s": None}) == {}


class TestAccLenDraftPositionMetrics:
    def test_accept_length_is_averaged_across_ranks(self):
        metrics = AccLenEvaluator._draft_position_metrics(
            [
                {"avg_spec_accept_length": 2.0},
                {"avg_spec_accept_length": 3.0},
            ],
            "d1",
        )

        assert metrics == {"accept_length": pytest.approx(2.5)}

    def test_no_reported_acceptance_length_yields_nothing(self):
        assert AccLenEvaluator._draft_position_metrics([{}], "d1") == {}

    def test_a_tree_draft_leaves_out_the_per_depth_breakdown(self):
        metrics = AccLenEvaluator._draft_position_metrics(
            [
                {
                    "avg_spec_accept_length": 2.0,
                    "spec_correct_drafts_histogram": [10, 6, 4],
                    "speculative_eagle_topk": 4,
                    "speculative_num_steps": 2,
                }
            ],
            "d1",
        )

        assert metrics == {"accept_length": pytest.approx(2.0)}

    def test_a_chain_draft_reports_accept_length_and_rate_per_position(self):
        metrics = AccLenEvaluator._draft_position_metrics(
            [
                {
                    "avg_spec_accept_length": 2.0,
                    "spec_correct_drafts_histogram": [10, 6, 4],
                    "speculative_eagle_topk": None,
                    "speculative_num_steps": 2,
                }
            ],
            "d1",
        )

        assert metrics == {
            "accept_length": pytest.approx(2.0),
            "accept_length_num_steps_1": pytest.approx(1.5),
            "accept_length_num_steps_2": pytest.approx(1.7),
            "accept_rate_position_1": pytest.approx(0.5),
            "accept_rate_position_2": pytest.approx(0.4),
        }

    def test_the_histogram_is_padded_to_the_configured_draft_width(self):
        metrics = AccLenEvaluator._draft_position_metrics(
            [
                {
                    "avg_spec_accept_length": 1.5,
                    "spec_correct_drafts_histogram": [10, 10],
                    "speculative_eagle_topk": 1,
                    "speculative_num_steps": 3,
                }
            ],
            "d1",
        )

        # num_steps 3 means depths 1..3 exist, even though nothing reached 2 or 3.
        assert "accept_length_num_steps_3" in metrics
        assert metrics["accept_rate_position_2"] == pytest.approx(0.0)

    def test_a_server_without_the_histogram_reports_accept_length_only(self):
        metrics = AccLenEvaluator._draft_position_metrics(
            [{"avg_spec_accept_length": 2.0, "spec_correct_drafts_histogram": []}],
            "d1",
        )

        assert metrics == {"accept_length": pytest.approx(2.0)}


class TestAccLenGetMetrics:
    def test_no_results_file_yields_no_metrics(self, tmp_path: Path):
        evaluator = _evaluator(tmp_path)

        assert evaluator.get_metrics() == {}

    def test_metrics_are_prefixed_per_dataset(self, tmp_path: Path):
        evaluator = _evaluator(tmp_path)
        evaluator.results_json_path.write_text(
            json.dumps({"d1": {"accept_length": 2.5, "median_itl_ms": 8.0}})
        )

        assert evaluator.get_metrics() == {
            "acc_len/d1/accept_length": pytest.approx(2.5),
            "acc_len/d1/median_itl_ms": pytest.approx(8.0),
        }

    def test_a_dataset_dropped_from_the_config_stops_being_republished(
        self, tmp_path: Path
    ):
        evaluator = _evaluator(tmp_path)
        evaluator.results_json_path.write_text(
            json.dumps(
                {"d1": {"accept_length": 2.5}, "renamed": {"accept_length": 9.0}}
            )
        )

        assert evaluator.get_metrics() == {
            "acc_len/d1/accept_length": pytest.approx(2.5)
        }

    def test_metrics_without_a_metrics_dir_yield_nothing(self, tmp_path: Path):
        evaluator = _evaluator(tmp_path)
        evaluator.metrics_dir = None

        assert evaluator.get_metrics() == {}


class TestAccLenJd:
    def test_after_run_tars_and_symlinks_the_metrics_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        evaluator = _evaluator(tmp_path)
        called: dict = {}
        monkeypatch.setattr(
            evaluator,
            "_send_metrics_to_jd",
            lambda metrics: called.update(sent=metrics),
        )
        monkeypatch.setattr(
            JD,
            "tar_metrics_dir",
            lambda metrics_dir, artifacts_path, _name: tmp_path / "acc_len.tar.zst",
        )
        monkeypatch.setattr(
            JD,
            "symlink_metrics_for_transfer",
            lambda tar_path, hf_path, _name: called.update(symlinked=tar_path),
        )

        evaluator.after_run({"acc_len/d1/accept_length": 2.5})

        assert called["sent"] == {"acc_len/d1/accept_length": 2.5}
        assert called["symlinked"] == tmp_path / "acc_len.tar.zst"

    def test_metrics_are_sent_under_the_acc_len_shooter(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        evaluator = _evaluator(tmp_path)
        sent: dict = {}
        monkeypatch.setattr(JD, "can_send_metrics", lambda **kwargs: True)
        monkeypatch.setattr(JD, "get_shooter_version", lambda path: f"v-{path}")
        monkeypatch.setattr(
            JD, "send_evaluator_metrics", lambda **kwargs: sent.update(kwargs)
        )

        evaluator._send_metrics_to_jd({"acc_len/d1/accept_length": 2.5})

        assert sent["shooter_name"] == "acc_len"
        assert sent["shooter_version"] == f"v-{ACC_LEN_VERSION_FILE}"
        assert sent["jd_metrics"] == {"ACC_LEN_D1_ACCEPT_LENGTH": 2.5}

    def test_nothing_is_sent_when_the_checkpoint_is_unknown(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        evaluator = _evaluator(tmp_path)
        monkeypatch.setattr(JD, "can_send_metrics", lambda **kwargs: False)
        monkeypatch.setattr(
            JD,
            "send_evaluator_metrics",
            lambda **kwargs: pytest.fail("must not send without a checkpoint"),
        )

        evaluator._send_metrics_to_jd({"acc_len/d1/accept_length": 2.5})


class TestAccLenDatasetConfig:
    def test_sharegpt_rejects_an_explicit_chat_template_choice(self):
        with pytest.raises(ValueError, match="apply_chat_template has no effect"):
            AccLenDatasetConfig(
                name="d1",
                path="/data/d1.jsonl",
                format="sharegpt",
                apply_chat_template=False,
            )

    def test_custom_requires_an_output_length(self):
        with pytest.raises(ValueError, match="custom format requires max_new_tokens"):
            AccLenDatasetConfig(name="d1", path="/data/d1.jsonl")

    def test_sharegpt_rejects_an_output_length(self):
        with pytest.raises(ValueError, match="max_new_tokens has no effect"):
            AccLenDatasetConfig(
                name="d1",
                path="/data/d1.jsonl",
                format="sharegpt",
                max_new_tokens=64,
            )

    def test_sharegpt_keeps_its_own_prompts_and_lengths(self):
        dataset = AccLenDatasetConfig(
            name="d1", path="/data/d1.jsonl", format="sharegpt"
        )

        assert dataset.dataset_format == "sharegpt"
        assert dataset.max_new_tokens is None


class TestAccLenEvaluatorConfig:
    def test_request_seed_defaults_to_the_prompt_seed(self):
        cfg = AccLenEvaluatorConfig(datasets=[_dataset()], seed=7)

        assert cfg.request_seed == 7

    def test_an_explicit_request_seed_is_kept(self):
        cfg = AccLenEvaluatorConfig(datasets=[_dataset()], seed=7, request_seed=9)

        assert cfg.request_seed == 9

    def test_duplicate_dataset_names_are_rejected(self):
        with pytest.raises(ValueError, match="dataset names must be unique"):
            AccLenEvaluatorConfig(datasets=[_dataset("d1"), _dataset("d1")])

    def test_request_timeout_defaults_to_the_shared_constant(self):
        from node_runner.src.config_model import DEFAULT_REQUEST_TIMEOUT

        cfg = AccLenEvaluatorConfig(datasets=[_dataset()])

        assert cfg.request_timeout == DEFAULT_REQUEST_TIMEOUT

    def test_base_url_must_be_applied_before_the_run(self):
        cfg = AccLenEvaluatorConfig(datasets=[_dataset()])

        with pytest.raises(ValueError, match="base_url must be set"):
            cfg.build_cli_args(dataset=cfg.datasets[0], output_path=Path("/tmp/o"))

    def test_apply_defaults_scales_concurrency_off_inference_instances(
        self, tmp_path: Path
    ):
        cfg = AccLenEvaluatorConfig(datasets=[_dataset()])

        cfg.apply_defaults(
            inference_instances=4,
            direct_base_url="http://127.0.0.1:29999",
            worker_urls=[WORKER],
            hf_checkpoint_path=None,
            max_seq_len=8192,
        )

        assert cfg.max_concurrency == 32
        assert cfg.base_url == "http://127.0.0.1:29999"
        assert cfg.worker_urls == [WORKER]
        assert cfg.context_len == 8192

    def test_apply_defaults_keeps_an_explicit_concurrency(self):
        cfg = AccLenEvaluatorConfig(datasets=[_dataset()], max_concurrency=3)

        cfg.apply_defaults(
            inference_instances=4,
            direct_base_url="http://127.0.0.1:29999",
            worker_urls=[],
            hf_checkpoint_path=None,
        )

        assert cfg.max_concurrency == 3

    @pytest.mark.parametrize(
        "files, expected",
        [
            ({"chat_template.jinja": "{{}}"}, True),
            ({"chat_template.json": "{}"}, True),
            ({"tokenizer_config.json": json.dumps({"chat_template": "{{}}"})}, True),
            ({"tokenizer_config.json": json.dumps({})}, False),
            ({"tokenizer_config.json": "not json"}, False),
            ({}, False),
        ],
    )
    def test_chat_template_is_applied_only_when_the_checkpoint_has_one(
        self, tmp_path: Path, files, expected
    ):
        checkpoint = tmp_path / "hf"
        checkpoint.mkdir()
        for name, content in files.items():
            (checkpoint / name).write_text(content)
        cfg = AccLenEvaluatorConfig(datasets=[_dataset()])

        cfg.apply_defaults(
            inference_instances=1,
            direct_base_url="http://127.0.0.1:29999",
            worker_urls=[],
            hf_checkpoint_path=checkpoint,
        )

        assert cfg.apply_chat_template is expected

    def test_a_pretrain_checkpoint_without_a_path_gets_no_template(self):
        cfg = AccLenEvaluatorConfig(datasets=[_dataset()])

        cfg.apply_defaults(
            inference_instances=1,
            direct_base_url="http://127.0.0.1:29999",
            worker_urls=[],
            hf_checkpoint_path=None,
        )

        assert cfg.apply_chat_template is False


class TestAccLenCliArgs:
    def test_custom_dataset_sends_its_output_length_and_template(self):
        cfg = _config()

        args = cfg.build_cli_args(dataset=cfg.datasets[0], output_path=Path("/tmp/o"))

        assert args[args.index("--dataset-name") + 1] == "custom"
        assert args[args.index("--sharegpt-output-len") + 1] == "64"
        assert args[args.index("--max-concurrency") + 1] == "4"
        assert "--apply-chat-template" in args
        assert "--flush-cache" in args
        assert "--chat-template-kwargs" not in args

    def test_reasoning_is_passed_to_the_chat_template_when_asked_for(self):
        cfg = _config(reasoning=True)

        args = cfg.build_cli_args(dataset=cfg.datasets[0], output_path=Path("/tmp/o"))

        assert json.loads(args[args.index("--chat-template-kwargs") + 1]) == {
            "reasoning": True
        }

    def test_a_dataset_can_opt_out_of_the_chat_template(self):
        cfg = _config(datasets=[_dataset(apply_chat_template=False)])

        args = cfg.build_cli_args(dataset=cfg.datasets[0], output_path=Path("/tmp/o"))

        assert "--apply-chat-template" not in args

    def test_sharegpt_sends_no_output_length(self):
        cfg = _config(datasets=[_dataset("sg", format="sharegpt", max_new_tokens=None)])

        args = cfg.build_cli_args(dataset=cfg.datasets[0], output_path=Path("/tmp/o"))

        assert "--sharegpt-output-len" not in args
        assert "--apply-chat-template" not in args

    def test_context_len_bounds_the_prompts_bench_serving_samples(self):
        cfg = _config()
        cfg.context_len = 4096

        args = cfg.build_cli_args(dataset=cfg.datasets[0], output_path=Path("/tmp/o"))

        assert args[args.index("--sharegpt-context-len") + 1] == "4096"

    def test_sampling_params_fall_back_to_the_evaluator_values(self):
        cfg = _config(temperature=0.7, top_p=0.9, repetition_penalty=1.1, seed=11)

        args = cfg.build_cli_args(dataset=cfg.datasets[0], output_path=Path("/tmp/o"))

        assert json.loads(args[args.index("--extra-request-body") + 1]) == {
            "temperature": 0.7,
            "top_p": 0.9,
            "seed": 11,
            "repetition_penalty": 1.1,
        }

    def test_a_dataset_overrides_the_evaluator_sampling(self):
        cfg = _config(
            temperature=0.7,
            datasets=[_dataset(temperature=0.0, top_p=0.5, repetition_penalty=1.5)],
        )

        args = cfg.build_cli_args(dataset=cfg.datasets[0], output_path=Path("/tmp/o"))

        body = json.loads(args[args.index("--extra-request-body") + 1])
        assert body["temperature"] == 0.0
        assert body["top_p"] == 0.5
        assert body["repetition_penalty"] == 1.5

    def test_no_penalty_is_sent_when_none_is_configured(self):
        cfg = _config()

        args = cfg.build_cli_args(dataset=cfg.datasets[0], output_path=Path("/tmp/o"))

        assert "repetition_penalty" not in json.loads(
            args[args.index("--extra-request-body") + 1]
        )
