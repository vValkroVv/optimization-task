import json
import math
from pathlib import Path

import numpy as np
import pytest
import zstandard

from node_runner.src.config_model import (
    EvaluatorsConfig,
    TOKEN_NLL_DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE,
    TokenNllEvaluatorConfig,
)
from node_runner.src.config_model import ModelInfoConfig as ModelInfo
from node_runner.src.job_dispatcher import JD
from node_runner.src.metrics import metrics_manager
from node_runner.src.metrics.evaluators import token_nll
from node_runner.src.metrics.evaluators.token_nll import TokenNllEvaluator

pytestmark = pytest.mark.unit

BOS = 1
# fake tokenizer: "a" -> 11, "b" -> 12, "c" -> 13
LOGPROBS_EARLY = {11: -0.5, 12: -1.0, 13: -0.2}
LOGPROBS_LATE = {11: -0.005, 12: -0.8, 13: -0.2}
# docs "abc" and "ab": scored tokens a,b,c,a,b under LOGPROBS_EARLY
EXPECTED_EARLY_VAL_LOSS = (0.5 + 1.0 + 0.2 + 0.5 + 1.0) / 5


class _FakeTokenizer:
    bos_token_id = BOS

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        return {"input_ids": [ord(char) - ord("a") + 11 for char in text]}

    def decode(self, ids) -> str:
        return "".join(chr(int(i) - 11 + ord("a")) for i in ids)


class _FakeTokenizerWithoutBos(_FakeTokenizer):
    bos_token_id = None


class _FakeServer:
    def __init__(self, logprobs: dict[int, float]):
        self.logprobs = logprobs
        self.requests: list[dict] = []

    def post(self, url: str, json: dict, timeout: int):
        self.requests.append(json)
        ids = json["input_ids"]
        input_token_logprobs = [[None, ids[0]]] + [
            [self.logprobs[token], token] for token in ids[1:]
        ]
        return _FakeResponse(
            {"meta_info": {"input_token_logprobs": input_token_logprobs}}
        )


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


def _write_dataset(path: Path, texts: list[str]) -> None:
    _write_lines(path, [json.dumps({"text": text}) for text in texts])


def _write_lines(path: Path, lines: list[str]) -> None:
    payload = "".join(line + "\n" for line in lines)
    path.write_bytes(zstandard.ZstdCompressor().compress(payload.encode("utf-8")))


def _config(dataset_path: Path, **overrides) -> TokenNllEvaluatorConfig:
    payload = {
        "datasets": {"ocr": {"path": str(dataset_path)}},
        "ref_lag_batches": 20000,
        "window_tokens": 8,
        **overrides,
    }
    cfg = TokenNllEvaluatorConfig.model_validate(payload)
    cfg.apply_defaults(inference_instances=1, base_url="http://127.0.0.1:30000")
    return cfg


def _run_evaluator(
    tmp_path: Path,
    monkeypatch,
    cfg: TokenNllEvaluatorConfig,
    step: int,
    server: _FakeServer,
    hf_name: str = "hf_checkpoint",
    tokenizer: _FakeTokenizer | None = None,
) -> TokenNllEvaluator:
    loaded = tokenizer or _FakeTokenizer()
    monkeypatch.setattr(token_nll.AutoTokenizer, "from_pretrained", lambda path: loaded)
    monkeypatch.setattr(token_nll.requests, "post", server.post)
    checkpoint_dir = tmp_path / "run" / f"ep0_ba{step}"
    evaluator = TokenNllEvaluator()
    evaluator.setup(
        autometric_runs_config=cfg,
        artifacts_path=checkpoint_dir / "artifacts",
        hf_checkpoint_path=checkpoint_dir / "artifacts" / hf_name,
        model_info=ModelInfo(model_class="pretrain", size="lite", username="u"),
    )
    evaluator.run()
    return evaluator


@pytest.fixture
def dataset_path(tmp_path: Path) -> Path:
    path = tmp_path / "ocr.jsonl.zstd"
    _write_dataset(path, ["abc", "ab"])
    return path


class TestConfig:
    @pytest.mark.parametrize(
        ("payload", "match"),
        [
            (
                {"datasets": {}, "ref_lag_batches": 1000, "window_tokens": 4090},
                "datasets",
            ),
            (
                {
                    "datasets": {"web": {"path": "/data/web.jsonl.zstd"}},
                    "ref_lag_batches": 1,
                },
                "window_tokens",
            ),
            (
                {
                    "datasets": {"web": {"path": "  "}},
                    "ref_lag_batches": 1,
                    "window_tokens": 4090,
                },
                "path can't be blank",
            ),
        ],
    )
    def test_required_fields_are_validated(self, payload, match):
        with pytest.raises(ValueError, match=match):
            TokenNllEvaluatorConfig.model_validate(payload)

    @pytest.mark.parametrize(
        ("concurrency", "expected"),
        [(None, 4 * TOKEN_NLL_DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE), (3, 3)],
    )
    def test_concurrency_scales_with_instances_unless_set(self, concurrency, expected):
        cfg = TokenNllEvaluatorConfig.model_validate(
            {
                "datasets": {"web": {"path": "/data/web.jsonl.zstd"}},
                "ref_lag_batches": 1,
                "window_tokens": 4090,
                "concurrency": concurrency,
            }
        )
        cfg.apply_defaults(inference_instances=4, base_url="http://127.0.0.1:30000")

        assert cfg.concurrency == expected

    def test_registered_in_evaluators_config_and_base_bucket(self):
        evaluators = EvaluatorsConfig.model_validate(
            {
                "token_nll": {
                    "datasets": {"web": {"path": "/data/web.jsonl.zstd"}},
                    "ref_lag_batches": 1,
                    "window_tokens": 4090,
                }
            }
        )

        assert [name for name, _ in evaluators] == ["token_nll"]
        assert metrics_manager.EVALUATORS["token_nll"] is TokenNllEvaluator


class TestValLoss:
    def test_val_loss_is_mean_nll_over_scored_tokens(
        self, tmp_path: Path, monkeypatch, dataset_path: Path
    ):
        server = _FakeServer(LOGPROBS_EARLY)
        evaluator = _run_evaluator(
            tmp_path, monkeypatch, _config(dataset_path), step=1000, server=server
        )

        metrics = evaluator.get_metrics()
        assert metrics["val_loss/ocr"] == pytest.approx(EXPECTED_EARLY_VAL_LOSS)
        assert metrics["val_ppl/ocr"] == pytest.approx(
            math.exp(EXPECTED_EARLY_VAL_LOSS)
        )
        assert metrics["val_loss/all"] == pytest.approx(EXPECTED_EARLY_VAL_LOSS)
        assert "memorization/ocr/proxy" not in metrics

    def test_windows_overlap_by_one_token_and_keep_every_token_scored(
        self, tmp_path: Path, monkeypatch, dataset_path: Path
    ):
        server = _FakeServer(LOGPROBS_EARLY)
        _run_evaluator(
            tmp_path,
            monkeypatch,
            _config(dataset_path, window_tokens=3),
            step=1000,
            server=server,
        )

        assert [request["input_ids"] for request in server.requests] == [
            [BOS, 11, 12],
            [12, 13],
            [BOS, 11, 12],
        ]
        assert server.requests[0]["return_logprob"] is True
        assert server.requests[0]["logprob_start_len"] == 0
        assert server.requests[0]["sampling_params"]["max_new_tokens"] == 1

    def test_unusable_rows_and_documents_are_skipped(self, tmp_path: Path, monkeypatch):
        path = tmp_path / "mixed.jsonl.zstd"
        _write_lines(
            path,
            [
                "",  # blank line
                json.dumps({"other": "no text field"}),
                json.dumps({"text": "a"}),  # one token without BOS: too short to score
                json.dumps({"text": "abc"}),
                json.dumps({"text": "ab"}),
                json.dumps({"text": "zz"}),  # beyond max_docs, never read
            ],
        )
        server = _FakeServer(LOGPROBS_EARLY)

        evaluator = _run_evaluator(
            tmp_path,
            monkeypatch,
            _config(path, max_docs=3),
            step=1000,
            server=server,
            tokenizer=_FakeTokenizerWithoutBos(),
        )

        # scored tokens are "bc" of "abc" and "b" of "ab"; "a" alone yields a single token
        assert evaluator.get_metrics()["val_loss/ocr"] == pytest.approx(
            (1.0 + 0.2 + 1.0) / 3
        )
        assert [request["input_ids"] for request in server.requests] == [
            [11, 12, 13],
            [11, 12],
        ]

    def test_dataset_without_scorable_documents_publishes_nothing(
        self, tmp_path: Path, monkeypatch
    ):
        path = tmp_path / "tiny.jsonl.zstd"
        _write_dataset(path, ["a", "b"])

        evaluator = _run_evaluator(
            tmp_path,
            monkeypatch,
            _config(path),
            step=1000,
            server=_FakeServer(LOGPROBS_EARLY),
            tokenizer=_FakeTokenizerWithoutBos(),
        )

        assert evaluator.get_metrics() == {}

    def test_truncated_logprobs_publish_no_metrics(
        self, tmp_path: Path, monkeypatch, dataset_path: Path
    ):
        class _TruncatedServer(_FakeServer):
            def post(self, url, json, timeout):
                self.requests.append(json)
                return _FakeResponse(
                    {
                        "meta_info": {
                            "input_token_logprobs": [[None, json["input_ids"][0]]]
                        }
                    }
                )

        evaluator = _run_evaluator(
            tmp_path,
            monkeypatch,
            _config(dataset_path),
            step=1000,
            server=_TruncatedServer({}),
        )

        assert evaluator.get_metrics() == {}

    def test_unwritable_all_metrics_path_does_not_lose_the_metrics(
        self, tmp_path: Path, monkeypatch, dataset_path: Path
    ):
        evaluator = _run_evaluator(
            tmp_path,
            monkeypatch,
            _config(dataset_path),
            step=1000,
            server=_FakeServer(LOGPROBS_EARLY),
        )
        evaluator.all_metrics_path = tmp_path / "missing_dir" / "all_metrics.json"

        metrics = evaluator.get_metrics()

        assert metrics["val_loss/ocr"] == pytest.approx(EXPECTED_EARLY_VAL_LOSS)
        assert not evaluator.all_metrics_path.exists()

    def test_cache_written_under_run_dir_and_reused_for_same_step(
        self, tmp_path: Path, monkeypatch, dataset_path: Path
    ):
        cfg = _config(dataset_path)
        first = _FakeServer(LOGPROBS_EARLY)
        _run_evaluator(tmp_path, monkeypatch, cfg, step=1000, server=first)
        second = _FakeServer(LOGPROBS_LATE)
        evaluator = _run_evaluator(tmp_path, monkeypatch, cfg, step=1000, server=second)

        cache_dir = tmp_path / "run" / "token_nll_cache" / "ocr"
        assert len(list(cache_dir.glob("tokens_*.npy"))) == 1
        assert len(list(cache_dir.glob("nll_*_ba1000.npy"))) == 1
        assert not list(cache_dir.glob("*.tmp.npy"))
        assert first.requests and not second.requests
        assert evaluator.get_metrics()["val_loss/ocr"] == pytest.approx(
            EXPECTED_EARLY_VAL_LOSS
        )

    def test_corrupt_cache_is_recomputed(
        self, tmp_path: Path, monkeypatch, dataset_path: Path
    ):
        cfg = _config(dataset_path)
        _run_evaluator(
            tmp_path, monkeypatch, cfg, step=1000, server=_FakeServer(LOGPROBS_EARLY)
        )
        cache_dir = tmp_path / "run" / "token_nll_cache" / "ocr"
        (nll_path,) = cache_dir.glob("nll_*_ba1000.npy")
        nll_path.write_bytes(b"garbage")

        server = _FakeServer(LOGPROBS_LATE)
        evaluator = _run_evaluator(tmp_path, monkeypatch, cfg, step=1000, server=server)

        assert server.requests
        expected = (0.005 + 0.8 + 0.2 + 0.005 + 0.8) / 5
        assert evaluator.get_metrics()["val_loss/ocr"] == pytest.approx(expected)
        assert np.load(nll_path).shape == (7,)

    def test_corrupt_reference_skips_memorization_but_keeps_val_loss(
        self, tmp_path: Path, monkeypatch, dataset_path: Path
    ):
        cfg = _config(dataset_path)
        _run_evaluator(
            tmp_path, monkeypatch, cfg, step=1000, server=_FakeServer(LOGPROBS_EARLY)
        )
        cache_dir = tmp_path / "run" / "token_nll_cache" / "ocr"
        (ref_path,) = cache_dir.glob("nll_*_ba1000.npy")
        ref_path.write_bytes(b"garbage")

        evaluator = _run_evaluator(
            tmp_path, monkeypatch, cfg, step=21000, server=_FakeServer(LOGPROBS_LATE)
        )

        metrics = evaluator.get_metrics()
        assert "val_loss/ocr" in metrics
        assert not [key for key in metrics if key.startswith("memorization/")]
        assert not ref_path.exists()

    def test_fp8_checkpoint_gets_its_own_cache(
        self, tmp_path: Path, monkeypatch, dataset_path: Path
    ):
        cfg = _config(dataset_path)
        _run_evaluator(
            tmp_path, monkeypatch, cfg, step=1000, server=_FakeServer(LOGPROBS_EARLY)
        )
        fp8 = _FakeServer(LOGPROBS_LATE)
        _run_evaluator(
            tmp_path,
            monkeypatch,
            cfg,
            step=1000,
            server=fp8,
            hf_name="hf_checkpoint_fp8",
        )

        cache_dir = tmp_path / "run" / "token_nll_cache" / "ocr"
        assert fp8.requests
        assert len(list(cache_dir.glob("nll_*_ba1000.npy"))) == 2

    def test_request_failure_publishes_no_metrics(
        self, tmp_path: Path, monkeypatch, dataset_path: Path
    ):
        monkeypatch.setattr(token_nll.time, "sleep", lambda seconds: None)

        class _BrokenServer(_FakeServer):
            def post(self, url, json, timeout):
                raise token_nll.requests.exceptions.ConnectionError("down")

        evaluator = _run_evaluator(
            tmp_path,
            monkeypatch,
            _config(dataset_path),
            step=1000,
            server=_BrokenServer({}),
        )

        assert evaluator.get_metrics() == {}

    def test_tokenizer_failure_publishes_no_metrics(
        self, tmp_path: Path, monkeypatch, dataset_path: Path
    ):
        class _BrokenTokenizer(_FakeTokenizer):
            def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
                raise OSError("broken tokenizer files")

        evaluator = _run_evaluator(
            tmp_path,
            monkeypatch,
            _config(dataset_path),
            step=1000,
            server=_FakeServer(LOGPROBS_EARLY),
            tokenizer=_BrokenTokenizer(),
        )

        assert evaluator.get_metrics() == {}


class TestMemorization:
    def test_proxy_against_lagged_reference_and_spans_file(
        self, tmp_path: Path, monkeypatch, dataset_path: Path
    ):
        cfg = _config(dataset_path)
        _run_evaluator(
            tmp_path, monkeypatch, cfg, step=1000, server=_FakeServer(LOGPROBS_EARLY)
        )
        evaluator = _run_evaluator(
            tmp_path, monkeypatch, cfg, step=21000, server=_FakeServer(LOGPROBS_LATE)
        )

        metrics = evaluator.get_metrics()
        # delta per scored token (a,b,c,a,b): 0.495, 0.2, 0, 0.495, 0.2 -> total 1.39
        # tokens with nll < 0.01 at the current step: both "a" -> 0.99
        assert metrics["memorization/ocr/proxy"] == pytest.approx(0.99 / 1.39)
        assert metrics["memorization/ocr/frac_tokens_below_thr"] == pytest.approx(2 / 5)
        assert metrics["memorization/ocr/delta_vs_ref"] == pytest.approx(1.39 / 5)
        assert metrics["memorization/ocr/ref_step"] == 1000

        spans_path = evaluator.metrics_dir / "memorized_spans_ocr.jsonl"
        spans = [json.loads(line) for line in spans_path.read_text().splitlines()]
        assert [(s["doc"], s["offset"], s["n_tokens"], s["text"]) for s in spans] == [
            (0, 1, 1, "a"),
            (1, 1, 1, "a"),
        ]
        assert spans[0]["contribution"] == pytest.approx(0.495)

    def test_half_written_reference_is_never_picked(
        self, tmp_path: Path, monkeypatch, dataset_path: Path
    ):
        cfg = _config(dataset_path)
        _run_evaluator(
            tmp_path, monkeypatch, cfg, step=1000, server=_FakeServer(LOGPROBS_EARLY)
        )
        cache_dir = tmp_path / "run" / "token_nll_cache" / "ocr"
        (ref_path,) = cache_dir.glob("nll_*_ba1000.npy")
        in_progress = ref_path.with_name(f".{ref_path.stem[:-4]}1500.tmp.npy")
        in_progress.write_bytes(b"partial")

        evaluator = _run_evaluator(
            tmp_path, monkeypatch, cfg, step=21000, server=_FakeServer(LOGPROBS_LATE)
        )

        assert evaluator.get_metrics()["memorization/ocr/ref_step"] == 1000

    def test_no_reference_within_lag_skips_memorization_metrics(
        self, tmp_path: Path, monkeypatch, dataset_path: Path
    ):
        cfg = _config(dataset_path, ref_lag_batches=30000)
        _run_evaluator(
            tmp_path, monkeypatch, cfg, step=1000, server=_FakeServer(LOGPROBS_EARLY)
        )
        evaluator = _run_evaluator(
            tmp_path, monkeypatch, cfg, step=21000, server=_FakeServer(LOGPROBS_LATE)
        )

        metrics = evaluator.get_metrics()
        assert "val_loss/ocr" in metrics
        assert not [key for key in metrics if key.startswith("memorization/")]

    def test_reference_with_different_tokens_is_ignored(
        self, tmp_path: Path, monkeypatch, dataset_path: Path
    ):
        cfg = _config(dataset_path)
        _run_evaluator(
            tmp_path, monkeypatch, cfg, step=1000, server=_FakeServer(LOGPROBS_EARLY)
        )
        _write_dataset(dataset_path, ["abc", "abc"])
        evaluator = _run_evaluator(
            tmp_path, monkeypatch, cfg, step=21000, server=_FakeServer(LOGPROBS_LATE)
        )

        metrics = evaluator.get_metrics()
        assert not [key for key in metrics if key.startswith("memorization/")]
        cache_dir = tmp_path / "run" / "token_nll_cache" / "ocr"
        assert len(list(cache_dir.glob("tokens_*.npy"))) == 2


class TestPureFunctions:
    @pytest.mark.parametrize(
        ("ref", "cur", "expected_proxy"),
        [
            ([1.0, 1.0, 1.0], [0.001, 0.5, 1.0], 0.999 / 1.499),
            ([1.0, 1.0], [0.5, 0.5], 0.0),
            ([math.nan, 1.0], [math.nan, 0.001], 1.0),
            ([0.5, 0.5], [1.0, 0.001], None),  # no overall improvement: proxy undefined
        ],
    )
    def test_memorization_proxy(self, ref, cur, expected_proxy):
        result = TokenNllEvaluator.memorization_proxy(
            np.array(ref, dtype=np.float32),
            np.array(cur, dtype=np.float32),
            threshold=0.01,
        )
        if expected_proxy is None:
            assert "proxy" not in result
        else:
            assert result["proxy"] == pytest.approx(expected_proxy)

    def test_delta_is_split_between_memorized_and_remaining_tokens(self):
        # scored tokens: two below the threshold now (delta 0.9, 0.5), one above (delta -0.2)
        result = TokenNllEvaluator.memorization_proxy(
            np.array([1.0, 0.6, 0.4], dtype=np.float32),
            np.array([0.1, 0.1, 0.6], dtype=np.float32),
            threshold=0.2,
        )

        assert result["delta_below_thr"] == pytest.approx((0.9 + 0.5) / 2)
        assert result["delta_rest"] == pytest.approx(-0.2)
        assert result["delta_vs_ref"] == pytest.approx((0.9 + 0.5 - 0.2) / 3)

    def test_group_deltas_omitted_when_a_group_is_empty(self):
        no_memorized = TokenNllEvaluator.memorization_proxy(
            np.array([1.0, 1.0], dtype=np.float32),
            np.array([0.5, 0.5], dtype=np.float32),
            threshold=0.01,
        )
        all_memorized = TokenNllEvaluator.memorization_proxy(
            np.array([1.0, 1.0], dtype=np.float32),
            np.array([0.001, 0.002], dtype=np.float32),
            threshold=0.01,
        )

        assert "delta_below_thr" not in no_memorized
        assert "delta_rest" in no_memorized
        assert "delta_rest" not in all_memorized
        assert all_memorized["delta_below_thr"] == pytest.approx(0.9985, rel=1e-3)

    def test_memorized_spans_groups_contiguous_tokens(self):
        ref = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        cur = np.array([0.001, 0.002, 0.5, 0.003, 0.5], dtype=np.float32)
        assert TokenNllEvaluator.memorized_spans(ref, cur, threshold=0.01) == [
            (0, 2, pytest.approx(1.997)),
            (3, 4, pytest.approx(0.997)),
        ]


@pytest.mark.usefixtures("scalar_ci_upload")
def test_after_run_sends_metrics_to_jd(tmp_path: Path, monkeypatch, dataset_path: Path):
    evaluator = _run_evaluator(
        tmp_path,
        monkeypatch,
        _config(dataset_path),
        step=1000,
        server=_FakeServer(LOGPROBS_EARLY),
    )
    sent: dict = {}
    monkeypatch.setattr(
        JD, "send_evaluator_metrics", lambda **kwargs: sent.update(kwargs)
    )
    monkeypatch.setattr(JD, "tar_metrics_dir", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        JD, "symlink_metrics_for_transfer", lambda *args, **kwargs: None
    )

    metrics = evaluator.get_metrics()
    evaluator.after_run(metrics)

    assert sent["shooter_name"] == "token_nll"
    assert sent["jd_metrics"] == metrics
    assert json.loads(evaluator.all_metrics_path.read_text()) == metrics


def test_after_run_skips_jd_when_metrics_are_not_sendable(
    tmp_path: Path, monkeypatch, dataset_path: Path
):
    evaluator = _run_evaluator(
        tmp_path,
        monkeypatch,
        _config(dataset_path),
        step=1000,
        server=_FakeServer(LOGPROBS_EARLY),
    )
    sent: list = []
    monkeypatch.setattr(JD, "can_send_metrics", lambda **kwargs: False)
    monkeypatch.setattr(
        JD, "send_evaluator_metrics", lambda **kwargs: sent.append(kwargs)
    )
    monkeypatch.setattr(JD, "tar_metrics_dir", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        JD, "symlink_metrics_for_transfer", lambda *args, **kwargs: None
    )

    evaluator.after_run(evaluator.get_metrics())

    assert sent == []


def test_tokenizer_loaded_from_plain_dir_when_present(
    tmp_path: Path, monkeypatch, dataset_path: Path
):
    artifacts = tmp_path / "run" / "ep0_ba10" / "artifacts"
    (artifacts / "tokenizer").mkdir(parents=True)
    (artifacts / "tokenizer" / "tokenizer_config.json").write_text("{}")
    seen: list[str] = []

    def fake_from_pretrained(path):
        seen.append(path)
        return _FakeTokenizer()

    monkeypatch.setattr(
        token_nll.AutoTokenizer, "from_pretrained", fake_from_pretrained
    )
    monkeypatch.setattr(token_nll.requests, "post", _FakeServer(LOGPROBS_EARLY).post)
    evaluator = TokenNllEvaluator()
    evaluator.setup(
        autometric_runs_config=_config(dataset_path, ref_lag_batches=5),
        artifacts_path=artifacts,
        hf_checkpoint_path=artifacts / "hf_checkpoint",
        model_info=ModelInfo(model_class="pretrain", size="lite", username="u"),
    )
    evaluator.run()

    assert seen == [(artifacts / "tokenizer").as_posix()]
    assert not evaluator._failed


def test_incomplete_token_mask_keeps_point_metric_without_ci(
    tmp_path, monkeypatch, dataset_path
):
    monkeypatch.setattr(
        TokenNllEvaluator,
        "_score",
        lambda self, tokens, offsets: np.array(
            [np.nan, 0.5, np.nan, 0.2, np.nan, 0.5, 1.0]
        ),
    )
    evaluator = _run_evaluator(
        tmp_path, monkeypatch, _config(dataset_path), 1000, _FakeServer(LOGPROBS_EARLY)
    )
    assert evaluator.get_metrics()["val_loss/ocr"] == pytest.approx(0.55)
    assert evaluator._loss_samples == {}


@pytest.mark.usefixtures("scalar_ci_upload")
@pytest.mark.parametrize("token_count", [0, 2])
def test_ci_withholds_combined_loss_for_incomplete_datasets(
    tmp_path, monkeypatch, dataset_path, token_count
):
    evaluator = _run_evaluator(
        tmp_path, monkeypatch, _config(dataset_path), 1000, _FakeServer(LOGPROBS_EARLY)
    )
    evaluator._loss_samples = {
        "other": ([("a", 0.0, 1), ("b", 1.0, 1)], token_count, "mask")
    }
    monkeypatch.setattr(JD, "send_evaluator_metrics", lambda **kwargs: None)
    evaluator._send_metrics_to_jd({"val_loss/other": 0.5, "val_loss/all": 0.5})
    rows = [
        json.loads(line)
        for line in (evaluator.metrics_dir / "ci_samples.jsonl")
        .read_text()
        .splitlines()
    ]
    assert {row["benchmark"] for row in rows} == (
        {"val_loss/other"} if token_count else set()
    )
