from __future__ import annotations

import json
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from metric_sampler.contract.metrics import ACTIVATION_LOG2_BINS, ENTROPY_TOKEN_STRIDE
from stand import validate_distributions as validate

pytestmark = pytest.mark.unit

_TOP_K = 2
_TOKENS_PER_MICROBATCH = 4
_TOKEN_BINS = {"count": 2, "start": 0.0, "width": 1.0}
_COUNT_METRICS = (
    "loss/stream_count", "loss/token_hist_by_stream", "expert_loss/count", "expert_loss/hist",
    "expert_loss/stream_count",
)
_SCALAR_METRICS = ("loss/stream_sum", "loss/ce", "expert_loss/sum", "grad/clip_total_norm")
_REGISTRY = [
    {"metric_id": metric_id, "name": name, **({"bins": _TOKEN_BINS} if name == "loss/token_hist_by_stream" else {})}
    for metric_id, name in enumerate((*_COUNT_METRICS, *_SCALAR_METRICS))
]
_ACTIVATION_BINS = ACTIVATION_LOG2_BINS.count
_ZERO_EXPONENT_BIN = -int(ACTIVATION_LOG2_BINS.start)


def _row(name: str, step: int, values: list, *, layer: int = 0, microbatch: int = 0, identical: bool = True) -> dict:
    return {
        "metric_id": [entry["name"] for entry in _REGISTRY].index(name), "node": "n0", "step": step,
        "layer": layer, "microbatch": microbatch, "contributors": 1, "identical": identical, "values": values,
    }


def _write_family(window: Path, family: str, rows: list[dict], value_type: pa.DataType) -> None:
    columns = {key: [row[key] for row in rows] for key in rows[0] if key != "values"}
    columns["values"] = pa.array([row["values"] for row in rows], type=pa.list_(value_type))
    pq.write_table(pa.table(columns), window / f"{family}.parquet")


def _step_rows(step: int, *, stream_count: int, identical: bool) -> tuple[list[dict], list[dict]]:
    counts = [
        _row("loss/stream_count", step, [stream_count]),
        _row("loss/token_hist_by_stream", step, [3, 1]),
        _row("expert_loss/count", step, [4, 4]),
    ]
    scalars = [
        _row("loss/stream_sum", step, [2.0]),
        _row("loss/ce", step, [0.5]),
        _row("expert_loss/sum", step, [2.0, 2.0]),
        _row("grad/clip_total_norm", step, [1.0], identical=identical),
    ]
    return counts, scalars


def test_main_reports_the_steps_that_fail(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_dir = tmp_path / "exp" / "run" / "run1"
    attempt = run_dir / "attempt=0"
    window = attempt / "node=n0" / "w00000001-00000002"
    window.mkdir(parents=True)
    (attempt / "metrics.json").write_text(json.dumps(_REGISTRY))
    (attempt / "streams.json").write_text(json.dumps(["a", "b", "c"]))
    healthy_counts, healthy_scalars = _step_rows(1, stream_count=4, identical=True)
    failing_counts, failing_scalars = _step_rows(2, stream_count=5, identical=False)
    detailed = [_row("expert_loss/hist", 1, [8]), _row("expert_loss/stream_count", 1, [8])]
    _write_family(window, "expert_counts", healthy_counts + detailed + failing_counts, pa.int64())
    _write_family(window, "scalars", healthy_scalars + failing_scalars, pa.float64())
    logged = attempt / "logged" / "w00000001-00000002"
    logged.mkdir(parents=True)
    pq.write_table(pa.table({"step": [1, 2]}), logged / "logged.parquet")

    validate.main(str(run_dir), _TOP_K, _TOKENS_PER_MICROBATCH)

    assert capsys.readouterr().out.splitlines() == [
        "FAIL n0 step 2: hist tokens 4 != stream count 5; layer 0: expert count 8 != top_k * tokens 10; "
        "stream mean 0.40000 vs loss/ce mean 0.50000; grad/clip_total_norm differs across ranks",
        "attempt=0: logged windows 1, rows 2",
        "attempt=0: streams 3",
        "checked 2 node-steps (1 detailed), failures 1",
    ]


def test_main_of_an_attempt_without_windows(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_dir = tmp_path / "exp" / "run" / "run1"
    attempt = run_dir / "attempt=0"
    attempt.mkdir(parents=True)
    (attempt / "metrics.json").write_text(json.dumps(_REGISTRY))

    validate.main(str(run_dir), _TOP_K, _TOKENS_PER_MICROBATCH)

    assert capsys.readouterr().out.splitlines() == [
        "attempt=0: logged windows 0, rows 0",
        "checked 0 node-steps (0 detailed), failures 0",
    ]


def _activation_metrics(
    point: str = "mlp_out", *, finite: float = 8.0, sum_of_squares: float = 8.0, token_total: int = 2,
    exponent_total: float | None = None, amax: float = 1.5, with_token_histogram: bool = True,
) -> dict:
    """One layer's statistics of an activation point: two tokens of root mean square one, `finite` elements."""
    token_histogram = [0] * _ACTIVATION_BINS
    token_histogram[_ZERO_EXPONENT_BIN] = token_total
    exponent_histogram = [0] * _ACTIVATION_BINS
    exponent_histogram[_ZERO_EXPONENT_BIN] = finite if exponent_total is None else exponent_total
    metrics = {
        "loss/ce": [{"microbatch": 0}],
        f"act/{point}/moments": [{
            "name": f"act/{point}/moments", "layer": 0, "contributors": 1,
            "values": [0.0, sum_of_squares, 0.0, finite],
        }],
        f"act/{point}/exp_hist": [{"layer": 0, "values": exponent_histogram}],
        f"act/{point}/amax": [{"layer": 0, "values": [amax]}],
    }
    if with_token_histogram:
        metrics[f"act/{point}/token_rms_hist"] = [{"layer": 0, "values": token_histogram}]
    return metrics


@pytest.mark.parametrize(
    ("metrics", "tokens_per_microbatch", "expected"),
    [
        pytest.param(_activation_metrics(), 2, [], id="consistent"),
        pytest.param(
            _activation_metrics(with_token_histogram=False), 2,
            ["act/mlp_out layer 0: moments without token_rms_hist"], id="without_token_histogram",
        ),
        pytest.param(
            _activation_metrics(), 3, ["act/mlp_out layer 0: rms hist tokens 2 != 3"], id="token_total",
        ),
        pytest.param(
            _activation_metrics(finite=9.0, sum_of_squares=9.0), 2,
            ["act/mlp_out layer 0: elements per token 4.5000 is not whole"], id="fractional_width",
        ),
        pytest.param(
            _activation_metrics(sum_of_squares=8.0 * 2**10), 2,
            ["act/mlp_out layer 0: log2 rms 5.000 outside token range [0, 1]"], id="outside_token_range",
        ),
        pytest.param(
            _activation_metrics(exponent_total=7.0), 2,
            ["act/mlp_out layer 0: exp hist 7 != finite 8"], id="exponent_total",
        ),
        pytest.param(
            _activation_metrics(amax=2.0), 2,
            ["act/mlp_out layer 0: amax 2.0 above top exponent bin 0"], id="amax_above_top_bin",
        ),
        pytest.param(
            {**_activation_metrics("residual"), **_activation_metrics("attn_out", finite=16.0, sum_of_squares=16.0)},
            2, ["activation widths differ: {4.0: ['residual@0'], 8.0: ['attn_out@0']}"], id="widths_differ",
        ),
    ],
)
def test_activation_problems(metrics: dict, tokens_per_microbatch: int, expected: list[str]) -> None:
    assert validate.activation_problems(metrics, tokens_per_microbatch) == expected


@pytest.mark.parametrize("shortfall", [0, 1])
def test_entropy_histogram_problems(shortfall: int) -> None:
    microbatches, ranks = 2, 2
    expected_total = math.ceil(_TOKENS_PER_MICROBATCH / ENTROPY_TOKEN_STRIDE) * microbatches * ranks
    metrics = {
        "loss/ce": [{"microbatch": 0}, {"microbatch": 1}],
        "loss/stream_count": [{"contributors": ranks}],
        "router/entropy_hist": [{"layer": 3, "values": [expected_total - shortfall, 0]}],
    }
    expected = [f"layer 3: entropy hist total {expected_total - shortfall} != {expected_total}"] if shortfall else []

    assert validate.entropy_histogram_problems(metrics, _TOKENS_PER_MICROBATCH) == expected


@pytest.mark.parametrize(
    ("stream_count_total", "expected"),
    [(4, []), (5, ["layer 0: expert_loss/stream_count total 5 != expert count 4"])],
)
def test_expert_histogram_problems(stream_count_total: int, expected: list[str]) -> None:
    metrics = {
        "expert_loss/count": [{"layer": 0, "values": [2, 2]}],
        "expert_loss/hist": [{"layer": 0, "values": [3, 1]}],
        "expert_loss/stream_count": [{"layer": 0, "values": [stream_count_total]}],
    }

    assert validate.expert_histogram_problems(metrics) == expected
