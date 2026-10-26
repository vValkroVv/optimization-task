from __future__ import annotations

import pytest

from metric_sampler.cloud import queries

pytestmark = pytest.mark.unit

_QUERY_NAMES = [
    "LOSS_PER_STEP",
    "LAYER_LOAD_HEATMAP",
    "STEP_DRILLDOWN",
    "HEALTH",
    "LOSS_QUANTILES",
    "STREAM_LOSS",
    "EXPERT_LOSS",
    "GRAD_PER_RANK",
    "ACTIVATION_PER_STEP",
    "ACTIVATION_EXP_QUANTILES",
    "ACTIVATION_MASSIVE_TOKENS",
]


@pytest.mark.parametrize("name", _QUERY_NAMES)
def test_query_is_a_single_select_statement(name: str) -> None:
    sql = getattr(queries, name)
    assert sql.strip().upper().startswith("SELECT")
    assert ";" not in sql


@pytest.mark.parametrize(
    "name, kwargs, expected_fragments",
    [
        (
            "LOSS_PER_STEP",
            {"exp": "expA", "run": "run1"},
            ["exp = 'expA'", "run = 'run1'", "metric_sampler.loss_per_step"],
        ),
        (
            "LAYER_LOAD_HEATMAP",
            {"exp": "expA", "run": "run1", "layer": 3, "step_from": 10, "step_to": 20},
            ["layer = 3", "step BETWEEN 10 AND 20", "metric_sampler.expert_totals_per_step"],
        ),
        (
            "STEP_DRILLDOWN",
            {"exp": "expA", "run": "run1", "step": 42},
            [
                "s.step = 42",
                "metric_sampler.scalars",
                "metric_sampler.metric_names",
                "metric_sampler.latest_attempt_step",
                "s.attempt",
            ],
        ),
        (
            "HEALTH",
            {"exp": "expA", "run": "run1"},
            ["exp = 'expA'", "run = 'run1'", "metric_sampler.health_per_step"],
        ),
        (
            "LOSS_QUANTILES",
            {"exp": "expA", "run": "run1"},
            ["exp = 'expA'", "run = 'run1'", "metric_sampler.loss_quantiles_per_step"],
        ),
        (
            "STREAM_LOSS",
            {"exp": "expA", "run": "run1"},
            ["exp = 'expA'", "run = 'run1'", "metric_sampler.stream_loss_per_step"],
        ),
        (
            "EXPERT_LOSS",
            {"exp": "expA", "run": "run1", "layer": 3},
            ["exp = 'expA'", "run = 'run1'", "layer = 3", "metric_sampler.expert_loss_per_step"],
        ),
        (
            "GRAD_PER_RANK",
            {"exp": "expA", "run": "run1"},
            ["exp = 'expA'", "run = 'run1'", "metric_sampler.grad_per_rank"],
        ),
        (
            "ACTIVATION_PER_STEP",
            {"exp": "expA", "run": "run1"},
            ["exp = 'expA'", "run = 'run1'", "metric_sampler.activation_per_step"],
        ),
        (
            "ACTIVATION_EXP_QUANTILES",
            {"exp": "expA", "run": "run1", "layer": 3, "point": "residual"},
            [
                "exp = 'expA'",
                "run = 'run1'",
                "layer = 3",
                "point = 'residual'",
                "metric_sampler.activation_exp_quantiles",
            ],
        ),
        (
            "ACTIVATION_MASSIVE_TOKENS",
            {"exp": "expA", "run": "run1"},
            ["exp = 'expA'", "run = 'run1'", "metric_sampler.activation_massive_tokens"],
        ),
    ],
)
def test_query_formats_with_its_expected_placeholders(
    name: str, kwargs: dict[str, object], expected_fragments: list[str]
) -> None:
    sql = getattr(queries, name).format(**kwargs)
    for fragment in expected_fragments:
        assert fragment in sql
    assert "{" not in sql
    assert "}" not in sql
