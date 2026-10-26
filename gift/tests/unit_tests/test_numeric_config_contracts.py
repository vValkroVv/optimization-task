"""Invalid arithmetic inputs fail at public config boundaries, before torch."""

import pytest

from torchtitan.config import run_pre_submit


@pytest.mark.parametrize(
    "path,value",
    [
        ("model.config.blocks.gdn_moe.moe.router.top_k", "0"),
        ("model.config.blocks.gdn_moe.moe.router.top_k", "100"),
        ("model.config.blocks.gdn_moe.moe.router.num_expert_groups", "0"),
        ("model.config.blocks.gdn_moe.moe.router_z_loss_coef", "-1"),
        ("model.config.blocks.gdn_moe.moe.router_z_loss_coef", ".nan"),
        ("model.config.blocks.gdn_moe.moe.router_z_loss_coef", ".inf"),
        ("model.config.blocks.gdn_moe.moe.experts.w1_hidden_z_loss_coef", ".nan"),
        ("model.config.blocks.gdn_moe.moe.experts.w3_hidden_z_loss_coef", "-1"),
        ("model.config.blocks.gdn_moe.moe.experts.hidden_dim", "0"),
        ("model.config.blocks.gdn_moe.moe.experts.num_experts", "0"),
        ("model.config.blocks.gdn_moe.moe.shared_experts.hidden_dim", "0"),
        ("lr_scheduler.min_lr_factor", "-0.1"),
        ("lr_scheduler.min_lr_factor", "1.1"),
        ("lr_scheduler.min_lr_factor", ".nan"),
        ("lr_scheduler.decay_ratio", "-0.1"),
        ("lr_scheduler.decay_ratio", "1.1"),
        ("lr_scheduler.decay_ratio", ".inf"),
        ("metrics.log_freq", "0"),
        ("metrics.log_freq", "-1"),
    ],
)
def test_invalid_scalar_has_the_written_field_path(path, value):
    result = run_pre_submit("configs/experiments/gift_3_5_debug.yaml", [(path, value)])
    assert not result.ok
    assert any(issue.path == path for issue in result.report.errors), (
        result.report.errors
    )


@pytest.mark.parametrize(
    "value,path",
    [
        ('{"*": -1}', "model.config.hidden_z_loss_patterns.*"),
        ('{"*": .nan}', "model.config.hidden_z_loss_patterns.*"),
        ('{"*": .inf}', "model.config.hidden_z_loss_patterns.*"),
        ('{"*": {coef: -1}}', "model.config.hidden_z_loss_patterns.*.coef"),
    ],
)
def test_hidden_loss_patterns_are_checked_before_placement(value, path):
    result = run_pre_submit(
        "configs/experiments/llama3_debug.yaml",
        [("model.config.hidden_z_loss_patterns", value)],
    )
    assert not result.ok
    assert any(
        issue.path == path
        or (
            issue.path == "model.config.hidden_z_loss_patterns.*"
            and path.endswith(".coef")
            and path in issue.message
        )
        for issue in result.report.errors
    ), result.report.errors


@pytest.mark.parametrize("value", ["0", "1"])
def test_scheduler_endpoints_and_zero_loss_remain_valid(value):
    result = run_pre_submit(
        "configs/experiments/gift_3_5_debug.yaml",
        [
            ("lr_scheduler.min_lr_factor", value),
            ("lr_scheduler.decay_ratio", value),
            ("model.config.blocks.gdn_moe.moe.router_z_loss_coef", "0"),
            ("model.config.blocks.gdn_moe.moe.router.top_k", "8"),
        ],
    )
    assert result.ok, result.report.errors


def test_shared_expert_dimension_mismatch_rejects_during_owner_completion():
    result = run_pre_submit(
        "configs/experiments/gift_3_5_debug.yaml",
        [
            ("model.config.blocks.gdn_moe.moe.shared_experts.dim", "128"),
        ],
    )
    assert not result.ok
    assert any(
        "shared_experts.dim must match experts.dim (256), got 128" in issue.message
        for issue in result.report.errors
    )
