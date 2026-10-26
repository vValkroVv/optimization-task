from collections import UserDict

import pytest
import torch

from composer.optim import DecoupledAdamW


def _optimizer(overrides):
    param = torch.nn.Parameter(torch.zeros(8))
    return DecoupledAdamW([param], per_fqn_overrides=overrides), param


@pytest.mark.parametrize(
    "overrides",
    [
        [{}],
        [{"param_str_match": "[", "lr_mult": 1.0}],
        [{"param_str_match": "weight", "lr_mult": -1.0}],
        [{"param_str_match": "weight", "wd_mult": float("inf")}],
        [{"param_str_match": "weight", "unknown": 1.0}],
    ],
)
def test_per_fqn_overrides_are_validated_at_construction(overrides):
    with pytest.raises((TypeError, ValueError)):
        _optimizer(overrides)


def test_per_fqn_overrides_accept_mapping_entries():
    optimizer, _ = _optimizer(
        [UserDict({"param_str_match": r"block\.weight", "wd_mult": 0.0})]
    )

    assert optimizer._resolve_fqn_mult("block.weight") == (1.0, 0.0)


def test_per_fqn_override_precedence_and_vector_exposure(monkeypatch):
    optimizer, param = _optimizer(
        [
            {"param_str_match": r"block\.", "lr_mult": 0.5, "wd_mult": 0.25},
            {"param_str_match": r"block\.second", "lr_mult": 2.0},
        ]
    )
    monkeypatch.setattr(optimizer, "get_fsdp_pg_rank", lambda _fqn: 0)
    monkeypatch.setattr(optimizer, "get_fsdp_pg_world_size", lambda _fqn: 1)

    optimizer._build_per_fqn_vec_for_param(
        param,
        ["block.first.weight", "block.second.weight"],
        [3, 5],
    )

    assert torch.equal(
        optimizer.get_per_fqn_lr_multiplier(param),
        torch.tensor([0.5, 0.5, 0.5, 2.0, 2.0, 2.0, 2.0, 2.0]),
    )
    assert torch.equal(
        optimizer.get_per_fqn_wd_multiplier(param),
        torch.tensor([0.25] * 8),
    )


def test_mismatched_packed_metadata_is_ignored(monkeypatch, caplog):
    optimizer, param = _optimizer(
        [
            {"param_str_match": r"block\.", "lr_mult": 0.5},
        ]
    )
    monkeypatch.setattr(optimizer, "get_fsdp_pg_rank", lambda _fqn: 0)
    monkeypatch.setattr(optimizer, "get_fsdp_pg_world_size", lambda _fqn: 1)

    optimizer._build_per_fqn_vec_for_param(param, ["block.weight"], [])

    assert optimizer.get_per_fqn_lr_multiplier(param) is None
    assert "fqn/numel length mismatch" in caplog.text
