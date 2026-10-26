# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""HF <-> native conversion of the fused ``(E, 2, F, D)`` routed-expert weight.

Every MoE family's state dict adapter, on the offline path (plain tensors) and
on the online one (DTensors sharded like EP and TP).
"""

import json

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file

from tests.unit_tests.config_helpers import load_model_spec
from torchtitan.models.common.moe.experts.gmm import GroupedExperts


_FAMILIES = ["qwen3", "deepseek_v3", "gift_moe", "gift_3_5", "llama4", "qwen3_vl"]
# HF stores gate and up per expert here; llama4 and qwen3_vl fuse them into
# one gate_up_proj per layer instead.
_PER_EXPERT_FAMILIES = ["qwen3", "deepseek_v3", "gift_moe", "gift_3_5"]


_PRESETS = {
    "qwen3": ("qwen3/debugmodel_moe.yaml", None),
    "deepseek_v3": ("deepseek_v3/debugmodel.yaml", None),
    "gift_moe": (
        "gift_moe/debugmodel.yaml",
        {"config": {"mtp": {"layers": [{"block": "mla_moe"}]}}},
    ),
    "gift_3_5": (
        "gift_moe/gift_3_5/debugmodel.yaml",
        {"config": {"mtp": {"layers": [{"block": "mtp"}]}}},
    ),
    "llama4": ("llama4/debugmodel.yaml", None),
    "qwen3_vl": ("qwen3_vl/debugmodel_moe.yaml", None),
}


def _spec(family):
    return load_model_spec(*_PRESETS[family])


def _adapter(family):
    """The family's adapter and a native state dict of its routed-expert weights."""
    spec = _spec(family)
    sd = {"lm_head.weight": torch.ones(2, 6, dtype=torch.bfloat16)}
    for fqn, cfg, _, _ in spec.model.traverse(GroupedExperts.Config):
        # Unequal input/output dimensions catch missing HF transposes.
        w1 = (
            torch.arange(cfg.num_experts * 4 * 8, dtype=torch.float32)
            .reshape(cfg.num_experts, 4, 8)
            .to(torch.bfloat16)
        )
        sd[f"{fqn}.w13_E2FD"] = torch.stack((w1, -w1 - 1), dim=1)
        sd[f"{fqn}.w2_EDF"] = w1.transpose(1, 2).contiguous()
    assert len(sd) > 1
    return spec.state_dict_adapter(spec.model, None), sd


def _assert_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


@pytest.mark.parametrize("family", _FAMILIES)
def test_hf_round_trip(family):
    adapter, sd = _adapter(family)
    hf = adapter.to_hf(sd)
    # The HF export goes through safetensors, which refuses strided views.
    assert all(value.is_contiguous() for value in hf.values())
    _assert_equal(adapter.from_hf(hf.copy()), sd)
    _assert_equal(adapter.to_hf(adapter.from_hf(hf.copy())), hf)


@pytest.mark.parametrize("family", _PER_EXPERT_FAMILIES)
def test_from_hf_is_key_order_independent(family):
    adapter, sd = _adapter(family)
    hf = adapter.to_hf(sd)
    _assert_equal(adapter.from_hf(dict(reversed(tuple(hf.items())))), sd)


def test_llama4_dense_ffn_round_trips():
    """Layers ``interleave_moe_layer_step`` leaves dense must reach HF too."""
    spec = _spec("llama4")
    adapter = spec.state_dict_adapter(spec.model, None)
    sd = {
        f"layers.0.feed_forward.{name}.weight": torch.full((4, 6), float(index))
        for index, name in enumerate(("w1", "w2", "w3"), start=1)
    }

    hf = adapter.to_hf(sd)
    assert set(hf) == {
        f"language_model.model.layers.0.feed_forward.{name}.weight"
        for name in ("gate_proj", "down_proj", "up_proj")
    }
    _assert_equal(adapter.from_hf(hf), sd)


def test_to_hf_rejects_wrong_expert_axis():
    adapter, sd = _adapter("qwen3")
    w13_key = next(key for key in sd if key.endswith(".w13_E2FD"))
    sd[w13_key] = sd[w13_key][:-1]

    with pytest.raises(ValueError, match="GroupedExperts weight"):
        adapter.to_hf(sd)


@pytest.mark.parametrize("family", _FAMILIES)
def test_load_hf_into_model_placeholders(family, tmp_path):
    """The trainer's initial_load_in_hf path: to_hf placeholders, dcp.load, from_hf."""
    adapter, sd = _adapter(family)
    hf = adapter.to_hf(sd)
    # clone(): the GIFT MTP clones alias lm_head / embed, which safetensors refuses.
    save_file(
        {key: value.clone() for key, value in hf.items()},
        tmp_path / "model.safetensors",
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {}, "weight_map": {key: "model.safetensors" for key in hf}}
        )
    )

    hf_placeholders = adapter.to_hf(
        {key: torch.zeros_like(value) for key, value in sd.items()}
    )
    dcp.load(hf_placeholders, storage_reader=dcp.HuggingFaceStorageReader(tmp_path))
    _assert_equal(adapter.from_hf(hf_placeholders), sd)


@pytest.mark.parametrize("family", _PER_EXPERT_FAMILIES)
def test_from_hf_rejects_missing_up_projection(family):
    adapter, sd = _adapter(family)
    hf = {
        key: value for key, value in adapter.to_hf(sd).items() if ".up_proj." not in key
    }
    with pytest.raises(ValueError, match="Incomplete gate/up"):
        adapter.from_hf(hf)


@pytest.mark.parametrize("family", _PER_EXPERT_FAMILIES)
def test_from_hf_rejects_incomplete_expert_set(family):
    adapter, sd = _adapter(family)
    hf = {
        key: value
        for key, value in adapter.to_hf(sd).items()
        if ".experts.0.gate_proj." not in key and ".experts.0.up_proj." not in key
    }
    with pytest.raises(ValueError, match="Incomplete expert weights"):
        adapter.from_hf(hf)


def _distributed_round_trip(rank, rendezvous):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import distribute_tensor, Replicate, Shard

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2
    )
    try:
        mesh = init_device_mesh("cpu", (1, 2), mesh_dim_names=("dp_replicate", "ep"))
        # Shard(0) is the EP layout, Shard(2) the TP layout of both grouped weights.
        for family in _FAMILIES:
            for expert_placement in (Shard(0), Shard(2)):
                adapter, sd = _adapter(family)
                native = {
                    key: distribute_tensor(value, mesh, (Replicate(), expert_placement))
                    if ".experts." in key
                    else value
                    for key, value in sd.items()
                }
                hf = adapter.to_hf(native)
                for key, value in hf.items():
                    local = value.to_local() if hasattr(value, "to_local") else value
                    assert local.is_contiguous(), key
                if expert_placement.dim == 2 and family in _PER_EXPERT_FAMILIES:
                    # F is dim 0 of a per-expert (F, D) gate/up and dim 1 of a
                    # per-expert (D, F) down; the TP shard must survive the split.
                    expected_dims = {"gate_proj": 0, "up_proj": 0, "down_proj": 1}
                    for key, value in hf.items():
                        if ".experts." in key:
                            projection = key.rsplit(".", 2)[-2]
                            assert value.placements[-1] == Shard(
                                expected_dims[projection]
                            ), key

                restored = adapter.from_hf(hf)
                assert restored.keys() == native.keys()
                for key, value in native.items():
                    if ".experts." not in key:
                        torch.testing.assert_close(
                            restored[key], sd[key], rtol=0, atol=0
                        )
                        continue
                    expected_placements = value.placements
                    if (
                        family not in _PER_EXPERT_FAMILIES
                        and expert_placement.dim == 2
                        and key.endswith("w13_E2FD")
                    ):
                        # HF's gate_up_proj fuses gate and up along F, so an F
                        # shard has no placement: the export gathers it.
                        expected_placements = (Replicate(), Replicate())
                    assert restored[key].placements == expected_placements, key
                    assert restored[key].to_local().is_contiguous()
                    torch.testing.assert_close(
                        restored[key].full_tensor(), sd[key], rtol=0, atol=0
                    )
    finally:
        dist.destroy_process_group()


def test_distributed_hf_round_trip(tmp_path):
    torch.multiprocessing.spawn(
        _distributed_round_trip,
        args=(str(tmp_path / "rendezvous"),),
        nprocs=2,
        join=True,
    )
