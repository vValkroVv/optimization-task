import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Replicate, Shard
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta

from tests.unit_tests.config_helpers import init_policy, load_model_spec
from torchtitan.config.loader import ReplaceMapping
from torchtitan.config.schema.models.common.moe.router import (
    TokenChoiceTopKRouterConfig,
)
from torchtitan.distributed.gfsdp import RaggedShard, RaggedShardDTensor
from torchtitan.models.gift_moe.const import _renorm_keep_std, _renorm_rows_keep_std
from torchtitan.config.schema.protocols.module import iter_module_configs
from torchtitan.protocols.module import resolve_param_init

# The base's router gates initialize like any linear; the hybrids renorm them.
LINEAR_INIT = init_policy("gift_base").router_gate
ROUTER_GATE_RENORM_INIT = init_policy("gift_hybrid").router_gate
INIT_STD = ROUTER_GATE_RENORM_INIT["weight"].init.std

_NUM_EXPERTS = 64
_DIM = 1920
_HYBRID_VERSIONS = ["gift_3_5", "gift_4"]
_MTP = {"config": {"mtp": {"layers": [{"block": "mla_moe"}]}}}
_HYBRID_MTP = {"config": {"mtp": {"layers": [{"block": "mtp"}]}}}
_RENORM = {
    "config": {
        **_MTP["config"],
        # The policy's router_gate entry, layered on the base's default
        # (LINEAR_INIT): weight replaced, bias kept.
        "init": {
            "router_gate": {
                # A different initializer type: replace the base's entry, not
                # merge into it.
                "weight": ReplaceMapping(
                    type="renorm_keep_std",
                    init={"type": "normal", "std": 0.006},
                )
            }
        },
    }
}


def _gates(model_config):
    return [
        module.gate
        for _, module in iter_module_configs(model_config)
        if isinstance(module, TokenChoiceTopKRouterConfig)
    ]


def _renorm_init(param: torch.Tensor) -> None:
    resolve_param_init(ROUTER_GATE_RENORM_INIT["weight"])(param)


def _assert_rows_renormed(weight: torch.Tensor):
    row_norms = weight.norm(dim=1)
    torch.testing.assert_close(row_norms, row_norms[0].expand_as(row_norms))
    assert weight.std().item() == pytest.approx(INIT_STD, rel=0.01)


def test_renorm_init_equalizes_row_norms_and_keeps_std():
    param = torch.nn.Parameter(torch.empty(_NUM_EXPERTS, _DIM))
    _renorm_init(param)
    _assert_rows_renormed(param.detach())


def test_renorm_router_weights_is_off_by_default():
    gates = _gates(load_model_spec("gift_moe/debugmodel.yaml", _MTP).model)
    assert gates
    assert all(gate.param_init == LINEAR_INIT for gate in gates)


def test_renorm_router_weights_swaps_only_router_gate_init():
    plain = load_model_spec("gift_moe/debugmodel.yaml", _MTP).model
    renormed = load_model_spec("gift_moe/debugmodel.yaml", _RENORM).model
    gates = _gates(renormed)
    assert gates
    assert all(gate.param_init == ROUTER_GATE_RENORM_INIT for gate in gates)
    for gate in gates:
        gate.param_init = LINEAR_INIT
    # The compact policy entry that produced them is the only other difference.
    renormed.init = plain.init
    assert repr(renormed) == repr(plain)


def test_renorm_router_weights_reaches_the_mtp_head_router():
    model = load_model_spec("gift_moe/debugmodel.yaml", _RENORM).model
    mtp_gates = _gates(model.mtp)
    assert mtp_gates
    assert all(gate.param_init == ROUTER_GATE_RENORM_INIT for gate in mtp_gates)


@pytest.mark.parametrize("version", _HYBRID_VERSIONS)
@pytest.mark.parametrize("size", ["debugmodel", "3b", "700b"])
def test_hybrid_builders_always_renorm_router_weights(version, size):
    gates = _gates(
        load_model_spec(f"gift_moe/{version}/{size}.yaml", _HYBRID_MTP).model
    )
    assert gates
    assert all(gate.param_init == ROUTER_GATE_RENORM_INIT for gate in gates)


def _renorm_distributed(rank, world_size, store_path, placements):
    dist.init_process_group(
        "gloo", init_method=f"file://{store_path}", rank=rank, world_size=world_size
    )
    try:
        torch.manual_seed(0)
        shape = (world_size,) if len(placements) == 1 else (world_size // 2, 2)
        mesh = init_device_mesh("cpu", shape)
        weight = distribute_tensor(torch.empty(_NUM_EXPERTS, _DIM), mesh, placements)
        _renorm_init(weight)
        _assert_rows_renormed(weight.full_tensor())
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "placements",
    [[Shard(0)], [Shard(1)], [Shard(0), Replicate()]],
    ids=["fsdp", "input_dim_sharded", "fsdp_tp"],
)
def test_renorm_init_under_dtensor_placements(tmp_path, placements):
    world_size = 4
    mp.spawn(
        _renorm_distributed,
        args=(world_size, str(tmp_path / "store"), placements),
        nprocs=world_size,
        join=True,
    )


# 60 elements: 15 per rank on a 4-rank ragged axis and 30 on a 2-rank one,
# so in both layouts a shard boundary falls mid-row.
_RAGGED_SHAPE = (5, 12)


def _ragged_param(full, mesh, ragged_axis):
    """``full`` laid out as gfsdp2 shards it: an even flat split on ``ragged_axis``."""
    num_shards = mesh.size(ragged_axis)
    shard_numel = full.numel() // num_shards
    index = mesh.get_coordinate()[ragged_axis]
    placement = RaggedShard(
        local_numel=shard_numel,
        global_offset=index * shard_numel,
        shard_numels=(shard_numel,) * num_shards,
    )
    placements = tuple(
        placement if axis == ragged_axis else Replicate() for axis in range(mesh.ndim)
    )
    spec = DTensorSpec(
        mesh, placements, TensorMeta(full.shape, full.stride(), full.dtype)
    )
    local = full.flatten()[index * shard_numel : (index + 1) * shard_numel].clone()
    return RaggedShardDTensor(local, spec, requires_grad=False)


def _renorm_ragged(rank, world_size, store_path, mesh_shape):
    dist.init_process_group(
        "gloo", init_method=f"file://{store_path}", rank=rank, world_size=world_size
    )
    try:
        mesh = init_device_mesh("cpu", mesh_shape)
        full = torch.randn(_RAGGED_SHAPE, generator=torch.Generator().manual_seed(0))
        weight = _ragged_param(full, mesh, ragged_axis=mesh.ndim - 1)
        _renorm_keep_std(weight)

        expected = full.clone()
        _renorm_rows_keep_std(expected)
        result = weight.full_tensor()
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        row_norms = result.norm(dim=1)
        torch.testing.assert_close(row_norms, row_norms[0].expand_as(row_norms))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("mesh_shape", [(4,), (2, 2)], ids=["fsdp", "hsdp"])
def test_renorm_init_under_gfsdp2_ragged_shards(tmp_path, mesh_shape):
    world_size = 4
    mp.spawn(
        _renorm_ragged,
        args=(world_size, str(tmp_path / "store"), mesh_shape),
        nprocs=world_size,
        join=True,
    )
