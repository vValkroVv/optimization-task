"""Focused actual-GFSDP checks for the opt-in BF16 expert cache."""

import shutil
import sys
import traceback
import uuid
from functools import wraps
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.checkpoint as dcp
import torch.distributed.tensor._random as dtensor_random
from torch.distributed.tensor import DTensor, Shard
from torch.testing._internal.common_distributed import (
    TIMEOUT_OVERRIDE,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import ActivationCheckpointConfig, ParallelismConfig
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.distributed.backends.gfsdp2 import GFSDP2Backend
from torchtitan.distributed.gfsdp._dcp import GFSDPCheckpointAdapter
from torchtitan.distributed.gfsdp._param import ExtensionsData
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.models.gift_moe._base.parallelize import apply_gfsdp
from tests.unit_tests.config_helpers import load_model_spec


_SEED = 78123
_NUM_LAYERS = 3
_NUM_MICROBATCHES = 2
_NUM_UPDATES = 2
_HIDDEN_SIZE = 256
_SEQUENCE_LENGTH = 4
_VOCAB_SIZE = 2048

# Bound failures in these small two-rank fixtures, rather than waiting on the
# distributed test default timeout when one rank has already raised.
TIMEOUT_OVERRIDE["test_cache_matches_uncached_training_and_weight_lifecycles"] = 90
TIMEOUT_OVERRIDE["test_all_gather_extension_keeps_original_output_buffer"] = 60


def _report_rank_exception(test):
    """Print a rank's original exception before ``with_comms`` tears PG down."""
    @wraps(test)
    def wrapped(self, *args, **kwargs):
        try:
            return test(self, *args, **kwargs)
        except BaseException:
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            raise

    return wrapped


def _expert_groups(model: nn.Module) -> list:
    """Return the expert group from each sequential checkpointed MoE unit."""
    groups = []
    for layer in model.layers.values():
        state = layer._get_fsdp_state()
        dense_group, expert_group = state._fsdp_param_groups
        assert dense_group._all_gather_process_group.size() == 2
        assert expert_group._all_gather_process_group.size() == 1
        groups.append(expert_group)
    assert len(groups) == _NUM_LAYERS
    return groups


def _group_stats(model: nn.Module) -> dict[str, dict[str, object]]:
    return {
        row["fqn"]: row for row in model.bf16_cache_stats(recurse=True)
    }


def _stats_for(model: nn.Module, group) -> dict[str, object]:
    return _group_stats(model)[group._module_fqn]


def _assert_fresh_bf16_cast(group) -> None:
    """The persistent input must match the currently active FP32 master."""
    master = group._padded_sharded_param_data[None]
    cached = group._all_gather_input[None]
    assert master.dtype is torch.float32
    assert cached.dtype is torch.bfloat16
    assert cached.shape == master.shape
    torch.testing.assert_close(cached, master.to(torch.bfloat16), rtol=0, atol=0)


def _master_data(groups: list) -> list[torch.Tensor]:
    return [group._padded_sharded_param_data[None].detach().cpu().clone() for group in groups]


def _all_group_masters(model: nn.Module) -> list[torch.Tensor]:
    masters = []
    for layer in model.layers.values():
        for group in layer._get_fsdp_state()._fsdp_param_groups:
            master = group._padded_sharded_param_data[None]
            assert master.dtype is torch.float32
            masters.append(master.detach().cpu().clone())
    return masters


def _group_gradients(groups: list) -> list[list[torch.Tensor]]:
    gradients = []
    for group in groups:
        group_gradients = []
        for fsdp_param in group.fsdp_params:
            grad = fsdp_param.sharded_param.grad
            assert grad is not None, f"missing expert gradient for {fsdp_param}"
            local_grad = getattr(grad, "_local_tensor", grad)
            group_gradients.append(local_grad.detach().cpu().clone())
        gradients.append(group_gradients)
    return gradients


def _parameter_identities(groups: list) -> list[tuple[tuple[int, int], ...]]:
    return [
        tuple(
            (id(fsdp_param.sharded_param), id(fsdp_param.unsharded_param))
            for fsdp_param in group.fsdp_params
        )
        for group in groups
    ]


def _sharded_parameter_ids(groups: list) -> list[tuple[int, ...]]:
    return [
        tuple(id(fsdp_param.sharded_param) for fsdp_param in group.fsdp_params)
        for group in groups
    ]


def _build_small_model(
    *,
    device_type: str,
    parallel_dims: ParallelDims,
    cache_enabled: bool,
) -> nn.Module:
    """Build three real GIFT MoE blocks on GFSDP with a size-one expert mesh."""
    model_config = load_model_spec(
        "gift_moe/debugmodel.yaml",
        {"config": {"layers": [{"block": "mla_moe"}] * _NUM_LAYERS}},
    ).model
    parallelism = ParallelismConfig(
        data_parallel_shard_degree=2,
        expert_parallel_degree=2,
        fsdp_backend="gfsdp2",
        enable_sequence_parallel=False,
        gfsdp_reuse_bf16_expert_weights=cache_enabled,
    )
    # This is the same model-specific configuration hook called by
    # GiftMoEModel.prepare_config before Module.parallelize in production.
    model_config.owner()._set_sharding_config(model_config, parallelism)
    # Trainer constructs models on meta, applies parallelism and GFSDP, then
    # calls to_empty and initializes the real weights. Match that order so
    # DTensor placement and GFSDP's move-to-mesh-device path are exercised.
    with torch.device("meta"):
        model = model_config.build()
    assert all(param.is_meta for param in model.parameters())
    # Pass deterministic floating hidden states so the input gradient is
    # observable without changing the model's actual block/parameter path.
    model.tok_embeddings = None
    model.parallelize(parallel_dims)
    _assert_expert_ep_sharding(model, parallel_dims)
    apply_ac(model, ActivationCheckpointConfig(mode="selective", early_stop=False))

    apply_gfsdp(
        model,
        parallel_dims.get_mesh("fsdp"),
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        pp_enabled=False,
        ep_degree=parallel_dims.ep,
        edp_mesh=parallel_dims.get_mesh("efsdp"),
        reuse_bf16_expert_weights=(
            parallelism.gfsdp_reuse_bf16_expert_weights
        ),
    )
    init_device = torch.device(device_type)
    model.to_empty(device=init_device)
    # Match Trainer's DTensor RNG initialization after model sharding. This
    # installs the tracker RaggedShardDTensor needs for rank-distinct draws.
    dtensor_random.manual_seed(_SEED, parallel_dims.world_mesh)
    backend = GFSDP2Backend()
    with torch.no_grad(), backend.weight_init_context(model):
        model.init_weights(buffer_device=init_device)
    model.train()
    model.lazy_init()
    model.setup_ema_buffer()
    model.update_ema_buffer_after_load(force_update=True)
    model.set_bf16_cache_debug(True)
    return model


def _assert_expert_ep_sharding(model: nn.Module, parallel_dims: ParallelDims) -> None:
    """Ensure the fixture's expert tensors are genuinely sharded over EP."""
    ep_mesh = parallel_dims.get_mesh("ep")
    assert ep_mesh.size() == 2
    assert parallel_dims.get_mesh("efsdp").size() == 1
    expert_params = list(model.layers["0"].moe.experts.parameters())
    assert expert_params
    for param in expert_params:
        assert isinstance(param, DTensor)
        assert param.device_mesh is ep_mesh
        assert param.placements == (Shard(0),)


def _input_for(device: torch.device, rank: int, update: int, microbatch: int):
    values = torch.arange(
        _SEQUENCE_LENGTH * _HIDDEN_SIZE, device=device, dtype=torch.float32
    ).reshape(1, _SEQUENCE_LENGTH, _HIDDEN_SIZE)
    values = (values.remainder(17) - 8) / 64
    values = values + rank / 32 + update / 16 + microbatch / 8
    return values.to(torch.bfloat16).detach().requires_grad_()


def _labels_for(device: torch.device, rank: int, update: int, microbatch: int):
    values = torch.arange(_SEQUENCE_LENGTH, device=device, dtype=torch.long)
    values = (values + 11 * rank + 7 * update + 3 * microbatch) % _VOCAB_SIZE
    return values.reshape(1, _SEQUENCE_LENGTH)


def _train_two_updates(model: nn.Module, groups: list, rank: int) -> dict[str, list]:
    optimizer = OptimizersContainer.Config(
        name="AdamW",
        lr=1e-3,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        weight_decay=0.1,
        implementation="fused",
    ).build(model_parts=[model])

    recompute_calls = [0] * _NUM_LAYERS
    for index, layer in enumerate(model.layers.values()):
        block = getattr(layer, "_checkpoint_wrapped_module", layer)

        def count_block_call(_module, _args, *, index=index):
            recompute_calls[index] += 1

        block.register_forward_pre_hook(count_block_call)

        layer_groups = tuple(layer._get_fsdp_state()._fsdp_param_groups)

        def check_weights_before_consumer(_module, _args, *, groups=layer_groups):
            for group in groups:
                _assert_fresh_bf16_cast(group)

        # GFSDP's prepended pre-hook unshards the group before this check runs.
        layer.register_forward_pre_hook(check_weights_before_consumer)

    results: dict[str, list] = {
        "losses": [],
        "input_grads": [],
        "weight_grads": [],
        "masters": [],
    }
    identities = _parameter_identities(groups)
    device = next(model.parameters()).device
    for update in range(_NUM_UPDATES):
        optimizer.zero_grad(set_to_none=True)
        for microbatch in range(_NUM_MICROBATCHES):
            inputs = _input_for(device, rank, update, microbatch)
            labels = _labels_for(device, rank, update, microbatch)
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.float().reshape(-1, _VOCAB_SIZE), labels.reshape(-1)
            ) / _NUM_MICROBATCHES
            loss.backward()
            results["losses"].append(loss.detach().cpu().clone())
            results["input_grads"].append(inputs.grad.detach().cpu().clone())
            for group in groups:
                _assert_fresh_bf16_cast(group)

        # Capture the actual reduced, accumulated expert gradients before AdamW.
        results["weight_grads"].append(_group_gradients(groups))
        optimizer.step()
        model.update_ema_params(alpha=0.25)
        assert _parameter_identities(groups) == identities
        results["masters"].append(_master_data(groups))

    results["recompute_calls"] = recompute_calls
    results["optimizer"] = optimizer
    return results


def _step_after_apply(
    model: nn.Module, groups: list, optimizer, rank: int
) -> dict[str, object]:
    """Exercise the existing optimizer against parameters after GFSDP _apply."""
    optimizer.zero_grad(set_to_none=True)
    inputs = _input_for(next(model.parameters()).device, rank, 8, 0)
    labels = _labels_for(inputs.device, rank, 8, 0)
    logits = model(inputs)
    loss = F.cross_entropy(
        logits.float().reshape(-1, _VOCAB_SIZE), labels.reshape(-1)
    )
    loss.backward()
    input_grad = inputs.grad.detach().cpu().clone()
    weight_grads = _group_gradients(groups)
    for group in groups:
        _assert_fresh_bf16_cast(group)
    optimizer.step()
    masters = _master_data(groups)
    with torch.no_grad():
        model(_input_for(inputs.device, rank, 9, 0))
    for group in groups:
        _assert_fresh_bf16_cast(group)
    return {
        "loss": loss.detach().cpu().clone(),
        "input_grad": input_grad,
        "weight_grads": weight_grads,
        "masters": masters,
    }


def _apply_on_auxiliary_stream_then_forward(model: nn.Module, inputs: torch.Tensor) -> None:
    """Queue a value-changing apply and consume it without a caller stream wait."""
    device = inputs.device
    current_stream = torch.cuda.current_stream(device)
    writer_stream = torch.cuda.Stream(device=device)
    writer_stream.wait_stream(current_stream)
    cuda_sleep = getattr(torch.cuda, "_sleep", None)

    with torch.cuda.stream(writer_stream):
        if callable(cuda_sleep):
            cuda_sleep(1_000_000)
        model.layers["0"]._apply(lambda tensor: tensor + 0.125)

    default_stream = torch.cuda.default_stream(device)
    with torch.cuda.stream(default_stream), torch.no_grad():
        model(inputs)


def _assert_exact_nested(test_case, expected, actual, path: str = "value") -> None:
    if isinstance(expected, torch.Tensor):
        test_case.assertIsInstance(actual, torch.Tensor, path)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=path)
        return
    test_case.assertEqual(type(actual), type(expected), path)
    if isinstance(expected, dict):
        test_case.assertEqual(actual.keys(), expected.keys(), path)
        for key in expected:
            _assert_exact_nested(test_case, expected[key], actual[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        test_case.assertEqual(len(actual), len(expected), path)
        for index, (want, got) in enumerate(zip(expected, actual, strict=True)):
            _assert_exact_nested(test_case, want, got, f"{path}[{index}]")
        return
    test_case.assertEqual(actual, expected, path)


class _ExtensionExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(
            torch.arange(16, dtype=torch.float32).reshape(4, 4)
        )


class _ExtensionMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = _ExtensionExperts()


class _ExtensionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.moe_enabled = True
        self.moe = _ExtensionMoE()

    def forward(self, x):
        return x @ self.moe.experts.weight


class _ExtensionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_embeddings = None
        self.layers = nn.ModuleDict({"0": _ExtensionBlock()})
        self.norm = None
        self.lm_head = None
        self.mtp = None
        self.enable_weight_tying = False

    def forward(self, x):
        return self.layers["0"](x)


class TestGFSDPBF16ExpertWeightCache(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    @skip_if_lt_x_gpu(2)
    @with_comms
    @_report_rank_exception
    def test_cache_matches_uncached_training_and_weight_lifecycles(self):
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=2,
            cp=1,
            tp=1,
            pp=1,
            ep=2,
            world_size=2,
        )
        parallel_dims.build_mesh()

        uncached = _build_small_model(
            device_type=self.device_type,
            parallel_dims=parallel_dims,
            cache_enabled=False,
        )
        cached = _build_small_model(
            device_type=self.device_type,
            parallel_dims=parallel_dims,
            cache_enabled=True,
        )
        uncached_groups = _expert_groups(uncached)
        cached_groups = _expert_groups(cached)
        for groups in (uncached_groups, cached_groups):
            for group in groups:
                for fsdp_param in group.fsdp_params:
                    self.assertTrue(fsdp_param.is_dtensor)
                    self.assertIs(
                        fsdp_param._tp_spec.mesh, parallel_dims.get_mesh("ep")
                    )
                    self.assertEqual(fsdp_param._tp_spec.placements, (Shard(0),))

        # The full-sized dense groups stay on the normal all-gather path.
        self.assertEqual(
            sum(bool(row["enabled"]) for row in _group_stats(uncached).values()), 0
        )
        for model, groups in ((uncached, uncached_groups), (cached, cached_groups)):
            for layer, expert_group in zip(
                model.layers.values(), groups, strict=True
            ):
                dense_group = layer._get_fsdp_state()._fsdp_param_groups[0]
                self.assertEqual(dense_group._all_gather_process_group.size(), 2)
                self.assertFalse(_stats_for(model, dense_group)["enabled"])
                self.assertNotEqual(
                    dense_group._all_gather_output[None].data_ptr(),
                    dense_group._all_gather_input[None].data_ptr(),
                )
                self.assertEqual(expert_group.orig_dtype, torch.float32)
                self.assertEqual(expert_group.param_dtype, torch.bfloat16)

        enabled_rows = [
            row for row in _group_stats(cached).values() if row["enabled"]
        ]
        self.assertEqual(len(enabled_rows), _NUM_LAYERS)
        for group in cached_groups:
            self.assertEqual(
                group._all_gather_output[None].data_ptr(),
                group._all_gather_input[None].data_ptr(),
            )
        # Layers 0 and 2 share the original alternating output pool, but each
        # keeps its own persistent BF16 input storage.
        self.assertIs(cached_groups[0].data_buffer_ctx, cached_groups[2].data_buffer_ctx)
        self.assertNotEqual(
            cached_groups[0]._all_gather_input[None].data_ptr(),
            cached_groups[2]._all_gather_input[None].data_ptr(),
        )
        for group in cached_groups:
            self.assertEqual(_stats_for(cached, group)["refreshes"], 0)

        for expected, actual in zip(
            _master_data(uncached_groups), _master_data(cached_groups), strict=True
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        uncached_result = _train_two_updates(
            uncached, uncached_groups, dist.get_rank()
        )
        cached_result = _train_two_updates(cached, cached_groups, dist.get_rank())
        for key in ("losses", "input_grads", "weight_grads", "masters"):
            _assert_exact_nested(self, uncached_result[key], cached_result[key], key)
        for calls in cached_result["recompute_calls"]:
            self.assertGreater(calls, _NUM_MICROBATCHES * _NUM_UPDATES)

        # Two optimizer updates invalidate once each. EMA shadow updates while
        # main weights are active do not invalidate the main-weight cache.
        for group in cached_groups:
            stats = _stats_for(cached, group)
            self.assertEqual(stats["generation"], _NUM_UPDATES)
            self.assertGreaterEqual(stats["invalidations"], _NUM_UPDATES)
            self.assertGreater(stats["hits"], 0)
            self.assertGreater(stats["direct_uses"], 0)

        # The first post-update consumer must prepare generation two exactly once.
        before_refreshes = [_stats_for(cached, group)["refreshes"] for group in cached_groups]
        with torch.no_grad():
            cached(_input_for(next(cached.parameters()).device, dist.get_rank(), 2, 0))
        for group, refreshes in zip(cached_groups, before_refreshes, strict=True):
            _assert_fresh_bf16_cast(group)
            self.assertEqual(_stats_for(cached, group)["refreshes"], refreshes + 1)

        # Save an actual GFSDP DCP checkpoint, then change one expert through
        # ordinary Module.load_state_dict and make DCP restore the saved values.
        checkpoint_masters = _master_data(cached_groups)
        adapter = GFSDPCheckpointAdapter([cached])
        checkpoint = adapter.model_save_state_dict()
        artifact_root = (
            Path(__file__).resolve().parents[3]
            / "runs"
            / "bf16-expert-cache"
            / "artifacts"
            / "raw"
        )
        artifact_root.mkdir(parents=True, exist_ok=True)
        path_holder = [
            str(artifact_root / f"focused-dcp-{uuid.uuid4().hex}")
            if dist.get_rank() == 0
            else None
        ]
        dist.broadcast_object_list(path_holder, src=0)
        checkpoint_path = path_holder[0]
        test_passed = False
        try:
            dcp.save(checkpoint, checkpoint_id=checkpoint_path)

            group_zero_before_load = (
                cached_groups[0]._padded_sharded_param_data[None].detach().clone()
            )
            state = cached.state_dict()
            changed_fqn = next(
                fqn for fqn in state if ".moe.experts." in fqn
            )
            changed_value = state[changed_fqn].clone() + 0.5
            load_result = cached.load_state_dict(
                {changed_fqn: changed_value}, strict=False
            )
            self.assertNotIn(changed_fqn, load_result.missing_keys)
            self.assertFalse(
                torch.equal(
                    group_zero_before_load,
                    cached_groups[0]._padded_sharded_param_data[None],
                )
            )
            with torch.no_grad():
                cached(_input_for(next(cached.parameters()).device, dist.get_rank(), 3, 0))
            _assert_fresh_bf16_cast(cached_groups[0])

            adapter.load(
                {},
                checkpoint_id=checkpoint_path,
                process_group=dist.group.WORLD,
                load_model=True,
            )
            for actual, expected in zip(
                _master_data(cached_groups), checkpoint_masters, strict=True
            ):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            with torch.no_grad():
                cached(_input_for(next(cached.parameters()).device, dist.get_rank(), 4, 0))
            for group in cached_groups:
                _assert_fresh_bf16_cast(group)

            # EMA weights use the same active-master buffer and must refresh on
            # both the swap into EMA and the restoration of main weights.
            main_masters = _master_data(cached_groups)
            cached.swap_ema_and_weights()
            self.assertTrue(
                any(
                    not torch.equal(active, main)
                    for active, main in zip(
                        _master_data(cached_groups), main_masters, strict=True
                    )
                )
            )
            with torch.no_grad():
                cached(_input_for(next(cached.parameters()).device, dist.get_rank(), 5, 0))
            for group in cached_groups:
                _assert_fresh_bf16_cast(group)

            cached.swap_ema_and_weights()
            for actual, expected in zip(
                _master_data(cached_groups), main_masters, strict=True
            ):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            with torch.no_grad():
                cached(_input_for(next(cached.parameters()).device, dist.get_rank(), 6, 0))
            for group in cached_groups:
                _assert_fresh_bf16_cast(group)

            # GFSDPModule._apply is a supported backing replacement path. Add a
            # same-dtype value change so stale-cache detection cannot depend on
            # a device/dtype conversion or Tensor._version.
            before_apply = _master_data([cached_groups[0]])[0]
            before_refresh = _stats_for(cached, cached_groups[0])["refreshes"]
            cached_ids = _sharded_parameter_ids(cached_groups)
            uncached_ids = _sharded_parameter_ids(uncached_groups)
            apply_input = _input_for(
                next(cached.parameters()).device, dist.get_rank(), 10, 0
            )
            for model in (cached, uncached):
                _apply_on_auxiliary_stream_then_forward(model, apply_input)
            self.assertEqual(_sharded_parameter_ids(cached_groups), cached_ids)
            self.assertEqual(_sharded_parameter_ids(uncached_groups), uncached_ids)
            self.assertFalse(
                torch.equal(before_apply, _master_data([cached_groups[0]])[0])
            )
            _assert_exact_nested(
                self,
                _all_group_masters(uncached),
                _all_group_masters(cached),
                "post_apply_all_masters_before_step",
            )
            cached_step = _step_after_apply(
                cached, cached_groups, cached_result["optimizer"], dist.get_rank()
            )
            uncached_step = _step_after_apply(
                uncached,
                uncached_groups,
                uncached_result["optimizer"],
                dist.get_rank(),
            )
            _assert_exact_nested(self, uncached_step, cached_step, "post_apply")
            self.assertGreater(
                _stats_for(cached, cached_groups[0])["refreshes"], before_refresh
            )
            test_passed = True
        finally:
            if test_passed:
                dist.barrier()
                if dist.get_rank() == 0:
                    shutil.rmtree(checkpoint_path, ignore_errors=True)
                dist.barrier()

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_all_gather_extension_keeps_original_output_buffer(self):
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=2,
            cp=1,
            tp=1,
            pp=1,
            ep=2,
            world_size=2,
        )
        parallel_dims.build_mesh()
        edp_mesh = parallel_dims.get_mesh("efsdp")
        dp_mesh = parallel_dims.get_mesh("fsdp")
        model = _ExtensionModel().to(self.device_type)
        apply_gfsdp(
            model,
            dp_mesh,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            pp_enabled=False,
            ep_degree=parallel_dims.ep,
            edp_mesh=edp_mesh,
            reuse_bf16_expert_weights=True,
        )
        expert_group = model.layers["0"]._get_fsdp_state()._fsdp_param_groups[0]
        extension_calls = {"pre": 0, "post": 0}
        seen_metadata = []
        for fsdp_param in expert_group.fsdp_params:
            local_tensor = fsdp_param._sharded_local_tensor

            def pre_all_gather(mesh, *, tensor=local_tensor):
                del mesh
                extension_calls["pre"] += 1
                return (tensor.detach(),), {"test_extension": True}

            def post_all_gather(
                inputs, metadata, dtype, out=None
            ):
                assert metadata is None or metadata == {"test_extension": True}
                extension_calls["post"] += 1
                seen_metadata.append(metadata)
                value = inputs[0].to(dtype=dtype)
                if out is None:
                    return value, metadata
                out.copy_(value)
                return out

            # The extension protocol belongs to the exact local tensor GFSDP
            # passes to its pre/post hooks. Install it there before lazy init,
            # since the small dense test model does not construct a production
            # float8 tensor subclass itself.
            local_tensor.fsdp_pre_all_gather = pre_all_gather
            local_tensor.fsdp_post_all_gather = post_all_gather
            fsdp_param._extensions_data = ExtensionsData()

        model.lazy_init()
        model.set_bf16_cache_debug(True)
        self.assertFalse(_stats_for(model, expert_group)["enabled"])
        self.assertNotEqual(
            expert_group._all_gather_output[None].data_ptr(),
            expert_group._all_gather_input[None].data_ptr(),
        )
        with torch.no_grad():
            model(
                torch.ones(
                    (1, 4, 4),
                    device=self.device_type,
                    dtype=torch.bfloat16,
                )
            )
        self.assertGreater(extension_calls["pre"], 0)
        self.assertGreater(extension_calls["post"], 0)
        self.assertIn({"test_extension": True}, seen_metadata)


def test_parallelism_config_defaults_bf16_expert_cache_off():
    assert ParallelismConfig().gfsdp_reuse_bf16_expert_weights is False


if __name__ == "__main__":
    import unittest

    unittest.main()
