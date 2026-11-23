#!/usr/bin/env python3
"""Run or compare the bounded native EP4 expert-owner numerical observation."""

import argparse
from contextlib import contextmanager
from contextvars import ContextVar
import importlib.util
import json
from math import prod
from pathlib import Path
import re
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SHARED = HERE.parent / "ep8.py"
spec = importlib.util.spec_from_file_location("_ep4_shared_numerics", SHARED)
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)

CONFIG_OVERRIDE = shared.CONFIG_OVERRIDE
STEPS = shared.STEPS
resolved_digest = shared.resolved_digest

RECEIPT = "ep4-owner-numerics-rank-{rank:02d}"
BASELINE, COMPACT = "baseline", "compact"
ROLES = (BASELINE, COMPACT)
FQNS = tuple(f"layers.{i}[efsdp]" for i in range(7, 0, -1))
PROJECTION_SHAPES = {
    "w13_E2FD": (32, 2, 1536, 4096),
    "w2_EDF": (32, 4096, 1536),
}
FULL_NUMEL = {f: prod(shape) for f, shape in PROJECTION_SHAPES.items()}
OWNER_ELEMENTS = sum(FULL_NUMEL.values()) // 2
CHUNK = OWNER_ELEMENTS // 2
TORCH_REVISION = "8b353065ce260336c5fdfa74d754e8bd9b91b062"
NCCL_REVISION = (2, 30, 7)
ALL_REQUIRED = shared.required_samples()


def require_source(expected, actual):
    if not valid_sha256(expected) or actual != expected:
        raise RuntimeError("Native GIFT source does not match the staged EP4 arm")


def valid_sha256(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def expected_pair(rank):
    first = rank % 4
    return [first, first + 4]


def owner_numel(rank, name):
    weight = name.rsplit(".", 1)[-1]
    # The flat slot packs W13 before W2. Each eFSDP owner gets one slot half.
    first, last = (rank // 4) * OWNER_ELEMENTS, (rank // 4 + 1) * OWNER_ELEMENTS
    start = 0 if weight == "w13_E2FD" else FULL_NUMEL["w13_E2FD"]
    return max(0, min(last, start + FULL_NUMEL[weight]) - max(first, start))


def required_for_rank(rank):
    empty = {name for name in shared.NAMES if owner_numel(rank, name) == 0}
    return {field for field in ALL_REQUIRED if field.partition("/")[2] not in empty}


def empty_fp32(value):
    import torch

    if value is None:
        return "absent"
    local = shared.local_tensor(value)
    if local.numel() or local.dtype != torch.float32:
        raise RuntimeError("A zero-owned EP4 parameter has unexpected local data")
    return "empty_fp32"


def validate_recipe(document):
    parallel = document["parallelism"]
    compile_config = document["compile"]
    training = document["training"]
    debug = document["debug"]
    if (
        parallel["expert_parallel_degree"] != 4
        or parallel["data_parallel_replicate_degree"] != 1
        or parallel["data_parallel_shard_degree"] != 8
        or any(
            parallel[key] != 1
            for key in (
                "tensor_parallel_degree",
                "pipeline_parallel_degree",
                "context_parallel_degree",
            )
        )
        or parallel["fsdp_backend"] != "gfsdp2"
        or parallel["enable_fsdp_symm_mem"]
        or not all(parallel.get(field) is True for field in shared.OPTIMIZATION_FIELDS)
        or (
            training["steps"],
            training["local_batch_size"],
            training["global_batch_size"],
            training["seq_len"],
        )
        != (1, 2, 512, 4096)
        or (training["mixed_precision_param"], training["mixed_precision_reduce"])
        != ("bfloat16", "float32")
        or training["enable_cpu_offload"]
        or not compile_config["enable"]
        or set(compile_config["components"]) != {"model", "loss"}
        or compile_config["backend"] != "inductor"
        or document["activation_checkpoint"]["mode"] != "selective"
        or not debug["deterministic"]
        or debug["deterministic_warn_only"]
        or debug["seed"] != 2048
    ):
        raise RuntimeError("Expected the pinned deterministic EP4 compiled/SAC recipe")


def validate_trainer(trainer):
    import torch

    config = trainer.config
    if (
        trainer.step != STEPS
        or trainer.gradient_accumulation_steps != shared.MICROBATCHES
        or len(trainer.model_parts) != 1
        or config.parallelism.expert_parallel_degree != 4
        or not config.debug.deterministic
        or config.debug.deterministic_warn_only
        or config.debug.seed != 2048
        or not torch.are_deterministic_algorithms_enabled()
        or torch.is_deterministic_algorithms_warn_only_enabled()
    ):
        raise RuntimeError(
            "Expected one strict deterministic EP4 update with accumulation 32"
        )


def full_indices(numel, device):
    import torch

    index = torch.arange(shared.SAMPLES, dtype=torch.int64)
    stratified = index * numel // shared.SAMPLES + (index * 4051 + 55) % (
        numel // shared.SAMPLES
    )
    return stratified.to(device)


def sample_full(tensor, shape, positions, compact):
    import torch

    expected_dtype = torch.bfloat16 if compact else torch.float32
    if (
        tensor.dtype != expected_dtype
        or tensor.shape != shape
        or not tensor.is_contiguous()
    ):
        raise RuntimeError(
            "Native full produced projection has the wrong shape, dtype or layout"
        )
    # index_select copies before the slot may be reused. Widen the selected BF16
    # values exactly, without converting the full shared slot.
    return tensor.view(-1).index_select(0, positions).to(torch.float32)


class Observation(shared.Observation):
    def __init__(self, output, requested, expected_config, role):
        super().__init__(output, requested, expected_config)
        self.compact = role == COMPACT
        self.role = role
        self.full_sample_indices = {}
        self.routed_parameters = {}
        self.reductions = {}
        self.expert_context = ContextVar("ep4_expert_reduce", default=None)
        self.owned_names = tuple(
            name for name in shared.NAMES if owner_numel(self.rank, name)
        )
        self.empty_owner = {
            name: {} for name in shared.NAMES if name not in self.owned_names
        }

    def validate_resolved(self, loaded):
        from torchtitan.config.save import get_typed_document

        super().validate_resolved(loaded)
        validate_recipe(
            get_typed_document(loaded.trainer_config, loaded.source_document["model"])
        )

    def start_update(self, trainer):
        import torch

        with patch.object(shared, "NAMES", self.owned_names):
            super().start_update(trainer)
        if (
            torch.version.git_version != TORCH_REVISION
            or torch.cuda.nccl.version() != NCCL_REVISION
            or torch.cuda.get_device_capability(trainer.device) != (9, 0)
            or "H100" not in torch.cuda.get_device_name(trainer.device)
            or torch.distributed.get_world_size() != shared.PROCESSES
            or torch.distributed.get_rank() != self.rank
        ):
            raise RuntimeError(
                "Native EP4 owner diagnostic requires the pinned H100/NCCL runtime"
            )
        (model,) = trainer.model_parts
        for layer in range(1, 8):
            experts = model.layers[str(layer)].moe.experts
            for weight in shared.WEIGHTS:
                p = getattr(experts, weight)
                if id(p) in self.routed_parameters:
                    raise RuntimeError("EP4 routed expert parameter is aliased")
                self.routed_parameters[id(p)] = f"layers.{layer}[efsdp]", weight
        for name in shared.NAMES:
            _, layer, _, _, weight = name.split(".")
            parameter = getattr(model.layers[layer].moe.experts, weight)
            local = shared.local_tensor(parameter)
            if local.numel() != owner_numel(self.rank, name):
                raise RuntimeError(
                    f"Wrong EP4 owner-local projection length for {name}"
                )
            if name in self.empty_owner:
                optimizers = [
                    optimizer
                    for optimizer in trainer.optimizers.optimizers
                    for group in optimizer.param_groups
                    if any(param is parameter for param in group["params"])
                ]
                (optimizer,) = optimizers
                self.owners[name] = optimizer, parameter
                self.names_by_parameter[id(parameter)] = name
                self.indices[name] = (0, None)
                self.empty_owner[name]["parameter"] = empty_fp32(parameter)
            self.full_sample_indices[name] = (
                PROJECTION_SHAPES[weight],
                full_indices(FULL_NUMEL[weight], local.device),
            )

    def gradient_shards(self, phase):
        for name, record in self.empty_owner.items():
            _, parameter = self.owners[name]
            record[phase] = empty_fp32(parameter.grad)
        with patch.object(shared, "NAMES", self.owned_names):
            super().gradient_shards(phase)

    def post_optimizer(self, trainer):
        for name, record in self.empty_owner.items():
            optimizer, parameter = self.owners[name]
            record["updated_parameter"] = empty_fp32(parameter)
            state = optimizer.state.get(parameter, {})
            if not state:
                record["adam"] = "absent"
                continue
            if any(
                empty_fp32(state[moment]) != "empty_fp32"
                for moment in ("exp_avg", "exp_avg_sq")
            ):
                raise RuntimeError(f"Missing empty optimizer moments for {name}")
            record["adam"] = "empty_moments"
            self.sample(f"optimizer_step/{name}", state["step"])
        with patch.object(shared, "NAMES", self.owned_names):
            super().post_optimizer(trainer)

    def before_reduce(self, params, group):
        for param in params:
            name = self.names_by_parameter.get(id(param.sharded_param))
            if name is None:
                continue
            if self.microbatch is None or group.size() != 2:
                raise RuntimeError(
                    "Expected a full EP4 producer gradient inside a native microbatch"
                )
            shape, positions = self.full_sample_indices[name]
            produced = param._unsharded_accumulated_grad
            if not produced.is_cuda or produced.device != positions.device:
                raise RuntimeError(
                    "EP4 pre-reduction capture must read the active CUDA slot"
                )
            value = sample_full(produced, shape, positions, self.compact)
            self.sample(f"pre_reduce/{self.microbatch:02d}/{name}", value)

    def reduce(
        self, native, params, active_params, output, input, group, *args, **kwargs
    ):
        import torch

        routed = [self.routed_parameters.get(id(p.sharded_param)) for p in params]
        if not any(routed):
            return native(params, active_params, output, input, group, *args, **kwargs)
        owners = {entry[0] for entry in routed if entry}
        weights = {entry[1] for entry in routed if entry}
        seq = self.reductions.setdefault(self.microbatch, [])
        expected_dtype = torch.bfloat16 if self.compact else torch.float32
        transport = kwargs.get("owner_transport")
        if (
            self.microbatch is None
            or len(routed) != 2
            or len(owners) != 1
            or weights != set(shared.WEIGHTS)
            or tuple(entry[1] for entry in routed if entry) != shared.WEIGHTS
            or params != active_params
            or len(seq) >= len(FQNS)
            or owners != {FQNS[len(seq)]}
            or group.size() != 2
            or torch.distributed.get_backend(group) != "nccl"
            or torch.distributed.get_process_group_ranks(group)
            != expected_pair(self.rank)
            or group.rank() != self.rank // 4
            or input.shape != (2 * OWNER_ELEMENTS,)
            or input.dtype != expected_dtype
            or output.shape != (OWNER_ELEMENTS,)
            or output.dtype != torch.float32
            or not input.is_cuda
            or output.device != input.device
            or not (output.is_contiguous() and input.is_contiguous())
            or self.compact != (transport is not None)
            or (self.compact and kwargs.get("owner_fqn") not in owners)
        ):
            raise RuntimeError(
                "EP4 expert reduction did not reach the expected source, mesh or layout"
            )
        if (
            self.compact
            and input.untyped_storage().nbytes() != input.numel() * input.element_size()
        ):
            raise RuntimeError(
                "Compact BF16 gradient slot uses an unexpected backing allocation"
            )
        calls = []
        token = self.expert_context.set((group, input, calls))
        try:
            event = native(params, active_params, output, input, group, *args, **kwargs)
        finally:
            self.expert_context.reset(token)
        if calls != ([COMPACT, COMPACT] if self.compact else [BASELINE]):
            raise RuntimeError(
                f"Wrong native EP4 expert collective sequence at {sorted(owners)}: {calls}"
            )
        seq.append(FQNS[len(seq)])
        return event

    def collective(self, name, native, output, input, *args, **kwargs):
        import torch

        active = self.expert_context.get()
        if active is not None:
            group, slot, calls = active
            if kwargs.get("group") is not group:
                raise RuntimeError(
                    "Expert gradient collective used a different process group"
                )
            if name == COMPACT:
                splits = [0, CHUNK] if group.rank() == 0 else [CHUNK, 0]
                peer = (1 - group.rank()) * OWNER_ELEMENTS
                offset = (peer + len(calls) * CHUNK) * slot.element_size()
                if (
                    len(calls) >= 2
                    or input.data_ptr() != slot.data_ptr() + offset
                    or input.shape != (CHUNK,)
                    or output.shape != (CHUNK,)
                    or input.dtype != torch.bfloat16
                    or output.dtype != torch.bfloat16
                    or output.untyped_storage().nbytes()
                    != CHUNK * output.element_size()
                    or output.untyped_storage().data_ptr()
                    == slot.untyped_storage().data_ptr()
                    or kwargs.get("input_split_sizes") != splits
                    or kwargs.get("output_split_sizes") != splits
                    or kwargs.get("async_op") is not False
                ):
                    raise RuntimeError("Unexpected physical BF16 owner exchange")
            elif (
                calls
                or input is not slot
                or output.shape != (OWNER_ELEMENTS,)
                or input.dtype != torch.float32
                or output.dtype != torch.float32
            ):
                raise RuntimeError(
                    "Unexpected physical native FP32 expert reduce-scatter"
                )
            calls.append(name)
        return native(output, input, *args, **kwargs)

    def finish_update(self, trainer):
        expected = {i: list(FQNS) for i in range(shared.MICROBATCHES)}
        if self.reductions != expected:
            raise RuntimeError(
                "EP4 did not complete seven expert reductions in each of 32 microbatches"
            )
        self.final["ep4_owner_transport"] = {
            "role": self.role,
            "pair": expected_pair(self.rank),
            "order": list(FQNS),
            "microbatches": len(self.reductions),
            "expert_reductions": len(FQNS) * len(self.reductions),
            "fp32_reduce_scatter_calls": 0
            if self.compact
            else len(FQNS) * len(self.reductions),
            "bf16_total_exchanges": 2 * len(FQNS) * len(self.reductions)
            if self.compact
            else 0,
            "packed_slot_dtype": "bfloat16" if self.compact else "float32",
            "packed_slot_elements": 2 * OWNER_ELEMENTS,
            "owner_local_elements": OWNER_ELEMENTS,
            "full_sample_numel": {
                name: prod(shape)
                for name, (shape, _) in self.full_sample_indices.items()
            },
        }
        self.final["empty_owner"] = self.empty_owner
        required = required_for_rank(self.rank)
        with (
            patch.object(shared, "NAMES", self.owned_names),
            patch.object(shared, "required_samples", lambda: required),
        ):
            super().finish_update(trainer)


@contextmanager
def observe(adapter, requested, expected_config, role):
    import torch.distributed as dist
    from torchtitan.distributed.gfsdp import _param_group

    if role not in ROLES or sorted(requested) != shared.OPTIMIZATIONS:
        raise RuntimeError(
            "EP4 owner diagnostic requires a source role and all four Expert GFSDP flags"
        )
    observation = Observation(
        adapter.options["output_dir"], requested, expected_config, role
    )
    native = _param_group.reduce_scatter
    native_rs = dist.reduce_scatter_tensor
    native_a2a = dist.all_to_all_single

    def reduce(*args, **kwargs):
        return observation.reduce(native, *args, **kwargs)

    def rs(*args, **kwargs):
        return observation.collective(BASELINE, native_rs, *args, **kwargs)

    def a2a(*args, **kwargs):
        return observation.collective(COMPACT, native_a2a, *args, **kwargs)

    with (
        patch.object(shared, "Observation", lambda *_: observation),
        patch.object(shared, "validate_trainer", validate_trainer),
        patch.object(shared, "RECEIPT", RECEIPT),
        patch.object(_param_group, "reduce_scatter", reduce),
        patch.object(dist, "reduce_scatter_tensor", rs),
        patch.object(dist, "all_to_all_single", a2a),
        shared.observe(adapter, requested, expected_config),
    ):
        yield


def validate_capture(path, role, source):
    import yaml

    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text())
    config = yaml.safe_load((path / "native/config/resolved_config.yaml").read_text())
    validate_recipe(config)
    require_source(
        source, metadata["source_snapshot_before"]["tracked_and_untracked_sha256"]
    )
    hashes = metadata["config_hashes_before"]
    if metadata["profiler"] != "disabled" or not all(
        valid_sha256(hashes.get(key))
        for key in (
            "expert_gfsdp_numerics_sha256",
            "expert_gfsdp_numerics_shared_sha256",
            "direct_dw_build_manifest_sha256",
        )
    ):
        raise RuntimeError(
            "Native EP4 capture is missing the diagnostic recipe or helper identity"
        )
    expected_helpers = (
        hashes["expert_gfsdp_numerics_shared_sha256"],
        hashes["direct_dw_build_manifest_sha256"],
    )
    minimum = required_for_rank(0)
    with (
        patch.object(shared, "RECEIPT", RECEIPT),
        patch.object(shared, "required_samples", lambda: minimum),
    ):
        for receipt, values in shared.load_run(path):
            rank = receipt["rank"]
            route = receipt["final_state"]["ep4_owner_transport"]
            expected_count = len(FQNS) * shared.MICROBATCHES
            if (
                receipt["requested"] != shared.OPTIMIZATIONS
                or route["role"] != role
                or route["pair"] != expected_pair(rank)
                or route["order"] != list(FQNS)
                or route["microbatches"] != shared.MICROBATCHES
                or route["expert_reductions"] != expected_count
                or route["fp32_reduce_scatter_calls"]
                != (expected_count if role == BASELINE else 0)
                or route["bf16_total_exchanges"]
                != (0 if role == BASELINE else 2 * expected_count)
                or route["packed_slot_dtype"]
                != ("float32" if role == BASELINE else "bfloat16")
                or route["packed_slot_elements"] != 2 * OWNER_ELEMENTS
                or route["owner_local_elements"] != OWNER_ELEMENTS
                or route["full_sample_numel"]
                != {name: FULL_NUMEL[name.rsplit(".", 1)[1]] for name in shared.NAMES}
                or receipt["selected_local_numel"]
                != {name: owner_numel(rank, name) for name in shared.NAMES}
            ):
                raise RuntimeError(
                    f"Invalid EP4 physical owner transport receipt at rank {rank}"
                )
            validate_samples(rank, receipt["final_state"]["empty_owner"], values)
            yield {**receipt, "shared_helper_and_build": expected_helpers}, values


def validate_samples(rank, empty_owner, values):
    required = required_for_rank(rank)
    empty = {name for name in shared.NAMES if owner_numel(rank, name) == 0}
    if set(empty_owner) != empty or not required.issubset(values):
        raise RuntimeError(
            f"EP4 rank {rank} has incomplete owner and producer coverage"
        )
    for name, record in empty_owner.items():
        if (
            record["parameter"] != "empty_fp32"
            or record["updated_parameter"] != "empty_fp32"
            or any(
                record[p] not in ("absent", "empty_fp32")
                for p in ("pre_clip", "post_clip")
            )
            or record["adam"] not in ("absent", "empty_moments")
        ):
            raise RuntimeError(f"Invalid zero-owned EP4 state for {name}, rank {rank}")
        observed = {key for key in values if key.partition("/")[2] == name}
        step = f"optimizer_step/{name}"
        expected = {step} if record["adam"] == "empty_moments" else set()
        if observed != expected or (
            expected and (values[step].shape != () or values[step].item() != 1)
        ):
            raise RuntimeError(
                f"Invalid zero-owned EP4 Adam counter for {name}, rank {rank}"
            )


def compare(left, right):
    rows = []
    for (a, x), (b, y) in zip(left, right, strict=True):
        rank = a["rank"]
        for key in (
            "rank",
            "recipe",
            "input_provenance",
            "helper",
            "shared_helper_and_build",
            "compiler_cache_policy",
        ):
            if a[key] != b[key]:
                raise RuntimeError(
                    f"EP4 diagnostic provenance differs at {key}, rank {rank}"
                )
        if a["initial_state"]["sha256"] != b["initial_state"]["sha256"]:
            raise RuntimeError(f"Initial state differs at EP4 rank {rank}")
        for key in ("data_sha256", "scheduler", "empty_owner"):
            if a["final_state"][key] != b["final_state"][key]:
                raise RuntimeError(
                    f"EP4 input or scheduler state differs at {key}, rank {rank}"
                )
        if x.keys() != y.keys():
            raise RuntimeError(f"EP4 sample coverage differs at rank {rank}")
        for key in (
            "global_valid_tokens",
            "tokens_seen",
            *(field for field in x if field.startswith("optimizer_step/")),
        ):
            if x[key].tobytes() != y[key].tobytes():
                raise RuntimeError(f"EP4 exact counter differs at {key}, rank {rank}")
        rows.extend(
            {"rank": rank, "field": field, **shared.metrics(x[field], y[field])}
            for field in sorted(x)
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("baseline_repeat", type=Path)
    parser.add_argument("compact", type=Path)
    parser.add_argument("--baseline-source-sha256", required=True)
    parser.add_argument("--compact-source-sha256", required=True)
    args = parser.parse_args()
    paths = [
        p.resolve(strict=True)
        for p in (args.baseline, args.baseline_repeat, args.compact)
    ]
    if (
        len(set(paths)) != 3
        or args.baseline_source_sha256 == args.compact_source_sha256
    ):
        parser.error(
            "Expected three distinct native attempts from two staged GIFT sources"
        )
    a, a2, b = [
        list(validate_capture(path, role, source))
        for path, role, source in zip(
            paths,
            (BASELINE, BASELINE, COMPACT),
            (
                args.baseline_source_sha256,
                args.baseline_source_sha256,
                args.compact_source_sha256,
            ),
            strict=True,
        )
    ]
    pairs = {
        "a1_vs_a2": compare(a, a2),
        "a1_vs_b": compare(a, b),
        "a2_vs_b": compare(a2, b),
    }
    exact = all(row["exact"] for pair in pairs.values() for row in pair)
    print(
        json.dumps(
            {
                "status": "EXACT_ON_CAPTURED_FIELDS"
                if exact
                else "INVESTIGATE_DIFFERENCES",
                "source": {
                    "baseline": args.baseline_source_sha256,
                    "compact": args.compact_source_sha256,
                },
                "capture_paths": [str(p) for p in paths],
                "pairs": pairs,
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0 if exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
