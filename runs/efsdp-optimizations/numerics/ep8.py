"""Observe one untimed original-recipe EP8 update without exporting model tensors."""

import argparse
from copy import deepcopy
import hashlib
import json
import os
import random
import struct
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

STEPS = 1
PROCESSES = 8
MICROBATCHES = 32
LAYERS = (1, 4, 7)
WEIGHTS = ("w13_E2FD", "w2_EDF")
NAMES = tuple(
    f"layers.{layer}.moe.experts.{weight}" for layer in LAYERS for weight in WEIGHTS
)
SAMPLES = 4096
CHUNK_BYTES = 16 * 1024**2
INITIAL_BYTES = 16 * 1024**3
PAYLOAD_BYTES = 4 * 1024**2
ORIGINAL_REVISION = "e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f"
OPTIMIZATIONS = ["bf16-cache", "direct-dw", "fp8-cache", "tail-retention"]
OPTIMIZATION_FIELDS = (
    "gfsdp_reuse_bf16_expert_weights",
    "gfsdp_fp8_expert_weights",
    "gfsdp_direct_expert_dw",
    "gfsdp_retain_tail_weights",
)
INPUT_FIELDS = (
    "dataset_sha256",
    "mds_index_sha256",
    "row_manifest_sha256",
    "tokenizer_file_sha256",
)
RECEIPT = "ep8-numerics-rank-{rank:02d}"
CONFIG_OVERRIDE = ("--debug.deterministic", "true")


def json_bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def checksum(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def resolved_digest(document):
    return checksum({k: v for k, v in document.items() if k != "dump_folder"})


def recipe_digest(document):
    normalized = deepcopy(document)
    parallelism = normalized["parallelism"]
    for field in OPTIMIZATION_FIELDS:
        parallelism.pop(field, None)
    for stream in normalized["dataloader"]["dataset"]["streams"]:
        stream.pop("local", None)
    return resolved_digest(normalized)


def raw_view(tensor):
    import torch

    return memoryview(tensor.reshape(-1).view(torch.uint8).numpy())


def chunks(tensor):
    """Traverse logical row-major bytes with a bound on each host copy."""
    limit = CHUNK_BYTES // tensor.element_size()
    if tensor.is_contiguous():
        yield from tensor.view(-1).split(limit)
    elif tensor.numel() <= limit:
        yield tensor
    else:
        for outer in tensor.unbind():
            yield from chunks(outer)


def local_tensor(value):
    import torch

    # RaggedShardDTensor.to_local() is only available with autograd off.
    with torch.no_grad():
        local = value.to_local() if hasattr(value, "to_local") else value
        return local.detach()


def tensor_metadata(value, local):
    fields = {
        "dtype": str(local.dtype),
        "global_shape": list(value.shape),
        "local_shape": list(local.shape),
        "local_stride": list(local.stride()),
    }
    if hasattr(value, "device_mesh"):
        mesh = value.device_mesh
        coordinate = mesh.get_coordinate()
        fields["mesh"] = {
            "shape": list(mesh.shape),
            "names": list(mesh.mesh_dim_names or ()),
            "coordinate": list(coordinate) if coordinate is not None else None,
        }
        fields["placements"] = [
            {
                "placement": str(placement),
                **{
                    name: getattr(placement, name)
                    for name in ("local_numel", "global_offset", "shard_numels")
                    if hasattr(placement, name)
                },
            }
            for placement in value.placements
        ]
    return fields


class StateDigest:
    def __init__(self):
        self.bytes_read = 0

    def tensor(self, value):
        import torch

        with torch.no_grad():
            local = local_tensor(value)
            num_bytes = local.numel() * local.element_size()
            self.bytes_read += num_bytes
            if self.bytes_read > INITIAL_BYTES:
                raise RuntimeError("initial state exceeds the 16 GiB/rank read budget")
            digest = hashlib.sha256()
            for chunk in chunks(local):
                cpu = chunk.to(device="cpu", memory_format=torch.contiguous_format)
                digest.update(raw_view(cpu.contiguous()))
            return {**tensor_metadata(value, local), "sha256": digest.hexdigest()}


def optimizer_inventory(trainer, names):
    owners = {}
    groups = []
    for optimizer in trainer.optimizers.optimizers:
        for group in optimizer.param_groups:
            parameters = []
            for param in group["params"]:
                aliases = names[id(param)]
                name = aliases[0]
                if name in owners:
                    raise RuntimeError(f"optimizer owns {name} more than once")
                owners[name] = (optimizer, param)
                parameters.append(aliases)
            groups.append(
                {
                    "type": type(optimizer).__name__,
                    "parameters": parameters,
                    "options": {
                        key: value for key, value in group.items() if key != "params"
                    },
                }
            )
    return owners, groups


def rng_state(digest):
    import numpy as np
    import torch

    name, words, offset, has_gauss, gauss = np.random.get_state()
    numpy_hash = hashlib.sha256(name.encode())
    numpy_hash.update(memoryview(words).cast("B"))
    numpy_hash.update(struct.pack("=q?d", offset, bool(has_gauss), gauss))
    return {
        "python": checksum(random.getstate()),
        "numpy": numpy_hash.hexdigest(),
        "torch_cpu": digest.tensor(torch.get_rng_state()),
        "torch_cuda": digest.tensor(torch.cuda.get_rng_state()),
    }


def initial_state(trainer):
    import torch

    (model,) = trainer.model_parts
    named = {}
    unique = {}
    for name, tensor in sorted(model.named_parameters(remove_duplicate=False)):
        named.setdefault(id(tensor), []).append(name)
        unique[id(tensor)] = tensor
    owners, groups = optimizer_inventory(trainer, named)
    digest = StateDigest()
    state = {
        "parameters": {
            names[0]: {"names": names, "value": digest.tensor(unique[key])}
            for key, names in named.items()
        },
        "buffers": {
            name: digest.tensor(tensor)
            for name, tensor in sorted(model.named_buffers())
        },
        "optimizer_groups": groups,
        "optimizer_state": {
            name: {
                key: digest.tensor(value) if torch.is_tensor(value) else value
                for key, value in optimizer.state.get(param, {}).items()
            }
            for name, (optimizer, param) in owners.items()
            if optimizer.state.get(param)
        },
        "scheduler": [
            scheduler.state_dict() for scheduler in trainer.lr_schedulers.schedulers
        ],
        "tokens_seen": digest.tensor(trainer.ntokens_seen),
        "rng": rng_state(digest),
    }
    return {"sha256": checksum(state), "bytes_read": digest.bytes_read, **state}, owners


def validate_trainer(trainer):
    import torch

    config = trainer.config
    if (
        trainer.step != STEPS
        or trainer.gradient_accumulation_steps != MICROBATCHES
        or len(trainer.model_parts) != 1
        or config.parallelism.expert_parallel_degree != PROCESSES
        or not config.debug.deterministic
        or config.debug.deterministic_warn_only
        or config.debug.seed != 2048
        or not torch.are_deterministic_algorithms_enabled()
        or torch.is_deterministic_algorithms_warn_only_enabled()
    ):
        raise RuntimeError(
            "parity requires strict deterministic native EP8 update 1 and accumulation 32"
        )


class Observation:
    def __init__(self, output, requested, expected_config):
        self.output = Path(output)
        self.rank = int(os.environ["RANK"])
        self.requested = sorted(requested)
        self.expected_config = expected_config
        self.config_validated = False
        self.update_started = False
        self.finished = False
        self.inputs = []
        self.microbatch = None
        self.losses = []
        self.initial = None
        self.streams = set()
        self.samples = {}
        self.indices = {}
        self.names_by_parameter = {}
        self.owners = {}
        self.final = {}

    def validate_resolved(self, loaded):
        from torchtitan.config.save import get_typed_document

        config = loaded.trainer_config
        typed = get_typed_document(config, loaded.source_document["model"])
        if (
            resolved_digest(typed) != self.expected_config
            or Path(config.dump_folder).resolve()
            != (self.output.parent / "native").resolve()
        ):
            raise RuntimeError(
                "effective EP8 configuration differs from the verified diagnostic recipe"
            )
        self.config_validated = True

    def sample(self, key, tensor, indices=None):
        import torch

        if key in self.samples:
            raise RuntimeError(f"duplicate EP8 numerical field {key}")
        with torch.no_grad():
            local = local_tensor(tensor)
            if indices is None:
                if local.numel() > SAMPLES:
                    raise RuntimeError(
                        f"scalar or router sample exceeds the per-field limit: {key}"
                    )
                if local.is_cuda:
                    self.streams.add(torch.cuda.current_stream(local.device))
                self.samples[key] = local.clone()
                return
            count, positions = indices
            if (
                not local.is_cuda
                or local.dtype != torch.float32
                or local.numel() != count
                or not local.is_contiguous()
            ):
                raise RuntimeError(
                    f"unexpected FP32 local shape/layout: {key} ({tuple(local.shape)})"
                )
            self.streams.add(torch.cuda.current_stream(local.device))
            self.samples[key] = local.view(-1).index_select(0, positions)

    def start_update(self, trainer):
        import torch

        if not self.config_validated:
            raise RuntimeError(
                "the EP8 configuration must pass validation before training"
            )
        validate_trainer(trainer)
        if self.update_started:
            raise RuntimeError("only one optimizer update is allowed")
        self.update_started = True
        self.initial, all_owners = initial_state(trainer)
        owners_by_parameter = {
            id(param): optimizer for optimizer, param in all_owners.values()
        }
        (model,) = trainer.model_parts
        for name in NAMES:
            _, layer, _, _, weight = name.split(".")
            experts = model.layers[layer].moe.experts
            parameter = getattr(experts, weight)
            param_id = id(parameter)
            if param_id in self.names_by_parameter:
                raise RuntimeError(f"selected expert parameter is aliased: {name}")
            self.owners[name] = (owners_by_parameter[param_id], parameter)
            self.names_by_parameter[param_id] = name
            local = local_tensor(parameter)
            count = local.numel()
            if count < SAMPLES:
                raise RuntimeError(f"insufficient local parameter samples for {name}")
            index = torch.arange(SAMPLES, dtype=torch.int64)
            # Sample all strata with distinct offsets within each stratum.
            positions = index * count // SAMPLES + (index * 4051 + 55) % (
                count // SAMPLES
            )
            self.indices[name] = (count, positions.to(local.device))
            self.sample(f"initial/{name}", parameter, self.indices[name])

    def before_reduce(self, params, group):
        for param in params:
            name = self.names_by_parameter.get(id(param.sharded_param))
            if name is None:
                continue
            if group.size() != 1 or self.microbatch is None:
                raise RuntimeError(
                    "expected local eFSDP=1 reduction inside a numbered microbatch"
                )
            # The native reduce stream already waits for this calling stream.
            # Copy now because subsequent reduction and owner reuse mutate the arena.
            self.sample(
                f"pre_reduce/{self.microbatch:02d}/{name}",
                param._unsharded_accumulated_grad,
                self.indices[name],
            )

    def gradient_shards(self, phase):
        for name in NAMES:
            _, parameter = self.owners[name]
            if parameter.grad is None:
                raise RuntimeError(
                    f"{phase} has no optimizer-visible gradient for {name}"
                )
            self.sample(f"{phase}/{name}", parameter.grad, self.indices[name])

    def router_state(self, trainer, phase):
        (model,) = trainer.model_parts
        for layer in LAYERS:
            moe = model.layers[str(layer)].moe
            for field in ("tokens_per_expert_E", "expert_bias_E"):
                value = getattr(moe, field)
                if value is not None:
                    self.sample(f"{phase}/layers.{layer}.moe.{field}", value)

    def post_optimizer(self, trainer):
        self.router_state(trainer, "post_optimizer")
        for name in NAMES:
            optimizer, parameter = self.owners[name]
            self.sample(f"parameter/{name}", parameter, self.indices[name])
            state = optimizer.state.get(parameter, {})
            self.sample(f"optimizer_step/{name}", state["step"])
            for moment in ("exp_avg", "exp_avg_sq"):
                self.sample(f"{moment}/{name}", state[moment], self.indices[name])
        self.final["scheduler"] = [
            scheduler.state_dict() for scheduler in trainer.lr_schedulers.schedulers
        ]

    def record_inputs(self, state):
        import torch

        digest = StateDigest()
        fields = dict(state.input_dict)
        if "labels" in fields:
            raise RuntimeError("native input unexpectedly includes a labels field")
        fields["labels"] = state.labels
        if any(
            not torch.is_tensor(value) or value.device.type != "cpu"
            for value in fields.values()
        ):
            raise RuntimeError("after_dataloader capture requires native CPU tensors")
        self.inputs.append(
            {name: digest.tensor(value) for name, value in sorted(fields.items())}
        )

    def finish_update(self, trainer):
        import numpy as np
        import torch

        if len(self.inputs) != MICROBATCHES or len(self.losses) != MICROBATCHES:
            raise RuntimeError(
                "parity did not observe all native accumulation microbatches"
            )
        self.sample("tokens_seen", trainer.ntokens_seen)
        self.sample("loss_per_microbatch", torch.stack(self.losses))
        missing = required_samples().difference(self.samples)
        if missing:
            raise RuntimeError(f"missing native observations: {sorted(missing)}")

        total = sum(
            value.numel() * value.element_size() for value in self.samples.values()
        )
        if total > PAYLOAD_BYTES:
            raise RuntimeError(f"bounded EP8 capture exceeded {PAYLOAD_BYTES} bytes")
        # D2H goes to a bounded pinned buffer only after the whole update.
        stream = torch.cuda.current_stream(trainer.device)
        for producer in self.streams:
            if producer != stream:
                stream.wait_stream(producer)
        hosts = {}
        with torch.cuda.stream(stream):
            for key, value in self.samples.items():
                if value.is_cuda:
                    pinned = torch.empty_like(value, device="cpu", pin_memory=True)
                    pinned.copy_(value, non_blocking=True)
                    hosts[key] = pinned
                else:
                    hosts[key] = value
        stream.synchronize()
        arrays = {key: value.numpy() for key, value in hosts.items()}
        if any(not np.isfinite(value).all() for value in arrays.values()):
            raise RuntimeError("first update contains nonfinite captured values")
        for key in (f"optimizer_step/{name}" for name in NAMES):
            if arrays[key].item() != 1:
                raise RuntimeError(
                    f"optimizer did not complete exactly one AdamW step for {key}"
                )
        self.final["data"] = self.inputs
        self.final["data_sha256"] = checksum(self.inputs)
        stem = self.output / RECEIPT.format(rank=self.rank)
        with stem.with_suffix(".npz").open("xb") as handle:
            np.savez(handle, **arrays)
        receipt = {
            "version": 1,
            "rank": self.rank,
            "requested": self.requested,
            "resolved_config_digest": self.expected_config,
            "layers": LAYERS,
            "microbatches": MICROBATCHES,
            "samples_per_weight": SAMPLES,
            "selected_local_numel": {k: n for k, (n, _) in self.indices.items()},
            "initial_state": self.initial,
            "final_state": self.final,
            "payload_sha256": hashlib.sha256(
                stem.with_suffix(".npz").read_bytes()
            ).hexdigest(),
            "payload_bytes": stem.with_suffix(".npz").stat().st_size,
        }
        with stem.with_suffix(".json").open("xb") as handle:
            handle.write(json_bytes(receipt) + b"\n")
        self.finished = True


def required_samples():
    return (
        {f"pre_reduce/{mb:02d}/{name}" for mb in range(MICROBATCHES) for name in NAMES}
        | {
            f"{phase}/{name}"
            for phase in (
                "initial",
                "pre_clip",
                "post_clip",
                "parameter",
                "optimizer_step",
                "exp_avg",
                "exp_avg_sq",
            )
            for name in NAMES
        }
        | {
            "global_valid_tokens",
            "grad_norm",
            "loss",
            "loss_per_microbatch",
            "tokens_seen",
        }
    )


@contextmanager
def observe(adapter, requested, expected_config):
    import torch
    from torchtitan import train
    from torchtitan.callbacks.manager import CallbackManager
    from torchtitan.distributed import utils
    from torchtitan.distributed.gfsdp import _param_group

    options = adapter.options
    if (
        int(os.environ["WORLD_SIZE"]) != PROCESSES
        or options["expected_steps"] != STEPS
        or options["warmup_updates"]
        or options["window_updates"]
        or options["profile_update"] is not None
        or not expected_config
    ):
        raise RuntimeError(
            "EP8 numerical mode requires the verified one-step untimed configuration"
        )
    observation = Observation(options["output_dir"], requested, expected_config)
    original_save = train.save_loaded_run_bundle
    original_step = adapter.trainer_class.train_step
    original_forward_backward = adapter.trainer_class.forward_backward_step
    original_emit = CallbackManager.emit
    original_clip = utils.clip_grad_norm_
    original_reduce = _param_group.reduce_scatter

    def save(loaded, *args, **kwargs):
        observation.validate_resolved(loaded)
        return original_save(loaded, *args, **kwargs)

    def train_step(trainer, *args, **kwargs):
        observation.start_update(trainer)
        result = original_step(trainer, *args, **kwargs)
        observation.finish_update(trainer)
        return result

    def forward_backward(trainer, *args, **kwargs):
        index = len(observation.losses)
        if index >= MICROBATCHES or observation.microbatch is not None:
            raise RuntimeError("unexpected nesting or additional EP8 microbatch")
        observation.microbatch = index
        try:
            loss = original_forward_backward(trainer, *args, **kwargs)
        finally:
            observation.microbatch = None
        with torch.no_grad():
            local_loss = local_tensor(loss)
            if local_loss.numel() != 1:
                raise RuntimeError(
                    "expected a single native normalized loss per microbatch"
                )
            observation.losses.append(local_loss.reshape(()))
        return loss

    def emit(manager, event, trainer, **fields):
        state = original_emit(manager, event, trainer, **fields)
        if event == "after_dataloader":
            observation.record_inputs(state)
        elif event == "before_optimizer_step":
            denominator = fields["global_valid_tokens"]
            if not torch.is_tensor(denominator):
                denominator = torch.tensor(denominator, dtype=torch.float64)
            observation.sample("global_valid_tokens", denominator)
            observation.gradient_shards("pre_clip")
            observation.router_state(trainer, "pre_clip")
        elif event == "after_optimizer_step":
            observation.post_optimizer(trainer)
        elif event == "batch_end":
            observation.sample("loss", fields["loss"])
        return state

    def clip(*args, **kwargs):
        norm = original_clip(*args, **kwargs)
        observation.sample("grad_norm", norm)
        observation.gradient_shards("post_clip")
        return norm

    def reduce(params, active_params, output, input, group, *args, **kwargs):
        observation.before_reduce(active_params, group)
        return original_reduce(
            params, active_params, output, input, group, *args, **kwargs
        )

    with ExitStack() as stack:
        for owner, name, wrapper in (
            (train, "save_loaded_run_bundle", save),
            (adapter.trainer_class, "train_step", train_step),
            (adapter.trainer_class, "forward_backward_step", forward_backward),
            (CallbackManager, "emit", emit),
            (utils, "clip_grad_norm_", clip),
            (_param_group, "reduce_scatter", reduce),
        ):
            stack.enter_context(patch.object(owner, name, wrapper))
        yield
        if not observation.finished:
            raise RuntimeError("EP8 numerical rank did not complete one native update")


def load_run(path):
    import numpy as np
    import yaml

    completion = json.loads((path / "completion.json").read_text())
    metadata = json.loads((path / "metadata.json").read_text())
    source = json.loads((path / "source-final.json").read_text())
    resolved_bytes = (path / "native/config/resolved_config.yaml").read_bytes()
    resolved = yaml.safe_load(resolved_bytes)
    digest = resolved_digest(resolved)
    recipe = recipe_digest(resolved)
    input_provenance = {field: metadata["input_data"][field] for field in INPUT_FIELDS}
    router_shapes = {
        f"{phase}/layers.{layer}.moe.{field}": (128,)
        for phase in ("pre_clip", "post_optimizer")
        for layer in LAYERS
        for field in ("tokens_per_expert_E", "expert_bias_E")
    }
    if (
        completion["status"] != "PASS"
        or hashlib.sha256(resolved_bytes).hexdigest()
        != completion["resolved_config_sha256"]
        or metadata["source_snapshot_before"]["tracked_and_untracked_sha256"]
        != source["tracked_and_untracked_sha256"]
        or metadata["steps"] != STEPS
        or metadata["warmup_updates"]
        or metadata["window_updates"]
    ):
        raise RuntimeError(
            f"invalid native completion, source identity or update scope: {path}"
        )

    for rank in range(PROCESSES):
        stem = path / "measurement" / RECEIPT.format(rank=rank)
        receipt = json.loads(stem.with_suffix(".json").read_text())
        payload = stem.with_suffix(".npz")
        if (
            receipt["rank"] != rank
            or receipt["version"] != 1
            or receipt["payload_bytes"] > PAYLOAD_BYTES + 1024**2
            or receipt["resolved_config_digest"] != digest
            or (
                not receipt["requested"]
                and metadata["source_head_before"] != ORIGINAL_REVISION
            )
            or receipt["microbatches"] != MICROBATCHES
            or receipt["layers"] != list(LAYERS)
            or receipt["samples_per_weight"] != SAMPLES
            or receipt["payload_bytes"] != payload.stat().st_size
            or receipt["payload_sha256"]
            != hashlib.sha256(payload.read_bytes()).hexdigest()
        ):
            raise RuntimeError(f"invalid numerical receipt: {stem}")
        with np.load(payload, allow_pickle=False) as arrays:
            if not required_samples().issubset(arrays.files):
                raise RuntimeError(f"incomplete sample payload: {stem}")
            values = {name: arrays[name] for name in arrays.files}
            if (
                sum(array.nbytes for array in values.values()) > PAYLOAD_BYTES
                or values["loss_per_microbatch"].shape != (MICROBATCHES,)
                or any(
                    value.shape != router_shapes.get(name, (SAMPLES,))
                    or value.dtype != np.float32
                    for name, value in values.items()
                    if name in router_shapes
                    or name.split("/", 1)[0]
                    in {
                        "pre_reduce",
                        "initial",
                        "pre_clip",
                        "post_clip",
                        "parameter",
                        "exp_avg",
                        "exp_avg_sq",
                    }
                )
            ):
                raise RuntimeError(
                    f"invalid native sampled tensor shapes, dtypes or budget: {stem}"
                )
            yield (
                {
                    **receipt,
                    "source": source["tracked_and_untracked_sha256"],
                    "recipe": recipe,
                    "input_provenance": input_provenance,
                    "helper": metadata["config_hashes_before"][
                        "expert_gfsdp_numerics_sha256"
                    ],
                    "compiler_cache_policy": metadata["compiler_cache_policy"],
                },
                values,
            )


def metrics(left, right):
    import numpy as np

    if left.shape != right.shape or left.dtype != right.dtype:
        raise RuntimeError("numeric field shape or dtype differs")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise RuntimeError("numeric field contains nonfinite values")
    a = left.astype(np.float64)
    b = right.astype(np.float64)
    error = np.abs(b - a)
    scale = max(np.linalg.norm(a), np.linalg.norm(b))
    return {
        "exact": left.tobytes() == right.tobytes(),
        "max_abs": float(error.max()),
        "p99_abs": float(np.quantile(error, 0.99)),
        "relative_l2": float(np.linalg.norm(error) / scale) if scale else 0.0,
    }


def compare(reference, other, is_control):
    comparisons = []
    for (a, left), (b, right) in zip(load_run(reference), load_run(other), strict=True):
        if (
            a["requested"]
            or b["requested"] != ([] if is_control else OPTIMIZATIONS)
            or a["recipe"] != b["recipe"]
            or a["input_provenance"] != b["input_provenance"]
            or a["helper"] != b["helper"]
            or a["compiler_cache_policy"] != b["compiler_cache_policy"]
            or (
                is_control
                and (
                    a["source"] != b["source"]
                    or a["resolved_config_digest"] != b["resolved_config_digest"]
                )
            )
        ):
            raise RuntimeError(
                "expected original A/A and original A versus flagged expert candidate"
            )
        if a["selected_local_numel"] != b["selected_local_numel"]:
            raise RuntimeError(
                f"selected local tensor size differs on rank {a['rank']}"
            )
        for key in ("data_sha256", "scheduler"):
            if a["final_state"][key] != b["final_state"][key]:
                raise RuntimeError(
                    f"input or scheduler state differs: {key} rank {a['rank']}"
                )
        if a["initial_state"]["sha256"] != b["initial_state"]["sha256"]:
            groups = [
                key
                for key, value in a["initial_state"].items()
                if value != b["initial_state"].get(key) and key != "sha256"
            ]
            raise RuntimeError(f"initial state differs on rank {a['rank']}: {groups}")
        if left.keys() != right.keys():
            raise RuntimeError(f"sample coverage differs on rank {a['rank']}")
        rank = a["rank"]
        for key in (
            "global_valid_tokens",
            "tokens_seen",
            *(f"optimizer_step/{name}" for name in NAMES),
        ):
            if left[key].tobytes() != right[key].tobytes():
                raise RuntimeError(f"exact counter {key} differs on rank {rank}")
        comparisons.extend(
            {"rank": rank, "field": key, **metrics(left[key], right[key])}
            for key in sorted(left)
        )
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("repeat_original", type=Path)
    parser.add_argument("candidate", type=Path, nargs="?")
    args = parser.parse_args()
    roots = [
        path.resolve(strict=True)
        for path in (args.original, args.repeat_original, args.candidate)
        if path is not None
    ]
    if len(set(roots)) != len(roots):
        parser.error("each comparison arm must be a distinct native run")
    control = compare(roots[0], roots[1], is_control=True)
    candidate = (
        compare(roots[0], roots[2], is_control=False) if args.candidate else None
    )
    control_exact = all(row["exact"] for row in control)
    candidate_exact = (
        all(row["exact"] for row in candidate) if candidate is not None else None
    )
    status = (
        "EXACT_ON_CAPTURED_FIELDS"
        if control_exact and candidate_exact is not False
        else "INVESTIGATE_DIFFERENCES"
    )
    print(
        json.dumps(
            {
                "status": status,
                "control_exact": control_exact,
                "candidate_exact": candidate_exact,
                "control": control,
                "candidate": candidate,
            },
            indent=2,
        )
    )
    return 0 if status == "EXACT_ON_CAPTURED_FIELDS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
