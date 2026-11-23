"""Optional metadata-only probe for the native layer-0 GFSDP initializers."""

from contextlib import contextmanager
import json
import logging
import os
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

logger = logging.getLogger(__name__)


@contextmanager
def capture(path):
    """Write one rank's receipt without making the training run depend on it."""
    path = Path(path)
    receipt = {
        "schema_version": 1,
        "rank": os.environ.get("RANK"),
        "contexts": 0,
        "completed_contexts": 0,
        "initializers": [],
        "draws": [],
        "errors": [],
    }

    def note_error(stage, exc):
        message = f"{stage}: {type(exc).__name__}: {exc}"
        if message not in receipt["errors"]:
            receipt["errors"].append(message)

    def cuda_rng(device, generator=None):
        try:
            if device.type != "cuda":
                raise RuntimeError(f"expected CUDA, got {device}")
            if generator is None:
                index = (
                    device.index
                    if device.index is not None
                    else torch.cuda.current_device()
                )
                generator = torch.cuda.default_generators[index]
            # Both queries read the host generator. No sample or CUDA tensor is read.
            return {"seed": generator.initial_seed(), "offset": generator.get_offset()}
        except Exception as exc:
            note_error("read CUDA generator", exc)
            return None

    @contextmanager
    def observe_init(backend, model):
        with weight_init_context(backend, model):
            receipt["contexts"] += 1
            try:
                names = {}
                for name, param in model.named_parameters(remove_duplicate=False):
                    fqn = ".".join(
                        p for p in name.split(".") if p != "_checkpoint_wrapped_module"
                    )
                    if fqn.startswith("layers.0."):
                        if id(param) in names:
                            raise RuntimeError(f"aliased layer-0 initializer: {fqn}")
                        names[id(param)] = fqn
                receipt["parameter_fqns"] = sorted(names.values())
                if not names:
                    raise RuntimeError(
                        "no layer-0 parameters in the weight-init context"
                    )
            except Exception as exc:
                note_error("identify layer 0", exc)
                yield
                return

            def observe_param(module, name, param):
                fqn = names.get(id(param))
                if fqn is not None:
                    try:
                        receipt["initializers"].append(
                            {
                                "fqn": fqn,
                                "initializer": type(module._param_init[name]).__name__,
                            }
                        )
                    except Exception as exc:
                        note_error("inspect initializer", exc)
                return init_param(module, name, param)

            def observe_fill(arg, func, args, kwargs):
                fqn = names.get(id(arg))
                if fqn is None:
                    return fill_random(arg, func, args, kwargs)
                draw = {"fqn": fqn, "operator": str(func), "samples": []}
                receipt["draws"].append(draw)
                try:
                    spec = arg._spec
                    axis = rng.get_ragged_shard_axis(spec)
                    placement = spec.placements[axis]
                    tracker = rng.random._rng_tracker
                    shard_dim = rng._fsdp2_shard_dims.get(id(arg), 0)
                    draw.update(
                        logical_shape=list(arg.shape),
                        local_numel=placement.local_numel,
                        local_offset=placement.global_offset,
                        per_rank_numels=list(placement.shard_numels),
                        mesh_coordinate=spec.mesh.get_coordinate(),
                        mesh_shape=list(spec.mesh.shape),
                        match_fsdp2_init=rng._match_fsdp2_init,
                        shard_dim=shard_dim,
                        reference_placements=[
                            str(value)
                            for value in rng.fsdp2_equivalent_placements(
                                spec, shard_dim=shard_dim
                            )
                        ],
                        tracker_enabled=tracker.distribute_region_enabled,
                        tracker_device=str(tracker._device),
                        sampler_args=list(args),
                        sampler_kwargs={
                            key: value
                            for key, value in kwargs.items()
                            if key != "generator"
                        },
                        entry_rng=cuda_rng(arg.device),
                    )
                except Exception as exc:
                    note_error("inspect GFSDP RNG context", exc)

                def observe_sample(tensor, *positional, **keywords):
                    generator = keywords.get("generator")
                    sample = {
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                        "device": str(tensor.device),
                        "explicit_generator": generator is not None,
                        "before": cuda_rng(tensor.device, generator),
                    }
                    draw["samples"].append(sample)
                    result = func(tensor, *positional, **keywords)
                    sample["after"] = cuda_rng(tensor.device, generator)
                    return result

                result = fill_random(arg, observe_sample, args, kwargs)
                draw["exit_rng"] = cuda_rng(arg.device)
                return result

            with (
                patch.object(Module, "_init_param", observe_param),
                patch.object(rng, "_fill_random", observe_fill),
            ):
                yield
        receipt["completed_contexts"] += 1

    try:
        try:
            import torch
            from torchtitan.distributed.backends.gfsdp2 import GFSDP2Backend
            from torchtitan.distributed.gfsdp import _random as rng
            from torchtitan.protocols.module import Module

            receipt["torch_version"] = str(torch.__version__)
            weight_init_context = GFSDP2Backend.weight_init_context
            init_param = Module._init_param
            fill_random = rng._fill_random
        except Exception as exc:
            note_error("install RNG probe", exc)
            yield
        else:
            with patch.object(GFSDP2Backend, "weight_init_context", observe_init):
                yield
    finally:
        receipt["status"] = (
            "unavailable"
            if not receipt["draws"]
            else "partial"
            if receipt["errors"]
            or receipt["contexts"] != 1
            or receipt["completed_contexts"] != 1
            else "complete"
        )
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(receipt, indent=2) + "\n")
            temporary.replace(path)
        except (OSError, TypeError, ValueError) as exc:
            logger.warning(
                "Could not write the optional initial RNG receipt %s: %s", path, exc
            )
