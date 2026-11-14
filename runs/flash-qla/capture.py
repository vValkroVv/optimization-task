"""Opt-in first-update GDN VJP boundary capture. Never import in timed runs."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import threading

import torch

NAMES = ("q", "k", "v", "raw_g", "beta", "A_log", "dt_bias")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def snapshot(tensor):
    """Preserve shape/strides while initializing all serialized padding bytes."""
    shape, stride = tuple(tensor.shape), tuple(tensor.stride())
    if any(s < 0 for s in stride):
        raise ValueError("Negative strides cannot be captured")
    size = 0 if tensor.numel() == 0 else 1 + sum((n - 1) * s for n, s in zip(shape, stride))
    storage = torch.zeros(size, dtype=tensor.dtype, device="cpu")
    result = storage.as_strided(shape, stride)
    result.copy_(tensor.detach())
    return result


def tensor_meta(t):
    return {"shape": list(t.shape), "stride": list(t.stride()),
            "dtype": str(t.dtype), "device": str(t.device),
            "storage_offset": t.storage_offset(), "requires_grad": t.requires_grad}


def install_capture(output_dir: Path, max_fixtures: int = 6):
    """Install before model construction in one dedicated first-update run.

    Hooks exist only in the real native autograd recomputation, not fake tracing.
    The caller owns restricting the training run to the intended update.
    """
    from fla.ops.backends import BackendRegistry, BaseBackend
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from torchtitan.experiments.flash_qla.backend import run_gdn

    if max_fixtures < 1:
        raise ValueError("max_fixtures must be positive")
    if os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") == "1":
        raise RuntimeError("FLA_DISABLE_BACKEND_DISPATCH=1 prevents capture backend activation")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    signature = inspect.signature(chunk_gated_delta_rule)
    source_identity = {"torch": torch.__version__,
                       "fla_upstream_revision": "1d2281ff1b7988d8070f00dba5ffa9747ae3a60e",
                       "flash_qla_upstream_revision": "a14dbf3474514e1674526431e960325f72081e9f",
                       "capture_sha256": sha256(Path(__file__)),
                       "native_backend_sha256": sha256(Path(inspect.getfile(run_gdn))),
                       "fla_chunk_sha256": sha256(Path(inspect.getfile(inspect.unwrap(chunk_gated_delta_rule))))}
    lock = threading.Lock()
    counters = {"calls": 0, "vjp_calls": 0, "saved": 0}
    rank = (torch.distributed.get_rank() if torch.distributed.is_initialized()
            else int(os.environ.get("RANK", "0")))

    class CaptureBackend(BaseBackend):
        backend_type = "task_first_update_capture"
        priority = 0

        def chunk_gated_delta_rule(self, *args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            # FLA keeps raw-gate flags and parameters in its public **kwargs.
            # Flatten that binding without copying or replacing tensor objects.
            a = dict(bound.arguments)
            for name, parameter in signature.parameters.items():
                if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                    a.update(a.pop(name))
            if a.get("initial_state") is not None or a.get("output_final_state", False):
                raise ValueError("Capture supports the actual zero-state, output-only GIFT path")
            if a.get("use_qk_l2norm_in_kernel", False) or not a.get("use_gate_in_kernel", False):
                raise ValueError("Capture requires GIFT's normalized-qk/raw-gate boundary")
            inputs = [a[n] for n in ("q", "k", "v", "g", "beta", "A_log", "dt_bias")]
            out, state = run_gdn("fla", *inputs, cu_seqlens=a.get("cu_seqlens"), scale=a.get("scale"))
            with lock:
                counters["calls"] += 1
                call = counters["calls"]
                eligible = rank == 0 and torch.is_grad_enabled() and out.requires_grad
                if eligible:
                    counters["vjp_calls"] += 1
                slot = counters["vjp_calls"]
            if eligible and slot <= max_fixtures:
                fired = False

                def save(grad):
                    nonlocal fired
                    if fired:
                        return grad
                    fired = True
                    payload = {n: snapshot(t) for n, t in zip(NAMES, inputs)}
                    payload.update(output=snapshot(out), grad_out=snapshot(grad),
                                   cu_seqlens=None if a.get("cu_seqlens") is None else snapshot(a["cu_seqlens"]),
                                   scale=a.get("scale"))
                    meta = {"schema": 1, "rank": rank, "call_counter": call,
                            "backward_order_slot": slot, "scope": "first-update actual VJP boundary; not post-warmup",
                            "layer_id": None, "initial_state": None, "output_final_state": False,
                            "source_identity": source_identity,
                            "artifact_layout": "shape/stride preserved; storage offset rebased to zero; padding initialized to zero",
                            "tensors": {n: tensor_meta(t) for n, t in zip(NAMES, inputs)},
                            "output": tensor_meta(out), "grad_out": tensor_meta(grad),
                            "pid": os.getpid()}
                    path = output_dir / f"fixture-{slot:02d}.pt"
                    if path.exists():
                        raise FileExistsError(path)
                    temporary = path.with_suffix(f".pt.{os.getpid()}.tmp")
                    torch.save(payload, temporary)
                    temporary.replace(path)
                    meta["fixture_sha256"] = sha256(path)
                    manifest = path.with_suffix(".json")
                    temporary = manifest.with_suffix(".json.tmp")
                    temporary.write_text(json.dumps(meta, indent=2) + "\n")
                    temporary.replace(manifest)
                    with lock:
                        counters["saved"] += 1
                    print(f"[GDN_CAPTURE] saved={path} slot={slot} sha256={meta['fixture_sha256']}", flush=True)
                    return grad

                out.register_hook(save)
            return out, state

    BackendRegistry.ensure_initialized("gated_delta_rule")
    registry = BackendRegistry._registries.get("gated_delta_rule")
    if registry is None:
        registry = BackendRegistry("gated_delta_rule")
    if any(b.priority <= 0 for b in registry._backends.values()):
        raise RuntimeError("Install capture in a fresh process without another priority-0 backend")
    registry.register(CaptureBackend())
    identity = {"backend": CaptureBackend.backend_type, "rank": rank,
                "max_fixtures": max_fixtures, "output_dir": str(output_dir), "counters": counters}
    print(f"[GDN_CAPTURE] installed rank={rank} max_fixtures={max_fixtures}", flush=True)
    return identity
