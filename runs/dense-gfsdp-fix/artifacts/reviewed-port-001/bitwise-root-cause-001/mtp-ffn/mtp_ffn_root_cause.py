#!/usr/bin/env python3
"""Locate the first differing MTP FFN backward boundary on a fixed native VJP."""

import argparse
from contextlib import ExitStack
import gc
from pathlib import Path
import sys
from unittest.mock import patch


PREFIX = "mtp.layers.0."
FFN = "block.feed_forward"
PHASES = ("pre-reduce", "taps", "fixed-ffn", "fp8")
WEIGHTS = (
    "block.feed_forward.",
    "input_proj.",
    "final_norm.",
    "token_norm.",
    "hidden_norm.",
    "block.ffn_norm.",
    "block.post_ffn_norm.",
)
MODULES = {
    "final_norm": "final_norm",
    "block.post_ffn_norm": "post_ffn_norm",
    "block.ffn_residual_scale": "ffn_residual_scale",
    FFN: "ffn",
    "block.ffn_norm": "ffn_norm",
    "input_proj": "input_proj",
    "token_norm": "token_norm",
    "hidden_norm": "hidden_norm",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def selected(name):
    return name.startswith(PREFIX) and name.removeprefix(PREFIX).startswith(WEIGHTS)


def clone(tensor):
    return tensor.detach().clone()


def watch_pre_reduce(model, canonical, stack):
    from torchtitan.distributed.gfsdp import GFSDPModule

    snapshots = {}
    seen = set()
    for module in model.modules():
        if not isinstance(module, GFSDPModule):
            continue
        for group in module._get_fsdp_state()._fsdp_param_groups:
            if id(group) in seen:
                continue
            seen.add(id(group))
            params = [
                (canonical(param._param_fqn), param)
                for param in group.fsdp_params
                if param._param_fqn and selected(canonical(param._param_fqn))
            ]
            if not params:
                continue
            original = group.post_backward

            def observed(*args, params=params, original=original):
                for name, param in params:
                    if param.unsharded_param.grad is not None:
                        require(
                            name not in snapshots, f"Duplicate full gradient: {name}"
                        )
                        snapshots[name] = clone(param.unsharded_grad_data)
                return original(*args)

            stack.enter_context(patch.object(group, "post_backward", observed))
    return snapshots


class Taps:
    """Opaque in-graph copies with an identity VJP, plus an optional fixed adjoint."""

    def __init__(self, torch, fixed):
        self.forward = {}
        self.backward = {}
        self.fixed = fixed
        self.pinned = None
        self.delivered = None

        @torch.library.custom_op("mtp_ffn_probe::forward", mutates_args=())
        def forward(x: torch.Tensor, label: str) -> torch.Tensor:
            self.forward.setdefault(label, []).append(clone(x))
            return x.clone()

        @forward.register_fake
        def forward_fake(x, label):
            return x.clone()

        @torch.library.custom_op("mtp_ffn_probe::backward", mutates_args=())
        def backward(x: torch.Tensor, label: str) -> torch.Tensor:
            self.backward.setdefault(label, []).append(clone(x))
            if self.fixed and label == "ffn.out":
                if self.pinned is None:
                    self.pinned = clone(x)
                require(
                    self.pinned.shape == x.shape and self.pinned.dtype == x.dtype,
                    "Fixed FFN cotangent differs in shape or dtype",
                )
                result = self.pinned.clone()
                self.delivered = clone(result)
                return result
            return x.clone()

        @backward.register_fake
        def backward_fake(x, label):
            return x.clone()

        def setup_context(ctx, inputs, output):
            ctx.label = inputs[1]

        def propagate(ctx, dy):
            return backward(dy, ctx.label), None

        forward.register_autograd(propagate, setup_context=setup_context)
        self.capture = forward

    def install(self, model, canonical, stack, *, fp8):
        targets = dict(MODULES)
        if fp8:
            # Input to the FFN projections can be a packed FP8 wrapper. Observe
            # the ordinary BF16 GEMM outputs at the SwiGLU producer boundaries.
            targets.update({FFN + ".w1": "gate_gemm", FFN + ".w3": "up_gemm"})
        for name, module in model.named_modules():
            relative = canonical(name).removeprefix(PREFIX)
            if relative not in targets:
                continue
            tag = targets[relative]
            original = module.forward
            tap = self.capture
            output_only = relative in (FFN + ".w1", FFN + ".w3")

            def observed(
                x, *args, original=original, tag=tag, output_only=output_only, **kwargs
            ):
                if not output_only:
                    x = tap(x, tag + ".in")
                return tap(original(x, *args, **kwargs), tag + ".out")

            stack.enter_context(patch.object(module, "forward", observed))

    def collect(self, base):
        require(
            self.forward.keys() == self.backward.keys(),
            "Incomplete compiled tap VJP coverage",
        )
        require(
            all(
                name in self.forward
                for name in (
                    "ffn.in",
                    "ffn.out",
                    "final_norm.in",
                    "final_norm.out",
                    "input_proj.in",
                    "input_proj.out",
                )
            ),
            "Missing mandatory compiled boundaries",
        )
        tensors, calls = {}, {}
        for name, forwards in self.forward.items():
            backwards = self.backward[name]
            calls[name] = {
                "forward": len(forwards),
                "backward": len(backwards),
                "forward_sha256": [base.digest(value.cpu()) for value in forwards],
            }
            require(
                len(backwards) == 1, f"Multiple backward contributions at tap {name}"
            )
            tensors["forward/" + name] = forwards[0].cpu()
            tensors["backward/" + name] = backwards[0].cpu()
        if self.fixed:
            require(self.delivered is not None, "Missing injected FFN cotangent")
            tensors["backward/ffn.out.used"] = self.delivered.cpu()
        self.forward.clear()
        self.backward.clear()
        self.delivered = None
        return tensors, calls


def measure(base, harness, fixture, model, device, rank, taps, phase):
    import torch

    tensors, receipt = harness.captured_inputs(fixture, device, rank, "mtp")
    model.zero_grad(set_to_none=True)
    with ExitStack() as patches:
        before = watch_pre_reduce(model, harness.canonical, patches)
        if taps is not None:
            taps.install(model, harness.canonical, patches, fp8=phase == "fp8")
        out = harness.microbatch(model, "mtp", tensors)
    results = {
        "forward/component": harness.cpu_tensor(out),
        "backward/hidden": harness.cpu_tensor(tensors[0].grad),
        "backward/token_embedding": harness.cpu_tensor(tensors[1].grad),
        "backward/component_cotangent": harness.cpu_tensor(tensors[2]),
        "forward/captured_hidden": harness.cpu_tensor(tensors[0]),
        "forward/captured_embedding": harness.cpu_tensor(tensors[1]),
    }
    post = {}
    for name, parameter in model.named_parameters():
        name = harness.canonical(name)
        if selected(name):
            require(
                parameter.grad is not None and parameter.grad.dtype == torch.float32,
                f"Expected reduced FP32 gradient: {name}",
            )
            post[name] = harness.cpu_tensor(parameter.grad)
    require(
        before.keys() == post.keys() and len(before) >= 6,
        "Incomplete pre-reduce gradient snapshots",
    )
    results.update(
        {"pre_reduce/" + name: value.cpu() for name, value in before.items()}
    )
    results.update({"post_reduce/" + name: value for name, value in post.items()})
    calls = {}
    if taps is not None:
        extra, calls = taps.collect(base)
        results.update(extra)
    return receipt, results, calls


def run(base, harness, options, args, dims, device, rank):
    import torch
    import torchtitan

    source = options.source_root.resolve(strict=True)
    require(
        Path(torchtitan.__file__).resolve().is_relative_to(source / "gift/torchtitan"),
        "Torchtitan originated from a different checkout",
    )
    taps = (
        Taps(torch, options.phase == "fixed-ffn")
        if options.phase != "pre-reduce"
        else None
    )
    reference = initial = weights_sha = rng = None
    evidence = {}
    for arm in ("A", "AA", "B"):
        torch._dynamo.reset()
        split = arm == "B"
        cfg = harness.resolved_config("mtp", split)
        cfg.dump_folder = str(args.output)
        model = harness.build(cfg, "mtp", dims, device)
        activation = base.inventory(harness, model, "mtp", split)
        weights = base.logical_weights(harness, model)
        sha = {name: base.digest(value) for name, value in weights.items()}
        if arm == "A":
            initial, weights_sha = weights, sha
            rng = (torch.get_rng_state(), torch.cuda.get_rng_state(device))
            restored = []
        else:
            require(
                weights_sha.keys() == sha.keys(),
                "MTP initial parameter inventory differs",
            )
            restored = [name for name in sha if sha[name] != weights_sha[name]]
            if restored:
                base.restore_weights(harness, model, initial, restored)
                sha = {
                    name: base.digest(value)
                    for name, value in base.logical_weights(harness, model).items()
                }
            require(sha == weights_sha, f"{arm} initial logical FP32 bytes differ")
        del weights
        torch.set_rng_state(rng[0])
        torch.cuda.set_rng_state(rng[1], device)
        fixture, tensors, calls = measure(
            base, harness, options.fixture, model, device, rank, taps, options.phase
        )
        if arm == "A":
            reference = tensors
            output = {name: base.digest(tensor) for name, tensor in tensors.items()}
        else:
            output = base.differences(reference, tensors)
        evidence[arm] = {
            "activation": activation,
            "restored_initial_fqns": restored,
            "input_fixture": fixture,
            "taps": calls,
            "tensors": output,
        }
        del tensors, model
        gc.collect()
        torch.cuda.empty_cache()
    return {
        "phase": options.phase,
        "rank": rank,
        "source": str(source),
        "diagnostic": str(Path(__file__).resolve()),
        "weights_sha256": weights_sha,
        "scope": "MTP VJP on one native captured operand tuple per rank; no optimizer",
        "arms": evidence,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--base-ffn-dir",
        type=Path,
        default=(Path(__file__).resolve().parents[2] / "all-final-bitwise-001/ffn"),
    )
    options = parser.parse_args()
    sys.path.insert(0, str(options.base_ffn_dir.resolve(strict=True)))
    import ffn_bitwise as base

    def probe(native, unused_fixture, unused_root, args, dims, device, rank):
        return run(base, native, options, args, dims, device, rank)

    with (
        patch.object(base, "run", probe),
        patch.object(
            sys,
            "argv",
            [
                sys.argv[0],
                "--component",
                "mtp",
                "--source-root",
                str(options.source_root),
                "--fixture",
                str(options.fixture),
                "--output",
                str(options.output),
            ],
        ),
    ):
        base.main()


if __name__ == "__main__":
    main()
