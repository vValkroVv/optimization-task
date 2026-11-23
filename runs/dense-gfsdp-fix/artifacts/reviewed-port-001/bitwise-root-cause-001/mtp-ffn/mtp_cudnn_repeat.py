#!/usr/bin/env python3
"""Replay the native cuDNN attention backward operands from one cached MTP VJP."""

import argparse
from pathlib import Path
import sys
from unittest.mock import patch


INPUTS = (
    "grad_out",
    "query",
    "key",
    "value",
    "out",
    "logsumexp",
    "philox_seed",
    "philox_offset",
    "attn_bias",
    "cum_seq_q",
    "cum_seq_k",
    "max_q",
    "max_k",
    "dropout_p",
    "is_causal",
)
OUTPUTS = ("dQ", "dK", "dV")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def freeze_operand(torch, tensor):
    """Keep native strides, storage offsets, and alignment on the same device."""
    shape, stride, offset = tensor.shape, tensor.stride(), tensor.storage_offset()
    if offset:
        extent = offset + (
            1 + sum((n - 1) * s for n, s in zip(shape, stride)) if tensor.numel() else 0
        )
        allocation = torch.empty(extent, dtype=tensor.dtype, device=tensor.device)
        frozen = allocation.as_strided(shape, stride, offset)
    else:
        frozen = torch.empty_strided(
            shape, stride, dtype=tensor.dtype, device=tensor.device
        )
    frozen.copy_(tensor)
    require(
        tensor.numel() == 0 or tensor.data_ptr() % 128 == frozen.data_ptr() % 128,
        "Cannot reproduce native cuDNN operand alignment",
    )
    return frozen


def output_tensors(torch, outputs):
    require(
        len(outputs) == 3 and all(isinstance(value, torch.Tensor) for value in outputs),
        "Native cuDNN backward did not return dQ/dK/dV",
    )
    return dict(
        zip(OUTPUTS, (value.detach().clone() for value in outputs), strict=True)
    )


def value_receipt(base, value):
    import torch

    if not isinstance(value, torch.Tensor):
        return value
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "storage_offset": value.storage_offset(),
        "alignment_mod_128": value.data_ptr() % 128,
        "sha256": base.digest(value.cpu()),
    }


def byte_identity(value):
    if not isinstance(value, dict):
        return value
    return value["dtype"], value["shape"], value["sha256"]


class NativeDispatch:
    def __init__(self, torch, native):
        self.torch = torch
        self.native = native
        self.calls = 0
        self.operands = None
        self.options = None
        self.outputs = None

    def __call__(self, *args, **kwargs):
        self.calls += 1
        require(
            self.calls == 1 and len(args) == len(INPUTS) and set(kwargs) == {"scale"},
            "Expected one pinned native cuDNN backward in the stock MTP executable",
        )
        with self.torch.no_grad():
            self.operands = tuple(
                freeze_operand(self.torch, value)
                if isinstance(value, self.torch.Tensor)
                else value
                for value in args
            )
        self.options = dict(kwargs)
        outputs = self.native(*args, **kwargs)
        self.outputs = output_tensors(self.torch, outputs)
        return outputs


def cpu_outputs(outputs):
    return {name: value.cpu() for name, value in outputs.items()}


def graph_activity(torch):
    counters = torch._dynamo.utils.counters
    return {
        "unique_graphs": counters["stats"]["unique_graphs"],
        "aot_autograd_total": counters["aot_autograd"]["total"],
    }


def native_once(torch, harness, fixture, model, device, rank, packet, native):
    tensors, receipt = harness.captured_inputs(fixture, device, rank, "mtp")
    hidden, embedding, upstream, positions, masks, token_mask = tensors
    model.zero_grad(set_to_none=True)
    # Cache selection happens before patching the runtime operator reference.
    result = model(
        hidden,
        embedding,
        positions=positions,
        attention_masks=masks,
        token_mask=token_mask,
    )
    capture = NativeDispatch(torch, native)
    with patch.object(packet, "default", capture):
        result.backward(upstream)
    require(
        capture.calls == 1,
        "Cached Inductor backward did not use the runtime interception",
    )
    return capture, receipt, result.detach(), hidden.grad, embedding.grad


def direct_repeats(torch, base, capture, repeats):
    first = cpu_outputs(
        output_tensors(torch, capture.native(*capture.operands, **capture.options))
    )
    comparisons = [
        base.differences(
            first,
            cpu_outputs(
                output_tensors(
                    torch, capture.native(*capture.operands, **capture.options)
                )
            ),
        )
        for _ in range(repeats - 1)
    ]
    return first, comparisons


def deterministic_replay(torch, base, capture, repeats, stock):
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn = torch.backends.cudnn.deterministic
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        first, repeat = direct_repeats(torch, base, capture, repeats)
    except RuntimeError as error:
        if "determin" not in str(error).lower():
            raise
        return {"status": "deterministic_runtime_error", "error": str(error)}
    finally:
        torch.backends.cudnn.deterministic = cudnn
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    return {
        "status": "completed",
        "same_setting_repeats": repeat,
        "versus_stock_first": base.differences(stock, first),
    }


def run(base, harness, options, args, dims, device, rank):
    import torch
    import torchtitan

    root = options.source_root.resolve(strict=True)
    require(
        Path(torchtitan.__file__).resolve().is_relative_to(root / "gift/torchtitan"),
        "Torchtitan originated from a different checkout",
    )
    cfg = harness.resolved_config("mtp", False)
    cfg.dump_folder = str(args.output)
    model = harness.build(cfg, "mtp", dims, device)
    activation = base.inventory(harness, model, "mtp", False)

    def weight_digests():
        return {
            key: base.digest(value)
            for key, value in base.logical_weights(harness, model).items()
        }

    weights = weight_digests()
    rng = (torch.get_rng_state(), torch.cuda.get_rng_state(device))
    warmup, _ = harness.captured_inputs(options.fixture, device, rank, "mtp")
    harness.microbatch(model, "mtp", warmup)
    del warmup
    graphs = graph_activity(torch)
    packet = torch.ops.aten._scaled_dot_product_cudnn_attention_backward
    native = packet.default
    require(
        isinstance(native, torch._ops.OpOverload),
        "cuDNN probe must invoke the original ATen overload",
    )
    records = []
    for _ in range(2):
        torch.set_rng_state(rng[0])
        torch.cuda.set_rng_state(rng[1], device)
        capture, fixture, fwd, hidden, embedding = native_once(
            torch, harness, options.fixture, model, device, rank, packet, native
        )
        operands = dict(
            zip(
                INPUTS,
                (value_receipt(base, value) for value in capture.operands),
                strict=True,
            )
        )
        boundary = cpu_outputs(
            {"output": fwd, "hidden_grad": hidden, "embedding_grad": embedding}
        )
        records.append((capture, fixture, operands, boundary))
    require(
        graph_activity(torch) == graphs,
        "Repeated VJPs did not reuse the cached stock executable",
    )
    require(
        weight_digests() == weights,
        "Logical FP32 weights changed during the repeat probe",
    )
    (
        (a, fixture, a_inputs, a_boundary),
        (aa, repeated_fixture, aa_inputs, aa_boundary),
    ) = records
    require(
        fixture == repeated_fixture and a.options == aa.options,
        "Stock repeat input fixture or attention scale differs",
    )

    actual = [cpu_outputs(record.outputs) for record in (a, aa)]
    policy = {
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "torch_deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    with torch.no_grad(), torch.cuda.device(device):
        fixed, repeat = direct_repeats(torch, base, a, options.repeats)
        deterministic = (
            deterministic_replay(torch, base, a, options.repeats, fixed)
            if options.deterministic_replay
            else None
        )
    return {
        "scope": "Native cached stock MTP A/A and same-operand cuDNN backward replay; no optimizer",
        "rank": rank,
        "activation": activation,
        "fixture": fixture,
        "weights_sha256": weights,
        "operator_schema": str(native._schema),
        "options": a.options,
        "compile_counters": graphs,
        "native_policy": policy,
        "a_operands": a_inputs,
        "aa_operands": aa_inputs,
        "native_operand_differences": [
            key for key in INPUTS if a_inputs[key] != aa_inputs[key]
        ],
        "native_operand_byte_differences": [
            key
            for key in INPUTS
            if byte_identity(a_inputs[key]) != byte_identity(aa_inputs[key])
        ],
        "native_attention_output_sha256": [
            {name: base.digest(value) for name, value in group.items()}
            for group in actual
        ],
        "native_attention_aa": base.differences(*actual),
        "native_component_boundary_aa": base.differences(a_boundary, aa_boundary),
        "fixed_operands_stock_replay": {
            "versus_original_native_call": base.differences(actual[0], fixed),
            "same_setting_repeats": repeat,
        },
        "fixed_operands_deterministic_replay": deterministic,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, choices=(2, 3, 4, 5), default=3)
    parser.add_argument("--deterministic-replay", action="store_true")
    parser.add_argument(
        "--base-ffn-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "all-final-bitwise-001/ffn",
    )
    args = parser.parse_args()
    sys.path.insert(0, str(args.base_ffn_dir.resolve(strict=True)))
    import ffn_bitwise as base

    def probe(native, unused_fixture, unused_root, native_args, dims, device, rank):
        return run(base, native, args, native_args, dims, device, rank)

    argv = [
        str(Path(__file__).resolve()),
        "--component",
        "mtp",
        "--source-root",
        str(args.source_root),
        "--fixture",
        str(args.fixture),
        "--output",
        str(args.output),
    ]
    with patch.object(base, "run", probe), patch.object(sys, "argv", argv):
        base.main()


if __name__ == "__main__":
    main()
