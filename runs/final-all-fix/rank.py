"""Run the native trainer with the verified expert writer and optional activation receipt."""

import argparse
from collections import Counter
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def describe(trainer):
    import torch.distributed as dist
    from torchtitan.distributed.gfsdp.fully_shard import _iter_fsdp_param_groups
    from torchtitan.distributed.gfsdp.row_owner_embedding import RowOwnerEmbedding
    from torchtitan.ops.float8.cuda_gemm import FP8Linear

    dims = trainer.parallel_dims
    parts = trainer.model_parts
    queue = trainer.dataloader._prefetch_iterator
    experts = {}
    embedding = {}
    tail_layers = []
    for part in parts:
        for group in _iter_fsdp_param_groups(part, recurse=True):
            slot = getattr(group, "_direct_expert_dw_slot", None)
            if slot is None:
                continue
            cache = group._fp8_cache
            experts[group._module_fqn] = {
                "slot": str(slot.dtype),
                "fp8_generation": cache.prepared_generation,
                "cache_ready": cache.ready_event is not None,
                "retain_tail_weights": group.comm_ctx.retain_tail_weights,
                "compact_owner": group._owner_transport is not None,
            }
        for name, module in part.named_modules():
            if isinstance(module, RowOwnerEmbedding):
                embedding[name] = {"compact_grad_used": module._compact_source_logged}
            if (
                isinstance(module, FP8Linear)
                and module._quantize_weight_tail
                and (module.in_features, module.out_features)
                in ((10944, 4096), (4096, 10944))
            ):
                tail_layers.append(name)
    return {
        "mesh": {
            name: dist.get_process_group_ranks(dims.get_mesh(name).get_group())
            for name in ("fsdp", "efsdp", "ep")
        },
        "prefetch": {
            "depth": queue._capacity,
            "live": queue._thread.is_alive(),
            "ready": len(queue._ready),
        },
        "experts": experts,
        "embedding": embedding,
        "dense_fp8_tail_layers": tail_layers,
        "microbatches_per_update": trainer.gradient_accumulation_steps,
    }


def activation_probe(stack, trainer_class, output):
    import torch

    original = trainer_class.forward_backward_step
    count = 0
    receipt = {"rank": int(os.environ["RANK"])}

    def observe(trainer, *args, **kwargs):
        nonlocal count
        count += 1
        if count != 2:
            return original(trainer, *args, **kwargs)
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        ) as profile:
            result = original(trainer, *args, **kwargs)
            torch.cuda.synchronize()
        counts = Counter(
            (event.device_type.name, event.name) for event in profile.events()
        )
        markers = (
            "torchtitan::",
            "gfsdp::",
            "cute_gated_norm",
            "gdn",
            "mla_rope",
            "gnfwd",
            "gnbwd",
            "fold",
            "centered",
            "write_owner",
            "hidden_z",
            "cudacherry",
            "quantize_blk",
            "blk128",
            "k_grouped",
            "capacity_mask",
        )
        receipt["events"] = {
            f"{device} {name}": n
            for (device, name), n in counts.items()
            if any(marker in name.lower() for marker in markers)
        }
        # The logical 10944-wide weight enters the CudaCherry quantizer directly.
        # A pre-padded weight has a different shape, even when the layer flag is on.
        tail_shapes = Counter()
        for event in profile.events():
            if "quant" in event.name.lower() and "blk" in event.name.lower():
                for shape in event.input_shapes:
                    if shape in ([10944, 4096], [4096, 10944]):
                        tail_shapes[f"{event.name} {shape}"] += 1
        receipt["dense_tail_quantizer_inputs"] = dict(tail_shapes)
        receipt.update(describe(trainer))
        return result

    stack.enter_context(patch.object(trainer_class, "forward_backward_step", observe))

    def save():
        receipt["completed_microbatches"] = count
        (output / f"activation-rank-{receipt['rank']}.json").write_text(
            json.dumps(receipt, sort_keys=True, indent=2) + "\n"
        )

    return save


def main():
    sep = sys.argv.index("--")
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type=int, choices=(4, 8), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--direct-dw-build", type=Path, required=True)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args(sys.argv[1:sep])
    rank = int(os.environ["LOCAL_RANK"])
    os.environ["TRITON_CACHE_DIR"] = str(
        Path(os.environ["TRITON_CACHE_DIR"]) / f"rank_{rank}"
    )

    tilelang = args.output / "cache" / "tilelang" / f"rank_{rank}"
    tilelang_tmp = tilelang / "tmp"
    tilelang_tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TILELANG_CACHE_DIR"] = str(tilelang)
    os.environ["TILELANG_TMP_DIR"] = str(tilelang_tmp)

    import torch

    torch.cuda.set_device(rank)
    import torchtitan
    from torchtitan import trainer
    from torchtitan.distributed.deepep import hybridep, routing
    from torchtitan.ops.float8.direct_expert_dw import configure_direct_expert_dw
    from torchtitan.ops.warmup import deep_gemm_fast_warmup
    from torchtitan.train import main as train
    from streaming_filelock import StreamingFileLockBindings
    from measure import DeepGemmCacheRedirect

    sys.path.insert(0, str(HERE / "direct-dw"))
    from direct_dw import load_direct_dw

    if not Path(torchtitan.__file__).resolve().is_relative_to(ROOT / "gift"):
        raise RuntimeError(
            "The trainer must be imported from the final-all-fix checkout"
        )
    extension = load_direct_dw(args.direct_dw_build, num_sms=116)
    configure_direct_expert_dw(extension.k_grouped_fp8_gemm_nt_contiguous_rounded_f32)
    with ExitStack() as stack:
        stack.enter_context(StreamingFileLockBindings())
        stack.enter_context(
            DeepGemmCacheRedirect(
                deep_gemm_fast_warmup,
                trainer,
                Path(os.environ["TASK_DEEP_GEMM_RANK_CACHE_ROOT"]),
            )
        )
        # All six donor comparisons used Original A routing and native FLA.
        stack.enter_context(
            patch.object(hybridep, "indices_to_map", routing.indices_to_map)
        )
        save = (
            activation_probe(stack, trainer.Trainer, args.output)
            if args.probe
            else None
        )
        train(sys.argv[sep + 1 :])
        if save:
            save()


if __name__ == "__main__":
    main()
