"""Compare unchanged and fixed-shape row owners on the pinned CUDA runtime."""

from contextlib import contextmanager, nullcontext
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch
import argparse
import json
import os
import runpy

import torch
from torch import distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from row_owner_route_experiment import SOURCE_ROOT, use_fixed_shape_route


WORLD = 8
PINNED_TORCH = "2.15.0.dev20260822+cu130"
PINNED_TORCH_GIT = "8b353065ce260336c5fdfa74d754e8bd9b91b062"
PASSES = ("initial", "accumulated", "reset")


def check_runtime():
    if (
        torch.__version__ != PINNED_TORCH
        or torch.version.git_version != PINNED_TORCH_GIT
        or torch.version.cuda != "13.0"
    ):
        raise RuntimeError("expected the pinned PyTorch/CUDA embedding replay build")
    if int(os.environ.get("WORLD_SIZE", "0")) != WORLD:
        raise RuntimeError("expected exactly eight torchrun ranks")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    if "H100" not in torch.cuda.get_device_name():
        raise RuntimeError("the replay requires H100 GPUs")
    return rank


class GradientMessages:
    """Capture each BF16 message on its actual communication stream."""

    def __init__(self):
        self.pending = []

    def __call__(self, name):
        return self._capture() if name == "gradient_exchange" else nullcontext()

    @contextmanager
    def _capture(self):
        exchange = dist.all_to_all_single

        def capture(output, input, output_split_sizes, input_split_sizes, **kwargs):
            if input.dtype != torch.bfloat16 or output.dtype != torch.bfloat16:
                raise RuntimeError("expected BF16 transport for both gradient messages")
            sent = input.clone()
            returned = exchange(
                output, input, output_split_sizes, input_split_sizes, **kwargs
            )
            self.pending.append(
                (
                    sent,
                    output.clone(),
                    list(input_split_sizes),
                    list(output_split_sizes),
                )
            )
            return returned

        with patch.object(dist, "all_to_all_single", capture):
            yield

    def collect(self, digest):
        if len(self.pending) != 1:
            raise RuntimeError("expected exactly one gradient exchange per backward")
        sent, received, send, receive = self.pending.pop()
        return {
            "sent_bf16_sha256": digest(sent, torch),
            "received_bf16_sha256": digest(received, torch),
            "sent_bytes": sent.numel() * sent.element_size(),
            "received_bytes": received.numel() * received.element_size(),
            "send_split_rows": send,
            "receive_split_rows": receive,
        }


def run_arm(harness, mesh, ids, derivative, seed, fixed):
    digest = harness["tensor_sha256"]
    components = harness["components"]
    harness["cleanup_cache"](torch)
    with use_fixed_shape_route() if fixed else nullcontext():
        model, _ = components.build_embedding(mesh, "owner", seed=seed)
        inventory = components.gradient_buffer_inventory(model)
        weight = model.tok_embeddings.weight
        master = digest(weight._local_tensor, torch)
        if not inventory["owner_scratch_aliases_private_lm_head"]:
            raise RuntimeError("embedding scratch must borrow the existing head bucket")
        messages = GradientMessages()
        model.tok_embeddings.stage_observer = messages
        passes = {}
        for name in PASSES:
            if name == "reset":
                weight.grad = None
            result = harness["pass_forward_backward"](
                model, ids, derivative, torch, measure_peak=False
            )
            gradient = harness["local_grad"](model, torch)
            passes[name] = {
                "output_bf16_sha256": digest(result["output_cpu"], torch),
                "gradient_fp32_sha256": digest(gradient, torch),
                "gradient_fp32_bytes": gradient.numel() * gradient.element_size(),
                "messages": messages.collect(digest),
            }
    return {
        "master_fp32_sha256": master,
        "pool_bytes": inventory["unique_storage_bytes"],
        "passes": passes,
    }


def compare(args, rank):
    harness = runpy.run_path(
        str(SOURCE_ROOT / "runs/ffn-row-owner/embedding_numeric.py")
    )
    capture = runpy.run_path(
        str(SOURCE_ROOT / "runs/production-data-abcd/component_fixtures.py")
    )
    dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    try:
        captured, receipt = capture["load_fixture"](args.input_fixtures, rank)
        ids = captured.pop("input_ids").cuda()
        del captured
        derivative, derivative_receipt = harness["build_shared_upstream"](
            ids, args.seed, rank, torch
        )
        mesh = init_device_mesh("cuda", (WORLD,), mesh_dim_names=("fsdp",))
        control = run_arm(harness, mesh, ids, derivative, args.seed, False)
        candidate = run_arm(harness, mesh, ids, derivative, args.seed, True)
        row = {
            "rank": rank,
            "input_ids_int64_sha256": receipt["input_ids_int64_sha256"],
            "fixture_tensors_sha256": receipt["tensors_sha256"],
            "upstream": derivative_receipt,
            "exact": control == candidate,
            "control": control,
            "candidate": candidate,
        }
        ranks = [None] * WORLD
        dist.all_gather_object(ranks, row)
        exact = all(row["exact"] for row in ranks)
        if rank == 0:
            report = {
                "exact": exact,
                "identity": "SHA-256 over tensor bytes in logical order",
                "seed": args.seed,
                "torch": torch.__version__,
                "torch_git": torch.version.git_version,
                "cuda": torch.version.cuda,
                "nccl": torch.cuda.nccl.version(),
                "ranks": ranks,
            }
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        if not exact:
            raise RuntimeError(
                "fixed-shape owner changed weights, output, gradients or messages"
            )
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.add_argument("--seed", type=int, default=2048)
    parser.add_argument("--input-fixtures", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--owner-checkpoint-dir", type=Path)
    args = parser.parse_args()
    if args.owner_checkpoint_dir is None and (
        args.input_fixtures is None or args.output is None
    ):
        parser.error("exact replay requires --input-fixtures and --output")
    rank = check_runtime()
    if args.owner_checkpoint_dir:
        tests = runpy.run_path(
            str(SOURCE_ROOT / "gift/tests/unit_tests/test_row_owner_embedding.py")
        )
        with use_fixed_shape_route():
            tests["_worker"](rank, WORLD, str(args.owner_checkpoint_dir), "nccl")
    else:
        compare(args, rank)


if __name__ == "__main__":
    main()
