"""Two-GPU transport numerics, run by the H100 integration suite."""

import argparse
import json
import os
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from torchtitan.config.moe import TokenDropConfig
from torchtitan.models.common.moe.dispatch import (
    DeepEPTokenDispatcher,
    HybridEPTokenDispatcher,
)


def _reference_keep(scores, ids, policy, *, num_experts, ep_size):
    """Select on CPU with stable Python sorting, independently of the kernels."""
    num_tokens, top_k = ids.shape
    keep = ids >= 0
    if policy is None:
        return keep
    if policy.strategy == "fixed_capacity":
        capacity = policy.expert_capacity_tokens // ep_size
    elif policy.strategy == "capacity_factor":
        capacity = int(
            (num_tokens * top_k // num_experts) * policy.expert_capacity_factor
        )
    else:
        capacity = (
            int(
                num_tokens
                * top_k
                * ep_size
                * policy.expert_capacity_factor
                / num_experts
            )
            // ep_size
        )
    if policy.drop_by_group_capacity:
        counts = torch.bincount(ids[keep], minlength=num_experts)
        dist.all_reduce(counts)
        loads = counts.view(ep_size, num_experts // ep_size).sum(1)
        if not bool((loads > capacity * num_experts).any()):
            return keep
    result = torch.zeros_like(keep)
    cpu_scores, cpu_ids = scores.detach().cpu(), ids.cpu()
    for expert in range(num_experts):
        routes = [
            (t, k)
            for t in range(num_tokens)
            for k in range(top_k)
            if cpu_ids[t, k] == expert
        ]
        routes.sort(key=lambda route: -float(cpu_scores[route]))
        for route in routes[:capacity]:
            result[route] = True
    return result


def _check_case(mesh, policy, *, backend, non_blocking, empty_receiver, name):
    rank, ep_size = dist.get_rank(), dist.get_world_size()
    num_experts, dim, num_tokens, top_k = 4, 256, 16, 2
    # Dyadic fixtures keep bf16 arithmetic exact enough to detect missing routes
    # even when every output/gradient should be zero. Scores deliberately tie.
    torch.manual_seed(42 + rank)
    x = (
        ((torch.randint(0, 2, (num_tokens, dim), device="cuda") * 2 - 1) / 8)
        .to(torch.bfloat16)
        .requires_grad_()
    )
    scores = (
        (
            (torch.arange(num_tokens, device="cuda")[:, None].expand(-1, top_k) % 2 + 1)
            / 4
        )
        .contiguous()
        .requires_grad_()
    )
    experts = (0, 1) if empty_receiver else (0, 2)
    ids = torch.tensor(experts, device="cuda").expand(num_tokens, -1).contiguous()
    counts = torch.bincount(ids.flatten(), minlength=num_experts)
    global_weights = torch.stack(
        [
            (e + 1) * torch.eye(dim, device="cuda", dtype=torch.bfloat16) / 4
            for e in range(num_experts)
        ]
    )
    weights = global_weights.chunk(ep_size)[rank].clone().requires_grad_()
    keep = _reference_keep(
        scores, ids, policy, num_experts=num_experts, ep_size=ep_size
    )
    expected_counts = torch.bincount(ids[keep], minlength=num_experts)
    dist.all_reduce(expected_counts)
    cls = {"deepep": DeepEPTokenDispatcher, "hybridep": HybridEPTokenDispatcher}[
        backend
    ]
    kwargs = {"token_drop": policy}
    if backend == "hybridep":
        kwargs.update(
            non_blocking_capacity_factor=non_blocking, fuse_permute=non_blocking is None
        )
    dispatcher = (
        cls.Config(**kwargs)
        .with_capacity_context(
            num_tokens=num_tokens, top_k=top_k, num_experts=num_experts, ep_size=ep_size
        )
        .build()
    )
    dispatcher.wire_meshes(ep_mesh=mesh)
    observations = []
    dispatcher.dispatch_monitor = observations.append
    routed, sizes, metadata = dispatcher.dispatch(
        x, scores, ids, counts, score_before_experts=False
    )
    actual_keep = observations[-1].capacity_keep_mask
    torch.testing.assert_close(
        actual_keep if actual_keep is not None else ids >= 0, keep
    )
    torch.testing.assert_close(
        sizes.to("cuda"), expected_counts.chunk(ep_size)[rank], check_dtype=False
    )
    sizes_list = sizes.tolist()
    populated = sum(sizes_list)
    parts = routed[:populated].split(sizes_list)
    expert_output = torch.cat(
        [part @ weight for part, weight in zip(parts, weights, strict=True)]
    )
    expert_output = torch.cat(
        [expert_output, expert_output.new_zeros((routed.shape[0] - populated, dim))]
    )
    output = dispatcher.combine(expert_output, metadata, x)
    dispatcher.sync_combine()
    output.float().square().sum().backward()

    xr, sr, wr = (
        v.detach().clone().requires_grad_() for v in (x, scores, global_weights)
    )
    expected = sum(
        (xr @ wr[e]) * (sr[:, k] * keep[:, k]).to(x.dtype)[:, None]
        for k, e in enumerate(experts)
    )
    expected.float().square().sum().backward()
    # Reference loss is local to each source; actual expert gradients accumulate
    # contributions dispatched from every source to the expert's owning rank.
    dist.all_reduce(wr.grad)
    errors = {}
    for label, actual, reference in (
        ("output", output, expected),
        ("input_grad", x.grad, xr.grad),
        ("score_grad", scores.grad, sr.grad),
        ("expert_grad", weights.grad, wr.grad.chunk(ep_size)[rank]),
    ):
        error = (actual.float() - reference.float()).abs().max()
        dist.all_reduce(error, op=dist.ReduceOp.MAX)
        errors[label] = error.item()
        assert errors[label] == 0, (
            f"{backend}/{name}/{label}: max absolute error={errors[label]}, "
            f"non_blocking={non_blocking}, policy={policy}"
        )
    result = dict(
        case=name,
        branchless=policy.branchless_capacity_drop if policy else None,
        non_blocking=non_blocking,
        empty_receiver=empty_receiver,
        retained_routes=int(expected_counts.sum()),
        max_abs_errors=errors,
    )
    if rank == 0:
        print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("deepep", "hybridep"), required=True)
    parser.add_argument("--dump_folder", type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=90))
    try:
        if dist.get_world_size() != 2:
            raise ValueError("Token-drop numerical tests require exactly two GPUs")
        mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("ep",))
        cases = (
            ("no_drop", None, False),
            ("no_drop_empty_receiver", None, True),
            ("relative", TokenDropConfig(expert_capacity_factor=0.5), False),
            (
                "fixed",
                TokenDropConfig(strategy="fixed_capacity", expert_capacity_tokens=8),
                False,
            ),
            (
                "peak",
                TokenDropConfig(
                    strategy="peak_capacity_factor", expert_capacity_factor=0.5
                ),
                False,
            ),
            ("empty_receiver", TokenDropConfig(expert_capacity_factor=0.5), True),
            (
                "group_skip",
                TokenDropConfig(
                    expert_capacity_factor=1.0, drop_by_group_capacity=True
                ),
                False,
            ),
            (
                "group_overload",
                TokenDropConfig(
                    expert_capacity_factor=1.0, drop_by_group_capacity=True
                ),
                True,
            ),
            (
                "all_dropped",
                TokenDropConfig(strategy="fixed_capacity", expert_capacity_tokens=1),
                False,
            ),
            (
                "all_dropped_group",
                TokenDropConfig(
                    strategy="fixed_capacity",
                    expert_capacity_tokens=1,
                    drop_by_group_capacity=True,
                ),
                True,
            ),
        )
        results = []
        for name, policy, empty_receiver in cases:
            for branchless in (False, True) if policy else (False,):
                for non_blocking in (
                    (None, 1.0) if args.backend == "hybridep" else (None,)
                ):
                    results.append(
                        _check_case(
                            mesh,
                            replace(policy, branchless_capacity_drop=branchless)
                            if policy
                            else None,
                            backend=args.backend,
                            non_blocking=non_blocking,
                            empty_receiver=empty_receiver,
                            name=name,
                        )
                    )
        if dist.get_rank() == 0:
            args.dump_folder.mkdir(parents=True, exist_ok=True)
            (args.dump_folder / "numerics.json").write_text(
                json.dumps(results, indent=2) + "\n"
            )
            print(
                f"PASS {args.backend}: {len(results)} output/gradient comparisons",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
