"""Run the compiled consuming CE and native GFSDP head on eight local ranks.

Set PYTHONPATH to the integrated GIFT checkout and supply the rebuilt CUDA CE
library and its digest through GIFT_EXACT_FOLDED_CE_LIBRARY and
GIFT_EXACT_FOLDED_CE_LIBRARY_SHA256. Launch with torchrun --nproc-per-node=8.
"""

import json
import os

import torch
import torch.distributed as dist
from torch.nn import functional as F

from head_bench import make_head


VOCAB, WIDTH, BATCH, SEQUENCE = 256, 128, 2, 19
CHUNKS, WORLD, IGNORE_INDEX = 3, 8, -100
WEIGHTS = (1.0, 0.15, 0.0, 0.3, 0.1, 0.05, 0.2, 0.11, 0.07, 0.25, 0.18)
CHUNK_WIDTHS = [7, 7, 5]


def fixture(rank, batch_index, device):
    generator = torch.Generator().manual_seed(31129 + 2 * rank + batch_index)
    shape = (len(WEIGHTS), BATCH, SEQUENCE, WIDTH)
    hidden = torch.randint(-8, 9, shape, generator=generator).to(device, torch.bfloat16) / 32
    labels = torch.randint(VOCAB, (BATCH, SEQUENCE), generator=generator).to(device)
    labels[:, [2 + batch_index, 8, 16]] = IGNORE_INDEX
    positions = torch.arange(SEQUENCE, device=device).repeat(BATCH, 1)
    positions[0, 13:] -= 13
    positions[1, 14:] -= 14
    return hidden, labels, positions


def eager_reference(hidden, labels, positions, weight, denominator):
    targets = [labels]
    for offset in range(1, len(WEIGHTS)):
        shifted = torch.full_like(labels, IGNORE_INDEX)
        same_document = positions[:, offset:] == positions[:, :-offset] + offset
        shifted[:, :-offset] = labels[:, offset:].masked_fill(~same_document, IGNORE_INDEX)
        targets.append(shifted)
    targets = torch.stack(targets)

    dh_chunks = []
    dw = weight.new_zeros(weight.shape, dtype=torch.float64)
    head_loss = weight.new_zeros(len(WEIGHTS), dtype=torch.float64)
    scales = (weight.new_tensor(WEIGHTS, dtype=torch.float32) / denominator).double()
    for x, y in zip(hidden.chunk(CHUNKS, dim=2), targets.chunk(CHUNKS, dim=2), strict=True):
        x = x.contiguous()
        # Match the BF16 projection boundary, then use eager FP64 CE and VJP.
        z = F.linear(x, weight).double().reshape(len(WEIGHTS), -1, VOCAB)
        z.requires_grad_()
        raw = torch.stack([
            F.cross_entropy(rows, targets, ignore_index=IGNORE_INDEX, reduction="sum")
            for rows, targets in zip(z, y.reshape(len(WEIGHTS), -1), strict=True)
        ])
        (dz,) = torch.autograd.grad(raw, z, grad_outputs=scales)
        head_loss.add_(raw.detach() / denominator)
        dz = dz.to(torch.bfloat16).view(-1, VOCAB).double()
        dh_chunks.append((dz @ weight.double()).reshape_as(x).to(torch.bfloat16))
        dw.addmm_(dz.T, x.view(-1, WIDTH).double())

    main = head_loss[0]
    mtp_weights = head_loss.new_tensor(WEIGHTS[1:])
    weighted_mtp = (head_loss[1:] * mtp_weights).sum()
    components = {"loss_ce": main, "loss_mtp": weighted_mtp / mtp_weights.sum()}
    components.update({f"loss_mtp{i}": head_loss[i] for i, w in enumerate(WEIGHTS[1:], 1) if w})
    return main + weighted_mtp, components, torch.cat(dh_chunks, dim=2), dw


def check_gradient(actual, expected, *, l2_limit, peak_limit):
    error = actual.double() - expected.double()
    relative_l2 = float(error.norm() / expected.double().norm())
    max_abs = float(error.abs().max())
    assert relative_l2 <= l2_limit and max_abs <= peak_limit * float(expected.abs().max()), (
        relative_l2, max_abs, l2_limit, peak_limit,
    )
    return {"relative_l2": relative_l2, "max_abs": max_abs}


def run(rank, device):
    from torchtitan.components.loss import MTPLoss
    from torchtitan.config import CompileConfig

    torch._inductor.config.triton.cudagraphs = False
    torch.backends.cuda.matmul.allow_tf32 = False
    weight = torch.randint(-8, 9, (VOCAB, WIDTH), generator=torch.Generator().manual_seed(441))
    weight = weight.to(device, torch.bfloat16) / 32
    inputs = [fixture(rank, index, device) for index in range(2)]
    count = sum(labels.ne(IGNORE_INDEX).sum() for _, labels, _ in inputs)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    denominator = float(count)

    head = make_head(weight, WORLD)
    with torch.no_grad():
        head(inputs[0][0][0, :, :1].contiguous())
    (group,) = head._get_fsdp_state()._fsdp_param_groups
    (parameter,) = group.fsdp_params
    loss = MTPLoss(
        MTPLoss.Config(num_chunks=CHUNKS, loss_weights=list(WEIGHTS[1:])),
        compile_config=CompileConfig(enable=True, components=["loss"], backend="inductor"),
    )
    loss.set_lm_head(head)
    exact = loss._exact_folded_ce
    launch = exact.compiled
    ce_shapes = []

    def record_ce(logits, labels, scales, heads, path, digest):
        ce_shapes.append((logits.shape, heads))
        return launch(logits, labels, scales, heads, path, digest)

    exact.compiled = record_ce
    original_gather = dist.all_gather_into_tensor
    original_scatter = dist.reduce_scatter_tensor
    accumulated_dw = torch.zeros_like(weight, dtype=torch.float64)
    for index, (hidden, labels, positions) in enumerate(inputs):
        expected_loss, expected_components, expected_dh, local_dw = eager_reference(
            hidden, labels, positions, weight, denominator,
        )
        reduced_dw = local_dw.clone()
        dist.all_reduce(reduced_dw, op=dist.ReduceOp.SUM)
        accumulated_dw.add_(reduced_dw)
        # Ensure beta=0 overwrites stale data on each microbatch. The previous
        # reduce-scatter can alias this input buffer when accumulating dWeight.
        torch.cuda.synchronize(device)
        parameter._unsharded_accumulated_grad.fill_(torch.nan)
        ce_shapes.clear()
        leases, gathered, scattered = [], [], []

        def record_head(_head, args, kwargs):
            leases.append((args[0].size(1), kwargs["staged_head_dw"]))

        def record_gather(output_tensor, input_tensor, *args, **kwargs):
            out, inp = output_tensor, input_tensor
            assert inp.dtype == out.dtype == torch.bfloat16 and out.numel() == VOCAB * WIDTH
            assert inp.untyped_storage().data_ptr() != out.untyped_storage().data_ptr()
            gathered.append(out.numel())
            return original_gather(out, inp, *args, **kwargs)

        def record_scatter(out, inp, *args, **kwargs):
            assert inp.dtype == out.dtype == torch.float32 and out.numel() * WORLD == inp.numel()
            assert kwargs.get("op", args[0] if args else dist.ReduceOp.SUM) == dist.ReduceOp.SUM
            scattered.append(check_gradient(inp.view_as(local_dw), local_dw, l2_limit=2e-3, peak_limit=5e-3))
            return original_scatter(out, inp, *args, **kwargs)

        dist.all_gather_into_tensor, dist.reduce_scatter_tensor = record_gather, record_scatter
        try:
            with head.register_forward_pre_hook(record_head, with_kwargs=True):
                loss.reset_components()
                value = loss(hidden.requires_grad_(), loss.pack_targets(labels, {"positions": positions}), denominator)
                value.backward()
        finally:
            dist.all_gather_into_tensor, dist.reduce_scatter_tensor = original_gather, original_scatter

        assert len(gathered) == len(scattered) == 1
        assert [width for width, _ in leases] == CHUNK_WIDTHS
        lease = leases[0][1]
        assert all(item is lease for _, item in leases)
        assert lease.mode == "generic" and lease.writes == lease.completed_callbacks == CHUNKS
        assert exact.scales_checked and not loss._exact_folded_ce_fallback_logged
        assert ce_shapes == [((len(WEIGHTS) * BATCH * w, VOCAB), len(WEIGHTS)) for w in CHUNK_WIDTHS]
        assert hidden.grad.dtype == torch.bfloat16
        # The outer backward ends at a leaf in this replay. Wait for the event
        # that the root GFSDP callback would join during full-model backward.
        reduce_event = group._post_reduce_event
        reduction_complete_at_return = reduce_event is None or reduce_event.query()
        if reduce_event is not None:
            torch.cuda.current_stream(device).wait_event(reduce_event)
        with torch.no_grad():
            actual_dw = head.weight.grad.to_local().view(VOCAB // WORLD, WIDTH)
        assert actual_dw.dtype == torch.float32
        dw_error = check_gradient(actual_dw, accumulated_dw.chunk(WORLD)[rank], l2_limit=2e-3, peak_limit=5e-3)
        dh_error = check_gradient(hidden.grad, expected_dh, l2_limit=4e-3, peak_limit=8e-3)
        torch.testing.assert_close(value.detach().double(), expected_loss, atol=1e-6, rtol=2e-6)
        components = loss.components()
        assert components.keys() == expected_components.keys()
        for name, expected in expected_components.items():
            torch.testing.assert_close(components[name].double(), expected, atol=1e-6, rtol=2e-6)
        print(json.dumps({
            "rank": rank, "microbatch": index, "shapes": ce_shapes, "sha256": exact.library_sha256,
            "loss_abs_error": float((value.detach() - expected_loss).abs()), "dhidden": dh_error,
            "local_dw": scattered[0], "accumulated_shard_dw": dw_error, "denominator": denominator,
            "reduction_complete_at_return": reduction_complete_at_return,
        }), flush=True)


if __name__ == "__main__":
    assert os.environ.get("GIFT_EXACT_FOLDED_CE_LIBRARY")
    assert os.environ.get("GIFT_EXACT_FOLDED_CE_LIBRARY_SHA256")
    os.environ.update(
        GIFT_EXPERIMENT_LM_HEAD_MATRIX="generic", GIFT_EXPERIMENT_LM_HEAD_MATRIX_CHUNKS="",
        GIFT_STABLE_CE_FOLDED_BACKWARD="1", GIFT_DIRECT_BF16_DHIDDEN="1",
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    assert int(os.environ["WORLD_SIZE"]) == WORLD and int(os.environ["RANK"]) == local_rank
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    from torchtitan.tools.logging import init_logger

    init_logger()
    try:
        run(local_rank, device)
    finally:
        dist.destroy_process_group()
