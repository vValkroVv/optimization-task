"""Verify staged head dW through real GFSDP callbacks and production inputs."""

import argparse
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from head_dw_component import (
    FLAG,
    FIXTURE_SHA256,
    ROOT,
    CompileConfig,
    MTPLoss,
    bits_equal,
    load_fixture,
    sha256,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.gfsdp import MixedPrecisionPolicy, fully_shard
from torchtitan.models.common.decoder_blocks.sharding import set_decoder_sharding_config
from torchtitan.models.common.nn_modules import Linear

STAGED_DW = "GIFT_STAGED_HEAD_DW"


class StopBeforeSecondChunk(RuntimeError):
    pass


def new_loss(head, staged, fixture, backend):
    os.environ[STAGED_DW] = "1" if staged else "0"
    loss = MTPLoss(
        MTPLoss.Config(num_chunks=2, loss_weights=fixture["head_weights"][1:]),
        compile_config=CompileConfig(enable=True, components=["loss"], backend=backend),
    )
    loss.set_lm_head(head)
    return loss


def run_gfsdp(head, loss, fixture, grad_context, observer, case, *, abort=False):
    hidden = fixture["hidden"].detach().requires_grad_()
    targets = loss.pack_targets(fixture["labels"], {"positions": fixture["positions"]})
    forwards = 0
    selected = []

    def observe(_head, args, kwargs):
        nonlocal forwards
        forwards += 1
        lease = kwargs.get("_staged_head_dw")
        selected.append(lease is not None)
        if lease is not None and forwards == 2:
            assert lease.completed_callbacks == lease.writes == 1
            assert lease.param.unsharded_accumulated_grad is None
            assert grad_context.owner is lease.group
            try:
                with grad_context.borrow(torch.cuda.current_stream()):
                    raise AssertionError("Borrowed live staged gradient bytes")
            except RuntimeError as err:
                if "live owner" not in str(err):
                    raise
        if abort and forwards == 2:
            raise StopBeforeSecondChunk

    with head.register_forward_pre_hook(observe, with_kwargs=True):
        output = loss(hidden, targets, fixture["denominator"])
        output.backward()
    assert forwards == 2 and selected[0] == selected[1]
    assert head._get_fsdp_state()._fsdp_param_groups[0]._staged_head_dw is None
    observer.append({"case": case, "staged_chunks": selected})
    return output.detach(), hidden.grad


def run_checks(fixture_path):
    fixture, captured = load_fixture(fixture_path)
    width = fixture["weight"].size(1)
    vocab = fixture["weight"].size(0)
    device = torch.device("cuda", torch.cuda.current_device())
    mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("fsdp",))
    dims = ParallelDims(
        dp_replicate=1,
        dp_shard=dist.get_world_size(),
        cp=1,
        tp=1,
        pp=1,
        ep=1,
        world_size=dist.get_world_size(),
    )
    dims.build_mesh()
    decoder = SimpleNamespace(
        tok_embeddings=SimpleNamespace(),
        norm=SimpleNamespace(),
        lm_head=Linear.Config(in_features=width, out_features=vocab, bias=False),
    )
    set_decoder_sharding_config(decoder, loss_parallel=True, enable_sp=True)
    with torch.device("meta"):
        head = decoder.lm_head.build()
    head.weight = torch.nn.Parameter(fixture["weight"].to(torch.float32))
    head.parallelize(dims)
    head = fully_shard(
        head,
        mesh=mesh,
        reshard_after_forward=True,
        mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            cast_forward_inputs=False,
        ),
    )
    del fixture["weight"]
    head.set_own_grad_buffer()
    head.set_gradient_divide_factor(1.0)
    with torch.no_grad():
        # Initialize the real GFSDP slots before invoking the split loss.
        head(fixture["hidden"][0, :, :1].contiguous())
    state = head._get_fsdp_state()
    (group,) = state._fsdp_param_groups
    (param,) = group.fsdp_params
    context = group.grad_buffer_ctx
    control = new_loss(head, False, fixture, captured["compile_backend"])
    candidate = new_loss(head, True, fixture, captured["compile_backend"])
    scratch_stream = torch.cuda.Stream(device=device)
    observed = []

    @torch.no_grad()
    def local_dw():
        return head.weight.grad.to_local()

    def run(loss, label, *, abort=False):
        return run_gfsdp(head, loss, fixture, context, observed, label, abort=abort)

    def borrow_scratch():
        # Same event and owner protocol used by RowOwnerEmbedding._scratch.
        with (
            torch.cuda.stream(scratch_stream),
            context.borrow(scratch_stream) as buffers,
        ):
            buffers[None][:4096].fill_(0x5A)
        assert context.owner is None and context.release_event is not None

    expected = [run(control, f"control:{idx}") for idx in range(2)]
    torch.cuda.synchronize()
    expected_dw = local_dw().clone()
    head.weight.grad = None

    actual = []
    for idx in range(2):
        borrow_scratch()
        actual.append(run(candidate, f"staged:{idx}"))
    torch.cuda.synchronize()
    for a, b in zip(expected, actual, strict=True):
        assert all(bits_equal(x, y) for x, y in zip(a, b, strict=True))
    assert bits_equal(expected_dw, local_dw())

    expected_deferred = None
    for label, loss in (("control", control), ("staged", candidate)):
        head.weight.grad = None
        head.set_requires_gradient_sync(False, recurse=False)
        run(loss, f"{label}:incoming-no-sync")
        assert context.owner is group
        assert param.unsharded_accumulated_grad is not None
        head.set_requires_gradient_sync(True, recurse=False)
        run(loss, f"{label}:prior-unsharded-gradient")
        assert context.owner is None
        assert param.unsharded_accumulated_grad is None
        torch.cuda.synchronize()
        if expected_deferred is None:
            expected_deferred = local_dw().clone()
        else:
            assert bits_equal(expected_deferred, local_dw())

    hook_calls = []
    with param.unsharded_param.register_hook(
        lambda grad: hook_calls.append(grad.shape)
    ):
        run(candidate, "staged:weight-hook")
    assert hook_calls == [torch.Size((vocab, width))] * 2

    head.weight.grad = None
    try:
        run(candidate, "staged:interrupted", abort=True)
    except StopBeforeSecondChunk:
        assert group._staged_head_dw is None
        assert context.owner is None
        assert param.unsharded_accumulated_grad is None
        assert state._state_ctx.is_last_backward and group.reduce_grads
    else:
        raise AssertionError("The injected second chunk interruption was not raised")
    borrow_scratch()
    run(candidate, "staged:retried")
    torch.cuda.synchronize()
    assert all(
        row["staged_chunks"]
        == [row["case"] in {"staged:0", "staged:1", "staged:retried"}] * 2
        for row in observed
    )
    return observed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(2048)
    os.environ[FLAG] = "1"
    os.environ["GIFT_DIRECT_BF16_DHIDDEN"] = "0"
    if sha256(args.fixture) != FIXTURE_SHA256:
        raise ValueError("The production fixture SHA-256 did not match")
    dist.init_process_group("nccl")
    try:
        observed = run_checks(args.fixture)
        output = args.output / f"rank{dist.get_rank()}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        result = {
            "passed": True,
            "source_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "world_size": dist.get_world_size(),
            "rank": dist.get_rank(),
            "fixture_sha256": FIXTURE_SHA256,
            "folded_ce": True,
            "active_routes": observed,
            "checks": [
                "Bitwise loss, dHidden, and accumulated sharded FP32 dW",
                "Live owner protects both chunks and subsequent asynchronous scratch borrowing",
                "Incoming no-sync and prior unsharded gradients use native callbacks",
                "Unsharded weight hooks still receive both native chunk gradients",
                "Aborted first stage releases its lease and the next pair succeeds",
            ],
        }
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(output, flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
