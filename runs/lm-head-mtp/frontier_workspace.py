"""Benchmark-only reuse of the staged LM-head projection output buffer.

This wrapper patches _StagedHeadLinear.forward in-process. It leaves the
training checkout and the existing stack harness unchanged.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch


TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / "gift"))

from torchtitan.distributed.gfsdp import _staged_head_dw as staged_head_dw  # noqa: E402


METRICS = {
    "forward_calls": 0,
    "workspace_allocations": 0,
    "workspace_reuses": 0,
    "reuse_order_checks": 0,
}
_BUFFER_ATTR = "_frontier_workspace_projection"
_USES_ATTR = "_frontier_workspace_uses"
_STREAM_ATTR = "_frontier_workspace_stream"


def install_workspace_forward():
    def workspace_forward(ctx, input, weight, lease):
        if (
            weight is not lease.param.unsharded_param
            or weight.dtype != torch.bfloat16
            or not weight.is_contiguous()
        ):
            raise RuntimeError("The staged LM head did not receive its BF16 weight")
        if input.dtype != weight.dtype or input.device != weight.device or input.ndim < 2:
            raise RuntimeError("The staged LM-head projection input changed dtype, device, or rank")
        if torch.is_grad_enabled():
            raise RuntimeError("The staged LM-head custom forward unexpectedly enabled autograd")

        stream = torch.cuda.current_stream(input.device)
        group_stream = lease.group.device_handle.current_stream()
        if stream != group_stream:
            raise RuntimeError("The staged projection is not on the GFSDP lease stream")

        prior_uses = getattr(lease, _USES_ATTR, 0)
        if prior_uses == 0:
            if lease.writes != 0 or lease.completed_callbacks != 0:
                raise RuntimeError("The first workspace use found an already-used dW lease")
            projection_stream = stream
            rows = input.numel() // input.size(-1)
            workspace = torch.empty(
                (rows, weight.size(0)),
                device=input.device,
                dtype=input.dtype,
            )
            setattr(lease, _BUFFER_ATTR, workspace)
            setattr(lease, _STREAM_ATTR, projection_stream)
            METRICS["workspace_allocations"] += 1
        else:
            if (
                lease.writes != prior_uses
                or lease.completed_callbacks != prior_uses
                or lease.stream != stream
                or getattr(lease, _STREAM_ATTR) != stream
            ):
                raise RuntimeError(
                    "Projection workspace reuse preceded same-stream head backward completion"
                )
            workspace = getattr(lease, _BUFFER_ATTR)
            METRICS["workspace_reuses"] += 1
            METRICS["reuse_order_checks"] += 1

        flat_input = input.flatten(0, -2)
        expected = (flat_input.size(0), weight.size(0))
        if (
            workspace.shape != expected
            or workspace.dtype != input.dtype
            or workspace.device != input.device
            or not workspace.is_contiguous()
        ):
            raise RuntimeError("The per-loss projection workspace no longer matches native output")

        ctx.save_for_backward(input, weight)
        ctx.lease = lease
        torch.mm(flat_input, weight.t(), out=workspace)
        setattr(lease, _USES_ATTR, prior_uses + 1)
        METRICS["forward_calls"] += 1
        return workspace.view(*input.shape[:-1], weight.size(0))

    staged_head_dw._StagedHeadLinear.forward = staticmethod(workspace_forward)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chunks", type=int, choices=(8, 16))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    install_workspace_forward()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import frontier_stack

    sys.argv[1:] = ["stack", str(args.chunks), str(args.output)]
    frontier_stack.main()

    receipt = json.loads(args.output.read_text())
    receipt["projection_workspace"] = {
        "active": True,
        "allocation": "lazy BF16 [flattened rows, vocabulary] buffer attached to each DirectHeadDw lease",
        "reuse_gate": "same stream and lease writes == completed_callbacks == prior workspace uses",
        "kernel": "torch.mm(flattened input, weight.T, out=workspace)",
        "logical_output": "native leading input dimensions plus vocabulary dimension",
        **METRICS,
    }
    receipt["workspace_wrapper_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
