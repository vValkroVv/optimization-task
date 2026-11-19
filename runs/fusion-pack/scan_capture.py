"""Capture the first real rank-zero GDN scan VJP in a diagnostic run."""

import importlib.util
import json
import os
from pathlib import Path

import torch


def install(output_dir: Path) -> None:
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else int(os.environ["RANK"])
    if rank != 0:
        return

    from torchtitan.models.common.attention.linear import fla_custom_ops as ops

    key = "gdn_chunk"
    if key in ops._DIRECT_VJP:
        raise ValueError("Scan capture requires the baseline GDN VJP")
    helper_path = Path(__file__).resolve().parents[1] / "flash-qla" / "capture.py"
    spec = importlib.util.spec_from_file_location("_fusion_pack_capture_helpers", helper_path)
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)

    output_dir.mkdir(parents=True, exist_ok=True)
    fixture = output_dir / "fixture-01.pt"
    manifest = fixture.with_suffix(".json")
    if fixture.exists() or manifest.exists():
        raise FileExistsError(fixture)

    def capture(leaves, grad_outputs, static, scalars, text):
        del ops._DIRECT_VJP[key]
        forward = ops._RECOMPUTE[key](static, scalars, text)
        output = None

        def forward_for_capture(*inputs):
            nonlocal output
            output = forward(*inputs)
            return output

        grads = ops.vjp_via_recompute(forward_for_capture, leaves, grad_outputs)
        (grad_o,) = grad_outputs
        (cu,) = static
        tensors = dict(zip(helpers.NAMES, leaves, strict=True))
        tensors.update(output=output, grad_out=grad_o)
        if cu is not None:
            tensors["cu_seqlens"] = cu
        temporary = fixture.with_suffix(".pt.tmp")
        torch.save({name: helpers.snapshot(value) for name, value in tensors.items()}, temporary)
        temporary.replace(fixture)
        result = {
            "rank": rank,
            "vjp": "external_gate baseline",
            "tensors": {name: helpers.tensor_meta(value) for name, value in tensors.items()},
            "artifact_layout": "shape and stride preserved, storage offset rebased to zero",
            "fixture_sha256": helpers.sha256(fixture),
            "capture_sha256": helpers.sha256(Path(__file__)),
        }
        temporary = manifest.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(manifest)
        print(f"[SCAN_CAPTURE] rank={rank} fixture={fixture} sha256={result['fixture_sha256']}", flush=True)
        return list(grads)

    ops._DIRECT_VJP[key] = capture
