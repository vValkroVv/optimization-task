"""Prepare a dense FFN fixture from a saved native layer-0 training boundary."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dense import (
    CASE,
    INIT_SEED,
    TAIL_ENV,
    compile_regions,
    layer0_crop,
    validate_ffn,
    validate_tensors,
)


def prepare(native_fixtures, rank, output):
    import torch
    from torchtitan.experiments.flash_qla.backend import install_backend

    # Rebuild the captured campaign's GDN branch with the explicit-pad reference.
    backend = install_backend("external_gate")
    os.environ[TAIL_ENV] = "0"
    source, config, crop = layer0_crop()
    fixture_source = source.fixture_module()
    records, native_receipt = fixture_source.load_fixture(native_fixtures, rank)
    row = records["layer0"]
    del records

    device = torch.device("cuda")
    torch.manual_seed(INIT_SEED)
    crop.to_empty(device=device)
    with torch.no_grad():
        crop.init_weights(buffer_device=device)
        for param in crop.parameters():
            # The isolated view uses the same parameter dtype as GFSDP forward.
            param.data = param.data.to(torch.bfloat16)
    crop.train()
    prefix, ffn, suffix = compile_regions(config, crop)
    validate_ffn(ffn, tail=False)

    hidden = row["hidden"].to(device).requires_grad_()
    upstream = row["upstream"].to(device)
    masks = fixture_source.restore(row["attention_masks"], device)
    positions = fixture_source.restore(row["positions"], device)
    token_mask = fixture_source.restore(row["token_mask"], device)
    residual, ffn_input = prefix(hidden, crop.freqs_cis, masks, positions, token_mask)
    residual = residual.detach().requires_grad_()
    ffn_input = ffn_input.detach().requires_grad_()

    # Derive dL/d(FFN output) through the actual suffix instead of reusing
    # dL/d(layer output), which sits after the gated norm and residual add.
    ffn_out = ffn(ffn_input).detach().requires_grad_()
    layer_out = suffix(residual, ffn_out)
    (tangent,) = torch.autograd.grad(layer_out, ffn_out, grad_outputs=upstream)
    payload = {
        "schema_version": 1,
        "case": CASE,
        "input": ffn_input.detach().cpu(),
        "tangent": tangent.detach().cpu(),
        "weights": {name: val.detach().cpu() for name, val in ffn.named_parameters()},
    }
    validate_tensors(payload)
    for name, val in (("input", payload["input"]), ("tangent", payload["tangent"])):
        if not val.isfinite().all() or not val.count_nonzero():
            raise ValueError(f"reconstructed {name} is nonfinite or entirely zero")

    output.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = output.with_suffix(".receipt.json")
    if output.exists() or receipt_path.exists():
        raise FileExistsError(f"dense fixture output already exists at {output}")
    with output.open("xb") as handle:
        torch.save(payload, handle)
    receipt = {
        "schema_version": 1,
        "case": CASE,
        "input": "native layer-0 hidden through the reconstructed production prefix",
        "tangent": "Reconstructed suffix VJP with native layer-0 upstream",
        "weights": "Single-device production crop init with FP32 masters and BF16 compute",
        "initialization_seed": INIT_SEED,
        "native_checkpoint_weights_recovered": False,
        "native_fixture": str(native_fixtures.resolve()),
        "native_rank": rank,
        "native_source_head": native_receipt["source_head"],
        "native_tensors_sha256": native_receipt["tensors_sha256"],
        "native_input_ids_sha256": native_receipt["input_ids_int64_sha256"],
        "sha256": fixture_source.digest(output),
        "torch": str(torch.__version__),
        "gdn_backend": backend,
        "precision": config.training.mixed_precision_param,
        "activation_checkpoint": config.to_dict()["activation_checkpoint"],
        "compile": config.to_dict()["compile"],
    }
    with receipt_path.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    print(json.dumps(receipt, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-fixtures", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0, choices=range(8))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.native_fixtures, args.rank, args.output)


if __name__ == "__main__":
    main()
