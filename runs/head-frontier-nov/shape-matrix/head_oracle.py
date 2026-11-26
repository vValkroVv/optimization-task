"""Evaluate the full isolated head against an untimed FP64 reference."""

import argparse
import hashlib
import math
import os
from pathlib import Path

import torch

from matrix_fixture import SHAPES, load_inputs


COMPONENT_NAMES = ("loss_ce", "loss_mtp")


def _fixture_metadata(record, chunks, world):
    def label_digest(tensor):
        if tensor is None:
            return None
        return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()

    hidden, weight = record["hidden"], record["weight"]
    if hidden.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("The oracle expects the native BF16 projection operands")
    if hidden.ndim != 4 or hidden.shape[0] != 2 or weight.shape[1] != hidden.shape[-1]:
        raise ValueError("The oracle expects one main head and one MTP head")
    if chunks < 1 or hidden.shape[2] % chunks or world < 1 or weight.shape[0] % world:
        raise ValueError("The sequence and vocabulary must split evenly for the oracle")
    denominator = float(record["denominator"])
    weights = list(record["head_weights"])
    if not math.isfinite(denominator) or denominator <= 0.0 or not all(map(math.isfinite, weights)):
        raise ValueError("The oracle needs a positive denominator and finite head weights")
    return {
        "format": 1,
        "provenance": record["provenance"],
        "hidden_shape": list(hidden.shape),
        "weight_shape": list(weight.shape),
        "labels_sha256": label_digest(record["labels"]),
        "positions_sha256": label_digest(record["positions"]),
        "ignore_index": int(record["ignore_index"]),
        "head_weights": weights,
        "denominator": denominator,
        "chunks": chunks,
        "world": world,
    }


def _cache_path(metadata, refdir):
    _, batch, seq, width = metadata["hidden_shape"]
    vocab, _ = metadata["weight_shape"]
    chunks, world = metadata["chunks"], metadata["world"]
    return Path(refdir) / f"oracle-b{batch}-s{seq}-d{width}-v{vocab}-c{chunks}-w{world}.pt"


@torch.no_grad()
def _reference(record, metadata, device):
    from torchtitan.components.loss import MTPLoss

    hidden = record["hidden"].to(device)
    weight = record["weight"].to(device)
    labels = record["labels"]
    positions = record["positions"]
    ignore_index = metadata["ignore_index"]
    targets = torch.stack((labels, MTPLoss._shift_labels(labels, 1, positions))).to(device)
    valid_targets = targets[targets != ignore_index]
    if (valid_targets < 0).any() or (valid_targets >= weight.shape[0]).any():
        raise ValueError("Oracle target outside the projection vocabulary")

    weight64 = weight.double()
    dw = torch.zeros_like(weight64)
    dh = torch.empty_like(hidden, dtype=torch.float64)
    losses = weight64.new_zeros(2)
    denom = metadata["denominator"]
    weights = metadata["head_weights"]
    steps = hidden.shape[2] // metadata["chunks"]

    for start in range(0, hidden.shape[2], steps):
        span = slice(start, start + steps)
        h = hidden[:, :, span, :].contiguous().view(-1, hidden.shape[-1])
        y = targets[:, :, span].contiguous().view(-1)
        # Use the same externally chunked BF16 projection as the native head.
        scores = (h @ weight.T).double()
        for head, (rows, target) in enumerate(zip(scores.chunk(2), y.chunk(2), strict=True)):
            valid = target != ignore_index
            safe = target.masked_fill(~valid, 0).unsqueeze(1)
            rows.sub_(rows.amax(dim=1, keepdim=True))
            selected = rows.gather(1, safe)
            rows.exp_()
            partition = rows.sum(dim=1, keepdim=True)
            loss = (partition.log() - selected).masked_fill_(~valid[:, None], 0.0)
            losses[head] += loss.sum() / denom

            # Softmax and its upstream scale stay FP64 through both GEMMs.
            rows.div_(partition)
            rows.scatter_add_(1, safe, -valid[:, None].double())
            rows.mul_(weights[head] / denom)
            rows.masked_fill_(~valid[:, None], 0.0)

        dh[:, :, span, :] = (scores @ weight64).view_as(hidden[:, :, span, :])
        dw.addmm_(scores.T, h.double())
        del rows, scores

    # Every rank replays the same local batch. GFSDP SUM-reduces dWeight,
    # while dHidden and the returned losses remain rank-local.
    dw.mul_(metadata["world"])
    expected_components = dict(zip(COMPONENT_NAMES, losses.tolist(), strict=True))
    total = sum(loss * weight for loss, weight in zip(expected_components.values(), weights, strict=True))
    return {
        "metadata": metadata,
        "dhidden": dh.cpu(),
        "dweight": dw.cpu(),
        "components": expected_components,
        "total": total,
    }


def precompute(record, chunks, world, refdir, device):
    """Compute all ranks' exact references once, before running timed candidates."""
    metadata = _fixture_metadata(record, chunks, world)
    path = _cache_path(metadata, refdir)
    if path.is_file():
        _load_cache(path, metadata)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    values = _reference(record, metadata, device)
    temp = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        torch.save(values, temp)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    return path


def _load_cache(path, metadata):
    expected = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if expected["metadata"] != metadata:
        raise ValueError(f"Oracle cache metadata differs from the current fixture at {path}")
    return expected


def _error_metrics(error_sq, norm_sq, max_abs):
    if not all(map(math.isfinite, (error_sq, norm_sq, max_abs))):
        raise ValueError("Nonfinite output or reference in the head oracle")
    return {
        "max_abs": max_abs,
        "relative_l2": math.sqrt(error_sq / norm_sq) if norm_sq else (None if error_sq else 0.0),
    }


def _tensor_error(actual, expected):
    if actual.shape != expected.shape:
        raise ValueError(f"Oracle shape mismatch: actual {actual.shape}, expected {expected.shape}")
    width = actual.shape[-1]
    rows_per_block = max(1, (1 << 20) // width)
    blocks = actual.reshape(-1, width).split(rows_per_block)
    reference_blocks = expected.view(-1, width).split(rows_per_block)
    sums = actual.new_zeros(2, dtype=torch.float64)
    maximum = actual.new_zeros((), dtype=torch.float64)
    for block, reference in zip(blocks, reference_blocks, strict=True):
        reference = reference.to(actual.device)
        delta = block.double() - reference
        sums += torch.stack((delta.square().sum(), reference.square().sum()))
        maximum = torch.maximum(maximum, delta.abs().amax())
    return _error_metrics(*sums.tolist(), maximum.item())


def _scalar_error(actual, expected):
    actual = float(actual)
    delta = abs(actual - expected)
    return {"actual": actual, "reference": expected, **_error_metrics(delta**2, expected**2, delta)}


@torch.no_grad()
def compare(record, chunks, rank, world, hidden_grad, weight_grad, components, total, refdir):
    """Compare every element of local dHidden and the rank's reduced dWeight shard."""
    if not 0 <= rank < world:
        raise ValueError(f"Rank {rank} outside oracle world of size {world}")
    if hidden_grad.dtype != torch.bfloat16 or weight_grad.dtype != torch.float32:
        raise ValueError("Expected BF16 dHidden and the reduced FP32 dWeight shard")
    metadata = _fixture_metadata(record, chunks, world)
    path = _cache_path(metadata, refdir)
    expected = _load_cache(path, metadata)
    shard_rows = metadata["weight_shape"][0] // world
    weight_ref = expected["dweight"].narrow(0, rank * shard_rows, shard_rows)
    return {
        "reference_cache": str(path),
        "dhidden": _tensor_error(hidden_grad, expected["dhidden"]),
        "dweight_shard": _tensor_error(weight_grad, weight_ref),
        **{name: _scalar_error(components[name], expected["components"][name]) for name in COMPONENT_NAMES},
        "total": _scalar_error(total, expected["total"]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=SHAPES)
    parser.add_argument("chunks", type=int)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("refdir", type=Path)
    parser.add_argument("--world", default=8, type=int)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    print(precompute(load_inputs(args.model, args.fixture), args.chunks, args.world, args.refdir, "cuda"), flush=True)
