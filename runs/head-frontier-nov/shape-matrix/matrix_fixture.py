"""Read native CE captures or prepare a measurement fixture from real tokenizer rows."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

SHAPES = {"100b": (2, 4096, 0.3), "700b": (3, 7168, 0.1)}
SEQ = 4096
VOCAB = 128256


def load_inputs(model, fixture, extra_fixture=None):
    payload = torch.load(fixture, map_location="cpu", weights_only=True, mmap=True)
    record = dict(payload["ce"])
    b, d, aux = SHAPES[model]
    assert record["hidden"].shape == (2, b, SEQ, d)
    assert record["weight"].shape == (VOCAB, d)
    assert record["labels"].shape == (b, SEQ)
    assert record["head_weights"] == [1.0, aux]
    record["provenance"] = payload.get("provenance", "native GIFT CE capture")
    if extra_fixture:
        z = torch.load(extra_fixture, weights_only=True, map_location="cpu", mmap=True)
        record["z_hidden"] = z["mtp"]["hidden"]
    return record


def make(model, dataset_dir, output):
    from streaming import LocalDataset

    b, d, aux = SHAPES[model]
    data = LocalDataset(local=str(dataset_dir), split="train")
    with (dataset_dir / "data-002-rows.jsonl").open() as manifest:
        rows = [json.loads(next(manifest)) for _ in range(b)]
    ids = []
    for row in rows:
        value = data[row["index"]]["tokens"]
        assert hashlib.sha256(value).hexdigest() == row["sha256"]
        ids.append(np.frombuffer(value, dtype="<u4").astype(np.int64))
    tokens = torch.from_numpy(np.stack(ids))
    assert tokens.shape == (b, SEQ)
    labels = torch.full_like(tokens, -100)
    labels[:, :-1] = tokens[:, 1:]
    indices = torch.arange(SEQ, dtype=torch.int32).expand(b, -1)
    starts = torch.zeros_like(indices)
    starts[:, 1:] = torch.where(tokens[:, :-1] == 2, indices[:, 1:], 0)
    positions = indices - starts.cummax(1).values
    gen = torch.Generator().manual_seed(2048)

    def normal(shape, std):
        out = torch.empty(shape, dtype=torch.bfloat16)
        flat = out.view(-1)
        for tile in flat.split(1 << 20):
            tile.copy_(torch.randn(tile.shape, generator=gen).mul_(std))
        return out

    record = dict(hidden=normal((2, b, SEQ, d), 1.0), weight=normal((VOCAB, d), 0.006),
                  labels=labels, positions=positions, head_weights=[1.0, aux], ignore_index=-100,
                  denominator=float((labels != -100).sum().item() * 8 * 32))
    if model == "100b":
        record["z_hidden"] = normal((b, SEQ, d), 1.0)
    description = dict(input_type="real data-002 targets and positions; seeded synthetic hidden, z-hidden and weights",
                       seed=2048, rows=rows, model=model)
    temp = output.with_suffix(".partial")
    torch.save(dict(ce=record, provenance=description), temp)
    temp.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=SHAPES)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    make(args.model, args.dataset_dir, args.output)
