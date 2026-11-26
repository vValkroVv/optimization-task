"""Read pinned cuBLASLt inputs and identify routes expected for a grid cell."""

import hashlib
import json
from pathlib import Path


ENV_FILES = {
    "dw": "HEAD_FULL_GRID_DW_CATALOG",
    "linear": "HEAD_FULL_GRID_LINEAR_MANIFEST",
}
WIDTH = {"100b": 4096, "700b": 7168}
VOCAB = 128256


def pin(path, digest):
    path = Path(path).resolve(strict=True)
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"Unexpected SHA-256 for {path}")
    return {"path": str(path), "sha256": digest}


def read_pinned(entry):
    path = Path(entry["path"])
    if pin(path, entry["sha256"]) != entry:
        raise ValueError(f"Manifest has a different resolved path: {path}")
    return json.loads(path.read_text())


def dw_plans(catalog):
    if catalog["format"] != 2:
        raise ValueError("Expected a format 2 dW catalog")
    by_shape = {}
    for entry in catalog["plans"]:
        shape = tuple(entry["shape"])
        if shape in by_shape or entry["format"] != 1 or len(entry["pair"]) != 2:
            raise ValueError(f"Invalid or duplicate dW plan for {shape}")
        by_shape[shape] = entry
    return by_shape


def linear_shape(descriptor):
    m, n, k = descriptor[:3]
    if descriptor[10]:
        return (m, k, n)
    return (n, k, m)


def expected_routes(model, chunks, seq, dw, linear):
    rows = 2 * seq // chunks
    width = WIDTH[model]
    shapes = {"projection": (rows, width, VOCAB), "dhidden": (rows, VOCAB, width)}
    linear_routes = {op: [] for op in shapes}
    for key in linear:
        _, _, descriptor = json.loads(key)
        for op, shape in shapes.items():
            if descriptor[11] == 1 and linear_shape(descriptor) == shape:
                linear_routes[op].append(key)
    return {"dw": dw.get((rows, VOCAB, width)), "linear": linear_routes}
