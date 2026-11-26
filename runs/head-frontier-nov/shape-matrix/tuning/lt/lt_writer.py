"""Experimental FP32 dW writer and cuBLASLt candidate enumeration."""

import ctypes as c
from contextlib import ExitStack
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import weakref

import torch
from torchtitan.distributed.gfsdp import _staged_head_dw_writer as native


WORKSPACE_LIMIT = 32 << 20
PREFERENCE_CAPS = (WORKSPACE_LIMIT, 0)
LOGGER = logging.getLogger(__name__)
CONFIG_NAMES = ("id", "tile", "split_k", "reduction", "swizzle", "custom", "stages", "inner", "cluster")


@dataclass
class Candidate:
    order: str
    layout: object
    algorithm: object
    workspace: int
    config: dict
    waves: float
    rank: int
    preference_cap: int

    def receipt(self):
        return {
            "order": self.order,
            "workspace_bytes": self.workspace,
            "config": self.config,
            "waves": self.waves,
            "heuristic_rank": self.rank,
            "preference_cap": self.preference_cap,
            "algorithm": bytes(self.algorithm).hex(),
        }


class Search(native._CublasLtMatmul):
    """Reuse the native C ABI, descriptors and in-place C/D contract."""

    def __init__(self, shape, alignments, stream, resources):
        super().__init__(None, *shape, stream)
        self.layouts = {
            order: self._layout(order == "row", *shape, resources)
            for order in ("column", "row")
        }
        self.alignments = alignments
        library = c.CDLL(f"libcublasLt.so.{torch.version.cuda.split('.')[0]}")
        self.algo_check = library.cublasLtMatmulAlgoCheck
        self.algo_check.argtypes = [c.c_void_p] * 8
        self.algo_check.restype = c.c_int

    def candidates(self, count=32):
        results = {}
        for order, layout in self.layouts.items():
            alignments = self.alignments if order == "row" else (
                self.alignments[1], self.alignments[0], self.alignments[2]
            )
            for cap in PREFERENCE_CAPS:
                for candidate in self._query(layout, order, (*alignments, alignments[2]), cap, count):
                    key = (candidate.order, bytes(candidate.algorithm))
                    results.setdefault(key, candidate)
        return list(results.values())

    def _query(self, layout, order, alignments, cap, count):
        with ExitStack() as resources:
            preference = native._Ptr()
            native._check(self.pref_create, c.byref(preference))
            resources.callback(native._check, self.pref_destroy, preference)
            native._set_attribute(self.pref_attr, preference, native._PREF_MAX_WORKSPACE_BYTES, native._U64(cap))
            native._set_attribute(
                self.pref_attr, preference, native._PREF_REDUCTION_SCHEME_MASK,
                c.c_uint32(native._FP32_REDUCTION_SCHEMES),
            )
            for offset, alignment in enumerate(alignments):
                native._set_attribute(
                    self.pref_attr, preference, native._PREF_MIN_ALIGNMENT_A_BYTES + offset,
                    c.c_uint32(alignment),
                )
            results, returned = (native._HeuristicResult * count)(), c.c_int()
            native._check(
                self.heuristic, self.handle, layout.operation, layout.a, layout.b, layout.cd,
                layout.cd, preference, count, results, c.byref(returned),
            )
            for rank, result in enumerate(results[:returned.value]):
                if result.state == 0 and result.workspace_bytes <= cap:
                    try:
                        yield self._candidate(order, result.algo, rank, cap)
                    except RuntimeError as error:
                        LOGGER.warning("Discarded %s heuristic %d at workspace cap %d: %s",
                                       order, rank, cap, error)

    def _candidate(self, order, algorithm, rank, cap):
        layout = self.layouts[order]
        checked = native._HeuristicResult()
        native._check(
            self.algo_check, self.handle, layout.operation, layout.a, layout.b,
            layout.cd, layout.cd, c.byref(algorithm), c.byref(checked),
        )
        if checked.state or checked.workspace_bytes > cap:
            raise RuntimeError(
                f"cuBLASLt algorithm check workspace={checked.workspace_bytes}, state={checked.state}"
            )
        config = {}
        for attr, name in enumerate(CONFIG_NAMES):
            value = c.c_uint16() if attr >= 7 else c.c_uint32()
            native._check(self.config_attr, c.byref(algorithm), attr, c.byref(value), c.sizeof(value), None)
            config[name] = value.value
        if config["reduction"] not in (0, 1, 2):
            raise RuntimeError(f"Unexpected cuBLASLt reduction scheme {config['reduction']}")
        return Candidate(
            order, layout, algorithm, checked.workspace_bytes, config, checked.waves, rank, cap,
        )

    def restore(self, entry):
        algorithm = native._Algo.from_buffer_copy(bytes.fromhex(entry["algorithm"]))
        candidate = self._candidate(
            entry["order"], algorithm, entry["heuristic_rank"], entry["preference_cap"],
        )
        if candidate.config != entry["config"] or candidate.workspace != entry["workspace_bytes"]:
            raise RuntimeError("cuBLASLt algorithm receipt does not match the available configuration")
        return candidate

    def launch(self, candidate, grad, hidden, slot, add, workspace):
        layout = candidate.layout
        a, b = (grad, hidden) if layout.row_major else (hidden, grad)
        native._check(
            self.matmul, self.handle, layout.operation, c.byref(native._ONE),
            a.data_ptr(), layout.a, b.data_ptr(), layout.b, c.byref(native._BETA[add]),
            slot.data_ptr(), layout.cd, slot.data_ptr(), layout.cd, c.byref(candidate.algorithm),
            workspace.data_ptr() if candidate.workspace else None,
            candidate.workspace, self.stream.cuda_stream,
        )


def operand_key(grad, hidden, slot):
    rows, vocab = grad.shape
    width = hidden.size(1)
    tensors = (grad, hidden, slot)
    if (
        min(rows, vocab, width) <= 0
        or hidden.shape != (rows, width) or slot.shape != (vocab, width)
        or any(t.device != slot.device or not t.is_contiguous() for t in tensors)
        or (grad.dtype, hidden.dtype, slot.dtype) != (torch.bfloat16, torch.bfloat16, torch.float32)
        or not slot.is_cuda
    ):
        raise ValueError("Expected contiguous CUDA BF16 [K,V]/[K,H] inputs and FP32 [V,H] output")
    alignments = tuple(min(256, ptr & -ptr) for ptr in (t.data_ptr() for t in tensors))
    return (rows, vocab, width), alignments


def workspace_for(device, *candidates):
    size = max(candidate.workspace for candidate in candidates)
    return torch.empty(size, dtype=torch.uint8, device=device) if size else None


class ReplayWriter:
    name = "cublasLtMatmul-replay"
    pair = "screened-per-beta"

    def __init__(self, receipt):
        data = json.loads(Path(receipt).read_text()) if isinstance(receipt, (str, Path)) else receipt
        if data["format"] == 1:
            entries = [data]
        elif data["format"] == 2:
            entries = data["plans"]
        else:
            raise ValueError(f"Unknown cuBLASLt receipt format {data['format']!r}")
        self.expected = {}
        for entry in entries:
            shape = tuple(entry["shape"])
            if entry["format"] != 1 or len(entry["pair"]) != 2 or shape in self.expected:
                raise ValueError(f"Expected one plan for each beta and no duplicate shape, got {shape!r}")
            self.expected[shape] = entry
        versions = {entry["cublasLt_version"] for entry in entries}
        self.version = next(iter(versions)) if len(versions) == 1 else None
        self.bound = None
        self.resources = ExitStack()
        self.cleanup = weakref.finalize(self, self.resources.close)
        self.plan_cache = {}
        self.calls = [0, 0]

    def _plan(self, shape, alignments, stream):
        if self.bound is None:
            self.bound = stream
        elif stream != self.bound:
            raise RuntimeError("cuBLASLt dW workspace must remain on its original stream")

        expected = self.expected.get(shape)
        compatible = (
            expected is not None
            and all(a >= b for a, b in zip(alignments, expected["alignments"], strict=True))
            and list(torch.cuda.get_device_capability(stream.device)) == expected["capability"]
            and torch.cuda.get_device_name(stream.device) == expected["device_name"]
        )
        if compatible:
            with ExitStack() as resources:
                search = Search(shape, alignments, stream, resources)
                if search.version == expected["cublasLt_version"]:
                    try:
                        pair = tuple(search.restore(item) for item in expected["pair"])
                    except RuntimeError as error:
                        LOGGER.warning("Falling back for cuBLASLt dW shape %s: %s", shape, error)
                    else:
                        workspace = workspace_for(stream.device, *pair)
                        self.resources.callback(resources.pop_all().close)
                        return search, pair, workspace
        return native.make_head_dw_writer("generic", 0, *shape, stream), None, None

    def __call__(self, grad, hidden, slot, add, stream=None):
        stream = stream if stream is not None else torch.cuda.current_stream(slot.device)
        if stream.device != slot.device or self.bound not in (None, stream):
            raise RuntimeError("cuBLASLt dW workspace must remain on its original CUDA stream")
        key = operand_key(grad, hidden, slot)
        plan = self.plan_cache.get(key)
        if plan is None:
            with torch.cuda.stream(stream):
                plan = self._plan(*key, stream)
            self.plan_cache[key] = plan
        search, pair, workspace = plan
        if pair is None:
            search(grad, hidden, slot, add, stream)
        else:
            search.launch(pair[add], grad, hidden, slot, add, workspace)
            self.calls[add] += 1

    def receipt(self):
        return {
            "route": self.name,
            "cublasLt_version": self.version,
            "tuned_calls_beta0_beta1": self.calls.copy(),
            "plans": [
                {"shape": shape, "alignments": alignments,
                 "pair": [item.receipt() for item in pair] if pair is not None else "native-generic"}
                for (shape, alignments), (_, pair, _) in self.plan_cache.items()
            ],
        }

    def close(self):
        self.cleanup()
        self.plan_cache.clear()


def make_dw_writer(policy="replay", *, receipt):
    if policy != "replay":
        raise ValueError(f"Unsupported cuBLASLt replay policy {policy!r}")
    return ReplayWriter(receipt)
