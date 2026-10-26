import ast
import types
import typing
from pathlib import Path

import pytest

_SRC_PATH = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py"
)
_SRC = _SRC_PATH.read_text(encoding="utf-8")


class T:
    def __init__(self, rows, base=None, begin=0):
        self._base = base if base is not None else rows
        self._begin = begin
        self.shape = (len(rows), len(rows[0]) if rows else 0)
        self.dtype = "bf16"

    @property
    def rows(self):
        return self._base[self._begin : self._begin + self.shape[0]]

    def is_contiguous(self):
        return True

    def new_empty(self, shape):
        return T([[0.0] * shape[1] for _ in range(shape[0])])

    def write(self, rows):
        for i, row in enumerate(rows):
            self._base[self._begin + i] = list(row)

    def __getitem__(self, s):
        b, e, _ = s.indices(self.shape[0])
        return T(self.rows[b:e], self._base, self._begin + b)

    def __setitem__(self, s, other):
        b, _e, _ = s.indices(self.shape[0])
        rows = other.rows
        for i, row in enumerate(rows):
            self._base[self._begin + b + i] = list(row)


def _load(chunk_size, calls):
    node = next(
        n
        for n in ast.parse(_SRC).body
        if isinstance(n, ast.FunctionDef) and n.name == "fused_experts_impl"
    )

    def _prepare_fused_moe_run(hs, w1, w2, ti, **kw):
        calls.append(hs.shape[0])
        assert hs.shape[0] == ti.shape[0]
        return (None, None, False, None, None, None)

    def _fused_moe_kernel_sequence(hs, w1, w2, tw, ti, *args, inplace, **kw):
        out = [
            [x * 10.0 + tw.rows[i][0] + ti.rows[i][0] for x in row]
            for i, row in enumerate(hs.rows)
        ]
        if inplace:
            hs.write(out)
            return hs
        return T(out)

    ns = {
        "torch": types.SimpleNamespace(
            Tensor=object, float32="f32", float16="f16", bfloat16="bf16"
        ),
        "Optional": typing.Optional,
        "List": typing.List,
        "padding_size": 0,
        "_use_aiter": False,
        "envs": types.SimpleNamespace(
            SGLANG_FUSED_MOE_CHUNK_SIZE=types.SimpleNamespace(get=lambda: chunk_size)
        ),
        "_prepare_fused_moe_run": _prepare_fused_moe_run,
        "_fused_moe_kernel_sequence": _fused_moe_kernel_sequence,
    }
    exec(compile(ast.Module([node], []), "<fused_moe>", "exec"), ns)
    return ns["fused_experts_impl"]


def _inputs(num_tokens, hidden=4):
    hs = T([[float(t * hidden + j) for j in range(hidden)] for t in range(num_tokens)])
    tw = T([[0.5 * t] for t in range(num_tokens)])
    ti = T([[float(t % 3)] for t in range(num_tokens)])
    w = types.SimpleNamespace(shape=(3, 8, hidden), is_contiguous=lambda: True)
    return hs, tw, ti, w


@pytest.mark.unit
@pytest.mark.parametrize("num_tokens,chunk_size", [(10, 4), (10, 10), (10, 0), (9, 3)])
def test_chunked_matches_unchunked(num_tokens, chunk_size):
    calls = []
    hs, tw, ti, w = _inputs(num_tokens)
    ref = _load(0, [])(hs, w, w, tw, ti)
    hs2, tw2, ti2, _ = _inputs(num_tokens)
    got = _load(chunk_size, calls)(hs2, w, w, tw2, ti2)

    assert got.rows == ref.rows
    assert max(calls) <= (chunk_size or num_tokens)
    assert sum(calls) == num_tokens


@pytest.mark.unit
def test_inplace_writes_every_chunk():
    calls = []
    hs, tw, ti, w = _inputs(7)
    ref_hs, _, _, _ = _inputs(7)
    _load(0, [])(ref_hs, w, w, tw, ti, inplace=True)
    out = _load(3, calls)(hs, w, w, tw, ti, inplace=True)

    assert out is hs
    assert hs.rows == ref_hs.rows
    assert calls == [3, 3, 1]
