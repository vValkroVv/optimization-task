"""DenseSplitAttnBackend.__getattr__ forwards only an explicit allowlist to the
DSA backend, so a typo raises AttributeError instead of silently returning the
wrong backend's value (feature-detection via getattr stays honest).

AST-loaded so the test needs no GPU / heavy sglang import.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).parents[3]
_SRC = _ROOT / "python/sglang/srt/layers/attention/dense_split_attn_backend.py"


def _load_getattr_and_allowlist():
    tree = ast.parse(_SRC.read_text(encoding="utf-8"))
    allowlist_assign = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "_FORWARDED_TO_DSA"
            for t in node.targets
        )
    )
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DenseSplitAttnBackend"
    )
    getattr_fn = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__getattr__"
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[allowlist_assign, getattr_fn], type_ignores=[])
    )
    ns = {"frozenset": frozenset}
    exec(compile(module, str(_SRC), "exec"), ns)
    return ns["__getattr__"], ns["_FORWARDED_TO_DSA"]


_getattr, _ALLOWLIST = _load_getattr_and_allowlist()


class _Wrapper:
    __getattr__ = _getattr

    def __init__(self, dsa_backend):
        object.__setattr__(self, "dsa_backend", dsa_backend)


@pytest.mark.unit
def test_forwards_allowlisted_attrs_to_dsa_backend():
    dsa = SimpleNamespace(**{name: name + "_val" for name in _ALLOWLIST})
    w = _Wrapper(dsa)
    for name in _ALLOWLIST:
        assert getattr(w, name) == name + "_val"


@pytest.mark.unit
def test_unknown_attr_raises_instead_of_forwarding():
    dsa = SimpleNamespace(enable_chunk_kv=True, enable_chunk_kvv="typo_target")
    w = _Wrapper(dsa)
    assert getattr(w, "enable_chunk_kv") is True
    with pytest.raises(AttributeError):
        _ = w.enable_chunk_kvv  # typo must not resolve to the DSA backend


@pytest.mark.unit
def test_feature_detection_attrs_are_forwarded():
    # getattr(backend, "enable_chunk_kv", ...) / "pack_prefix_chunk_kv" must reach
    # the DSA backend, and a missing one falls back to the default (not this wrapper).
    for name in ("enable_chunk_kv", "pack_prefix_chunk_kv", "use_mha"):
        assert name in _ALLOWLIST
    w = _Wrapper(SimpleNamespace())  # dsa_backend has none of them
    assert getattr(w, "enable_chunk_kv", "default") == "default"


@pytest.mark.unit
def test_sub_backend_names_are_not_self_forwarded():
    # dsa_backend / dense_backend are real instance attrs; they must never be
    # served by __getattr__ (would recurse / hide a construction bug).
    assert "dsa_backend" not in _ALLOWLIST
    assert "dense_backend" not in _ALLOWLIST
