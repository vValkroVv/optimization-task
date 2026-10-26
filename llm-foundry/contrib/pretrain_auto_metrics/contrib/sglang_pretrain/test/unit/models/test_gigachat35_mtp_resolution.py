"""Config-level resolution matrix for the Gigachat35 MTP/NextN DSA mode.

Exercises the REAL `ModelConfig._resolve_gigachat35_mtp_dsa`, its checkpoint
scan (`_gigachat35_ckpt_has_mtp_indexer`) and `gigachat35_mtp_is_dsa` /
`is_deepseek_dsa` from `configs/model_config.py`. Importing the module normally
would pull the full sglang runtime, so -- like test_gigachat35.py -- we parse the
source, keep only the functions/methods under test, and exec them against a light
namespace. The scan reads on-disk weight manifests, so each case builds a tmp
checkpoint dir in the layout it wants to cover.
"""
import ast
import json
import os
import struct
import textwrap
import types
import typing
from pathlib import Path

import pytest

_MC_SRC = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/configs/model_config.py"
).read_text(encoding="utf-8")

_FUNCS = {"_hf_attr", "_hf_arch", "_unwrap_dsa_config", "is_deepseek_dsa",
          "gigachat35_mtp_is_dsa"}
_METHODS = {"_gigachat35_ckpt_has_mtp_indexer", "_resolve_gigachat35_mtp_dsa"}


def _load_resolver_ns():
    tree = ast.parse(_MC_SRC)
    func_src, method_src = {}, {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in _FUNCS:
            func_src[node.name] = ast.get_source_segment(_MC_SRC, node)
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for m in node.body:
                if isinstance(m, ast.FunctionDef) and m.name in _METHODS:
                    method_src[m.name] = textwrap.dedent(ast.get_source_segment(_MC_SRC, m))
    missing = (_FUNCS - set(func_src)) | (_METHODS - set(method_src))
    assert not missing, f"model_config.py is missing expected symbols: {sorted(missing)}"

    warnings: list = []
    ns = {
        "os": os, "json": json, "typing": typing, "Optional": typing.Optional,
        "logger": types.SimpleNamespace(warning=lambda *a, **k: warnings.append(a),
                                        info=lambda *a, **k: None),
        "PretrainedConfig": lambda **kw: types.SimpleNamespace(**kw),
        "_CONFIG_CLASS_TO_ARCHITECTURE": {},
    }
    for src in list(func_src.values()) + list(method_src.values()):
        exec(src, ns)  # noqa: S102 -- our own repo source, under test
    return ns, warnings


_NS, _WARNINGS = _load_resolver_ns()


class _MC:
    _gigachat35_ckpt_has_mtp_indexer = _NS["_gigachat35_ckpt_has_mtp_indexer"]
    _resolve_gigachat35_mtp_dsa = _NS["_resolve_gigachat35_mtp_dsa"]

    def __init__(self, hf_config, model_path):
        self.hf_config = hf_config
        self.model_path = model_path


def _cfg(**kw):
    kw.setdefault("architectures", ["Gigachat35ForCausalLMNextN"])
    kw.setdefault("num_hidden_layers", 40)
    kw.setdefault("index_topk", 2048)
    kw.setdefault("nextn_is_dsa", None)
    return types.SimpleNamespace(**kw)


def _write_st_index(d, keys):
    json.dump({"weight_map": {k: "shard-0.safetensors" for k in keys}},
              open(os.path.join(d, "model.safetensors.index.json"), "w"))


def _write_bin_index(d, keys):
    json.dump({"weight_map": {k: "pytorch_model-0.bin" for k in keys}},
              open(os.path.join(d, "pytorch_model.bin.index.json"), "w"))


def _write_single_safetensors(d, keys):
    header = {k: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]} for k in keys}
    blob = json.dumps(header).encode()
    with open(os.path.join(d, "model.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(b"\x00\x00")


def _mtp_keys(has_indexer, n=40):
    keys = [f"model.layers.{i}.self_attn.kv_a_proj.weight" for i in range(n + 1)]
    if has_indexer:
        keys.append(f"model.layers.{n}.self_attn.indexer.wk.weight")
    return keys


def _resolve(tmp_path, *, nextn_is_dsa=None, has_indexer=True, index_topk=2048,
             layout="st_index", write=True):
    d = str(tmp_path)
    if write:
        keys = _mtp_keys(has_indexer)
        {"st_index": _write_st_index, "bin_index": _write_bin_index,
         "single": _write_single_safetensors}[layout](d, keys)
    cfg = _cfg(nextn_is_dsa=nextn_is_dsa, index_topk=index_topk)
    _WARNINGS.clear()
    _MC(cfg, d)._resolve_gigachat35_mtp_dsa()
    return cfg


# --- explicit nextn_is_dsa -------------------------------------------------
def test_explicit_true_with_indexer(tmp_path):
    cfg = _resolve(tmp_path, nextn_is_dsa=True, has_indexer=True)
    assert cfg.nextn_is_dsa is True
    assert _NS["is_deepseek_dsa"](cfg) is True


def test_explicit_true_without_indexer_raises(tmp_path):
    with pytest.raises(ValueError):
        _resolve(tmp_path, nextn_is_dsa=True, has_indexer=False)


def test_explicit_false_with_indexer_warns(tmp_path):
    cfg = _resolve(tmp_path, nextn_is_dsa=False, has_indexer=True)
    assert cfg.nextn_is_dsa is False
    assert _WARNINGS, "expected a warning about ignoring the trained indexer"


def test_explicit_false_without_indexer(tmp_path):
    cfg = _resolve(tmp_path, nextn_is_dsa=False, has_indexer=False)
    assert cfg.nextn_is_dsa is False


# --- unset -> follow the checkpoint scan -----------------------------------
def test_unset_follows_indexer_present(tmp_path):
    cfg = _resolve(tmp_path, nextn_is_dsa=None, has_indexer=True)
    assert cfg.nextn_is_dsa is True


def test_unset_follows_indexer_absent(tmp_path):
    cfg = _resolve(tmp_path, nextn_is_dsa=None, has_indexer=False)
    assert cfg.nextn_is_dsa is False


def test_unset_no_manifest_raises(tmp_path):
    with pytest.raises(ValueError):
        _resolve(tmp_path, nextn_is_dsa=None, write=False)


# --- no DSA anywhere -------------------------------------------------------
def test_index_topk_none_is_dense(tmp_path):
    cfg = _resolve(tmp_path, nextn_is_dsa=None, index_topk=None, write=False)
    assert cfg.nextn_is_dsa is False


# --- scan covers every local weight layout ---------------------------------
@pytest.mark.parametrize("layout", ["st_index", "bin_index", "single"])
def test_scan_detects_indexer_across_layouts(tmp_path, layout):
    cfg = _resolve(tmp_path, nextn_is_dsa=None, has_indexer=True, layout=layout)
    assert cfg.nextn_is_dsa is True


@pytest.mark.parametrize("layout", ["st_index", "bin_index", "single"])
def test_scan_detects_absent_indexer_across_layouts(tmp_path, layout):
    cfg = _resolve(tmp_path, nextn_is_dsa=None, has_indexer=False, layout=layout)
    assert cfg.nextn_is_dsa is False
