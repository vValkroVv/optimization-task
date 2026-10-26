# Copyright 2024 LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
"""Static regression checks for ScMoE layer-index plumbing.

The papaya_mix decoder passes ``layer_idx`` into every MoE implementation.
Keep this test torch-free: importing the real ScMoE module pulls CUDA-heavy expert
dependencies, while the constructor contract can be checked directly from AST.
"""

import ast
import re
from pathlib import Path


_MOE_LAYERS = (
    Path(__file__).resolve().parents[3]
    / "llmfoundry"
    / "models"
    / "layers"
    / "moe"
    / "moe_layers.py"
)
_BLOCKS = (
    Path(__file__).resolve().parents[3]
    / "llmfoundry"
    / "models"
    / "layers"
    / "blocks.py"
)
_EXPERTS = (
    Path(__file__).resolve().parents[3]
    / "llmfoundry"
    / "models"
    / "layers"
    / "moe"
    / "experts.py"
)


def _scmoe_class() -> ast.ClassDef:
    module = ast.parse(_MOE_LAYERS.read_text())
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "ScMoEBlock":
            return node
    raise AssertionError("ScMoEBlock class not found")


def _init_method(cls: ast.ClassDef) -> ast.FunctionDef:
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            return node
    raise AssertionError("ScMoEBlock.__init__ not found")


def test_scmoe_accepts_layer_idx_from_decoder_block():
    init = _init_method(_scmoe_class())
    arg_names = [arg.arg for arg in init.args.args + init.args.kwonlyargs]
    assert "layer_idx" in arg_names


def test_scmoe_uses_layer_idx_for_moe_included_projection_z_loss():
    cls = _scmoe_class()
    src = ast.get_source_segment(_MOE_LAYERS.read_text(), cls)
    assert "resolve_included_proj_coef" in src
    assert '"up_moe"' in src
    assert '"gate_moe"' in src
    assert "self.layer_idx" in src


def test_decoder_block_passes_layer_idx_to_dense_and_shared_mlps():
    src = _BLOCKS.read_text()
    assert "build_ffn(config.tp_size, config, layer_idx=layer_idx)" in src
    assert "SharedExpertLlamaMLP(config, layer_idx=layer_idx)" in src
    assert re.search(
        r"SharedExpertParallelLlamaMLP\(\s*config,\s*layer_idx=layer_idx\s*\)",
        src,
    )


def test_shared_experts_forward_layer_idx_to_mlp_base_classes():
    src = _EXPERTS.read_text()
    assert "def __init__(self, config, layer_idx: int | None = None)" in src
    assert src.count("layer_idx=layer_idx") >= 2
