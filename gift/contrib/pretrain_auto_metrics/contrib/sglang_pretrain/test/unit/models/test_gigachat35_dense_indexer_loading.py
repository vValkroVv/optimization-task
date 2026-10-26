import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


_MODEL_PATH = (
    Path(__file__).parents[3] / "python/sglang/srt/models/gigachat35.py"
)


def _load_weight_filter():
    tree = ast.parse(_MODEL_PATH.read_text(encoding="utf-8"), filename=str(_MODEL_PATH))
    source_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "Gigachat35HybridGDNWeightLoaderMixin"
    )
    helper = next(
        node
        for node in source_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_is_removed_dense_indexer_weight"
    )
    load_method = next(
        node
        for node in source_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "do_load_weights"
    )
    subset = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__",
                    names=[ast.alias(name="annotations")],
                    level=0,
                ),
                ast.ClassDef(
                    name="WeightLoaderSubset",
                    bases=[],
                    keywords=[],
                    body=[helper],
                    decorator_list=[],
                ),
            ],
            type_ignores=[],
        )
    )
    namespace = {}
    exec(compile(subset, str(_MODEL_PATH), "exec"), namespace)
    return namespace["WeightLoaderSubset"], load_method


@pytest.mark.unit
def test_dense_layer_skips_extra_checkpoint_indexer_weights():
    loader_class, load_method = _load_weight_filter()
    loader = loader_class()
    loader.config = SimpleNamespace(dense_attention_layers=[1])

    assert loader._is_removed_dense_indexer_weight(
        "model.layers.1.self_attn.indexer.wq_b.weight", 1
    )
    assert not loader._is_removed_dense_indexer_weight(
        "model.layers.3.self_attn.indexer.wq_b.weight", 3
    )
    assert not loader._is_removed_dense_indexer_weight(
        "model.layers.1.self_attn.q_b_proj.weight", 1
    )

    skip_branches = [
        node
        for node in ast.walk(load_method)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Call)
        and isinstance(node.test.func, ast.Attribute)
        and node.test.func.attr == "_is_removed_dense_indexer_weight"
        and any(isinstance(child, ast.Continue) for child in node.body)
    ]
    assert len(skip_branches) == 1
