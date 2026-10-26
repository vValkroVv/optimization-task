"""Derived diagnostics identify compact YAML fields and their winning sources."""

import copy
import pickle
from pathlib import Path
from typing import Any

from torchtitan.config.cli import load_document
from torchtitan.config.configurable import config, Configurable, to_plain
from torchtitan.config.decode import decode
from torchtitan.config.pre_submit import run_pre_submit
from torchtitan.config.save import dump_yaml
from torchtitan.config.schema.models.common.embeddings.token_embeddings import (
    EmbeddingConfig,
)
from torchtitan.config.sections import ModelSection
from torchtitan.config.validation import issue_for_field


@config
class ListOwnerConfig(Configurable.Config):
    values: list[int]


def _importing(tmp_path, name, *, imports, model=None):
    path = tmp_path / name
    data: dict[str, Any] = {
        "config_version": 1,
        "imports": [str(item) for item in imports],
    }
    if model is not None:
        data["model"] = {"config": model}
    path.write_text(dump_yaml(data))
    return path


def test_tp_head_error_points_to_compact_path_and_winning_file(tmp_path):
    base = Path("configs/experiments/llama3_debug.yaml").resolve()
    first = _importing(
        tmp_path,
        "first.yaml",
        imports=[base],
        model={"blocks": {"dense": {"attention": {"n_heads": 8}}}},
    )
    entry = _importing(
        tmp_path,
        "entry.yaml",
        imports=[first],
        model={"blocks": {"dense": {"attention": {"n_heads": 16}}}},
    )
    result = run_pre_submit(str(entry), [("parallelism.tensor_parallel_degree", "3")])
    issue = next(item for item in result.report.errors if item.path.endswith("n_heads"))
    assert issue.path == "model.config.blocks.dense.attention.n_heads"
    assert issue.runtime_path == "model.layers.0.attention.n_heads"
    assert issue.source is not None
    assert issue.source.endswith("entry.yaml")
    assert "source:" in str(issue)


def test_cli_origin_wins_over_imported_file():
    result = run_pre_submit(
        "configs/experiments/llama3_debug.yaml",
        [
            ("parallelism.tensor_parallel_degree", "3"),
            ("model.config.blocks.dense.attention.n_heads", "16"),
        ],
    )
    issue = next(item for item in result.report.errors if item.path.endswith("n_heads"))
    assert issue.source == "CLI override"


def test_vocab_warning_points_to_original_yaml_field():
    result = run_pre_submit(
        "configs/experiments/llama3_debug.yaml",
        [
            ("parallelism.tensor_parallel_degree", "3"),
            ("model.config.vocab_size", "256"),
        ],
    )
    issue = next(
        item for item in result.report.warnings if "loss parallel" in item.message
    )
    assert issue.path == "model.config.vocab_size"
    assert issue.runtime_path == "model.tok_embeddings.num_embeddings"
    assert issue.source == "CLI override"


def test_repeated_unchanged_import_retains_the_actual_value_source(tmp_path):
    base = Path("configs/experiments/llama3_debug.yaml").resolve()
    shared = _importing(
        tmp_path,
        "shared.yaml",
        imports=[base],
        model={"blocks": {"dense": {"attention": {"n_heads": 16}}}},
    )
    entry = _importing(tmp_path, "entry.yaml", imports=[shared, shared])
    result = run_pre_submit(str(entry), [("parallelism.tensor_parallel_degree", "3")])
    issue = next(item for item in result.report.errors if item.path.endswith("n_heads"))
    assert issue.source is not None
    assert issue.source.endswith("shared.yaml")


def test_derived_renamed_field_does_not_guess_from_equal_values():
    document, _ = load_document(
        "configs/experiments/llama3_debug.yaml",
        [
            ("model.config.dim", "256"),
            ("model.config.vocab_size", "256"),
        ],
    )
    owner = decode(
        ModelSection, document.data["model"], path="model", origins=document.origins
    ).config
    embedding = EmbeddingConfig.derive(
        owner,
        field_sources={
            "num_embeddings": "vocab_size",
            "embedding_dim": "dim",
        },
    )
    issue = issue_for_field(
        embedding,
        key="num_embeddings",
        path="model.tok_embeddings.num_embeddings",
        message="bad vocab",
    )
    assert issue.path == "model.config.vocab_size"
    dim_issue = issue_for_field(
        embedding,
        key="embedding_dim",
        path="model.tok_embeddings.embedding_dim",
        message="bad width",
    )
    assert dim_issue.path == "model.config.dim"
    assert issue.source == "CLI override"


def test_equal_values_from_different_files_keep_their_own_sources(tmp_path):
    base = Path("configs/experiments/llama3_debug.yaml").resolve()
    width = _importing(tmp_path, "width.yaml", imports=[base], model={"dim": 256})
    vocab = _importing(
        tmp_path, "vocab.yaml", imports=[width], model={"vocab_size": 256}
    )
    document, _ = load_document(str(vocab))
    owner = decode(
        ModelSection, document.data["model"], path="model", origins=document.origins
    ).config
    embedding = EmbeddingConfig.derive(
        owner, field_sources={"num_embeddings": "vocab_size", "embedding_dim": "dim"}
    )
    vocab_issue = issue_for_field(
        embedding,
        key="num_embeddings",
        path="model.tok_embeddings.num_embeddings",
        message="bad vocab",
    )
    dim_issue = issue_for_field(
        embedding,
        key="embedding_dim",
        path="model.tok_embeddings.embedding_dim",
        message="bad width",
    )
    assert vocab_issue.source is not None
    assert dim_issue.source is not None
    assert vocab_issue.source.endswith("vocab.yaml")
    assert dim_issue.source.endswith("width.yaml")


def test_inherited_norm_field_reports_its_actual_default_source():
    document, _ = load_document(
        "configs/experiments/gift_4_debug.yaml", [("model.config.norm.eps", "1e-5")]
    )
    owner = decode(
        ModelSection, document.data["model"], path="model", origins=document.origins
    ).config
    issue = issue_for_field(
        owner.blocks["mla_moe"].attention.kv_norm,
        key="eps",
        path="model.layers.2.attention.kv_norm.eps",
        message="bad eps",
    )
    assert issue.path == "model.config.blocks.mla_moe.attention.kv_norm.eps"
    assert issue.source.endswith("gift_3_5/debugmodel.yaml")


def test_source_metadata_survives_copies_without_affecting_serialization():
    document, _ = load_document("configs/experiments/llama3_debug.yaml")
    owner = decode(
        ModelSection, document.data["model"], path="model", origins=document.origins
    ).config
    original = to_plain(owner)
    for copied in (copy.deepcopy(owner), owner._copy()):
        assert copied._source_origins is owner._source_origins
        attention = copied.blocks["dense"].attention
        assert attention._source_path == "model.config.blocks.dense.attention"
        assert to_plain(copied) == original
    assert all(not key.startswith("_source") for key in original)


def test_resolved_reload_rebases_origin_to_the_resolved_file(tmp_path):
    document, _ = load_document("configs/experiments/llama3_debug.yaml")
    resolved = tmp_path / "resolved.yaml"
    resolved.write_text(dump_yaml(document.data))
    result = run_pre_submit(
        str(resolved), [("parallelism.tensor_parallel_degree", "3")]
    )
    issue = next(item for item in result.report.errors if item.path.endswith("n_heads"))
    assert issue.source is not None
    assert issue.source.endswith("resolved.yaml")


def test_origin_metadata_uses_canonical_list_paths_and_drops_replaced_descendants(
    tmp_path,
):
    path = tmp_path / "entry.yaml"
    path.write_text("values: [{a: 1, b: 2}]\n")
    from torchtitan.config.loader import load_raw_document

    document = load_raw_document(str(path), overrides=[{"values": [{"a": 3}]}])
    assert document.origins["values[0].a"] == "CLI override"
    assert "values[0].b" not in document.origins


def test_immutable_source_map_is_safe_for_copies_and_worker_serialization():
    from torchtitan.config.provenance import SourceMap

    values = {"model.config.dim": "model.yaml"}
    source = SourceMap(values)
    values["model.config.dim"] = "changed.yaml"
    assert source["model.config.dim"] == "model.yaml"
    assert copy.deepcopy(source) is source
    assert dict(pickle.loads(pickle.dumps(source))) == dict(source)


def test_deriving_semantic_fields_does_not_retain_mutable_owner_values():
    owner = ListOwnerConfig(values=[1, 2])
    derived = ListOwnerConfig.derive(owner, field_sources={"values": "values"})
    derived.values.append(3)
    assert owner.values == [1, 2]
