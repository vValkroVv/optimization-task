"""Service metadata and compact invariants must survive runtime materialization."""

import copy
from dataclasses import fields

import pytest

from torchtitan.config.cli import load_document
from torchtitan.config.decode import decode
from torchtitan.config.pre_submit import run_pre_submit
from torchtitan.config.schema.components.lora import FrozenConfig
from torchtitan.config.schema.models.common.regularization import HiddenZLoss
from torchtitan.config.sections import ModelSection


def _compact(path="configs/experiments/llama3_debug.yaml", overrides=()):
    document, _ = load_document(path, list(overrides))
    return decode(
        ModelSection, document.data["model"], path="model", origins=document.origins
    ).config


def test_metadata_is_not_a_runtime_configuration_field():
    compact = _compact()
    compact.blocks["dense"].attention._explicit_param_init_names = frozenset({"weight"})
    for node in (compact, compact.blocks["dense"].attention):
        assert not any(field.name.startswith("_source") for field in fields(node))
        assert "_explicit_param_init_names" not in {
            field.name for field in fields(node)
        }
        for cloned in (copy.copy(node), copy.deepcopy(node), node._copy()):
            assert cloned._source_path == node._source_path
            assert cloned._source_origins is node._source_origins
    assert copy.deepcopy(
        compact.blocks["dense"].attention
    )._explicit_param_init_names == frozenset({"weight"})


def test_frozen_metadata_and_copy_are_independent_from_inner():
    inner = _compact().blocks["dense"].attention
    wrapped = FrozenConfig(inner=inner)
    wrapped._source_path = "wrapper"
    assert inner._source_path == "model.config.blocks.dense.attention"
    for cloned in (wrapped._copy(), copy.deepcopy(wrapped)):
        assert cloned._source_path == "wrapper"
        assert cloned.inner._source_path == inner._source_path


def test_frozen_wrapper_respects_new_service_properties_without_a_name_allowlist():
    from torchtitan.config.configurable import config

    @config
    class AnnotatedFrozen(FrozenConfig):
        @property
        def annotation(self):
            return self._get_metadata("annotation", None)

        @annotation.setter
        def annotation(self, value):
            self._set_metadata("annotation", value)

    inner = _compact().blocks["dense"].attention
    wrapped = AnnotatedFrozen(inner=inner)
    wrapped.annotation = "wrapper-owned"
    assert wrapped.annotation == "wrapper-owned"
    assert not hasattr(inner, "annotation")
    clone = wrapped._copy()
    clone.annotation = "copied"
    assert wrapped.annotation == "wrapper-owned"


def test_root_hidden_z_loss_survives_compact_materialization():
    compact = _compact()
    compact.hidden_z_loss = HiddenZLoss(coef=0.01, on="input")
    runtime = compact.materialize()
    assert runtime.hidden_z_loss == compact.hidden_z_loss
    assert runtime.hidden_z_loss is not compact.hidden_z_loss


@pytest.mark.parametrize(
    "field,value",
    [
        ("blocks.dense.attention.n_heads", "7"),
        ("blocks.dense.feed_forward.hidden_dim", "0"),
    ],
)
def test_invalid_compact_field_has_exact_source(field, value):
    result = run_pre_submit(
        "configs/experiments/llama3_debug.yaml", [(f"model.config.{field}", value)]
    )
    assert len(result.report.errors) == 1
    issue = result.report.errors[0]
    assert issue.path == f"model.config.{field}"
    assert issue.source == "CLI override"
    assert "CLI override" in str(issue)


def test_invalid_field_reports_winning_import_file(tmp_path):
    from pathlib import Path
    from torchtitan.config.save import dump_yaml

    imported = tmp_path / "invalid.yaml"
    imported.write_text(
        dump_yaml(
            {
                "config_version": 1,
                "imports": [
                    str(Path("configs/experiments/llama3_debug.yaml").resolve())
                ],
                "model": {
                    "config": {"blocks": {"dense": {"attention": {"n_heads": 7}}}}
                },
            }
        )
    )
    entry = tmp_path / "entry.yaml"
    entry.write_text(dump_yaml({"config_version": 1, "imports": [str(imported)]}))
    result = run_pre_submit(str(entry))
    assert len(result.report.errors) == 1
    assert result.report.errors[0].path == "model.config.blocks.dense.attention.n_heads"
    assert Path(result.report.errors[0].source).resolve() == imported.resolve()


def test_repeated_template_diagnostics_are_deduplicated():
    result = run_pre_submit(
        "configs/experiments/llama3_debug.yaml",
        [("parallelism.tensor_parallel_degree", "3")],
    )
    issues = [issue for issue in result.report.errors if issue.path.endswith("n_heads")]
    assert len(issues) == 1


def test_hybrid_cp_rejects_hybrid_support_without_none_kernel_noise():
    result = run_pre_submit(
        "configs/experiments/gift_3_5_debug.yaml",
        [("parallelism.context_parallel_degree", "2")],
    )
    rendered = "\n".join(map(str, result.report.errors))
    assert "Context Parallel does not support ['gdn']." in rendered
    assert "NoneType" not in rendered


@pytest.mark.parametrize(
    "overrides,path",
    [
        (
            [
                ("parallelism.tensor_parallel_degree", "2"),
                ("parallelism.enable_sequence_parallel", "true"),
            ],
            "parallelism.enable_sequence_parallel",
        ),
        (
            [("parallelism.context_parallel_degree", "2")],
            "parallelism.context_parallel_degree",
        ),
    ],
)
def test_mtp_parallelism_errors_point_to_the_incompatible_option(overrides, path):
    result = run_pre_submit(
        "configs/experiments/gift_moe_debug.yaml",
        [("model.config.mtp.layers", "[{block: mla_moe}]"), *overrides],
    )
    issues = [
        issue
        for issue in result.report.errors
        if "MTP is incompatible" in issue.message
    ]
    assert issues
    assert all(issue.path == path for issue in issues)


def test_sparse_fp8_issues_deduplicate_original_template_fields():
    from torchtitan.config.configs import FP8Config
    from torchtitan.config.schema.models.common.cross_section import model_fits_fp8
    from torchtitan.config.configs import ParallelismConfig

    model = _compact(
        "configs/experiments/gift_moe_debug_hybridep_fp8.yaml",
        [("model.config.dim", "256")],
    ).materialize()
    issues = model_fits_fp8(
        model, FP8Config(sparse_use_grouped_gemm=True), ParallelismConfig()
    )
    dims = [issue for issue in issues if "dim % 512" in issue.message]
    assert len(dims) == 1
    assert dims[0].path == "model.config.dim"


def test_gift_cp_checks_later_softmax_layers_and_reports_linear_layers():
    from torchtitan.config.configs import ParallelismConfig
    from torchtitan.config.schema.models.common.attention.softmax.kernels.flex import (
        FlexAttentionConfig,
    )

    model = _compact("configs/experiments/gift_3_5_debug.yaml").materialize()
    # Hybrid debug starts with linear attention; a later MLA must still be checked.
    softmax = [
        layer.attention
        for layer in model.layers
        if hasattr(layer.attention, "inner_attention")
    ]
    assert softmax
    softmax[-1].inner_attention = FlexAttentionConfig()
    errors = model.validate_parallelism(ParallelismConfig(context_parallel_degree=2))
    rendered = "\n".join(map(str, errors))
    # The linear layers are reported by their own capability, the MLA by its
    # kernel; nothing is reported for the kernel-less linear attention.
    assert "Context Parallel does not support ['gdn', 'mla with flex']." in rendered
    assert "NoneType" not in rendered
