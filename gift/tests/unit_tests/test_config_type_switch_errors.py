"""Direct schema failures explain retained fields without changing merge semantics."""

from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import pytest

from torchtitan.config import load_config, load_document
from torchtitan.config.build import build_trainer_config, load_model_spec
from torchtitan.config.cli import load_trainer_config
from torchtitan.config.errors import ConfigSchemaError
from torchtitan.config.pre_submit import run_pre_submit
from torchtitan.config.save import dump_yaml

EXPERIMENT = "configs/experiments/llama3_debug.yaml"
SWITCH = [("dataloader.type", "streaming_text")]


def _assert_switch(error):
    assert str(error).startswith("dataloader.dataset:")
    assert "from 'hf_text' to 'streaming_text'" in str(error)
    assert "CLI override" in str(error)
    assert "!replace {type: streaming_text, ...}" in str(error)
    assert error.source.endswith("presets/train/debug.yaml")


def test_public_loader_explains_the_retained_field_before_runtime_imports():
    with pytest.raises(ConfigSchemaError) as caught:
        load_config(EXPERIMENT, overrides=["--dataloader.type", "streaming_text"])
    _assert_switch(caught.value)


def test_direct_trainer_builder_preserves_raw_document_and_source():
    document, _ = load_document(EXPERIMENT, SWITCH)
    before = deepcopy(document.data)
    with pytest.raises(ConfigSchemaError) as caught:
        build_trainer_config(document, with_model_spec=False)
    _assert_switch(caught.value)
    assert document.data == before
    assert document.data["dataloader"]["dataset"] == "c4_test"


def test_pre_submit_keeps_structured_error_and_existing_warning():
    result = run_pre_submit(EXPERIMENT, SWITCH)
    assert not result.ok
    assert len(result.report.warnings) == len(result.report.errors) == 1
    error = result.report.errors[0]
    assert error.path == "dataloader.dataset"
    assert "!replace" in error.message
    assert error.source.endswith("presets/train/debug.yaml")
    assert f"(source: {error.source})" not in error.message


def test_cli_logger_receives_actionable_direct_error():
    logger = Mock()
    with pytest.raises(SystemExit) as caught:
        load_trainer_config(
            [EXPERIMENT, "--dataloader.type", "streaming_text"], logger=logger
        )
    assert caught.value.code == 1
    assert any("!replace" in str(arg) for arg in logger.error.call_args.args)


@pytest.mark.parametrize(
    "overrides",
    [
        [
            ("dataloader.type", "hf_text_interleaved"),
            ("dataloader.dataset", "null"),
            ("dataloader.typo", "1"),
        ],
        [("dataloader", "!replace {type: hf_text_interleaved, typo: 1}")],
        [
            ("dataloader", "null"),
            ("dataloader", "{type: hf_text_interleaved, typo: 1}"),
        ],
        [("dataloader.type", "unknown")],
    ],
)
def test_explicit_typos_unknown_types_and_cleared_mappings_get_no_stale_hint(overrides):
    document, _ = load_document(EXPERIMENT, overrides)
    with pytest.raises(ConfigSchemaError) as caught:
        build_trainer_config(document, with_model_spec=False)
    assert "!replace" not in str(caught.value)


@pytest.mark.parametrize(
    "overrides",
    [
        [("dataloader", "!replace {type: hf_text_interleaved}")],
        [("dataloader", "null"), ("dataloader", "{type: hf_text_interleaved}")],
    ],
)
def test_explicit_replacement_and_null_reset_keep_their_existing_semantics(overrides):
    result = run_pre_submit(EXPERIMENT, overrides)
    assert result.ok, result.report.to_dict()
    assert not result.report.warnings
    assert "dataset" not in result.data["dataloader"]


def _model_switch(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text(
        dump_yaml(
            {
                "imports": [
                    str(Path("configs/presets/models/llama3/debugmodel.yaml").resolve())
                ],
                "model": {
                    "config": {"norm": {"type": "rmsnorm", "elementwise_affine": True}}
                },
            }
        )
    )
    intermediate = tmp_path / "intermediate.yaml"
    intermediate.write_text("imports: [base.yaml]\n")
    entry = tmp_path / "model.yaml"
    entry.write_text(
        dump_yaml(
            {
                "imports": [str(intermediate)],
                "model": {
                    "config": {
                        "norm": {
                            "type": "groupnorm",
                            "num_groups": 4,
                            "num_channels": 256,
                        }
                    }
                },
            }
        )
    )
    return entry, base


def test_model_only_loader_explains_nested_imported_type_switch_before_runtime(
    tmp_path,
):
    entry, base = _model_switch(tmp_path)
    with pytest.raises(ConfigSchemaError) as caught:
        load_model_spec(str(entry))
    error = caught.value
    assert str(error).startswith("model.config.norm.elementwise_affine:")
    assert "from 'rmsnorm' to 'groupnorm'" in str(error)
    assert "!replace {type: groupnorm, ...}" in str(error)
    assert Path(error.source).resolve() == base.resolve()


def test_real_new_component_requirement_is_not_a_retained_field_error():
    document, _ = load_document(
        EXPERIMENT,
        [
            ("dataloader.type", "hf_chat"),
            ("dataloader.dataset", "null"),
        ],
    )
    with pytest.raises(ConfigSchemaError) as caught:
        build_trainer_config(document, with_model_spec=False)
    assert "sample_processor" in str(caught.value)
    assert "!replace" not in str(caught.value)


def test_inherited_field_error_after_intentional_subclass_transition_gets_no_hint():
    result = run_pre_submit(
        "configs/experiments/ft_llama3_debug.yaml",
        [("model.config.full_attention.n_heads", "7")],
    )
    assert not result.ok
    assert not result.report.warnings
    assert "!replace" not in result.report.errors[0].message
