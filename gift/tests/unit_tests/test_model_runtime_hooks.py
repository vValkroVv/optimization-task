"""CPU/meta coverage of the model runtime preparation boundary."""

import re

import pytest
import torch

from tests.unit_tests.config_helpers import load_model_spec, PRESET_ROOT
from tests.unit_tests.test_config_parallelism_review import CASES
from torchtitan.config import load_document, load_raw_document
from torchtitan.config.build import build_model_spec, build_trainer_config
from torchtitan.config.decode import decode
from torchtitan.config.schema.components.lora import FrozenConfig
from torchtitan.config.schema.models.common.nn_modules import LinearConfig
from torchtitan.config.sections import ModelSection
from torchtitan.models.common.nn_modules import Linear


@pytest.mark.parametrize("variant", ["debugmodel", "debugmodel_moe"])
def test_graph_qwen_runtime_accepts_dense_and_moe_blocks(variant):
    # Dense and MoE are the same model type with different blocks.
    spec = load_model_spec(
        f"qwen3/{variant}.yaml", {"config": {"type": "graph_trainer/qwen3"}}
    )
    from torchtitan.experiments.graph_trainer.qwen3.model import GraphTrainerQwen3Model

    with torch.device("meta"):
        model = spec.model.build()
    assert isinstance(model, GraphTrainerQwen3Model)
    assert model.layers
    assert all(parameter.is_meta for parameter in model.parameters())


@pytest.mark.parametrize("experiment,overrides,message", CASES)
def test_runtime_owner_rejects_invalid_run_before_module_construction(
    experiment, overrides, message
):
    document, _ = load_document("configs/experiments/" + experiment, overrides)
    trainer_config = build_trainer_config(document, with_model_spec=False)
    model_config = decode(ModelSection, document.data["model"]).config.materialize()
    with pytest.raises(ValueError, match=re.escape(message)):
        model_config.owner().prepare_config(model_config, config=trainer_config)


def test_frozen_factory_preserves_module_type_and_parameter_names():
    from torchtitan.components.lora import build_frozen

    assert FrozenConfig.owner() is build_frozen
    inner = LinearConfig(in_features=4, out_features=3, bias=True)
    with torch.device("meta"):
        model = FrozenConfig(inner=inner).build()
    assert type(model) is Linear
    assert set(model.state_dict()) == {"weight", "bias"}
    assert set(dict(model.named_parameters())) == {"weight", "bias"}
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_compact_build_and_model_spec_apply_converters_once():
    document = load_raw_document(str(PRESET_ROOT / "llama3/debugmodel.yaml"))
    document.data["model"]["config"]["converters"] = [
        {"type": "lora", "rank": 2, "target_modules": ["w1"]}
    ]
    section = decode(ModelSection, document.data["model"])
    compact = section.config
    spec = build_model_spec(section)
    with torch.device("meta"):
        direct_model = compact.build()
        built_model = spec.model.build()
    assert set(direct_model.state_dict()) == set(built_model.state_dict())
    assert any("lora_a" in name for name in direct_model.state_dict())
    assert not compact.materialized
    assert compact.blocks["dense"].feed_forward.w1 is None


def test_export_opt_out_is_preserved_by_runtime_constructor():
    document = load_raw_document(str(PRESET_ROOT / "llama3/debugmodel.yaml"))
    document.data["model"]["config"]["converters"] = [{"type": "lora", "rank": 2}]
    section = decode(ModelSection, document.data["model"])
    spec = build_model_spec(section, apply_converters=False)
    with torch.device("meta"):
        model = spec.model.build()
    assert not any("lora_a" in name for name in model.state_dict())


def test_decoder_pipeline_costing_rejects_unmaterialized_model_configs():
    from torchtitan.config.configs import ActivationCheckpointConfig
    from torchtitan.distributed.pipeline_balance import model_pipeline_units

    document = load_raw_document(str(PRESET_ROOT / "llama3/debugmodel.yaml"))
    compact = decode(ModelSection, document.data["model"]).config
    assert not compact.materialized
    with pytest.raises(ValueError, match="requires a decoder model"):
        model_pipeline_units(
            torch.nn.Identity(),
            compact,
            8,
            ActivationCheckpointConfig(),
            segments=([], [], []),
            block_fqns=set(),
        )
