"""Materialized models expose editable runtime modules independent of the written blocks."""

import pytest

from tests.unit_tests.config_helpers import PRESET_ROOT
from torchtitan.config import load_raw_document
from torchtitan.config.decode import decode
from torchtitan.config.schema.models.common.decoder_blocks.decoder import (
    DecoderConfig,
)
from torchtitan.config.schema.models.common.norms.rmsnorm import RMSNormConfig
from torchtitan.config.sections import ModelSection


@pytest.mark.parametrize("preset", sorted(PRESET_ROOT.rglob("*.yaml")))
def test_materialized_presets_place_every_layer(preset):
    document = load_raw_document(str(preset))
    written = decode(ModelSection, document.data["model"]).config
    runtime = written.materialize()
    assert runtime.materialized and not written.materialized
    assert runtime is not written
    if isinstance(runtime, DecoderConfig):
        assert runtime.n_layers == len(runtime.layers) == written.n_layers
        assert len({id(layer) for layer in runtime.layers}) == len(runtime.layers)


def test_actual_layer_edits_are_preserved_and_written_blocks_are_independent():
    document = load_raw_document(str(PRESET_ROOT / "llama3/debugmodel.yaml"))
    written = decode(ModelSection, document.data["model"]).config
    runtime = written.materialize()
    assert isinstance(runtime, DecoderConfig)
    template_norm = written.blocks["dense"].attention_norm
    original_eps = template_norm.eps
    first_norm = runtime.layers[0].attention_norm
    second_norm = runtime.layers[1].attention_norm
    assert isinstance(first_norm, RMSNormConfig)
    assert isinstance(second_norm, RMSNormConfig)
    first_norm.eps = original_eps * 2
    assert template_norm.eps == original_eps
    assert second_norm.eps == original_eps
    runtime.layers.pop()
    assert runtime.n_layers == written.n_layers - 1
    again = runtime.materialize()
    assert again.materialized
    assert again.n_layers == runtime.n_layers
    assert again.layers[0].attention_norm.eps == original_eps * 2
