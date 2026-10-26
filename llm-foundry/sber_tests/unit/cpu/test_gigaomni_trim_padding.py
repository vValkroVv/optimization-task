"""Microbatch padding trim, the SP re-pad after video token reduction, and the
pad mask the language model gets.

The omni ``TextCollator`` pads a whole device batch to its longest row, rounded
up to ``divider`` (``2 * tp_sp``), and its ``split_batch`` cuts every
microbatch back to its own longest row. Video token reduction then drops an
arbitrary number of tokens inside the model, so ``VideoOnlyModality`` pads the
sequence back to the multiple the model puts on ``GigaOmniDataState``. The
model hands a language model that opts in (giga_mix) ``padding_mask`` from
those final ids, so the MoE router skips both kinds of pads.
``import_modules`` loads the modules on a CPU-only host.
"""

import inspect
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sber_tests.test_utils.gigaomni_cpu import import_modules

text_collator, flash_attn_collator, modelling, video_only, utils = import_modules(
    "llmfoundry.data.collators.text_collator",
    "llmfoundry.data.collators.text_flash_attn_collator",
    "llmfoundry.models.gigaomni.modelling_gigaomni",
    "llmfoundry.models.gigaomni.modalities.video_only_modality",
    "llmfoundry.models.gigaomni.utils_gigaomni",
)
GigaOmniDataState = modelling.GigaOmniDataState
GigaOmniForCausalLM = modelling.GigaOmniForCausalLM
VideoOnlyModality = video_only.VideoOnlyModality

pytestmark = [pytest.mark.unit, pytest.mark.fast]

PAD_ID = 0
IGNORE_INDEX = -100
PLACEHOLDER_ID = 99
HIDDEN = 4
# gigachat-utils `convertation/mapper_omni_utils.py` ships exactly these siblings
# of the modeling file in every hf_omni export.
EXPORTED_SIBLINGS = {"configuration_gigaomni", "utils_gigaomni", "modalities"}


def _round_up(length, multiple):
    return -(-length // multiple) * multiple


def _collator(monkeypatch, sp_size, **kwargs):
    """The omni pipeline's ``TextCollator`` (``scale=2.0``) at ``sp_size``."""
    monkeypatch.setattr(
        text_collator,
        "dist",
        SimpleNamespace(get_tp_sp_group_size=lambda: sp_size),
    )
    return text_collator.TextCollator(pad_token_id=PAD_ID, scale=2.0, **kwargs)


def _examples(real_lengths):
    """Samples of ``real_lengths`` tokens (ids ``1..``, labels ``101..``)."""
    return [
        {"input_ids": torch.arange(1, n + 1), "labels": torch.arange(101, n + 101)}
        for n in real_lengths
    ]


@pytest.mark.parametrize("sp_size", [None, 4])
def test_each_microbatch_is_cut_to_its_own_longest_row(monkeypatch, sp_size):
    collator = _collator(monkeypatch, sp_size)
    divider = 2 * (sp_size or 1)
    real_lengths = [5, 30, 3]
    batch = collator(_examples(real_lengths))
    assert batch["input_ids"].shape[1] == _round_up(30, divider)

    microbatches = collator.split_batch(batch, microbatch_size=1)

    for row, (microbatch, real) in enumerate(zip(microbatches, real_lengths)):
        length = _round_up(real, divider)
        assert torch.equal(
            microbatch["input_ids"], batch["input_ids"][row : row + 1, :length]
        )
        assert torch.equal(
            microbatch["labels"], batch["labels"][row : row + 1, :length]
        )


def test_microbatch_of_several_rows_keeps_its_longest_row(monkeypatch):
    collator = _collator(monkeypatch, sp_size=2)
    batch = collator(_examples([5, 9, 30, 3]))

    microbatches = collator.split_batch(batch, microbatch_size=2)

    assert [tuple(part["input_ids"].shape) for part in microbatches] == [
        (2, 12),
        (2, 32),
    ]
    assert microbatches[0]["labels"].is_contiguous()


def test_attention_mask_is_cut_with_the_tokens(monkeypatch):
    collator = _collator(monkeypatch, sp_size=None, return_attention_mask=True)
    batch = collator(_examples([3, 10]))

    short, _ = collator.split_batch(batch, microbatch_size=1)

    assert short["attention_mask"].tolist() == [[True, True, True, False]]


def test_trailing_pad_id_with_a_real_label_is_kept(monkeypatch):
    collator = _collator(monkeypatch, sp_size=None)
    examples = _examples([5, 10])
    examples[0]["input_ids"][-1] = PAD_ID

    short, _ = collator.split_batch(collator(examples), microbatch_size=1)

    assert short["input_ids"].tolist() == [[1, 2, 3, 4, PAD_ID, PAD_ID]]
    assert short["labels"].tolist() == [[101, 102, 103, 104, 105, IGNORE_INDEX]]


def test_left_padded_microbatch_is_not_cut(monkeypatch):
    collator = _collator(monkeypatch, sp_size=None, left_padding=True)
    batch = collator(_examples([3, 10]))

    short, _ = collator.split_batch(batch, microbatch_size=1)

    assert short["input_ids"].shape == (1, 10)


def test_pad_only_microbatch_is_not_cut(monkeypatch):
    collator = _collator(monkeypatch, sp_size=None)
    batch = {
        "input_ids": torch.full((1, 8), PAD_ID),
        "labels": torch.full((1, 8), IGNORE_INDEX),
    }

    (microbatch,) = collator.split_batch(batch, microbatch_size=1)

    assert microbatch["input_ids"].shape == (1, 8)


def test_flash_attn_collator_cuts_position_ids_with_the_tokens(monkeypatch):
    collator = object.__new__(flash_attn_collator.TextFlashAttnCollator)
    collator.base_collator = _collator(monkeypatch, sp_size=None)
    batch = collator.base_collator(_examples([3, 10]))
    batch["position_ids"] = torch.arange(10).expand(2, -1)

    short, full = collator.split_batch(batch, microbatch_size=1)

    assert short["position_ids"].tolist() == [[0, 1, 2, 3]]
    assert full["position_ids"].tolist() == [list(range(10))]


# The modality's own config, enough for `process_inputs_for_language_model`.
VIDEO = SimpleNamespace(config=SimpleNamespace(modality_output={"name": "video"}))


def _reduced_state(length, drop, multiple, pad_token_id=PAD_ID):
    """The state once the adapter dropped ``drop`` video tokens of a row."""
    text_mask = torch.ones(1, length, dtype=torch.bool)
    text_mask[0, 2 : 2 + drop] = False
    input_ids = torch.arange(1, length + 1).unsqueeze(0)
    inputs_embeds = torch.randn(1, length, HIDDEN)
    state = GigaOmniDataState(pad_to_multiple_of=multiple, pad_token_id=pad_token_id)
    state.update("encoder_outputs", "text", {"inputs_embeds": inputs_embeds})
    state.update(
        "adapter_outputs",
        "video",
        {
            "input_ids": input_ids[text_mask].unsqueeze(0),
            "labels": (input_ids + 100)[text_mask].unsqueeze(0),
            "attention_mask": torch.ones(1, length - drop, dtype=torch.bool),
            "text_mask": text_mask,
        },
    )
    return state, inputs_embeds[text_mask]


@pytest.mark.parametrize("multiple", [2, 8])
def test_reduced_sequence_is_padded_back_to_the_multiple(multiple):
    state, kept_embeds = _reduced_state(16, drop=5, multiple=multiple)

    VideoOnlyModality.process_inputs_for_language_model(VIDEO, state)

    text = state.modality_inputs["text"]
    inputs_embeds = state.encoder_outputs["text"]["inputs_embeds"]
    expected = _round_up(11, multiple)
    assert inputs_embeds.shape == (1, expected, HIDDEN)
    torch.testing.assert_close(inputs_embeds[0, :11], kept_embeds)
    assert (inputs_embeds[0, 11:] == 0).all()
    assert text["input_ids"].shape == (1, expected)
    assert (text["input_ids"][0, 11:] == PAD_ID).all()
    assert (text["labels"][0, 11:] == IGNORE_INDEX).all()
    assert text["attention_mask"].dtype == torch.bool
    assert text["attention_mask"][0, :11].all()
    assert not text["attention_mask"][0, 11:].any()


def test_reduced_sequence_on_the_multiple_is_not_padded():
    state, _ = _reduced_state(16, drop=4, multiple=4)

    VideoOnlyModality.process_inputs_for_language_model(VIDEO, state)

    assert state.encoder_outputs["text"]["inputs_embeds"].shape == (1, 12, HIDDEN)
    assert state.modality_inputs["text"]["input_ids"].shape == (1, 12)


def test_pad_token_id_from_the_state_fills_the_tail():
    state, _ = _reduced_state(16, drop=5, multiple=4, pad_token_id=7)

    VideoOnlyModality.process_inputs_for_language_model(VIDEO, state)

    assert state.modality_inputs["text"]["input_ids"][0, 11:].tolist() == [7]


def test_default_state_leaves_the_reduced_sequence_as_is():
    state, _ = _reduced_state(16, drop=5, multiple=1)

    VideoOnlyModality.process_inputs_for_language_model(VIDEO, state)

    assert state.encoder_outputs["text"]["inputs_embeds"].shape == (1, 11, HIDDEN)


def test_state_without_the_pad_fields_is_not_padded():
    """An older bundled ``modelling_gigaomni.py`` builds a state without them."""
    inputs_embeds = torch.randn(1, 11, HIDDEN)
    state = SimpleNamespace(
        modality_inputs={"text": {"input_ids": torch.ones(1, 11, dtype=torch.long)}},
        encoder_outputs={"text": {"inputs_embeds": inputs_embeds}},
    )

    video_only._pad_to_multiple(state)

    assert state.encoder_outputs["text"]["inputs_embeds"] is inputs_embeds
    assert state.modality_inputs["text"]["input_ids"].shape == (1, 11)


class _CountingEmbedding(nn.Embedding):
    calls = 0

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.calls += 1

        return super().forward(input_ids)


def _owner(pad_token_id=None, accepts_padding_mask=True):
    """``GigaOmniForCausalLM`` with a counting embedding and one video modality."""
    embed_tokens = _CountingEmbedding(128, HIDDEN, padding_idx=PAD_ID)
    model = object.__new__(GigaOmniForCausalLM)
    model.language_model = SimpleNamespace(
        accepts_padding_mask=accepts_padding_mask,
        get_input_embeddings=lambda: embed_tokens,
        config=SimpleNamespace(pad_token_id=pad_token_id, return_dict=True),
        forward=lambda **kwargs: (kwargs, kwargs["inputs_embeds"]),
    )
    model.encoders = {}
    model.adapters = {}
    model.modalities = [
        SimpleNamespace(
            name="video",
            config=SimpleNamespace(
                modality_output={
                    "type": "adapter_outputs",
                    "name": "video",
                    "key": "embeddings",
                    "placeholder_token_id": PLACEHOLDER_ID,
                },
                data_flow=SimpleNamespace(
                    encoders=[
                        SimpleNamespace(
                            inputs=[
                                SimpleNamespace(
                                    type="modality_inputs",
                                    name="video",
                                    keys=["pixel_values"],
                                )
                            ]
                        )
                    ]
                ),
            ),
            update_modality_inputs_with_dummy_values=lambda inputs: None,
            process_inputs_for_component=lambda state, component_type: None,
        ),
    ]
    return model, embed_tokens


@pytest.mark.parametrize(
    ("sp_size", "pad_token_id", "expected"),
    [(None, None, (2, 0)), (4, 7, (8, 7))],
)
def test_forward_puts_twice_sp_and_the_pad_id_on_the_state(
    monkeypatch, sp_size, pad_token_id, expected
):
    monkeypatch.setattr(modelling, "get_tp_sp_group_size", lambda: sp_size)
    model, _ = _owner(pad_token_id)
    seen = []

    def record(state):
        seen.append((state.pad_to_multiple_of, state.pad_token_id))
        state.update("adapter_outputs", "video", {"embeddings": torch.zeros(0, HIDDEN)})

    model.modalities[0].process_inputs_for_language_model = record

    GigaOmniForCausalLM.forward(model, torch.tensor([[1, 2]]), modality_inputs={})

    assert seen == [expected]


def test_forward_hands_the_lm_the_reduced_sequence_padded_to_twice_sp(monkeypatch):
    monkeypatch.setattr(modelling, "get_tp_sp_group_size", lambda: 2)
    model, embed_tokens = _owner()
    input_ids = torch.arange(1, 13).unsqueeze(0)
    text_mask = torch.tensor([[True] * 4 + [False] * 3 + [True] * 5])
    video = model.modalities[0]

    def reduce_video_tokens(state):
        text = state.modality_inputs["text"]
        state.update(
            "adapter_outputs",
            "video",
            {
                "input_ids": text["input_ids"][text_mask].unsqueeze(0),
                "labels": text["labels"][text_mask].unsqueeze(0),
                "text_mask": text_mask,
                "embeddings": torch.zeros(0, HIDDEN),
            },
        )
        VideoOnlyModality.process_inputs_for_language_model(video, state)

    video.process_inputs_for_language_model = reduce_video_tokens

    lm_kwargs, _ = GigaOmniForCausalLM.forward(
        model, input_ids, modality_inputs={}, labels=input_ids.clone()
    )

    assert embed_tokens.calls == 1
    assert lm_kwargs["inputs_embeds"].shape == (1, 12, HIDDEN)
    torch.testing.assert_close(
        lm_kwargs["inputs_embeds"][0, :9], embed_tokens.weight[input_ids[text_mask]]
    )
    assert (lm_kwargs["inputs_embeds"][0, 9:] == 0).all()
    assert lm_kwargs["labels"].tolist() == [[1, 2, 3, 4, 8, 9, 10, 11, 12] + [-100] * 3]
    assert lm_kwargs["padding_mask"].tolist() == [[True] * 9 + [False] * 3]


def test_forward_masks_the_collator_pads_with_the_lm_pad_id(monkeypatch):
    monkeypatch.setattr(modelling, "get_tp_sp_group_size", lambda: None)
    model, _ = _owner(pad_token_id=7)
    model.modalities[0].process_inputs_for_language_model = lambda state: state.update(
        "adapter_outputs", "video", {"embeddings": torch.zeros(0, HIDDEN)}
    )
    input_ids = torch.tensor([[5, PAD_ID, 6, 7], [5, 6, 7, 7]])

    lm_kwargs, _ = GigaOmniForCausalLM.forward(model, input_ids, modality_inputs={})

    assert lm_kwargs["padding_mask"].tolist() == [
        [True, True, True, False],
        [True, True, False, False],
    ]
    assert lm_kwargs["attention_mask"] is None


def test_forward_keeps_attention_mask_zeros_out_of_the_router(monkeypatch):
    """A real token the attention mask hides is not counted by the router."""
    monkeypatch.setattr(modelling, "get_tp_sp_group_size", lambda: None)
    model, _ = _owner()
    model.modalities[0].process_inputs_for_language_model = lambda state: state.update(
        "adapter_outputs", "video", {"embeddings": torch.zeros(0, HIDDEN)}
    )
    attention_mask = torch.tensor([[1, 0, 1, 0]])

    lm_kwargs, _ = GigaOmniForCausalLM.forward(
        model,
        torch.tensor([[5, 6, 7, PAD_ID]]),
        modality_inputs={},
        attention_mask=attention_mask,
    )

    assert lm_kwargs["padding_mask"].tolist() == [[True, False, True, False]]
    assert lm_kwargs["attention_mask"].tolist() == [[1, 0, 1, 0]]


def test_forward_leaves_padding_mask_out_for_a_model_that_does_not_take_it(
    monkeypatch,
):
    """The language models an hf_omni export vendors do not take the kwarg."""
    monkeypatch.setattr(modelling, "get_tp_sp_group_size", lambda: None)
    model, _ = _owner(accepts_padding_mask=False)
    model.modalities[0].process_inputs_for_language_model = lambda state: state.update(
        "adapter_outputs", "video", {"embeddings": torch.zeros(0, HIDDEN)}
    )

    lm_kwargs, _ = GigaOmniForCausalLM.forward(
        model, torch.tensor([[5, 6, PAD_ID]]), modality_inputs={}
    )

    assert "padding_mask" not in lm_kwargs


def test_language_models_that_opt_in_take_padding_mask():
    opted_in = {
        name: inspect.signature(model_class.forward).parameters
        for name, model_class in utils.GIGAOMNI_MODEL_REGISTRY.items()
        if getattr(model_class, "accepts_padding_mask", False)
    }

    if utils.GIGAOMNI_MODEL_REGISTRY["giga_mix_causal_lm"] is None:
        pytest.skip("giga_mix is not importable on this host")

    assert "giga_mix_causal_lm" in opted_in

    for name, parameters in opted_in.items():
        assert "padding_mask" in parameters, name


def test_bundled_modeling_files_only_import_exported_siblings():
    gigaomni_dir = Path(modelling.__file__).parent

    for name in ("modelling_gigaomni", "utils_gigaomni", "configuration_gigaomni"):
        source = (gigaomni_dir / f"{name}.py").read_text()
        imports = re.findall(r"^from (\.+)([\w.]*) import", source, re.M)
        outside = {
            dots + module
            for dots, module in imports
            if dots != "." or module.split(".")[0] not in EXPORTED_SIBLINGS
        }
        assert not outside, (
            f"{name}.py imports {outside}, which the hf_omni exporter does not ship"
        )


def test_video_modality_imports_neither_composer_nor_llmfoundry():
    source = Path(video_only.__file__).read_text()

    assert not re.search(r"^\s*(from|import) (composer|llmfoundry)\b", source, re.M)
