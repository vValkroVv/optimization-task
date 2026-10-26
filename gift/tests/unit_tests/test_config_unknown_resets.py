"""A ``null`` that deletes nothing must name a key the schema has.

The decode checks each such reset against the class at the reset's parent path.
"""

import pytest

from tests.unit_tests.config_helpers import decode_document
from tests.unit_tests.torch_free import REPO_ROOT
from torchtitan.config import ConfigSchemaError, load_document, TrainingConfig
from torchtitan.config.build import load_model_spec
from torchtitan.config.decode import checking_missing_cleared_paths, decode
from torchtitan.config.loader import deep_merge
from torchtitan.config.sections import ModelSection

EXPERIMENTS = REPO_ROOT / "configs" / "experiments"
FLUX_DEBUG = EXPERIMENTS / "flux_debug.yaml"
LLAMA3_DEBUG = EXPERIMENTS / "llama3_debug.yaml"
QWEN3_MOE_DEBUG = EXPERIMENTS / "qwen3_moe_debug.yaml"
SFT_LLAMA3_DEBUG = EXPERIMENTS / "sft_llama3_debug.yaml"


def _experiment(tmp_path, body: str, base=LLAMA3_DEBUG) -> str:
    """An experiment that imports *base* and adds *body*."""
    path = tmp_path / "exp.yaml"
    imports = f"imports: [{base}]\n" if base else ""
    path.write_text(f"config_version: 1\n{imports}{body}", encoding="utf-8")
    return str(path)


@pytest.mark.parametrize("prior", [False, True])
def test_reset_field_inside_a_literal_pattern_key(tmp_path, prior):
    path = _experiment(
        tmp_path,
        "model:\n  config:\n    hidden_z_loss_patterns:\n"
        '      "layers.*.attention.wo": {coef: 0.1'
        + (", on: input" if prior else "")
        + "}\n",
        base=EXPERIMENTS / "gift_moe_debug.yaml",
    )
    data = decode_document(
        path,
        [
            (
                "model.config.hidden_z_loss_patterns",
                '{"layers.*.attention.wo": {on: null}}',
            )
        ],
    )
    assert data["model"]["config"]["hidden_z_loss_patterns"] == {
        "layers.*.attention.wo": {"coef": 0.1}
    }


@pytest.mark.parametrize(
    "body,message",
    [
        # inherited section: the parent exists, the key does not
        ("training: {stpes: null}\n", "training.stpes: .*Did you mean 'steps'"),
        ("trainng: null\n", "trainng: .*Did you mean 'training'"),
        ("trainng: {steps: null}\n", "trainng: .*Did you mean 'training'"),
        ("dataloader: {dataste: null}\n", "dataloader.dataste: .*'dataset'"),
        ("model: {config: {dmi: null}}\n", "model.config.dmi: .*Did you mean 'dim'"),
        (
            "model: {config: {converters: [{type: lora, rnka: null}]}}\n",
            r"model.config.converters\[0\].rnka: .*Did you mean 'rank'",
        ),
        ("model: {label_name: null}\n", "model.label_name: .*Did you mean 'label'"),
        ("trainer: {kind: null}\n", "trainer.kind: unknown key"),
    ],
)
def test_reset_of_an_unknown_key_is_an_error(tmp_path, body, message):
    with pytest.raises(ConfigSchemaError, match=rf"^{message}"):
        decode_document(_experiment(tmp_path, body))


@pytest.mark.parametrize(
    "overrides",
    [
        [("training.stpes", "null")],
        [("training", "{stpes: null, steps: 3}")],
    ],
)
def test_cli_reset_of_an_unknown_key_is_the_same_error(overrides):
    with pytest.raises(
        ConfigSchemaError,
        match=r"^training.stpes: unknown key \(reset with null\). Did you mean 'steps'",
    ):
        decode_document(LLAMA3_DEBUG, overrides)


def test_reset_is_checked_against_the_dispatched_class(tmp_path):
    """The family variant the ``type`` picked (rope ``scaling``, qwen3 ``moe``)
    decides which keys exist."""
    scaling = "model.config.rope.scaling"
    decode_document(LLAMA3_DEBUG, [(f"{scaling}", "{low_freq_factor: null}")])
    with pytest.raises(
        ConfigSchemaError,
        match=rf"^{scaling}.factr: unknown key \(reset with null\).*LlamaScaling",
    ):
        decode_document(LLAMA3_DEBUG, [(f"{scaling}.factr", "null")])
    moe = "model.config.blocks.moe.moe"
    decode_document(QWEN3_MOE_DEBUG, [(f"{moe}.experts.token_dispatcher", "null")])
    with pytest.raises(ConfigSchemaError, match=rf"^{moe}.router.topk: unknown key"):
        decode_document(QWEN3_MOE_DEBUG, [(f"{moe}.router.topk", "null")])


def test_reset_is_checked_against_the_default_variant():
    """After ``scaling.type: null`` the family default (``none``) judges later
    resets; a second ``type: null`` deletes nothing and passes."""
    scaling = "model.config.rope.scaling"
    document, _ = load_document(
        str(LLAMA3_DEBUG), [(f"{scaling}.type", "null"), (f"{scaling}.type", "null")]
    )
    with checking_missing_cleared_paths(document.missing_cleared_paths):
        section = decode(ModelSection, document.data["model"], path="model")
    assert type(section.config.rope.scaling).__name__ == "NoScaling"
    for reset in ("low_freq_factor", "factr"):
        with pytest.raises(
            ConfigSchemaError,
            match=rf"^{scaling}.{reset}: unknown key \(reset with null\).*NoScaling",
        ):
            decode_document(
                LLAMA3_DEBUG,
                [(f"{scaling}.type", "null"), (f"{scaling}.{reset}", "null")],
            )


def test_reset_of_type_that_deletes_nothing_always_passes():
    """It passes even on a plain dataclass, which has no ``type`` field."""
    decode_document(LLAMA3_DEBUG, [("training.type", "null")])


def test_reset_of_a_field_nothing_set_pins_the_default(tmp_path):
    """No import sets ``max_norm`` or ``steps``: the ``null`` deletes nothing
    and is still fine, it names a field."""
    data = decode_document(_experiment(tmp_path, "training: {max_norm: null}\n"))
    assert "max_norm" not in data["training"]
    document, _ = load_document(
        _experiment(tmp_path, "training: {steps: null}\n", base=None)
    )
    assert "training" not in document.data
    assert document.missing_cleared_paths == (("training", "steps"),)
    data = decode_document(LLAMA3_DEBUG, [("training.max_norm", "null")])
    assert "max_norm" not in data["training"]


def test_sft_reset_of_the_previous_dataloaders_key_passes():
    """``dataset: null`` after ``type: hf_chat`` pops the ``hf_text`` key the
    debug preset set; ``hf_chat`` has no ``dataset`` field, but nothing is
    left to check, so no type-switch rule is needed."""
    data = decode_document(SFT_LLAMA3_DEBUG)
    assert data["dataloader"]["type"] == "hf_chat"
    assert "dataset" not in data["dataloader"]


def test_reset_inside_an_open_mapping_passes(tmp_path):
    path = _experiment(
        tmp_path,
        "dataloader: {load_dataset_kwargs: {anything: null}}\n",
        base=SFT_LLAMA3_DEBUG,
    )
    data = decode_document(path)
    assert "anything" not in data["dataloader"]["load_dataset_kwargs"]


def test_reset_inside_a_list_item_takes_the_default(tmp_path):
    data = decode_document(
        _experiment(
            tmp_path, "model: {config: {converters: [{type: lora, rank: null}]}}\n"
        )
    )
    assert data["model"]["config"]["converters"] == [{"type": "lora"}]
    section = decode(ModelSection, data["model"], path="model")
    assert section.config.converters[0].rank == 8


_GATE = "model.config.blocks.dense.attention.gate"


@pytest.mark.parametrize(
    "overrides",
    [
        [(f"{_GATE}.gate_proj", "null")],
        [(_GATE, "null"), (f"{_GATE}.gate_proj", "null")],
    ],
)
def test_reset_under_an_absent_module_does_not_create_it(overrides):
    """An empty ``gate: {}`` would enable the gate."""
    data = decode_document(LLAMA3_DEBUG, overrides)
    assert "gate" not in data["model"]["config"]["blocks"]["dense"]["attention"]


@pytest.mark.parametrize(
    "path,message",
    [
        ("trainng.steps", r"trainng: unknown top-level section \(reset with null\)"),
        (f"{_GATE}.gate_projj", rf"{_GATE}.gate_projj: .*Did you mean 'gate_proj'"),
        (f"{_GATE}e.gate_proj", rf"{_GATE}e: unknown key .*Did you mean 'gate'"),
        # LinearConfig is its family's base and its default.
        (
            "model.config.lm_head.out_featuers",
            r"model.config.lm_head.out_featuers: unknown key",
        ),
        (
            f"{_GATE}.gate_proj.out_featuers",
            rf"{_GATE}.gate_proj.out_featuers: unknown key",
        ),
        (
            "model.config.blocks.densee.attention",
            r"model.config.blocks.densee: no entry to reset. Did you mean 'dense'",
        ),
    ],
)
def test_reset_under_an_absent_container_is_checked(path, message):
    """The decode never reaches these paths: the static type judges them."""
    with pytest.raises(ConfigSchemaError, match=rf"^{message}"):
        decode_document(LLAMA3_DEBUG, [(path, "null")])


@pytest.mark.parametrize("gate_set_first", [False, True])
def test_an_ancestor_reset_supersedes_the_resets_below_it(gate_set_first):
    overrides = [(f"{_GATE}.gate_projj", "null"), (_GATE, "null")]
    if gate_set_first:
        overrides.insert(0, (_GATE, "{}"))
    data = decode_document(LLAMA3_DEBUG, overrides)
    assert "gate" not in data["model"]["config"]["blocks"]["dense"]["attention"]


@pytest.mark.parametrize(
    "path",
    ["model.config.blocks.densee", "parallelism.pipeline_parallel_module_costs_ms.foo"],
)
def test_reset_of_an_absent_closed_dict_entry_is_an_error(path):
    with pytest.raises(ConfigSchemaError, match=rf"^{path}: no entry to reset"):
        decode_document(LLAMA3_DEBUG, [(path, "null")])


def test_a_repeated_reset_of_a_deleted_entry_is_not_recorded_as_missing():
    deleted: list[tuple[str | int, ...]] = []
    missing: list[tuple[str | int, ...]] = []
    merged = {"blocks": {"dense": {"x": 1}}}
    for _ in range(2):
        merged = deep_merge(
            merged, {"blocks": {"dense": None}}, _deleted=deleted, _missing=missing
        )
    assert (merged, deleted, missing) == ({"blocks": {}}, [("blocks", "dense")], [])


def test_reset_of_a_required_field_reaches_the_decoder(tmp_path):
    """``model.config`` is a field: the reset check passes and the decoder
    reports the missing required key, as without the ``null``."""
    path = _experiment(tmp_path, "model: {config: null, label: tiny}\n", base=None)
    document, _ = load_document(path)
    assert document.data["model"] == {"label": "tiny"}
    with (
        pytest.raises(ConfigSchemaError, match="model.config: missing required key"),
        checking_missing_cleared_paths(document.missing_cleared_paths),
    ):
        decode(ModelSection, document.data["model"], path="model")


def test_typo_set_in_a_base_and_cleared_by_the_same_typo_passes(tmp_path):
    """Accepted limitation: the child's reset pops a key that exists, so it
    is not checked. The base alone fails pre-submit (strict decode)."""
    base = tmp_path / "base.yaml"
    base.write_text(f"imports: [{LLAMA3_DEBUG}]\ntraining: {{stpes: 5}}\n")
    data = decode_document(
        _experiment(tmp_path, "training: {stpes: null}\n", base=base)
    )
    assert "stpes" not in data["training"]


def test_load_model_spec_checks_model_resets(tmp_path):
    """The model-only path (presets, checkpoint conversion) checks ``model.*``
    resets while decoding the model config, before anything imports torch."""
    preset = REPO_ROOT / "configs" / "presets" / "models" / "llama3" / "debugmodel.yaml"
    path = tmp_path / "preset.yaml"
    path.write_text(f"imports: [{preset}]\nmodel: {{config: {{dmi: null}}}}\n")
    with pytest.raises(ConfigSchemaError, match=r"^model.config.dmi: unknown key"):
        load_model_spec(str(path))


def test_reset_is_judged_by_the_final_type_whatever_the_argv_order():
    """``fuse_permute`` is a field of the hybridep dispatcher only, and the
    ``type`` that selects it may come before or after the reset."""
    dispatcher = "model.config.blocks.moe.moe.experts.token_dispatcher"
    for overrides in (
        [(f"{dispatcher}.type", "hybridep"), (f"{dispatcher}.fuse_permute", "null")],
        [(f"{dispatcher}.fuse_permute", "null"), (f"{dispatcher}.type", "hybridep")],
    ):
        decode_document(QWEN3_MOE_DEBUG, overrides)
    with pytest.raises(
        ConfigSchemaError,
        match=rf"^{dispatcher}.fuse_permute: unknown key .*DeepEPTokenDispatcherConfig",
    ):
        decode_document(
            QWEN3_MOE_DEBUG,
            [(f"{dispatcher}.type", "deepep"), (f"{dispatcher}.fuse_permute", "null")],
        )


def test_top_level_reset_is_judged_by_the_trainer_types_sections():
    """``encoder`` is a section of the flux trainer only."""
    with pytest.raises(
        ConfigSchemaError,
        match=r"^encodr: unknown top-level section \(reset with null\) for "
        r"trainer 'flux'. Did you mean 'encoder'",
    ):
        decode_document(FLUX_DEBUG, [("encodr", "null")])
    with pytest.raises(
        ConfigSchemaError,
        match=r"^encoder: unknown top-level section \(reset with null\) for "
        r"trainer 'default'",
    ):
        decode_document(LLAMA3_DEBUG, [("encoder", "null")])


def test_load_model_spec_checks_converter_resets(tmp_path):
    preset = REPO_ROOT / "configs" / "presets" / "models" / "llama3" / "debugmodel.yaml"
    path = tmp_path / "preset.yaml"
    path.write_text(
        f"imports: [{preset}]\n"
        "model: {config: {converters: [{type: lora, rnka: null}]}}\n"
    )
    with pytest.raises(
        ConfigSchemaError, match=r"^model.config.converters\[0\].rnka: unknown key"
    ):
        load_model_spec(str(path))


def test_the_check_ends_with_its_decode():
    """A failed decode leaves no reset keys behind for the next one."""
    with pytest.raises(ConfigSchemaError, match="training.stpes"):
        decode_document(LLAMA3_DEBUG, [("training.stpes", "null")])
    assert decode(TrainingConfig, {}, path="training") == TrainingConfig()


def test_null_cannot_remove_an_entry_of_an_init_table(tmp_path):
    """``init.linear.bias`` comes from the imported init preset: a null would
    drop it from the run's table, and the projection's bias would have no
    initializer. That holds whichever file set the entry last."""
    bias = "model.config.init.linear.bias"
    message = rf"^{bias}: null cannot remove an entry of an init table"
    with pytest.raises(ConfigSchemaError, match=message):
        decode_document(LLAMA3_DEBUG, [(bias, "null")])
    body = "model: {config: {init: {linear: {bias: {type: ones}}}}}\n"
    with pytest.raises(ConfigSchemaError, match=message):
        decode_document(_experiment(tmp_path, body), [(bias, "null")])
