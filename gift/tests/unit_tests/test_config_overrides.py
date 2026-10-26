"""CLI overrides are YAML layers merged after the experiment file.

A command-line value means what the same text means in a YAML file.
"""

import pytest

from tests.unit_tests.config_helpers import decode_document
from tests.unit_tests.torch_free import REPO_ROOT
from torchtitan.config import load_document
from torchtitan.config.errors import ConfigSchemaError
from torchtitan.config.overrides import build_override_layers

EXPERIMENTS = REPO_ROOT / "configs" / "experiments"


def _load(experiment: str, *overrides: tuple[str, str]) -> dict:
    document, _ = load_document(str(EXPERIMENTS / experiment), list(overrides))
    return document.data


def _decode(experiment: str, *overrides: tuple[str, str]) -> dict:
    return decode_document(EXPERIMENTS / experiment, list(overrides))


def _get(data: dict, path: str):
    for part in path.split("."):
        data = data[part]
    return data


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2000", 2000),
        ("1.0e-4", 1.0e-4),
        ("off", "off"),  # YAML 1.2: only true / false are booleans
        ("yes", "yes"),
        ("010", 10),  # YAML 1.2: decimal, not octal
        ("hello", "hello"),
        ("a,b,c", ["a", "b", "c"]),
        ("1,2", [1, 2]),
        ("1F1B", "1F1B"),
        ("null", None),
    ],
)
def test_override_value_reads_as_yaml(raw, expected):
    assert build_override_layers([("x", raw)])[1]["x"] == expected


@pytest.mark.parametrize(
    "path,raw,expected",
    [
        # A string YAML would read as something else is quoted, as in a file.
        ("dump_folder", "'2024'", "2024"),
        ("dump_folder", '"a,b"', "a,b"),
        ("dump_folder", "''", ""),
        ("dump_folder", "./out/run1", "./out/run1"),
        ("optimizer.lr", "5e-5", 5e-5),  # ConfigYAMLLoader: 1e-4 is a float
        ("training.steps", "010", 10),
        ("dump_folder", "12:30", "12:30"),  # YAML 1.2: no base-60 ints
        ("dump_folder", "2024-02-30", "2024-02-30"),  # nor timestamps
        ("compile.enable", "true", True),
        ("compile.enable", "False", False),
        # A bare comma is a flow list; spaces around the items do not survive.
        ("compile.components", "model,loss", ["model", "loss"]),
        ("compile.components", "model, loss", ["model", "loss"]),
        ("compile.components", '[model, "a,b"]', ["model", "a,b"]),
        ("parallelism.context_parallel_load_balancer", "disable", "disable"),
        ("model.config.dim", "512", 512),
    ],
)
def test_the_decode_takes_the_yaml_value(path, raw, expected):
    overrides = [(path, raw)]
    if path == "model.config.dim":
        # The explicit preset RoPE width must track the new per-head width.
        overrides.append(("model.config.rope.dim", "32"))
    assert _get(_decode("llama3_debug.yaml", *overrides), path) == expected


@pytest.mark.parametrize(
    "path,raw,message",
    [
        ("dump_folder", "2024", r"expected a string, got 2024; quote it \('...'\)"),
        ("training.steps", "1_000", "expected an int, got '1_000'"),  # YAML 1.2
        ("compile.enable", "yes", "expected a bool"),
        ("dump_folder", "a,b", r"expected a string, got \['a', 'b'\]"),
        ("dump_folder", "true", "expected a string, got True"),
        ("training.steps", "5e-5", "expected an int"),
        ("optimizer.lr", "fast", "expected a float"),
        ("parallelism", "tp", "expected a mapping"),
    ],
)
def test_the_decode_rejects_a_value_of_the_wrong_type(path, raw, message):
    with pytest.raises(ConfigSchemaError, match=rf"^{path}: {message}"):
        _decode("llama3_debug.yaml", (path, raw))


@pytest.mark.parametrize(
    "path,raw,message",
    [
        ("training.steps", "", "training.steps: empty command-line value"),
        ("dump_folder", "#tag", "dump_folder: '#tag' reads as null"),
        ("dump_folder", "*x", r"dump_folder: '\*x' is not a YAML value"),
        ("dump_folder", "!!int a", "is not a YAML value"),
        ("dump_folder", "&a [*a]", "alias cycle"),
        ("model.config.converters[0].rank", "8", "malformed override path"),
        ("training..steps", "8", "malformed override path"),
    ],
)
def test_a_malformed_override_is_a_config_error(path, raw, message):
    with pytest.raises(ConfigSchemaError, match=message):
        _load("llama3_debug.yaml", (path, raw))


def test_null_resets_whatever_the_type():
    data = _decode(
        "llama3_debug.yaml", ("dump_folder", "null"), ("training.steps", "~")
    )
    assert "dump_folder" not in data
    assert "steps" not in data["training"]


def test_a_flow_mapping_merges_into_the_section():
    """As in YAML: the siblings stay, a null inside resets only its key."""
    base = _load("llama3_debug.yaml")["training"]
    data = _decode("llama3_debug.yaml", ("training", "{steps: null, seq_len: 64}"))
    assert data["training"] == {
        **{key: value for key, value in base.items() if key != "steps"},
        "seq_len": 64,
    }


def test_overrides_fold_in_argv_order():
    data = _decode(
        "llama3_debug.yaml", ("training.steps", "5"), ("training.steps", "null")
    )
    assert "steps" not in data["training"]
    data = _decode(
        "llama3_debug.yaml", ("training", "{steps: 7}"), ("training.steps", "9")
    )
    assert data["training"]["steps"] == 9


def test_raw_override_values_are_not_coerced_before_decode():
    """The raw loader preserves YAML values; typed decode validates them later."""
    base = "model.config.full_attention.inner_attention.kernel_options"
    data = _load(
        "llama3_debug_flex_attn.yaml",
        (f"{base}.BLOCKS", "[1,2]"),
        (f"{base}.BLOCK_M", "64"),
        (f"{base}.NAMES", "a,b"),
        (f"{base}.LABEL", '"a,b"'),
        (f"{base}.MODE", "nan"),
    )
    assert _get(data, base) == {
        "BLOCKS": [1, 2],
        "BLOCK_M": 64,
        "NAMES": ["a", "b"],
        "LABEL": "a,b",
        "MODE": "nan",
    }


def test_override_order_does_not_type_the_value():
    """Nothing is typed when the overrides are applied, so ``--x.type`` may
    come before or after the fields of the class it selects."""
    for overrides in (
        [("dataloader.type", "hf_multimodal"), ("dataloader.video_dir", "'2024'")],
        [("dataloader.video_dir", "'2024'"), ("dataloader.type", "hf_multimodal")],
    ):
        data = _load("llama3_debug.yaml", *overrides)
        assert data["dataloader"]["video_dir"] == "2024"


def test_applied_overrides_are_flat_and_ordered():
    override_layers, applied_overrides = build_override_layers(
        [
            ("training.steps", "5"),
            ("compile.enable", "true"),
            ("training.steps", "null"),
        ]
    )
    assert override_layers == [
        {"training": {"steps": 5}},
        {"compile": {"enable": True}},
        {"training": {"steps": None}},
    ]
    # Last position for a repeated path: replaying in order gives the document.
    assert list(applied_overrides.items()) == [
        ("compile.enable", True),
        ("training.steps", None),
    ]
    _, applied_overrides = build_override_layers(
        [("training.steps", "1"), ("training", "null"), ("training.steps", "2")]
    )
    assert list(applied_overrides.items()) == [
        ("training", None),
        ("training.steps", 2),
    ]


def test_cli_resets_join_the_missing_cleared_paths():
    """The loader reports the command line's resets of nothing like a file's."""
    document, _ = load_document(
        str(EXPERIMENTS / "llama3_debug.yaml"),
        [("training.seq_len", "null"), ("optimizer", "{lr: 0.1, eps: null}")],
    )
    assert "seq_len" not in document.data["training"]  # existed: popped
    assert document.missing_cleared_paths == (("optimizer", "eps"),)
    assert document.data["optimizer"]["lr"] == 0.1
