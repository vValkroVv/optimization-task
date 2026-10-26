"""YAML mapping validation before configuration merging and typed decoding."""

import logging

import pytest
import yaml

from torchtitan.config.errors import ConfigFileError
from torchtitan.config.loader import load_raw_document, parse_yaml_value
from torchtitan.config.save import dump_yaml


@pytest.mark.parametrize(
    ("content", "key", "first_line", "duplicate_line"),
    [
        ("training: {}\ntraining: {}\n", "training", 1, 2),
        ("training:\n  steps: 10\n  steps: 20\n", "steps", 2, 3),
        ("imports:\n  optimizer: a.yaml\n  optimizer: b.yaml\n", "optimizer", 2, 3),
        ("callbacks:\n  - type: timer\n    type: monitor\n", "type", 2, 3),
        ("training: {steps: 10, steps: 10}\n", "steps", 1, 1),
        ('training: {steps: 10, "steps": 20}\n', "steps", 1, 1),
        ("training:\n  <<: {steps: 10}\n  <<: {seq_len: 20}\n", "<<", 2, 3),
        ("training:\n  <<: &defaults {steps: 10, steps: 20}\n", "steps", 2, 2),
    ],
)
def test_duplicate_yaml_keys_are_errors(
    tmp_path, content, key, first_line, duplicate_line
):
    path = tmp_path / "experiment.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ConfigFileError) as exc_info:
        load_raw_document(str(path))

    message = str(exc_info.value)
    assert str(path) in message
    assert f"duplicate YAML key {key!r}" in message
    assert f"line {first_line}," in message
    assert f"line {duplicate_line}," in message


def test_duplicate_keys_in_imported_yaml_are_errors(tmp_path):
    preset = tmp_path / "preset.yaml"
    preset.write_text("optimizer: {lr: 0.1, lr: 0.2}\n", encoding="utf-8")
    entry = tmp_path / "experiment.yaml"
    entry.write_text("imports: preset.yaml\n", encoding="utf-8")

    with pytest.raises(ConfigFileError) as exc_info:
        load_raw_document(str(entry))

    assert str(preset) in str(exc_info.value)
    assert "duplicate YAML key 'lr'" in str(exc_info.value)


def test_yaml_merge_defaults_and_shared_aliases_remain_supported(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text(
        "defaults: &defaults {steps: 10, seq_len: 128}\n"
        "other: &other {steps: 20, batch_size: 2}\n"
        "training: &training {<<: [*defaults, *other], steps: 30}\n"
        "copy: {<<: *training}\n",
        encoding="utf-8",
    )

    data = load_raw_document(str(path)).data

    assert data["training"] == {"steps": 30, "seq_len": 128, "batch_size": 2}
    assert data["copy"] == data["training"]


def test_duplicate_check_does_not_modify_pyyaml_global_loader(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text("training: {}\n", encoding="utf-8")
    load_raw_document(str(path))

    assert yaml.safe_load("steps: 10\nsteps: 20\n") == {"steps": 20}
    assert yaml.safe_load("lr: 1e-4\n") == {"lr": "1e-4"}  # YAML 1.1 as is
    assert yaml.safe_load("a: yes\n") == {"a": True}


def test_exponent_without_a_dot_is_a_float(tmp_path):
    """YAML 1.1 wants ``1.0e-4``; the config loader (YAML 1.2) also reads ``1e-4``."""
    path = tmp_path / "experiment.yaml"
    path.write_text(
        "a: 1e-4\nb: 5E3\nc: -2e+5\nd: 1_000e-3\ne: 1.0e-4\nf: '1e-4'\ng: e5\nh: 1e\n"
        "i: 1.e3\nj: 0x1e5\n",
        encoding="utf-8",
    )

    data = load_raw_document(str(path)).data

    assert data == {
        "a": 1e-4,
        "b": 5e3,
        "c": -2e5,
        "d": "1_000e-3",  # YAML 1.2 numbers have no digit separators
        "e": 1e-4,
        "f": "1e-4",  # quoted: a string, as always
        "g": "e5",
        "h": "1e",
        "i": 1e3,
        "j": 0x1E5,  # hexadecimal: the int resolver comes first
    }


@pytest.mark.parametrize(
    "text, expected",
    [
        # Plain scalars that YAML 1.1 (PyYAML's default) reads differently.
        ("yes", "yes"),
        ("on", "on"),
        ("Off", "Off"),
        ("010", 10),
        ("08", 8),
        ("0o10", 8),
        ("0x10", 16),
        ("-0x1F", "-0x1F"),
        ("0b101", "0b101"),
        ("1_000", "1_000"),
        ("12:30", "12:30"),
        ("2024-01-02", "2024-01-02"),
        ("5E3", 5e3),
        # Read alike by both.
        ("true", True),
        ("FALSE", False),
        ("~", None),
        ("Null", None),
        (".inf", float("inf")),
        ("-.5", -0.5),
        ("+1", 1),
    ],
)
def test_values_read_by_the_yaml_1_2_core_schema(text, expected):
    value = parse_yaml_value(text, source="test")
    assert value == expected and type(value) is type(expected)


def test_merge_keys_still_work(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text(
        "base: &base {steps: 1, seed: 2}\ntraining: {<<: *base, steps: 3}\n",
        encoding="utf-8",
    )
    assert load_raw_document(str(path)).data["training"] == {"steps": 3, "seed": 2}


def test_a_value_yaml_cannot_construct_is_a_config_error(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text("dump_folder: !!int a\n", encoding="utf-8")

    with pytest.raises(ConfigFileError, match="invalid YAML: invalid literal"):
        load_raw_document(str(path))


def test_a_dumped_document_loads_back_identically(tmp_path):
    """A string the loader would read bare as a float is quoted when dumped."""
    data = {"a": "1e3", "b": "3E-4", "c": 1e-05, "d": "1.0e-4", "e": {"1e5": "key"}}
    # Strings that YAML 1.1 or YAML 1.2 reads bare as something else.
    data |= {k: k for k in ("yes", "0o10", "010", "1:30", "2024-01-02", "1_000")}
    path = tmp_path / "resolved.yaml"
    path.write_text(dump_yaml(data), encoding="utf-8")

    assert load_raw_document(str(path)).data == data
    # A bundle reads back the same under PyYAML's YAML 1.1 loader.
    assert yaml.safe_load(path.read_text(encoding="utf-8")) == data
    assert yaml.safe_dump({"a": "1e3"}) == "a: 1e3\n"  # SafeDumper as is


# ---------------------------------------------------------------------------
# file-level errors are config errors that name the file
# ---------------------------------------------------------------------------


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return str(path)


def test_missing_file_is_a_config_error(tmp_path):
    missing = tmp_path / "nope.yaml"
    with pytest.raises(ConfigFileError, match=f"config file not found: {missing}"):
        load_raw_document(str(missing))


def test_directory_is_a_config_error(tmp_path):
    with pytest.raises(ConfigFileError, match="is a directory, not a config file"):
        load_raw_document(str(tmp_path))


def test_invalid_utf8_is_a_config_error(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_bytes(b"training: \xff\xfe\n")
    with pytest.raises(
        ConfigFileError, match=f"{path}: config file is not valid UTF-8"
    ):
        load_raw_document(str(path))


def test_unreadable_file_is_a_config_error(tmp_path, monkeypatch):
    path = write(tmp_path / "secret.yaml", "training: {}\n")
    real_open = open

    def denied_open(file, *args, **kwargs):
        if str(file) == path:
            raise PermissionError(13, "Permission denied")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr("builtins.open", denied_open)
    with pytest.raises(ConfigFileError, match="cannot read config file") as exc_info:
        load_raw_document(path)
    assert path in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, PermissionError)


def test_imported_bad_file_names_the_declaring_file_and_the_import(
    tmp_path, monkeypatch
):
    # Declaring files use the shorter cwd-relative name; resolved imports stay explicit.
    monkeypatch.chdir(tmp_path)
    bad = tmp_path / "presets" / "bad.yaml"
    bad.parent.mkdir()
    bad.write_bytes(b"training: \xff\n")
    entry = write(tmp_path / "exp.yaml", "imports: presets/bad.yaml\n")

    with pytest.raises(ConfigFileError) as exc_info:
        load_raw_document(entry)

    message = str(exc_info.value)
    assert message.startswith(
        f"exp.yaml: import 'presets/bad.yaml' (resolved to {bad}): "
    )
    assert message.endswith(f"{bad}: config file is not valid UTF-8")
    # a nested import error is wrapped once, by the file that declares it
    write(tmp_path / "middle.yaml", "imports: presets/bad.yaml\n")
    outer = write(tmp_path / "outer.yaml", "imports: middle.yaml\n")
    with pytest.raises(ConfigFileError) as exc_info:
        load_raw_document(outer)
    assert str(exc_info.value).count("import ") == 1
    assert str(exc_info.value).startswith("middle.yaml: import 'presets/bad.yaml'")


@pytest.mark.parametrize(
    ("content", "described"),
    [
        ("7: seven\n", "int key 7"),
        ("true: 1\n", "bool key True"),
        ("null: 1\n", "NoneType key None"),
    ],
)
def test_non_string_root_keys_are_errors(tmp_path, content, described):
    path = write(tmp_path / "exp.yaml", "optimizer: {}\n" + content)
    with pytest.raises(ConfigFileError) as exc_info:
        load_raw_document(path)
    assert str(exc_info.value) == (
        f"{path}: top-level config keys must be strings; got {described}"
    )


def test_nested_non_string_keys_are_left_to_the_schema(tmp_path):
    path = write(tmp_path / "exp.yaml", "training: {7: seven}\n")
    assert load_raw_document(path).data == {"training": {7: "seven"}}


@pytest.mark.parametrize(
    ("content", "key_path"),
    [
        ("training: &self\n  nested: *self\n", "training.nested"),
        ("training:\n  steps: 1\n  keep: &l [1, *l]\n", "training.keep[1]"),
        ("a: &a\n  b: {c: [*a]}\n", "a.b.c[0]"),
    ],
)
def test_alias_cycles_are_errors(tmp_path, content, key_path):
    path = write(tmp_path / "exp.yaml", content)
    with pytest.raises(ConfigFileError) as exc_info:
        load_raw_document(path)
    assert str(exc_info.value) == (
        f"{path}: recursive YAML structure (alias cycle) at {key_path}"
    )


def test_merge_key_self_reference_is_not_a_cycle(tmp_path):
    """PyYAML expands `<<: *x` into x's own entries, which does not nest x in
    itself; the loader keeps whatever PyYAML constructed."""
    path = write(tmp_path / "exp.yaml", "training: &x {<<: *x, steps: 1}\n")
    assert load_raw_document(path).data == {"training": {"steps": 1}}


def test_shared_aliases_are_not_cycles(tmp_path):
    path = write(
        tmp_path / "exp.yaml",
        "defaults: &d {steps: 1}\n"
        "keep: &k [1, 2]\n"
        "a: {x: *d, y: *d, lists: [*k, *k]}\n"
        "b: *d\n",
    )
    data = load_raw_document(path).data
    assert data["a"] == {
        "x": {"steps": 1},
        "y": {"steps": 1},
        "lists": [[1, 2], [1, 2]],
    }
    assert data["b"] == {"steps": 1}


@pytest.mark.parametrize(
    ("content", "where"),
    [
        ("imports: ''\n", "imports"),
        ("imports: ['']\n", r"imports\[0\]"),
        ("imports: [a.yaml, '  ']\n", r"imports\[1\]"),
        ("imports: {optimizer: ''}\n", "imports.optimizer"),
    ],
)
def test_empty_import_paths_are_errors(tmp_path, content, where):
    write(tmp_path / "a.yaml", "optimizer: {}\n")
    path = write(tmp_path / "exp.yaml", content)
    with pytest.raises(
        ConfigFileError, match=f"{where}: import path must not be empty"
    ):
        load_raw_document(path)


# ---------------------------------------------------------------------------
# imports flatten to one ordered list of layers; one left-to-right merge
# ---------------------------------------------------------------------------


@pytest.fixture
def base(tmp_path):
    write(
        tmp_path / "base.yaml",
        "training: {steps: 99, seq_len: 128}\noptimizer: {lr: 1}\n",
    )
    write(tmp_path / "reset.yaml", "training: null\n")
    return tmp_path


def test_root_reset_from_a_later_import(base):
    entry = write(base / "exp.yaml", "imports: [base.yaml, reset.yaml]\n")
    assert load_raw_document(entry).data == {"optimizer": {"lr": 1}}


def test_nested_reset_in_a_child_that_imports_the_base(base):
    write(base / "child.yaml", "imports: base.yaml\ntraining: {steps: null}\n")
    entry = write(base / "exp.yaml", "imports: [base.yaml, child.yaml]\n")
    assert load_raw_document(entry).data == {
        "training": {"seq_len": 128},
        "optimizer": {"lr": 1},
    }


def test_reset_then_partial_in_a_child(base):
    """The child resets `training` through its own import and then sets one
    field: nothing from the outer base may leak back into the section."""
    write(base / "child.yaml", "imports: reset.yaml\ntraining: {steps: 7}\n")
    entry = write(base / "exp.yaml", "imports: [base.yaml, child.yaml]\n")
    assert load_raw_document(entry).data == {
        "training": {"steps": 7},
        "optimizer": {"lr": 1},
    }


def test_reset_then_empty_mapping(base):
    write(base / "child.yaml", "imports: reset.yaml\ntraining: {}\n")
    entry = write(base / "exp.yaml", "imports: [base.yaml, child.yaml]\n")
    assert load_raw_document(entry).data == {"training": {}, "optimizer": {"lr": 1}}


def test_later_explicit_value_after_a_reset(base):
    write(base / "refill.yaml", "training: {seq_len: 128}\n")
    entry = write(
        base / "exp.yaml",
        "imports: [base.yaml, reset.yaml, refill.yaml]\ntraining: {steps: 5}\n",
    )
    assert load_raw_document(entry).data == {
        "training": {"steps": 5, "seq_len": 128},
        "optimizer": {"lr": 1},
    }
    # Importing base.yaml again instead would undo the reset: an error.
    entry = write(base / "exp.yaml", "imports: [base.yaml, reset.yaml, base.yaml]\n")
    with pytest.raises(ConfigFileError, match="would change training,"):
        load_raw_document(entry)


def test_section_import_of_a_null_section_imports_the_reset(base):
    write(base / "mix.yaml", "imports: {training: reset.yaml}\n")
    entry = write(base / "exp.yaml", "imports: [base.yaml, mix.yaml]\n")
    assert load_raw_document(entry).data == {"optimizer": {"lr": 1}}
    # a section the file never mentions is still an error
    write(base / "other.yaml", "optimizer: {lr: 2}\n")
    entry = write(base / "exp2.yaml", "imports: {training: other.yaml}\n")
    with pytest.raises(
        ConfigFileError, match="has no 'training' section; top-level keys: optimizer"
    ):
        load_raw_document(entry)


def test_reset_inside_a_section_import_travels(base):
    """A `null` in a sectioned file's tree pops the key the outer document
    inherited from an earlier import."""
    write(base / "run.yaml", "imports: base.yaml\ntraining: {seq_len: null}\n")
    write(base / "outer.yaml", "training: {seq_len: 4096, steps: 1}\n")
    write(base / "mix.yaml", "imports: {training: run.yaml}\n")
    entry = write(base / "exp.yaml", "imports: [outer.yaml, mix.yaml]\n")
    assert load_raw_document(entry).data == {"training": {"steps": 99}}


def test_scalar_replaced_then_remapped(base):
    """No `null` involved: a child replaces the section with a scalar through
    its import and re-creates it as a mapping; the outer base must not merge
    back into the new mapping."""
    write(base / "s1.yaml", "training: 5\n")
    write(base / "s2.yaml", "imports: s1.yaml\ntraining: {steps: 1}\n")
    entry = write(base / "exp.yaml", "imports: [base.yaml, s2.yaml]\n")
    assert load_raw_document(entry).data == {
        "training": {"steps": 1},
        "optimizer": {"lr": 1},
    }


def test_null_fields_of_mappings_inside_lists_are_dropped(tmp_path):
    path = write(
        tmp_path / "exp.yaml",
        "model: {converters: [{type: lora, rank: null}, {type: fp8}]}\n"
        "items: [null, {a: null}, [null, {b: null, c: 1}], 3]\n",
    )
    assert load_raw_document(path).data == {
        "model": {"converters": [{"type": "lora"}, {"type": "fp8"}]},
        "items": [None, {}, [None, {"c": 1}], 3],
    }


# ---------------------------------------------------------------------------
# missing_cleared_paths: the resets that deleted nothing
# ---------------------------------------------------------------------------


def test_missing_cleared_paths_report_resets_that_found_nothing(tmp_path):
    path = write(
        tmp_path / "exp.yaml",
        "training: {stpes: null, steps: 3}\n"
        "trainng: null\n"
        "model: {converters: [{type: lora, rank: null}]}\n"
        "items: [null, {a: null}]\n",
    )
    document = load_raw_document(path)
    assert document.missing_cleared_paths == (
        ("training", "stpes"),
        ("trainng",),
        ("model", "converters", 0, "rank"),
        ("items", 1, "a"),
    )
    assert document.data == {
        "training": {"steps": 3},
        "model": {"converters": [{"type": "lora"}]},
        "items": [None, {}],
    }


def test_missing_cleared_paths_skip_resets_that_popped_or_were_overwritten(base):
    """A reset that popped an inherited key, or that a later layer set
    again, has an effect (or none at all) and is not reported."""
    write(base / "child.yaml", "training: {steps: null, seq_len: null}\n")
    write(base / "child2.yaml", "training: {steps: null, seq_len: null}\n")
    write(base / "again.yaml", "training: {seq_len: 64}\n")
    entry = write(base / "exp.yaml", "imports: [child.yaml, base.yaml, child2.yaml]\n")
    assert load_raw_document(entry).missing_cleared_paths == ()
    entry = write(base / "exp2.yaml", "imports: [child.yaml, again.yaml]\n")
    document = load_raw_document(entry)
    assert document.missing_cleared_paths == (("training", "steps"),)
    assert document.data == {"training": {"seq_len": 64}}


def test_missing_cleared_paths_include_absent_keys_of_inherited_mappings(base):
    """The parent mapping exists (inherited ``training``, the root) but the
    key does not: ``deep_merge`` records the path."""
    write(base / "child.yaml", "training: {stpes: null}\ntrainng: null\n")
    entry = write(base / "exp.yaml", "imports: [base.yaml, child.yaml]\n")
    document = load_raw_document(entry)
    assert document.missing_cleared_paths == (("training", "stpes"), ("trainng",))
    assert document.data == {
        "training": {"steps": 99, "seq_len": 128},
        "optimizer": {"lr": 1},
    }
    assert load_raw_document(base / "base.yaml").missing_cleared_paths == ()


def test_repeated_full_import_that_changes_nothing_is_allowed(base):
    """A preset layered on a preset the experiment also imports changes nothing
    on the repeat, so it is allowed and the graph records both occurrences."""
    write(base / "layer.yaml", "imports: base.yaml\ntraining: {steps: 7}\n")
    entry = write(base / "exp.yaml", "imports: [base.yaml, layer.yaml]\n")
    document = load_raw_document(entry)
    assert document.data["training"] == {"steps": 7, "seq_len": 128}
    assert [n.path.endswith("base.yaml") for n in document.graph.imports] == [
        True,
        False,
    ]
    assert document.graph.imports[1].imports[0].path.endswith("base.yaml")


@pytest.mark.parametrize(
    "mid_values,changed",
    [
        ("training: {steps: 8}", "training.steps"),
        # A reset between the two occurrences would be undone as well.
        ("optimizer: null", "optimizer"),
    ],
)
def test_repeated_full_import_that_would_undo_a_layer_is_an_error(
    base, mid_values, changed
):
    """`exp: [mid, base]` with `mid: [base] + overrides`: merging base again
    would silently revert mid's values; the error names the keys and both
    importers."""
    write(base / "mid.yaml", f"imports: base.yaml\n{mid_values}\n")
    entry = write(base / "exp.yaml", "imports: [mid.yaml, base.yaml]\n")
    with pytest.raises(ConfigFileError) as excinfo:
        load_raw_document(entry)
    message = str(excinfo.value)
    assert "base.yaml is imported by" in message
    assert "mid.yaml and again by" in message and "exp.yaml" in message
    assert f"would change {changed}," in message
    # The entry file or the command line may still set the key after it.
    write(base / "exp.yaml", f"imports: [mid.yaml, base.yaml]\n{mid_values}\n")
    load_raw_document(entry)


def test_same_file_for_two_section_imports_is_not_a_repeat(base, caplog):
    """`imports: {a: run.yaml, b: run.yaml}` is how several sections are taken
    out of one run; nothing overlaps, so nothing is logged. Repeats inside
    each projected sub-tree do not count either."""
    write(base / "run.yaml", "imports: base.yaml\nlr_scheduler: {warmup_steps: 7}\n")
    entry = write(
        base / "exp.yaml",
        "imports: {lr_scheduler: run.yaml, optimizer: run.yaml}\n",
    )
    with caplog.at_level(logging.WARNING, logger="torchtitan.config.loader"):
        document = load_raw_document(entry)
    assert caplog.records == []
    assert document.data == {
        "lr_scheduler": {"warmup_steps": 7},
        "optimizer": {"lr": 1},
    }
