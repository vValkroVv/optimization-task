"""Snapshot compression preserves values and keeps ordinary edits reviewable."""

import copy
import difflib
import json
from types import SimpleNamespace

import pytest
import yaml

from tests.unit_tests import config_snapshot
from tests.unit_tests.config_snapshot import dump_snapshot


def _canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _projection(**changes):
    return {
        "__type__": "linear:linear",
        "in_features": 256,
        "out_features": 512,
        "bias": False,
        "hidden_z_loss": None,
        "init_role": "linear",
        "param_init": {"weight": {"__type__": "param_init:normal", "std": 0.02}},
        "sharding_config": None,
        **changes,
    }


def test_only_equal_subtrees_share_aliases():
    first, second = _projection(), _projection()
    source = [
        first,
        second,
        _projection(out_features=1024),
        _projection(__type__="linear:specialized"),
    ]
    text = dump_snapshot(source)
    restored = yaml.safe_load(text)
    assert _canonical(restored) == _canonical(source)
    assert restored[0] is restored[1]
    assert restored[0] is not restored[2]
    assert restored[0] is not restored[3]
    assert "&linear_linear_" in text and "*linear_linear_" in text
    assert "<<:" not in text and "!" not in text
    assert first is not second


@pytest.mark.parametrize("values", [(True, 1), (False, 0), (1, 1.0), (0.0, -0.0)])
def test_python_equal_scalars_keep_distinct_json_types_and_values(values):
    source = [_projection(option=value) for value in values]
    restored = yaml.safe_load(dump_snapshot(source))
    assert _canonical(restored) == _canonical(source)
    assert restored[0] is not restored[1]


def test_null_missing_fields_order_and_unusual_strings_survive():
    source = {
        "present": None,
        "ordered": [3, 1, 2, None, {}, []],
        "strings": ["null", "true", "on", "012", "2026-10-15", "# text", "", "☃"],
        "<<": {"literal": "<<"},
        "objects": [_projection(option=None), _projection()],
    }
    restored = yaml.safe_load(dump_snapshot(source))
    assert _canonical(restored) == _canonical(source)
    assert "option" in restored["objects"][0]
    assert "option" not in restored["objects"][1]


def test_nested_lists_and_component_types_remain_distinct():
    repeated = [_projection(), {"nested": [None, _projection()]}]
    source = [repeated, copy.deepcopy(repeated), [{"__type__": "different"}]]
    restored = yaml.safe_load(dump_snapshot(source))
    assert _canonical(restored) == _canonical(source)
    assert restored[0] is restored[1]
    assert restored[0][0] is restored[0][1]["nested"][1]
    assert restored[0] is not restored[2]


def test_small_values_do_not_gain_alias_indirection():
    assert "&" not in dump_snapshot([{}, {}, {"value": None}, {"value": None}])


def test_output_is_deterministic_and_does_not_mutate_input():
    source = {"z": _projection(), "a": _projection(), "middle": [2, 1]}
    before = copy.deepcopy(source)
    text = dump_snapshot(source)
    assert source == before
    assert text == dump_snapshot(source)
    assert text == dump_snapshot(dict(reversed(list(source.items()))))
    assert text == dump_snapshot(yaml.safe_load(text))


def test_one_layer_edit_keeps_other_aliases_and_diff_local():
    source = {"layers": [_projection() for _ in range(64)]}
    edited = copy.deepcopy(source)
    edited["layers"][0]["out_features"] = 1024
    before, after = dump_snapshot(source), dump_snapshot(edited)
    difference = list(difflib.unified_diff(before.splitlines(), after.splitlines()))
    changes = [line for line in difference if line[:1] in ("+", "-")][2:]
    assert len(changes) < 30
    before_names = set(
        line.strip() for line in before.splitlines() if "*linear_linear_" in line
    )
    after_names = set(
        line.strip() for line in after.splitlines() if "*linear_linear_" in line
    )
    assert before_names <= after_names
    assert _canonical(yaml.safe_load(after)) == _canonical(edited)


def test_anchor_digest_collision_is_rejected(monkeypatch):
    monkeypatch.setattr(
        config_snapshot.hashlib,
        "sha256",
        lambda data: SimpleNamespace(hexdigest=lambda: "0" * 64),
    )
    source = [
        _projection(),
        _projection(),
        _projection(out_features=1024),
        _projection(out_features=1024),
    ]
    with pytest.raises(ValueError, match="snapshot anchor collision"):
        dump_snapshot(source)
