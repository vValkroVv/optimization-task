"""Golden changes expose field-level JSON differences instead of opaque hashes."""

import yaml

import pytest

from tests.unit_tests import config_golden


def test_snapshots_retain_unset_options_and_report_readable_changes(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(config_golden, "GOLDEN_DIR", tmp_path)
    monkeypatch.setenv(config_golden.UPDATE_ENV, "1")
    key = "experiments/example.yaml"
    config_golden.check_golden(key, {"width": 4, "optional": None})
    path = tmp_path / "experiments/example.yaml"
    assert yaml.safe_load(path.read_text()) == {"width": 4, "optional": None}
    assert key in config_golden.load_golden()
    monkeypatch.delenv(config_golden.UPDATE_ENV)
    config_golden.check_golden(key, {"width": 4, "optional": None})
    with pytest.raises(AssertionError, match='"width": 8'):
        config_golden.check_golden(key, {"width": 8, "optional": None})
    with pytest.raises(AssertionError, match='"new_option": null'):
        config_golden.check_golden(
            key, {"width": 4, "optional": None, "new_option": None}
        )


def test_experiment_snapshot_includes_declared_and_materialized_config():
    snapshot = config_golden.resolve("experiments/llama3_debug.yaml")
    assert snapshot["trainer"].model_spec is None
    assert not snapshot["model"].config.materialized
    assert snapshot["materialized_model"].materialized
    assert (
        len(snapshot["materialized_model"].layers) == snapshot["model"].config.n_layers
    )


def test_aliases_expand_before_comparison_and_failure_diagnostics(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(config_golden, "GOLDEN_DIR", tmp_path)
    monkeypatch.setenv(config_golden.UPDATE_ENV, "1")
    key = "experiments/aliased.yaml"
    layer = {
        "__type__": "example:layer",
        "width": 256,
        "optional": None,
        "description": "A repeated layer with enough content to benefit from a YAML alias.",
    }
    config_golden.check_golden(key, {"layers": [dict(layer), dict(layer)]})
    stored = (tmp_path / key).read_text()
    assert "&example_layer_" in stored and "*example_layer_" in stored
    monkeypatch.delenv(config_golden.UPDATE_ENV)
    config_golden.check_golden(key, {"layers": [dict(layer), dict(layer)]})
    with pytest.raises(AssertionError, match='"width": 512') as failure:
        config_golden.check_golden(
            key, {"layers": [dict(layer), {**layer, "width": 512}]}
        )
    assert "*example_layer_" not in str(failure.value)
