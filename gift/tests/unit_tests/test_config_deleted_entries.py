"""Only deletions that survive the winning merge constrain initializer tables."""

import pytest

from torchtitan.config import load_raw_document
from torchtitan.config import build
from torchtitan.config.pre_submit import run_pre_submit
from tests.unit_tests.torch_free import REPO_ROOT

BASE = REPO_ROOT / "configs/experiments/llama3_debug.yaml"
BIAS = "model.config.init.linear.bias"


def _layers(tmp_path, bodies):
    previous = BASE
    for index, body in enumerate(bodies):
        path = tmp_path / f"layer{index}.yaml"
        path.write_text(f"config_version: 1\nimports: [{previous}]\n{body}\n")
        previous = path
    return str(previous)


@pytest.mark.parametrize(
    "winning",
    [
        "model: {config: {init: {linear: !replace {weight: {type: zeros}}}}}",
        "model: {config: {init: !replace {linear: {weight: {type: zeros}}}}}",
        "model: {config: {init: null}}",
    ],
)
def test_replaced_subtree_discards_historical_deletion(tmp_path, monkeypatch, winning):
    path = _layers(
        tmp_path,
        ["model: {config: {init: {linear: {bias: null}}}}", winning],
    )
    document = load_raw_document(path)
    assert tuple(BIAS.split(".")) not in document.deleted_paths
    # Compare with the same final mapping reached without the discarded layer.
    direct_dir = tmp_path / "direct"
    direct_dir.mkdir()
    direct = _layers(direct_dir, [winning])
    assert document.data == load_raw_document(direct).data
    actual = run_pre_submit(path)
    expected = run_pre_submit(direct)
    assert actual.ok == expected.ok
    assert [(i.path, i.message) for i in actual.report.issues] == [
        (i.path, i.message) for i in expected.report.issues
    ]

    # Exercise the public model-only loader up to its runtime import boundary.
    class RuntimeBoundaryError(Exception):
        pass

    def stop_runtime_import(*args, **kwargs):
        raise RuntimeBoundaryError

    monkeypatch.setattr(build, "_build_model_spec", stop_runtime_import)
    for candidate in (path, direct):
        if expected.ok:
            with pytest.raises(RuntimeBoundaryError):
                build.load_model_spec(candidate)


def test_live_deletion_still_rejected_and_readdition_clears_it(tmp_path):
    path = _layers(tmp_path, ["model: {config: {init: {linear: {bias: null}}}}"])
    assert tuple(BIAS.split(".")) in load_raw_document(path).deleted_paths
    result = run_pre_submit(path)
    assert not result.ok
    assert any("null cannot remove an entry" in i.message for i in result.report.issues)
    result = run_pre_submit(path, [(BIAS, "{type: zeros}")])
    assert result.ok, result.report.issues


def test_later_null_after_replacement_is_still_live(tmp_path):
    path = _layers(
        tmp_path,
        [
            "model: {config: {init: {linear: {bias: null}}}}",
            "model: {config: {init: {linear: !replace {bias: {type: zeros}}}}}",
            "model: {config: {init: {linear: {bias: null}}}}",
        ],
    )
    assert tuple(BIAS.split(".")) in load_raw_document(path).deleted_paths
    assert not run_pre_submit(path).ok


def test_override_replacement_supersedes_imported_deletion(tmp_path):
    path = _layers(tmp_path, ["model: {config: {init: {linear: {bias: null}}}}"])
    result = run_pre_submit(
        path, [("model.config.init.linear", "!replace {weight: {type: zeros}}")]
    )
    assert result.ok, result.report.issues


def test_readded_parent_starts_a_new_mapping(tmp_path):
    path = _layers(
        tmp_path,
        [
            "model: {config: {init: {linear: {bias: null}}}}",
            "model: {config: {init: {linear: null}}}",
            "model: {config: {init: {linear: {weight: {type: zeros}}}}}",
        ],
    )
    assert not any(
        p[:4] == ("model", "config", "init", "linear")
        for p in load_raw_document(path).deleted_paths
    )
    result = run_pre_submit(path)
    assert result.ok, result.report.issues
