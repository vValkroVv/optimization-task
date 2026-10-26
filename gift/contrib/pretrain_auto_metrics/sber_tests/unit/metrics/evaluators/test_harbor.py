import json

import pytest

from node_runner.src.metrics.evaluators.harbor import task_rewards

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("binary", "expected"), [(False, {"a": [0.4, 1.0]}), (True, {"a": [0.0, 1.0]})]
)
def test_complete_trials_keep_task_grouping(tmp_path, binary, expected):
    (tmp_path / "result.json").write_text('{"n_total_trials": 2}')
    for i, reward in enumerate([0.4, 1.0]):
        trial = tmp_path / str(i)
        trial.mkdir()
        (trial / "result.json").write_text(
            json.dumps(
                {
                    "task_name": "a",
                    "verifier_result": {
                        "rewards": {"score": reward, "label": "ignored"}
                    },
                }
            )
        )
    assert task_rewards(tmp_path / "result.json", 2, 2, binary=binary) == expected


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("verifier_result", [None, {}, {"rewards": {"reward": None}}])
def test_missing_rewards_count_as_zero(tmp_path, binary, verifier_result):
    path = tmp_path / "result.json"
    path.write_text('{"n_total_trials": 1}')
    trial = tmp_path / "trial"
    trial.mkdir()
    (trial / "result.json").write_text(
        json.dumps({"task_name": "a", "verifier_result": verifier_result})
    )
    assert task_rewards(path, 1, 1, binary=binary) == {"a": [0.0]}


@pytest.mark.parametrize(
    ("total", "attempts", "expected", "trials", "message"),
    [
        (0, 1, 0, [], "different trial population"),
        (2, 1, 1, [], "different trial population"),
        (1, 1, 1, [{"score": "invalid"}], "ungraded or invalid"),
        (1, 1, 1, [{"score": float("inf")}], "ungraded or invalid"),
        (2, 2, 2, [{"score": 1}], "incomplete"),
        (1, 2, 1, [{"score": 1}], "incomplete"),
    ],
)
def test_incomplete_or_ungraded_trials_are_rejected(
    tmp_path, total, attempts, expected, trials, message
):
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"n_total_trials": total}))
    for i, rewards in enumerate(trials):
        trial = tmp_path / str(i)
        trial.mkdir()
        (trial / "result.json").write_text(
            json.dumps({"task_name": "a", "verifier_result": {"rewards": rewards}})
        )
    with pytest.raises(ValueError, match=message):
        task_rewards(path, attempts, expected)
