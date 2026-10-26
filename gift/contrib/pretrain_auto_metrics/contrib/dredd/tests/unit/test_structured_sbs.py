import json
from pathlib import Path

import pytest
import requests

pytestmark = pytest.mark.unit


def test_pair_uses_router_and_server_reverse(structured_runner, structured_http, structured_result, read_jsonl):
    assert structured_runner.run() == {"succeeded": 1, "failed": 0, "skipped": False}
    structured_http.assert_called_once()
    call = structured_http.call_args
    assert call.args == ("http://judge-router/v1/chat/completions",)
    assert call.kwargs["timeout"] == 42
    payload = call.kwargs["json"]
    assert set(payload) == {"model", "messages"}
    assert payload["model"] == "gemma4-meanvar"
    assert len(payload["messages"]) == 1
    assert payload["messages"][0]["role"] == "user"
    pair = json.loads(payload["messages"][0]["content"])
    assert pair == {
        "schema": "gemma-mean-var-v1", "question": '"quoted"\nВопрос 🐈',
        "answer_a": "B", "answer_b": "C", "domain": "science",
    }
    row, = read_jsonl(structured_runner.output_file)
    assert row["category"] == "test"
    assert [g["score"] for g in row["games"]] == ["B>A", "A>B"]
    assert [g["role_a"] for g in row["games"]] == ["baseline", "candidate"]
    assert row["games"][0]["answer_a_hash"] == row["games"][1]["answer_b_hash"]
    for i, scores in enumerate((structured_result, structured_result["reverse"])):
        game = row["games"][i]
        assert game["mean_var"]["probabilities"] == scores["probabilities"]
        assert game["mean_var"]["rewards"] == scores["rewards"]
        assert game["mean_var"]["export_manifest_sha256"] == "e"
        assert json.loads(game["prompt"][0]["content"])["answer_a"] == ("B", "C")[i]


@pytest.mark.parametrize("failure", ["json", "verdict", "reverse"])
def test_invalid_response_fails_without_record(structured_runner, structured_http, structured_result, failure):
    if failure == "verdict":
        structured_result["verdict"] = "unknown"
    if failure == "reverse":
        del structured_result["reverse"]
    content = "not json" if failure == "json" else json.dumps(structured_result)
    structured_http.return_value.json.return_value["choices"][0]["message"]["content"] = content
    with pytest.raises(RuntimeError, match="1 failed"):
        structured_runner.run()
    assert not structured_runner.output_file.exists()


def test_transient_error_uses_existing_api_retries(structured_runner, structured_http):
    structured_http.side_effect = [requests.Timeout("queue timeout"), structured_http.return_value]
    assert structured_runner.run()["succeeded"] == 1
    assert structured_http.call_count == 2


@pytest.mark.parametrize("attempts", [1, 3])
def test_exhausted_retries_fail_run(structured_runner, structured_http, attempts):
    structured_runner.judgment_func.__self__.model.num_retries = attempts
    structured_http.side_effect = requests.ConnectionError("router unavailable")
    with pytest.raises(RuntimeError, match="1 failed"):
        structured_runner.run()
    assert structured_http.call_count == attempts
    assert not structured_runner.output_file.exists()


@pytest.mark.parametrize("overwrite", [False, True])
def test_legacy_judgments_are_rejudged(structured_runner, structured_http, write_jsonl, overwrite):
    stale = {"uid": "q1", "games": [{"score": "A>B"}]}
    write_jsonl(structured_runner.output_file, [stale])
    before = structured_runner.output_file.read_bytes()
    generation = structured_runner.candidate_path.read_bytes()
    structured_runner.overwrite = overwrite
    assert structured_runner.run()["succeeded"] == 1
    if not overwrite:
        archived, = structured_runner.output_file.parent.glob("previous-judgments-*/judgment.jsonl")
        assert archived.read_bytes() == before
    assert structured_runner.candidate_path.read_bytes() == generation


@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("case", ["partial", "complete", "judge", "prompt", "legacy", "mixed", "transport", "tail", "newline", "bundle"])
def test_identity_aware_resume(
    structured_runner, structured_http, structured_result, write_jsonl, read_jsonl, mocker, structured, case
):
    runner = structured_runner
    if not structured:
        if case == "bundle":
            pytest.skip("Bundle hashes are specific to structured judging")
        runner.judge.config["input_format"] = "prompt_template"
        runner.judgment_func = mocker.Mock(side_effect=lambda args: write_jsonl(
            Path(args["output_file"]), [{"uid": args["question"]["uid"], "games": [{"score": "A>B"}]}]
        ))
    for path in (runner.questions_path, runner.candidate_path, runner.baseline_path):
        template, = read_jsonl(path)
        write_jsonl(path, [{**template, "uid": f"q{i}"} for i in range(1, 4)])
    generation = runner.candidate_path.read_bytes(), runner.baseline_path.read_bytes()
    runner.run()
    rows = read_jsonl(runner.output_file)
    if case != "complete":
        write_jsonl(runner.output_file, rows[:2])
    if case == "judge":
        runner.judge.model.name = "another-judge"
    elif case == "prompt":
        runner.judge.system_prompt = "Another system prompt"
        runner.judge_settings.clear()
    elif case == "legacy":
        write_jsonl(runner.output_file, [{k: v for k, v in row.items() if k != "judge_identity"} for row in rows[:2]])
    elif case == "mixed":
        write_jsonl(runner.output_file, [{**rows[0], "judge_identity": "another-judge"}, rows[1]])
    elif case == "transport":
        runner.judge.config.update(api_base="http://another-router", request_timeout=777, num_retries=9)
        runner.judge.model.api_base = "http://another-router"
    elif case == "tail":
        with runner.output_file.open("ab") as stream:
            stream.write(b'{"uid": "q3", "text": "\xd0')
    elif case == "newline":
        runner.output_file.write_bytes(runner.output_file.read_bytes().rstrip(b"\n"))
    elif case == "bundle":
        structured_result["export_manifest_sha256"] = "different-export"
        structured_http.return_value.json.return_value["choices"][0]["message"]["content"] = json.dumps(structured_result)
    old_bytes = runner.output_file.read_bytes()
    runner.output_file.with_name("judgment_summary.jsonl").write_text("old summary")
    calls = structured_http if structured else runner.judgment_func
    calls.reset_mock()
    runner.run()
    changed = case in {"judge", "prompt", "legacy", "mixed", "bundle"}
    expected_calls = 3 if changed else (1 if structured else 0) + (case != "complete")
    assert calls.call_count == expected_calls
    actual = read_jsonl(runner.output_file)
    assert {row["uid"] for row in actual} == {"q1", "q2", "q3"}
    assert len({row["judge_identity"] for row in actual}) == 1
    archives = list(runner.output_file.parent.glob("previous-judgments-*"))
    assert bool(archives) == changed
    if changed:
        assert (archives[0] / "judgment.jsonl").read_bytes() == old_bytes
        assert (archives[0] / "judgment_summary.jsonl").read_text() == "old summary"
    else:
        assert rows[1] in actual
    assert generation == (runner.candidate_path.read_bytes(), runner.baseline_path.read_bytes())


def test_mixed_current_bundles_fail(structured_runner, structured_http, structured_result, write_jsonl, read_jsonl):
    runner = structured_runner
    for path in (runner.questions_path, runner.candidate_path, runner.baseline_path):
        row, = read_jsonl(path)
        write_jsonl(path, [row, {**row, "uid": "q2"}])
    replies = [dict(structured_result), {**structured_result, "export_manifest_sha256": "different-export"}]
    structured_http.return_value.json.side_effect = [
        {"choices": [{"message": {"content": json.dumps(reply)}}]} for reply in replies
    ]
    runner.fail_on_error = False
    with pytest.raises(RuntimeError, match="identity changed"):
        runner.run()
    rows = read_jsonl(runner.output_file)
    assert len(rows) == 1
    assert rows[0]["games"][0]["mean_var"]["export_manifest_sha256"] == "e"
