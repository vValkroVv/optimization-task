import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

import export_ci
from src.utils.judgment_summary import compute_judgment_summary

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "case",
    ["complete", "missing_question", "missing_verdict", "missing_file", "malformed",
     "record_identity", "mixed_identity", "legacy_mixture"],
)
def test_export_preserves_canonical_scores_and_removes_stale_evidence(
    tmp_path: Path, monkeypatch, write_jsonl, case: str
):
    questions = write_jsonl(
        tmp_path / "questions.jsonl", [{"uid": uid} for uid in ("a", "b", "c")]
    )
    monkeypatch.setattr(export_ci, "get_questions_path", lambda arena: questions)
    config = {
        "datasets": [
            {
                "name": "arena",
                "baseline": {"name": "baseline"},
                "model": {"name": "candidate", "temperature": 0.0, "api_key": "secret"},
                "judge": {"name": "judge", "regex_pattern": r"\[\[([AB<>=]+)\]\]"},
            }
        ]
    }
    config_path = tmp_path / "dredd_merged_config.yaml"
    OmegaConf.save(OmegaConf.create(config), config_path)
    directory = tmp_path / "arenas/arena/judgments/baseline"
    rows = [
        {"uid": "a", "score": "A>B"},
        {"uid": "a", "score": "B>A"},
        {"uid": "b", "games": [{"score": "B>A"}, {"score": "A>B"}]},
        {"uid": "c", "judgment": "[[A=B]]"},
    ]
    if case == "missing_question":
        rows.pop()
    elif case == "missing_verdict":
        rows[-1] = {"uid": "c", "score": "N/A"}
    if case in {"record_identity", "mixed_identity", "legacy_mixture"}:
        for row in rows:
            row["judge_identity"] = "judge-fingerprint"
        if case == "mixed_identity":
            rows[-1]["judge_identity"] = "different-judge"
        elif case == "legacy_mixture":
            rows[-1].pop("judge_identity")
    judgment = write_jsonl(directory / "judgment.jsonl", rows)
    if case == "missing_file":
        judgment.unlink()
    elif case == "malformed":
        judgment.write_text('{"uid": "a", "games": [null]}\n')
    output = directory / "ci_quality.json"
    output.write_text("stale evidence")

    export_ci.export(tmp_path)

    if case not in {"complete", "record_identity"}:
        assert not output.exists()
        return
    data = json.loads(output.read_text())
    assert data["samples"] == [["a", 0.0], ["b", 1.0], ["c", 0.5]]
    assert sum(value for _, value in data["samples"]) / 3 == (
        compute_judgment_summary(judgment)["score_0_100"] / 100
    )
    config["datasets"][0]["model"].update(
        name="another-model", api_key="another-secret"
    )
    OmegaConf.save(OmegaConf.create(config), config_path)
    export_ci.export(tmp_path)
    assert json.loads(output.read_text())["identity"] == data["identity"]
    config["datasets"][0]["judge"]["name"] = "another-judge"
    OmegaConf.save(OmegaConf.create(config), config_path)
    export_ci.export(tmp_path)
    assert (json.loads(output.read_text())["identity"] == data["identity"]) == (case == "record_identity")
    if case == "record_identity":
        for row in rows:
            row["judge_identity"] = "different-judge"
        write_jsonl(judgment, rows)
        export_ci.export(tmp_path)
        assert json.loads(output.read_text())["identity"] != data["identity"]


@pytest.mark.parametrize("hash_key", ["protocol_sha256", "export_manifest_sha256"])
@pytest.mark.parametrize("mixed_question", ["a", "b"])
def test_export_tracks_mean_var_hashes_and_rejects_mixed_bundles(
    tmp_path, monkeypatch, write_jsonl, hash_key, mixed_question
):
    questions = write_jsonl(
        tmp_path / "questions.jsonl", [{"uid": uid} for uid in ("tie", "a", "b")]
    )
    monkeypatch.setattr(export_ci, "get_questions_path", lambda arena: questions)
    OmegaConf.save(OmegaConf.create({"datasets": [{
        "name": "arena", "baseline": {"name": "baseline"},
        "model": {"name": "candidate"},
        "judge": {"name": "gemma4-meanvar", "regex_pattern": r"\[\[([AB<>=]+)\]\]"},
    }]}), tmp_path / "dredd_merged_config.yaml")
    directory = tmp_path / "arenas/arena/judgments/baseline"
    judgment = directory / "judgment.jsonl"
    output = directory / "ci_quality.json"
    rows = [{"uid": "tie", "games": [{"score": "A=B"}]}] + [
        {"uid": uid, "games": [{
            "score": score,
            "mean_var": {"protocol_sha256": "a" * 64, "export_manifest_sha256": "b" * 64},
        } for score in ("B>A", "A>B")]}
        for uid in ("a", "b")
    ]
    write_jsonl(judgment, rows)
    export_ci.export(tmp_path)
    before = json.loads(output.read_text())
    assert before["samples"] == [["a", 1.0], ["b", 1.0], ["tie", 0.5]]
    for row in rows[1:]:
        for game in row["games"]:
            game["mean_var"][hash_key] = "c" * 64
    write_jsonl(judgment, rows)
    export_ci.export(tmp_path)
    after = json.loads(output.read_text())
    assert after["identity"] != before["identity"]
    assert after["samples"] == before["samples"]
    mixed_games = rows[1]["games"][1:] if mixed_question == "a" else rows[2]["games"]
    for game in mixed_games:
        game["mean_var"][hash_key] = "d" * 64
    write_jsonl(judgment, rows)
    export_ci.export(tmp_path)
    assert not output.exists()
