"""Export question scores using DREDD's own order/deduplication policy."""

import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

from omegaconf import OmegaConf

from dredd import get_questions_path
from src.judges.runner import JudgeRunner
from src.utils.io import load_jsonl
from src.utils.judgment_summary import compute_judgment_summary


def export(metrics_dir: Path) -> None:
    config = OmegaConf.to_container(
        OmegaConf.load(metrics_dir / "dredd_merged_config.yaml"), resolve=True
    )
    for dataset in config["datasets"]:
        arena, baseline = dataset["name"], dataset["baseline"]["name"]
        directory = metrics_dir / "arenas" / arena / "judgments" / baseline
        output = directory / "ci_quality.json"
        output.unlink(missing_ok=True)
        try:
            questions_path = get_questions_path(arena)
            expected = {
                str(row.get("uid") or row["question_id"])
                for row in load_jsonl(str(questions_path))
            }
            rows = load_jsonl(str(directory / "judgment.jsonl"))
            by_question = {}
            for row in rows:
                uid = str(row.get("uid") or row.get("question_id") or "")
                by_question.setdefault(uid, []).append(row)
            if not expected or set(by_question) != expected:
                print(
                    f"DREDD CI unavailable for {arena}: incomplete question population"
                )
                continue
            pattern = re.compile(dataset["judge"]["regex_pattern"])
            valid = True
            judge_hashes = set()
            judge_identities = set()
            for records in by_question.values():
                latest = records[-1]
                judge_identities.add(latest.get("judge_identity"))
                games = latest.get("games", [latest])
                judge_hashes.update(
                    (game["mean_var"]["protocol_sha256"], game["mean_var"]["export_manifest_sha256"])
                    for game in games[:2] if "mean_var" in game
                )
                if not games or len(judge_hashes) > 1 or len(judge_identities) > 1 or any(
                    game.get("score") == "N/A"
                    or JudgeRunner._extract_verdict_from_game(game, pattern) is None
                    for game in games[:2]
                ):
                    valid = False
                    break
            if not valid:
                print(f"DREDD CI unavailable for {arena}: missing judge verdicts or mixed judge hashes")
                continue
            identity = {
                "questions": hashlib.sha256(questions_path.read_bytes()).hexdigest(),
                "baseline": baseline,
            }
            if judge_hashes:
                identity["mean_var"] = next(iter(judge_hashes))
            for role in ("model", "judge"):
                identity[role] = {
                    key: value
                    for key, value in dataset[role].items()
                    if key
                    not in {
                        "api_base",
                        "api_key",
                        "parallel_api_calls",
                        "path",
                        "artifacts_dir",
                        *(("name",) if role == "model" else ()),
                    }
                }
            if judge_identities != {None}:
                identity["judge"] = next(iter(judge_identities))
            samples = []
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "question.jsonl"
                for uid, records in sorted(by_question.items()):
                    path.write_text("".join(json.dumps(row) + "\n" for row in records))
                    result = compute_judgment_summary(
                        path, regex_pattern=dataset["judge"]["regex_pattern"]
                    )
                    samples.append((uid, result["score_0_100"] / 100))
            output.write_text(
                json.dumps(
                    {
                        "samples": samples,
                        "identity": hashlib.sha256(
                            json.dumps(identity, sort_keys=True).encode()
                        ).hexdigest(),
                    }
                )
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(f"DREDD CI unavailable for {arena}: {error}")


if __name__ == "__main__":
    export(Path(sys.argv[1]))
