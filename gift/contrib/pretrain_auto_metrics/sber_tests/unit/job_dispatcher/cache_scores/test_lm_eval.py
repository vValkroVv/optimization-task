import json
from pathlib import Path

import pytest

from node_runner.src.job_dispatcher.cache_scores import (
    LmEvalCacheScoresBuilder,
    iter_compressed_jsonl_bytes,
)
from node_runner.src.metrics.evaluators.lm_eval import LmEvalEvaluator

pytestmark = pytest.mark.unit

TIMESTAMP = "2026-08-29T17-23-21.150946"
LATER_TIMESTAMP = "2026-08-30T14-52-59.798671"


def write_run(
    metrics_dir: Path,
    run_name: str,
    timestamp: str,
    results: dict,
    group_subtasks: dict,
    n_shot: dict,
    n_samples: dict | None = None,
    date: float | None = None,
) -> None:
    run = {
        "results": results,
        "group_subtasks": group_subtasks,
        "n-shot": n_shot,
        "n-samples": n_samples or {},
    }
    if date is not None:
        run["date"] = date
    (metrics_dir / f"{run_name}_{timestamp}.json").write_text(
        json.dumps(run), encoding="utf-8"
    )


def write_samples(metrics_dir: Path, task: str, timestamp: str, rows: list) -> None:
    (metrics_dir / f"samples_{task}_{timestamp}.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )


def build(
    metrics_dir: Path, exclude=lambda name: False, remote=None, run_files=None
) -> list:
    evaluator = LmEvalEvaluator()
    evaluator.should_exclude_metric = exclude
    builder = LmEvalCacheScoresBuilder(metrics_dir, evaluator, run_files=run_files)
    if remote is not None:
        builder.existing_rows_loader = lambda: remote
    path = builder.build(metrics_dir)
    return list(iter_compressed_jsonl_bytes(path.read_bytes()))


def remote_row(subset: str, uid: str, score: float, schema_version: int = 1) -> dict:
    return {
        "schema_version": schema_version,
        "scoring_kind": "mean",
        "benchmark": "mmlu/5-shot/acc",
        "subset": subset,
        "uid": uid,
        "metric_name": "acc",
        "score": score,
    }


def test_group_rows_carry_the_subtask_as_subset(tmp_path: Path) -> None:
    write_run(
        tmp_path,
        "mmlu",
        TIMESTAMP,
        results={
            "mmlu": {"acc,none": 2 / 3, "acc_stderr,none": 0.1, "alias": "mmlu"},
            "mmlu_management": {"acc,none": 1.0},
            "mmlu_nutrition": {"acc,none": 0.5},
        },
        group_subtasks={"mmlu": ["mmlu_management", "mmlu_nutrition"]},
        n_shot={"mmlu_management": 5},
        n_samples={
            "mmlu_management": {"effective": 1},
            "mmlu_nutrition": {"effective": 2},
        },
    )
    write_samples(
        tmp_path,
        "mmlu_management",
        TIMESTAMP,
        [{"doc_id": 0, "doc_hash": "ab12cd34ef", "filter": "none", "acc": 1.0}],
    )
    write_samples(
        tmp_path,
        "mmlu_nutrition",
        TIMESTAMP,
        [
            {"doc_id": 0, "doc_hash": "ff00", "filter": "none", "acc": 1.0},
            {"doc_id": 1, "doc_hash": "ee11", "filter": "none", "acc": 0.0},
        ],
    )

    rows = build(tmp_path)

    assert rows == [
        {
            "schema_version": 1,
            "scoring_kind": "mean",
            "benchmark": "mmlu/5-shot/acc",
            "subset": "mmlu_management",
            "uid": "ab12cd34ef#0",
            "metric_name": "acc",
            "score": 1.0,
        },
        {
            "schema_version": 1,
            "scoring_kind": "mean",
            "benchmark": "mmlu/5-shot/acc",
            "subset": "mmlu_nutrition",
            "uid": "ff00#0",
            "metric_name": "acc",
            "score": 1.0,
        },
        {
            "schema_version": 1,
            "scoring_kind": "mean",
            "benchmark": "mmlu/5-shot/acc",
            "subset": "mmlu_nutrition",
            "uid": "ee11#0",
            "metric_name": "acc",
            "score": 0.0,
        },
    ]


def test_task_without_subtasks_scores_the_all_subset(tmp_path: Path) -> None:
    write_run(
        tmp_path,
        "gsm8k",
        TIMESTAMP,
        results={
            "gsm8k": {
                "exact_match,flexible-extract": 1.0,
                "exact_match,strict-match": 0.0,
            }
        },
        group_subtasks={"gsm8k": []},
        n_shot={"gsm8k": 8},
    )
    write_samples(
        tmp_path,
        "gsm8k",
        TIMESTAMP,
        [
            {
                "doc_id": 0,
                "doc_hash": "aa",
                "filter": "flexible-extract",
                "exact_match": 1.0,
            },
            {
                "doc_id": 0,
                "doc_hash": "aa",
                "filter": "strict-match",
                "exact_match": 0.0,
            },
        ],
    )

    rows = build(tmp_path)

    assert [(row["benchmark"], row["subset"], row["score"]) for row in rows] == [
        ("gsm8k/8-shot/exact_match.flexible-extract", "ALL", 1.0),
        ("gsm8k/8-shot/exact_match.strict-match", "ALL", 0.0),
    ]


def test_excluded_benchmarks_are_left_out(tmp_path: Path) -> None:
    write_run(
        tmp_path,
        "gsm8k",
        TIMESTAMP,
        results={
            "gsm8k": {
                "exact_match,flexible-extract": 1.0,
                "exact_match,strict-match": 0.0,
            }
        },
        group_subtasks={"gsm8k": []},
        n_shot={"gsm8k": 8},
    )
    write_samples(
        tmp_path,
        "gsm8k",
        TIMESTAMP,
        [
            {
                "doc_id": 0,
                "doc_hash": "aa",
                "filter": "flexible-extract",
                "exact_match": 1.0,
            },
            {
                "doc_id": 0,
                "doc_hash": "aa",
                "filter": "strict-match",
                "exact_match": 0.0,
            },
        ],
    )

    rows = build(tmp_path, exclude=lambda name: name.endswith("strict-match"))

    assert [row["benchmark"] for row in rows] == [
        "gsm8k/8-shot/exact_match.flexible-extract"
    ]


def test_context_length_files_score_their_task(tmp_path: Path) -> None:
    write_run(
        tmp_path,
        "niah_multikey_1",
        TIMESTAMP,
        results={"niah_multikey_1": {"16384,none": 1.0, "32768,none": 0.0}},
        group_subtasks={"niah_multikey_1": []},
        n_shot={"niah_multikey_1": 0},
    )
    write_samples(
        tmp_path,
        "niah_multikey_1_16384",
        TIMESTAMP,
        [{"doc_id": 0, "doc_hash": "aa", "filter": "none", "16384": 1.0}],
    )
    write_samples(
        tmp_path,
        "niah_multikey_1_32768",
        TIMESTAMP,
        [{"doc_id": 0, "doc_hash": "aa", "filter": "none", "32768": 0.0}],
    )

    rows = build(tmp_path)

    assert [(row["benchmark"], row["score"]) for row in rows] == [
        ("niah_multikey_1/0-shot/16384", 1.0),
        ("niah_multikey_1/0-shot/32768", 0.0),
    ]


def test_latest_run_of_a_task_wins(tmp_path: Path) -> None:
    for run_name, timestamp, value, date in (
        ("winogrande", TIMESTAMP, 1.0, 1.0),
        ("winogrande_5shot", LATER_TIMESTAMP, 0.0, 2.0),
    ):
        write_run(
            tmp_path,
            run_name,
            timestamp,
            results={"winogrande": {"acc,none": value}},
            group_subtasks={"winogrande": []},
            n_shot={"winogrande": 5},
            date=date,
        )
        write_samples(
            tmp_path,
            "winogrande",
            timestamp,
            [{"doc_id": 0, "doc_hash": "aa", "filter": "none", "acc": value}],
        )

    rows = build(tmp_path)

    assert [(row["benchmark"], row["score"]) for row in rows] == [
        ("winogrande/5-shot/acc", 0.0)
    ]


def test_group_aggregated_without_sample_weights_is_skipped(tmp_path: Path) -> None:
    write_run(
        tmp_path,
        "slava_32",
        TIMESTAMP,
        results={
            "slava_32": {"mean@1,scoring": 0.5},
            "slava_qa_32": {"mean@1,scoring": 0.75},
            "slava_mcqa_32": {"mean@1,scoring": 0.25},
        },
        group_subtasks={"slava_32": ["slava_qa_32", "slava_mcqa_32"]},
        n_shot={"slava_qa_32": 5},
        n_samples={
            "slava_qa_32": {"effective": 1},
            "slava_mcqa_32": {"effective": 3},
        },
    )
    write_samples(
        tmp_path,
        "slava_qa_32",
        TIMESTAMP,
        [{"doc_id": 0, "doc_hash": "aa", "filter": "scoring", "mean@1": 0.75}],
    )

    assert build(tmp_path) == []


def test_non_numeric_scores_are_skipped(tmp_path: Path) -> None:
    write_run(
        tmp_path,
        "gsm8k",
        TIMESTAMP,
        results={"gsm8k": {"exact_match,none": 1.0}},
        group_subtasks={"gsm8k": []},
        n_shot={"gsm8k": 8},
    )
    write_samples(
        tmp_path,
        "gsm8k",
        TIMESTAMP,
        [
            {
                "doc_id": 0,
                "doc_hash": "aa",
                "filter": "none",
                "exact_match": float("nan"),
            },
            {"doc_id": 1, "doc_hash": "bb", "filter": "none", "exact_match": "1.0"},
            {"doc_id": 2, "doc_hash": "cc", "filter": "none"},
            {"doc_id": 3, "doc_hash": "dd", "filter": "none", "exact_match": True},
        ],
    )

    rows = build(tmp_path)

    assert [(row["uid"], row["score"]) for row in rows] == [("dd#0", 1.0)]


@pytest.mark.parametrize("published", [None, "N/A"])
def test_non_numeric_published_metric_is_not_scored(
    published: object, tmp_path: Path
) -> None:
    write_run(
        tmp_path,
        "gsm8k",
        TIMESTAMP,
        results={"gsm8k": {"exact_match,none": published}},
        group_subtasks={"gsm8k": []},
        n_shot={"gsm8k": 8},
    )
    write_samples(
        tmp_path,
        "gsm8k",
        TIMESTAMP,
        [{"doc_id": 0, "doc_hash": "aa", "filter": "none", "exact_match": 1.0}],
    )

    assert build(tmp_path) == []


def test_uid_survives_renumbering_and_separates_repeated_documents(
    tmp_path: Path,
) -> None:
    write_run(
        tmp_path,
        "gsm8k",
        TIMESTAMP,
        results={
            "gsm8k": {
                "exact_match,none": 0.5,
                "exact_match,flexible-extract": 0.5,
            }
        },
        group_subtasks={"gsm8k": []},
        n_shot={"gsm8k": 8},
    )
    write_samples(
        tmp_path,
        "gsm8k",
        TIMESTAMP,
        [
            {"doc_id": 7, "doc_hash": "aa", "filter": "none", "exact_match": 1.0},
            {
                "doc_id": 7,
                "doc_hash": "aa",
                "filter": "flexible-extract",
                "exact_match": 1.0,
            },
            {"doc_id": 9, "doc_hash": "aa", "filter": "none", "exact_match": 0.0},
            {
                "doc_id": 9,
                "doc_hash": "aa",
                "filter": "flexible-extract",
                "exact_match": 0.0,
            },
        ],
    )

    rows = build(tmp_path)

    assert [(row["benchmark"], row["uid"], row["score"]) for row in rows] == [
        ("gsm8k/8-shot/exact_match", "aa#0", 1.0),
        ("gsm8k/8-shot/exact_match.flexible-extract", "aa#0", 1.0),
        ("gsm8k/8-shot/exact_match", "aa#1", 0.0),
        ("gsm8k/8-shot/exact_match.flexible-extract", "aa#1", 0.0),
    ]


def test_rerun_of_one_subtask_keeps_the_other_subtasks_of_its_benchmark(
    tmp_path: Path,
) -> None:
    write_run(
        tmp_path,
        "mmlu",
        TIMESTAMP,
        results={
            "mmlu": {"acc,none": 1.0},
            "mmlu_management": {"acc,none": 1.0},
        },
        group_subtasks={"mmlu": ["mmlu_management"]},
        n_shot={"mmlu_management": 5},
        n_samples={"mmlu_management": {"effective": 1}},
    )
    write_samples(
        tmp_path,
        "mmlu_management",
        TIMESTAMP,
        [{"doc_id": 0, "doc_hash": "ab12", "filter": "none", "acc": 1.0}],
    )
    remote = [
        remote_row("mmlu_management", "stale#0", 0.0),
        remote_row("mmlu_nutrition", "ff00#0", 1.0),
    ]

    rows = build(tmp_path, remote=remote)

    assert [(row["subset"], row["uid"], row["score"]) for row in rows] == [
        ("mmlu_management", "ab12#0", 1.0),
        ("mmlu_nutrition", "ff00#0", 1.0),
    ]


def test_remeasured_subtask_without_scorables_drops_its_stale_rows(
    tmp_path: Path,
) -> None:
    write_run(
        tmp_path,
        "mmlu",
        TIMESTAMP,
        results={
            "mmlu": {"acc,none": 2 / 3},
            "mmlu_management": {"acc,none": 1.0},
            "mmlu_nutrition": {"acc,none": 0.5},
        },
        group_subtasks={"mmlu": ["mmlu_management", "mmlu_nutrition"]},
        n_shot={"mmlu_management": 5},
        n_samples={
            "mmlu_management": {"effective": 1},
            "mmlu_nutrition": {"effective": 2},
        },
    )
    write_samples(
        tmp_path,
        "mmlu_management",
        TIMESTAMP,
        [{"doc_id": 0, "doc_hash": "ab12", "filter": "none", "acc": None}],
    )
    remote = [
        remote_row("mmlu_management", "stale#0", 0.0),
        remote_row("mmlu_nutrition", "ff00#0", 1.0),
    ]

    assert build(tmp_path, remote=remote) == []


def test_rerun_without_log_samples_drops_the_stale_rows_of_its_benchmark(
    tmp_path: Path,
) -> None:
    write_run(
        tmp_path,
        "mmlu",
        LATER_TIMESTAMP,
        results={"mmlu": {"acc,none": 0.0}, "mmlu_management": {"acc,none": 0.0}},
        group_subtasks={"mmlu": ["mmlu_management"]},
        n_shot={"mmlu_management": 5},
    )
    other = {**remote_row("ALL", "cc#0", 1.0), "benchmark": "gsm8k/8-shot/acc"}
    remote = [remote_row("mmlu_management", "ab12#0", 1.0), other]

    assert build(tmp_path, remote=remote) == [other]


def test_rerun_without_log_samples_outranks_the_samples_of_an_earlier_run(
    tmp_path: Path,
) -> None:
    for timestamp, value, date in ((TIMESTAMP, 1.0, 1.0), (LATER_TIMESTAMP, 0.0, 2.0)):
        write_run(
            tmp_path,
            "mmlu",
            timestamp,
            results={
                "mmlu": {"acc,none": value},
                "mmlu_management": {"acc,none": value},
            },
            group_subtasks={"mmlu": ["mmlu_management"]},
            n_shot={"mmlu_management": 5},
            date=date,
        )
    write_samples(
        tmp_path,
        "mmlu_management",
        TIMESTAMP,
        [{"doc_id": 0, "doc_hash": "ab12", "filter": "none", "acc": 1.0}],
    )
    remote = [remote_row("mmlu_management", "ab12#0", 1.0)]

    assert build(tmp_path, remote=remote) == []


def test_bucketed_ruler_rows_name_their_subtask_once(tmp_path: Path) -> None:
    write_run(
        tmp_path,
        "ruler",
        TIMESTAMP,
        results={
            "ruler": {"4096,none": 0.5},
            "ruler_cwe": {"4096,none": 0.0},
            "ruler_vt": {"4096,none": 1.0},
        },
        group_subtasks={"ruler": ["ruler_cwe", "ruler_vt"]},
        n_shot={"ruler_cwe": 0},
        n_samples={"ruler_cwe": {"effective": 1}, "ruler_vt": {"effective": 1}},
    )
    write_samples(
        tmp_path,
        "ruler_cwe",
        TIMESTAMP,
        [{"doc_id": 0, "doc_hash": "bb", "filter": "none", "4096": 0.0}],
    )
    write_samples(
        tmp_path,
        "ruler_vt",
        TIMESTAMP,
        [{"doc_id": 0, "doc_hash": "aa", "filter": "none", "4096": 1.0}],
    )

    rows = build(tmp_path)

    assert [
        (row["benchmark"], row["subset"], row["uid"], row["score"]) for row in rows
    ] == [
        ("ruler/0-shot/4096", "ruler_cwe", "bb#0", 0.0),
        ("ruler/0-shot/4096", "ruler_vt", "aa#0", 1.0),
    ]


def test_excluded_ruler_subtask_drops_the_whole_bucket(tmp_path: Path) -> None:
    write_run(
        tmp_path,
        "ruler",
        TIMESTAMP,
        results={
            "ruler": {"4096,none": 0.5},
            "ruler_cwe": {"4096,none": 0.0},
            "ruler_vt": {"4096,none": 1.0},
        },
        group_subtasks={"ruler": ["ruler_cwe", "ruler_vt"]},
        n_shot={"ruler_cwe": 0},
        n_samples={"ruler_cwe": {"effective": 1}, "ruler_vt": {"effective": 1}},
    )
    write_samples(
        tmp_path,
        "ruler_cwe",
        TIMESTAMP,
        [{"doc_id": 0, "doc_hash": "bb", "filter": "none", "4096": 0.0}],
    )
    write_samples(
        tmp_path,
        "ruler_vt",
        TIMESTAMP,
        [{"doc_id": 0, "doc_hash": "aa", "filter": "none", "4096": 1.0}],
    )

    assert build(tmp_path, exclude=lambda name: name == "ruler/vt/0-shot/4096") == []


def test_ruler_bucket_of_unequally_sized_subtasks_is_skipped(tmp_path: Path) -> None:
    write_run(
        tmp_path,
        "ruler",
        TIMESTAMP,
        results={
            "ruler": {"4096,none": 0.25},
            "ruler_cwe": {"4096,none": 0.0},
            "ruler_vt": {"4096,none": 1.0},
        },
        group_subtasks={"ruler": ["ruler_cwe", "ruler_vt"]},
        n_shot={"ruler_cwe": 0},
        n_samples={"ruler_cwe": {"effective": 3}, "ruler_vt": {"effective": 1}},
    )
    write_samples(
        tmp_path,
        "ruler_cwe",
        TIMESTAMP,
        [
            {"doc_id": 0, "doc_hash": "bb", "filter": "none", "4096": 0.0},
            {"doc_id": 1, "doc_hash": "cc", "filter": "none", "4096": 0.0},
            {"doc_id": 2, "doc_hash": "dd", "filter": "none", "4096": 0.0},
        ],
    )
    write_samples(
        tmp_path,
        "ruler_vt",
        TIMESTAMP,
        [{"doc_id": 0, "doc_hash": "aa", "filter": "none", "4096": 1.0}],
    )

    assert build(tmp_path) == []


def test_samples_file_without_a_results_file_is_reported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    errors = []
    monkeypatch.setattr(
        "node_runner.src.job_dispatcher.cache_scores.lm_eval.Logger.error",
        errors.append,
    )
    write_run(
        tmp_path,
        "winogrande",
        TIMESTAMP,
        results={"winogrande": {"acc,none": 1.0}},
        group_subtasks={"winogrande": []},
        n_shot={"winogrande": 5},
    )
    write_samples(
        tmp_path,
        "gsm8k",
        TIMESTAMP,
        [{"doc_id": 0, "doc_hash": "aa", "filter": "none", "exact_match": 1.0}],
    )

    assert build(tmp_path) == []
    assert errors == [f"samples_gsm8k_{TIMESTAMP}.jsonl: no results file describes it"]


def test_task_whose_metrics_are_all_excluded_is_not_reported_as_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    errors = []
    monkeypatch.setattr(
        "node_runner.src.job_dispatcher.cache_scores.lm_eval.Logger.error",
        errors.append,
    )
    write_run(
        tmp_path,
        "gsm8k",
        TIMESTAMP,
        results={"gsm8k": {"exact_match,strict-match": 0.0}},
        group_subtasks={"gsm8k": []},
        n_shot={"gsm8k": 8},
    )
    write_samples(
        tmp_path,
        "gsm8k",
        TIMESTAMP,
        [
            {
                "doc_id": 0,
                "doc_hash": "aa",
                "filter": "strict-match",
                "exact_match": 0.0,
            }
        ],
    )

    assert build(tmp_path, exclude=lambda name: name.endswith("strict-match")) == []
    assert errors == []


def test_remote_rows_of_another_schema_version_are_dropped(tmp_path: Path) -> None:
    write_run(
        tmp_path,
        "gsm8k",
        TIMESTAMP,
        results={"gsm8k": {"exact_match,none": 1.0}},
        group_subtasks={"gsm8k": []},
        n_shot={"gsm8k": 8},
    )
    write_samples(
        tmp_path,
        "gsm8k",
        TIMESTAMP,
        [{"doc_id": 0, "doc_hash": "aa", "filter": "none", "exact_match": 1.0}],
    )
    remote = [
        remote_row("mmlu_management", "ff00#0", 1.0, schema_version=0),
        remote_row("mmlu_nutrition", "ee11#0", 1.0),
    ]

    rows = build(tmp_path, remote=remote)

    assert [(row["benchmark"], row["subset"]) for row in rows] == [
        ("gsm8k/8-shot/exact_match", "ALL"),
        ("mmlu/5-shot/acc", "mmlu_nutrition"),
    ]


def test_samples_file_without_a_run_timestamp_is_skipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    warnings = []
    monkeypatch.setattr(
        "node_runner.src.job_dispatcher.cache_scores.lm_eval.Logger.warning",
        warnings.append,
    )
    (tmp_path / "samples_gsm8k.jsonl").write_text(
        json.dumps(
            {"doc_id": 0, "doc_hash": "aa", "filter": "none", "exact_match": 1.0}
        )
        + "\n",
        encoding="utf-8",
    )

    assert build(tmp_path) == []
    assert warnings == ["No run timestamp in samples_gsm8k.jsonl; skipping"]


def test_samples_without_doc_hash_are_reported_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    errors = []
    monkeypatch.setattr(
        "node_runner.src.job_dispatcher.cache_scores.lm_eval.Logger.error",
        errors.append,
    )
    write_run(
        tmp_path,
        "gsm8k",
        TIMESTAMP,
        results={"gsm8k": {"exact_match,none": 0.0}},
        group_subtasks={"gsm8k": []},
        n_shot={"gsm8k": 8},
    )
    write_samples(
        tmp_path,
        "gsm8k",
        TIMESTAMP,
        [
            {"doc_id": 0, "filter": "none", "exact_match": 1.0},
            {"doc_id": 1, "doc_hash": "", "filter": "none", "exact_match": 1.0},
            {"doc_id": 2, "doc_hash": 17, "filter": "none", "exact_match": 1.0},
            {"doc_id": 3, "doc_hash": "aa", "filter": "none", "exact_match": 0.0},
        ],
    )

    rows = build(tmp_path)

    assert [(row["uid"], row["score"]) for row in rows] == [("aa#0", 0.0)]
    assert errors == [
        f"samples_gsm8k_{TIMESTAMP}.jsonl: "
        "samples without doc_hash; no scorables for it"
    ]


def test_unreadable_results_file_is_reported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    errors = []
    monkeypatch.setattr(
        "node_runner.src.job_dispatcher.cache_scores.lm_eval.Logger.error",
        errors.append,
    )
    results_file = tmp_path / f"gsm8k_{TIMESTAMP}.json"
    results_file.write_text("{not json", encoding="utf-8")
    write_samples(
        tmp_path,
        "gsm8k",
        TIMESTAMP,
        [{"doc_id": 0, "doc_hash": "aa", "filter": "none", "exact_match": 1.0}],
    )

    assert build(tmp_path) == []
    assert errors[0].startswith(f"Unreadable lm_eval results {results_file}: ")
    assert errors[1:] == [
        f"samples_gsm8k_{TIMESTAMP}.jsonl: no results file describes it"
    ]


@pytest.mark.parametrize(
    "file_name, run",
    [
        (
            "gsm8k.json",
            {
                "results": {"gsm8k": {"exact_match,none": 0.5}},
                "group_subtasks": {"gsm8k": []},
            },
        ),
        (
            f"gsm8k_{TIMESTAMP}.json",
            {"results": {}, "group_subtasks": {"gsm8k": []}},
        ),
        (
            f"gsm8k_{TIMESTAMP}.json",
            {"results": {"gsm8k": {"exact_match,none": 0.5}}, "group_subtasks": {}},
        ),
    ],
    ids=["no run timestamp", "no results", "no group_subtasks"],
)
def test_json_files_that_describe_no_run_are_ignored(
    file_name: str, run: dict, tmp_path: Path
) -> None:
    (tmp_path / file_name).write_text(json.dumps(run), encoding="utf-8")
    write_samples(
        tmp_path,
        "gsm8k",
        TIMESTAMP,
        [{"doc_id": 0, "doc_hash": "aa", "filter": "none", "exact_match": 1.0}],
    )

    assert build(tmp_path) == []


def test_ruler_task_without_subtasks_scores_its_bucket(tmp_path: Path) -> None:
    write_run(
        tmp_path,
        "ruler",
        TIMESTAMP,
        results={"ruler": {"4096,none": 1.0}},
        group_subtasks={"ruler": []},
        n_shot={"ruler": 0},
    )
    write_samples(
        tmp_path,
        "ruler",
        TIMESTAMP,
        [{"doc_id": 0, "doc_hash": "aa", "filter": "none", "4096": 1.0}],
    )

    rows = build(tmp_path)

    assert [(row["benchmark"], row["subset"], row["score"]) for row in rows] == [
        ("ruler/0-shot/4096", "ruler", 1.0)
    ]


def test_ruler_bucket_absent_from_the_group_aggregate_is_still_scored(
    tmp_path: Path,
) -> None:
    write_run(
        tmp_path,
        "ruler",
        TIMESTAMP,
        results={
            "ruler": {"4096,none": 0.5},
            "ruler_cwe": {"4096,none": 0.0, "8192,none": 0.0},
            "ruler_vt": {"4096,none": 1.0, "8192,none": 1.0},
        },
        group_subtasks={"ruler": ["ruler_cwe", "ruler_vt"]},
        n_shot={"ruler_cwe": 0},
        n_samples={"ruler_cwe": {"effective": 1}, "ruler_vt": {"effective": 1}},
    )
    write_samples(
        tmp_path,
        "ruler_cwe",
        TIMESTAMP,
        [{"doc_id": 0, "doc_hash": "bb", "filter": "none", "4096": 0.0, "8192": 0.0}],
    )
    write_samples(
        tmp_path,
        "ruler_vt",
        TIMESTAMP,
        [{"doc_id": 0, "doc_hash": "aa", "filter": "none", "4096": 1.0, "8192": 1.0}],
    )

    rows = build(tmp_path)

    assert [(row["benchmark"], row["subset"], row["score"]) for row in rows] == [
        ("ruler/0-shot/4096", "ruler_cwe", 0.0),
        ("ruler/0-shot/8192", "ruler_cwe", 0.0),
        ("ruler/0-shot/4096", "ruler_vt", 1.0),
        ("ruler/0-shot/8192", "ruler_vt", 1.0),
    ]


def test_ruler_metric_that_is_not_a_context_length_is_skipped(tmp_path: Path) -> None:
    write_run(
        tmp_path,
        "ruler",
        TIMESTAMP,
        results={
            "ruler": {"acc,none": 0.5},
            "ruler_cwe": {"acc,none": 0.0},
            "ruler_vt": {"acc,none": 1.0},
        },
        group_subtasks={"ruler": ["ruler_cwe", "ruler_vt"]},
        n_shot={"ruler_cwe": 0},
        n_samples={"ruler_cwe": {"effective": 1}, "ruler_vt": {"effective": 1}},
    )
    write_samples(
        tmp_path,
        "ruler_cwe",
        TIMESTAMP,
        [{"doc_id": 0, "doc_hash": "bb", "filter": "none", "acc": 0.0}],
    )

    assert build(tmp_path) == []


def test_ruler_bucket_scores_the_subgroup_that_survives_exclusion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    warnings = []
    monkeypatch.setattr(
        "node_runner.src.job_dispatcher.cache_scores.lm_eval.Logger.warning",
        warnings.append,
    )
    write_run(
        tmp_path,
        "ruler",
        TIMESTAMP,
        results={
            "ruler": {"4096,none": 0.5},
            "ruler_niah": {"4096,none": 0.5},
            "ruler_niah_1": {"4096,none": 0.0},
            "ruler_niah_2": {"4096,none": 1.0},
        },
        group_subtasks={
            "ruler": ["ruler_niah"],
            "ruler_niah": ["ruler_niah_1", "ruler_niah_2"],
        },
        n_shot={"ruler_niah_1": 0},
        n_samples={"ruler_niah_1": {"effective": 1}, "ruler_niah_2": {"effective": 1}},
    )
    write_samples(
        tmp_path,
        "ruler_niah_1",
        TIMESTAMP,
        [{"doc_id": 0, "doc_hash": "bb", "filter": "none", "4096": 0.0}],
    )
    write_samples(
        tmp_path,
        "ruler_niah_2",
        TIMESTAMP,
        [{"doc_id": 0, "doc_hash": "cc", "filter": "none", "4096": 1.0}],
    )

    rows = build(tmp_path, exclude=lambda name: name != "ruler/niah/0-shot/4096")

    assert [(row["benchmark"], row["subset"], row["score"]) for row in rows] == [
        ("ruler/0-shot/4096", "ruler_niah_1", 0.0),
        ("ruler/0-shot/4096", "ruler_niah_2", 1.0),
    ]
    assert warnings == []


@pytest.mark.parametrize(
    "results, n_samples",
    [
        (
            {
                "mmlu": {"acc,none": 0.75},
                "mmlu_management": {"acc,none": 1.0},
                "mmlu_nutrition": {},
            },
            {"mmlu_management": {"effective": 1}, "mmlu_nutrition": {"effective": 2}},
        ),
        (
            {
                "mmlu": {"acc,none": 0.75},
                "mmlu_management": {"acc,none": 1.0},
                "mmlu_nutrition": {"acc,none": 0.5},
            },
            {"mmlu_management": {"effective": 1}},
        ),
    ],
    ids=["subtask without the metric", "subtask without a sample count"],
)
def test_group_with_incomplete_subtask_results_is_skipped(
    results: dict, n_samples: dict, tmp_path: Path
) -> None:
    write_run(
        tmp_path,
        "mmlu",
        TIMESTAMP,
        results=results,
        group_subtasks={"mmlu": ["mmlu_management", "mmlu_nutrition"]},
        n_shot={"mmlu_management": 5},
        n_samples=n_samples,
    )
    write_samples(
        tmp_path,
        "mmlu_management",
        TIMESTAMP,
        [{"doc_id": 0, "doc_hash": "aa", "filter": "none", "acc": 1.0}],
    )

    assert build(tmp_path) == []


@pytest.mark.parametrize(
    ("measured_run", "expected_benchmarks"),
    [
        pytest.param(
            None,
            ["hellaswag/5-shot/acc", "winogrande/5-shot/acc"],
            id="whole-metrics-dir",
        ),
        pytest.param(
            ("hellaswag", LATER_TIMESTAMP),
            ["hellaswag/5-shot/acc"],
            id="this-measurement-only",
        ),
    ],
)
def test_only_the_runs_of_this_measurement_reach_the_cache(
    tmp_path: Path,
    measured_run: tuple[str, str] | None,
    expected_benchmarks: list[str],
) -> None:
    for task, timestamp in (("winogrande", TIMESTAMP), ("hellaswag", LATER_TIMESTAMP)):
        write_run(
            tmp_path,
            task,
            timestamp,
            results={task: {"acc,none": 1.0}},
            group_subtasks={task: []},
            n_shot={task: 5},
            date=1.0,
        )
        write_samples(
            tmp_path,
            task,
            timestamp,
            [{"doc_id": 0, "doc_hash": "aa", "filter": "none", "acc": 1.0}],
        )

    run_files = None
    if measured_run is not None:
        task, timestamp = measured_run
        run_files = {
            tmp_path / f"{task}_{timestamp}.json",
            tmp_path / f"samples_{task}_{timestamp}.jsonl",
        }

    rows = build(tmp_path, run_files=run_files)

    assert sorted({row["benchmark"] for row in rows}) == expected_benchmarks
