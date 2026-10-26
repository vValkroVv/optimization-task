import json

import pytest

from node_runner.src.job_dispatcher.cache_scores.base import iter_compressed_jsonl_bytes
from node_runner.src.job_dispatcher.cache_scores.scalar import ScalarCacheScoresBuilder

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("metrics", "samples"),
    [
        ({"quality": "failed"}, [("a", 0), ("b", 1)]),
        ({"other": 0.5}, [("a", 0), ("b", 1)]),
        ({"quality": 0.5, "quality/child": 1}, [("a", 0), ("b", 1)]),
        ({"quality": 0.5, "quality (child)": 1}, [("a", 0), ("b", 1)]),
        ({"quality": 0.5}, [("a", 0)]),
        ({"quality": 0.5}, [("a", 0), ("a", 1)]),
        ({"quality": 0.5}, [("", 0), ("b", 1)]),
        ({"quality": 0.5}, [("a", float("nan")), ("b", 1)]),
        ({"quality": 0.5}, [("a", 1), ("b", 1)]),
    ],
)
def test_invalid_or_ambiguous_evidence_is_withheld(tmp_path, metrics, samples):
    scores = ScalarCacheScoresBuilder(tmp_path, metrics)
    scores.add_metric("quality", samples, {"trials": 1})
    assert scores.records == {}


def test_scalar_roundtrip_replaces_covered_metrics_and_retains_compatible_rows(
    tmp_path,
):
    scores = ScalarCacheScoresBuilder(tmp_path, {"quality": 0.5, "errors": 3})
    scores.add_metric("quality", [("b", 1), ("a", 0)], {"trials": 1})
    row = scores.records["quality"][0]
    retained = {**row, "benchmark": "other"}
    scores.existing_rows_loader = lambda: iter(
        [
            row,
            {**row, "benchmark": "errors"},
            retained,
            {**row, "benchmark": "legacy", "schema_version": 2},
            {**row, "benchmark": "nonlinear", "scoring_kind": "f1"},
        ]
    )
    artifact = scores.build(tmp_path)
    rows = list(iter_compressed_jsonl_bytes(artifact.read_bytes()))
    assert rows == scores.records["quality"] + [retained]
    assert [r["score"] for r in rows[:2]] == [0, 1]
    assert {
        k: row[k] for k in ("schema_version", "scoring_kind", "subset", "metric_name")
    } == {
        "schema_version": 1,
        "scoring_kind": "mean",
        "subset": "ALL",
        "metric_name": "quality",
    }
    assert [
        json.loads(line) for line in scores.source_path.read_text().splitlines()
    ] == rows[:2]


@pytest.mark.parametrize(
    ("samples", "identity", "pairs"),
    [
        ([("b", 1), ("a", 0)], {"trials": 1}, True),
        ([("a", 0), ("b", 1)], {"trials": 2}, False),
        ([("a", 0), ("c", 1)], {"trials": 1}, False),
    ],
)
def test_pairing_requires_identical_methodology_and_population(
    tmp_path, samples, identity, pairs
):
    scores = ScalarCacheScoresBuilder(tmp_path, {"quality": 0.5})
    scores.add_metric("quality", [("a", 0), ("b", 1)], {"trials": 1})
    original = {row["uid"] for row in scores.records["quality"]}
    scores.add_metric("quality", samples, identity)
    current = {row["uid"] for row in scores.records["quality"]}
    assert (original == current) if pairs else original.isdisjoint(current)


@pytest.mark.parametrize(
    ("categories", "weights", "score"),
    [
        ({"a": [("1", 0), ("2", 1)]}, {"a": 1}, "failed"),
        ({"a": [("1", 0), ("2", 1)]}, {}, 0.5),
        ({"a": [("1", 0), ("2", 1)]}, {"b": 1}, 0.5),
        ({"a": [("1", 1)]}, {"a": 1}, 1),
        ({"a": [("1", 0), ("1", 1)]}, {"a": 1}, 0.5),
        ({"a": [("", 0), ("2", 1)]}, {"a": 1}, 0.5),
        ({"a": [("1", float("nan")), ("2", 1)]}, {"a": 1}, 0.5),
        ({"a": [("1", 0), ("2", 1)]}, {"a": 1}, 0.6),
    ],
)
def test_weighted_metric_withholds_invalid_evidence(
    tmp_path, categories, weights, score
):
    scores = ScalarCacheScoresBuilder(tmp_path, {"quality": score})
    scores.add_weighted_metric("quality", categories, weights, {"method": "test"})
    assert scores.records == {}


@pytest.mark.parametrize(
    ("metrics", "samples"),
    [
        ({"quality": "failed"}, [("a", 0.0, 1), ("b", 1.0, 1)]),
        ({"other": 0.5}, [("a", 0.0, 1), ("b", 1.0, 1)]),
        ({"quality": 0.5}, [("a", 1.0, 1)]),
        ({"quality": 0.5}, [("a", 0.0, 1), ("a", 1.0, 1)]),
        ({"quality": 0.5}, [("a", 0.0, 0), ("b", 1.0, 1)]),
        ({"quality": 0.5}, [("a", float("nan"), 1), ("b", 1.0, 1)]),
        ({"quality": 0.6}, [("a", 0.0, 1), ("b", 1.0, 1)]),
    ],
)
def test_ratio_metric_withholds_invalid_evidence(tmp_path, metrics, samples):
    scores = ScalarCacheScoresBuilder(tmp_path, metrics)
    scores.add_ratio_metric("quality", samples, {"method": "test"})
    assert scores.records == {}
