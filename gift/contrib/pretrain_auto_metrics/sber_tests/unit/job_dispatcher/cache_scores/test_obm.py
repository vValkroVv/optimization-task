import gzip
import hashlib
import json
from pathlib import Path

import pytest

from node_runner.src.job_dispatcher.cache_scores import iter_compressed_jsonl_bytes
from node_runner.src.job_dispatcher.cache_scores.base import (
    CacheScoresBuilder,
    CacheScoresSpec,
    encoded_path,
    iter_jsonl,
    project_row,
)
from node_runner.src.job_dispatcher.cache_scores.obm import ObmCacheScoresBuilder

pytestmark = pytest.mark.unit


def _write_population(directory: Path, rows: list[dict]) -> None:
    path = directory / "all_metrics.jsonl"
    entries = list(iter_jsonl(path))
    population = {
        "schema_version": 1,
        "rows": {
            str(row["uid"]): hashlib.sha256(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode()
            ).hexdigest()
            for row in rows
        },
    }
    for entry in entries:
        entry.update(size=len(population["rows"]), cache_population=population)
    path.write_text("".join(json.dumps(entry) + "\n" for entry in entries))


def _full_rows(directory: Path, benchmark: str) -> list[dict]:
    output = ObmCacheScoresBuilder(directory).build(directory)
    return [
        row
        for row in iter_compressed_jsonl_bytes(output.read_bytes())
        if row["benchmark"] == benchmark
    ]


def test_project_row_keeps_only_requested_fields() -> None:
    row = {"uid": "1", "score": 0.5, "prompt": "large"}

    assert project_row(row, ["uid", "score", "missing"], {"benchmark": "a"}) == {
        "benchmark": "a",
        "uid": "1",
        "score": 0.5,
    }


def test_iter_jsonl_skips_blank_and_invalid_lines(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"uid": "1"}\n\nnot-json\n', encoding="utf-8")
    warnings = []
    monkeypatch.setattr(
        "node_runner.src.job_dispatcher.cache_scores.base.Logger.warning",
        warnings.append,
    )

    assert list(iter_jsonl(path)) == [{"uid": "1"}]
    assert warnings == [f"Skipping unparsable line in {path}"]


def test_obm_builder_projects_local_rows_and_merges_remote_benchmarks(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "alpha.jsonl").write_text(
        json.dumps({"uid": "local", "score": 0.8, "prompt": "drop"}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "all_metrics.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"name": "ignored", "metric": {}}),
                json.dumps({"name": "alpha", "metric": {"name": "accuracy"}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    remote_rows = [
        {"benchmark": "alpha", "uid": "stale", "score": 0.1},
        {"benchmark": "beta", "uid": "remote", "score": 0.6},
    ]
    builder = ObmCacheScoresBuilder(tmp_path)
    builder.existing_rows_loader = lambda: remote_rows

    output_path = builder.build(tmp_path)

    assert output_path is not None
    rows = list(iter_compressed_jsonl_bytes(output_path.read_bytes()))
    assert rows == [
        {
            "benchmark": "alpha",
            "metric_name": "accuracy",
            "uid": "local",
            "score": 0.8,
        },
        {"benchmark": "beta", "uid": "remote", "score": 0.6},
    ]


@pytest.mark.parametrize(
    ("metric", "rows", "expected"),
    [
        (
            "accuracy/pass@2",
            [
                {"uid": "a", "rollouts": [{"matched": False}, {"matched": True}]},
                {"uid": "b", "rollouts": [{"matched": False}, {"matched": False}]},
            ],
            [1.0, 0.0],
        ),
        (
            "accuracy/mean@2",
            [
                {"uid": "a", "rollouts": [{"matched": True}, {"matched": False}]},
                {"uid": "b", "rollouts": [{"matched": False}, {"matched": False}]},
            ],
            [0.5, 0.0],
        ),
        (
            "accuracy",
            [
                {"uid": "a", "rollouts": [{"matched": True}, {"matched": False}]},
                {"uid": "b", "rollouts": [{"matched": False}, {"matched": False}]},
            ],
            [0.5, 0.0],
        ),
        (
            "keyword_score",
            [{"uid": "a", "score": 0.8}, {"uid": "b", "score": 0.2}],
            [0.8, 0.2],
        ),
        (
            "accuracy",
            [
                {"uid": "a", "matched": {"name": True}},
                {"uid": "b", "matched": {"name": False}},
            ],
            [1.0, 0.0],
        ),
    ],
)
def test_obm_builder_exports_full_metric_keys(
    tmp_path: Path, metric: str, rows: list[dict], expected: list[float]
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "alpha.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (tmp_path / "all_metrics.jsonl").write_text(
        json.dumps(
            {
                "name": "alpha",
                "metric": {"name": metric, "value": sum(expected) / len(expected)},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    _write_population(tmp_path, list(iter_jsonl(cache_dir / "alpha.jsonl")))
    output_path = ObmCacheScoresBuilder(tmp_path).build(tmp_path)

    assert output_path is not None
    exported = list(iter_compressed_jsonl_bytes(output_path.read_bytes()))
    full_rows = [row for row in exported if row["benchmark"] == f"alpha/{metric}"]
    assert all(len(row["uid"].split(":")[0]) == 64 for row in full_rows)
    assert [{**row, "uid": row["uid"].split(":", 1)[1]} for row in full_rows] == [
        {
            "benchmark": f"alpha/{metric}",
            "uid": uid,
            "metric_name": "mean",
            "score": score,
        }
        for uid, score in zip(("a", "b"), expected)
    ]


def test_obm_builder_exports_both_aime_rollout_metrics(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "aime.jsonl").write_text(
        '{"uid": "a", "rollouts": [{"matched": true}, {"matched": false}]}\n'
        '{"uid": "b", "rollouts": [{"matched": false}, {"matched": true}]}\n',
        encoding="utf-8",
    )
    (tmp_path / "all_metrics.jsonl").write_text(
        json.dumps(
            {
                "name": "aime",
                "metrics": [
                    {"name": "accuracy/pass@2", "value": 1.0},
                    {"name": "accuracy/mean@2", "value": 0.5},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    _write_population(tmp_path, list(iter_jsonl(cache_dir / "aime.jsonl")))
    output_path = ObmCacheScoresBuilder(tmp_path).build(tmp_path)

    assert output_path is not None
    exported = list(iter_compressed_jsonl_bytes(output_path.read_bytes()))
    assert [row["benchmark"] for row in exported] == [
        "aime",
        "aime",
        "aime/accuracy/pass@2",
        "aime/accuracy/pass@2",
        "aime/accuracy/mean@2",
        "aime/accuracy/mean@2",
    ]


def test_obm_builder_reads_encrypted_obm_without_naming_it_in_logs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "aime.jsonl").write_text(
        '{"uid": "a", "score": 1.0}\n{"uid": "b", "score": 0.0}\n', encoding="utf-8"
    )
    (tmp_path / "all_metrics.jsonl").write_text(
        json.dumps({"name": "aime", "metric": {"name": "accuracy", "value": 0.5}})
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "plain").mkdir()
    (tmp_path / "encrypted").mkdir()
    plain = ObmCacheScoresBuilder(tmp_path).build(tmp_path / "plain")
    for path in [cache_dir / "aime.jsonl", tmp_path / "all_metrics.jsonl"]:
        with gzip.open(encoded_path(path), "wt", encoding="utf-8") as f:
            f.write(path.read_text(encoding="utf-8"))
        path.unlink()
    logs = []
    for level in ("info", "warning", "error"):
        monkeypatch.setattr(
            f"node_runner.src.job_dispatcher.cache_scores.base.Logger.{level}",
            logs.append,
        )

    encrypted = ObmCacheScoresBuilder(tmp_path).build(tmp_path / "encrypted")

    assert encrypted is not None
    assert list(iter_compressed_jsonl_bytes(encrypted.read_bytes())) == list(
        iter_compressed_jsonl_bytes(plain.read_bytes())
    )
    assert logs and not any("aime" in message for message in logs)


def test_obm_builder_drops_unverified_full_key_rows_but_keeps_legacy(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "alpha.jsonl").write_text(
        '{"uid": "a", "score": 0.25}\n', encoding="utf-8"
    )
    (tmp_path / "all_metrics.jsonl").write_text(
        '{"name": "alpha", "metric": {"name": "keyword_score", "value": 0.75}}\n',
        encoding="utf-8",
    )

    output_path = ObmCacheScoresBuilder(tmp_path).build(tmp_path)

    assert output_path is not None
    assert list(iter_compressed_jsonl_bytes(output_path.read_bytes())) == [
        {
            "benchmark": "alpha",
            "metric_name": "keyword_score",
            "uid": "a",
            "score": 0.25,
        }
    ]


@pytest.mark.parametrize(
    ("metric", "rows"),
    [
        (
            {"name": "accuracy/pass@2", "value": 1.0},
            [{"uid": "a", "rollouts": [{"matched": True}]}],
        ),
        (
            {"name": "keyword_score", "value": 1.0},
            [{"uid": "a", "score": float("nan")}],
        ),
        (
            {"name": "keyword_score", "value": float("nan")},
            [{"uid": "a", "score": 1.0}],
        ),
        (
            {"name": "accuracy", "value": 1.0},
            [{"uid": "a", "matched": True}],
        ),
        (
            {
                "name": "accuracy",
                "value": 0.5,
                "additional": {"accuracy_ci_95": 0.1},
            },
            [{"uid": "a", "rollouts": [{"matched": True}, {"matched": False}]}],
        ),
    ],
)
def test_obm_builder_does_not_publish_invalid_task_mean(
    tmp_path: Path, metric: dict, rows: list[dict]
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "alpha.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (tmp_path / "all_metrics.jsonl").write_text(
        json.dumps({"name": "alpha", "metric": metric}) + "\n", encoding="utf-8"
    )

    _write_population(tmp_path, list(iter_jsonl(cache_dir / "alpha.jsonl")))
    output_path = ObmCacheScoresBuilder(tmp_path).build(tmp_path)

    assert output_path is not None
    assert all(
        row["benchmark"] != f"alpha/{metric['name']}"
        for row in iter_compressed_jsonl_bytes(output_path.read_bytes())
    )


def test_obm_builder_uses_exact_scored_row_despite_stale_duplicates(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    selected = [
        {"uid": "a", "subset": "first", "score": 1.0, "cache_fingerprint": "current"},
        {"uid": "b", "subset": "second", "score": 0.0, "cache_fingerprint": "current"},
    ]
    stale = {"uid": "a", "subset": "first", "score": 0.0, "cache_fingerprint": "old"}
    (cache / "alpha.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in [stale, *selected, stale])
    )
    (tmp_path / "all_metrics.jsonl").write_text(
        json.dumps({"name": "alpha", "metric": {"name": "keyword_score", "value": 0.5}})
    )
    _write_population(tmp_path, selected)
    rows = _full_rows(tmp_path, "alpha/keyword_score")
    assert [
        (row["uid"].split(":")[-1], row["subset"], row["score"]) for row in rows
    ] == [("a", "first", 1.0), ("b", "second", 0.0)]


def test_obm_builder_keeps_other_benchmarks_if_variant_cache_is_missing(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "beta.jsonl").write_text('{"uid": "b"}\n', encoding="utf-8")
    (tmp_path / "all_metrics.jsonl").write_text(
        '{"metric": {"name": "ignored", "value": 1.0}}\n'
        '{"name": "alpha", "metric": {"name": "keyword_score", "value": 1.0}}\n',
        encoding="utf-8",
    )

    output_path = ObmCacheScoresBuilder(tmp_path).build(tmp_path)

    assert output_path is not None
    assert list(iter_compressed_jsonl_bytes(output_path.read_bytes())) == [
        {"benchmark": "beta", "uid": "b"}
    ]


@pytest.mark.parametrize(
    ("contents", "metric"),
    [
        ("", "keyword_score"),
        ('{"uid": "a", "score": 1.0}\n{"uid": "b"}\n', "keyword_score"),
        (
            '{"uid": "a", "matched": {"name": true}}\n{"uid": "b", "matched": false}\n',
            "accuracy",
        ),
    ],
)
def test_obm_builder_with_incomplete_cache_keeps_legacy_only(
    tmp_path: Path, contents: str, metric: str
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "alpha.jsonl").write_text(contents, encoding="utf-8")
    (tmp_path / "all_metrics.jsonl").write_text(
        json.dumps({"name": "alpha", "metric": {"name": metric, "value": 1.0}}) + "\n",
        encoding="utf-8",
    )

    _write_population(tmp_path, list(iter_jsonl(cache_dir / "alpha.jsonl")))
    output_path = ObmCacheScoresBuilder(tmp_path).build(tmp_path)

    assert output_path is not None
    assert all(
        row["benchmark"] == "alpha"
        for row in iter_compressed_jsonl_bytes(output_path.read_bytes())
    )


def test_obm_builder_returns_none_without_cache_files(tmp_path: Path) -> None:
    builder = ObmCacheScoresBuilder(tmp_path)

    assert list(builder.existing_rows()) == []
    assert builder.build(tmp_path) is None


def test_obm_builder_drops_the_previous_file_when_remote_merge_fails(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "alpha.jsonl").write_text('{"uid": "local"}\n', encoding="utf-8")
    output_path = tmp_path / "cache_scores.jsonl.zst"
    output_path.write_bytes(b"previous")

    def fail_to_load_remote_rows():
        raise RuntimeError("S3 unavailable")

    builder = ObmCacheScoresBuilder(tmp_path)
    builder.existing_rows_loader = fail_to_load_remote_rows

    assert builder.build(tmp_path) is None
    assert not output_path.exists()
    assert list(tmp_path.glob("*.part")) == []


def test_builder_without_merge_key_writes_rows(tmp_path: Path) -> None:
    source_path = tmp_path / "source.jsonl"
    source_path.write_text('{"uid": "1", "drop": true}\n', encoding="utf-8")

    class BuilderWithoutMerge(CacheScoresBuilder):
        def iter_source_files(self):
            return [source_path]

        def derived_fields(self, source_file: Path) -> dict:
            return {"source": source_file.stem}

    builder = BuilderWithoutMerge(tmp_path, CacheScoresSpec(raw_fields=["uid"]))

    assert list(builder.existing_rows()) == []
    output_path = builder.build(tmp_path)
    assert output_path is not None
    assert list(iter_compressed_jsonl_bytes(output_path.read_bytes())) == [
        {"source": "source", "uid": "1"}
    ]


def test_builder_handles_missing_local_merge_key(tmp_path: Path) -> None:
    source_path = tmp_path / "source.jsonl"
    source_path.write_text('{"uid": "local"}\n', encoding="utf-8")

    class BuilderWithoutLocalKey(CacheScoresBuilder):
        def iter_source_files(self):
            return [source_path]

        def derived_fields(self, source_file: Path) -> dict:
            return {}

        def existing_rows(self):
            return [{"benchmark": "remote", "uid": "remote"}]

    builder = BuilderWithoutLocalKey(
        tmp_path, CacheScoresSpec(raw_fields=["uid"], merge_keys=("benchmark",))
    )

    output_path = builder.build(tmp_path)

    assert output_path is not None
    assert list(iter_compressed_jsonl_bytes(output_path.read_bytes())) == [
        {"uid": "local"},
        {"benchmark": "remote", "uid": "remote"},
    ]


@pytest.mark.parametrize(
    "case",
    ["old_summary", "missing", "extra", "replaced", "wrong_size", "unknown_version"],
)
def test_full_key_ci_requires_exact_current_population(tmp_path, case):
    cache = tmp_path / "cache"
    cache.mkdir()
    selected = [{"uid": "a", "score": 0.0}, {"uid": "b", "score": 1.0}]
    (tmp_path / "all_metrics.jsonl").write_text(
        json.dumps({"name": "alpha", "metric": {"name": "keyword_score", "value": 0.5}})
    )
    if case != "old_summary":
        _write_population(tmp_path, selected)
    if case in {"wrong_size", "unknown_version"}:
        entry = json.loads((tmp_path / "all_metrics.jsonl").read_text())
        if case == "wrong_size":
            entry["size"] = 4
        else:
            entry["cache_population"]["schema_version"] = 99
        (tmp_path / "all_metrics.jsonl").write_text(json.dumps(entry))
    rows = selected[:1] if case == "missing" else selected
    if case == "extra":
        rows = selected + [{"uid": "c", "score": 0.0}, {"uid": "d", "score": 1.0}]
    if case == "replaced":
        rows = [{"uid": "c", "score": 0.0}, {"uid": "d", "score": 1.0}]
    (cache / "alpha.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    full = _full_rows(tmp_path, "alpha/keyword_score")
    assert len(full) == (2 if case == "extra" else 0)


def test_full_key_pairing_requires_same_population_and_protocol(tmp_path):
    ids = []
    for index, (uids, model, version) in enumerate(
        [
            (["a", "b"], "model-a", 1),
            (["a", "b"], "model-b", 1),
            (["a", "b", "c", "d"], "model-b", 1),
            (["a", "b"], "model-b", 2),
        ]
    ):
        directory = tmp_path / str(index)
        (directory / "cache").mkdir(parents=True)
        rows = [{"uid": uid, "score": 0.5} for uid in uids]
        (directory / "cache/alpha.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
        (directory / "all_metrics.jsonl").write_text(
            json.dumps(
                {
                    "name": "alpha",
                    "metric": {"name": "keyword_score", "value": 0.5},
                    "version": {"dataset": version},
                    "implementation_details": {
                        "solver": {"model": model, "params": {"temperature": 0}}
                    },
                }
            )
        )
        _write_population(directory, rows)
        ids.append({row["uid"] for row in _full_rows(directory, "alpha/keyword_score")})
    assert ids[0] == ids[1]
    assert ids[0].isdisjoint(ids[2])
    assert ids[0].isdisjoint(ids[3])


def test_missing_current_cache_invalidates_remote_full_key(tmp_path):
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache/beta.jsonl").write_text('{"uid":"b"}\n')
    (tmp_path / "all_metrics.jsonl").write_text(
        '{"name":"alpha","metric":{"name":"keyword_score","value":1}}'
    )
    builder = ObmCacheScoresBuilder(tmp_path)
    builder.existing_rows_loader = lambda: iter(
        [
            {"benchmark": "alpha/keyword_score", "uid": "old", "score": 0.0},
            {"benchmark": "gamma", "uid": "keep", "score": 0.4},
        ]
    )
    output = builder.build(tmp_path)
    assert list(iter_compressed_jsonl_bytes(output.read_bytes())) == [
        {"benchmark": "beta", "uid": "b"},
        {"benchmark": "gamma", "uid": "keep", "score": 0.4},
    ]


@pytest.mark.parametrize(
    ("run_files", "expected_sources"),
    [
        pytest.param(None, ["from_a_previous_run", "from_this_run"], id="whole-dir"),
        pytest.param({"from_this_run"}, ["from_this_run"], id="this-measurement-only"),
    ],
)
def test_obm_builder_reads_only_the_cache_files_it_was_given(
    tmp_path: Path,
    run_files: set[str] | None,
    expected_sources: list[str],
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    for benchmark in ("from_a_previous_run", "from_this_run"):
        (cache_dir / f"{benchmark}.jsonl").write_text(
            '{"uid": "1"}\n', encoding="utf-8"
        )

    builder = ObmCacheScoresBuilder(
        tmp_path,
        (
            None
            if run_files is None
            else {cache_dir / f"{name}.jsonl" for name in run_files}
        ),
    )

    assert list(builder.iter_source_files()) == [
        cache_dir / f"{name}.jsonl" for name in expected_sources
    ]


def test_iter_jsonl_preserves_complete_rows_from_truncated_gzip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = encoded_path(tmp_path / "private_benchmark.jsonl")
    path.write_bytes(gzip.compress(b'{"uid": "complete"}\n')[:-8])
    warnings = []
    monkeypatch.setattr(
        "node_runner.src.job_dispatcher.cache_scores.base.Logger.warning",
        warnings.append,
    )

    assert list(iter_jsonl(path)) == [{"uid": "complete"}]
    assert len(warnings) == 1
    assert path.name in warnings[0]
    assert "private_benchmark" not in warnings[0]


def test_obm_builder_ignores_non_jsonl_cache_files(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "alpha.jsonl").write_text('{"uid": "a"}\n', encoding="utf-8")
    encoded_path(cache / "notes.txt").write_bytes(gzip.compress(b"notes"))

    output = ObmCacheScoresBuilder(tmp_path).build(tmp_path)

    assert list(iter_compressed_jsonl_bytes(output.read_bytes())) == [
        {"benchmark": "alpha", "uid": "a"}
    ]
