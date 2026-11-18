"""CPU-only contract tests for offline timing-attempt adjudication."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

sys.dont_write_bytecode = True
_SCRIPT = Path(__file__).with_name("adjudicate.py")
_SPEC = importlib.util.spec_from_file_location("schedule_embedding_adjudicate_tests", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
adjudicate = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = adjudicate
_SPEC.loader.exec_module(adjudicate)


def test_nullable_missing_stage_is_rejected_by_old_guard_and_allowed_with_all_evidence() -> None:
    expected = {
        "parallelism.required_control": False,
        f"parallelism.{adjudicate.STAGES_FIELD}": None,
    }
    resolved = {"parallelism": {"required_control": False}}
    error = adjudicate._strict_guard_error(resolved, expected)
    assert error == f"resolved config lacks parallelism.{adjudicate.STAGES_FIELD}"

    evidence = {key: True for key in (
        "typed_default_none",
        "typed_annotation_nullable",
        "resolved_yaml_omits_selector",
        "explicit_child_null_override",
        "native_typed_decode_is_none",
        "all_rank_module_stage_receipts_valid",
    )}
    assert adjudicate._check_expected_config(
        resolved, expected, nullable_stage_evidence=evidence
    ) == f"parallelism.{adjudicate.STAGES_FIELD}"


def test_nullable_exception_does_not_accept_missing_or_wrong_required_controls() -> None:
    evidence = {key: True for key in (
        "typed_default_none",
        "typed_annotation_nullable",
        "resolved_yaml_omits_selector",
        "explicit_child_null_override",
        "native_typed_decode_is_none",
        "all_rank_module_stage_receipts_valid",
    )}
    expected = {
        "parallelism.required_control": False,
        f"parallelism.{adjudicate.STAGES_FIELD}": None,
    }
    with pytest.raises(adjudicate.AdjudicationError, match="resolved config lacks parallelism.required_control"):
        adjudicate._check_expected_config(
            {"parallelism": {}}, expected, nullable_stage_evidence=evidence
        )
    with pytest.raises(adjudicate.AdjudicationError, match="resolved config .* expected"):
        adjudicate._check_expected_config(
            {
                "parallelism": {
                    "required_control": False,
                    adjudicate.STAGES_FIELD: ["compact", "routing"],
                }
            },
            {**expected, f"parallelism.{adjudicate.STAGES_FIELD}": ["compact"]},
            nullable_stage_evidence=evidence,
        )
    with pytest.raises(adjudicate.AdjudicationError, match="resolved config lacks"):
        adjudicate._check_expected_config(
            {"parallelism": {"required_control": False}},
            {**expected, f"parallelism.{adjudicate.STAGES_FIELD}": ["compact"]},
            nullable_stage_evidence=evidence,
        )


def test_missing_stage_rejected_when_nullable_evidence_is_incomplete() -> None:
    expected = {f"parallelism.{adjudicate.STAGES_FIELD}": None}
    evidence = {"typed_default_none": True, "all_rank_module_stage_receipts_valid": False}
    with pytest.raises(adjudicate.AdjudicationError, match="resolved config lacks"):
        adjudicate._check_expected_config(
            {"parallelism": {}}, expected, nullable_stage_evidence=evidence
        )


def _stage_receipts() -> list[dict[str, object]]:
    source = {"campaign.py": "a" * 64}
    return [
        {
            "run_id": "schedule-embedding-a-timing-140-003",
            "rank": rank,
            "world_size": 8,
            "variant": "A",
            "source_sha256": source,
            "capture_stage": "after_trainer_init_before_first_optimizer_update",
            "device_reads": 0,
            "expected_module_count": 0,
            "expected_stages": [],
            "modules": [],
        }
        for rank in range(8)
    ]


def test_all_rank_empty_stage_receipts_validate_and_missing_rank_fails() -> None:
    receipts = _stage_receipts()
    result = adjudicate._check_stage_receipts(
        receipts,
        run_id="schedule-embedding-a-timing-140-003",
        variant="A",
        source_hashes={"campaign.py": "a" * 64},
        expected_stages=[],
    )
    assert result["rank_count"] == 8
    with pytest.raises(adjudicate.AdjudicationError, match="expected 8 stage receipts"):
        adjudicate._check_stage_receipts(
            receipts[:-1],
            run_id="schedule-embedding-a-timing-140-003",
            variant="A",
            source_hashes={"campaign.py": "a" * 64},
            expected_stages=[],
        )


def _complete_summary() -> dict[str, object]:
    return {
        "status": "complete",
        "errors": [],
        "training_steps": 140,
        "world_size": 8,
        "mode": "complete_window",
        "window": {
            "warmup_updates": 40,
            "start_after_update": 40,
            "end_after_update": 140,
            "updates": 100,
            "global_envelope_convention": "max(rank end monotonic timestamp) - min(rank start timestamp)",
            "primary_elapsed_s": 1000.0,
            "per_rank": {
                str(rank): {
                    "physical_input_tokens_local": 26_214_400,
                    "supervised_tokens_global": 209_664_000,
                }
                for rank in range(8)
            },
        },
        "tokens": {
            "physical_input_tokens_global": 209_715_200,
            "supervised_tokens_global": 209_664_000,
        },
        "memory": {
            "by_rank": {
                str(rank): {
                    "peak_bytes": {
                        "allocated_bytes.all.peak": 1,
                        "reserved_bytes.all.peak": 2,
                    },
                    "allocator_counter_deltas": {"num_alloc_retries": 0, "num_ooms": 0},
                }
                for rank in range(8)
            }
        },
    }


def test_exact_window_summary_passes_and_incomplete_window_fails() -> None:
    summary = _complete_summary()
    adjudicate._validate_timing_summary(summary)  # type: ignore[arg-type]
    summary["window"]["updates"] = 99  # type: ignore[index]
    with pytest.raises(adjudicate.AdjudicationError, match=r"40\+100 token window"):
        adjudicate._validate_timing_summary(summary)  # type: ignore[arg-type]


def test_source_and_config_mismatch_fail() -> None:
    source = {"head": "head", "tracked_and_untracked_sha256": "tree"}
    config = {"base": "a" * 64, "flash_qla_bridge_sources": {"flash": "b" * 64}}
    freeze = {
        "gift_source_snapshot": source,
        "base_config_hashes": {"base": "a" * 64},
        "flash_source_identity": {"bridge_sources": {"flash": "b" * 64}},
    }
    adjudicate._check_source_config_identity(
        source_before=source,
        source_after=source,
        config_before=config,
        config_after=config,
        freeze=freeze,
    )
    changed_source = {**source, "tracked_and_untracked_sha256": "changed"}
    with pytest.raises(adjudicate.AdjudicationError, match="final source snapshot"):
        adjudicate._check_source_config_identity(
            source_before=source,
            source_after=changed_source,
            config_before=config,
            config_after=config,
            freeze=freeze,
        )
    with pytest.raises(adjudicate.AdjudicationError, match="final config hashes"):
        adjudicate._check_source_config_identity(
            source_before=source,
            source_after=source,
            config_before=config,
            config_after={**config, "base": "changed"},
            freeze=freeze,
        )


def test_input_and_cache_identity_mismatches_fail() -> None:
    cache = Path("/task/runs/production-data-abcd/artifacts/raw/d/cache")
    expected_input = {"dataset_sha256": "d" * 64}
    metadata = {
        "input_data": expected_input,
        "compiler_cache_policy": "reuse",
        "compiler_cache_root": str(cache),
        "compiler_cache_producer_run_id": "proddata-d-timing-001",
    }
    adjudicate._check_input_cache_metadata(metadata, expected_input, cache)
    with pytest.raises(adjudicate.AdjudicationError, match="input receipt"):
        adjudicate._check_input_cache_metadata(
            {**metadata, "input_data": {"dataset_sha256": "wrong"}},
            expected_input,
            cache,
        )
    with pytest.raises(adjudicate.AdjudicationError, match="cache policy"):
        adjudicate._check_input_cache_metadata(
            {**metadata, "compiler_cache_root": "/other/cache"}, expected_input, cache
        )


def test_capacity_bootstrap_receipts_reject_wrong_original_overlay() -> None:
    controls = {
        "gfsdp_mtp_transition_prefetch": False,
        "gfsdp_row_owner_embedding_optimization_stages": None,
        "gfsdp_row_owner_embedding_optimizations": False,
        "gfsdp_row_owner_embeddings": False,
        "gfsdp_split_layer0_ffn": False,
        "gfsdp_split_mtp_ffn": False,
    }
    source = {"campaign.py": "a" * 64}
    receipts = []
    for rank in range(8):
        receipts.append({
            "rank": rank,
            "local_rank": rank,
            "native_main_completed": True,
            "schedule_embedding": {
                "variant": "A",
                "controls": controls,
                "source_sha256": source,
                "capacity_control": {
                    "module_name": "torchtitan.models.common.moe.dispatch.capacity",
                    "module_sha256": "2f034b072fdaf02ac844936366134fafcd5e815588ce750b0dc7dfcc50467120",
                    "loaded_before_native_imports": True,
                },
                "capacity_control_final_check": {
                    "registered_same_object": True,
                    "module_path_is_original_snapshot": True,
                    "current_capacity_file_unchanged": True,
                },
            },
        })
    adjudicate._validate_bootstrap(
        receipts, variant="A", controls=controls, source_hashes=source
    )
    receipts[4]["schedule_embedding"]["capacity_control"]["module_sha256"] = "bad"  # type: ignore[index]
    with pytest.raises(adjudicate.AdjudicationError, match="rank 4 capacity"):
        adjudicate._validate_bootstrap(
            receipts, variant="A", controls=controls, source_hashes=source
        )


def test_repair_004_plan_and_retained_freeze_are_pinned() -> None:
    plan, plan_hash = adjudicate._load_pinned_plan(adjudicate.EXPECTED_PLAN_PATH)
    freeze, freeze_hash = adjudicate._load_retained_freeze(plan)
    assert plan_hash == "1a193606963e2f3c7086a6afb1d6d56ad78692dae54121e257c05c18fc8a5968"
    assert freeze_hash == adjudicate.EXPECTED_FREEZE_SHA256
    assert adjudicate._json_digest(freeze) == adjudicate.EXPECTED_FREEZE_SHA256
    assert adjudicate._plan_run(plan, "A")[0] == "schedule-embedding-a-timing-140-003"
    assert adjudicate._plan_run(plan, "B")[0] == "schedule-embedding-b-timing-140-004"
    assert adjudicate._plan_run(plan, "SE-fused")[0] == "schedule-embedding-se-fused-timing-140-004"
    assert adjudicate._expected_controls("SE")[adjudicate.STAGES_FIELD] == [
        "compact", "owner_add", "owner_lookup"
    ]
    assert adjudicate._expected_controls("SE-fused")[adjudicate.STAGES_FIELD] == [
        "compact", "owner_add", "owner_lookup", "routing"
    ]


def test_repair_plan_rejects_changed_run_id() -> None:
    plan, _ = adjudicate._load_pinned_plan(adjudicate.EXPECTED_PLAN_PATH)
    plan["runs"][0]["run_id"] = "schedule-embedding-b-timing-140-003"
    with pytest.raises(adjudicate.AdjudicationError, match="argv identity"):
        adjudicate._validate_repair_plan(plan)


def test_matrix_gate_blocks_active_attempts_before_validation(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(adjudicate, "RAW_ROOT", tmp_path)
    plan = {"retained_A_run_id": "A", "runs": [{"run_id": "B"}]}
    for run_id, status in (("A", "FAIL"), ("B", "RUNNING")):
        attempt = tmp_path / run_id
        attempt.mkdir()
        (attempt / "status.txt").write_text(status + "\n")
        (attempt / "process.json").write_text(
            '{"process_group_stopped": true, "orphaned_process_group": false}\n'
        )
    with pytest.raises(adjudicate.AdjudicationError, match="every planned timing attempt is terminal"):
        adjudicate._require_all_timing_attempts_terminal(plan)
    (tmp_path / "B/status.txt").write_text("PASS\n")
    adjudicate._require_all_timing_attempts_terminal(plan)


def _frozen_file_fixture() -> tuple[dict[str, object], dict[str, str]]:
    freeze: dict[str, object] = {
        "executed_helpers": {},
        "schedule_source_sha256": {},
        "flash_source_identity": {"bridge_sources": {}},
    }
    digests: dict[str, str] = {}
    for index, (name, path) in enumerate(adjudicate.EXPECTED_EXECUTED_HELPERS.items(), start=1):
        digest = f"{index:064x}"
        freeze["executed_helpers"][name] = {"path": str(path), "sha256": digest}  # type: ignore[index]
        digests[str(path)] = digest
    for index, relative in enumerate(adjudicate.EXPECTED_SCHEDULE_SOURCE_FILES, start=20):
        digest = adjudicate.EXPECTED_CAMPAIGN_SHA256 if relative.endswith("campaign.py") else f"{index:064x}"
        freeze["schedule_source_sha256"][relative] = digest  # type: ignore[index]
        digests[str(adjudicate.ROOT / relative)] = digest
    for index, relative in enumerate(adjudicate.EXPECTED_FLASH_BRIDGE_FILES, start=30):
        digest = f"{index:064x}"
        freeze["flash_source_identity"]["bridge_sources"][relative] = digest  # type: ignore[index]
        digests[str(adjudicate.ROOT / relative)] = digest
    return freeze, digests


def test_matching_retained_runtime_hashes_authenticate_before_import() -> None:
    freeze, digests = _frozen_file_fixture()
    original_hash_file = adjudicate._sha256_file
    try:
        adjudicate._sha256_file = lambda path: digests[str(path)]
        verified = adjudicate._authenticate_frozen_files(freeze)
    finally:
        adjudicate._sha256_file = original_hash_file
    assert len(verified) == 14


@pytest.mark.parametrize("tamper", ["overnight_parser", "overnight_runner", "flash_run"])
def test_tampered_helper_or_bridge_fails_before_runtime_or_parser(
    tamper: str, tmp_path: Path, monkeypatch
) -> None:
    freeze, digests = _frozen_file_fixture()
    if tamper == "flash_run":
        target = str(adjudicate.ROOT / "runs/flash-qla/run.py")
        expected_message = "frozen FlashQLA bridge source changed"
    else:
        target = str(adjudicate.EXPECTED_EXECUTED_HELPERS[tamper])
        expected_message = "frozen helper hash mismatch"
    monkeypatch.setattr(adjudicate, "_sha256_file", lambda path: "f" * 64 if str(path) == target else digests[str(path)])

    raw_root = tmp_path / "raw"
    attempt = raw_root / "schedule-embedding-a-timing-140-003"
    attempt.mkdir(parents=True)
    output_root = tmp_path / "adjudication"
    monkeypatch.setattr(adjudicate, "RAW_ROOT", raw_root)
    monkeypatch.setattr(adjudicate, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(
        adjudicate,
        "_load_pinned_plan",
        lambda _path: ({"retained_A_run_id": attempt.name, "runs": []}, "1" * 64),
    )
    monkeypatch.setattr(
        adjudicate,
        "_load_retained_freeze",
        lambda _plan: (freeze, adjudicate.EXPECTED_FREEZE_SHA256),
    )
    monkeypatch.setattr(adjudicate, "_require_all_timing_attempts_terminal", lambda _plan: None)
    monkeypatch.setattr(adjudicate, "_load_raw_hash_paths", lambda _attempt: pytest.fail("raw hashing ran early"))
    monkeypatch.setattr(adjudicate, "_load_runtime", lambda: pytest.fail("runtime import ran before hash check"))
    monkeypatch.setattr(adjudicate, "_load_module", lambda *_args: pytest.fail("parser loaded before hash check"))

    with pytest.raises(adjudicate.AdjudicationError, match=expected_message):
        adjudicate.adjudicate(
            "A",
            attempt,
            output_root / "A",
            Path("repair-004-plan.json"),
        )
    assert not (output_root / "A").exists()


def test_terminal_gate_precedes_frozen_file_hash_reads(tmp_path: Path, monkeypatch) -> None:
    raw_root = tmp_path / "raw"
    attempt = raw_root / "schedule-embedding-a-timing-140-003"
    attempt.mkdir(parents=True)
    output_root = tmp_path / "adjudication"
    monkeypatch.setattr(adjudicate, "RAW_ROOT", raw_root)
    monkeypatch.setattr(adjudicate, "OUTPUT_ROOT", output_root)
    plan = {"retained_A_run_id": attempt.name, "runs": []}
    monkeypatch.setattr(adjudicate, "_load_pinned_plan", lambda _path: (plan, "1" * 64))
    monkeypatch.setattr(adjudicate, "_load_retained_freeze", lambda _plan: ({}, "2" * 64))
    monkeypatch.setattr(
        adjudicate,
        "_require_all_timing_attempts_terminal",
        lambda _plan: (_ for _ in ()).throw(adjudicate.AdjudicationError("active gate")),
    )
    monkeypatch.setattr(adjudicate, "_authenticate_frozen_files", lambda _freeze: pytest.fail("hashes ran before gate"))
    monkeypatch.setattr(adjudicate, "_load_runtime", lambda: pytest.fail("runtime import ran before gate"))
    with pytest.raises(adjudicate.AdjudicationError, match="active gate"):
        adjudicate.adjudicate("A", attempt, output_root / "A", Path("repair-004-plan.json"))
    assert not (output_root / "A").exists()


def _diagnostic_profile_parser():
    class Parser:
        @staticmethod
        def _expected_profile_spec(target_update: int, warmup_updates: int) -> dict[str, object]:
            return {
                "target_update": target_update,
                "warmup_updates": warmup_updates,
                "profile_ranks": [0, 1],
                "settings": {
                    "profiler.enable_profiling": True,
                    "profiler.enable_memory_snapshot": False,
                    "profiler.profile_ranks": [0, 1],
                    "profiler.save_traces_folder": "profiling/traces",
                    "profiler.record_shapes": False,
                    "profiler.profile_freq": 2,
                    "profiler.profiler_warmup": 1,
                    "profiler.profiler_active": 1,
                    "profiler.profiler_repeat": 1,
                    "profiler.profiler_skip_first": target_update - 2,
                    "profiler.profiler_skip_first_wait": 0,
                },
                "native_hardcoded": {"with_stack": False, "profile_memory": False},
                "trace_directory": f"native/profiling/traces/iteration_{target_update}",
            }

    return Parser()


def _diagnostic_summary_and_metadata() -> tuple[dict[str, object], dict[str, object]]:
    parser = _diagnostic_profile_parser()
    profile = parser._expected_profile_spec(21, 20)
    profile["profile_microbatches"] = 3
    calls = {
        str(rank): {
            "microbatch_calls": 32,
            "train_step_calls": 1,
            "optimizer_step_calls": 1,
            "native_profiler_build_calls": 1,
            "native_profiler_created": rank in (0, 1),
        }
        for rank in range(8)
    }
    traces = {}
    for rank in (0, 1):
        traces[str(rank)] = {
            "path": f"native/profiling/traces/iteration_21/rank{rank}_trace.json",
            "size_bytes": 100,
            "sha256": f"{rank + 1:064x}",
            "marker_counts": {
                f"task.profile_update.21.microbatch.rank{rank}": 3,
                f"task.profile_update.21.train_step.rank{rank}": 1,
            },
            "gpu_projection_counts": {
                f"task.profile_update.21.microbatch.rank{rank}": 3,
            },
        }
    summary: dict[str, object] = {
        "status": "complete",
        "mode": "profile_update",
        "run_id": "schedule-embedding-se-diagnostic-140-004",
        "training_steps": 21,
        "measurement_status": "disabled",
        "profile_capture": {
            "target_update": 21,
            "warmup_updates": 20,
            "profile_ranks": [0, 1],
            "settings": profile["settings"],
            "native_hardcoded": profile["native_hardcoded"],
            "call_counts_by_rank": calls,
            "traces_by_rank": traces,
        },
        "errors": [],
    }
    return summary, {"profile_capture": profile}


def test_diagnostic_plan_maps_exact_pinned_update_21_argv() -> None:
    plan, _ = adjudicate._load_pinned_plan(adjudicate.EXPECTED_PLAN_PATH)
    for variant, row in zip(adjudicate.DIAGNOSTIC_VARIANTS, plan["diagnostics"]):
        run_id, argv = adjudicate._plan_run(plan, variant, "diagnostic")
        assert run_id == row["run_id"] == f"schedule-embedding-{variant.lower()}-diagnostic-140-004"
        assert argv == row["argv"]
        assert adjudicate._option_values(argv, "--steps") == ["21"]
        assert adjudicate._option_values(argv, "--warmup") == ["20"]
        assert adjudicate._option_values(argv, "--window") == ["0"]
        assert adjudicate._option_values(argv, "--profile-update") == ["21"]
        assert adjudicate._option_values(argv, "--profile-microbatches") == ["3"]


def test_cli_defaults_to_timing_and_exposes_diagnostic_mode() -> None:
    common = ["--variant", "D0", "--run-dir", "/tmp/raw", "--output-dir", "/tmp/out", "--plan", "/tmp/plan"]
    assert adjudicate._parse_args(common).mode == "timing"
    assert adjudicate._parse_args(["--mode", "diagnostic", *common]).mode == "diagnostic"


def test_diagnostic_summary_records_parser_pass_with_tail_still_pending() -> None:
    summary, metadata = _diagnostic_summary_and_metadata()
    proof = adjudicate._validate_diagnostic_summary(
        summary, metadata, _diagnostic_profile_parser()
    )
    assert proof["status"] == "NATIVE_PROFILE_PARSER_PASS"
    assert proof["trace_tail_coverage"]["status"] == "PENDING_PERFETTO"
    assert proof["trace_tail_coverage"]["overall_acceptance_complete"] is False
    adjudicate._check_original_trace_hashes(proof, proof["expected_original_trace_hashes"])
    hashes = dict(proof["expected_original_trace_hashes"])
    hashes[next(iter(hashes))] = "f" * 64
    with pytest.raises(adjudicate.AdjudicationError, match="original trace bytes"):
        adjudicate._check_original_trace_hashes(proof, hashes)


def test_diagnostic_summary_rejects_wrong_mode_steps_and_full_profile_spec() -> None:
    summary, metadata = _diagnostic_summary_and_metadata()
    parser = _diagnostic_profile_parser()
    summary["training_steps"] = 20
    with pytest.raises(adjudicate.AdjudicationError, match="update-21 profile"):
        adjudicate._validate_diagnostic_summary(summary, metadata, parser)

    summary, metadata = _diagnostic_summary_and_metadata()
    metadata["profile_capture"]["settings"]["profiler.record_shapes"] = True
    with pytest.raises(adjudicate.AdjudicationError, match="full frozen profile specification"):
        adjudicate._validate_diagnostic_summary(summary, metadata, parser)

    summary, metadata = _diagnostic_summary_and_metadata()
    summary["profile_capture"]["call_counts_by_rank"]["1"]["microbatch_calls"] = 31
    with pytest.raises(adjudicate.AdjudicationError, match="rank 1 native profile call counters"):
        adjudicate._validate_diagnostic_summary(summary, metadata, parser)

    summary, metadata = _diagnostic_summary_and_metadata()
    summary["status"] = "incomplete"
    with pytest.raises(adjudicate.AdjudicationError, match="complete update-21 profile"):
        adjudicate._validate_diagnostic_summary(summary, metadata, parser)


def test_nullable_wrapper_failure_is_accepted_only_for_exact_d0_or_s_guard() -> None:
    values = {
        "status": "FAIL",
        "exit_code": 1,
        "failure": {
            "status": "FAIL",
            "error": adjudicate.KNOWN_NULLABLE_GUARD_ERROR,
            "timed_out": False,
        },
        "completion": None,
        "raw_summary": None,
        "summary": {"status": "complete"},
    }
    for variant in ("D0", "S"):
        assert adjudicate._validate_diagnostic_wrapper_receipts(
            variant=variant, **values
        ) == adjudicate.KNOWN_NULLABLE_GUARD_ERROR
    for variant in ("SE", "SE-fused"):
        with pytest.raises(adjudicate.AdjudicationError, match="unsupported original diagnostic"):
            adjudicate._validate_diagnostic_wrapper_receipts(variant=variant, **values)

    wrong_failure = {**values, "failure": {**values["failure"], "error": "RuntimeError: other"}}
    with pytest.raises(adjudicate.AdjudicationError, match="exact pre-parser"):
        adjudicate._validate_diagnostic_wrapper_receipts(variant="D0", **wrong_failure)
    with pytest.raises(adjudicate.AdjudicationError, match="exact pre-parser"):
        adjudicate._validate_diagnostic_wrapper_receipts(
            variant="S", **{**values, "raw_summary": {"status": "complete"}}
        )

    summary = {"status": "complete", "run_id": "schedule-embedding-se-diagnostic-140-004"}
    passed = {
        "status": "PASS",
        "exit_code": 0,
        "failure": None,
        "completion": {
            "status": "PASS",
            "source_unchanged": True,
            "config_unchanged": True,
        },
        "raw_summary": summary,
        "summary": summary,
    }
    assert adjudicate._validate_diagnostic_wrapper_receipts(variant="SE", **passed) is None
    passed["completion"]["config_unchanged"] = False
    with pytest.raises(adjudicate.AdjudicationError, match="PASS wrapper receipts"):
        adjudicate._validate_diagnostic_wrapper_receipts(variant="SE-fused", **passed)


def test_diagnostic_task_tmpdir_cleanup_requires_matching_removed_receipts(tmp_path: Path) -> None:
    attempt = tmp_path / "attempt"
    (attempt / "preflight").mkdir(parents=True)
    missing_tmpdir = tmp_path / "removed-run-tmpdir"
    metadata = {"task_tmpdir": {"path": str(missing_tmpdir)}}
    (attempt / "preflight/task-tmpdir.json").write_text(
        '{"run_id":"D0","path":"' + str(missing_tmpdir) + '"}\n'
    )
    (attempt / "task-tmpdir-cleanup.json").write_text(
        '{"path":"' + str(missing_tmpdir) + '","removed":true}\n'
    )
    completion = {"task_tmpdir": str(missing_tmpdir), "task_tmpdir_removed": True}
    adjudicate._validate_diagnostic_task_tmpdir_cleanup(
        attempt, run_id="D0", metadata=metadata, completion=completion
    )
    (attempt / "task-tmpdir-cleanup.json").write_text(
        '{"path":"' + str(missing_tmpdir) + '","removed":false}\n'
    )
    with pytest.raises(adjudicate.AdjudicationError, match="cleanup is not proven"):
        adjudicate._validate_diagnostic_task_tmpdir_cleanup(
            attempt, run_id="D0", metadata=metadata, completion=None
        )


def _diagnostic_plan(run_id: str) -> dict[str, object]:
    return {
        "retained_A_run_id": "schedule-embedding-a-timing-140-003",
        "runs": [],
        "diagnostics": [
            {
                "name": f"{variant}-diagnostic",
                "run_id": run_id
                if variant == "D0"
                else f"schedule-embedding-{variant.lower()}-diagnostic-140-004",
                "argv": ["pinned", "diagnostic"],
            }
            for variant in adjudicate.DIAGNOSTIC_VARIANTS
        ],
    }


def test_diagnostic_prerequisite_gate_precedes_raw_hashing_and_imports(
    tmp_path: Path, monkeypatch
) -> None:
    raw_root = tmp_path / "raw"
    attempt = raw_root / "schedule-embedding-d0-diagnostic-140-004"
    output_root = tmp_path / "adjudication-diagnostic-004"
    events: list[str] = []
    plan = _diagnostic_plan(attempt.name)
    for row in plan["diagnostics"]:
        diagnostic_attempt = raw_root / row["run_id"]
        diagnostic_attempt.mkdir(parents=True)
        (diagnostic_attempt / "status.txt").write_text("PASS\n")
        (diagnostic_attempt / "process.json").write_text(
            '{"process_group_stopped":true,"orphaned_process_group":false}\n'
        )
    monkeypatch.setattr(adjudicate, "RAW_ROOT", raw_root)
    monkeypatch.setattr(adjudicate, "DIAGNOSTIC_OUTPUT_ROOT", output_root)
    monkeypatch.setattr(adjudicate, "_load_pinned_plan", lambda _path: (plan, "1" * 64))
    monkeypatch.setattr(adjudicate, "_load_retained_freeze", lambda _plan: ({}, "2" * 64))
    monkeypatch.setattr(adjudicate, "_require_all_timing_attempts_terminal", lambda _plan: events.append("timing-terminal"))
    terminal_gate = adjudicate._require_all_diagnostic_attempts_terminal

    def record_diagnostic_terminal_gate(current_plan):
        events.append("diagnostics-terminal")
        terminal_gate(current_plan)

    monkeypatch.setattr(adjudicate, "_require_all_diagnostic_attempts_terminal", record_diagnostic_terminal_gate)

    def fail_prerequisite(_plan, _freeze, _plan_hash):
        events.append("timing-adjudications")
        raise adjudicate.AdjudicationError("timing prerequisites missing")

    monkeypatch.setattr(adjudicate, "_require_all_timing_adjudications_pass", fail_prerequisite)
    monkeypatch.setattr(adjudicate, "_authenticate_frozen_files", lambda _freeze: pytest.fail("frozen hashes ran before timing prerequisites"))
    monkeypatch.setattr(adjudicate, "_load_raw_hash_paths", lambda *_args, **_kwargs: pytest.fail("raw hashing ran before timing prerequisites"))
    monkeypatch.setattr(adjudicate, "_load_runtime", lambda: pytest.fail("runtime import ran before timing prerequisites"))
    monkeypatch.setattr(adjudicate, "_load_module", lambda *_args: pytest.fail("parser import ran before timing prerequisites"))

    with pytest.raises(adjudicate.AdjudicationError, match="timing prerequisites missing"):
        adjudicate.adjudicate(
            "D0", attempt, output_root / "D0", Path("repair-004-plan.json"), mode="diagnostic"
        )
    assert events == ["timing-terminal", "diagnostics-terminal", "timing-adjudications"]
    assert not (output_root / "D0").exists()


@pytest.mark.parametrize("other_status", ["RUNNING", None])
def test_all_diagnostic_attempts_must_be_terminal_before_timing_reads(
    other_status: str | None, tmp_path: Path, monkeypatch
) -> None:
    raw_root = tmp_path / "raw"
    attempt = raw_root / "schedule-embedding-d0-diagnostic-140-004"
    attempt.mkdir(parents=True)
    (attempt / "status.txt").write_text("PASS\n")
    (attempt / "process.json").write_text(
        '{"process_group_stopped":true,"orphaned_process_group":false}\n'
    )
    other = raw_root / "schedule-embedding-s-diagnostic-140-004"
    other.mkdir(parents=True)
    if other_status is not None:
        (other / "status.txt").write_text(other_status + "\n")
    output_root = tmp_path / "adjudication-diagnostic-004"
    plan = _diagnostic_plan(attempt.name)
    monkeypatch.setattr(adjudicate, "RAW_ROOT", raw_root)
    monkeypatch.setattr(adjudicate, "DIAGNOSTIC_OUTPUT_ROOT", output_root)
    monkeypatch.setattr(adjudicate, "_load_pinned_plan", lambda _path: (plan, "1" * 64))
    monkeypatch.setattr(adjudicate, "_load_retained_freeze", lambda _plan: ({}, "2" * 64))
    monkeypatch.setattr(adjudicate, "_require_all_timing_attempts_terminal", lambda _plan: None)
    monkeypatch.setattr(adjudicate, "_require_all_timing_adjudications_pass", lambda *_args: pytest.fail("timing proofs read while diagnostic is active"))
    monkeypatch.setattr(adjudicate, "_authenticate_frozen_files", lambda _freeze: pytest.fail("frozen hashes read while diagnostic is active"))
    monkeypatch.setattr(adjudicate, "_load_runtime", lambda: pytest.fail("runtime import while diagnostic is active"))

    with pytest.raises(adjudicate.AdjudicationError, match="requested attempt is terminal"):
        adjudicate.adjudicate(
            "D0", attempt, output_root / "D0", Path("repair-004-plan.json"), mode="diagnostic"
        )
    assert not (output_root / "D0").exists()


def test_all_seven_final_timing_proofs_must_match_current_method_and_raw_hashes(
    tmp_path: Path, monkeypatch
) -> None:
    plan, plan_hash = adjudicate._load_pinned_plan(adjudicate.EXPECTED_PLAN_PATH)
    freeze, _digests = _frozen_file_fixture()
    raw_root = tmp_path / "raw"
    output_root = tmp_path / "adjudication-140-004"
    retained_path = tmp_path / "historical-A-recovery-003.json"
    monkeypatch.setattr(adjudicate, "RAW_ROOT", raw_root)
    monkeypatch.setattr(adjudicate, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(adjudicate, "RETAINED_FREEZE_PATH", retained_path)
    monkeypatch.setattr(
        adjudicate,
        "_load_raw_hash_paths",
        lambda attempt: [attempt / "status.txt"],
    )
    method_hash = adjudicate._sha256_file(Path(adjudicate.__file__).resolve())
    expected_runtime_hashes = adjudicate._frozen_file_hash_manifest(freeze)

    for variant in adjudicate.VARIANTS:
        run_id, _argv = adjudicate._plan_run(plan, variant, "timing")
        attempt = raw_root / run_id
        attempt.mkdir(parents=True)
        (attempt / "status.txt").write_text("PASS\n")
        (attempt / "exit-code.txt").write_text("0\n")
        record_dir = output_root / variant
        record_dir.mkdir(parents=True)
        summary_path = record_dir / "summary.json"
        summary_path.write_text('{"status":"complete"}\n')
        raw_hashes = adjudicate._raw_hashes([attempt / "status.txt"], attempt)
        record = {
            "status": "ADJUDICATED_PASS",
            "variant": variant,
            "run_id": run_id,
            "original_wrapper_status": "PASS",
            "original_exit_code": 0,
            "raw_receipts_unchanged": True,
            "source_freeze_sha256": adjudicate.EXPECTED_FREEZE_SHA256,
            "source_freeze": freeze,
            "launch_plan_path": str(adjudicate.EXPECTED_PLAN_PATH),
            "launch_plan_sha256": plan_hash,
            "adjudicator_method_sha256": method_hash,
            "authenticated_runtime_files": expected_runtime_hashes,
            "source_freeze_provenance": {
                "retained_record": str(retained_path),
                "current_full_source_walk_recomputed": False,
            },
            "raw_attempt": str(attempt.resolve()),
            "raw_receipt_hashes": raw_hashes,
            "adjudication_path": str((record_dir / "adjudication.json").resolve()),
            "parser_summary_path": str(summary_path.resolve()),
            "parser_summary_sha256": adjudicate._sha256_file(summary_path),
        }
        adjudicate._write_json_new(record_dir / "adjudication.json", record)

    assert all(
        adjudicate._timing_adjudication_path(variant)
        == output_root / variant / "adjudication.json"
        for variant in adjudicate.VARIANTS
    )
    adjudicate._require_all_timing_adjudications_pass(plan, freeze, plan_hash)

    late_record_path = output_root / "SE-fused/adjudication.json"
    late_record_payload = late_record_path.read_text()
    late_record = json.loads(late_record_payload)
    late_record["adjudicator_method_sha256"] = "f" * 64
    late_record_path.write_text(json.dumps(late_record))
    original_raw_hashes = adjudicate._raw_hashes
    raw_hash_calls: list[Path] = []
    monkeypatch.setattr(
        adjudicate,
        "_raw_hashes",
        lambda paths, attempt: raw_hash_calls.append(attempt) or {},
    )
    with pytest.raises(adjudicate.AdjudicationError, match="SE-fused lacks a matching"):
        adjudicate._require_all_timing_adjudications_pass(plan, freeze, plan_hash)
    assert raw_hash_calls == []
    monkeypatch.setattr(adjudicate, "_raw_hashes", original_raw_hashes)
    late_record_path.write_text(late_record_payload)

    changed_status = raw_root / "schedule-embedding-a-timing-140-003/status.txt"
    changed_status.write_text("FAIL\n")
    with pytest.raises(adjudicate.AdjudicationError, match="original timing status differs"):
        adjudicate._require_all_timing_adjudications_pass(plan, freeze, plan_hash)
