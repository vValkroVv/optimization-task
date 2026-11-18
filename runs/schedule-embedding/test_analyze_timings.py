"""Targeted tests for the schedule-embedding timing receipt analyzer."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import analyze_timings as analyzer
from analyze_timings import (
    LABELS,
    STAGE_RECEIPT_CAPTURE,
    STAGE_TUPLES,
    _cadence_analysis,
    _controls_without_stage_selector,
    _published_run,
    _stage_selector_fields,
    analyze_campaign,
    compare_runs,
    extract_receipt,
    percentile,
    primary_window_metrics,
    summarize,
    validate_cadence_steps,
    validate_run_receipt,
    validate_stage_selection_receipts,
)


ROOT = Path(__file__).resolve().parents[2]
OLD_SE = (
    ROOT
    / "runs/schedule-embedding/artifacts/raw/schedule-embedding-se-timing-001"
)
ACTUAL_TIMING_RESULT = ROOT / "runs/schedule-embedding/timing-results-140.json"


class TimingMathTests(unittest.TestCase):
    def test_linear_percentiles_and_summary(self) -> None:
        self.assertEqual(percentile([0, 10, 20, 30, 40], 0.10), 4.0)
        self.assertEqual(percentile([0, 10, 20, 30, 40], 0.90), 36.0)
        summary = summarize([1, 2, 3, 4, 5])
        self.assertEqual(summary["median"], 3)
        self.assertEqual(summary["mean"], 3)
        self.assertEqual(summary["p10"], 1.4)
        self.assertEqual(summary["p90"], 4.6)

    def test_primary_envelope_uses_latest_end_minus_earliest_start(self) -> None:
        result = primary_window_metrics(
            {0: 10_000_000_000, 1: 10_020_000_000},
            {0: 20_000_000_000, 1: 20_030_000_000},
            physical_tokens=1_000,
            supervised_tokens=900,
            measured_updates=10,
        )
        self.assertAlmostEqual(result["global_envelope_s"], 10.03)
        self.assertAlmostEqual(result["mean_update_s"], 1.003)
        self.assertAlmostEqual(result["physical_tokens_per_second_global"], 1_000 / 10.03)
        self.assertAlmostEqual(result["supervised_tokens_per_second_global"], 900 / 10.03)


    def test_comparison_ratios_use_candidate_over_reference(self) -> None:
        memory = {
            str(rank): {
                "peak_bytes": {
                    "allocated_bytes.all.peak": 200,
                    "reserved_bytes.all.peak": 400,
                }
            }
            for rank in range(8)
        }
        counters = {str(rank): {"num_alloc_retries": 0, "num_ooms": 0} for rank in range(8)}
        reference = {
            "label": "A",
            "primary": {
                "physical_tokens_per_second_global": 100.0,
                "supervised_tokens_per_second_global": 90.0,
                "mean_update_s": 4.0,
            },
            "memory_by_rank": memory,
            "allocator_counter_deltas_by_rank_from_parser": counters,
        }
        candidate = copy.deepcopy(reference)
        candidate["label"] = "B"
        candidate["primary"] = {
            "physical_tokens_per_second_global": 200.0,
            "supervised_tokens_per_second_global": 180.0,
            "mean_update_s": 3.0,
        }
        for value in candidate["memory_by_rank"].values():
            value["peak_bytes"]["allocated_bytes.all.peak"] = 400
            value["peak_bytes"]["reserved_bytes.all.peak"] = 800
        comparison = compare_runs(candidate, reference, "B/A")
        self.assertEqual(comparison["throughput"]["physical_tokens_per_second_global"]["ratio"], 2.0)
        self.assertEqual(comparison["mean_update_time"]["ratio"], 0.75)
        self.assertEqual(comparison["memory_maxima"]["allocated_bytes.all.peak"]["ratio"], 2.0)

    def test_measured_window_boundaries_and_ten_update_bins(self) -> None:
        cadence = {
            str(rank): [
                {"step": step, "interval_s": float(step + rank)}
                for step in range(41, 141)
            ]
            for rank in range(8)
        }
        analysis = _cadence_analysis({
            "measured_cadence_by_rank": cadence,
            "observed_window_updates": {"start_after_update": 40, "end_after_update": 140},
        })
        self.assertEqual(analysis["measured_updates"], [41, 140])
        self.assertEqual(len(analysis["ten_update_bins"]), 10)
        self.assertEqual(analysis["ten_update_bins"][0]["updates"], [41, 50])
        self.assertEqual(analysis["ten_update_bins"][-1]["updates"], [131, 140])
        windows = analysis["first20_vs_last20"]["window_updates"]
        self.assertEqual(windows["first20"], [41, 60])
        self.assertEqual(windows["last20"], [121, 140])
        self.assertEqual(len(analysis["full_measured_series_by_rank"]["0"]), 100)

    def test_full_run_and_warmup_bins_keep_startup_cadence(self) -> None:
        cadence = {
            str(rank): [
                {"step": step, "interval_s": None if step == 1 else float(step + rank)}
                for step in range(1, 141)
            ]
            for rank in range(8)
        }
        measured = {
            rank: [row for row in rows if row["step"] >= 41]
            for rank, rows in cadence.items()
        }
        analysis = _cadence_analysis({
            "cadence_by_rank": cadence,
            "measured_cadence_by_rank": measured,
            "observed_window_updates": {"start_after_update": 40, "end_after_update": 140},
        })
        self.assertEqual(len(analysis["full_training_series_by_rank"]["0"]), 140)
        self.assertIsNone(analysis["full_training_series_by_rank"]["0"][0])
        self.assertEqual(analysis["full_training_series_by_rank"]["0"][1], 2.0)
        self.assertEqual(analysis["startup_updates_1_10"]["largest_interval"]["update"], 10)
        self.assertEqual(
            [item["updates"] for item in analysis["warmup_ten_update_bins"]],
            [[11, 20], [21, 30], [31, 40]],
        )
        self.assertEqual(analysis["warmup_ten_update_bins"][0]["all_rank_statistics"]["count"], 80)

    def test_missing_cadence_step_is_rejected(self) -> None:
        records = [{"step": step} for step in range(41, 141)]
        records.remove({"step": 80})
        errors = validate_cadence_steps(records, range(41, 141))
        self.assertEqual(len(errors), 1)
        self.assertIn("41..140", errors[0])


class PublishedMarkdownTests(unittest.TestCase):
    def test_renders_actual_complete_timing_result_schema(self) -> None:
        result = json.loads(ACTUAL_TIMING_RESULT.read_text(encoding="utf-8"))

        markdown = analyzer.markdown_fragment(result)

        self.assertIn("## Seven-run native timing results", markdown)
        self.assertIn("| Variant | Receipt acceptance |", markdown)
        for label in LABELS:
            self.assertIn(f"| {label} | ADJUDICATED_PASS", markdown)
        run = result["runs"]["A"]
        allocated_gib = (
            run["memory"]["maximum_peak_bytes_across_ranks"]["allocated_bytes.all.peak"]
            / (1024 ** 3)
        )
        self.assertIn("| A | ADJUDICATED_PASS", markdown)
        self.assertIn(f"{allocated_gib:.3f}", markdown)
        self.assertIn("### Incremental and cumulative comparisons", markdown)


class ExistingReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not OLD_SE.is_dir():
            raise AssertionError(f"existing native receipt fixture is missing: {OLD_SE}")
        cls.receipt = extract_receipt(OLD_SE)

    def test_extracts_real_historical_ten_plus_ten_receipt(self) -> None:
        receipt = self.receipt
        self.assertEqual(receipt["run_id"], "schedule-embedding-se-timing-001")
        self.assertEqual(receipt["metadata"]["steps"], 20)
        self.assertEqual(receipt["metadata"]["warmup_updates"], 10)
        self.assertEqual(receipt["metadata"]["window_updates"], 10)
        self.assertEqual(receipt["observed_window_updates"]["first_observed_cache_update_interval"], "(10,20]")
        self.assertEqual(len(receipt["cadence_by_rank"]["0"]), 20)
        self.assertEqual(
            [item["step"] for item in receipt["measured_cadence_by_rank"]["0"]],
            list(range(11, 21)),
        )
        self.assertEqual(
            receipt["actual_embedding_stages_by_rank"]["0"]["tuples"],
            [["compact", "owner_add", "owner_lookup", "routing"]],
        )
        self.assertEqual(
            receipt["actual_embedding_stages_by_rank"]["0"]["source"],
            "historical_20_step_log_fallback",
        )
        self.assertEqual(receipt["missing_embedding_stage_selection_receipts"], list(range(8)))

    def test_old_receipt_cannot_pass_the_new_campaign_gate(self) -> None:
        failures = validate_run_receipt(self.receipt, "SE")
        self.assertTrue(any("metadata steps must be 140" in item for item in failures))
        self.assertTrue(any("window must contain 100 updates" in item for item in failures))
        self.assertTrue(any("embedding-stage-selection receipts must cover ranks 0..7" in item for item in failures))

    def test_published_record_keeps_series_and_hash_refs_without_raw_nested_summary(self) -> None:
        run = copy.deepcopy(self.receipt)
        run["label"] = "SE"
        run["cadence_analysis"] = _cadence_analysis(run)
        run["cache_observation_semantics"] = "boundary evidence only"
        published = _published_run(run)
        encoded = json.dumps(published, allow_nan=False)
        self.assertIn("raw_artifact_sha256", published)
        self.assertIn("actual_embedding_stages_by_rank", published["configuration_controls"])
        self.assertEqual(len(published["all_rank_cadence_records"]["0"]), 20)
        self.assertNotIn("rank_sidecars", published)
        self.assertNotIn("summary", published)
        self.assertTrue(encoded)
        self.assertEqual(published["acceptance"]["mode"], "native_completion")

    def test_strict_campaign_manifest_rejects_historical_matrix_before_output(self) -> None:
        manifest = {
            "_manifest_path": str(ROOT / "runs/schedule-embedding/manifest.json"),
            "runs": [
                {"label": label, "path": str(OLD_SE)} for label in LABELS
            ],
        }
        with self.assertRaisesRegex(ValueError, "seven-run campaign not accepted"):
            analyze_campaign(manifest)

    def test_missing_rank_cadence_is_detected(self) -> None:
        damaged = copy.deepcopy(self.receipt)
        damaged["cadence_by_rank"].pop("7")
        damaged["measured_cadence_by_rank"].pop("7")
        failures = validate_run_receipt(damaged, "SE")
        self.assertTrue(any("cadence does not cover ranks 0..7" in item for item in failures))

    def test_missing_rank_sidecar_is_detected(self) -> None:
        damaged = copy.deepcopy(self.receipt)
        damaged["rank_sidecars"].pop("7")
        failures = validate_run_receipt(damaged, "SE")
        self.assertTrue(any("rank sidecars do not cover ranks 0..7" in item for item in failures))

    def test_native_stage_selection_receipts_are_authoritative(self) -> None:
        frozen_sources = {
            "runs/schedule-embedding/campaign.py": "a" * 64,
            "runs/schedule-embedding/measure.py": "b" * 64,
            "runs/schedule-embedding/run.py": "c" * 64,
        }
        for label in ("SE", "SE-fused"):
            stages = list(STAGE_TUPLES[label])
            modules = [{"name": "tok_embeddings", "optimization_stages": stages}]
            receipts = {
                str(rank): {
                    "schema_version": 1,
                    "run_id": "new-run",
                    "rank": rank,
                    "world_size": 8,
                    "variant": label,
                    "source_sha256": frozen_sources,
                    "capture_stage": STAGE_RECEIPT_CAPTURE,
                    "device_reads": 0,
                    "expected_module_count": 1,
                    "expected_stages": stages,
                    "modules": modules,
                }
                for rank in range(8)
            }
            run = {
                "run_id": "new-run",
                "embedding_stage_selection_receipts_by_rank": receipts,
                "expected_frozen_schedule_source_sha256": frozen_sources,
            }
            self.assertEqual(validate_stage_selection_receipts(run, label), [])
            damaged = copy.deepcopy(run)
            damaged["embedding_stage_selection_receipts_by_rank"]["3"]["modules"][0]["optimization_stages"] = []
            self.assertTrue(any(
                "stages differ from expected" in item
                for item in validate_stage_selection_receipts(damaged, label)
            ))

    def test_se_control_comparison_allows_only_selector_list_to_differ(self) -> None:
        shared = {
            "gfsdp_row_owner_embedding_optimizations": False,
            "gfsdp_row_owner_embeddings": True,
            "gfsdp_split_layer0_ffn": True,
            "gfsdp_split_mtp_ffn": True,
            "gfsdp_mtp_transition_prefetch": True,
            "gfsdp_reuse_bf16_expert_weights": True,
        }
        stage_key = "gfsdp_row_owner_embedding_optimization_stages"
        se = {**shared, stage_key: list(STAGE_TUPLES["SE"])}
        fused = {**shared, stage_key: list(STAGE_TUPLES["SE-fused"])}
        self.assertEqual(_controls_without_stage_selector(se), _controls_without_stage_selector(fused))
        self.assertEqual(_stage_selector_fields(se)[stage_key], list(STAGE_TUPLES["SE"]))


class AdjudicatedCampaignTests(unittest.TestCase):
    BASE_RAW = ROOT / "runs/schedule-embedding/artifacts/raw/schedule-embedding-a-timing-140-003"
    BASE_SUMMARY = (
        ROOT / "runs/schedule-embedding/artifacts/raw/adjudication-140-003/A-recovery-003/summary.json"
    )
    BASE_PROOF = (
        ROOT / "runs/schedule-embedding/artifacts/raw/adjudication-140-003/A-recovery-003/adjudication.json"
    )

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        task_root = Path(self.temp.name) / "task"
        evidence_root = task_root / "runs/schedule-embedding"
        self.raw_root = evidence_root / "artifacts/raw"
        self.adjudication_root = self.raw_root / "adjudication-140-004"
        self.plan_path = self.raw_root / "newlaunchrecords/repair-004-plan.json"
        self.method_path = evidence_root / "adjudicate.py"

        plan_bytes = analyzer.REPAIR_PLAN_PATH.read_bytes()
        self.plan_path.parent.mkdir(parents=True, exist_ok=True)
        self.plan_path.write_bytes(plan_bytes)
        self.plan = json.loads(plan_bytes)
        self.method_path.parent.mkdir(parents=True, exist_ok=True)
        self.method_path.write_text("reviewed synthetic timing method\n", encoding="utf-8")
        self.method_sha256 = analyzer._sha256(self.method_path)

        self.base_metadata = json.loads((self.BASE_RAW / "metadata.json").read_text(encoding="utf-8"))
        self.base_source_final = json.loads((self.BASE_RAW / "source-final.json").read_text(encoding="utf-8"))
        self.base_config_final = json.loads((self.BASE_RAW / "config-hashes-final.json").read_text(encoding="utf-8"))
        self.base_process = json.loads((self.BASE_RAW / "process.json").read_text(encoding="utf-8"))
        self.source_freeze = json.loads(self.BASE_PROOF.read_text(encoding="utf-8"))["source_freeze"]
        self.summary_base = json.loads(self.BASE_SUMMARY.read_text(encoding="utf-8"))
        self.schedule_source_hashes = self.source_freeze["schedule_source_sha256"]

        self.patches = [
            patch.object(analyzer, "RAW_ROOT", self.raw_root),
            patch.object(analyzer, "ADJUDICATION_ROOT", self.adjudication_root),
            patch.object(analyzer, "REPAIR_PLAN_PATH", self.plan_path),
            patch.object(analyzer, "ADJUDICATOR_METHOD_PATH", self.method_path),
        ]
        for patcher in self.patches:
            patcher.start()

        self.manifest = {
            "_manifest_path": str(evidence_root / "manifest.json"),
            "adjudicator_method_sha256": self.method_sha256,
            "runs": [],
        }
        self.records: dict[str, dict[str, Any]] = {}
        for label in LABELS:
            self._build_run(label)

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def tearDown(self) -> None:
        for patcher in reversed(self.patches):
            patcher.stop()
        self.temp.cleanup()

    def _controls(self, label: str) -> dict[str, Any]:
        row_owner = label not in {"A", "B", "C"}
        return {
            "gfsdp_split_mtp_ffn": row_owner,
            "gfsdp_split_layer0_ffn": row_owner,
            "gfsdp_row_owner_embeddings": row_owner,
            "gfsdp_mtp_transition_prefetch": label in {"S", "SE", "SE-fused"},
            "gfsdp_row_owner_embedding_optimizations": False,
            "gfsdp_row_owner_embedding_optimization_stages": (
                list(STAGE_TUPLES[label]) if label in {"SE", "SE-fused"} else None
            ),
        }

    def _build_run(self, label: str) -> None:
        run_id = analyzer._run_id_from_repair_plan(self.plan, label)
        raw_path = self.raw_root / run_id
        raw_path.mkdir(parents=True)
        output_dir = self.adjudication_root / label
        output_dir.mkdir(parents=True)
        summary_path = output_dir / "summary.json"
        adjudication_path = output_dir / "adjudication.json"
        is_pass = label in {"SE", "SE-fused"}
        raw_status = "PASS" if is_pass else "FAIL"
        raw_exit_code = 0 if is_pass else 1
        controls = self._controls(label)
        stages = list(STAGE_TUPLES[label])
        module_count = analyzer.STAGE_MODULE_COUNTS[label]

        metadata = copy.deepcopy(self.base_metadata)
        metadata["run_id"] = run_id
        metadata["requested_parallelism_features"] = {
            key: controls[key]
            for key in (
                "gfsdp_split_mtp_ffn",
                "gfsdp_split_layer0_ffn",
                "gfsdp_row_owner_embeddings",
            )
        }
        metadata["num_prefetch_batches"] = 32
        metadata["gfsdp_reuse_bf16_expert_weights"] = True
        argv = metadata.get("argv")
        if isinstance(argv, list):
            for index, token in enumerate(argv[:-1]):
                if token == "--variant":
                    argv[index + 1] = label
                elif token == "--run-id":
                    argv[index + 1] = run_id

        schedule_bridge = self._read_json(self.BASE_RAW / "schedule-embedding-bridge.json")
        schedule_bridge.update({
            "variant": label,
            "attempt": run_id,
            "controls": controls,
            "controls_sha256": analyzer._json_sha256(controls),
            "variant_sha256": hashlib.sha256(label.encode("utf-8")).hexdigest(),
            "source_sha256": self.schedule_source_hashes,
        })
        flash_bridge = self._read_json(self.BASE_RAW / "flash-qla-bridge.json")
        flash_bridge["run_id"] = run_id

        for relative in ("config/experiment.yaml",):
            destination = raw_path / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((self.BASE_RAW / relative).read_bytes())

        resolved_path = raw_path / "native/config/resolved_config.yaml"
        if is_pass:
            resolved_lines = ["parallelism:"]
            resolved_lines.append("  gfsdp_row_owner_embedding_optimization_stages:")
            resolved_lines.extend(f"    - {stage}" for stage in stages)
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            resolved_path.write_text("\n".join(resolved_lines) + "\n", encoding="utf-8")
        else:
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            resolved_path.write_bytes((self.BASE_RAW / "native/config/resolved_config.yaml").read_bytes())

        source_final = copy.deepcopy(self.base_source_final)
        config_final = copy.deepcopy(self.base_config_final)
        process = copy.deepcopy(self.base_process)
        failure = None if is_pass else {
            "error": analyzer.NULLABLE_STAGE_FAILURE,
            "finished_at_utc": "2026-11-15T00:00:00+00:00",
            "pid": process.get("pid"),
            "status": "FAIL",
            "timed_out": False,
        }
        summary = copy.deepcopy(self.summary_base)
        summary["run_id"] = run_id
        self._write_json(summary_path, summary)

        raw_receipts = {
            "config-hashes-final.json": config_final,
            "flash-qla-bridge.json": flash_bridge,
            "metadata.json": metadata,
            "process.json": process,
            "schedule-embedding-bridge.json": schedule_bridge,
            "source-final.json": source_final,
        }
        for relative, value in raw_receipts.items():
            self._write_json(raw_path / relative, value)
        (raw_path / "exit-code.txt").write_text(f"{raw_exit_code}\n", encoding="utf-8")
        (raw_path / "status.txt").write_text(f"{raw_status}\n", encoding="utf-8")
        if is_pass:
            self._write_json(raw_path / "summary.json", summary)
            self._write_json(raw_path / "completion.json", {
                "status": "PASS",
                "source_unchanged": True,
                "config_unchanged": True,
                "resolved_config_sha256": analyzer._sha256(resolved_path),
            })
        else:
            self._write_json(raw_path / "failure.json", failure)

        measurement = raw_path / "measurement"
        measurement.mkdir()
        for rank in range(8):
            sidecar = self._read_json(self.BASE_RAW / f"measurement/rank-{rank:02d}.json")
            sidecar["run_id"] = run_id
            self._write_json(measurement / f"rank-{rank:02d}.json", sidecar)

            stage_receipt = self._read_json(
                self.BASE_RAW / f"measurement/embedding-stage-selection-rank-{rank:02d}.json"
            )
            modules = (
                [{"name": "tok_embeddings", "optimization_stages": stages}]
                if module_count else []
            )
            stage_receipt.update({
                "run_id": run_id,
                "variant": label,
                "source_sha256": self.schedule_source_hashes,
                "expected_module_count": module_count,
                "expected_stages": stages,
                "modules": modules,
            })
            self._write_json(measurement / f"embedding-stage-selection-rank-{rank:02d}.json", stage_receipt)

            bootstrap = self._read_json(
                self.BASE_RAW / f"measurement/flash-qla-bootstrap-rank-{rank:02d}.json"
            )
            bootstrap_schedule = bootstrap["schedule_embedding"]
            bootstrap_schedule.update({
                "variant": label,
                "controls": controls,
                "source_sha256": self.schedule_source_hashes,
            })
            self._write_json(measurement / f"flash-qla-bootstrap-rank-{rank:02d}.json", bootstrap)
            triton = self._read_json(
                self.BASE_RAW / f"measurement/triton-startup-rank-{rank:02d}.json"
            )
            triton["run_id"] = run_id
            self._write_json(measurement / f"triton-startup-rank-{rank:02d}.json", triton)

        raw_hashes = analyzer._current_raw_receipt_hashes(raw_path, raw_status)
        module_names = {
            str(rank): (["tok_embeddings"] if module_count else []) for rank in range(8)
        }
        nullable_exception = None
        known_guard = None
        if not is_pass:
            known_guard = analyzer.NULLABLE_STAGE_GUARD
            nullable_exception = {
                "field": "parallelism.gfsdp_row_owner_embedding_optimization_stages",
                "reason": "typed None default and all-rank native stage receipts were checked",
                "evidence": {
                    "typed_default_none": True,
                    "typed_annotation_nullable": True,
                    "resolved_yaml_omits_selector": True,
                    "explicit_child_null_override": True,
                    "native_typed_decode_is_none": True,
                    "all_rank_module_stage_receipts_valid": True,
                },
            }
        record = {
            "schema_version": 1,
            "status": "ADJUDICATED_PASS",
            "variant": label,
            "run_id": run_id,
            "original_wrapper_status": raw_status,
            "original_exit_code": raw_exit_code,
            "original_failure": failure,
            "original_process": process,
            "known_strict_guard_failure": known_guard,
            "nullable_exception": nullable_exception,
            "actual_native_completion": {
                key: summary[key]
                for key in ("status", "training_steps", "window", "tokens", "memory", "errors")
            },
            "validated_stage_receipts": {
                "rank_count": 8,
                "module_count_per_rank": module_count,
                "module_names_by_rank": module_names,
            },
            "validation_checks": [
                "140/40/100 native cadence and clean child process group",
                "typed selector decode, resolved config, and per-rank module-stage receipts",
            ],
            "raw_attempt": str(raw_path.resolve()),
            "raw_receipt_hashes": raw_hashes,
            "raw_receipts_unchanged": True,
            "source_freeze_sha256": analyzer.SOURCE_FREEZE_SHA256,
            "source_freeze": self.source_freeze,
            "source_freeze_provenance": {"retained_record": str(self.BASE_PROOF)},
            "launch_plan_path": str(self.plan_path.resolve()),
            "launch_plan_sha256": analyzer.REPAIR_PLAN_SHA256,
            "adjudicator_method_sha256": self.method_sha256,
            "authenticated_runtime_files": analyzer._expected_authenticated_runtime_files(self.source_freeze),
            "parser_summary_path": str(summary_path.resolve()),
            "parser_summary_sha256": analyzer._sha256(summary_path),
            "adjudication_path": str(adjudication_path.resolve()),
        }
        self._write_json(adjudication_path, record)
        self.records[label] = record
        self.manifest["runs"].append({
            "label": label,
            "path": str(raw_path.resolve()),
            "adjudication_path": str(adjudication_path.resolve()),
        })

    def _rewrite_record(self, label: str, *, refresh_raw_hashes: bool = False) -> None:
        record = self.records[label]
        if refresh_raw_hashes:
            raw_path = self.raw_root / record["run_id"]
            record["raw_receipt_hashes"] = analyzer._current_raw_receipt_hashes(
                raw_path, record["original_wrapper_status"]
            )
        path = Path(record["adjudication_path"])
        self._write_json(path, record)

    def _analyze(self) -> dict[str, Any]:
        return analyzer.analyze_campaign(self.manifest)

    def test_accepts_complete_uniform_seven_run_proof_set(self) -> None:
        result = self._analyze()
        self.assertEqual(result["campaign_order"], list(LABELS))
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["runs"]["A"]["acceptance"]["mode"], "adjudicated_nullable_guard_fail")
        self.assertEqual(result["runs"]["A"]["acceptance"]["raw_wrapper_status"], "FAIL")
        self.assertEqual(result["runs"]["A"]["acceptance"]["raw_wrapper_exit_code"], "1")
        self.assertEqual(result["runs"]["SE"]["acceptance"]["mode"], "adjudicated_raw_pass")
        self.assertEqual(result["runs"]["SE"]["acceptance"]["raw_wrapper_status"], "PASS")
        self.assertEqual(result["runs"]["SE"]["acceptance"]["raw_wrapper_exit_code"], "0")
        self.assertEqual(
            Path(result["runs"]["A"]["acceptance"]["adjudication_proof"]["adjudication_path"]).parent.name,
            "A",
        )

    def test_requires_external_proof_for_every_final_row(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["runs"][3].pop("adjudication_path")
        with self.assertRaisesRegex(ValueError, "every final timing row requires"):
            analyzer.analyze_campaign(manifest)

    def test_rejects_wrong_run_path_and_variant(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["runs"][2]["path"] = manifest["runs"][1]["path"]
        with self.assertRaisesRegex(ValueError, "plan-pinned attempt"):
            analyzer.analyze_campaign(manifest)

        self.records["D0"]["variant"] = "S"
        self._rewrite_record("D0")
        with self.assertRaisesRegex(ValueError, "variant differs"):
            self._analyze()

    def test_rejects_nonempty_parser_errors(self) -> None:
        record = self.records["B"]
        summary_path = Path(record["parser_summary_path"])
        summary = self._read_json(summary_path)
        summary["errors"] = ["synthetic parser error"]
        record["actual_native_completion"]["errors"] = summary["errors"]
        self._write_json(summary_path, summary)
        record["parser_summary_sha256"] = analyzer._sha256(summary_path)
        self._rewrite_record("B")
        with self.assertRaisesRegex(ValueError, "parser summary errors must use the canonical empty-list"):
            self._analyze()

    def test_rejects_stage_source_map_mismatch(self) -> None:
        record = self.records["B"]
        raw_path = self.raw_root / record["run_id"]
        stage_path = raw_path / "measurement/embedding-stage-selection-rank-00.json"
        stage = self._read_json(stage_path)
        stage["source_sha256"]["runs/schedule-embedding/run.py"] = "f" * 64
        self._write_json(stage_path, stage)
        self._rewrite_record("B", refresh_raw_hashes=True)
        with self.assertRaisesRegex(ValueError, "source_sha256 differs from frozen schedule sources"):
            self._analyze()

    def test_rejects_stage_module_count_mismatch(self) -> None:
        record = self.records["B"]
        raw_path = self.raw_root / record["run_id"]
        stage_path = raw_path / "measurement/embedding-stage-selection-rank-00.json"
        stage = self._read_json(stage_path)
        stage["expected_module_count"] = 1
        stage["modules"] = [{"name": "tok_embeddings", "optimization_stages": []}]
        self._write_json(stage_path, stage)
        self._rewrite_record("B", refresh_raw_hashes=True)
        with self.assertRaisesRegex(ValueError, "expected_module_count must be 0"):
            self._analyze()

    def test_rejects_nullable_exception_on_raw_pass_and_wrong_status(self) -> None:
        self.records["SE"]["nullable_exception"] = {"field": "parallelism.fake"}
        self._rewrite_record("SE")
        with self.assertRaisesRegex(ValueError, "raw PASS proof must not claim a nullable-selector exception"):
            self._analyze()

        self.records["SE"]["nullable_exception"] = None
        self.records["SE"]["original_wrapper_status"] = "FAIL"
        self._rewrite_record("SE")
        with self.assertRaisesRegex(ValueError, "original wrapper status differs"):
            self._analyze()

    def test_rejects_raw_hash_tampering(self) -> None:
        record = self.records["C"]
        config_path = self.raw_root / record["run_id"] / "config/experiment.yaml"
        config_path.write_text("tampered config\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "raw receipt hashes differ"):
            self._analyze()

    def test_rejects_raw_and_external_summary_mismatch(self) -> None:
        record = self.records["SE-fused"]
        raw_summary = self.raw_root / record["run_id"] / "summary.json"
        summary = self._read_json(raw_summary)
        summary["host"] = "tampered raw summary host"
        self._write_json(raw_summary, summary)
        self._rewrite_record("SE-fused", refresh_raw_hashes=True)
        with self.assertRaisesRegex(ValueError, "raw PASS parser summary differs"):
            self._analyze()


if __name__ == "__main__":
    unittest.main()
