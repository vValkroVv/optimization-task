import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import pytest
from node_runner.src.subprocess_manager import SubprocessManager

pytestmark = pytest.mark.e2e

SKIP_NODE_RUN_ENV = "PAM_LMMS_EVAL_E2E_SKIP_NODE_RUN"
UPDATE_GOLDENS_ENV = "PAM_LMMS_EVAL_E2E_UPDATE_GOLDENS"
MODEL_PATH_ENV = "PAM_LMMS_EVAL_E2E_MODEL_PATH"
CONFIG_PATH_ENV = "PAM_LMMS_EVAL_E2E_CONFIG"
GOLDEN_ROOT_ENV = "PAM_LMMS_EVAL_E2E_GOLDEN_ROOT"
METRIC_TOLERANCE_ENV = "PAM_LMMS_EVAL_E2E_METRIC_TOLERANCE"
SAMPLE_SPECS_ENV = "PAM_LMMS_EVAL_E2E_SAMPLE_SPECS"

DEFAULT_MODEL_PATH = "/home/jovyan/models/10b_vision_test_no_delete"
DEFAULT_GOLDEN_MODEL_DIRNAME = "10b_vision"
DEFAULT_SAMPLE_SPECS = (
    "ai2d_smoke:ai2d,"
    "hallucination_group_smoke:pope,"
    "mixed_overlap_smoke:ai2d,"
    "mixed_overlap_smoke:chartqa_giga,"
    "mixed_overlap_smoke:pope,"
    "sequential_mixed_smoke:docvqa_val,"
    "sequential_mixed_smoke:ocrbench_v2,"
    "sequential_mixed_smoke:pope"
)
SAMPLE_FIELDS = (
    "doc_id",
    "target",
    "filtered_resps",
    "exact_match",
    "accuracy",
    "anls",
    "full_accuracy",
    "parsed_accuracy",
    "not_parsed",
    "overall",
    "relaxed_overall",
    "input",
)


def _enabled(name: str) -> bool:
    return os.getenv(name, "").lower() in {"1", "true", "yes"}


def _read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_metrics_payload(path: Path) -> tuple[str, dict[str, float]]:
    payload = _read_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("metrics"), dict):
        raise AssertionError(f"Invalid metrics payload: {path}")

    if payload.get("step") is not None:
        step = str(payload["step"])
        raw_metrics = payload["metrics"]
    else:
        step_items = list(payload["metrics"].items())
        if len(step_items) != 1 or not isinstance(step_items[0][1], dict):
            raise AssertionError(f"Expected exactly one metrics step in {path}")
        step, raw_metrics = step_items[0]

    metrics = {}
    for metric_name, metric_value in raw_metrics.items():
        if isinstance(metric_value, dict):
            metrics[metric_name] = metric_value["value"]
            metrics.update(metric_value["submetrics"])
        else:
            metrics[metric_name] = metric_value

    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, bool) or not isinstance(metric_value, (int, float)):
            raise AssertionError(
                f"Metric {metric_name!r} is not numeric in {path}: {metric_value!r}"
            )
        metric_value = float(metric_value)
        if not math.isfinite(metric_value):
            raise AssertionError(
                f"Metric {metric_name!r} is not finite in {path}: {metric_value!r}"
            )
        metrics[metric_name] = metric_value
    return str(step), metrics


def _relative_diff(actual: float, expected: float) -> float:
    diff = abs(actual - expected)
    if expected == 0:
        return diff
    return diff / abs(expected)


def _assert_metrics_match(
    actual_metrics_file: Path,
    golden_file: Path,
    tolerance: float,
) -> None:
    actual_step, actual = _load_metrics_payload(actual_metrics_file)
    golden_step, expected = _load_metrics_payload(golden_file)
    if golden_step != actual_step:
        raise AssertionError(
            f"Step mismatch: actual={actual_step}, golden={golden_step}"
        )

    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    mismatched = []
    for metric_name, expected_value in sorted(expected.items()):
        if metric_name not in actual:
            continue
        actual_value = actual[metric_name]
        diff = _relative_diff(actual_value, expected_value)
        if diff > tolerance:
            mismatched.append(
                f"{metric_name}: expected={expected_value}, actual={actual_value}, "
                f"relative_diff={diff:.6f}, tolerance={tolerance:.6f}"
            )

    if missing or extra or mismatched:
        parts = [
            f"Metrics mismatch:\n  actual={actual_metrics_file}\n  golden={golden_file}"
        ]
        if missing:
            parts.append("Missing metrics:\n  " + "\n  ".join(missing))
        if extra:
            parts.append("Extra metrics:\n  " + "\n  ".join(extra))
        if mismatched:
            parts.append("Metrics out of tolerance:\n  " + "\n  ".join(mismatched))
        raise AssertionError("\n".join(parts))


def _latest_metrics_file(metrics_dir: Path) -> Path:
    files = sorted(metrics_dir.glob("metrics_step_*.json"))
    if not files:
        raise AssertionError(f"No metrics_step_*.json found under {metrics_dir}")
    return max(files, key=lambda path: path.stat().st_mtime)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise AssertionError(
                    f"JSONL row {line_number} is not an object: {path}"
                )
            rows.append(row)
    return rows


def _project_sample(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row[field] for field in SAMPLE_FIELDS if field in row}


def _parse_sample_specs(raw_specs: str) -> list[tuple[str, str]]:
    specs = []
    for raw_spec in raw_specs.split(","):
        raw_spec = raw_spec.strip()
        if not raw_spec:
            continue
        if ":" not in raw_spec:
            raise AssertionError(
                f"Invalid sample spec {raw_spec!r}. Expected '<run>:<task>'."
            )
        run_name, task_name = raw_spec.split(":", 1)
        specs.append((run_name.strip(), task_name.strip()))
    return specs


def _find_run_sample_file(metrics_dir: Path, run_name: str, task: str) -> Path:
    run_dir = metrics_dir / run_name
    matches = list(run_dir.rglob(f"samples_{task}.jsonl"))
    matches.extend(run_dir.rglob(f"*_samples_{task}.jsonl"))
    matches = [
        path
        for path in set(matches)
        if path.is_file() and "submissions" not in path.relative_to(run_dir).parts
    ]
    if not matches:
        raise AssertionError(f"No samples file for run={run_name!r}, task={task!r}")
    return max(matches, key=lambda path: path.stat().st_mtime)


def _sample_golden_path(golden_root: Path, run_name: str, task: str) -> Path:
    return golden_root / "samples" / f"{run_name}__{task}.json"


def _assert_samples_match(actual_file: Path, golden_file: Path) -> None:
    actual = [_project_sample(row) for row in _load_jsonl(actual_file)]
    expected = _read_json(golden_file)
    if actual != expected:
        if len(actual) != len(expected):
            raise AssertionError(
                f"Sample count mismatch:\n  actual={actual_file}\n"
                f"  golden={golden_file}\n  actual_count={len(actual)}\n"
                f"  expected_count={len(expected)}"
            )
        for index, (actual_row, expected_row) in enumerate(zip(actual, expected)):
            if actual_row != expected_row:
                raise AssertionError(
                    f"Sample mismatch at row {index}:\n  actual={actual_file}\n"
                    f"  golden={golden_file}\n  actual_row={actual_row!r}\n"
                    f"  expected_row={expected_row!r}"
                )


def _write_goldens(
    metrics_dir: Path,
    golden_root: Path,
    sample_specs: list[tuple[str, str]],
) -> None:
    actual_metrics_file = _latest_metrics_file(metrics_dir)
    _load_metrics_payload(actual_metrics_file)
    _write_json(golden_root / actual_metrics_file.name, _read_json(actual_metrics_file))

    for run_name, task in sample_specs:
        actual_sample_file = _find_run_sample_file(metrics_dir, run_name, task)
        projected_rows = [
            _project_sample(row) for row in _load_jsonl(actual_sample_file)
        ]
        _write_json(_sample_golden_path(golden_root, run_name, task), projected_rows)


class TestE2EVision:
    @property
    def pam_root(self) -> Path:
        return Path(__file__).parent.parent.parent

    def _default_config_path(self) -> Path:
        return (
            self.pam_root
            / "sber_tests"
            / "test_data"
            / "configs"
            / "10b_test_vision.yaml"
        )

    def _default_golden_root(self) -> Path:
        return (
            self.pam_root
            / "sber_tests"
            / "test_data"
            / "goldens"
            / DEFAULT_GOLDEN_MODEL_DIRNAME
            / "metrics"
            / "lmms_eval"
        )

    def test_lmms_eval_golden_smoke(self, tmp_path: Path):
        model_path = Path(os.getenv(MODEL_PATH_ENV, DEFAULT_MODEL_PATH))
        config_path = Path(os.getenv(CONFIG_PATH_ENV, self._default_config_path()))
        golden_root = Path(os.getenv(GOLDEN_ROOT_ENV, self._default_golden_root()))
        sample_specs = _parse_sample_specs(
            os.getenv(SAMPLE_SPECS_ENV, DEFAULT_SAMPLE_SPECS)
        )
        tolerance = float(os.getenv(METRIC_TOLERANCE_ENV, "0.0"))

        artifacts_dir = model_path / "artifacts"
        lmms_metrics_dir = artifacts_dir / "metrics" / "lmms_eval"
        if not _enabled(SKIP_NODE_RUN_ENV):
            if lmms_metrics_dir.exists():
                shutil.rmtree(lmms_metrics_dir)
            cmd = [
                "python3",
                "node_runner.py",
                "--checkpoint",
                str(model_path),
                "--config",
                str(config_path),
                "--mode",
                "autometrics",
            ]
            sm = SubprocessManager(
                cmd,
                log_file=tmp_path / "node_runner_lmms_eval.log",
                cwd=self.pam_root / "node_runner",
            )
            sm.start()
            sm.wait_for_completion()
            sm.raise_for_exit_code()

        if _enabled(UPDATE_GOLDENS_ENV):
            _write_goldens(lmms_metrics_dir, golden_root, sample_specs)

        actual_metrics_file = _latest_metrics_file(lmms_metrics_dir)
        golden_metrics_file = _latest_metrics_file(golden_root)
        _assert_metrics_match(
            actual_metrics_file=actual_metrics_file,
            golden_file=golden_metrics_file,
            tolerance=tolerance,
        )

        checked_samples = []
        for run_name, task in sample_specs:
            actual_sample_file = _find_run_sample_file(lmms_metrics_dir, run_name, task)
            _assert_samples_match(
                actual_file=actual_sample_file,
                golden_file=_sample_golden_path(golden_root, run_name, task),
            )
            checked_samples.append(f"{run_name}:{task}")

        actual_step, actual_metrics = _load_metrics_payload(actual_metrics_file)
        print(
            "PAM vision e2e golden check passed:\n"
            f"  metrics_step={actual_step}\n"
            f"  metric_keys={', '.join(sorted(actual_metrics))}\n"
            f"  actual_metrics={actual_metrics_file}\n"
            f"  golden_metrics={golden_metrics_file}\n"
            f"  samples={', '.join(checked_samples)}"
        )
