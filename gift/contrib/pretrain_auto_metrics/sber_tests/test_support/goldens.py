import json
import math
from collections import Counter
from pathlib import Path


def _adaptive_tolerance(
    base_tolerance: float, expected: float, growth_coeff: float
) -> float:
    normalized_expected = min(abs(expected), 1.0)
    return base_tolerance * (1.0 + growth_coeff * (1.0 - normalized_expected))


def _load_step_metrics(metrics_file: Path) -> dict[str, float]:
    if not metrics_file.exists():
        raise AssertionError(f"Metrics file not found: {metrics_file}")
    if not metrics_file.is_file():
        raise AssertionError(f"Metrics path is not a file: {metrics_file}")

    with open(metrics_file, encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise AssertionError(f"Invalid metrics payload type in {metrics_file}")

    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise AssertionError(f"Missing or invalid 'metrics' in {metrics_file}")

    step_metrics = metrics.get("0")
    if not isinstance(step_metrics, dict):
        raise AssertionError(f"Missing or invalid 'metrics[\"0\"]' in {metrics_file}")

    normalized: dict[str, float] = {}
    for metric_name, metric_value in step_metrics.items():
        if isinstance(metric_value, bool) or not isinstance(metric_value, (int, float)):
            raise AssertionError(
                f"Metric '{metric_name}' in {metrics_file} is not numeric: {metric_value!r}"
            )
        metric_value = float(metric_value)
        if not math.isfinite(metric_value):
            raise AssertionError(
                f"Metric '{metric_name}' in {metrics_file} is not finite: {metric_value!r}"
            )
        normalized[metric_name] = metric_value

    return normalized


def _relative_diff(actual: float, expected: float) -> float:
    diff = abs(actual - expected)
    if expected == 0:
        return diff
    return diff / abs(expected)


def _assert_metrics_file_matches_golden(
    golden_file: Path,
    actual_file: Path,
    tolerance: float,
    tolerance_growth_coeff: float = 0.0,
) -> None:
    expected_metrics = _load_step_metrics(golden_file)
    actual_metrics = _load_step_metrics(actual_file)

    missing_metrics: list[str] = []
    diff_failures: list[str] = []
    for metric_name, expected_value in sorted(expected_metrics.items()):
        if metric_name not in actual_metrics:
            missing_metrics.append(metric_name)
            continue
        actual_value = actual_metrics[metric_name]
        rel_diff = _relative_diff(actual_value, expected_value)
        effective_tolerance = _adaptive_tolerance(
            base_tolerance=tolerance,
            expected=expected_value,
            growth_coeff=tolerance_growth_coeff,
        )
        if rel_diff > effective_tolerance:
            diff_failures.append(
                (
                    f"{metric_name}: expected={expected_value}, actual={actual_value}, "
                    f"relative_diff={rel_diff:.6f}, tolerance={effective_tolerance:.6f}, "
                    f"base_tolerance={tolerance:.6f}, growth_coeff={tolerance_growth_coeff:.6f}"
                )
            )

    if missing_metrics or diff_failures:
        actual_only_metrics = (
            sorted(actual_metrics.keys() - expected_metrics.keys())
            if missing_metrics
            else []
        )
        parts: list[str] = [
            f"Metrics mismatch for:\n  golden={golden_file}\n  actual={actual_file}"
        ]
        if missing_metrics:
            parts.append("Missing metrics:\n  " + "\n  ".join(missing_metrics))
        if actual_only_metrics:
            parts.append(
                "Actual-only metrics (may indicate renamed metrics):\n  "
                + "\n  ".join(actual_only_metrics)
            )
        if diff_failures:
            parts.append("Metrics out of tolerance:\n  " + "\n  ".join(diff_failures))
        raise AssertionError("\n".join(parts))


REPORTED_ERRORS = 5


def _evaluator_status(log_lines: list[str], evaluator: str) -> str:
    if any(f"[planner] evaluator {evaluator} failed" in line for line in log_lines):
        return "failed"
    if any(f"[planner] done  {evaluator} " in line for line in log_lines):
        return "done"
    if any(f"[planner] start {evaluator} " in line for line in log_lines):
        return "started, never finished"
    return "never started"


def _errors(lines: list[str]) -> list[str]:
    errors = []
    in_traceback = False
    for line in lines:
        if line.startswith("Traceback"):
            in_traceback = True
        elif in_traceback and line and not line[0].isspace():
            errors.append(line.strip())
            in_traceback = False
        elif " | ERROR | " in line:
            errors.append(line.split(" | ERROR | ", 1)[1].strip())
    return errors


def _run_summary(artifacts_path: Path, evaluators: list[Path]) -> str:
    logs = {
        log_file: log_file.read_text(encoding="utf-8", errors="replace").splitlines()
        for log_file in sorted(artifacts_path.glob("logs/**/*.log"))
    }
    log_lines = [line for lines in logs.values() for line in lines]
    rows = [
        f"  {evaluator}: {_evaluator_status(log_lines, evaluator.name)}, "
        + (
            "metrics file written"
            if (artifacts_path / "metrics" / evaluator / "metrics_step_0.json").exists()
            else "NO metrics file"
        )
        for evaluator in evaluators
    ]
    errors = Counter(
        (log_file.relative_to(artifacts_path / "logs"), error[:300])
        for log_file, lines in logs.items()
        for error in _errors(lines)
    )
    return "\n".join(
        [
            f"Evaluators, by {artifacts_path / 'logs'}:",
            *rows,
            "Most frequent errors in the logs:",
            *(
                f"  {log_file}: {count}x {error}"
                for (log_file, error), count in errors.most_common(REPORTED_ERRORS)
            ),
            *([] if errors else ["  none"]),
        ]
    )


def assert_metrics_against_goldens(
    model_path: str,
    golden_root: Path,
    tolerance: float,
    tolerance_growth_coeff: float = 0.0,
) -> None:
    actual_metrics_root = Path(model_path) / "artifacts" / "metrics"
    if not actual_metrics_root.exists():
        raise AssertionError(f"Metrics directory not found: {actual_metrics_root}")
    if not actual_metrics_root.is_dir():
        raise AssertionError(f"Metrics path is not a directory: {actual_metrics_root}")
    if not golden_root.exists() or not golden_root.is_dir():
        raise AssertionError(f"Golden root not found: {golden_root}")

    golden_files = sorted(golden_root.rglob("metrics_step_0.json"))
    if not golden_files:
        raise AssertionError(
            f"No golden files matching 'metrics_step_0.json' under: {golden_root}"
        )

    failures: list[str] = []
    evaluators: list[Path] = []
    for golden_file in golden_files:
        relative_to_golden_root = golden_file.relative_to(golden_root)
        if relative_to_golden_root.parts[0] == "metrics":
            actual_relative_path = Path(*relative_to_golden_root.parts[1:])
        else:
            actual_relative_path = relative_to_golden_root
        actual_file = actual_metrics_root / actual_relative_path
        evaluators.append(actual_relative_path.parent)
        try:
            if not actual_file.exists():
                raise AssertionError(
                    f"{actual_relative_path.parent}: no {actual_file.name} "
                    f"at {actual_file.parent}"
                )
            _assert_metrics_file_matches_golden(
                golden_file=golden_file,
                actual_file=actual_file,
                tolerance=tolerance,
                tolerance_growth_coeff=tolerance_growth_coeff,
            )
        except AssertionError as exc:
            failures.append(str(exc))

    if failures:
        raise AssertionError(
            f"{len(failures)} of {len(golden_files)} golden checks failed\n\n"
            + _run_summary(actual_metrics_root.parent, evaluators)
            + "\n\n"
            + "\n\n".join(failures)
        )
