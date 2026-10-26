from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).absolute().parent.parent))

from contrib.job_submitter.job_submitter.jobs.any_script import AnyScriptJob

LLM_F_ROOT = Path(__file__).absolute().parent.parent


USER_LOGIN = "<CLOUD_USER_LOGIN>"  # required: cloud job owner/login.
WORKSPACE = (
    "<PATH_TO_CLOUD_WORKSPACE_DIR>"  # required: parent dir containing llm-foundry.
)
LLMF_DIR = "llm-foundry"  # required: repo directory name under WORKSPACE.
STABILITY_DIR = f"{WORKSPACE}/{LLMF_DIR}/sber_scripts/analytics/stability"
CLOUD_QUEUE_NAME = "pretrain-ml"  # required: cloud queue, e.g. "pretrain-ml".
CLOUD_N_GPUS = 1  # required by the submitter; TB tools themselves are CPU-heavy.
EXPERIMENT_NAME = "<EXPERIMENT_NAME>"  # required: used only for job and output names.

# Parent directory with one subdirectory per TensorBoard run / hyperparameter setup.
TB_ROOT = "<PATH_TO_TENSORBOARD_ROOT_DIR>"

# Explicit run directory names under TB_ROOT. Leave empty to select by TB_RUNS_PATTERN.
TB_RUNS: list[str] = [
    "<TB_RUN_DIR_NAME_A>",
    "<TB_RUN_DIR_NAME_B>",
    "<TB_RUN_DIR_NAME_C>",  # optional; remove when comparing only two runs.
]

# Alternative selection mode. Used only when TB_RUNS is empty.
TB_RUNS_PATTERN: str | None = None

# Compare needs 2+ runs. If empty, the filtered TB_RUNS list is reused.
TB_COMPARE_RUNS: list[str] = []
TB_COMPARE_RUNS_PATTERN: str | None = None

TB_OUTPUT_DIR = f"{STABILITY_DIR}/output/tensorboard/{EXPERIMENT_NAME}"

RUN_SINGLE_REPORTS = True
RUN_SINGLE_HEATMAPS = True
RUN_SINGLE_STORIES = True
RUN_MULTI_TOOLS = True
RUN_COMPARE = True
RUN_MULTI_STORY = True

INSTABILITY_WORKERS = 8
INSTABILITY_HEATMAP_TAGS = [
    "moe_load_balance",
    "moe_maxvio",
    "moe_expert_utilization",
]
INSTABILITY_HEATMAP_THRESHOLD = 4
INSTABILITY_COMPARE_N_STEPS = 8
COMPARE_STORY_ROWS = 12
TB_EVENT_GLOB = "events.out.tfevents*"
TB_SKIP_EXISTING_OUTPUTS = True


def _job() -> AnyScriptJob:
    return AnyScriptJob(
        user_login=USER_LOGIN,
        exp_name=f"{EXPERIMENT_NAME}_tb_stability_compare",
        script=f"python3 {STABILITY_DIR}/submit_stability_tb_tools_cloud_example.py --run-tools",
        extra_env={
            "PYTHONPATH": STABILITY_DIR,
            "PYTHONUNBUFFERED": "1",
            "MPLBACKEND": "Agg",
        },
        n_gpus=CLOUD_N_GPUS,
        n_workers=1,
        queue_name=CLOUD_QUEUE_NAME,
        priority_class="medium",
        script_type="binary",
        max_retry=None,
    )


def _run(title: str, comment: str, argv: list[str]) -> None:
    print(f"\n===== {title} =====", flush=True)
    print(f"# {comment}", flush=True)
    print(" ".join(argv), flush=True)
    subprocess.run(argv, check=True)


def _write_json(path: str, data: dict) -> None:
    p = Path(path)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"[story config] wrote {p}", flush=True)


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "run"


def _runs_args(runs: list[str], pattern: str | None) -> list[str]:
    if runs:
        return ["--runs", *runs]
    if pattern:
        return ["--runs-pattern", pattern]
    return []


def _run_has_events(run_name: str) -> bool:
    run_dir = Path(_run_dir(run_name))
    if not run_dir.exists():
        print(f"[runs discovery] skip missing run dir: {run_dir}", flush=True)
        return False
    try:
        next(run_dir.rglob(TB_EVENT_GLOB))
    except StopIteration:
        print(f"[runs discovery] skip run without event files: {run_dir}", flush=True)
        return False
    return True


def _filter_runs_with_events(runs: list[str]) -> list[str]:
    return [run for run in runs if _run_has_events(run)]


def _selected_runs() -> list[str]:
    if TB_RUNS:
        return _filter_runs_with_events(TB_RUNS)
    if not TB_RUNS_PATTERN:
        return []

    root = Path(TB_ROOT)
    if not root.exists():
        print(f"[runs discovery] TB_ROOT does not exist yet: {TB_ROOT}", flush=True)
        return []

    runs = sorted(
        child.name
        for child in root.iterdir()
        if child.is_dir() and fnmatch.fnmatch(child.name, TB_RUNS_PATTERN)
    )
    return _filter_runs_with_events(runs)


def _compare_runs_args(runs: list[str]) -> list[str]:
    if TB_COMPARE_RUNS:
        return _runs_args(_filter_runs_with_events(TB_COMPARE_RUNS), None)
    if len(runs) >= 2:
        return _runs_args(runs, None)
    return []


def _run_dir(run_name: str) -> str:
    return f"{TB_ROOT}/{run_name}"


def _report_dir(run_name: str) -> str:
    return f"{TB_OUTPUT_DIR}/report_{_slug(run_name)}"


def _heatmap_dir(run_name: str) -> str:
    return f"{TB_OUTPUT_DIR}/heatmap_{_slug(run_name)}"


def _story_config_path(run_name: str) -> str:
    return f"{TB_OUTPUT_DIR}/story_{_slug(run_name)}_config.json"


def _story_html_path(run_name: str) -> str:
    return f"{TB_OUTPUT_DIR}/story_{_slug(run_name)}.html"


def _single_story_config(run_name: str) -> dict:
    return {
        "title": f"instability story - {run_name}",
        "subtitle": f"TensorBoard run: {_run_dir(run_name)}",
        "summary": (
            "Generated by submit_stability_tb_tools.py after instability.report. "
            "Read report data/summary.md and CSV files for the diagnostics; "
            "this HTML is a figure gallery."
        ),
        "overview_note": "Single TensorBoard run report.",
        "overview_image": "01_overview.png",
        "metric_sections": [
            {
                "metric": "max_vio per layer",
                "direction_note": "higher = more severe routing imbalance",
                "prose": "Depth-time max_vio heatmap from instability.report.",
                "image": "02_max_vio_heatmap.png",
            },
            {
                "metric": "MoE health",
                "direction_note": "router and expert-utilization monitors",
                "prose": "Curated MoE health panel.",
                "image": "04_moe_health.png",
            },
            {
                "metric": "activation depth",
                "direction_note": "large late-depth values can indicate saturation",
                "prose": "Activation depth panel.",
                "image": "05_activations_depth.png",
            },
        ],
    }


def _float_or_none(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        x = float(value)
    except ValueError:
        return None
    return x if math.isfinite(x) else None


def _comparison_diffs(compare_dir: str, runs: list[str]) -> list[dict[str, Any]]:
    csv_path = Path(compare_dir) / "run_comparison_diffs.csv"
    if not csv_path.exists():
        return []
    out: list[dict[str, Any]] = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            values = {
                run: value
                for run in runs
                if (value := _float_or_none(row.get(f"val__{run}"))) is not None
            }
            out.append(
                {
                    "metric": row.get("tag", ""),
                    "layer": row.get("layer", ""),
                    "group": row.get("group", ""),
                    "values": values,
                    "divergence": _float_or_none(row.get("divergence")) or 0.0,
                    "note": (
                        f"highest={row.get('highest_run', '')}, "
                        f"lowest={row.get('lowest_run', '')}, "
                        f"peak_step={row.get('peak_step', '')}"
                    ),
                }
            )
            if len(out) >= COMPARE_STORY_ROWS:
                break
    return out


def _multi_story_config(runs: list[str], compare_dir: str) -> dict:
    return {
        "title": f"instability comparison story - {EXPERIMENT_NAME}",
        "subtitle": f"TensorBoard root: {TB_ROOT}",
        "summary": (
            "Generated after instability.compare plus per-run instability.report. "
            "Use compare/run_comparison_diffs.csv and each report_<run>/data/summary.md "
            "for the actual diagnostics; this HTML is a combined gallery."
        ),
        "comparison": {
            "runs": runs,
            "intro": "Two learning-rate setups compared at shared TensorBoard steps.",
            "image": "compare/figures/01_divergence_by_group.png",
            "diffs": _comparison_diffs(compare_dir, runs),
        },
        "overview_note": "Cross-run max_vio overlay from instability.compare.",
        "overview_image": "compare/figures/00_max_vio_overlay.png",
        "findings": [
            {
                "title": run,
                "verdict": "inconclusive",
                "prose": (
                    "Per-run report generated; read data/summary.md for this run "
                    "before committing to a verdict."
                ),
                "images": [
                    f"report_{_slug(run)}/figures/01_overview.png",
                    f"report_{_slug(run)}/figures/02_max_vio_heatmap.png",
                    f"report_{_slug(run)}/figures/04_moe_health.png",
                ],
            }
            for run in runs
        ],
        "next_steps": [
            "Read compare/run_comparison_group_summary.csv first.",
            "Read compare/run_comparison_diffs.csv for concrete tag/layer divergences.",
            "Read report_<run>/data/summary.md for each run before interpreting causes.",
        ],
    }


def _run_single_tools(run_name: str) -> None:
    run_dir = _run_dir(run_name)
    report_dir = _report_dir(run_name)
    heatmap_dir = _heatmap_dir(run_name)

    if RUN_SINGLE_REPORTS:
        report_done = Path(report_dir, "data", "signals.json").exists()
        if TB_SKIP_EXISTING_OUTPUTS and report_done:
            print(
                f"[skip existing] instability report {run_name}: {report_dir}",
                flush=True,
            )
        else:
            # instability.report: single-run report. It reads all events.out.tfevents.*
            # files inside run_dir and writes figures/, data/, atlas/, summary.md.
            _run(
                f"instability report {run_name}",
                "Single TensorBoard run report over all event files in this run directory.",
                [
                    "python3",
                    "-m",
                    "instability.report",
                    "--tb-dir",
                    run_dir,
                    "--out-dir",
                    report_dir,
                    "--workers",
                    str(INSTABILITY_WORKERS),
                    "--name",
                    run_name,
                ],
            )

    if RUN_SINGLE_HEATMAPS:
        heatmaps_done = all(
            Path(heatmap_dir, f"{tag}.png").exists() for tag in INSTABILITY_HEATMAP_TAGS
        )
        if TB_SKIP_EXISTING_OUTPUTS and heatmaps_done:
            print(
                f"[skip existing] instability heatmap {run_name}: {heatmap_dir}",
                flush=True,
            )
        else:
            # instability.heatmap: focused layer x step heatmaps for selected tag families.
            _run(
                f"instability heatmap {run_name}",
                "Focused layer-by-step heatmaps for selected TensorBoard tag families.",
                [
                    "python3",
                    "-m",
                    "instability.heatmap",
                    "--tb-dir",
                    run_dir,
                    "--out-dir",
                    heatmap_dir,
                    "--tags",
                    *INSTABILITY_HEATMAP_TAGS,
                    "--threshold",
                    str(INSTABILITY_HEATMAP_THRESHOLD),
                ],
            )

    if RUN_SINGLE_STORIES:
        story_html = _story_html_path(run_name)
        if TB_SKIP_EXISTING_OUTPUTS and Path(story_html).exists():
            print(
                f"[skip existing] instability story {run_name}: {story_html}",
                flush=True,
            )
        else:
            # instability.story: HTML gallery over this run's report figures.
            story_config = _story_config_path(run_name)
            _write_json(story_config, _single_story_config(run_name))
            _run(
                f"instability story {run_name}",
                "Shareable HTML gallery over the single-run instability.report figures.",
                [
                    "python3",
                    "-m",
                    "instability.story",
                    story_config,
                    f"{report_dir}/figures",
                    "--out",
                    story_html,
                    "--figures-url-prefix",
                    f"report_{_slug(run_name)}/figures",
                ],
            )


def _run_multi_tools(runs: list[str]) -> None:
    if not RUN_MULTI_TOOLS:
        return
    runs_args = _runs_args(runs, None)
    multi_dir = f"{TB_OUTPUT_DIR}/multi"

    if RUN_COMPARE:
        compare_args = _compare_runs_args(runs)
        if compare_args:
            # instability.compare: N-run diff. It aligns tags across runs and ranks
            # concrete (tag, layer) divergences at shared steps.
            _run(
                "instability compare",
                "Cross-run diff for selected TensorBoard runs.",
                [
                    "python3",
                    "-m",
                    "instability.compare",
                    "--tb-root",
                    TB_ROOT,
                    "--out-dir",
                    f"{TB_OUTPUT_DIR}/compare",
                    *compare_args,
                    "--n-steps",
                    str(INSTABILITY_COMPARE_N_STEPS),
                    "--max-workers",
                    str(INSTABILITY_WORKERS),
                ],
            )
        else:
            print("\n===== skip instability compare =====", flush=True)
            print(
                "# Set TB_COMPARE_RUNS or TB_COMPARE_RUNS_PATTERN for 2+ runs.",
                flush=True,
            )

    # instability.eda: multi-run inventory and monitor/max_vio correlations.
    _run(
        "instability eda",
        "Multi-run inventory and monitor/max_vio correlation tables.",
        [
            "python3",
            "-m",
            "instability.eda",
            "--tb-root",
            TB_ROOT,
            "--out-dir",
            multi_dir,
            *runs_args,
        ],
    )

    # instability.temporal: lead/lag analysis across selected runs.
    _run(
        "instability temporal",
        "Lead/lag views: predictive tests, cross-correlation, AR-X and event study.",
        [
            "python3",
            "-m",
            "instability.temporal",
            "--tb-root",
            TB_ROOT,
            "--out-dir",
            multi_dir,
            *runs_args,
        ],
    )

    # instability.projections: projection-level grouping of EDA/temporal signals.
    _run(
        "instability projections",
        "Per-projection grouping and plots for the selected TensorBoard runs.",
        [
            "python3",
            "-m",
            "instability.projections",
            "--tb-root",
            TB_ROOT,
            "--out-dir",
            multi_dir,
            *runs_args,
        ],
    )

    if RUN_MULTI_STORY and RUN_COMPARE and len(runs) >= 2:
        # instability.story: one combined HTML whose gallery covers compare figures
        # and all per-run report figures under TB_OUTPUT_DIR.
        story_config = f"{TB_OUTPUT_DIR}/story_compare_config.json"
        _write_json(story_config, _multi_story_config(runs, f"{TB_OUTPUT_DIR}/compare"))
        _run(
            "instability story compare",
            "Combined comparison HTML over compare figures and per-run report figures.",
            [
                "python3",
                "-m",
                "instability.story",
                story_config,
                TB_OUTPUT_DIR,
                "--out",
                f"{TB_OUTPUT_DIR}/story_compare.html",
                "--figures-url-prefix",
                ".",
            ],
        )


def run_tools() -> None:
    runs = _selected_runs()
    if RUN_SINGLE_REPORTS or RUN_SINGLE_HEATMAPS or RUN_SINGLE_STORIES:
        for run_name in runs:
            _run_single_tools(run_name)

    if runs:
        _run_multi_tools(runs)
    else:
        print("\n===== skip instability multi-run tools =====", flush=True)
        print("# No selected TensorBoard run has event files.", flush=True)

    print(f"\n===== done =====\nOutputs: {TB_OUTPUT_DIR}", flush=True)


def submit_all() -> None:
    print("tb:", _job().submit())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tools", action="store_true")
    args = parser.parse_args()

    if args.run_tools:
        run_tools()
    else:
        submit_all()


if __name__ == "__main__":
    main()
