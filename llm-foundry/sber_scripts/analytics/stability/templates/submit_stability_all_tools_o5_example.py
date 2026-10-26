from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).absolute().parent.parent))

from contrib.job_submitter.job_submitter.jobs.any_script import AnyScriptJob
from contrib.job_submitter.job_submitter.jobs.image_registry import LLMF_BASE_IMAGES

LLM_F_ROOT = Path(__file__).absolute().parent.parent


USER_LOGIN = "<O_USER_LOGIN>"  # required: O-cluster job owner/login.
REGION = "o5"  # required: "o1" or "o5".
WORKSPACE = "<PATH_TO_O_SHARED_WORKSPACE_DIR>"  # required: shared workspace/root dir.

LLMF_DIR = "<PATH_TO_LLM_FOUNDRY_REPO_RELATIVE_TO_WORKSPACE>"
STABILITY_DIR = f"{WORKSPACE}/{LLMF_DIR}/sber_scripts/analytics/stability"
EXPERIMENT_NAME = "<EXPERIMENT_NAME>"  # required: used only for job and output names.
JOB_LOGDIR: str | None = (
    "<PATH_TO_JOB_LOG_DIR>"  # optional; set to None for submitter default.
)
SLURM_LOGDIR: str | None = (
    "<PATH_TO_SLURM_LOG_DIR>"  # optional; set to None for submitter default.
)

checkpoint_run_dir = "<PATH_TO_CHECKPOINT_RUN_DIR>"

# Put one iteration here for the old single-checkpoint behavior, or several
# iterations for cross-checkpoint / trajectory tools. Each iteration must contain
# artifacts/hf_checkpoint/ under checkpoint_run_dir.
CHECKPOINT_ITERS = [
    "<CHECKPOINT_ITER_A>",
    "<CHECKPOINT_ITER_B>",
]
# False means pass --no-skip-existing, so batch really reruns analyze/joint.
WEIGHTS_BATCH_SKIP_EXISTING = False

output_root = f"{STABILITY_DIR}/output/{EXPERIMENT_NAME}"
if len(CHECKPOINT_ITERS) > 1:
    out_dir = f"{output_root}/{'_vs_'.join(CHECKPOINT_ITERS)}"
else:
    out_dir = f"{output_root}/{CHECKPOINT_ITERS[0]}"

# weights report --trajectory needs one role[/proj[/block]] per call. These are
# common model groups; edit/extend after reading weights_report_deltas.txt.
WEIGHTS_TRAJECTORIES = [
    "linear_attn/ba",
    "linear_attn/in_ba",
    "moe_router/weight",
    "moe_expert/gate",
    "moe_expert/up",
    "moe_expert/down",
]


WEIGHTS_REPORT_TOP_N = 12
WEIGHTS_REPORT_ALL_METRICS = False
WEIGHTS_DASHBOARD_MAX_HIST = 20
WEIGHTS_JOINT_K = 4


def _iter_label() -> str:
    if len(CHECKPOINT_ITERS) == 1:
        return CHECKPOINT_ITERS[0]
    return f"{len(CHECKPOINT_ITERS)}ckpts_{CHECKPOINT_ITERS[0]}_{CHECKPOINT_ITERS[-1]}"


def _hf_dir(iter_name: str) -> str:
    return f"{checkpoint_run_dir}/{iter_name}/artifacts/hf_checkpoint"


def _iter_out_dir(iter_name: str) -> str:
    return f"{output_root}/{iter_name}"


def _analyze_json(iter_name: str) -> str:
    return f"{_iter_out_dir(iter_name)}/analyze.json"


def _joint_json(iter_name: str) -> str:
    return f"{_iter_out_dir(iter_name)}/joint.json"


def _weights_report(iter_name: str) -> str:
    return f"{_iter_out_dir(iter_name)}/reports/weights_report.txt"


def _weights_dashboard_dir(iter_name: str) -> str:
    return f"{_iter_out_dir(iter_name)}/dashboard"


def _weights_story_config_path(iter_name: str) -> str:
    return f"{_iter_out_dir(iter_name)}/story_config.json"


def _weights_story_html(iter_name: str) -> str:
    return f"{_iter_out_dir(iter_name)}/story.html"


def _multi_out_dir() -> str:
    return f"{output_root}/multi_{'_'.join(CHECKPOINT_ITERS)}"


def _job() -> AnyScriptJob:
    return AnyScriptJob(
        user_login=USER_LOGIN,
        region=REGION,
        exp_name=f"{EXPERIMENT_NAME}_stability_all_{_iter_label()}",
        script=f"python3 {STABILITY_DIR}/submit_stability_all_tools_o5_example.py --run-tools",
        extra_env={
            "PYTHONPATH": ":".join([str(LLM_F_ROOT), STABILITY_DIR]),
            "PYTHONUNBUFFERED": "1",
            "MPLBACKEND": "Agg",
        },
        n_gpus=8,
        n_workers=1,
        script_type="binary",
        priority_class="medium",
        base_image=LLMF_BASE_IMAGES[REGION],
        job_logdir=JOB_LOGDIR,
        slurm_logdir=SLURM_LOGDIR,
        max_retry=None,
    )


def _weights_story_config(iter_name: str) -> dict:
    return {
        "title": f"weights story - {EXPERIMENT_NAME} {iter_name}",
        "subtitle": f"HF checkpoint: {_hf_dir(iter_name)}",
        "summary": (
            "Generated after weights analyze/joint/report/dashboard. "
            "Use reports/weights_report.txt for the textual diagnostics; "
            "this HTML is a dashboard gallery."
        ),
        "overview_note": "One checkpoint, analyze.json plus joint.json.",
        "overview_image": "00_overview.png",
        "metric_sections": [
            {
                "metric": "base.max_abs",
                "direction_note": "high = large weight outliers",
                "prose": "Per-group overview from weights dashboard.",
                "image": "01_metric_overview/base.max_abs.png",
            },
            {
                "metric": "spectral.stable_rank",
                "direction_note": "low = rank collapse",
                "prose": "Stable-rank overview across tensor families.",
                "image": "01_metric_overview/spectral.stable_rank.png",
            },
            {
                "metric": "rowcol.row.max_over_median",
                "direction_note": "high = row norm concentration",
                "prose": "Row norm concentration overview.",
                "image": "01_metric_overview/rowcol.row.max_over_median.png",
            },
        ],
        "joint": {
            "prose": "MoE gate/up alignment is shown when joint.json has alignment entries.",
            "image": "03_joint_alignment.png",
        },
    }


def _run(title: str, comment: str, argv: list[str]) -> None:
    print(f"\n===== {title} =====", flush=True)
    print(f"# {comment}", flush=True)
    print(" ".join(argv), flush=True)
    subprocess.run(argv, check=True)


def _write_json(path: str, data: dict) -> None:
    p = Path(path)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"[story config] wrote {p}", flush=True)


def _trajectory_out_name(spec: str) -> str:
    return "trajectory_" + spec.replace("/", "_") + ".txt"


def _run_single_checkpoint_producers(iter_name: str) -> None:
    hf_dir = _hf_dir(iter_name)
    analyze_json = _analyze_json(iter_name)
    joint_json = _joint_json(iter_name)

    # weights analyze: GPU pass over the HF checkpoint; produces analyze.json.
    _run(
        f"weights analyze {iter_name}",
        "Per-tensor base, row/column and spectral statistics for the HF checkpoint.",
        [
            "python3",
            "-m",
            "weights",
            "analyze",
            hf_dir,
            "--out",
            analyze_json,
            "--device",
            "cuda",
        ],
    )

    # weights joint: GPU MoE joint spectrum/alignment pass; produces joint.json.
    _run(
        f"weights joint {iter_name}",
        "MoE gate/up joint spectrum and alignment for the same HF checkpoint.",
        [
            "python3",
            "-m",
            "weights",
            "joint",
            hf_dir,
            "--out",
            joint_json,
            "--device",
            "cuda",
            "-k",
            str(WEIGHTS_JOINT_K),
        ],
    )


def _run_weights_batch() -> None:
    for iter_name in CHECKPOINT_ITERS:
        batch_cmd = [
            "python3",
            "-m",
            "weights",
            "batch",
            checkpoint_run_dir,
            "--out",
            output_root,
            "--commands",
            "analyze",
            "joint",
            "--pattern",
            iter_name,
            "--device",
            "cuda",
            "--joint-k",
            str(WEIGHTS_JOINT_K),
        ]
        if not WEIGHTS_BATCH_SKIP_EXISTING:
            batch_cmd.append("--no-skip-existing")

        # weights batch: GPU producer for exactly one whitelisted checkpoint
        # iteration. CHECKPOINT_ITERS is the source of truth; no broad glob.
        _run(
            f"weights batch analyze+joint {iter_name}",
            "Runs weights analyze and joint for this exact checkpoint iteration.",
            batch_cmd,
        )


def _run_weights_consumers_for_iter(iter_name: str) -> None:
    analyze_json = _analyze_json(iter_name)
    joint_json = _joint_json(iter_name)
    weights_report = _weights_report(iter_name)
    weights_dashboard_dir = _weights_dashboard_dir(iter_name)
    weights_story_config = _weights_story_config_path(iter_name)
    weights_story_html = _weights_story_html(iter_name)

    report_cmd = [
        "python3",
        "-m",
        "weights",
        "report",
        analyze_json,
        joint_json,
        "--top-n",
        str(WEIGHTS_REPORT_TOP_N),
        "--out",
        weights_report,
    ]
    if WEIGHTS_REPORT_ALL_METRICS:
        report_cmd.append("--all-metrics")

    # weights report: CPU text summary over analyze.json/joint.json.
    _run(
        f"weights report {iter_name}",
        "Direction-aware text report; consumes the analyze.json and joint.json above.",
        report_cmd,
    )

    # weights dashboard: CPU PNG dashboard over analyze.json/joint.json.
    _run(
        f"weights dashboard {iter_name}",
        "PNG dashboard: inventory, metric overviews, layer profiles and MoE heatmaps.",
        [
            "python3",
            "-m",
            "weights",
            "dashboard",
            analyze_json,
            joint_json,
            "--out-dir",
            weights_dashboard_dir,
            "--max-hist",
            str(WEIGHTS_DASHBOARD_MAX_HIST),
        ],
    )

    # weights story: CPU shareable HTML; needs a small JSON config plus dashboard.
    _write_json(weights_story_config, _weights_story_config(iter_name))
    _run(
        f"weights story {iter_name}",
        "Shareable HTML gallery over the weights dashboard.",
        [
            "python3",
            "-m",
            "weights",
            "story",
            weights_story_config,
            weights_dashboard_dir,
            "--out",
            weights_story_html,
            "--dashboard-url-prefix",
            "dashboard",
        ],
    )


def _run_multi_checkpoint_weights() -> None:
    if len(CHECKPOINT_ITERS) < 2:
        print("\n===== skip weights multi-checkpoint tools =====", flush=True)
        print(
            "# Set CHECKPOINT_ITERS to 2+ entries to run deltas, trajectories and compare.",
            flush=True,
        )
        return

    analyze_files = [_analyze_json(iter_name) for iter_name in CHECKPOINT_ITERS]
    multi_out = _multi_out_dir()

    report_cmd = [
        "python3",
        "-m",
        "weights",
        "report",
        *analyze_files,
        "--top-n",
        str(WEIGHTS_REPORT_TOP_N),
        "--out",
        f"{multi_out}/weights_report_deltas.txt",
    ]
    if WEIGHTS_REPORT_ALL_METRICS:
        report_cmd.append("--all-metrics")

    # weights report over 2+ analyze.json: CPU per-file summaries plus DELTAS.
    _run(
        "weights report multi-checkpoint",
        "Compares checkpoint iterations within one run and appends a DELTAS section.",
        report_cmd,
    )

    # weights report --trajectory: CPU layer x iteration tables for selected groups.
    for spec in WEIGHTS_TRAJECTORIES:
        traj_cmd = [
            "python3",
            "-m",
            "weights",
            "report",
            *analyze_files,
            "--top-n",
            str(WEIGHTS_REPORT_TOP_N),
            "--trajectory",
            spec,
            "--out",
            f"{multi_out}/{_trajectory_out_name(spec)}",
        ]
        if WEIGHTS_REPORT_ALL_METRICS:
            traj_cmd.append("--all-metrics")
        _run(
            f"weights trajectory {spec}",
            "Layer-by-iteration trajectory for one role/proj group across checkpoints.",
            traj_cmd,
        )

    # weights compare: CPU diff over 2+ analyze.json files. This is formally a
    # cross-run tool, but with explicit labels it is also useful for two checkpoints.
    compare_cmd = [
        "python3",
        "-m",
        "weights",
        "compare",
        *analyze_files,
        "--labels",
        *CHECKPOINT_ITERS,
        "--top-n",
        str(WEIGHTS_REPORT_TOP_N),
        "--csv-dir",
        f"{multi_out}/compare_csv",
        "--out",
        f"{multi_out}/weights_compare.txt",
    ]
    if WEIGHTS_REPORT_ALL_METRICS:
        compare_cmd.append("--all-metrics")
    _run(
        "weights compare multi-checkpoint",
        "Ranks tensors/metrics where the selected checkpoint iterations differ most.",
        compare_cmd,
    )


def run_tools() -> None:
    if len(CHECKPOINT_ITERS) == 1:
        _run_single_checkpoint_producers(CHECKPOINT_ITERS[0])
    else:
        _run_weights_batch()

    for iter_name in CHECKPOINT_ITERS:
        _run_weights_consumers_for_iter(iter_name)

    _run_multi_checkpoint_weights()

    print(f"\n===== done =====\nOutputs: {output_root}", flush=True)


def submit_all() -> None:
    print("all:", _job().submit())


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
