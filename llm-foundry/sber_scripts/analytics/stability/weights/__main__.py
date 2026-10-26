"""Single CLI entry point: ``python -m weights <subcommand> [args]``.

Subcommands:

* ``analyze HF_DIR --out PATH [...]``
    Per-tensor base + rowcol + spectral stats for an HF safetensors export.
* ``joint HF_DIR --out PATH [...]``
    Joint MoE Gram + gate↔up alignment for an HF safetensors export.
* ``batch RUN_DIR --out OUT_DIR --commands analyze,joint [...]``
    Discover ``ep0_baN/artifacts/hf_checkpoint/`` iterations under RUN_DIR
    and run commands on each. Skips iterations whose output already exists.
* ``report FILE [FILE ...] [--top-n N] [--all-metrics] [--trajectory ROLE/PROJ]``
    Read analyze/joint JSONs, print a compact textual report (HEADLINES +
    per-group aggregates + outliers + multi-file deltas).
* ``compare FILE FILE [...] [--labels A B ...] [--csv-dir DIR]``
    Cross-RUN diff: rank where N analyze.json runs disagree per
    ``(tensor, metric)``, grouped by role/proj/block. With files spanning
    several shared iterations, also emit a divergence trajectory.
* ``dashboard FILE [FILE ...] --out-dir DIR``
    Render a folder of diagnostic PNGs from analyze.json (+ optional
    joint.json): role inventory, per-metric overview, per-group layer
    profiles, per-expert × layer heatmaps, joint alignment scatter,
    abs-hist for headline tensors.
* ``story CONFIG.json DASHBOARD_DIR --out STORY.html``
    Render a single shareable HTML page combining narrative prose (from
    a small config JSON the agent writes) with the dashboard PNGs and
    a filterable gallery of every PNG in the folder.

``HF_DIR`` is the safetensors export directory (the one with
``model.safetensors.index.json`` and ``shard-*.safetensors`` /
``model-*-of-*.safetensors``). Typically lives at
``<iter_dir>/artifacts/hf_checkpoint/``.

Run any subcommand with ``--help`` for details.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# NOTE: heavy modules (analyze / joint / batch) pull in `torch` at import.
# `weights report` is stdlib-only and must work without torch (e.g. when
# an LLM agent runs the report from a non-GPU shell). Each _cmd_*
# imports what it needs lazily.


def _cmd_analyze(args: argparse.Namespace) -> None:
    from weights import analyze

    cfg = analyze.Config(
        device=args.device,
        topk_sigma=args.topk_sigma,
        svd_niter=args.svd_niter,
        debug_timing=args.debug_timing,
    )
    analyze.run_hf(
        args.hf_dir,
        args.out,
        config=cfg,
        include=args.include,
        exclude=args.exclude,
        limit=args.limit,
    )


def _cmd_joint(args: argparse.Namespace) -> None:
    from weights import joint

    cfg = joint.Config(
        device=args.device,
        k=args.k,
        gpu_gram_cap_gb=args.gpu_gram_cap_gb,
        max_gram_gb=args.max_gram_gb,
        cpu_fallback=args.cpu_fallback,
    )
    joint.run_hf(args.hf_dir, args.out, config=cfg)


def _cmd_report(args: argparse.Namespace) -> None:
    from weights import report as report_mod

    report_mod.run(
        args.files,
        top_n=args.top_n,
        out=args.out,
        all_metrics=args.all_metrics,
        trajectory=args.trajectory,
    )


def _cmd_compare(args: argparse.Namespace) -> None:
    from weights import compare as compare_mod

    compare_mod.run(
        args.files,
        labels=args.labels,
        top_n=args.top_n,
        all_metrics=args.all_metrics,
        csv_dir=args.csv_dir,
        out=args.out,
    )


def _cmd_dashboard(args: argparse.Namespace) -> None:
    import json
    from pathlib import Path as _Path
    from weights import dashboard as dashboard_mod

    analyze_path: _Path | None = None
    joint_path: _Path | None = None
    for p in args.files:
        blob = json.loads(p.read_text())
        if "tensors" in blob:
            analyze_path = p
        elif "joint" in blob or "alignment" in blob:
            joint_path = p
        else:
            raise SystemExit(f"can't determine file kind: {p}")
    if analyze_path is None:
        raise SystemExit("dashboard needs at least one analyze.json")
    dashboard_mod.run(
        analyze_path,
        joint_path,
        args.out_dir,
        max_hist=args.max_hist,
    )


def _cmd_story(args: argparse.Namespace) -> None:
    from weights import story as story_mod

    story_mod.run(
        args.config,
        args.dashboard_dir,
        args.out,
        dashboard_url_prefix=args.dashboard_url_prefix,
    )


def _cmd_batch(args: argparse.Namespace) -> None:
    from weights import analyze, batch, joint

    analyze_cfg = analyze.Config(
        device=args.device,
        topk_sigma=args.topk_sigma,
        svd_niter=args.svd_niter,
    )
    joint_cfg = joint.Config(
        device=args.device,
        k=args.joint_k,
        gpu_gram_cap_gb=args.gpu_gram_cap_gb,
        max_gram_gb=args.max_gram_gb,
        cpu_fallback=args.cpu_fallback,
    )
    cfg = batch.BatchConfig(
        commands=args.commands,
        pattern=args.pattern,
        skip_existing=not args.no_skip_existing,
        analyze_config=analyze_cfg,
        joint_config=joint_cfg,
    )
    batch.run_batch(args.root, args.out, config=cfg)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="weights",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="command", required=True)

    # ---- analyze --------------------------------------------------------
    p = sub.add_parser(
        "analyze",
        help="per-tensor base + rowcol + spectral stats on an HF export",
    )
    p.add_argument(
        "hf_dir",
        type=Path,
        help="HF safetensors export directory (e.g. <iter>/artifacts/hf_checkpoint/)",
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--include",
        type=str,
        default=None,
        help="regex; only analyze tensors whose FQN matches",
    )
    p.add_argument(
        "--exclude",
        type=str,
        default=None,
        help="regex; skip tensors whose FQN matches",
    )
    p.add_argument("--device", default=None, help="cpu / cuda (auto if omitted)")
    p.add_argument(
        "--topk-sigma",
        type=int,
        default=16,
        help="top-K randomised SVD width (default 16). "
        "Pass 0 to skip spectral stats entirely.",
    )
    p.add_argument("--svd-niter", type=int, default=2)
    p.add_argument(
        "--debug-timing",
        action="store_true",
        help="print per-tensor `base done` / `rowcol done` lines",
    )
    p.add_argument(
        "--limit", type=int, default=None, help="(debug) stop after N tensors"
    )
    p.set_defaults(func=_cmd_analyze)

    # ---- joint ----------------------------------------------------------
    p = sub.add_parser(
        "joint",
        help="joint MoE spectrum + gate↔up alignment on an HF export",
    )
    p.add_argument("hf_dir", type=Path, help="HF safetensors export directory")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default=None)
    p.add_argument(
        "-k", type=int, default=4, help="top-k joint singular directions to keep"
    )
    p.add_argument(
        "--gpu-gram-cap-gb",
        type=float,
        default=30.0,
        help="if Gram exceeds this on GPU, skip the (layer, "
        "proj) group by default. Use --cpu-fallback to "
        "retry on CPU instead. Default 30 GB.",
    )
    p.add_argument(
        "--cpu-fallback",
        action="store_true",
        help="when GPU Gram cap is exceeded, retry on CPU "
        "instead of skipping. Slow for big N; use only "
        "if you really need those layers.",
    )
    p.add_argument(
        "--max-gram-gb",
        type=float,
        default=200.0,
        help="hard cap even with --cpu-fallback. Default 200 GB.",
    )
    p.set_defaults(func=_cmd_joint)

    # ---- report (stdlib-only; no torch) ---------------------------------
    p = sub.add_parser(
        "report",
        help="compact textual report over analyze / joint JSON(s)",
    )
    p.add_argument(
        "files",
        type=Path,
        nargs="+",
        help="one or more analyze.json / joint.json files. "
        "Multiple analyze.json files in iteration order "
        "add a delta section at the end.",
    )
    p.add_argument(
        "--top-n", type=int, default=8, help="rows per ranking table (default 8)"
    )
    p.add_argument(
        "--all-metrics",
        action="store_true",
        help="emit every numeric metric found in records "
        "instead of just the curated diagnostic set",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="if set, also write the report to this file",
    )
    p.add_argument(
        "--trajectory",
        default=None,
        metavar="ROLE[/PROJ[/BLOCK]]",
        help="layer × iteration tables for ONE group across multiple "
        "analyze.json files. Use after a first-pass report to "
        "track a suspect (role, proj) through time.",
    )
    p.set_defaults(func=_cmd_report)

    # ---- compare (stdlib-only; cross-run diff) --------------------------
    p = sub.add_parser(
        "compare",
        help="cross-RUN diff: rank where N analyze.json runs disagree per "
        "(tensor, metric); use after `report` flags the suspect groups",
    )
    p.add_argument(
        "files",
        type=Path,
        nargs="+",
        help="2+ analyze.json files, one per run (or per (run, iter) "
        "for a divergence trajectory across shared iterations)",
    )
    p.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="run label per file (overrides path-derived run names)",
    )
    p.add_argument("--top-n", type=int, default=12, help="rows per table (default 12)")
    p.add_argument(
        "--all-metrics",
        action="store_true",
        help="compare every numeric metric, not just the curated set",
    )
    p.add_argument(
        "--csv-dir",
        type=Path,
        default=None,
        help="also write run_comparison_*.csv into this directory",
    )
    p.add_argument(
        "--out", type=Path, default=None, help="also tee the text report to this file"
    )
    p.set_defaults(func=_cmd_compare)

    # ---- dashboard ------------------------------------------------------
    p = sub.add_parser(
        "dashboard",
        help="render PNG dashboard folder from analyze.json (+ optional joint.json)",
    )
    p.add_argument(
        "files",
        type=Path,
        nargs="+",
        help="analyze.json (required), optionally joint.json too",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--max-hist",
        type=int,
        default=20,
        help="cap on abs-hist PNGs for headline tensors",
    )
    p.set_defaults(func=_cmd_dashboard)

    # ---- story ----------------------------------------------------------
    p = sub.add_parser(
        "story",
        help="render shareable HTML page (narrative + gallery) from a "
        "story config JSON + dashboard PNG folder",
    )
    p.add_argument(
        "config", type=Path, help="story config JSON (see weights.story docstring)"
    )
    p.add_argument(
        "dashboard_dir",
        type=Path,
        help="dashboard PNG folder produced by `weights dashboard`",
    )
    p.add_argument("--out", type=Path, required=True, help="output HTML path")
    p.add_argument(
        "--dashboard-url-prefix",
        default="dashboard",
        help="URL prefix for <img> tags relative to the HTML "
        "location (default: 'dashboard')",
    )
    p.set_defaults(func=_cmd_story)

    # ---- batch ----------------------------------------------------------
    p = sub.add_parser(
        "batch",
        help="run commands across all checkpoint iterations under RUN_DIR",
    )
    p.add_argument(
        "root",
        type=Path,
        help="run directory containing ep0_baN/ subfolders "
        "(each with artifacts/hf_checkpoint/), OR a "
        "single iteration directory",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output root; results land at <out>/<iter_name>/",
    )
    # Mirror of weights.batch.KNOWN_COMMANDS — inlined so argparse
    # construction doesn't import batch (which pulls in torch).
    p.add_argument(
        "--commands",
        nargs="+",
        default=["analyze"],
        choices=("analyze", "joint"),
        help="which commands to run on each iteration",
    )
    p.add_argument(
        "--pattern",
        default="ep0_ba*",
        help="glob pattern for iteration subfolder names",
    )
    p.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="re-run even if the output file already exists",
    )
    # analyze knobs (forwarded)
    p.add_argument("--device", default=None, help="cpu / cuda (auto if omitted)")
    p.add_argument("--topk-sigma", type=int, default=16)
    p.add_argument("--svd-niter", type=int, default=2)
    # joint knobs (forwarded)
    p.add_argument("--joint-k", type=int, default=4)
    p.add_argument("--gpu-gram-cap-gb", type=float, default=30.0)
    p.add_argument("--cpu-fallback", action="store_true")
    p.add_argument("--max-gram-gb", type=float, default=200.0)
    p.set_defaults(func=_cmd_batch)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
