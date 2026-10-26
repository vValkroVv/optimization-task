"""Run ``weights`` commands on many checkpoint iterations at once.

Each iteration is expected to expose its HF safetensors export at::

    <iter_dir>/artifacts/hf_checkpoint/

The script discovers iteration directories (e.g. ``ep0_baN``) and runs
the requested commands on each. Output: ``out_root/<iter_name>/<cmd>.json``.

Idempotent — by default skips iterations whose output JSON already
exists, so a crashed batch can be re-launched without redoing finished
work.
"""

from __future__ import annotations

import re
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from weights import analyze, joint


# What output file each command writes by convention.
OUTPUT_FILENAMES: dict[str, str] = {
    "analyze": "analyze.json",
    "joint": "joint.json",
}

KNOWN_COMMANDS: tuple[str, ...] = tuple(OUTPUT_FILENAMES.keys())

# Path within an iteration directory where the HF export lives.
HF_SUBPATH = Path("artifacts") / "hf_checkpoint"


@dataclass
class BatchConfig:
    commands: list[str] = field(default_factory=lambda: ["analyze"])
    pattern: str = "ep0_ba*"
    skip_existing: bool = True
    analyze_config: analyze.Config | None = None
    joint_config: joint.Config | None = None


def _step_from_name(name: str) -> int | None:
    m = re.search(r"ba(\d+)", name)
    return int(m.group(1)) if m else None


def _has_hf(p: Path) -> bool:
    """A directory is a valid iteration if it has the HF subpath."""
    return (p / HF_SUBPATH).is_dir()


def discover_checkpoints(path: Path, pattern: str = "ep0_ba*") -> list[Path]:
    """Find iteration directories under ``path``.

    Two cases:
    * ``path`` itself is an iteration (contains ``artifacts/hf_checkpoint/``)
      — returned as a 1-element list.
    * ``path`` is a run directory — return its subfolders matching
      ``pattern`` that have an HF subpath.

    Sorted by parsed step (``ep0_ba2000`` < ``ep0_ba4000``).
    """
    if not path.is_dir():
        raise FileNotFoundError(f"not a directory: {path}")

    if _has_hf(path):
        return [path]

    children = sorted(
        p for p in path.iterdir() if p.is_dir() and p.match(pattern) and _has_hf(p)
    )
    if not children:
        raise RuntimeError(
            f"no iterations with {HF_SUBPATH} under {path} (pattern={pattern!r})"
        )
    return sorted(children, key=lambda p: _step_from_name(p.name) or 0)


def _run_one(
    cmd: str,
    ckpt_dir: Path,
    out_dir: Path,
    cfg: BatchConfig,
) -> Path | None:
    """Execute one command on one iteration. Returns output path or None on error."""
    out_file = out_dir / OUTPUT_FILENAMES[cmd]
    if cfg.skip_existing and out_file.exists():
        print(f"  [skip] {ckpt_dir.name}/{cmd}: {out_file.name} already exists")
        return out_file

    out_dir.mkdir(parents=True, exist_ok=True)
    hf_dir = ckpt_dir / HF_SUBPATH
    print(f"\n>>> {ckpt_dir.name} / {cmd}   (HF: {hf_dir})", flush=True)
    try:
        if cmd == "analyze":
            analyze.run_hf(
                hf_dir,
                out_file,
                config=cfg.analyze_config or analyze.Config(),
            )
        elif cmd == "joint":
            joint.run_hf(
                hf_dir,
                out_file,
                config=cfg.joint_config or joint.Config(),
            )
        else:
            raise ValueError(f"unknown command {cmd!r}")
        return out_file
    except Exception as e:  # noqa: BLE001
        print(
            f"!!! {ckpt_dir.name}/{cmd} FAILED: {e!r}\n{traceback.format_exc()}",
            flush=True,
        )
        return None


def run_batch(
    root: Path | str,
    out_root: Path | str,
    *,
    config: BatchConfig | None = None,
) -> dict[Path, dict[str, Path | None]]:
    """Run commands on every discovered iteration under ``root``."""
    cfg = config or BatchConfig()
    root = Path(root)
    out_root = Path(out_root)
    for c in cfg.commands:
        if c not in KNOWN_COMMANDS:
            raise ValueError(f"unknown command {c!r}; pick from {KNOWN_COMMANDS}")

    ckpts = discover_checkpoints(root, cfg.pattern)
    print(
        f"[batch] discovered {len(ckpts)} iteration(s) under {root}; "
        f"commands={cfg.commands}; out={out_root}",
        flush=True,
    )
    for c in ckpts:
        print(f"  - {c.name}", flush=True)

    results: dict[Path, dict[str, Path | None]] = {}
    for ckpt in ckpts:
        ckpt_out = out_root / ckpt.name
        per: dict[str, Path | None] = {}
        for cmd in cfg.commands:
            per[cmd] = _run_one(cmd, ckpt, ckpt_out, cfg)
        results[ckpt] = per

    print("\n" + "=" * 60)
    print("[batch] summary")
    print("=" * 60)
    for ckpt, per in results.items():
        statuses = " ".join(
            f"{cmd}={'OK' if path else 'FAIL'}" for cmd, path in per.items()
        )
        print(f"  {ckpt.name}: {statuses}")
    return results
