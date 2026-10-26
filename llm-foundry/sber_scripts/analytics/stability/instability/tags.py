"""Tag inventory + structural classification for a TB run.

Loads *every* scalar tag a run logs, classifies each via ``features.parse_tag``
(layer, monitor group, projection, family template), and reports what's there —
so you can see the full monitor set a run produces without hand-maintaining
pattern lists, and spot anything new or unclassified.

Three things it surfaces:

* **By group** — how the discovered families map onto the coarse monitor groups
  (with the structural fallback, a brand-new monitor still lands in a sensible
  group, not ``other``).
* **UNRECOGNIZED** — families that still fall to ``other`` and aren't known global
  scalars: the only thing that ever needs a human rule.
* **NOT LOADED BY DEFAULT** — families the default deny-list
  (``tb_helpers.default_load_keep``) drops, i.e. monitors that ``report`` /
  ``compare`` / ``temporal`` skip. Should normally be only eval/profiler/timing
  noise + slash-form non-residual aggregates.

Usage::

    python -m instability.tags --tb-dir /path/to/RUN [--csv tags.csv] [--full]
"""

from __future__ import annotations

import argparse
import csv as csvmod
from collections import defaultdict
from pathlib import Path

# _SCALAR_PREFIXES is the single source of truth for "global scalar, not a model
# monitor" — features uses it to keep such tags at group "other"; we reuse it so the
# UNRECOGNIZED filter below stays in lockstep instead of duplicating the list.
from instability.features import _SCALAR_PREFIXES, parse_tag, register_tag_groups
from instability.tb_helpers import default_load_keep, load_run_scalars


def inventory(tags: list[str]) -> dict[str, dict]:
    """Aggregate tags into ``{family: {group, projection, layers, n_tags, sample}}``."""
    fams: dict[str, dict] = {}
    for tag in tags:
        info = parse_tag(tag)
        f = fams.setdefault(
            info.family,
            {
                "group": info.group,
                "projections": set(),
                "layers": set(),
                "n_tags": 0,
                "sample": tag,
                "recognized": info.recognized,
                "in_default": default_load_keep(tag),
            },
        )
        f["projections"].add(info.projection)
        if info.layer is not None:
            f["layers"].add(info.layer)
        f["n_tags"] += 1
    return fams


def _table(rows: list[list[str]], headers: list[str]) -> None:
    if not rows:
        print("  (none)")
        return
    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))
    fmt = "  " + "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print("  " + "  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt.format(*r))


def report(fams: dict[str, dict], *, full: bool) -> None:
    n_tags = sum(f["n_tags"] for f in fams.values())
    print(
        f"\n{'=' * 78}\n  TAG INVENTORY — {len(fams)} families, {n_tags} tags\n{'=' * 78}"
    )

    # ---- by group ----------------------------------------------------------
    by_group: dict[str, list[str]] = defaultdict(list)
    for fam, f in fams.items():
        by_group[f["group"]].append(fam)
    print("\n  -- families by monitor group --")
    rows = []
    for group in sorted(by_group, key=lambda g: (-len(by_group[g]), g)):
        members = by_group[group]
        n_t = sum(fams[m]["n_tags"] for m in members)
        sample = min(members, key=len)
        rows.append([group, str(len(members)), str(n_t), sample[:60]])
    _table(rows, ["group", "families", "tags", "sample family"])

    # ---- unrecognized ------------------------------------------------------
    unknown = [
        fam
        for fam, f in fams.items()
        if not f["recognized"] and not f["sample"].startswith(_SCALAR_PREFIXES)
    ]
    print(f"\n  -- UNRECOGNIZED ({len(unknown)}) — group=other, not a known scalar --")
    print("     (these are the only families that may need a monitor_group rule)")
    _table(
        [
            [fam[:64], str(len(fams[fam]["layers"])), fams[fam]["sample"][:50]]
            for fam in sorted(unknown)
        ],
        ["family", "layers", "sample tag"],
    )

    # ---- dropped by the default deny-list -----------------------------------
    skipped = sorted(fam for fam, f in fams.items() if not f["in_default"])
    print(f"\n  -- LOGGED BUT NOT LOADED BY DEFAULT ({len(skipped)}) --")
    print(
        "     (default deny-list drops these — eval/profiler/timing noise + slash-form non-residual aggregates)"
    )
    _table(
        [[fam[:64], fams[fam]["group"], str(fams[fam]["n_tags"])] for fam in skipped],
        ["family", "group", "tags"],
    )

    # ---- full per-family table ---------------------------------------------
    if full:
        print("\n  -- all families --")
        rows = []
        for fam in sorted(fams, key=lambda m: (fams[m]["group"], m)):
            f = fams[fam]
            rows.append(
                [
                    fam[:60],
                    f["group"],
                    ",".join(sorted(f["projections"]))[:24],
                    str(len(f["layers"])),
                    str(f["n_tags"]),
                ]
            )
        _table(rows, ["family", "group", "projections", "layers", "tags"])


def dump_csv(tags: list[str], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csvmod.writer(fh)
        w.writerow(
            [
                "tag",
                "group",
                "projection",
                "layer",
                "family",
                "recognized",
                "in_default",
            ]
        )
        for tag in sorted(tags):
            info = parse_tag(tag)
            w.writerow(
                [
                    tag,
                    info.group,
                    info.projection,
                    "" if info.layer is None else info.layer,
                    info.family,
                    info.recognized,
                    default_load_keep(tag),
                ]
            )
    print(f"\n[tags] wrote per-tag classification → {out}")


def run(
    tb_dir: Path,
    *,
    csv: Path | None = None,
    full: bool = False,
    tag_groups: Path | None = None,
) -> dict[str, dict]:
    if tag_groups is not None:
        import json as _json

        register_tag_groups(_json.loads(Path(tag_groups).read_text()))
    # Load EVERYTHING (prefixes=("",) matches all tags) so the audit is complete;
    # this gets its own cache entry, separate from the default-prefix loads.
    series = load_run_scalars(str(tb_dir), prefixes=("",))
    tags = list(series)
    if not tags:
        raise SystemExit(f"no tags found in {tb_dir}")
    fams = inventory(tags)
    report(fams, full=full)
    if csv:
        dump_csv(tags, csv)
    return fams


def _cli() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tb-dir", required=True, type=Path, help="TB run directory")
    ap.add_argument(
        "--csv", type=Path, default=None, help="dump per-tag classification"
    )
    ap.add_argument(
        "--full", action="store_true", help="print every family, not just summaries"
    )
    ap.add_argument(
        "--tag-groups",
        type=Path,
        default=None,
        help="JSON {substring: group} of your headline families, merged on "
        "top of the built-ins (same as $INSTABILITY_TAG_GROUPS)",
    )
    args = ap.parse_args()
    run(args.tb_dir, csv=args.csv, full=args.full, tag_groups=args.tag_groups)


if __name__ == "__main__":
    _cli()
