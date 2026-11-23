"""Reuse the eight-rank replay on the two source trees from the training matrix."""

import hashlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "gather-repeat/replay8.py"
TRAIN = ROOT / "train-ep8"
BASELINE = ROOT.parents[2] / "a10-cute-001"
REPLAY_SHA256 = "3f035c7e8f18d7c0656a87204f2521018e807d5c2127f88eac3c73e2eb2c822b"


def replace(source, before, after, occurrences=1):
    if source.count(before) != occurrences:
        raise RuntimeError(f"Unexpected distributed replay source around {before!r}")
    return source.replace(before, after)


def main():
    original_bytes = ORIGINAL.read_bytes()
    if hashlib.sha256(original_bytes).hexdigest() != REPLAY_SHA256:
        raise RuntimeError(f"Unverified replay source {ORIGINAL}")
    source = original_bytes.decode()
    source = replace(source, '("baseline", "regular", "alias")', '("baseline", "regular")')
    source = replace(source, "choices=(2, 8)", "choices=(2, 8, 32)")
    source = replace(
        source, "source = CURRENT if candidate else replay.OLD",
        "source = CURRENT if candidate else BASELINE", occurrences=2,
    )
    source = replace(
        source, '"e3871b36c0be1cb5c027b811d1684558fb11b02b"',
        '"e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f"',
    )
    source = replace(source, 'PAIRS = {2: "9.14", 8: "5.5"}', 'PAIRS = {2: "9.14", 8: "5.5", 32: "5.5"}')
    source = replace(
        source, '    launcher = None\n    if candidate:',
        '    launcher = None\n'
        '    if not candidate:\n'
        '        from torchtitan.experiments.four_fusion import stable_ce\n'
        '        stable_ce.cross_entropy_loss = loss_module.cross_entropy_loss\n'
        '    if candidate:',
    )
    source = replace(
        source, 'assert allocator_settings["expandable_segments"] == candidate, allocator_settings',
        'assert allocator_settings["expandable_segments"], allocator_settings',
    )
    source = replace(
        source, "        receipt = run(args, rank, device, visible)",
        "        if candidate:\n"
        "            with staged_writer(args.chunks):\n"
        "                receipt = run(args, rank, device, visible)\n"
        "        else:\n"
        "            receipt = run(args, rank, device, visible)\n"
        '        receipt["isolated_pair_harness_sha256"] = file_sha(PAIR_HARNESS)',
    )
    sys.path[:0] = [str(ORIGINAL.parent), str(TRAIN)]
    from final_child import staged_writer

    namespace = {
        "__name__": "_isolated_pair_replay",
        "__file__": str(ORIGINAL),
        "BASELINE": BASELINE,
        "PAIR_HARNESS": Path(__file__).resolve(),
        "staged_writer": staged_writer,
    }
    exec(compile(source, str(ORIGINAL), "exec"), namespace)
    namespace["main"]()


if __name__ == "__main__":
    main()
