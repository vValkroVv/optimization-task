"""Compare exact CUDA CE with the current native compiled head replay."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

RUNS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RUNS / "layout-order"))
sys.path.insert(0, str(RUNS / "fused-ce"))
import replay_native


def install(launcher):
    """Bind the existing fused replay factory to the active native CE route."""
    import ce_storage
    import integrated
    from torchtitan.experiments.four_fusion import stable_ce

    native_make = ce_storage.make
    scale_checks = integrated.install(launcher)
    fused_make = ce_storage.make
    ce_storage.make = native_make
    prepared = []

    def native_factory(backend):
        consume = fused_make(backend, True)
        prepared.append(backend)
        return consume

    stable_ce.make_consuming_folded_cross_entropy = native_factory
    return scale_checks, prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chunks", type=int, choices=(2, 8))
    parser.add_argument("arm", choices=("control", "fused"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--method", choices=("cuda-13", "cuda-16"), default="cuda-13")
    args = parser.parse_args()

    native_make_head = replay_native.replay.make_head
    launcher = None
    if args.arm == "fused":
        import fused

        launcher = fused.Launcher(args.method)

        def install_on_native_head(weight):
            nonlocal scale_checks, prepared
            scale_checks, prepared = install(launcher)
            return native_make_head(weight)

        scale_checks = prepared = None
        replay_native.replay.make_head = install_on_native_head

    replay_native.run(args.chunks, args.output, arm=args.arm)
    receipt = json.loads(args.output.read_text())
    assert len(receipt["elapsed_ms"]) == len(receipt["memory"]) == receipt["warmup"] == 10
    assert receipt["gather"]["input_data_ptr"] != receipt["gather"]["output_data_ptr"]
    assert receipt["compile"]["backend"] == "inductor"
    assert receipt["compile"]["counters"]["unique_graphs"] == 1
    if launcher is not None:
        assert prepared == ["inductor"]
        assert launcher.calls == 23 * args.chunks
        (incoming, expected), = scale_checks
        actual = torch.stack(incoming).float()
        assert torch.equal(actual, expected), (actual, expected)
        binary = Path(fused.__file__).with_name(os.environ.get("FUSED_CE_BINARY", "centered.so"))
        receipt["candidate"] = {
            "method": launcher.name,
            "calls": launcher.calls,
            "incoming_fp32_scales": actual.tolist(),
            "prepared_fp32_scales": expected.tolist(),
            "launcher_sha256": hashlib.sha256(Path(fused.__file__).read_bytes()).hexdigest(),
            "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        }
    receipt["gather_mode"] = "regular"
    receipt["driver_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
