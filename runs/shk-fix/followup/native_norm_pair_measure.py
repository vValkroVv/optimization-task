"""Run the existing GIFT measurement worker with the opt-in norm-pair probe."""

import importlib.util
import os
from pathlib import Path
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
BRIDGE = HERE.parents[1] / "flash-qla/measure.py"
CANDIDATE = HERE.parent / "artifacts/raw/norm-pair-tile-reuse-002/candidate.py"
SWITCH = "GIFT_CUTE_NORM_PAIR"
ACTIVATION = "CuTe norm pair candidate activated"


def run_candidate(train_argv, adapter, native_main):
    if any(
        os.environ.get(name) != "1"
        for name in (
            "GIFT_CUTE_NORM_FUSE_RESIDUAL",
            "GIFT_CUTE_NORM_SAC_STATS_ONLY",
            "GIFT_FP8_QUANTIZE_WEIGHT_TAIL",
        )
    ):
        raise ValueError("The norm-pair candidate needs the three-fix preset")

    import torch
    from norm_f1_component import load_variant
    from norm_pair_tile_compiled import FuseNormPairAfterSAC
    from norm_pair_tile_ops import make_fused_pair
    from norm_skip import digest
    from torchtitan.models.common.norms.cute_gated_norm import compiler, forward

    device = torch.device("cuda", torch.cuda.current_device())
    if (
        torch.cuda.get_device_capability(device) != (9, 0)
        or forward.resolve_num_sms(device) != 116
    ):
        raise ValueError("The norm-pair candidate requires SM90 and the 116-SM preset")

    candidate = load_variant(forward, CANDIDATE, "forward_norm_pair_tile_reuse")
    compile_pair = candidate.compile_norm_pair
    active = False

    def compile_and_record(*args):
        nonlocal active
        compiled = compile_pair(*args)
        if not active:
            print(ACTIVATION, flush=True)
            active = True
        return compiled

    candidate.compile_norm_pair = compile_and_record
    op = make_fused_pair(candidate)
    rewrite = FuseNormPairAfterSAC(
        compiler.GatedNormStatsOnlyPass(),
        op._opoverload,
        tuple(
            digest(path)
            for path in (
                Path(__file__),
                HERE / "norm_pair_tile_compiled.py",
                HERE / "norm_pair_tile_ops.py",
                CANDIDATE,
            )
        ),
    )
    # The original bootstrap has already selected this rank's device and caches.
    with patch.object(compiler, "GatedNormStatsOnlyPass", return_value=rewrite):
        result = native_main(train_argv, adapter)
    if not active:
        raise RuntimeError(
            f"No fused CUDA call. Rewrites: {rewrite.rewrites}; skipped: {rewrite.skipped}"
        )
    return result


def main():
    enabled = os.environ[SWITCH]
    if enabled not in ("0", "1"):
        raise ValueError("Expected the binary norm-pair experiment switch")
    spec = importlib.util.spec_from_file_location("shk_norm_pair_flash_measure", BRIDGE)
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load the original measurement worker at {BRIDGE}"
        )
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    if enabled == "0":
        return bridge.main()

    bootstrap = bridge._bootstrap_and_run

    def start(train_argv, adapter, native_main):
        def native_with_candidate(argv, current_adapter):
            return run_candidate(argv, current_adapter, native_main)

        return bootstrap(train_argv, adapter, native_with_candidate)

    with patch.object(bridge, "_bootstrap_and_run", start):
        return bridge.main()


if __name__ == "__main__":
    main()
