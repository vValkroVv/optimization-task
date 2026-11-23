"""Compose the verified norm-pair and routed SwiGLU native measurement workers."""

import importlib.util
import os
from unittest.mock import patch

from native_norm_pair_measure import BRIDGE, run_candidate
from native_norm_pair_measure import SWITCH as NORM_PAIR_SWITCH
from native_routed_swiglu_fwd_measure import run_native
from native_routed_swiglu_fwd_measure import SWITCH as SWIGLU_SWITCH


def main():
    swiglu = os.environ.get(SWIGLU_SWITCH)
    if os.environ.get(NORM_PAIR_SWITCH) != "1" or swiglu not in ("0", "1"):
        raise ValueError(f"Expected {NORM_PAIR_SWITCH}=1 and {SWIGLU_SWITCH}=0 or 1")

    spec = importlib.util.spec_from_file_location("shk_norm_pair_swiglu_flash_measure", BRIDGE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load the original measurement worker at {BRIDGE}")
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    bootstrap = bridge._bootstrap_and_run

    def start(train_argv, adapter, native_main):
        def with_swiglu(argv, current_adapter):
            return run_native(argv, current_adapter, native_main, swiglu == "1")

        def with_norm_pair(argv, current_adapter):
            return run_candidate(argv, current_adapter, with_swiglu)

        return bootstrap(train_argv, adapter, with_norm_pair)

    # Both wrappers run after the shared bootstrap selects the device and caches.
    with patch.object(bridge, "_bootstrap_and_run", start):
        return bridge.main()


if __name__ == "__main__":
    main()
