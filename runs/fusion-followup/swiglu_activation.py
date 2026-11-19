"""Record the first compiled expert SwiGLU launch with the selected native geometry."""

import os
from pathlib import Path
import sys


def install():
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

    from torch._inductor.runtime.runtime_utils import cache_dir

    original = CachingAutotuner.run

    def observe(kernel, *args, **kwargs):
        result = original(kernel, *args, **kwargs)
        name = kernel.inductor_meta.get("kernel_name", "")
        constants = kernel.triton_meta.get("constants", {})
        if name.startswith("_swiglu_quant_bwd_kernel") and constants.get("hid_dim") == 1536:
            frame = sys._getframe(1)
            while frame is not None:
                code = frame.f_code
                if code.co_name == "call" and Path(code.co_filename).is_relative_to(cache_dir()):
                    break
                frame = frame.f_back
            if frame is None or kwargs.get("benchmark_run", False):
                return result
            rows, cols = constants["ROW_BLOCK"], constants["COL_BLOCK"]
            if (rows, cols) != (16, 128):
                raise RuntimeError(f"Unexpected compiled expert SwiGLU tile: {rows}x{cols}")
            CachingAutotuner.run = original
            print(
                f"fusion_followup executed=swiglu-row16 rank={os.environ['RANK']} "
                f"tile={rows}x{cols} kernel={name} graph={Path(code.co_filename).name}",
                flush=True,
            )
        return result

    CachingAutotuner.run = observe
