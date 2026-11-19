"""Record the first full-size compiled CuTe statistics replay on each rank."""

import os
from pathlib import Path
import sys


def install():
    from torch._inductor.runtime.runtime_utils import cache_dir

    from torchtitan.models.common.norms.cute_gated_norm.op import _forward_stats

    rank = int(os.environ["RANK"])
    kernels = _forward_stats._backend_fns
    original = kernels["cuda"]

    def observe(x, *args, **kwargs):
        result = original(x, *args, **kwargs)
        if x.shape == (2, 4096, 4096):
            frame = sys._getframe(1)
            while frame is not None:
                code = frame.f_code
                if code.co_name == "call" and Path(code.co_filename).is_relative_to(
                    cache_dir()
                ):
                    kernels["cuda"] = original
                    print(
                        f"fusion_followup executed=cute-replay-stats rank={rank} "
                        f"graph={Path(code.co_filename).name} line={frame.f_lineno} "
                        f"shape=2,4096,4096 device={x.device}",
                        flush=True,
                    )
                    break
                frame = frame.f_back
        return result

    kernels["cuda"] = observe
