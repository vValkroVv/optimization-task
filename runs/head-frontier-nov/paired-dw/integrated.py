"""A single fresh native GFSDP head replay with per-pair rather than per-chunk dW stores."""
import argparse
from pathlib import Path
import sys

import triton

from pair import PairWriter

sys.path.insert(0,str(Path("/workspace-SR008.nfs2/vkropotin/optimization-task/runs/head-frontier-nov/layout-order")))
import replay_native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chunks",type=int,choices=(2,8))
    parser.add_argument("arm")
    args = parser.parse_args()
    candidate = args.arm.startswith("pair")
    group = []

    def hook(native, chunks, rows, vocab, width, stream):
        class Trial:
            name = "paired_triton_tma"
            pair = "sequential_fp32_tma_config_5"
            version = triton.__version__
            def __init__(self):
                self.impl = PairWriter(5)
            def __call__(self, grad, hidden, slot, add, stream):
                self.impl(grad, hidden, slot, add)
            @property
            def receipt(self):
                return {"stores": self.impl.stores, "pending": self.impl.pending is not None}
        writer = Trial()
        group.append(writer)
        return writer

    # Digest of the full native outputs is taken after the unprofiled hot timings.
    replay_native.run(args.chunks,Path(f"c{args.chunks}-{args.arm}.json").resolve(),
                      arm=args.arm,writer_hook=hook if candidate else None)
    if candidate:
        assert group[0].receipt == {"stores": 23 * args.chunks // 2, "pending": False}


if __name__ == "__main__":
    main()
