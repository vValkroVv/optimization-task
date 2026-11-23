#!/usr/bin/env python3
"""Check whether retaining eager BF16 casts explains the MTP split delta."""

from pathlib import Path
import sys
from unittest.mock import patch

FFN = Path(__file__).resolve().parents[1] / "all-final-bitwise-001" / "ffn"
sys.path.insert(0, str(FFN))
import ffn_bitwise


def main():
    import torch._inductor.config as config

    original_run = ffn_bitwise.run

    def capture(harness, fixtures, root, args, dims, device, rank):
        if args.component != "mtp":
            raise ValueError("This diagnostic targets the MTP FFN split.")
        if config.emulate_precision_casts:
            raise RuntimeError("Default-run precision emulation was already enabled.")
        results = {}
        for label, preserve_casts in (("original", False), ("preserve_bf16_casts", True)):
            if rank == 0:
                print(f"MTP precision probe: {label}", flush=True)
            settings = {"emulate_precision_casts": preserve_casts}
            if rank == 0:
                settings.update({"trace.enabled": True, "trace.debug_dir": str(args.output / label)})
            with config.patch(settings):
                results[label] = original_run(harness, fixtures, root, args, dims, device, rank)
        return results

    with patch.object(ffn_bitwise, "run", capture), patch.object(ffn_bitwise, "CASES", {"captured": None}):
        ffn_bitwise.main()


if __name__ == "__main__":
    main()
