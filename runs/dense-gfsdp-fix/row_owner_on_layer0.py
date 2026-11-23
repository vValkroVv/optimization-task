#!/usr/bin/env python3
"""Run the conditional row-owner arm under the pinned native campaign."""

from pathlib import Path
from subprocess import CalledProcessError
import sys

import campaign


ARM = "layer0-row-owner"


def main():
    # Original A's direct row-owner arm reached Dynamo's fullgraph cache limit.
    # Both sides of this contrast use the already-tested first-FFN split.
    campaign.ARMS[ARM] = (campaign.LAYER0, campaign.ROW_OWNER)
    campaign.PER_FIX = ((8, ARM), (4, ARM))
    campaign.SCRIPT = Path(__file__).resolve()
    return campaign.main(sys.argv[1:])


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, ValueError, CalledProcessError) as error:
        print(f"ERROR {error}", file=sys.stderr)
        raise SystemExit(1)
