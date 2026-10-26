from __future__ import annotations

import sys
from pathlib import Path

from metric_sampler.capture.kernels import activation_ext


def main() -> None:
    build_root = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    extension = activation_ext.build(build_root)
    print(f"built {extension.__file__}")


if __name__ == "__main__":
    main()
