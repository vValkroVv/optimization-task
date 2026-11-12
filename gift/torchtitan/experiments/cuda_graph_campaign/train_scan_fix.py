"""Training entrypoint with the campaign-scoped HybridEP scan correction."""

from __future__ import annotations

from torchtitan.experiments.cuda_graph_campaign.backend_scan_fix import (
    install_hybrid_ep_scan_fix,
)
from torchtitan.tools.logging import init_logger


def main(argv: list[str] | None = None) -> None:
    init_logger()
    install_hybrid_ep_scan_fix()

    from torchtitan.train import main as native_main

    native_main(argv)


if __name__ == "__main__":
    main()
