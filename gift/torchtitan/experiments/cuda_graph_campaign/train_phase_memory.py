"""Training entrypoint with the scan correction and native phase-memory tap."""

from __future__ import annotations

import logging
import os

from torchtitan.experiments.cuda_graph_campaign.backend_scan_fix import (
    install_hybrid_ep_scan_fix,
)
from torchtitan.experiments.cuda_graph_campaign.phase_memory import (
    DEFAULT_PHASE_BOUNDARIES,
    install_phase_memory_observer,
)
from torchtitan.tools.logging import init_logger


logger = logging.getLogger(__name__)


def _report_secondary(
    primary: BaseException, label: str, secondary: BaseException
) -> None:
    note = f"phase-memory {label} also failed: {type(secondary).__name__}: {secondary}"
    add_note = getattr(primary, "add_note", None)
    if callable(add_note):
        add_note(note)
    logger.error(note)


def main(argv: list[str] | None = None) -> None:
    init_logger()

    output_dir = os.environ.get("TORCHTITAN_PHASE_MEMORY_OUTPUT_DIR")
    if not output_dir:
        raise RuntimeError(
            "set TORCHTITAN_PHASE_MEMORY_OUTPUT_DIR to a new shared output directory"
        )
    observer = install_phase_memory_observer(output_dir, DEFAULT_PHASE_BOUNDARIES)

    try:
        install_hybrid_ep_scan_fix()
        from torchtitan.train import main as native_main

        native_main(argv)
    except BaseException as primary:
        try:
            observer.finish_and_flush(primary)
        except BaseException as flush_error:
            _report_secondary(primary, "flush", flush_error)
        try:
            observer.restore()
        except BaseException as restore_error:
            _report_secondary(primary, "restoration", restore_error)
        raise
    else:
        cleanup_error: BaseException | None = None
        try:
            observer.finish_and_flush()
        except BaseException as exc:
            cleanup_error = exc
        try:
            observer.restore()
        except BaseException as exc:
            if cleanup_error is None:
                cleanup_error = exc
            else:
                _report_secondary(cleanup_error, "restoration", exc)
        if cleanup_error is not None:
            raise cleanup_error.with_traceback(cleanup_error.__traceback__)


if __name__ == "__main__":
    main()
