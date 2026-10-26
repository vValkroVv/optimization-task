from __future__ import annotations

import time
from collections.abc import Callable

import torch

from stand.activation_kernel_checks.fake_recorder import make_recorder
from stand.activation_kernel_checks.kernel_calls import launch_statistics_kernel, observe_statistics

# A launch that takes longer than this for a new row count compiled a kernel.
RECOMPILE_SECONDS = 0.1
SEEN_ROWS = 49152
EXTRA_ROWS = 64
WIDTH = 1024
NEW_ROW_COUNTS = (49153, 49160, 49171, 49201, 49216)


def check_no_recompile_on_row_count(device: torch.device) -> list[str]:
    # MoE points see a different row count every step; a fresh compile per count costs seconds.
    recorder = make_recorder(device)
    activations = torch.randn(SEEN_ROWS + EXTRA_ROWS, WIDTH, device=device, dtype=torch.bfloat16)

    def launch(rows: int) -> None:
        launch_statistics_kernel(activations[:rows], recorder.float_words)

    return _slow_launches(launch, "")


def check_extension_no_slowdown_on_row_count(device: torch.device, extension) -> list[str]:
    # The extension has no row-count-dependent specialization, but a call with an unseen row count should still
    # stay fast: this guards against a regression that adds one back.
    recorder = make_recorder(device)
    activations = torch.randn(SEEN_ROWS + EXTRA_ROWS, WIDTH, device=device, dtype=torch.bfloat16)

    def launch(rows: int) -> None:
        observe_statistics(extension, activations[:rows], recorder.float_words, False)

    return _slow_launches(launch, "extension ")


def _slow_launches(launch: Callable[[int], None], label_prefix: str) -> list[str]:
    problems = []
    for detailed in (False, True):
        launch(SEEN_ROWS)
        torch.cuda.synchronize()
        for rows in NEW_ROW_COUNTS:
            started = time.perf_counter()
            launch(rows)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if elapsed > RECOMPILE_SECONDS:
                problems.append(
                    f"{label_prefix}rows={rows} detailed={detailed}: {elapsed:.2f} s, looks like a recompile"
                )
    return problems
