"""Label actual dW and dHidden kernel launches in one warmed replay sample."""

import sys

import torch

import replay_native


original_overlay = replay_native.overlay
reset_peaks = torch.cuda.reset_peak_memory_stats
synchronize = torch.cuda.synchronize
started = False
ended = False


def start_trace(*args, **kwargs):
    global started
    reset_peaks(*args, **kwargs)
    if not started:
        torch.cuda.check_error(torch.cuda.cudart().cudaProfilerStart())
        torch.cuda.nvtx.range_push("isolated_lm_head_sample")
        started = True


def stop_trace(*args, **kwargs):
    global ended
    synchronize(*args, **kwargs)
    if started and not ended:
        torch.cuda.nvtx.range_pop()
        torch.cuda.check_error(torch.cuda.cudart().cudaProfilerStop())
        ended = True


def overlay(module, function, replacements):
    if function.__name__ != "backward":
        return original_overlay(module, function, replacements)
    old, order = replacements[-1]
    stages = order.splitlines()
    labeled = []
    for stage in stages:
        label = "head_dhidden" if "_benchmark_mm" in stage else "head_dw_including_pack"
        labeled.extend((f'    with torch.cuda.nvtx.range("{label}"):', "    " + stage))
    return original_overlay(module, function, [*replacements[:-1], (old, "\n".join(labeled))])


if __name__ == "__main__":
    torch.cuda.reset_peak_memory_stats = start_trace
    torch.cuda.synchronize = stop_trace
    replay_native.overlay = overlay
    sys.argv.append("--stages")
    # The variant overlays add NVTX labels, which are independently checked in SQLite.
    sys.argv.append("--samples")
    sys.argv.append("1")
    replay_native.main()
    assert ended
