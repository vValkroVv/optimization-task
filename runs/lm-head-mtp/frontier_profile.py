"""Capture the first post-warmup native or storage replay for diagnosis."""

import sys

import frontier_lifetime
import replay


def main():
    cuda = replay.torch.cuda
    reset = cuda.reset_peak_memory_stats
    synchronize = cuda.synchronize
    started = False
    ended = False

    def start(*args, **kwargs):
        nonlocal started
        reset(*args, **kwargs)
        if not started:
            cuda.check_error(cuda.cudart().cudaProfilerStart())
            cuda.nvtx.range_push("isolated_lm_head_sample")
            started = True

    def stop(*args, **kwargs):
        nonlocal ended
        synchronize(*args, **kwargs)
        if started and not ended:
            cuda.nvtx.range_pop()
            cuda.check_error(cuda.cudart().cudaProfilerStop())
            ended = True

    cuda.reset_peak_memory_stats = start
    cuda.synchronize = stop
    assert "--dump" not in sys.argv
    frontier_lifetime.main()
    assert ended


if __name__ == "__main__":
    main()
