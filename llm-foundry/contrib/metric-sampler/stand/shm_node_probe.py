from __future__ import annotations

import json
import os
import shutil
import statistics
import time
from multiprocessing import shared_memory

import torch
import torch.distributed as dist

LAYERS = 61
EXPERTS = 256
MICROBATCHES = 4
SLOTS = 16
STEPS = 200
WARMUP_STEPS = 10
COUNT_LIMIT = 5000
COUNT_TENSORS_PER_MICROBATCH = 2
SCALARS_PER_MICROBATCH = 6
SCALARS_PER_STEP = 8
MATMUL_SIZE = 8192
WORD_BYTES = 4
CUDA_HOST_REGISTER_DEFAULT = 0
MILLISECONDS_PER_SECOND = 1000
BYTES_PER_KIB = 1024
BYTES_PER_GIB = 2**30


def capture_tensors(device: torch.device) -> list[torch.Tensor]:
    tensors = []
    for _ in range(LAYERS):
        tensors.append(torch.randint(0, COUNT_LIMIT, (EXPERTS,), device=device, dtype=torch.int32))
        tensors.append(torch.rand(EXPERTS, device=device))
        tensors.append(torch.rand(EXPERTS, device=device))
        for _ in range(MICROBATCHES * COUNT_TENSORS_PER_MICROBATCH):
            tensors.append(torch.randint(0, COUNT_LIMIT, (EXPERTS,), device=device, dtype=torch.int32))
    for _ in range(MICROBATCHES * SCALARS_PER_MICROBATCH + SCALARS_PER_STEP):
        tensors.append(torch.rand(1, device=device))
    return tensors


def time_steps(
    device: torch.device, sources: list[torch.Tensor], views: list[torch.Tensor], buffer: torch.Tensor,
    slots: torch.Tensor,
) -> tuple[list[float], list[float], list[float]]:
    copy_stream = torch.cuda.Stream(device=device)
    work_matrix = torch.randn(MATMUL_SIZE, MATMUL_SIZE, device=device, dtype=torch.bfloat16)
    record_milliseconds, step_end_milliseconds, overlap_milliseconds = [], [], []
    for step in range(STEPS):
        torch.matmul(work_matrix, work_matrix)
        start = time.perf_counter()
        torch._foreach_copy_(views, sources)
        buffer[-1] = step
        record_milliseconds.append((time.perf_counter() - start) * MILLISECONDS_PER_SECOND)
        start = time.perf_counter()
        copy_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(copy_stream):
            slots[step % SLOTS].copy_(buffer, non_blocking=True)
            event = torch.cuda.Event()
            event.record(copy_stream)
        buffer.record_stream(copy_stream)
        step_end_milliseconds.append((time.perf_counter() - start) * MILLISECONDS_PER_SECOND)
        torch.matmul(work_matrix, work_matrix)
        start = time.perf_counter()
        event.synchronize()
        overlap_milliseconds.append((time.perf_counter() - start) * MILLISECONDS_PER_SECOND)
    torch.cuda.synchronize()
    return record_milliseconds, step_end_milliseconds, overlap_milliseconds


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    job_id = os.environ.get("SLURM_JOB_ID", "local")

    sources = capture_tensors(device)
    words = sum(tensor.numel() for tensor in sources)
    buffer = torch.zeros(words, dtype=torch.int32, device=device)
    views = []
    offset = 0
    for tensor in sources:
        destination = buffer[offset:offset + tensor.numel()]
        views.append(destination if tensor.dtype == torch.int32 else destination.view(torch.float32))
        offset += tensor.numel()

    slot_bytes = words * WORD_BYTES
    name = f"msprobe_{job_id}_r{local_rank}"
    segment = shared_memory.SharedMemory(name=name, create=True, size=slot_bytes * SLOTS)
    host = torch.frombuffer(segment.buf, dtype=torch.int32)
    register_status = int(
        torch.cuda.cudart().cudaHostRegister(host.data_ptr(), slot_bytes * SLOTS, CUDA_HOST_REGISTER_DEFAULT)
    )
    slots = host.view(SLOTS, words)

    record_milliseconds, step_end_milliseconds, overlap_milliseconds = time_steps(
        device, sources, views, buffer, slots
    )

    usage = shutil.disk_usage("/dev/shm")
    report = {
        "rank": rank,
        "local_rank": local_rank,
        "host": os.uname().nodename,
        "payload_kib": slot_bytes / BYTES_PER_KIB,
        "cuda_host_register_status": register_status,
        "host_is_pinned": bool(host.is_pinned()),
        "record_ms_median": statistics.median(record_milliseconds[WARMUP_STEPS:]),
        "step_end_ms_median": statistics.median(step_end_milliseconds[WARMUP_STEPS:]),
        "event_wait_after_matmul_ms_median": statistics.median(overlap_milliseconds[WARMUP_STEPS:]),
        "dev_shm_total_gib": usage.total / BYTES_PER_GIB,
        "dev_shm_free_gib": usage.free / BYTES_PER_GIB,
        "tensors_per_step": len(sources),
    }
    print("PROBE", json.dumps(report), flush=True)
    dist.barrier()

    if local_rank == 0:
        visible = []
        for peer_rank in range(local_world_size):
            peer = shared_memory.SharedMemory(name=f"msprobe_{job_id}_r{peer_rank}")
            last = int(torch.frombuffer(peer.buf, dtype=torch.int32).view(SLOTS, words)[(STEPS - 1) % SLOTS][-1])
            visible.append({"local_rank": peer_rank, "last_step_word": last})
            peer.close()
        print("PROBE_NODE", json.dumps({"host": os.uname().nodename, "segments": visible}), flush=True)
    dist.barrier()

    torch.cuda.cudart().cudaHostUnregister(host.data_ptr())
    del host, slots
    segment.close()
    segment.unlink()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
