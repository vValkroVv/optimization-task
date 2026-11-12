# Lane A004 allocation-history result

## Result

The single graph-on diagnostic failed with exit code 1 at the same dense block
0 FORWARD warmup CUDA Graph Trees pool guard as A003. All eight rank error
records tie bad pointer 597168226304 to the allocation stack:

    at::empty(...)
    cuda_gemm::(anonymous namespace)::get_cached_workspace(...)
    cuda_gemm::gemm(...)

The pointer and producer appear in each rank's allocator-history section. This
confirms the CudaCherry workspace producer for the guarded allocation. The
loaded extension path and binary hash were not exposed by the existing route,
so runtime binary identity remains open. The built-in report does not include a
block size or standalone snapshot. The same pointer appeared in A003.

The error occurred during the first forward warmup before an optimizer update.
No successful capture/replay, numerical comparison, throughput result, or
performance claim was produced.

## Run and cleanup

- Popen PID 515371 started at 2026-11-11T23:50:05.822927Z. Runner exit code 1
  was recorded at 2026-11-11T23:50:35.441029Z.
- Rank PIDs 515455 through 515462 and memory sampler PID 515384 exited.
- Post-unwind, all eight same-UUID H100s reported 0 MiB and 0% utilization; the
  compute-app query was empty. No kill was needed.
- Root's launch preflight and invocation receipts are retained as
  launch-preflight.txt and launch-invocation.json.
- Complete interleaved output is in supervisor.log. Each rank's full error and
  allocation history is in
  torchrun/01a36e84-c7c0-45b8-aa67-bebf178f6694_cpo7bqkf/attempt_0/{0..7}/error.json,
  with corresponding stderr.log and stdout.log files.

The supervisor.log SHA-256 is
5ab9d3c1759c832a369e6857aecdf5fddb343bf7add373d2a141434cbba31cb8.
failure-summary.json records SHA-256 values for all eight rank error files.

## Frozen source and launch

SOURCE-SNAPSHOT/003 remains unchanged: manifest SHA-256
eae89a656a0e59efb2248dd921395d418afa2485ab2ae320d1e834a14ce97526, archive
SHA-256 ea215abaac14d98a21c36e8684dbd8ce88a1d48f47cec5fe4f85f61be5dd4e27,
and preset SHA-256 b341701bd1eef7774c110fbf7ed69cf6280ba8691100b05be457dee3e7d50a7b.
The exact 3B MoE, EP8/FSDP2, FP8, FlexAttention, selective AC, model/loss
compilation, batch 8/global 256, sequence 4096, four-microbatch accumulation,
seed 42, 25-step limit, 30000-step schedule, and A003 profiler flags were
preserved. The only diagnostic setting was the built-in
torch._inductor.config.triton.cudagraph_trees_history_recording=True option,
set by the attempt-local bootstrap before importing the unchanged entrypoint.
The allocator guard remained active.

## Preparation and extraction notes

- Review caught that an early A004 launcher draft checked for supervisor.log
  and launch.json after Popen creates them. That draft was discarded before
  launch; the known A003 launcher was restored with only the A004 attempt path
  and history_entry.py target substitutions.
- One source-guard invocation initially resolved the repository-relative
  manifest path under the raw-artifact root twice and failed before hashing.
  The corrected source guard passed all 22 pinned source files, the archive,
  and preset before launch; the launcher repeated its own frozen-source guard.
- Two patch-generation commands failed before changing files because of command
  quoting/syntax. A post-run extraction command also failed on escaped newline
  matching. Direct per-rank JSON extraction succeeded.
- The first supervisor-only reading showed an unsymbolized traceback tail.
  Full rank error.json records contain the producer frames above.

These setup and extraction failures did not change source, package state, or
training behavior.
