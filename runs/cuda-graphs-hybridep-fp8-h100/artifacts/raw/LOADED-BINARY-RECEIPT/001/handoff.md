# Loaded binary receipt helper

## Scope

This directory contains a one-shot CPU-only collector for a later live fault
diagnostic. It does not launch or inspect a training run. Invoke it once from
the external owner after native update 1 is confirmed complete on all eight
ranks, using that owner's exact rank-to-PID mapping. No live capture was made
while preparing this helper.

The collector reads only the eight supplied `/proc/<PID>/stat`, `cmdline`, and
`maps` files. It checks all initial rank/PID/attempt bindings before it reads
any maps, checks each binding again immediately before its maps read, after the
maps read, and after hashing. Command-line matching records the argument
indices only; it does not save complete command lines or read process
environments. It imports only the Python standard library and does not import
Torch or CudaCherry.

## Invocation

Provide one `--rank-pid RANK:PID` option for each externally confirmed rank.
The output must be a new path; existing receipts are never overwritten.

```sh
python capture_loaded_binaries.py \
  --attempt-dir /absolute/path/to/this/attempt \
  --rank-pid 0:PID0 --rank-pid 1:PID1 \
  --rank-pid 2:PID2 --rank-pid 3:PID3 \
  --rank-pid 4:PID4 --rank-pid 5:PID5 \
  --rank-pid 6:PID6 --rank-pid 7:PID7 \
  --output /absolute/path/to/this/attempt/loaded-binary-receipt.json
```

Replace every `PIDn` with the exact positive PID for that rank. The attempt
directory string must be absolute and normalized, and must occur in every
worker's NUL-delimited command-line arguments. Exit code 0 means the receipt's
rank coverage and unique mapped-file hashes are complete; exit code 1 means a
partial receipt was saved; exit code 2 means the output could not be created.

## Receipt contents and limits

The JSON records per-rank timestamps, PID start-time checks, command-line
scope, matching raw map rows and parsed path/device/inode, separate quantize
and GEMM coverage, unique mapped-file identities, SHA-256/bytes, and path/open
stat snapshots before and after hashing. Matching files are grouped by map
device/inode and hashed once across ranks. A deleted-only mapping, missing or
unreadable file, map/stat identity mismatch, changing file, changed PID, absent
family, malformed map row, or incomplete rank map remains explicit and keeps
the overall status partial. Matching is conservative: the mapped shared-object
pathname must name CudaCherry and the quantize or GEMM family, including a
recognized JIT name. An opaque JIT filename remains uncovered.

When a mapped pathname reveals a `cudacherry` package root, the collector also
hashes `_loader.py` and `quantization/blockwise.py` if present. These are
optional adjacent-source records and do not substitute for mapped binary
coverage. The collector makes no `sys.modules` claim. Its maps and file hashes
do not attest an extension's registered operator body, build origin, executed
branch, or fault causality.

## Provenance context and cost

The recurrent-fault handoff requires quantize and GEMM provenance separately
in the same attempt; see
[`RECURRENT-CUDA-FAULT/002/handoff.md`](../../RECURRENT-CUDA-FAULT/002/handoff.md).
A006 recorded only the GEMM object at
`/opt/torchtitan-venv/lib/python3.12/site-packages/cudacherry/gemm/_C_gemm.cpython-312-x86_64-linux-gnu.so`,
SHA-256 `9548e0fdb0be6033a4bbd0e159ee9b04e20fcc12874b0d968e497ea18c53aaa5`,
13,384,960 bytes, inode 74482301. A006 did not retain the exact map line and
has no quantize binary receipt.

Read-only inspection of the installed `_loader.py` shows prebuilt import first
and JIT fallback through `torch.utils.cpp_extension.load`; its JIT names are
`cudacherry_C_gemm` and `cudacherry_C_quantize`. The installed
`quantization/blockwise.py` calls `load_family("quantize")`. The collector
derives optional static paths from the actual mapped package path and hashes
them without importing either file.

The only expected cost during the later external capture is CPU I/O: a few
small `/proc` reads per supplied PID, one SHA-256 pass for each distinct
mapped quantize/GEMM file identity, optional hashes of the two small Python
sources per discovered package root, and one compact JSON write. It adds no
training callback, barrier, CUDA wait, extension import/reload, or training
runtime work. The helper itself was verified with synthetic CPU fixtures; live
rank binding and loaded-binary provenance remain open until the authorized
external capture.

## Checks

Run the focused fixtures from this directory with:

```sh
python verify_cpu.py
```

The fixtures cover path spaces and deleted suffixes, stable hashes deduplicated
across eight ranks, static-source hashing, wrong attempt scope and missing PID
before maps reads, duplicate PIDs, missing files, inode replacement, changed
PID, absent family coverage, and exclusive output creation. They do not prove
live `/proc` access, GPU behavior, or training correctness.
