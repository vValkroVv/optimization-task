# Mapped CudaCherry binary metadata: bounded provenance check

## Decision

The actual mapped quantize and GEMM ELF files expose build IDs, compiler
identity/flags, and host DWARF source/build paths. Those paths are consistent
with the local package origin recorded by the installed distribution. The
inspected metadata contains no VCS revision or source-content digest, so it
cannot validate the installed CudaCherry source bytes reviewed in
[003](../003/handoff.md), and it does not contradict them. The actual
source-to-binary/build attestation gate remains unresolved.

This check does not identify an originating kernel or establish fault
causality. GIFT source snapshot 008 was not edited or used as binary build
provenance; its opt-in memory adapter is outside these CudaCherry shared
objects.

## Binary identity and observed metadata

The same-run [receipt](../../SANITIZER-ORIGIN/attempts/001/binary-provenance-receipt.json)
records stable map/hash checks for all eight ranks. Current CPU-side SHA-256
hashes match those receipt entries exactly:

| Mapped shared object | SHA-256 | GNU build ID |
| --- | --- | --- |
| cudacherry/quantization/_C_quantize.cpython-312-x86_64-linux-gnu.so (18,017,064 bytes) | e134ff643cc1e40cdd5443e4cf744b7221fe719ef46b28d69f1866185d44bc87 | 8703442a0fa15f4c0c14620183e5169e18cf4832 |
| cudacherry/gemm/_C_gemm.cpython-312-x86_64-linux-gnu.so (13,384,960 bytes) | 9548e0fdb0be6033a4bbd0e159ee9b04e20fcc12874b0d968e497ea18c53aaa5 | d20a408f45639ae37584f483aebc165a9f4c3f08 |

Both ELF files are x86-64, dynamically linked, and contain .nv_fatbin,
.debug_info, .debug_line, and .comment. Their .comment sections say
GCC: (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0. The first host DWARF
compilation unit in each reports producer:
GNU C++20 13.3.0 -mtune=generic -march=x86-64 -g -O2 -O3 -std=c++20
-fno-strict-overflow -fPIC -fasynchronous-unwind-tables
-fstack-protector-strong -fstack-clash-protection -fcf-protection.

- Quantize DWARF names
  /opt/cudacherry/src/cudacherry/csrc/quantization/blockwise/bindings.cpp
  and /opt/cudacherry/src/cudacherry/csrc/tensor/qtensor_bridge.cpp.
- GEMM DWARF names
  /opt/cudacherry/src/cudacherry/csrc/gemm/bindings.cpp.
- Both report compilation directory
  /opt/cudacherry/build/temp.linux-x86_64-cpython-312.

The installed cudacherry-0.1.0.dist-info metadata says version 0.1.0,
installer uv, setuptools wheel generator 78.1.0, and local direct URL
file:///opt/cudacherry; uv_build.json is empty. The targeted RECORD
rows match each current shared-object SHA-256 and size. These fields identify
the installed artifact and a local build/install route; RECORD is not a source
attestation, and the direct URL has no VCS commit ID.

## Limits and checks

Inspected only the two receipt-pinned CPU shared objects and their package
install metadata. GNU readelf was /usr/bin/readelf, Binutils 2.42; file was
5.45. The SHA check used the exact paths below; the other ELF commands ran on
each path.

~~~sh
sha256sum /opt/torchtitan-venv/lib/python3.12/site-packages/cudacherry/quantization/_C_quantize.cpython-312-x86_64-linux-gnu.so /opt/torchtitan-venv/lib/python3.12/site-packages/cudacherry/gemm/_C_gemm.cpython-312-x86_64-linux-gnu.so
file <path>
readelf -n <path>
readelf -p .comment <path>
readelf -SW <path> | rg '\.debug_(info|abbrev|line|str)|\.comment|\.note\.gnu\.build-id|\.nv_fatbin|\.nvFatBinSegment'
readelf --debug-dump=info --dwarf-depth=1 <path> | rg -m 60 'Compilation Unit|DW_TAG_compile_unit|DW_AT_(name|comp_dir|producer|stmt_list|linkage_name)'
~~~

In this command list, path denotes either absolute shared-object path in the
linked same-run receipt.
Python standard-library reads covered direct_url.json, uv_build.json, INSTALLER,
WHEEL, METADATA, and the two RECORD rows, including independent SHA-256/size
comparisons.

GNU build IDs provide identifiers for these linked ELF outputs;
they do not identify their source revision. DWARF evidence here is limited to
host compilation-unit headers. No CUDA fatbin/PTX or kernel code was decoded,
no whole-file string sweep was done, and no source contents were reread after
consulting 003. Thus an unexamined embedded string or external build record is
not ruled out, but the inspected ELF and installation metadata do not attest
source contents or revision. No imports, CUDA/GPU queries, launch, source edits,
package changes, or cleanup were performed.

This read-only inventory adds no production runtime cost.
