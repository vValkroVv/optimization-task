# E=4 scan generated-cost comparison

## Result

The matched original and corrected SM90 scan artifacts have **different emitted instructions and identical reported resources**. This closes the question of static code/resource equivalence for this one specialization; it does not establish equal runtime cost or fault causality.

| Property | Original rank 2 | Corrected smoke rank 2 |
| --- | ---: | ---: |
| Registers per thread | 69 | 69 |
| Shared memory per block, bytes | 1,472 | 1,472 |
| Stack / local memory, bytes | 0 / 0 | 0 / 0 |
| Constant bank 0, bytes | 640 | 640 |
| Function text, bytes | 31,744 | 31,744 |
| Static instruction slots (including 11 NOPs) | 1,984 | 1,984 |
| Local-memory LDL/STL instructions | 0 | 0 |

Of 1,984 matching instruction offsets, 99 have different full encodings (including scheduling/control bits); 48 have different printed assembly. The first differing offset is `0x1960`, the last `0x5190`. Opcode/mnemonic histograms are identical. The diff includes instruction ordering, registers/predicates, operands and scheduling encodings. These are real emitted differences; no addresses, registers or predicates were normalized away. No additional local-memory traffic or register/shared allocation increase is observed. The retained ptxas compilation log is unavailable, so compiler-reported spill counts are not asserted; zero local allocation and absence of LDL/STL are the available binary evidence.

## Identity and method

Both objects contain exactly the filtered target function `hybrid_ep::scan<256,108,128,64,8,1,4>(...)`, with SM90 and embedded nvlink toolkit 13.2 V13.2.51 metadata. `receipts.json` records exact binary paths/hashes and cuobjdump argv/status/output files. `comparison.json` records header hashes, resource values, opcode histograms, raw-dump hashes and caveats. `*.sass.txt`, `*.resources.txt`, `*.elf.txt` and `*.symbols.txt` are raw outputs; `sass.diff` and `elf.diff` retain differences.

Instruction slots were parsed from cuobjdump address-labelled lines; the next encoding-only line was appended to the same slot, preserving both 64-bit halves. Counts compare corresponding offsets, with no semantic normalization. The ELF text-section size is `0x7c00`; this independently matches 1,984 16-byte slots. The resource dumps are byte-identical. SASS/ELF dumps differ; symbol dumps are byte-identical.

The original artifact is the recorded failed-baseline task PID 459018 rank-2 cache object. The corrected artifact is rank 2 PID 471082 from the completed native smoke, matching that smoke's recorded SHA-256. Header hashes were reread and verified. Original exact source-to-binary provenance retains the earlier limitation: installed header, timestamps, PID and include route support it, but no preprocessed translation unit/exact compiler command survived. The corrected manifest establishes its one-bound staging delta and fresh cache; this artifact comparison alone cannot assign every emitted difference specifically to that edit.

## Interpretation and remaining gates

The source fix does not increase the reported register, shared, local or stack demand on this specialization. Equal instruction counts/resources do not prove equal latency: changed dependencies or scheduling can change execution. No device compute, host submission, synchronization/idle, communication/copy latency, integrated runtime or paired memory peak was measured. Those require the campaign's authorized comparative runtime evidence. No speedup, regression magnitude, fault cause, numerical parity or sustained distributed correctness is claimed here. Binary differences do not prove this kernel caused the earlier asynchronous CUDA fault.

This was one offline comparison using cuobjdump and Python standard-library file parsing. No CUDA initialization, GPU queries/execution, source/package changes, compilation or other kernels were requested. Only this evidence directory was written. Acceptance is complete for the bounded static comparison; campaign runtime and numerical gates remain with the owner.
