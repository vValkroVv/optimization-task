# KGroupedContiguous caller, workspace lifetime and stream review

## Result

No reachable caller or temporary-workspace lifetime defect was identified that explains the rank3 CTA82 TMA-B illegal instruction in attempt002. The installed DeepGEMM host binary allocates per-call PyTorch scratch, submits the kernel on the current CUDA stream, and releases the scratch tensor after submission. This is the normal same-stream allocator lifetime contract. Prioritize the separate inner-kernel descriptor/scheduler investigation; this review does not establish first-corruption causality or a stable baseline.

Finite acceptance: trace active grouped-wgrad inputs and ownership; verify the installed binding and scratch launch/release; inspect compiled/custom-op and HybridEP stream boundaries; relate findings to CTA82; document remaining uncertainty. All read-only review items are complete. No source changes, GPU calls, training, package changes, or benchmarks were performed. Evidence files were written outside training source.

## Proven path and contracts

1. `gift/torchtitan/models/common/moe/experts/gmm.py:GroupedExperts._fp8_experts_forward` passes the same `FP8DispatchState` padded sizes/CPU copy into both grouped GEMMs. `grouped_gemm.py:_GroupedGemm.forward` saves weight storage and quantized input with `save_for_backward`; metadata is held on `ctx`. In backward, DGRAD precedes two row-to-column requantizations; the second GEMM's WGRAD is the matching `[4,1536,1024]` case. WGRAD zeros its output and invokes `_DeepGemmHandle.k_grouped_fp8_gemm_contiguous` with explicit A/B data, A/B scales, output and count arguments.
2. `deep_gemm_handle.py:_k_grouped_fp8_gemm_contiguous` is a CUDA `torch.library.custom_op` with mutable `d`, no output aliases and a fake implementation returning `None`. The actual installed schema is captured in [identities.json](identities.json): `Tensor(a4!) d`, all data/scales/counts explicit inputs. The eager body reads `ks_cpu.tolist()` when available, validates groups/alignment, optionally trims capacity, and calls installed `deep_gemm.k_grouped_fp8_gemm_nt_contiguous((a_data,a_scale),(b_data,b_scale),d,ks,ks_tensor,d)`. The binding docstring agrees with that positional signature; `c=d` selects accumulation. No Python descriptor or pointer is retained by the GIFT handle; its cache contains only the wrapper/library selection and SM budget.
3. Installed `cudacherry/quantization/row2col_dq.py:requantize_rowwise_to_columnwise` allocates output data/scales with PyTorch, then calls an explicit mutable-output registered operator. Installed `csrc/quantization/row2col_dq/bindings.cpp` declares mutation of `Y` and `S`. Its `row2col_kernel.cu:launch_kernel` uses `at::cuda::getCurrentCUDAStream()` at line322. The temporary int32 contiguous counts conversion at line353 is also on that current stream. No side stream is created here.
4. `fp8_tensor.py:clear_tensor_data` is a no-op during compilation, so traced backward does not execute its eager `.data` storage swap. The inspected generated Inductor cache provides concrete evidence beyond the decorator: `NATIVE-SMOKE/001/cache/torchinductor/qq/cqqwonglgwwmh2ntiynjjfeetzskg4f3vqlyukyarcyu7mnknu6b.py:4410–4469` copies a common counts producer to distinct live count buffers, runs both row2col operators, zeros `[4,1536,1024]`, and calls the KGrouped custom op with the resulting data/scales/counts. B/count buffers are deleted only after that call. These custom operations survive compilation despite returning `None`; their mutation dependencies are retained. This is a cached graph of the matching production shape, not proof that this exact cache file was the faulting attempt002 invocation.
5. `gift/torchtitan/distributed/deepep/hybridep.py:_launch_on_comm_stream` waits for compute before dispatch; records compute-produced CUDA inputs on the communication stream; joins compute back to communication and records returned CUDA tensors on compute. `_dispatch_impl` calls that join before returning payload/count outputs. The active resolved config selects `comm_stream: true`, `fuse_permute: true`, pad128, blocking dispatch (no nonblocking capacity factor). Blocking dispatch produces completed host counts, copies them to device, and preserves a CPU int32 copy. `activation_checkpoint.py:_get_save_ops` saves HybridEP dispatch/combine, avoiding replayed communication through SAC. SAC/compiled recomputation does not introduce a new stream or a GIFT-global descriptor workspace on this path.

## Installed DeepGEMM host binary

The package includes device headers but lacks the original C++ host API source. The short build commit `f093a4c` is not publicly retrievable at the attempted raw primary-source URL (404), so no upstream source was substituted as installed source. The later upstream `1f6f3f378920ccb5cc036ef43eb3f5972e921713` API was located, but is not used to assert installed behavior. Actual installed binary inspection resolves the material lifetime question.

[Installed disassembly](installed-host-disassembly.txt), from `_C.cpython-312-x86_64-linux-gnu.so`, establishes:

| Address / symbol | Observation |
| --- | --- |
| `k_grouped_fp8_gemm_nt_contiguous` `0x68120` | Active binding host function; transforms both scale layouts before allocation. |
| `0x684b3–0x6850f` | Gets SM count, shifts left9, invokes `torch::empty` for a per-call local tensor. At116 SMs this is59392 byte-sized elements. |
| `0x68b1e–0x68b52` | Takes that local tensor's storage pointer and stores it in runtime Args. |
| `0x68822`, `0x68882` | Makes A/B base TMA descriptors before launch; scale and output descriptors follow. |
| `0x69175` | Calls `LaunchRuntime<SM90FP8Gemm1D1DRuntime>::launch`. |
| `0x6919f–0x691a2` | Releases the workspace tensor only after launch returns; transformed scales are released afterward. |
| Launch `0x7767f` | Calls `c10::cuda::getCurrentCUDAStream(-1)`; converts it to raw stream and constructs the launch config. |
| Launch `0x777ab–0x777f4` | Supplies argument storage to `cuLaunchKernelEx`; no stream switch occurs between allocation and this launch. |

Pinned Torch's installed `_tensor_docs.py:3966–4008` explicitly documents same-creation-stream allocator safety after host deallocation; `record_stream` is needed for unmanaged use on another stream. Therefore host return before asynchronous kernel completion does not itself free scratch for concurrent reuse. A same-stream next call may reuse the address safely, because GPU accesses are ordered. Adding unconditional `record_stream`, synchronization, retained tensor lists or workspace cloning is not justified by this path.

## Fit to CTA82 and remaining uncertainty

Installed `deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh:30–42` passes base descriptors by grid-constant value and scratch by pointer. Lines92–93 give each CTA four128-byte descriptor slots. With116 CTAs, host allocation covers all464 slots; CTA82 B slots330/331 occupy bytes42240–42495, within59392 bytes. There is no CTA82 host workspace bounds violation. Host base descriptors are copied as launch arguments; Python scope ending does not invalidate them. GIFT/CudaCherry handles FP8 payloads/scales/counts, not the internal dynamic descriptor slots.

A caller data lifetime failure would primarily threaten operand contents; it does not alone demonstrate malformed TMA descriptor parameters. Corrupted counts or an earlier out-of-bounds producer could still indirectly corrupt maps or sizes. The core omits global/shared memory, so actual failing counts, descriptor bytes, allocation addresses and first corruption remain unknown. This inspection does not exclude such corruption or an inner-kernel dynamic descriptor hazard. No independent no-GPU test can establish those GPU facts.

## Runtime cost and decision

This evidence-only review adds no production work. Existing per-wgrad costs include two requantization launches and output buffers, zeroed BF16 WGRAD, opaque-op Python/dispatcher submission, optional layout transforms, and per-call59392-byte scratch; no added synchronization or communication is proposed. Magnitudes are unmeasured. The CPU host-count read in the active blocking route adds no device-value synchronization; device `.tolist()` remains only the fallback route. Do not change production precision, SAC, compilation, routing, or streams based on this review. Remaining acceptance is causal repair and the already required affected-path GPU verification/native stability passes owned by the coordinator.

## Reproduction and verification

Read-only commands used the exact task `PYTHONPATH` and `/opt/torchtitan-venv/bin/python3.12`, printed installed paths/binding/schema, and checked `torch.cuda.is_initialized() == False`. Torch is `2.15.0.dev20260822+cu130`; DeepGEMM public version is `2.3.0`, `_build.py` pins `2.3.0+f093a4c`. [Identities](identities.json) records SHA256 hashes for the binary, installed producer source, GIFT callers and inspected generated graph. Disassembly is reproducible with `objdump -d -C --start-address=0x68120 --stop-address=0x69370 <binary>` and the launch range `0x77550–0x77c00`. No GPU runtime was initialized. Local links and trailing whitespace were checked after writing this report.
