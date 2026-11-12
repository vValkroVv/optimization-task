# Baseline followup with post-failure details

## Result

**Clean bounded nonreproduction:** one fresh process completed all2112 GEMMs, checking every output exactly. Positive groups passed1024 calls, empty-final group passed1024 calls, and production-scale K passed64 calls. No exception, mismatch or core occurred. No further execution or candidate patch was attempted.

This does not erase [001's call2 numerical failure](../001/REPORT.md), establish original-fault causality, or establish native-training stability. The requested mismatch coordinates/group counts were unavailable because this followup did not fail. Positive-group failure alone would not select a descriptor-drain repair: the same schedule also contains missing-release CTAs20–51. Patch selection remains unresolved.

## Scope and identities

[fixture.py](fixture.py) preserves001's oracle, inputs, actual GIFT call, same-stream D resets,32-output ring, chunk boundaries,120-second workload cap, and1024/1024/64 budgets including each admission call. The only numerical-path addition is a branch after a detected mismatch to record failing calls, per-group mismatch counts, first16 coordinates/values, maximum absolute error and finiteness. That branch did not execute and did not perturb any failing prefix. No full tensors were written.

The dormant summary branch's `tile_cta` formula is an **unvalidated row-major hypothesis**, not an established mapping from DeepGEMM's scheduler. No CTA field was emitted. Raw coordinates would be the primary evidence if this branch ran; the actual scheduler/swizzle would need inspection before interpreting a CTA. Do not reuse that derived field as causal evidence.

001's executed fixture remains byte-identical: SHA256 `302862ff7a5500023b70ea81c86f8f237cd2e78a860423225d3147ac54aa3fa0`. This executed002 fixture SHA256 is `c77b9964ee393c108c6200c527189602ef8691afb268cf998074ce5685acbf23`. Checks and per-case totals are in [verification.json](verification.json).

The launch reused001's known compiled cache. Actual loaded key `04254286e5f40c1e389b04ec1a89cd8d`, cubin SHA256 `dafcb551bb26cc90fb1e37c647262ca1dfaf1fd539ca9440e34326843b813f4c`, matches001. Its generated source matches the original fault-associated template;001 already demonstrated exact SASS equality with the original cubin. Case1 logged the actual load; cases2/3 reused the same in-process specialization without another load. See [events.jsonl](events.jsonl).

Fresh [launch admission](launch.json) verified all eight exact reconciliation002 UUIDs, no compute applications, sufficient GPU/filesystem capacity, and made only authorized GPU0 visible. Exact capture flags, fresh core path, graph0/mixorder0, interpreter, PYTHONPATH, and command are recorded there. [completion.json](completion.json) records returncode0. No profiler, sanitizer, launch blocking, training, dependency/source edit, installation or package change occurred.

## Checks and decision

CPU packing/oracle checks and actual interface inspection passed again for this002 file; see [cpu-check.json](cpu-check.json). Ruff F821/F811/F841 passed on fixture.py and launch.py. Local document links and whitespace passed. The invocation was `/opt/torchtitan-venv/bin/python3.12 runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/BASELINE-STABILITY/reproducer/002/launch.py` from the task root, following the same CPU command as001 with002 substituted.

Runtime-cost review is unchanged from001: this task-only diagnostic retains384MiB of output-ring storage, same-stream zeroing, boundary synchronization and device equality kernels, plus debug host logging. The additional detail branch has no cost on this clean path. No latency, throughput or speed claim is made.

Combined evidence:001 failed at call2 after an exact first-call pass;002 passed the complete fixed campaign using the same machine code. This supports an intermittent fixture-path failure but identifies neither the corrupted output tile nor its mechanism. The next decision belongs to the independent integration/oracle review; no descriptor repair is selected. Original TMA-fault causality, a matched causal fail/pass comparison, integration and required native stability passes remain open. Work stops at this agreed boundary.
