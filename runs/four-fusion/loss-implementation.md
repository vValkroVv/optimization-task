# Independent loss implementations and scoped CUDA checks

Status: initial implementation and scoped checks recorded below; production evaluation
remains open. The later [native denominator correction](loss-native-denominator-fix.md)
resolves the Tensor-only assumption missed by these initial checks and owns the
current scalar/compiler evidence. Initial source `da6774a1c2191bfec0e1cb4aa1dc34019b3b7df8`; root fast-forwarded
to `33b336a5` while this work was in progress. Only `.gitignore` and the newly
available [profile ideas](../../docs/profile-my-ideas.md) changed upstream; loss
source bytes were unchanged. The new profile document does not change the settled
[loss design](loss-design.md), and expert BF16 reuse is outside this assignment.
Applied `code-rules` and `perf-experiments`; reused the design's source/research
pins and preimplementation runtime-cost review. No dependencies were installed.

## Finite acceptance and result

- Implement independently activatable experiments 3 and 4 using the settled FLA
  and Kindulov directions: complete, [linear patch](linear-ce.patch) and
  [CE patch](ce.patch). Apply exactly one against V2; never stack them.
- Preserve native outer chunking, head `__call__`, MTP/normalization, hidden/head
  gradients, wrapper flags, compile, evaluation, probes and hooks: scoped native
  and single-device GFSDP checks pass; eight-rank production integration is open.
- Correct the absolute-LSE and target-rounding defects: actual CUDA eager and
  compiled regressions pass, and the unchanged donor fails as expected.
- Keep FP32 arithmetic and internal dW accumulation, and verify the installed
  FP32-output GEMM API: complete within the scoped checks.
- Preserve exact commands, failures, runtime identity, final costs and remaining
  gates: recorded below; raw evidence is in ignored `artifacts/raw/loss-checks/`.
- Restore original core before handoff and check both patches independently:
  complete. `gift/torchtitan/components/loss.py` SHA256 is
  `14d546d63b069a69041ec4a379d10a06f9b6a7888f25a68dddcadf27293b7f7c`, identical
  to the saved V2 bytes. No patch is currently active.

## Source and integration

[linear_ce.py](../../gift/torchtitan/experiments/four_fusion/linear_ce.py) adapts
FLA's token-tile/GEMM design (upstream
`1d2281ff1b7988d8070f00dba5ffa9747ae3a60e`; attribution retained). At production
folded chunk shape `[4,2048,4096]`, it uses eight internal 1,024-row tiles. These
are inside the unchanged two outer sequence chunks. One FP32 dW accumulator
spans every internal tile and both main/MTP contributions, followed by one BF16
output cast. `torch.addmm(..., out_dtype=torch.float32, out=accumulator)` was
executed successfully on the installed runtime, including each candidate test.
No FP32 parameter-size temporary or operand casts are used for those GEMMs.

The thin patch creates one adapter on each chunked loss and installs an
instance-local `forward` on its wrapped head. The argument remains named `input`.
The loss calls **`head(...)`**, so GFSDP gathers and its differentiable input and
output hooks run normally. The head adapter calls a separately compiled,
fake/autograd-registered `four_fusion::native_linear_ce` custom op. Both output
loss terms are summed immediately by the unchanged inner-backward loop. This
bridge deliberately supports **only native unit-upstream inner backward** after
known head weights and the shared denominator have entered dlogits. It does not
support arbitrary unknown incoming gradients, higher derivatives, or taking an
individual head term's gradient. There is no device predicate checking upstream.

Selection retains the original head/CE path for evaluation without input grads,
shape probes, acceptance argmax sinks, DTensor inputs/labels, unsupported head
implementations, biased heads, non-BF16 hidden inputs, nonzero head softcap, and
non-FSDP head hooks. Only known `Unembedding.forward` / `nn.Linear.forward`
implementations are selected. GFSDP/FSDP hooks remain active. No supported mode
is rejected to activate the experiment. Existing FP8 trunk, SAC, hidden z-loss,
loader/RNG, optimizer and checkpoint structures are untouched. No parameters or
buffers are added; state-dict keys and parameter identities pass direct checks.

[stable_ce.py](../../gift/torchtitan/experiments/four_fusion/stable_ce.py) adapts
the cap-zero row algorithm from Kindulov
`b983803337dfcc34fdd571b38aaf5e01df583bbe` (BSD attribution retained). Original
reference/contrib bytes are unchanged. Two separate FP32 row statistics preserve
`(logit - maximum) - log_denominator` in loss and backward. SUM stays FP32;
labels and row offsets stay signed int64. Target subtraction and incoming scale
happen in FP32 before the single logits-dtype gradient store. Ignored rows are
exactly zero. Its fake/autograd custom ops work through the existing compiled
`self.fn`; DTensor inputs retain the original distributed CE path. CPU uses
PyTorch FP32 CE. CUDA derivatives are explicitly first order only.

Experiment 3 shares these centered row kernels as an implementation dependency;
importing them does not activate experiment 4. The CE patch changes only the CE
callable; the linear patch changes only its constructor/head/outer-loss boundary.
No trainer, feature knob, preset, runner or contributor-owned file was changed.

## Executed verification

Only physical **device 2**, UUID `GPU-2f380105-fce8-292b-f1a6-54dbea10840a`, was
used. `nvidia-smi` showed no compute processes and 0 MiB usage before these
checks and after release. It is an NVIDIA H100 80GB HBM3. Runtime:
`/opt/torchtitan-venv/bin/python3.12`, Python 3.12.3,
Torch `2.15.0.dev20260822+cu130`, build commit
`8b353065ce260336c5fdfa74d754e8bd9b91b062`, CUDA runtime 13.0, Triton 3.8.0.
The coordinator's installed API/source inventory owns native linear-CE source
hashes; native API existence was not used as evidence of semantic compatibility.

Every Python CUDA command used this prefix from the task root:

```bash
source /etc/profile.d/wm.sh
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=gift
export TORCHINDUCTOR_CACHE_DIR=$PWD/runs/four-fusion/artifacts/raw/loss-checks/inductor
```

With `linear-ce.patch` applied:

```bash
/opt/torchtitan-venv/bin/python3.12 -m pytest -q gift/tests/unit_tests/test_fusion_losses.py
/opt/torchtitan-venv/bin/python3.12 -m pytest -q gift/tests/unit_tests/test_chunked_loss_fsdp_protocol.py gift/tests/unit_tests/test_aux_loss_owner.py gift/tests/unit_tests/test_monitoring_loss_chunks.py
/opt/torchtitan-venv/bin/python3.12 runs/four-fusion/artifacts/raw/loss-checks/gfsdp_single_device.py
```

- Final focused suite: **17 passed**, no skipped tests, 9.71 s
  (`pytest-linear-final2.log`). Eager and `fullgraph=True` compiled paths execute.
  Loss/component totals, every dH/dW element, packed MTP boundaries, state and
  parameter identity, logits-hook fallback, eval and shape probes pass.
- Existing flag/lifecycle/monitoring checks: **31 passed**, 5.73 s
  (`protocol-linear.log`), including borrowed flags and exception cleanup.
- Matched GFSDP head control/candidate, each with 32 microbatches and two outer
  chunks: eager and compiled pass, **64 native FLCE calls** each
  (`gfsdp-matched-final.log`). Both use FP32 master parameters, BF16 unsharded
  head compute, FP32 reduction/accumulation, own head gradient buffer, divide
  factor 1.0 and unchanged MTP1 weight 0.3. Maximum absolute dH difference:
  `1.9073486328125e-6`; accumulated dW: `2.384185791015625e-7`. Exact sharded
  parameter identity, state-dict keys and post-call flags pass.

After reversing linear and applying `ce.patch`, the same focused test command
passed **17 tests**, none skipped, 6.79 s (`pytest-ce.log`), including its actual
compiled native loss/head path. The same GFSDP harness passed both eager and
compiled with **128 selected CE calls** each (`gfsdp-ce.log`), with the same
maximum dH/dW differences. The matched GFSDP fixture is `H2/B2/S8/D32/V520`;
the focused dense gradient fixture uses 32 rows, D32/V517, seed 17. These scoped
fixtures do not establish full crop-size or eight-rank behavior.

The mandatory large-offset regression uses **V128256**, three BF16 rows at
`2**28`, labels `[5,-100,128255]`, eager and compiled. Stable loss is
`23.52356719970703` for two valid rows, approximately `2*log(V)`. Elementwise
BF16 gradients, including incoming `0.3/17`, match centered expected values
exactly, and ignored rows are zero. The unchanged donor returns loss **0**,
non-target gradients **1** and target gradients **0**: the regression exposes
its defect in both modes. Additional checks cover partial vocabulary tiles,
mixed/all-ignored rows, negative nonunit CE upstream, BF16 head loss/dH/dW with
known 1.0/0.3 scales, and a near-certain target whose true nonzero target gradient
would disappear if BF16 probability were rounded before target subtraction.

Retained failures/limitations of earlier checks:

- `pytest-linear-final.log`: two failures came from a test extension calling
  `.components` as an attribute; corrected to the existing `.components()` API.
  Numerical/identity checks had already passed before that assertion. The final
  suite passes. No failing test or log was removed.
- `gfsdp-single.log` used a BF16-accumulating unwrapped control and therefore
  could not isolate FP32 accumulation; it is superseded by the matched GFSDP
  control in `gfsdp-matched*.log`. Its loose initial dW tolerance establishes no
  final numerical claim.
- Initial `ruff format --check` requested formatting in all three new files;
  formatting was applied, and the final format check passes.
- Initial smoke artifacts were moved from root `artifacts/raw/loss-checks/`
  into the ignored run-owned path above, without deleting their evidence.

Final scoped lint/format commands and whitespace checks pass:

```bash
/opt/torchtitan-venv/bin/ruff check --select F821,F811,F841 --no-cache gift/torchtitan/experiments/four_fusion/linear_ce.py gift/torchtitan/experiments/four_fusion/stable_ce.py gift/tests/unit_tests/test_fusion_losses.py gift/torchtitan/components/loss.py
/opt/torchtitan-venv/bin/ruff format --check gift/torchtitan/experiments/four_fusion/linear_ce.py gift/torchtitan/experiments/four_fusion/stable_ce.py gift/tests/unit_tests/test_fusion_losses.py
git diff --check
git apply --check runs/four-fusion/linear-ce.patch
git apply --check runs/four-fusion/ce.patch
```

Both candidate patches were reversed after their checks. Full-repository lint
and tests were not run; there was no installation, training, TPS run or profiling.

## Final runtime-cost review and remaining gates

Experiment 3 keeps projection, CE, dH and dW GEMMs, while scheduling their
precomputation inside the forward custom op. This moves timing ownership, not
work removal. At crop dimensions its transient BF16 logits are 250.5 MiB per
internal tile, against the ordinary full folded chunk's 1.957 GiB. It adds one
1.957 GiB FP32 full-head dW accumulator/zero-fill and a 0.979 GiB final BF16 dW
that coexists with existing GFSDP buffers. Smaller token GEMMs, repeated row
passes, per-tile dispatch and head-gradient lifetime can lower efficiency or
increase peak memory. There is **one** FP32 accumulator, not one per head.
Known head weights are constexpr kernel inputs; no per-chunk CPU-to-GPU weights
tensor or target-count `.item()` is introduced. Metric loss reductions and
scaling inside the opaque custom-op body still have eager launch costs.

Experiment 4 retains the full BF16 logits and dlogits and the original projection
and backward GEMMs. It uses two FP32 row-stat vectors plus the row-loss vector;
there is no claimed GiB-scale full-FP32-buffer saving. Its remaining hypothesis
is replacing Inductor's vocabulary-wide one-hot-gradient reduction. A row-online
loop can be slower. SUM reduction, wrapper and launch costs remain part of the
head/loss region. Neither implementation adds host-value predicates or
collectives; both can change overlap and live-buffer timing.

The accepted V2 workload remains prefetch32/epoch51200, FP8 trunk/BF16 head,
MTP1, compile/SAC, accumulation32 and two outer chunks. Source-level integration
and the scoped head checks preserve these contracts, but do not establish the
complete workload. Open gates owned by the coordinator: independent final
integration/runtime-cost review; eight-rank GFSDP gather/reduce ordering and
all-rank production activation; full-shape memory fit; fresh monitoring-off
10+10 control/candidate comparisons; matched bounded Perfetto traces; complete
head/loss elapsed time and downstream effect; paired phase memory; and retained
candidate numerical assessment where required. No speedup, production readiness
or candidate retention is claimed here.
