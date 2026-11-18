# Capacity group-counter change: pre-implementation review

## Baseline and affected contract

- Task root revision: `e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f`.
- Baseline `capacity.py` SHA256: `2f034b072fdaf02ac844936366134fafcd5e815588ce750b0dc7dfcc50467120`.
- `select_capacity_routes` builds int64 expert counts (including its inactive
  sentinel bucket), computes the local expert-over-capacity predicate from
  `counts[:num_experts]`, then for group gating sums the E expert counters
  across EP and groups the global vector by destination. The final condition
  remains `local_expert_overload AND any(global_group_load > capacity * E)`.
- `BaseEPTokenDispatcher._drop_routes` consumes the returned keep mask and
  invalidates dropped IDs; local, AllToAll, DeepEP and HybridEP dispatchers
  share this path. The mask affects routing and backward through the existing
  dispatcher. It adds no persistent state or checkpoint format.
- The functional `funcol.all_reduce(..., "sum", ep_group)` must remain present
  for every source, including empty sources, so compile/SAC retain the same
  collective participation and order.

## Chosen bounded change

Keep the local E-element int64 expert histogram. Sum it locally into one
int64 load per destination group, then functional-SUM those `ep_size` values.
For E=128/EP=8 this changes the collective input from 128 int64 counters
(1,024 bytes) to 8 counters (64 bytes). The predicate, capacity arithmetic,
selection, tie order, masks, route IDs, scores and collective participants are
unchanged. The existing design note and screening JSON establish this as an
exact algebraic reordering and explicitly warn that reduced payload does not
prove a latency or step-time gain.

## Expected runtime costs by path

- **Computation and launches:** the same integer group sum is required. It
  moves before the collective; the code still expresses one `sum(1)` over
  `(EP, E / EP)`. This can move a reduction kernel before communication and
  change fusion/launch scheduling. No kernel/launch count or latency claim is
  made without target measurements.
- **Memory:** local expert counts remain live and unchanged for the local
  predicate and route selection. The group-load result still has EP int64
  values; the all-reduce input/output metadata is smaller. No persistent
  allocation or cache is introduced. Peak live temporary memory is expected
  not to increase materially; GPU measurement remains the evidence gate.
- **Host/device interaction:** no new scalar read, `.item()`, transfer, or
  synchronization is added. The existing host predicate read and conditional
  selection remain at the same frequency and after the collective.
- **Communication:** one functional EP SUM per grouped-capacity call remains
  in the same order with all EP ranks, including empty sources. Only payload
  changes from E to EP counters. The 1 KiB to 64 B figures are input tensor
  sizes, not wire-byte or latency claims.
- **Compile/SAC and lifetime:** retain the same functional collective operator
  and call path; only its input shape changes. No new state crosses forward,
  backward recomputation, optimizer, evaluation or checkpoint boundaries.

## Acceptance before handoff

1. Focused bounded-int oracle compares local and global group loads, the exact
   two-part predicate and final keep masks; a separate E=128/EP=8 assertion
   fails against the old payload shape.
2. Existing token-drop/Gloo transport and gradient checks, CPU fullgraph
   compile checks, and real Gloo compile/SAC checkpoint check pass with CUDA
   hidden. Existing empty and unequal source cases remain covered.
3. Run scoped `ruff check --select F821,F811,F841 --no-cache` on changed Python
   files and preserve the result.
4. Keep the original-order source snapshot/patch reproducible. Independent
   integration/runtime-cost review is assigned after this source handoff.

## Explicit remaining gates

CPU/Gloo and source checks cannot establish NCCL behavior, launch/fusion costs,
GPU performance, the native 10+10 pair, or the final candidate trace. Those
remain assigned GPU gates; this review does not treat the byte reduction as a
speedup.

## Post-implementation reconciliation

The final source keeps `counts[:num_experts]` unchanged for the local overload
predicate and selection. It computes the same `(EP, E / EP)` local group sum
before the functional collective, then applies the original
`needs_drop = local_overload AND any(group_loads > capacity * E)` predicate.
No other production path or persistent state changed. The expected costs above
still apply: fewer collective input bytes, the same number/order of collectives
and host predicate reads, with possible changes to reduction launch/fusion
placement. No NCCL timing or whole-training performance result is claimed in
this CPU handoff.
