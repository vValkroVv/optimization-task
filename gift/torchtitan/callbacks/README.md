# TorchTitan Callbacks

This package contains lightweight callbacks for the training loop. A callback is
a `Configurable` subscriber to trainer events: it reads state, records metrics,
or changes an explicitly declared part of the current training step.

## Current Model

`CallbackManager` builds callbacks from the run's `callbacks` dictionary.
`Trainer` invokes them at explicit lifecycle points through
`callbacks.emit(Event.x, self, ...)`; `Event` enumerates 14 events.

Callback types define their permissions:

- **`Observer`** reads the step and publishes metrics. Its `ObserverState` is a
  snapshot without a `Trainer` reference. It exposes modules (`model_parts`),
  `parallel_dims`, `device`, `config`, event payload (`input_dict`, `labels`,
  `loss`, `grad_norm`, ...), and copies of optimizer hyperparameters through
  `optimizer_groups()` and `optimizer_state(param)`. Publish through
  `state.record(name, value)` or `state.record_all(mapping)`. The snapshot has
  no optimizer, scheduler, or `metrics_processor` handle for changing training
  state.
- **`Mutator`** changes the step and declares its mutation axes, for example
  `mutates = frozenset({Mutation.lr})` (`batch`, `lr`, `weights`,
  `optimizer_state`, ...). This declaration is conservative: every overridden
  hook counts as writing every declared axis. If some hooks only maintain
  bookkeeping, decorate the writing hooks with `@mutates(Mutation.lr)` instead.
  Only decorated hooks then count as writers in conflict checks and `describe()`;
  the class's mutation set is their union. Mixing the two declaration styles is
  an error. A mutator receives `MutableState`, containing the full `Trainer`
  and event payload. Batch replacement through `state.input_dict` and
  `state.labels` is supported only in `after_dataloader`, where the trainer
  reads those fields back. A `{batch}` mutator without that hook is rejected
  when the handler is built; decorating another hook with
  `@mutates(Mutation.batch)` prevents class creation. A mutator without a
  declaration also fails at class creation.

Every event runs **mutators → observers → post-mutators → aggregators**, with
preset dictionary order inside each group. An `Aggregator` observes the other
callbacks' event results: `TimersMonitor` drains the gigatimer registry they
populate, so it runs last regardless of its key's position. A `PostMutator`
uses observer results: `DynamicBlockLRRescaler` reads `DispatcherSource`'s
`max_vio` after observers have computed it. It runs before aggregators so its
`record()` is included in the same event's drain. Observers see the step before
the post-mutator's change, which becomes visible on the next event.

Two mutators writing the same axis on the same event are rejected when the
handler is built; post-mutators participate in that check. The final
`CallbackManager.describe()` schedule is logged and written to TensorBoard
(`callbacks/schedule`) at `fit_start`. See [Execution Order](#execution-order).

In a `Mutator`, access persistent training objects through `state.trainer`:

- `state.trainer.model_parts`
- `state.trainer.optimizers`
- `state.trainer.metrics_processor`
- `state.trainer.parallel_dims`
- `state.trainer.config`

Use `State` fields for data a callback can replace during an event:

- `state.input_dict`
- `state.labels`
- `state.loss`
- `state.grad_norm`
- `state.global_valid_tokens`
- `state.metrics`

Do not retain a `State` reference after its event: the next event has a different
context. Copy only values needed by future steps, or keep tensors and buffers
owned by the callback.

## Stateful Callbacks

Use `torch.distributed.checkpoint.stateful.Stateful` only for state that must
survive checkpoint save/load to resume training correctly.

A stateful callback must:

1. Implement `state_dict()`.
2. Implement `load_state_dict()`.
3. Return itself from `stateful_objects()` under a stable top-level key.

```python
def stateful_objects(self):
    return {"ema": self}
```

`Trainer` collects these objects before constructing `CheckpointManager`, so
DCP saves them with the ordinary training state.

Logging runtime metrics alone does not make a callback stateful. Forward hooks,
temporary metric buffers, and interval counters usually need no checkpointing
unless recovery correctness depends on them.

## `Observer` or `Mutator`

Inherit `Mutator` only when the callback changes training semantics:

- Replacing `state.input_dict` / `state.labels` (`SeqLengthWarmup`, `Mutation.batch`).
- Changing group learning rates / scheduler `base_lrs`
  (`DynamicBlockLRRescaler`, `Mutation.lr`).
- Scheduling `ResidualScale.alpha` (`ProResWarmup`, `Mutation.residual_scale`).

Other callbacks are observers: monitors, timers, EMA (which writes only its own
shadow weights), and Autometrics. Monitoring sources (`MonitorSource`) are
observers by construction. The contract is checked by a bitwise comparison of
training with observers enabled and disabled.

## Current Callbacks

Package layout: `base.py` defines `Callback` / `CallbackHandler`; `metrics/`
contains the monitoring engine; `sources/` contains `MonitoringCallback` sources;
`builtin/` contains callbacks; `manager.py` is the composition root that builds
configured owners. Paths below are relative to `torchtitan/callbacks/`.

| Callback | File | Stateful | Type | Main events | Purpose |
| --- | --- | --- | --- | --- | --- |
| `SystemMetricsMonitor` | `builtin/system_monitor.py` | No | Observer | `batch_end` | Logs allocator counters from `device_module.memory_stats()` in absolute GiB (`memory/*`, `memory_log_interval` cadence), host CPU/RAM/swap/disk/network statistics through psutil once per node, and NVML GPU statistics once per rank's device (`system_metrics/*`, `host_log_interval` cadence). An interval of 0 disables its metric family. |
| `MonitoringCallback` | `builtin/monitoring.py` | No | Observer | `fit_start`, `batch_start`, `microbatch_start`, `before_dataloader`, `after_dataloader`, `after_forward`, `before_optimizer_step`, `after_optimizer_step`, `validation_start`, `validation_end`, `batch_end`, `close` | Unified data/model/trainer/optimizer monitoring through sources and metric pipelines. |
| `DispatcherSource` | `sources/dispatcher_source.py` | No | Observer | Through `MonitoringCallback` | Dispatcher load: local/global tokens per expert, `max_vio`, and `routed_tokens`. Dispatchers are not `nn.Module`s, so `dispatch()` cannot use a module forward hook. Instead, `fit_start` assigns `dispatch_monitor` and `close` removes it. Captures every batch, because `max_vio` averages the entire logging window and feeds `DynamicBlockLRRescaler` every step. It is separate from `ModellingSource`, whose hooks run only on logged steps. |
| `EMA` | `builtin/ema.py` | Yes | Observer | `after_optimizer_step` | Maintains DCP-compatible EMA shadow model weights. |
| `SeqLengthWarmup` | `builtin/seq_length_warmup.py` | No | Mutator `{batch}` | `fit_start`, `after_dataloader`, `batch_end` | Truncates batch tensors to a step-dependent sequence length before valid-token counting and forward execution. |
| `ProResWarmup` | `builtin/prores_warmup.py` | No | Mutator `{residual_scale}` | `fit_start`, `batch_start`, `after_checkpoint_save` | Sets each `ResidualScale.alpha` from its module-configured step schedule; warns when Autometrics converts a save with `alpha < 1` (trainer HF saves are checked before the run). |
| `UpdateDynamicGate` | `builtin/update_dynamic_gate.py` | No | Mutator `{routing}` | `after_optimizer_step` | Calls `switch2uniform(step)` on modules supporting it; reserved for dynamic-gate modules and currently a no-op. |

## MonitoringCallback

`MonitoringCallback` is the standard mechanism for training runtime statistics.
It replaces specialized monitor callbacks with one path:

```text
MonitorSource -> Sample -> MetricPipeline -> ParallelReducer -> MetricStore -> MetricEmitter
```

The main abstractions are:

- `MonitorSource`: knows where to capture a `Sample` in the trainer lifecycle,
  without implementing complex statistics itself.
- `Sample`: a monitored tensor/scalar, name, source type, axis names, and list
  of `MetricConfig` entries.
- `MetricConfig`: one metric's ordered `steps` and `dtensor_mode`.
- `MetricStepConfig`: a `stat` or `reduce` pipeline step; order is preserved.
- `StatOpConfig`: an atomic statistic such as `mean`, `std`, `min`, `max`,
  `abs_max`, `rms`, `count`, `finite_ratio`, or `bincount`.
- `ReduceStageConfig`: one distributed reduction over a mesh group.
- `ParallelReducer`: executes online reductions over `ParallelDims` groups.
- `MetricStore`: optionally saves metrics to CPU/disk for offline processing.
- `MetricEmitter`: writes final scalar metrics to `metrics_processor`.

### `include_patterns` / `exclude_patterns` Filters

`modelling`, `optimizer`, and `dispatcher` support FQN filters
(`FqnFilterConfig`). Patterns are regexes matching **whole dotted segments** of
canonical FQNs, with AC/FSDP/compile wrappers removed. `layers.1` selects
`layers.1.*`, including `mtp.layers.1.*`, but not `layers.10`. `^layers\.1`
restricts the match to the top level and excludes that MTP head. `wq` does not
match `wq_norm`; use `.*proj` to match a substring within a segment.

An empty include list selects everything; otherwise only matches remain.
Excludes apply afterwards and win. Filters resolve targets at `fit_start`, so
excluded modules receive no hooks. They apply to every FQN a source uses,
including modelling MoE units: `include=['attention']` also disables `moe/*`
metric families. A nonempty filter logs retained counts at `fit_start`
("keep 97 of 134 modules", "9 of 14 MoE units"). Unmatched patterns warn, so a
typo is distinguishable from an intentional selection. `data` and `trainer`
have no FQN filters.

### Ownership and Data Flow

Sources are observers held and drained by one `MonitoringCallback`:

```text
Callback
├── Observer  receives ObserverState (snapshot; publishes through record())
│   ├── MonitoringCallback ── owns and drains MonitorSource instances
│   │     └── MonitorSource (Configurable + Observer)
│   │           ├── ModellingSource    module hooks; MoE router; expert weights
│   │           ├── OptimizerSource    weights, gradients, param groups, moments
│   │           ├── TrainerSource      loss, learning rate, throughput, accuracy
│   │           ├── DispatcherSource   dispatch_monitor slot on the dispatcher
│   │           └── DataSource         dataloader timing and batch statistics
│   └── TimersMonitor, SystemMetricsMonitor, MtpAccLenMonitor, EMA, Autometrics
└── Mutator   receives MutableState (the trainer) and declares mutation axes
      ├── DynamicBlockLRRescaler  {lr} (PostMutator)
      ├── SeqLengthWarmup         {batch}
      └── UpdateDynamicGate       {routing}
```

One value's path from capture to TensorBoard:

```text
Sample(value, axes, metrics)          source captures the value
  → MetricPipeline.compute_stats      statistics; collectives only in the
                                      DTensor partial_stats path
  → MetricRecord(value, pending_stages, post_stats)
  → finalize_pending_reductions       one collective per (group, op) per step
  → apply_post_stats                  rank-local; no collectives
  → MetricEmitter.emit                one stack, one host copy,
                                      one gather across PP stages
  → metrics_processor.log_extra_metrics → TensorBoard
```

### Sources

Sources and the `MonitorSource` protocol live in `sources/`; the monitoring
engine lives in `metrics/`.

`ModellingSource` hooks `nn.Module`s in `state.model_parts`. All model, router,
and logits forward hooks check `MonitorSource._is_recompute` to skip activation
checkpointing replays during backward. Otherwise sum tags would double, and
metrics without reduction stages could fail when combining duplicate captures.
It covers:

- Forward activations (module outputs).
- Input activations with `capture_inputs`, tagged
  `activations_{stat}/{fqn}_input.{i}`. Norm inputs expose the residual stream:
  `ffn_norm` receives the residual after attention.
- Backward gradients of module outputs.
- Tuple outputs with `output="all_tensors"`.
- MoE router balance and expert-weight statistics through the `moe` section
  (`MoECaptureConfig`).

Router statistics come from a forward hook on `moe.router` and accumulate
**per batch**: `tokens_per_expert` sums across microbatches;
`scores_per_expert` averages across forward passes (llm-f divides its accumulator
by `_upd_counter`). Configured metrics run once at `batch_end` on those vectors.
The summed histogram's violation is less noisy than averaging microbatch
violations, and is a different quantity.

Models expose no monitoring-specific API. `sources/moe_units.py` finds MoE layers
through `named_modules()` and `isinstance(module, MoE)`, then reads their router,
experts, and dispatcher. Names use `canonical_fqn`, without AC/FSDP/compile
wrapper segments. The same traversal handles MoE in MTP heads. `MoE`'s
`tokens_per_expert_E` is training state for auxiliary-loss-free bias, not a metric.

Each source's declarative `metrics` list (`MetricConfig`) defines its statistics;
see [Metric Pipelines and Reductions](#metric-pipelines-and-reductions).
YAML presets configure it in `configs/presets/callbacks/monitored.yaml`.
Every `ModellingSource` metric must declare reduction stages: hooks run per
microbatch, and gradient accumulation yields multiple captures per batch. The
first reduction combines them; `Config.__post_init__` rejects a metric without
one. Defaults (`amax`, `amean`, `l2norm`) reduce over `dp` and `cp`, whose ranks
hold different batches and sequence slices respectively. `OptimizerSource`
gradients, weights, and moments are DTensors; `partial_stats` reduces their shard
axes, so explicit stages are optional there. For example:
`ReduceConfig(stages=[ReduceStageConfig(group="dp", op="max"),
ReduceStageConfig(group="cp", op="max")])`. These groups do not cross the PP
mesh, avoiding collectives between different pipeline stages' hooks.

`ModellingSource` uses the same metric list to produce
`activations_{metric.name}/...` families. Default names `amax`, `amean`
(`abs().mean()`), and `l2norm` preserve historical dashboards. `OptimizerSource`
uses fixed `l2_norm/...` and `amax/...` layouts.

`OptimizerSource` reads parameters, gradients, optimizer parameter groups, and
optimizer state around the optimizer step. It provides gradient, weight,
learning-rate, Adam-moment, and related optimizer statistics.

`TrainerSource` reads `ObserverState`, without a trainer reference. It provides
progress counters, batch geometry, valid tokens, and top-k next-token accuracy.
`topk_accuracy=[1,5,20]` hooks the logits module selected by
`logits_module_pattern`. Chunked loss invokes it per chunk; under loss
parallelism, sharded top-k results merge over TP, and hits/valid counts sum over
DP and CP.

`DataSource` measures `next(data_iterator)` wall time between `before_dataloader`
and `after_dataloader`. It can also collect supervised/ignored label-token
statistics. Values are local to the logging rank.

Try existing sources first. Add a source only for a new lifecycle point or a
fundamentally different way to access data.

### Modeling Rule

Monitoring must not add Python logging, distributed collectives, or
callback-specific APIs inside model files.

To monitor an intermediate `forward` value, expose the operation as a small
`nn.Module`, then select that module by name in `ModellingSource`. Replace:

```python
h = some_intermediate_expression(x)
```

with:

```python
h = self.some_intermediate_module(x)
```

Hooks on `some_intermediate_module` then collect statistics without changing
monitoring infrastructure. Exceptions must represent actual modeling semantics
that exist independently of monitoring, and still use ordinary `nn.Module`s.

The module rule covers activations and their gradients. Distinguish three cases:

1. **Expressions within one forward** (`h = a * b + c`): hooks see module inputs
   and outputs, not intermediate values; use the small-module approach above.
2. **Routing and dispatch quantities** (tokens per expert, histograms, `max_vio`):
   these originate internally rather than at module boundaries. Use the
   dispatcher's `dispatch_monitor` slot and router hook.
3. **Step-level quantities** (loss and components, clipped gradient norms,
   learning rates, timings): use `TrainerSource` / `OptimizerSource`.

### Metric Pipelines and Reductions

Simple metrics can use the `pipeline + reduce` shorthand:

```python
MetricConfig(
    name="activation_abs_max",
    pipeline=[StatOpConfig(op="abs_max")],
    reduce=ReduceConfig(group="tp", op="max"),
)
```

Use ordered `steps` to interleave statistics and distributed reductions:

```python
MetricConfig(
    name="mean_then_dp_mean_then_sp_sum",
    steps=[
        MetricStepConfig(kind="stat", stat=StatOpConfig(op="mean")),
        MetricStepConfig(kind="reduce", reduce=ReduceStageConfig(group="dp", op="mean")),
        MetricStepConfig(kind="reduce", reduce=ReduceStageConfig(group="sp", op="sum")),
    ],
)
```

Canonical groups match `ParallelDims` mesh names: `tp`, `ep`, `cp`, `fsdp`,
`efsdp`, `batch`, `loss`, and `dp_replicate`. Aliases are:

- `sp` and `sequence`: `tp`, because sequence parallelism uses the TP mesh.
- `dp` and `data`: `batch`.
- `global`: `world`.

`pp` is not a reduction group, and `global` / `world` are forbidden with pipeline
parallelism. Records exist per model part, so stages can have different value
counts; `ParallelReducer` raises `ValueError`. Metrics from different stages
reach the logging stage through `gather_metrics_over_pp`, not a reduction
collective.

`ReduceStageConfig(group=("dp_replicate", "fsdp"), op="mean")` reduces over a
multidimensional mesh if `ParallelDims` provides that combination.

Keep expensive collectives explicit. Express an intermediate reduction,
normalization, and further reduction through ordered `steps`, or add a named
statistic/metric function beside the pipeline implementation. Do not hide this
work in a source.

### DTensor and Debug Modes

`MetricConfig.dtensor_mode` controls DTensor handling:

- `local` reads only the shard through `DTensor.to_local()`. It is cheap and
  describes only that shard.
- `partial_stats` is the default. For supported global statistics without `dim`,
  it computes sufficient local statistics and reduces over sharded DTensor mesh
  axes, including `_StridedShard` with FSDP2+TP. `Partial` placements first reduce
  to `Replicate`, because their logical value is the sum of local tensors.
  Replicated placements are not reduced, preventing overcounting. Supported
  operations are `sum`, `mean`, `std`, `var`, `rms`, `count`, `min`, `max`,
  `abs_max`, `norm` (including `p=inf`), `nan_count`, `inf_count`,
  `nonzero_count`, `finite_ratio`, and `zero_ratio`.
- `full_tensor` materializes the logical tensor with `DTensor.full_tensor()`
  before applying the statistic. This debug mode is useful for correctness
  checks and axis-wise statistics, but can consume substantial memory and
  communication. Avoid frequent production use on large activations or
  optimizer states.

Only the metrics rank (the first rank of the last PP stage) writes to
TensorBoard. `gather_metrics_over_pp` in `reducers.py` gathers metrics from other
stages to it. The `MonitoringCallback` emitter and MoE monitors invoke the gather
synchronously across the PP mesh on every logged step. Without PP it is a no-op.

`--callbacks.monitoring.debug_full_tensor` computes each DTensor
`partial_stats` metric through `full_tensor` as well, logging it as
`debug_full/{tag}` beside the production tag. Agreement within floating-point
error validates the distributed statistics. Use short debug runs only.

Axis-wise DTensor operations such as `mean(dim="hidden")` are not yet
placement-aware in `partial_stats`. They and unsupported operations fall back to
`full_tensor`, rather than returning a shard-local result silently. If that is
too expensive, express the sufficient-statistics and reduction pipeline explicitly.

## Adding a Callback

1. Create `torchtitan/callbacks/builtin/<name>.py` with a class inheriting
   `Configurable` and `Observer`, or `Mutator` with declared mutations.
2. Declare its torch-free schema in
   `torchtitan/config/schema/callbacks/builtin/<name>.py`, using
   `@config(type="<name>")`, `CallbackConfig`, and the runtime `_owner_spec`.
   Expose it as the runtime class's `Config` alias.
3. Implement only the needed event methods, accepting `ObserverState` or
   `MutableState`. Publish through `state.record`; tensor values are copied to
   the host by the emitter together with the rest of the step's metrics.
4. Implement `stateful_objects()` only for state that DCP must save.
5. Add an instance key and `type` to the preset's `callbacks` mapping. No
   `CallbackManager` edits are needed. Declare constructor dependencies in
   `Config.requires` if the callback needs another built object.
6. Export the runtime class from `callbacks/builtin/__init__.py` and
   `callbacks/__init__.py`.
7. Pin the new `type` in `COMPONENT_TYPES["callback"]` in
   `tests/unit_tests/test_config_torch_free.py`. Add focused tests for its
   validation and event behavior; update any affected experiment snapshots.

Example schema (`config/schema/callbacks/builtin/my_callback.py`):

```python
from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.callbacks.base import CallbackConfig


@config(type="my_callback")
class MyCallbackConfig(CallbackConfig):
    _owner_spec: ClassVar[str] = "torchtitan.callbacks.builtin.my_callback:MyCallback"
    log_interval: int = 10

    def __post_init__(self) -> None:
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")
```

Runtime owner (`callbacks/builtin/my_callback.py`):

```python
from typing import TypeAlias

from torchtitan.callbacks.base import Observer, ObserverState
from torchtitan.config import Configurable
from torchtitan.config.schema.callbacks.builtin.my_callback import MyCallbackConfig


class MyCallback(Configurable, Observer):
    Config: TypeAlias = MyCallbackConfig

    def __init__(self, config: Config) -> None:
        self.config = config

    def batch_end(self, state: ObserverState) -> None:
        if state.step % self.config.log_interval == 0:
            state.record("my/step", state.step)
```

After adding these modules, enable the callback in an experiment. Save this
file as `configs/experiments/llama3_callback.yaml`:

```yaml
# Llama 3 debug run with a custom step observer.
# torchrun --nproc_per_node=1 -m torchtitan.train configs/experiments/llama3_callback.yaml
config_version: 1
imports:
  - llama3_debug.yaml
callbacks:
  progress: {type: my_callback, log_interval: 5}
```

The instance key `progress` determines the override path; `my_callback` selects
the schema and runtime class. No new trainer field or loader branch is needed:

```bash
python scripts/config_check.py configs/experiments/llama3_callback.yaml \
    --callbacks.progress.log_interval 2 --print-resolved
NGPU=1 ./run_train.sh configs/experiments/llama3_callback.yaml \
    --callbacks.progress.log_interval 2
```

For an existing component, adding a parameter follows the same path: declare
it on the owning schema, validate it, read it in the runtime owner, and set it
at that component's YAML/CLI path. Decoding exposes fields automatically;
runtime behavior changes only when code consumes the value. Preserve inherited
validation by calling the parent explicitly when overriding `__post_init__`
or `validate` (see [Extending Trainer.Config](/docs/extension.md#extending-trainerconfig)).
Test the default, a non-default value that changes behavior, and invalid input;
pin a new experiment with the [golden snapshot command](/docs/config.md#golden-snapshots).

Rules:

- There is no `enable` flag: a callback is built exactly when the preset lists
  its key in `callbacks`. Absence of that key disables it.
- Observers access persistent state through the snapshot (`state.model_parts`,
  `state.optimizer_groups()`). Only mutators have `state.trainer`.
- Mutators replace event-local values through `MutableState` fields where
  available, rather than writing those values directly to the trainer.
- Keep monitoring communication in callbacks or trainer utilities, outside
  model modules.
- Avoid Python instrumentation inside `torch.compile` regions.
- Document ordering dependencies and register callbacks accordingly; never
  silently rely on another callback's position.

## Adding an Event

Add an event only for a real lifecycle boundary that existing events cannot
express clearly.

1. Add an `Event` member and a matching no-op `Callback` method in `base.py`.
2. Emit `self.callbacks.emit(Event.<event>, self, ...)` at the exact point in
   `Trainer`. A test compares emitted events with the enum.
3. Pass only payload fields valid at that point. Add new observer fields to
   `MutableState` and `ObserverState.from_mutable` when needed.
4. Document which fields mutators may change there.
5. Name the event after trainer semantics, independently of any particular
   callback.

Names such as `after_dataloader`, `after_forward`, and `before_optimizer_step`
describe stable training-loop boundaries. Avoid callback-specific events such
as `before_ema_update`; use existing events and callback order where possible.

## Execution Order

`CallbackHandler` builds the same ordering once for all events:

1. All `Mutator`s in `callbacks` dictionary order.
2. All `Observer`s in dictionary order.
3. All `PostMutator`s, whose changes depend on observer results. Currently,
   `DynamicBlockLRRescaler` reads `DispatcherSource.max_vio_by_unit`: this reads
   already computed data after observers, rather than triggering computation.
   It precedes aggregators so its `record()` is drained in the same event.
   The old dependency relied on a "first reader computes" cache; type now
   expresses the order.
4. All `Aggregator`s, which observe the other callbacks' event results rather
   than the step itself. Currently this is `TimersMonitor`.
   `build_all_timers_metrics` clears each timer's history when reading it, so
   a callback running after the drain would fall into the next interval.
   Previously this depended on listing `timers` last; its type now makes it
   last regardless of dictionary position.
5. One `ObserverState` snapshot is created after the first mutator group.
   Every observer and aggregator therefore sees that group's changes.
   Post-mutator changes are absent from the snapshot: observers see the step
   used by the event, and see the post-mutator's change on the next event.

Order is defined by two visible properties: callback type and preset key
position. There are no numeric priorities, phases, or explicit "X after Y"
relations. Key order matters only within a group, whose members must be
interchangeable by contract: observers do not consume each other's results,
and neither do aggregators. `Stateful` does not affect execution order;
checkpointing and scheduling are independent.

These guarantees are enforced:

- `describe()` prints the schedule at `fit_start` and writes
  `callbacks/schedule` to TensorBoard for every run.
- `_check_mutation_conflicts` rejects two mutators declaring the same
  `Mutation` axis on the same event.
- `Config.requires` declares build dependencies. `CallbackManager._dependency`
  resolves them **by constructor parameter type**. A dependency listed later
  in the dictionary produces an explicit error instead of `None`.
- A callback that must run last declares `Aggregator`. A test lists `timers`
  first and checks that it is scheduled last.

## Automatic evaluation of saved checkpoints

Add this block to an experiment YAML using the `fsdp2` backend with HF export
enabled. `autometrics` submits a separate evaluation job; the training process
does not run the evaluation itself. Set your login and the queue, region,
workspace, and image for your environment. The top-level `hf_assets_path` must
point to a directory containing tokenizer files, and the model adapter must
support a loadable HF export.

```yaml
checkpoint:
  enable: true
  interval: 1000
  keep_latest_k: 0  # Keep checkpoints available while PAM jobs are queued or running.
  save_in_hf: true
callbacks:
  evaluation:
    type: autometrics
    user: your_login
    exp_name: pretrain_evaluation
    submission_type: cloud  # For Slurm, use slurm and its submission settings.
    region: SR008
    queue_name: your_queue
    workspace_name: your_workspace
    base_image: your_registry/evaluation:tag
    interval: 1000  # Must be a multiple of checkpoint.interval.
    tasks:
      - name: hellaswag
        num_fewshot: 0
    inference_tp: 1
    inference_dp: 8
    n_workers: 1
    n_gpus: 8  # Must equal inference_tp * inference_dp.
```

The name `evaluation` is the callback instance key, so change its interval with
`--callbacks.evaluation.interval 2000`. To remove it from an inherited set,
use `callbacks: {evaluation: null}`; the other callbacks remain enabled.
For `gfsdp2`, PAM receives the DCP checkpoint and uses `gfsdp2_mapper`; this
workflow does not require HF export. See the available fields in
[`AutometricsConfig`](../config/schema/callbacks/builtin/autometrics.py).

## Мониторинг и torch.compile

С `compile.components=["model"]` блоки скомпилированы, а module-хуки внутри
скомпилированного блока не вызываются. Поэтому на лог-шагах (раз в
`log_interval`) MonitoringCallback запускает блоки eager — только если какой-то
источник читает модель через хуки: `modelling` с любым `capture_*`
(включая `capture_swiglu_clips`) или с роутерными MoE-метриками
(`token_metrics`, `score_metrics`, `per_expert_tokens`). Источник, который после шага читает одни веса
(`moe.expert_weights`, `bias_metrics`), хуков не вешает и окно не открывает;
об открытом окне предупреждает лог на старте. **Throughput и пик памяти на
eager-шагах отличаются от скомпилированных** — не пугайтесь провала tps и
другого профиля памяти раз в `log_interval` шагов; `log_interval=1` означает
обучение целиком без компила. Диспетчер-монитор (max_vio и маски окон)
оттрейсен внутрь блоков и работает каждый шаг без eager-окна. Метадата-проба
PP-расписания (первый шаг процесса, forward+backward на `torch.empty`) идёт
eager и не попадает ни в счётчики диспетчера, ни в хуки
(`in_metadata_probe`). SeqLengthWarmup компил не трогает: каждая новая длина
— рекомпиляция блока, поэтому расписание длин должно быть ступенчатым, а не
непрерывным.
