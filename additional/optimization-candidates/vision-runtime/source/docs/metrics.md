We support automatically collecting metrics such as
1. High level system metrics such as MFU, average loss, max loss and words per second along with some
2. Memory metrics to measure max VRAM consumption and the number of OOMs
3. Timing metrics to measure data loading bottlenecks

Those metrics can then be visualized in either a TensorBoard or WandDB dashboard

## TensorBoard

To visualize TensorBoard metrics of models trained on a remote server via a local web browser:

1. Make sure `metrics.enable_tensorboard` option is set to true in model training (either from a config_registry function or from CLI).

2. Set up SSH tunneling, by running the following from local CLI
```
ssh -L 6006:127.0.0.1:6006 [username]@[hostname]
```

3. Inside the SSH tunnel that logged into the remote server, go to the torchtitan repo, and start the TensorBoard backend
```
tensorboard --logdir=./outputs/tb
```

4. In the local web browser, go to the URL it provides OR to http://localhost:6006/.

### Event file rotation

The event file is rotated by size: once the current file exceeds
`metrics.tb_rotate_mib` (default 64 MiB), the writer closes it and the next
write opens a fresh file, so a `tb/` directory holds a series of
`events.out.tfevents.*` files instead of one ever-growing one. TensorBoard
reads such a directory natively. Closed files never change again, which lets
incremental transfer tools (rsync, rclone) re-upload only the newest file.
Set `metrics.tb_rotate_mib` to 0 to keep a single file per run.

Note the default only triggers on runs with a large scalar volume (e.g. the
monitoring callbacks with per-layer and per-expert series); a plain run writes
a few hundred bytes per log call and stays in its first file.

## Weights and Biases

Weights and Biases will automatically send metrics to a remote server if you login with `wandb login`

So all you need to do is make sure that `metrics.enable_wandb` is enabled

For an example you can inspect the Llama 3 [config_registry.py](../torchtitan/models/llama3/config_registry.py)

Note that if both W&B and Tensorboard are enabled then we will prioritize W&B.


## Per-rank allocator statistics

Set `metrics.save_rank_memory_stats=true` to write line-buffered JSONL to
`<dump_folder>/memory/rankN.jsonl` on every global rank, independently of
TensorBoard/W&B rank selection. With `metrics.log_freq=1`, each optimizer
update produces one row with its step, rank, hostname, timestamp, interval
timing, allocator current/peak allocated and reserved bytes, capacity and
allocation retry/OOM counters. Peaks reset after each metrics log; the first
interval includes setup since the metrics processor was constructed.
Fault-tolerant replicas use separate `memory/replica_N` directories.

These are PyTorch allocator statistics. External CUDA allocations, including
some communication allocations, are not included. No allocator event history,
extra collective or CUDA synchronization is added by this output.

For lightweight CPU/CUDA traces, use `profiler.record_shapes=false`,
`profile_memory=false` and `enable_memory_snapshot=false`. Memory timelines
require `record_shapes=true` and include stacks. A fresh run with
`profile_freq=10`, `profiler_warmup=1`, `profiler_active=1` and
`profiler_repeat=3` records updates 10, 20 and 30; warmup is 9, 19 and 29.
`profile_ranks: null` records every rank. Trace export occurs after the
recorded update and can affect the following step's timing; exclude those
intervals from steady-state comparisons.


### Input throughput compatible with Composer DataSpec

When trainer throughput is enabled, `throughput/input_tokens_per_sec`
counts the sum of input tensor positions over data-parallel ranks and accumulated
microbatches, including padding and positions with ignored labels.
`throughput/device/input_tokens_per_sec` divides that global rate by world size.
Both use the configured wall-clock throughput window; corresponding
`throughput/step/input_tokens_per_sec` and
`throughput/device/step/input_tokens_per_sec` measure one update.
Existing `throughput/tokens_per_sec` remains the supervised-label rate.
The input count shares the loss-count collective, without another all-reduce.
Its cumulative counter is run-local; starting or resuming training resets the
throughput window, so checkpoint totals are not mixed with a new elapsed time.
Compare runs with the same window, packing and token accounting, excluding
startup and profiled steps from steady performance results.
