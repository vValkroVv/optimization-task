We support automatically collecting metrics such as
1. High level system metrics such as MFU, average loss, max loss and words per second along with some
2. Memory metrics to measure max VRAM consumption and the number of OOMs
3. Timing metrics to measure data loading bottlenecks

Those metrics can then be visualized in either a TensorBoard or WandDB dashboard

## TensorBoard

To visualize TensorBoard metrics of models trained on a remote server via a local web browser:

1. Make sure `metrics.enable_tensorboard` option is set to true in model training (in the `metrics:` section of the experiment YAML or via `--metrics.enable_tensorboard`).

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

## CUDA cache before optimizer monitoring

On CUDA monitoring steps, unused PyTorch cache is released before optimizer
monitoring to make room for external allocations such as NCCL buffers. Live
tensors are unaffected. Set
`callbacks.monitoring.empty_cuda_cache_before_optimizer: false` to disable it.
Cache release and subsequent allocations may slow down the monitoring step and
the steps that follow.

## Total peak memory

The memory monitor reports memory sizes in GiB (`2**30` bytes): all byte
counters under `memory/*` and the GPU, RAM, swap-capacity (`total`, `used`,
`free`), and disk-capacity fields under `system_metrics/*`. Percentages, event
counts, timings, and I/O counters retain their units, including swap `sin` and
`sout` in bytes. Older logs used decimal GB under `memory/*` and bytes for
system memory sizes.

`memory/non_torch_mem` is `total - free - reserved_current`, using
`mem_get_info()`, and
`memory/total_peak_mem` is `reserved_peak + non_torch_mem`, both in GiB.
This includes external allocations such as NCCL and HybridEP buffers at the
report boundary; the PyTorch reserved peak follows the existing reset cadence.

The built-in logger reports `memory/train/total_peak_mem` and
`memory/train/non_torch_mem`; validation uses `memory/valid/`. These prefixes
avoid duplicate callback and built-in events for the same tag and step. Stdout
uses the same built-in report: `memory:` is total peak GiB and `non_torch:` is
its external-memory component. The startup memory report also uses total peak.
No total-peak percentage is added; existing `max_reserved_pct` and
`max_active_pct` retain their PyTorch-only meaning.

The callback's `/pp_rankN` tags retain the per-pipeline-rank readings. The
built-in TensorBoard series describe the logging rank's device: with PP this
is the first rank of the last stage, except for ZBVZeroBubble, which uses rank
0. Stdout defaults to rank 0 through `GIFT_LOG_RANK`, so compare readings from
the same rank. The callback runs before built-in logging and may sample a
different external-memory value. Neither report measures transient external
peaks between samples; reserved peaks cover the interval since the last
`reset_peak_memory_stats()`, including resets after built-in logging.

## Weights and Biases

Weights and Biases will automatically send metrics to a remote server if you login with `wandb login`

So all you need to do is make sure that `metrics.enable_wandb` is enabled

For example: `./run_train.sh configs/experiments/llama3_8b.yaml --metrics.enable_wandb`.

Note that if both W&B and Tensorboard are enabled then we will prioritize W&B.
