# gift_scripts

Submit and run torchtitan jobs on ML Space or Slurm. Job submission is handled
by `job_submitter`.

## Setup (once)

```bash
git submodule update --init --recursive
```

## 1. Submit a training job

Edit the `TrainJobGIFT` arguments in `submit_train_job.py`, then run it from the
target cloud notebook or Slurm login node:

```bash
python gift_scripts/submit_train_job.py
```

`TrainJobGIFT` selects the cloud or Slurm backend from the current environment
and resolves the matching Docker/SQSH image for the detected cluster from
`gift_docker/images.json`, the single source of truth for pinned torchtitan
images. The stable `gift_scripts/train.py`
entrypoint runs under the `pytorch2` launcher, which starts one process per GPU
and provides `RANK`, `WORLD_SIZE`, and `LOCAL_RANK` to TorchTitan. It also sets
the working directory to the checkout root so relative config paths work in
both backends. The run is an experiment YAML under `configs/experiments/`
(`config_file=`); before submission `TrainJobGIFT` runs the checkout's
`scripts/config_check.py` over it together with `script_args`, the same
torch-free pre-submit check the CI lint stage runs, so an invalid config fails
before cluster resources are requested. The container repeats the check and
performs the strict typed decode before anything is built.

Dot-path overrides go in `script_args`, for example:

```python
script_args="--training.steps 100 --metrics.log_freq 10",
```

To override auto-detection, pass `region`, `submission_type`, `workspace_name`,
or `base_image` to `TrainJobGIFT` in the submit script.

## 2. Run PAM metrics

GIFT uses the same `pretrain-auto-metrics` pipeline and PAM images as
LLM Foundry. Initialize the recursive submodules, then edit and run:

```bash
python gift_scripts/submit_autometrics_job.py
```

`checkpoint_dir` must be a complete Hugging Face checkpoint containing
`config.json` and `*.safetensors`. Native TorchTitan DCP checkpoints are not
accepted by PAM. Enable `checkpoint.last_save_in_hf` for the final GIFT save and
make sure the output directory also contains the matching config and tokenizer
assets from `hf_assets_path`.

The PAM config selects the inference layout and metric shooters (`lm_eval`, OBM,
DREDD, and others). Results are written under
`<checkpoint_dir>/artifacts/metrics`; logs go to
`<checkpoint_dir>/artifacts/logs`. `mode="inference"` starts serving without the
metric stage, while the default `mode="autometrics"` runs the configured metrics.

To measure an already running OpenAI-compatible or IFT endpoint, use the
external API job:

```bash
python gift_scripts/submit_external_api_metrics_job.py
```

This path is Cloud-only and does not require a local checkpoint. API credentials
and evaluator settings belong in the PAM YAML, not in the submit script.

## 3. Submit an infinite job

An *infinite job* allocates a container that stays up (`sleep infinity`), so you
can SSH in and iterate without re-queuing on every run.

Edit the config block at the top of `submit_infinite_job.py` (login,
`N_WORKERS`, queue, priority), then run it **from the ML Space notebook**.
The image comes from `gift_docker/images.json` for the detected cluster:

```bash
python gift_scripts/submit_infinite_job.py
```

When it is `Running`, connect to rank 0:

```bash
mlspace ssh by-host <JOB-ID>-mpimaster-0
# or: ssh user@<JOB-ID>-mpimaster-0.<JOB-ID> -p 2222 -i ~/.ssh/<key>
```

## 4. Train inside the container (multi-node)

From `mpimaster-0`, `run.sh` starts one `torchrun` per node (ranks rendezvous on
the master). Pick the run with `CONFIG_FILE` (an experiment YAML under
`configs/experiments/`); anything in `EXTRA_ARGS` is forwarded to
`torchtitan.train` as dot-path overrides. Validate first with
`python scripts/config_check.py <yaml>`.

```bash
# llama3 8B (default)
NNODES=2 NGPUS_PER_NODE=8 bash gift_scripts/infinite_jobs/run.sh

# gift_moe debug model
NNODES=2 CONFIG_FILE=configs/experiments/gift_moe_debug.yaml \
  bash gift_scripts/infinite_jobs/run.sh

# extra training flags
EXTRA_ARGS="--training.steps 100 --metrics.log_freq 10" \
  bash gift_scripts/infinite_jobs/run.sh
```

| env              | default                               | meaning                                        |
| ---------------- | ------------------------------------- | ---------------------------------------------- |
| `CONFIG_FILE`    | `configs/experiments/llama3_8b.yaml`  | experiment YAML, relative to the checkout root |
| `NNODES`         | `1`                                   | number of nodes (= `N_WORKERS`)                |
| `NGPUS_PER_NODE` | `8`                                   | GPUs per node                                  |
| `EXTRA_ARGS`     | –                                     | dot-path overrides for `torchtitan.train` |

**Single node?** Skip MPI and use the model wrapper directly, e.g. gift on DeepEP:

```bash
NGPU=2 CONFIG_FILE=configs/experiments/gift_moe_debug_deepep.yaml torchtitan/models/gift_moe/train.sh
```

## 5. Run tests (inside the container, single node)

```bash
bash tests/run_tests.sh                 # unit + integration (features, models)
SUITES=h100 bash tests/run_tests.sh integration   # gift / MoE DeepEP suite
```

## 6. Clean up

After a job stops or crashes, kill stragglers and free shared memory on all nodes:

```bash
NNODES=2 bash gift_scripts/infinite_jobs/clean.sh
```
