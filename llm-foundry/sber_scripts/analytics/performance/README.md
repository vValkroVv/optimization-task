# Model Performance Benchmark

> **Stage 4.3 · performance.** Vendored as-is from llm-foundry (MR !1323), GPU/cluster-only, and
> deliberately **outside the strict gates** — mypy `exclude` + ruff `extend-exclude` skip
> `sber_scripts/analytics/performance/`, and pytest does not collect it. It moved here from the
> former top-level `sber_benchmarks/` in the restructure; the scripts still run directly (not via
> the `analytics` CLI).

End-to-end inference benchmark for an llm-foundry config.

```text
Input
  llm_foundry_config.yaml
  tokenizer path
  optional existing artifacts/random_hf_checkpoint_uniform[_fp8]

Command
  bash sber_scripts/analytics/performance/run_model_perf_benchmark.sh

Flow
  launcher
    -> submits inf_bench InfiniteJob
       -> create/reuse random HF checkpoint
       -> optionally convert HF checkpoint to FP8
       -> submit nested AutometricsJobLLMF inference job
       -> wait for SGLang readiness
       -> run aiperf
       -> optionally collect profiler traces
       -> write status/artifacts and clean up inference job

Output
  <config_dir>/artifacts/random_hf_checkpoint_uniform/
  <config_dir>/artifacts/random_hf_checkpoint_uniform_fp8/   when QUANTIZE_FP8=true
  <config_dir>/artifacts/model_perf/<timestamp>/status.json
  <config_dir>/artifacts/model_perf/<timestamp>/aiperf/
  <config_dir>/artifacts/model_perf/<timestamp>/traces/      when COLLECT_TRACES=1
```

## Usage

Edit the assignment block at the top of:

```text
sber_scripts/analytics/performance/run_model_perf_benchmark.sh
```

Run:

```bash
bash sber_scripts/analytics/performance/run_model_perf_benchmark.sh
```

The launcher prints the full `model_perf_benchmark.py` command, submits the `inf_bench` job, and optionally monitors it depending on `MONITOR`.

## Launcher Parameters

All user-facing benchmark settings are direct shell assignments in `run_model_perf_benchmark.sh`.

### Config and Checkpoint Init

| Parameter | Effect |
| --- | --- |
| `CONFIG_PATH` | llm-foundry YAML config used to create or reuse the random HF checkpoint. Default artifacts are written next to this config. |
| `QUEUE_NAME` | Queue for the `inf_bench` job. Also used by nested inference when `INFERENCE_QUEUE_NAME` is empty. |
| `PRIORITY_CLASS` | Priority for the `inf_bench` job. Also used by nested inference when `INFERENCE_PRIORITY_CLASS` is empty. |
| `BASE_IMAGE` | Image for the `inf_bench` job. |
| `PREP_N_GPUS` | GPU count for checkpoint init, quantization, aiperf, and traces inside the `inf_bench` job. |
| `PREP_N_WORKERS` | Worker count for the `inf_bench` job. |
| `FORCE_INIT` | Recreate the random HF checkpoint even if it already exists. |
| `HF_CHECKPOINT_DIR` | Optional random HF checkpoint path. Empty means `<artifacts>/random_hf_checkpoint_uniform`. |
| `INCLUDE_MTP` | Passed to `init_random_hf_model.py --include_mtp`. |
| `USE_CUSTOM_SGLANG` | Passed to `init_random_hf_model.py --use_custom_sglang`. |
| `MODEL_TYPE` | Custom model type for checkpoint init, usually `gigachat35`. |
| `MODEL_TAG` | Model tag for non-custom SGLang init mode. Usually empty for this benchmark. |
| `INIT_DEVICE` | Init device for `init_random_hf_model.py`. |
| `INIT_CUDA_VISIBLE_DEVICES` | `CUDA_VISIBLE_DEVICES` for checkpoint init. |
| `UNIFORM_MOE` | Passed to `init_random_hf_model.py --uniform_moe`. |
| `EXPAND_CONTEXT_LEN_TO` | Context length written into the generated HF config. Current launcher value is `262144`. |

### FP8 Quantization

| Parameter | Effect |
| --- | --- |
| `QUANTIZE_FP8` | `true`: create/reuse FP8 checkpoint and serve it. `false`: skip quantization and serve the random BF16/HF checkpoint. |
| `FORCE_QUANT` | Recreate the FP8 checkpoint even if it already exists. |
| `FP8_CHECKPOINT_DIR` | Optional FP8 checkpoint path. Empty means `<artifacts>/random_hf_checkpoint_uniform_fp8`. |
| `QUANT_DEVICE` | Device for layer-by-layer FP8 quantization. |
| `WEIGHT_BLOCK_SIZE` | Two integers passed as `--weight_block_size`. |
| `ACTIVATION_SCHEME` | Passed as `--activation_scheme`. Values: `dynamic`, `static`. |
| `MODULES_TO_NOT_CONVERT` | Space-separated skip patterns passed explicitly to `--modules_to_not_convert`. Current value is the quantizer default skip list without `self_attn.gate_proj` and `shared_expert_sigmoid`. |

### Inference Job and SGLang

| Parameter | Effect |
| --- | --- |
| `INFERENCE_N_GPUS` | GPU count for the nested custom-SGLang inference job. |
| `INFERENCE_N_WORKERS` | Worker count for the nested inference job. |
| `INFERENCE_QUEUE_NAME` | Optional inference queue override. Empty means `QUEUE_NAME`. |
| `INFERENCE_PRIORITY_CLASS` | Optional inference priority override. Empty means `PRIORITY_CLASS`. |
| `INFERENCE_BASE_IMAGE` | Optional inference image override. Empty means `AutometricsJobLLMF` resolves the image from `sglang_tag="custom"`. |
| `TP` | Tensor parallel size. |
| `EP` | Expert parallel size. |
| `DP` | Optional data parallel size. Empty value is not written to the PAM config. |
| `PP` | Optional pipeline parallel size. Empty value is not written to the PAM config. |
| `ROUTER_PORT` | SGLang router port. |
| `SGLANG_COMMUNICATION_PORT` | PAM/SGLang communication port. |
| `MAX_SEQ_LEN` | Top-level PAM `max_seq_len`. Current launcher value is `262144`. |
| `GATED_NORM_IMPL` | `autometrics.inference.gated_norm_impl`. Values accepted by the wrapper: `native`, `torch_compile`, `triton`. |
| `MOE_RUNNER_BACKEND` | `autometrics.inference.moe_runner_backend`. |
| `MEM_FRACTION_STATIC` | `autometrics.inference.mem_fraction_static`. |
| `TOKENIZER_PATH` | Tokenizer used by checkpoint init, PAM/OBM proxy, aiperf, and trace `bench_serving`. |
| `COPY_TOKENIZER_TO_ARTIFACTS` | Copies tokenizer directory to `<artifacts>/tokenizer` and uses that copied path in generated state. |

The wrapper does not set `max_running_requests` or `max_queued_requests` in the generated PAM config. SGLang/PAM defaults apply.

### Timing, Monitoring, and Cleanup

| Parameter | Effect |
| --- | --- |
| `SERVER_TIMEOUT_MINUTES` | Readiness timeout for the nested inference server after it has been submitted. This does not limit checkpoint init, quantization, aiperf, traces, or total `inf_bench` job runtime. `0` means wait without a timeout. |
| `SERVER_POLL_SECONDS` | Interval between readiness probes from the `inf_bench` job to the nested inference server. Each probe tries `/health`, then a tiny `/generate` request. Current launcher value is `15`. |
| `MONITOR` | Controls only the local launcher process. `1` keeps the launcher attached to `job.monitor(...)` for the `inf_bench` job. `0` submits the job and returns after submission. |
| `MONITOR_POLL_SECONDS` | Poll interval used by local `job.monitor(...)` for the `inf_bench` job. It is not a timeout and does not affect benchmark execution. Current launcher value is `30`. |
| `KEEP_INFERENCE_JOB` | `1` leaves the nested inference job running after benchmark completion or failure. `0` kills it in cleanup. |

### aiperf Workload

| Parameter | Effect |
| --- | --- |
| `SKIP_AIPERF` | Skip `aiperf_benchmark.sh` when enabled. |
| `AIPERF_VENV` | Optional virtualenv sourced before running aiperf. |
| `AIPERF_BIN` | Optional direct aiperf executable path. |
| `AIPERF_PYTHON` | Optional Python executable used as `python -m aiperf profile`. |
| `AIPERF_MODEL` | `--model` for `aiperf profile`. |
| `AIPERF_ENDPOINT_TYPE` | `--endpoint-type` for `aiperf profile`. |
| `AIPERF_STREAMING` | Adds `--streaming` when true. |
| `AIPERF_INPUT_LEN` | `--synthetic-input-tokens-mean`. Current launcher value is `1024`. |
| `AIPERF_OUTPUT_LEN` | `--output-tokens-mean`; also used for default `max_tokens` and `min_tokens`. Current launcher value is `8192`. |
| `AIPERF_INPUT_STDDEV` | `--synthetic-input-tokens-stddev`. |
| `AIPERF_CONCURRENCY` | `--concurrency`. Current launcher value is `64`. |
| `AIPERF_REQUESTS` | `--request-count`. Current launcher value is `128`. |
| `AIPERF_OUTPUT_DIR` | Optional output directory. Empty means `<run_dir>/aiperf`. |
| `AIPERF_EXTRA_INPUTS` | Optional semicolon-separated `--extra-inputs` list. Empty means `max_tokens:<AIPERF_OUTPUT_LEN>;min_tokens:<AIPERF_OUTPUT_LEN>;ignore_eos:true`. |

### Trace Workload

| Parameter | Effect |
| --- | --- |
| `COLLECT_TRACES` | Run `get_inference_traces.py` after aiperf when enabled. |
| `TRACE_INPUT_LEN` | Random input length for `sglang.bench_serving`. Current launcher value is `10240`. |
| `TRACE_OUTPUT_LEN` | Random output length for `sglang.bench_serving`. Current launcher value is `1024`. |
| `TRACE_NUM_PROMPTS` | Number of prompts for trace workload. Current launcher value is `32`. |
| `TRACE_MAX_CONCURRENCY` | Adds `--max-concurrency` when greater than zero. Current launcher value is `0`, so the flag is omitted. |
| `TRACE_RPS` | Adds `--request-rate` when greater than zero. Current launcher value is `8`. |
| `TRACE_SAVE_DETAILS` | Writes `details.json` when true. |
| `TRACE_WARMUP_STEPS` | Sent as `start_step` in `/start_profile` when greater than zero. Current launcher value is `5`. |
| `TRACE_OUTPUT_DIR` | Optional trace output directory. Empty means `<run_dir>/traces`. |

## Artifacts and Reuse

Default artifact layout:

```text
<config_dir>/artifacts/
  random_hf_checkpoint_uniform/
  random_hf_checkpoint_uniform_fp8/
  tokenizer/
  tensorboard/<timestamp>/
  model_perf/<timestamp>/
    state.json
    inf_bench_job.sh
    status.json
    inference_pam_config.yaml
    aiperf/
    traces/
```

Checkpoint reuse rules:

| Path | Reuse check | Force flag |
| --- | --- | --- |
| `HF_CHECKPOINT_DIR` or `<artifacts>/random_hf_checkpoint_uniform` | Directory contains `config.json` and at least one `*.safetensors` or `pytorch_model*.bin` file. | `FORCE_INIT=1` |
| `FP8_CHECKPOINT_DIR` or `<artifacts>/random_hf_checkpoint_uniform_fp8` | HF checkpoint check plus `config.json.quantization_config.quant_method == "fp8"`. | `FORCE_QUANT=1` |

`status.json` records run status, checkpoint paths, nested inference job name, server URL, and error text on failure.

## `model_perf_benchmark.py` Architecture

`model_perf_benchmark.py` is the main orchestrator. It has two execution modes.

### Submission Mode

This is the mode used by `run_model_perf_benchmark.sh`.

1. Parse launcher arguments.
2. Resolve `CONFIG_PATH`, artifacts directory, run directory, HF checkpoint directory, FP8 checkpoint directory, aiperf output directory, and trace output directory.
3. Resolve tokenizer path. If `COPY_TOKENIZER_TO_ARTIFACTS=true`, copy the tokenizer directory to `<artifacts>/tokenizer` and use the copied path in all later steps.
4. Write `<run_dir>/state.json`.
5. Write `<run_dir>/inf_bench_job.sh`.
6. Submit an `InfiniteJob` named by the launcher settings. This is the `inf_bench` job.
7. If `MONITOR=1`, call `job.monitor(...)` for the `inf_bench` job with `MONITOR_POLL_SECONDS` as the polling interval.

`inf_bench_job.sh` runs:

```text
python sber_scripts/analytics/performance/model_perf_benchmark.py --run-in-job --state-json <run_dir>/state.json
```

### `--run-in-job` Mode

This mode runs inside the submitted `inf_bench` job.

1. Load `<run_dir>/state.json`.
2. Create `<run_dir>/status.json` with `status="running"`.
3. `maybe_init_checkpoint(state)`:
   - checks the HF checkpoint reuse condition;
   - skips init when reuse is valid and `FORCE_INIT=0`;
   - otherwise runs `sber_scripts/analytics/performance/run_init_random_hf_model.sh`;
   - validates that the HF checkpoint exists after init.
4. Quantization decision:
   - if `QUANTIZE_FP8=true`, `maybe_quantize_checkpoint(state)` checks/reuses/creates the FP8 checkpoint with `fine_grained_fp8_quantization_layer_by_layer.py`;
   - if `QUANTIZE_FP8=false`, the BF16/HF checkpoint path is used directly for inference.
5. `write_pam_config(state)` writes `<run_dir>/inference_pam_config.yaml`.
6. `submit_inference_job(state)` submits nested `AutometricsJobLLMF` with:
   - `mode="inference"`;
   - `debug=True`;
   - `sglang_tag="custom"`;
   - checkpoint path selected by `QUANTIZE_FP8`;
   - the generated PAM config.
7. Build the nested server URL:

```text
http://<job_name>-mpimaster-0.<job_name>:<ROUTER_PORT>
```

8. `wait_for_server(...)` polls readiness. It tries `GET /health`, then a tiny `POST /generate` fallback. `SERVER_TIMEOUT_MINUTES=0` disables the readiness timeout.
9. `run_aiperf(state, base_url)` runs `sber_scripts/analytics/performance/aiperf_benchmark.sh` with explicit environment variables derived from launcher settings.
10. `run_traces(state, base_url)` runs `sber_scripts/analytics/performance/get_inference_traces.py` when `COLLECT_TRACES=1`.
11. Mark `status.json` as `completed` or `failed`.
12. If an inference job was submitted and `KEEP_INFERENCE_JOB=0`, stop it through job_submitter/client APIs.

### Generated PAM Config

The generated config skips PAM conversion/preparation because the benchmark already points inference at an HF checkpoint:

```json
{
  "max_seq_len": 262144,
  "autometrics": {
    "globals": {
      "delete_converted_hf": false,
      "sglang_communication_port": 50000
    },
    "router": {
      "host": "0.0.0.0",
      "port": 30000
    },
    "convertation": {
      "skip": true,
      "mappers": ["map_gigafsdp_to_hf_safetensor"],
      "input_tp_size": 1
    },
    "preparation": {"skip": true},
    "inference": {
      "tp": 8,
      "ep": 8,
      "trust_remote_code": true,
      "allow_auto_truncate": true,
      "mem_fraction_static": 0.8,
      "gated_norm_impl": "torch_compile",
      "moe_runner_backend": "triton_llmf"
    }
  }
}
```

`dp`, `pp`, and `tokenizer_path` are added only when configured. The wrapper does not add `max_running_requests` or `max_queued_requests`.

## Other Files Used by the Benchmark

- `sber_scripts/analytics/performance/run_model_perf_benchmark.sh`: user-facing launcher with all benchmark settings.
- `sber_scripts/analytics/performance/run_init_random_hf_model.sh`: shell adapter that calls `init_random_hf_model.py` with environment variables prepared by `model_perf_benchmark.py`.
- `sber_scripts/analytics/performance/init_random_hf_model.py`: creates a random-weight HuggingFace checkpoint from the llm-foundry config.
- `contrib/gigachat-utils/convertation/quantization/fine_grained_fp8_quantization_layer_by_layer.py`: layer-by-layer BF16 to FP8 converter used when `QUANTIZE_FP8=true`.
- `contrib/job_submitter/job_submitter/jobs/llmf/autometrics.py`: submits the nested inference job; `sglang_tag="custom"` adds the custom SGLang fork to remote code libraries.
- `sber_scripts/analytics/performance/aiperf_benchmark.sh`: runs `aiperf profile` against the nested SGLang server.
- `sber_scripts/analytics/performance/get_inference_traces.py`: starts SGLang profiling, runs `sglang.bench_serving`, then stops profiling and writes trace files.
