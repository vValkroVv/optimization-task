# MultiPL-E HumanEval

## Paper

MultiPL-E: A Scalable and Polyglot Approach to Benchmarking Neural Code Generation
https://arxiv.org/abs/2208.08227

MultiPL-E translates OpenAI's HumanEval (Python) benchmark into 18+ languages.
This task implements **10 non-Python languages** from the canonical subset used in
Qwen2.5-Coder, DeepSeek-Coder, Code Llama, StarCoder2 evaluations.

Dataset: `nuprl/MultiPL-E` on HuggingFace. 161 tasks per language.

## Languages

| # | Language | Task name | Stop tokens |
|---|----------|-----------|-------------|
| 1 | C++ | `multipl_e_humaneval_cpp` | `\n}` |
| 2 | C# | `multipl_e_humaneval_cs` | `\n    }\n` |
| 3 | Java | `multipl_e_humaneval_java` | `\n    }\n` |
| 4 | JavaScript | `multipl_e_humaneval_js` | `\nfunction `, `\n/*`, `\n//`, `\nconsole.log` |
| 5 | TypeScript | `multipl_e_humaneval_ts` | `\nfunction `, `\n/*`, `\n//`, `\nclass` |
| 6 | PHP | `multipl_e_humaneval_php` | `\nfunction`, `\n?>`, `\n//`, `\n#` |
| 7 | Rust | `multipl_e_humaneval_rs` | `\n}` |
| 8 | R | `multipl_e_humaneval_r` | `\n#`, `\n\`\`\`` |
| 9 | Racket | `multipl_e_humaneval_rkt` | `\n(define `, `\n#\|`, `\n;`, `\n(` |
| 10 | Bash | `multipl_e_humaneval_sh` | `\n}` |

## How it works

**Generation**: lm-eval sends the `prompt` field from the dataset to the model.
Stop tokens (per-language, in each yaml's `generation_kwargs.until`) are passed
to sglang as server-side stop sequences. The `extract_code` filter prepends the
prompt to the generated completion to form a complete runnable program.

**Validation**: `pass_at_1` in `utils.py` POSTs the program + tests to a
**multipl_e_executor** service via YT TaskProxy. The executor compiles/runs the
code in a sandboxed container and returns `{status, exit_code, stdout, stderr}`.
pass@1 = 1.0 if `status == "OK"` and `exit_code == 0`, else 0.0.

On any service/network error, the metric **raises an exception** (fail-fast)
to stop the benchmark immediately rather than silently producing 0.0 scores.

## Architecture: multipl_e_executor on YTsaurus

The executor runs as a **YT vanilla job** on a YTsaurus cluster. It is a FastAPI
service inside a Docker container with 10 language runtimes (g++, javac, node,
tsc, php, mono, rustc, Rscript, racket, bash).

```
lm-eval (PAM job)
    │
    │  POST /execute {language, program, timeout}
    │  Headers: Authorization, x-yt-taskproxy-operation-alias, ...
    ▼
YT TaskProxy (configured through YT_TASKPROXY_BASE_URL)
    │
    │  routes by alias → operation → task → port
    ▼
multipl_e_executor (YT vanilla job, Docker container)
    │
    │  asyncio subprocess: compile + run
    ▼
{status: "OK"|"SyntaxError"|"Timeout"|"RuntimeError", exit_code, stdout, stderr}
```

**Key details**:
- TaskProxy alias: `multipl_e_executor`, task: `run_multipl_e_executor`
- Subprocess stdout/stderr truncated to 100 bytes (prevents multi-GB responses
  from infinite print loops)
- Subprocess memory limited via `RLIMIT_DATA` (configurable, default 2048 MB)
- Java and Node.js use `limit_memory=False` because their VMs reserve large
  virtual address spaces at startup

## Environment variables

Required by `utils.py` at evaluation time:

| Variable | Default | Description |
|----------|---------|-------------|
| `YT_TOKEN` | *(required)* | YT auth token for TaskProxy |
| `YT_TASKPROXY_BASE_URL` | *(required)* | TaskProxy endpoint |
| `YT_TASKPROXY_OPERATION_ALIAS` | `multipl_e_executor` | Operation alias |
| `YT_TASKPROXY_TASK_NAME` | `run_multipl_e_executor` | Task name |
| `YT_TASKPROXY_SERVICE` | `port0` | Service port name |

## Usage in PAM

Add tasks to the PAM config. Both required connection values must be in
`extra_env`:

```python
from contrib.job_submitter.job_submitter.jobs.pam.autometrics import AutometricsJobPAM

AutometricsJobPAM(
    user_login="<user>",
    exp_name="...",
    checkpoint_dir="<path>",
    extra_env={
        "YT_TOKEN": "<token>",
        "YT_TASKPROXY_BASE_URL": "<taskproxy-url>",
    },
).submit()
```

In `pam_config.yaml`, use the group name to run all 10 languages at once:

```yaml
    lm_eval:
      multipl_e_humaneval:
        num_fewshot: 0
        log_samples: true
```

This is equivalent to listing all 10 `multipl_e_humaneval_*` tasks individually.
The group also computes an aggregate `pass_at_1` (mean across languages, weighted
by number of tasks per language).

## Retry logic

`utils.py` has two layers of retry:

1. **urllib3 retries** (transport-level): `total=20, connect=5, read=5`,
   `backoff_factor=5, backoff_jitter=10`, for HTTP 429/500/502/503/504
2. **Application-level retries**: 5 attempts with 15-30s random delay,
   catches all `requests.exceptions.RequestException` (covers
   ChunkedEncodingError and other connection-level failures)

## Files

```
lm_eval/tasks/multipl_e/
├── _multipl_e_humaneval.yaml   # task group (all 10 languages + aggregate metric)
├── _humaneval_template.yaml    # shared config (dataset, metric, filter)
├── multipl_e_humaneval_*.yaml  # 10 per-language configs (stop tokens, dataset_name)
├── utils.py                    # build_reference, extract_code, pass_at_1
└── README.md
```

## Citation

```
@inproceedings{cassano2023multipl,
  title={MultiPL-E: A Scalable and Polyglot Approach to Benchmarking Neural Code Generation},
  author={Federico Cassano and John Gouwar and Daniel Nguyen and Sydney Nguyen and Luna Phipps-Costin and Donald Pinckney and Ming-Ho Yee and Yangtian Zi and Carolyn Jane Anderson and Molly Q Feldman and Arjun Guha and Michael Greenberg and Abhinav Jangda},
  journal={IEEE Transactions on Software Engineering},
  year={2023}
}
```
