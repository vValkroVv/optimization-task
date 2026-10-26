# BigCodeBench

`bigcodebench` — the "complete" split of [BigCodeBench](https://github.com/bigcode-project/bigcodebench)
(`bigcode/bigcodebench`, `v0.1.4`, 1140 tasks): the model completes a function
from its signature and docstring; the task's `unittest` cases decide pass/fail.

Programs are executed by the `bcb_executor` gigagym service through YT
TaskProxy, one YT job per program. Required environment:

```
YT_TOKEN                        token with read access to the executor alias
YT_TASKPROXY_BASE_URL           TaskProxy base URL
BCB_TASKPROXY_OPERATION_ALIAS   default bcb_executor
BCB_TASKPROXY_TASK_NAME         default run_bcb_executor
BCB_CACHE_ONLY=1                skip execution (dataset prefetch at image build)
```

Unlike upstream, the tree-sitter `sanitize` step is not applied: completions
are cut by the stop strings, which include the file continuation markers
(`<|end_of_file|>`, `<|begin_of_file_name|>` and their `<｜end▁of▁file｜>`,
`<｜begin▁of▁file▁name｜>` forms), and anything else the model writes after the
function counts as a failure.
