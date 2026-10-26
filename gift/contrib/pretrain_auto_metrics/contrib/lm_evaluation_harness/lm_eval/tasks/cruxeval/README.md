# CRUXEval

Code Reasoning, Understanding, and eXecution Evaluation — benchmark of 800 Python functions with input-output pairs.

## Tasks

- **cruxeval** — group that aggregates both subtasks (recommended)
- **cruxeval_output_cot** — given code + input, predict output (CRUXEval-O)
- **cruxeval_input_cot** — given code + output, predict input (CRUXEval-I)

## Paper

Gu et al., "CRUXEval: A Benchmark for Code Reasoning, Understanding and Execution", 2024
https://arxiv.org/abs/2401.03065

## Links

- Original repo: https://github.com/facebookresearch/cruxeval
- Dataset (HuggingFace): https://huggingface.co/datasets/cruxeval-org/cruxeval
- Prompts adapted from: https://github.com/facebookresearch/cruxeval/blob/main/prompts.py

## Details

- 800 samples, 0-shot, greedy decoding
- CoT prompts with `[THOUGHT]...[/THOUGHT]` and `[ANSWER]...[/ANSWER]` tags
- Answer extraction via regex, validation via `hf_evaluate.code_eval` (same sandbox as HumanEval)
- Metric: `pass_at_1` (unified, takes `mode: input | output` kwarg)

## PAM config example

Use the group — runs both subtasks and reports one aggregated score:

```yaml
evaluators:
  lm_eval:
    cruxeval:
      num_fewshot: 0
      log_samples: true
```

Or pick an individual subtask:

```yaml
evaluators:
  lm_eval:
    cruxeval_output_cot:
      num_fewshot: 0
      log_samples: true
    cruxeval_input_cot:
      num_fewshot: 0
      log_samples: true
```

## Output

Running `--tasks cruxeval` produces three entries in `results`:

- `cruxeval` — group-level aggregated `pass_at_1` (weighted mean across subtasks)
- `cruxeval_output_cot` — subtask `pass_at_1` (CRUXEval-O)
- `cruxeval_input_cot` — subtask `pass_at_1` (CRUXEval-I)

## Reference results

Validated on **Qwen2.5-Coder-7B (base)** — reproduces numbers from the Qwen2.5-Coder Technical Report (https://arxiv.org/abs/2409.12186):

| | Our run | Qwen paper |
|---|---:|---:|
| CRUXEval-O (output_cot, pass@1) | 57.25% | 56.0% |
| CRUXEval-I (input_cot, pass@1)  | 56.38% | 56.5% |
| cruxeval (group aggregate)      | 56.81% | — |
