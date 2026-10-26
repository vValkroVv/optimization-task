# codeelo

lm-eval task that scores a checkpoint on Codeforces contests using the
[CodeElo](https://arxiv.org/abs/2501.01257) protocol: generate solutions, submit
to Codeforces NAPI, compute per-contest Elo against real participants, average.

## Variants

| Task name | Prompt style | When to use |
|---|---|---|
| `codeelo_div{1,2,3}_{cpp,python}` | LiveCodeBench base-model one-shot: one worked example (Watermelon), then the problem and an open ```` ```cpp ```` / ```` ```python ```` fence; the closing fence ends the answer. Python is submitted as PyPy 3 | Pretrain checkpoints. One task per division and language, each with a default contest set (see [Default contest sets](#default-contest-sets)) |
| `codeelo_instruct_div{1,2,3}_{cpp,python}` | LiveCodeBench-style instruct prompt asking for one fenced block; the last fenced block of the answer is submitted. No stop strings and 32K new tokens so a thinking model can reason first; its reasoning arrives separately as `reasoning_content` and never reaches the judge | Chat and thinking models. One task per division and language, with the same default contest sets |

All tasks share `_codeelo_template.yaml` (dataset, metrics). The one-shot
division tasks build on `_codeelo_oneshot_{cpp,python}.yaml`, the instruct ones
on `_codeelo_instruct_{cpp,python}.yaml`.

## How to enable in PAM

### 1. `pam_config.yaml`

Pretrain checkpoints use the one-shot division tasks under `lm_eval.pretrain`,
chat and thinking models the instruct division tasks under `lm_eval.instruct`:

```yaml
    lm_eval:
      instruct:
        codeelo_instruct_div1_cpp:
          log_samples: true
          process_results_workers: 16
        codeelo_instruct_div1_python:
          log_samples: true
          process_results_workers: 16
```

`metadata` is optional for the division tasks: each carries a default
`contest_ids`. PAM's `metadata` is a JSON string forwarded as `--metadata` and
overrides the yaml key by key, so a run can pick other contests or drop single
problems:

```yaml
          metadata: '{"contest_ids": [2273, 2268], "exclude_problems": ["2273D"]}'
```

`exclude_problems` leaves those problems out of generation and judging. The
contest rating still counts them as unsolved, so read only pass@k from such a
run.

PAM passes every key of a task entry as a `--<key>` flag and lm-eval parses its
arguments strictly, so an unknown key aborts the run.

## Default contest sets

The division tasks default to the last 10 rated rounds of each division at the
time of writing, interactive problems included. Educational and combined
Div. 1+2 rounds are not in any set.

| Division | Contests | Dates |
|---|---|---|
| Div. 1 | 2273, 2268, 2262, 2255, 2249, 2239, 2223, 2219, 2201, 2196 | 2026-02-11 … 2026-10-11 |
| Div. 2 | 2274, 2269, 2267, 2264, 2263, 2258, 2257, 2256, 2252, 2248 | 2026-08-01 … 2026-10-11 |
| Div. 3 | 2275, 2266, 2259, 2254, 2244, 2241, 2236, 2227, 2200, 2195 | 2026-02-15 … 2026-10-07 |

## How the task works

### Lifecycle

| Phase | lm-eval hook | What happens |
|---|---|---|
| Load | `custom_dataset` → `utils.build_dataset_from_napi` | For each contest in `contest_ids` (from `--metadata`), call NAPI `/problemStatements?contestId=X`. Build a `DatasetDict({"test": Dataset.from_list(...)})` with one row per problem. |
| Prompt | `doc_to_text` → `utils.format_oneshot_prompt{,_python}` or `utils.format_instruct_prompt{,_python}` | One-shot: a worked example, the statement and an open code fence. Instruct: the LiveCodeBench instruct prompt asking for one fenced block. |
| Generate | sglang via lm-eval generate_until | 8 samples per problem (`repeats: 8`). Pretrain tasks: the paper's sampling (T 0.7, top_p 0.8, top_k 20, repetition penalty 1.1), 4096 new tokens. Instruct tasks: T 1.0, top_p 1.0, 32,768 new tokens, plus PAM's `obm_proxy.reasoning-token-budget` when thinking is on. |
| Submit + judge | `process_results` → `utils.submit_oneshot_problem{,_python}` | Per sample (after the filter: the generation up to the closing fence, or the last fenced block of an instruct answer): empty code is `NO_CODE`, over 65,535 characters `SOURCE_TOO_LONG`; anything else goes to the proxy's `judge` action. The 8 verdicts are packed into `{contest_id, prob, verdicts}` and returned under every metric key. |
| Aggregate | `utils.aggregate_*` | After all problems are processed, group by `contest_id`, fetch each contest's standings summary from the proxy, and call `calc_rating.contest_rating(summary, verdicts)`. |

### What `calc_rating` does

The rating follows the simulated-contest method of OpenAI's *Competitive
Programming with Large Reasoning Models* (arXiv 2502.06807, appendix B), the
one DeepSeek-V4 reports its Codeforces rating with.

1. The proxy's `standings` action returns, per problem, the median score of
   the human contestants who solved it with each number of failed attempts
   (points under CF scoring, penalty minutes under ICPC), and every rated
   contestant's result and rating before the contest.
2. A problem's samples are submitted in a random order: the number of failed
   attempts is the position of the first `OK`. The model scores the human
   median for that number of failures; a problem nobody solved with that many
   failures counts as unsolved (reported as `unmatched`).
3. The model's rating is the maximum-likelihood `R` given its place in the
   standings: `Σ 1/(1 + 10^((r_i - R)/400))` over rated contestants equals the
   number it beat. A tie counts as a loss, as Codeforces places tied
   contestants at the bottom of their group. Newcomers (rating 0) count as
   1400, their rating inside the Codeforces calculation.
4. Steps 2–3 are averaged over 1000 random orderings (seeded), and the
   per-contest ratings over contests.

### Reported metrics

The results JSON ends up in the checkpoint's metrics dir with:

- `codeelo_avg_rating` — mean of per-contest Elo ratings; `NaN` if the standings of any contest could not be loaded (always so without the proxy), while pass@k is still reported.
- `codeelo_pass_at_{1,2,4,8}` — unbiased pass@k over problems.

