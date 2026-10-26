---
name: pam-review
description: >
  Review a diff, branch or MR in pretrain-auto-metrics (PAM) and the repos around it
  (lm-evaluation-harness, job-submitter, sglang-pretrain, composer, new_obm).
  Encodes the review comments the maintainer repeats on every MR, so they get caught
  before a human reads the diff. Use whenever asked to review PAM code, a PAM MR,
  or a new benchmark shooter/evaluator.
---

# PAM review

Review as the maintainer who has left the same comment two hundred times and is
tired of it. Report findings; do not rewrite the branch unless asked.

Read the code the diff touches before judging it — a rule below applied to a
misread flow is worse than no review.

For what the code *should* look like: `/pam-code-style`
(`.agents/skills/pam-code-style/SKILL.md`) and, for evaluators,
`/pam-create-evaluator` (`.agents/skills/pam-create-evaluator/SKILL.md`).
This skill is the review pass over the result. Do not restate those rules —
check them.

## The checklist

Every item is a review comment the maintainer repeats on every MR.

### 1. No `raise` inside an evaluator at runtime
Not for validation, not for "impossible" states, not wrapped in `try/except` and
re-raised.
- Subprocess failed → `SubprocessManager.log_error()` (`subprocess_manager.py:143`), not `raise_for_exit_code()`.
- Artifacts missing or unparseable → return empty metrics and `Logger.error` with the path.
- A phase failed and nothing is worth saving (generation of a `ScoringMetricsEvaluator`, scoring whose outputs are unusable) → raising is right: the scheduler skips scoring and saves nothing.
- Config is wrong → that belongs to the config model (item 2), not to `run()`.

The actual reason is what a raise throws away: `DynamicScheduler._run_task`
(`metrics_manager.py`) catches `Exception` per evaluator step and then skips
`save_evaluator_metrics`, so a raise discards whatever the benchmark had already
measured.

The counterpart obligation: an evaluator may not vanish quietly.
`save_evaluator_metrics` returns silently on empty metrics, so a swallowed failure
disappears from the report. Log the absence explicitly.

### 2. Config work belongs to the config model
- Dumping the config to benchmark args is a method on the config model, not on the evaluator.
- `model_dump` with an explicit `exclude` set — not a hand-rolled `build_overrides`, not `setdefault` chains over a dict.
- URL assembly, defaults derived from inference instances / base URL / checkpoint path: `apply_defaults()`. Judge URLs and instances come from `self.judge_endpoints[name]`, never from new `*_base_urls` fields or kwargs.
- The evaluator does not mutate the config.
- Validation lives here and *may* raise: failing at startup is right, failing eight hours in is not. Meaningless values must not validate — zero retries, zero timeout, empty task list. `ge=0` on a retry count is a bug (`config_model.py:571`, `config_model.py:1429`).

### 3. Configuration: surface and coverage
Every field is a question the user is forced to answer. Benchmark-internal
constants belong in the evaluator or the config model as defaults, not in the
user-facing config. Ask, per field: who sets this, and to what other value?

Coverage is the other half: a new evaluator is declared in every config that
lists evaluators, not only in the domain configs the author happened to run.
`terminal_bench`, `tau3` and `swe_bench` are in all 25 base `configs/*.yaml`;
an evaluator that lives only under `configs/alignment_prod/` is invisible to everyone
who measures from a base config.

### 4. The evaluator must not know where it shoots
No endpoint passed in as a parameter, no branching on SLURM env vars, no
per-shooter internet detection, no hardcoded environment-specific value of any
kind. PAM decides RLI vs proxy once, using the existing availability check in the
evaluator interface, and hands the shooter a single well-known env var. A change
here must be made in one place, not in every shooter.

### 5. Reproducible builds; data and repos in the image
- Benchmark data on NFS or in `contrib/` → move it into the image.
- Cloning a repo inside the Dockerfile → clone in `gitlab-ci.yaml`, `COPY` into the image.
- Tests running inside a Dockerfile → remove them.
- Duplicated Docker layers → separate build stage.
- Secrets → docker secret, never build args or files.
- Dependencies pinned, the source reachable, mirrored content verified. An unpinned requirements file is a future breakage with no owner.

### 6. "What case does this protect us from?"
Model-name slugs, `isinstance` guards on results, defensive normalizations, extra
validation on top of a pydantic model. Ask for the concrete failing input. No
answer → delete it. Same question for a constant introduced for a single use and
for a field or computed value with no downstream consumer: if nothing reads it,
it does not exist.

### 7. Skipping a phase needs a real signal
A phase must not be skipped because an output directory exists — a killed or
preempted run leaves one behind. Require the metrics themselves, or a marker
written as the last step. Symmetrically, a corrupt or half-written artifact must
not block reruns forever: the recovery path is to reject and redo it, not to
refuse to start.

### 8. Cache and resume identity
The fingerprint covers everything that changes the result — model, checkpoint,
prompts, sampling, task selection, judge — and deliberately excludes runtime
shape: parallelism, timeouts, instance and node counts. Too narrow serves a stale
score; too wide means the cache never hits and every minimal config diff re-runs
the benchmark. Configurations that can produce different numbers must not share
an identity. An artifact whose fingerprint version is unknown or older is
rejected, never accepted on a guess.

### 9. Infrastructure failures are not model quality
An API error, a dead proxy, a verifier crash, a container that never started, a
DNS-less search tool: these remove the metric, they do not lower it. A benchmark
that scored the model on requests that never reached it must publish nothing and
say why. Check that the failure share is counted, gated, and visible — a silently
depressed score is worse than a missing one.

### 10. Timeout and parallelism
Both configurable, both wired to what actually reaches the served model, no
hardcoded numbers — the shared default is 6400 seconds with 20 retries, not a
locally invented 900. Ask whether the concurrency was measured or guessed, and
how long the set takes end to end. `/pam-create-evaluator` has the mandatory shape.

### 11. Comments and docstrings
Generated-sounding comments restating the code: delete. A docstring is warranted
only where the transformation is non-obvious, and then it states what comes in,
what goes out, which errors are possible, and which cases it covers. Documentation
has one canonical home; a second copy is a future contradiction. Nobody reads a
README inside a vendored tool.

### 12. Diff hygiene and blast radius
- A feature MR must not quietly change shared core or the behavior of neighboring components. A safety-shooter MR editing terminal_bench is two MRs.
- Changes to shared code or to someone else's component: agreed with the owner up front, or split out.
- Submodule pointer moved without that being the point of the MR → revert it.
- Code moved between files for no stated reason → put it back.
- Function defined inside a function unless it makes the main algorithm clearer; variables used once instead of being inlined.

### 13. Computational results need reproducible evidence
An MR that claims a number — a speedup, a score, a fixed metric — carries the
config used, the path to the artifacts, and a comparison against a baseline or a
published result. "It got faster" without a run to point at is not reviewable.

### 14. Cache, resume and restart fixes carry a regression test
This is the one exception to the repository default of not adding tests: a fix to
caching, resumption, locking or restart logic ships with a test that reproduces
the original breakage. These bugs are invisible on a green run and only surface
when a job is preempted at the wrong moment.

### 15. Follow-ups have boundaries
If a finding needs separate architectural work, it is recorded as a follow-up with
an explicit scope and owner — in the MR description or an issue, never as a vague
"later" and never as a TODO left in the code.

### 16. lm-evaluation-harness
- A set of related tasks ships as one group task, not as N task names pasted into a PAM config.
- No separate task path, no shim other tasks do not need.
- Nobody reads the lm-eval README; documentation there is not a deliverable.

## Output

The result is a list of comments ready to be pasted into the MR. Nothing else —
no summary of the change, no praise, no "overall this looks good".

Group findings as **Blocking / Worth fixing / Nit**, most severe first. Per
finding, exactly four parts:

1. `path/to/file.py:LINE` — the line number on the **new** side of the diff, at the
   start of the problematic block. A comment that lands on the wrong line is useless.
2. A short title, up to eight words.
3. The offending code, verbatim from the diff, in a fenced block. Up to ten lines —
   enough to recognize the place without scrolling, nothing more.
4. **Коммент:** the text to paste, and nothing but that text.

Example of the shape:

### Blocking

**`node_runner/src/metrics/evaluators/widesearch.py:143` — рейз в судейской фазе**

```python
sm.start()
sm.wait_for_completion()
sm.raise_for_exit_code()
```

**Коммент:** Рейз здесь выбросит всё, что шутер успел посчитать: `_run_task`
поймает исключение и не сохранит метрики. Нужен `sm.log_error()` и то, что
удалось распарсить.

## Writing the comment

Write it in the language of the MR discussion — in these repos that is Russian.
One to three sentences. It goes into the thread as-is, so it must read like a
person wrote it in thirty seconds, not like a generated review.

What the comment contains:

- **What breaks and when.** The concrete consequence: which run dies, which number
  lies, which node hangs. That is the whole justification. A rule name is not a
  justification, and neither is "это не соответствует принятым практикам".
- **The replacement**, when the answer is known. Name the method, the field, the file.
- **The question**, when the code may have a reason you cannot see. One question,
  the one whose answer decides it: "от чего это нас спасает?", "какие значения тут
  бывают?", "ты подбирал параллельность или это на глаз?".

What it never contains:

- an opening compliment, a closing compliment, or a softening sandwich around either
- a retelling of what the code does — the author wrote it and can read it
- stacked hedges: "возможно, стоило бы рассмотреть вариант"
- a principle in place of a consequence: "нарушает single responsibility", "не по SOLID"
- references to this checklist, to rule numbers, or to a skill
- more than one issue in one comment — one thread, one thing to fix

Slop, for calibration:

> В данном месте используется raise, что потенциально может привести к
> нежелательному поведению. Рекомендуется рассмотреть возможность использования
> механизма логирования, поскольку это в большей степени соответствует принятым
> в проекте подходам к обработке ошибок.

Not slop:

> Рейз тут убивает фазу целиком. Логируй и верни пустые метрики.

Clean diff → one line saying so. Do not invent findings to look thorough, and do
not downgrade a real finding to a Nit to soften the list.
