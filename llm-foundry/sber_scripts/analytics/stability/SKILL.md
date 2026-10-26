---
name: stability-analysis
description: Diagnose training stability from TB logs (`sber_scripts/analytics/stability/instability/`) and HF safetensors checkpoint exports (`sber_scripts/analytics/stability/weights/`). Use when the user provides a TB run directory, an HF checkpoint (typically `<iter>/artifacts/hf_checkpoint/`), an `analyze.json`/`joint.json`, or asks "что с этим раном/чекпойнтом". Before execution, ask whether cluster job submission is allowed; if yes, run framework commands through cluster jobs, otherwise run all applicable local CPU-only steps. The deliverable is ALWAYS a single self-contained HTML diagnostic report — for one run, a full causal diagnosis; for several, that plus a run-vs-run comparison with paired plots. The chat reply is just the report path + a 2–3 sentence verdict.
allowed-tools: Bash, Read, Write, Edit, Grep
---

# stability-analysis

Two toolkits under `llm-foundry/sber_scripts/analytics/stability/`. In cluster mode,
they are invoked inside job scripts as `python3 -m <package>.<command>` from
that directory:

* `instability/` — TB-log analysis (CPU; a warm cache makes it near-instant).
* `weights/` — HF safetensors checkpoint analysis. `analyze` / `joint` need GPU
  and must be submitted as cluster jobs when cluster execution is permitted;
  `report` / `compare` / `dashboard` / `story` are CPU/stdlib but should still
  be job-submitted for full analyses when cluster execution is permitted.
  Input is the HF export folder (`<iter>/artifacts/hf_checkpoint/` with
  `model.safetensors.index.json`).

---

## Pre-flight permission gate

Before starting any stability-analysis execution, the agent **must ask the user**
one explicit question:

> May I submit jobs to the cluster, or should I do only what can be done locally?

Do not submit any cluster job until the user explicitly permits it. If the user
allows cluster jobs, follow the job-submitted workflows below. If the user says to
stay local, chooses CPU-only/local-only, or does not permit cluster submission,
run a **local CPU-only** analysis: execute every requested or inferable framework
command that does not require GPU in the local Python environment available for
that platform.

If the user did not give a narrower instruction such as "run exactly these
tools" or "compute exactly this metric", local CPU-only means: run all applicable
non-GPU framework steps for the provided inputs. On cloud, use the same active
conda/Python environment from which a cloud submitter would be launched. On O,
use the global `python3` environment from which an O submitter would be launched.
GPU producers (`weights analyze`, `weights joint`, `weights batch` with CUDA)
remain forbidden locally; omit those parts unless existing JSON artifacts are
already available.

The final deliverable is still a single self-contained HTML report in local
CPU-only mode. It may be shorter and must clearly omit GPU-only sections whose
inputs could not be produced locally, but it is still an HTML report generated
from the available deterministic framework outputs, not a chat-only analysis.

---

## What you deliver

**Always one self-contained HTML report.** This requirement applies to full
cluster-backed analyses and to local CPU-only analyses. Not a chat write-up, not
a pile of PNGs, not a Markdown summary — a single `story.html` built with
`instability.story` / `weights story` that a teammate can open and read end to
end. The analytical prose lives *inside* the HTML; your chat message is only the
path + a 2–3 sentence verdict.

The report answers, for the data given:

* **What happened during training** — the trajectory, the regime, the outcome.
* **What went well** — the signals that stayed healthy, argued (not just asserted).
* **What went wrong, and why** — each issue with (a) the plot that shows it,
  (b) what *detects* it (which monitor/metric crossed what), (c) the probable
  *cause* (a mechanism), and (d) what *leads to it in the data* (the upstream
  signal and the step/layer where it started).
* **For several runs** — all of the above per run, **plus** a comparison: what
  differs, what's the same, which run is the outlier, where and when the gap
  opened, and why. **Every cross-run claim is shown with paired plots** — if
  run A's plot X proves it went bad, the report shows the *same* plot X for B
  and C (or all runs superimposed) so the reader sees the difference directly.

**Two unbreakable rules for the report:**

1. **Every point is backed by a data plot.** No claim without a figure that
   shows it. A sentence that cites a number must point at the plot that plots it.
2. **Real data plots only.** Time-series, layer×step heatmaps, depth/per-layer
   curves, run overlays — the actual measured signals. **No synthetic summary
   plots** (traffic-light boards, gauges, "health dashboards", score bars). They
   hide the data behind a verdict; the reader must see the data.

Use **all** the data, not just the curated figures: the report's `atlas/` tree
plots every monitor family, and you pick the heatmaps that actually move. Text in
the report is clear, concise, readable — short paragraphs, each tied to a figure.

The HTML is the work product. It must include the verdict, healthy signals,
anomalies, causal chain, omitted-data notes, captions, and next steps inside the
report itself. The chat reply must not substitute for missing report prose. For
local CPU-only analyses, add a short "Scope" note in the HTML saying which
GPU-required parts were skipped and which existing artifacts, if any, were used.

---

## When to invoke

* A path with `events.out.tfevents.*` (TB run dir), or several such dirs.
* An `artifacts/hf_checkpoint/` dir, or an `analyze.json` / `joint.json`.
* "Проанализируй", "что с раном/чекпойнтом", "is this run healthy",
  "сравни раны", "проверь стабильность".

---

## Rules

1. **Deliver one HTML report; reply with its path + a short verdict.** Never
   paste the HTML or dump the full analysis into chat.
2. **Every claim → a plot, every plot → a caption.** If you can't point at a
   figure, make/find one (atlas, heatmap, compare overlay) or drop the claim;
   and every plot you place carries a super-short clear caption (≤ ~8 words).
3. **Real data plots only** — no traffic-light / gauge / "dashboard" summary art.
4. **Numbers are evidence; commit to a verdict.** No "looks interesting" /
   "worth watching". Each cited number is anchored (vs a threshold or the run's
   own sister-group p50) and labelled `anomaly` / `expected` / `inconclusive`.
5. **Never name artifact files in the report.** The CSVs / JSONs are *your*
   inputs — the reader never sees them. Cite the fact, not its source: write
   "max_vio at L17 climbs from 1.2 to 9 after step 8k", never "see
   `run_comparison_diffs.csv`" or "per `max_vio_comparison.csv`". No filenames,
   no `.csv`/`.json`/`.png` paths in the prose.
6. **Use short run labels.** Real run names are huge and 90% identical;
   `instability.compare` prints a "run key" (short alias → full name) and uses the
   aliases in every figure. Use those aliases in the report, and include the run
   key once (in the comparison intro) so the reader can map them back.
7. **Use all the data.** Browse `figures/atlas/` and the CSVs; don't stop at the
   14 curated figures. Note in one sentence anything you checked and dismissed.
8. **Cover what's healthy too**, and *why* it's healthy — not only the anomalies.
9. **For several runs: comparison is mandatory and paired.** Run the cross-run
   `compare` **and** a per-run diagnosis on each, then fuse into one HTML; show
   every divergence with the same plot for every run.
10. **Ask before jobs, then obey the answer.** GPU producers (`weights analyze`,
    `weights joint`, `weights batch`) must never be run directly in the local
    terminal; submit them as cluster jobs only after the user permits cluster
    submission. For full TB/CPU analyses with cluster permission, submit
    `instability/*` and `weights report|compare|dashboard|story` as jobs too;
    local shell work is for inspecting artifacts, writing story configs, and
    preparing submitters. If the user chooses local CPU-only, run all applicable
    non-GPU framework commands locally and still produce the required HTML.
11. **Clean up temporary code and logs.** After the report is built, remove
    ad-hoc submitters, temporary Python wrappers, scratch configs, transient job
    helper scripts, manual logs, and other intermediate files that are not
    deterministic framework outputs. Allowed final artifacts are only the single
    HTML report and deterministic outputs produced by the stability framework
    under the selected `output/<exp>/...` directory.

---


## Local CPU-only mode

Use this mode only when the user does not permit cluster submission. It is not a
reason to stop early. Run every non-GPU command that can add evidence from the
provided inputs:

* TB logs: `instability.report`, `instability.compare`, `instability.temporal`,
  `instability.eda`, `instability.projections`, `instability.heatmap`, and
  `instability.story` are eligible when their inputs are available and the user
  has not narrowed the request.
* Existing checkpoint JSONs: `weights report`, `weights compare`,
  `weights dashboard`, and `weights story` are eligible. Do not run
  `weights analyze`, `weights joint`, or CUDA-backed `weights batch` locally.
* Missing GPU-produced JSONs: state in the HTML scope note that those sections
  were skipped because cluster submission was not permitted and no existing
  JSONs were provided.

Use the same local environment that would be used to submit jobs: the active
cloud conda/Python environment for cloud workflows, and global `python3` for O
workflows. Keep output paths explicit under `sber_scripts/analytics/stability/output/...`;
never write to `/tmp` or into the source run/checkpoint directory.

---

## Job submission (mandatory)

The agent must submit stability-analysis work as cluster jobs **only after the
pre-flight permission gate allows cluster submission**. Do not replace the
workflow above with a generic "run this all-in-one script" instruction: choose
the specific capability, write the smallest submitter for that request, submit
it, wait for completion when asked, then build the required HTML story from the
artifacts.

A single job may run several ordered framework commands when that is the natural
unit of work: for example `weights analyze` followed by `weights joint` for one
checkpoint, or `instability.report` followed by `instability.heatmap` and
`instability.story` for one TensorBoard run. Keep the order explicit and print
every command before executing it so the job log is auditable.

Generated submitters should usually have two modes:

* default mode: construct `AnyScriptJob` objects and call `.submit()`;
* `--run-tools` mode: execute the actual `python3 -m ...` commands in order.

Recommended command helper inside `--run-tools`:

```python
def run(title: str, argv: list[str]) -> None:
    print(f"\n===== {title} =====", flush=True)
    print(" ".join(argv), flush=True)
    subprocess.run(argv, check=True)
```

### Cloud jobs (default)

Assume cloud unless the user explicitly asks for O or provides O-specific paths.
The local submitter should be run from the active workspace user's Python/conda
environment. On shared notebooks, infer the workspace user from the current
workspace path, not from the Linux username; source that user's `.zshrc` or
`.bashrc` if needed so the same conda environment used for `job_submitter` is
active.

If the user's shell exports HTTP proxy variables, unset them around the local
submitter invocation; otherwise submit can fail before a job id is created:

```bash
env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY \
    -u https_proxy -u http_proxy -u all_proxy \
    python3 SUBMITTER.py
```

After the submitted job is no longer needed, delete the local ad-hoc submitter
file and any scratch wrapper scripts/log captures you created. Keep the
framework's deterministic output directory and the final HTML report.

Cloud jobs use `queue_name`, not an O region:

```python
from contrib.job_submitter.job_submitter.jobs.any_script import AnyScriptJob

job = AnyScriptJob(
    user_login=USER_LOGIN,
    exp_name=EXP_NAME,
    script="python3 -m <package>.<command> <args> --out-dir <OUT_DIR>",
    extra_env={
        "PYTHONPATH": STABILITY_DIR,
        "PYTHONUNBUFFERED": "1",
        "MPLBACKEND": "Agg",
    },
    n_gpus=N_GPUS,
    n_workers=1,
    queue_name="pretrain-ml",
    priority_class="medium",
    script_type="binary",
    max_retry=None,
)
print(job.submit())
```

Use `n_gpus=1` for `weights analyze` / `weights joint` unless there is a clear
reason to request more. TensorBoard jobs are CPU-heavy; set `n_gpus=0` if the
submitter accepts it, otherwise use the minimum accepted GPU count.

### O jobs

Use O only when the user asks for it or the paths/settings are clearly O-specific.
Valid regions are normally `o1` and `o5`. O jobs use `region` and the matching
base image; they do not use `queue_name`. Inside the job script, use `python3`;
plain `python` is not a safe alias on O.

```python
from contrib.job_submitter.job_submitter.jobs.any_script import AnyScriptJob
from contrib.job_submitter.job_submitter.jobs.image_registry import LLMF_BASE_IMAGES

job = AnyScriptJob(
    user_login=USER_LOGIN,
    region=REGION,  # "o1" or "o5"
    exp_name=EXP_NAME,
    script="python3 -m <package>.<command> <args> --out-dir <OUT_DIR>",
    extra_env={
        "PYTHONPATH": ":".join([LLM_F_ROOT, STABILITY_DIR]),
        "PYTHONUNBUFFERED": "1",
        "MPLBACKEND": "Agg",
    },
    n_gpus=N_GPUS,
    n_workers=1,
    script_type="binary",
    priority_class="medium",
    base_image=LLMF_BASE_IMAGES[REGION],
    job_logdir=JOB_LOGDIR,
    slurm_logdir=SLURM_LOGDIR,
    max_retry=None,
)
print(job.submit())
```

### Example submitters

The repository may contain `*_example.py` submitters for cloud/O and weights/TB.
Treat them only as templates for job construction and command ordering. Do not
tell the user to run an example blindly, and do not turn this skill into "run the
big example script". Copy the relevant pattern, fill in `USER_LOGIN`, workspace,
input paths, output paths, labels, and requested capabilities, then submit the
job(s). Remove the copied/ad-hoc submitter after completion; it is not a final
artifact.

---

## Toolkits & commands

Outputs land under `output/<experiment>/<subdir>/`. Pass `--out-dir`
(instability, dashboard) or `--out` (weights report/story) explicitly; never
write to `/tmp` or into the run directory.

### `instability/` (TB logs)

| script | what it does | input | key outputs |
|---|---|---|---|
| `instability.report` | full single-run diagnostic: 14 curated PNGs + an **atlas** (one layer×step heatmap per monitor family — plot-everything) + `summary.md` + per-figure CSVs | one TB dir | `OUT/figures/01–14_*.png`, `OUT/figures/atlas/<category>/<family>.png`, `OUT/data/*.csv\|md`, `signals.json`, `summary.md` |
| `instability.compare` | **N-run diff** — aligns every monitor across runs, ranks `(tag, layer)` by cross-run divergence at the shared reference step; `--steps`/`--n-steps` adds a divergence trajectory; writes run-overlay figures (paired plots) | TB root + `--runs A B C…` (2+) | `run_comparison_diffs.csv`, `…_trajectory.csv`, `…_group_summary.csv`, `max_vio_comparison.csv`, `max_vio_by_layer.csv`, `figures/` overlays |
| `instability.temporal` | lead/lag (forward-lag predictive F-test on absolute levels, cross-correlation, Lasso AR-X) + event-study around growth_onset — answers *what leads to what* | sample of runs | `granger_results.csv`, `ccf_results.csv`, `lasso_arx_results.csv`, `event_pre_onset_results.csv` |
| `instability.eda` | many-run inventory + monitor↔max_vio correlations | TB root | `run_inventory.csv`, `run_summary.csv`, `metric_correlations.csv` |
| `instability.projections` | per-projection grouping of eda+temporal | sample of runs | `projection_*.csv` + PNGs |
| `instability.heatmap` | ad-hoc `layer × step` heatmap + CSV for any `<tag>/layer_N` family | one TB dir | `<tag>.png` + `.csv` |
| `instability.story` | render the HTML report from a config JSON + a figures folder; auto-builds a filterable gallery of every PNG | config + figures dir | one HTML |

### `weights/` (checkpoints)

| subcommand | what it does | input | output |
|---|---|---|---|
| `weights report` | direction-aware text report; many analyze files of one run → DELTAS / `--trajectory` (iteration progression) | JSON(s) | stdout |
| `weights compare` | **cross-RUN diff** — ranks where N runs disagree per `(tensor, metric)`, grouped by role/proj/block, direction-aware `worst_run`; multi-iteration files add a divergence trajectory | 2+ analyze.json | stdout + `--csv-dir` (`run_comparison_*.csv`) |
| `weights dashboard` | PNG tree from analyze.json (+ joint.json): per-metric overview, per-group layer profiles, per-expert×layer heatmaps, alignment scatter, abs-hist | JSON(s) + `--out-dir` | PNG folder |
| `weights story` | render the HTML report from a config JSON + a dashboard folder | config + dashboard dir | one HTML |

---

## Workflow — single TB run

Goal: one `story.html` with the full causal diagnosis.

```bash
cd llm-foundry/sber_scripts/analytics/stability
python3 -m instability.report --tb-dir /path/to/RUN \
    --out-dir output/<exp>/report_<run>
```

1. **Read `data/summary.md` first** — depth correlations + findings that anchor
   everything. Then the curated figures `01–14` for the usual suspects.
2. **Ground every flagged signal** in its `data/<NN>_*.csv`; note which layers
   drive it (top vs bottom of the stack = different failure modes).
3. **Use all the data**: browse `figures/atlas/<category>/` and pick the
   layer×step heatmaps that actually move. This is where families the curated
   figures don't cover (residual contribution, moments, massive activations,
   per-projection norms, effective_lr, …) show up.
4. **Establish causality — "what leads to what":** find the signal that moved
   *first*. Use `02_max_vio_heatmap` + the atlas heatmaps to read the **timing**
   (which family/layer departed baseline earliest), `14_spike_clusters` for
   co-firing spikes, `12_phase_diagnosis` for regime changes, and the
   `summary.md` depth correlations. State the order ("grad-norm spikes at L17
   from step 8k precede the max_vio rise at 9k"), not just the endpoint.
5. **Anchor** every value against the run's own sister-group p50 / trajectory and
   the rough thresholds in "Anchors" before calling anything an anomaly.
6. **Write the story config** (schema below) covering: a verdict `summary`
   (mechanism, not location), `metric_sections` for what's healthy and what
   moved, `findings` (one per issue — each with cause + what-detects-it +
   what-led-to-it, and the plot(s) that show it), `next_steps`. **Every section
   names its image.**
7. **Render and reply:**

```bash
python3 -m instability.story output/<exp>/report_<run>/story_config.json \
    output/<exp>/report_<run>/figures \
    --out output/<exp>/report_<run>/story.html
```

Reply: the `story.html` path + a 2–3 sentence verdict. Done.

---

## Workflow — several TB runs

Goal: one combined `story.html` = full per-run diagnosis **+** run-vs-run
comparison with paired plots. Three mandatory steps.

**Step 1 — compare the runs.** Ranks where they actually diverge and writes
run-overlay figures (all runs on one axis = paired by construction).

```bash
python3 -m instability.compare --tb-root /path/to/tb_parent \
    --runs RUN_A RUN_B RUN_C \
    --out-dir output/<exp>/compare_<set>
# --runs-pattern 'glob' to select; --steps 5000 10000 25000 (or --n-steps 5)
# adds a divergence trajectory → WHEN the gap opened.
```

Read `run_comparison_group_summary.csv` (which families differ most —
`max_divergence`, `highest_run_mode`), then `run_comparison_diffs.csv` (ranked
`(tag, layer)` with each run's `val__<run>`, `divergence`, `peak_step`,
`highest_run`/`lowest_run`; `signed_diff` for two runs),
`run_comparison_trajectory.csv` (cite the step where `divergence` leaves ~0), and
`max_vio_comparison.csv`. **`divergence`** = relative spread across runs,
`(max−min)/mean|·|`; 0 = identical, ~1 = the gap is as big as the values.

**Step 2 — diagnose every run on its own** (the comparison says *where* they
differ; the per-run report says *why each one* behaves that way):

```bash
for R in RUN_A RUN_B RUN_C; do
  python3 -m instability.report --tb-dir /path/to/tb_parent/$R \
      --out-dir output/<exp>/report_$R
done
```

Run the single-run workflow on each: read each `report_$R/data/summary.md`,
ground its signals, browse its `atlas/`.

**Step 3 — one combined HTML.** Point the story at the experiment dir with
`--figures-url-prefix .` so its gallery sweeps **every** PNG (compare overlays +
all per-run figures + atlas) with no copying; reference images by their path
relative to `output/<exp>/`:

```bash
python3 -m instability.story output/<exp>/story_<set>_config.json \
    output/<exp>/ --figures-url-prefix . \
    --out output/<exp>/story_<set>.html
```

The config carries:

* `summary` — the cross-run **mechanism**: which run is the outlier, what failed,
  where in the stack, and when the gap opened (from the trajectory).
* `comparison` block — the diff table (one column per run, worst cell
  highlighted) from the top `run_comparison_diffs.csv` rows, with
  `"image": "compare_<set>/figures/01_divergence_by_group.png"`.
* `findings` — **paired**. For each divergence, show it the same way for every
  run. `compare` writes two views per top series: a **side-by-side panel**
  (`compare_<set>/figures/03_top*_panels_<tag>.png` — one sub-plot per run, shared
  y-axis, run alias as the panel title — the default, no giant legend) and an
  overlay (`02_top*` — all runs superimposed). Prefer the panel; add the overlay
  when the superposition tells the story better. `00b_max_vio_panels.png` is the
  side-by-side max_vio. You can also pair the *same* atlas heatmap from each run
  (`report_RUN_A/figures/atlas/<cat>/<fam>.png`, …). Never "A is bad" with A's
  plot only — show all runs the same way.
* `next_steps`.

Reply: the single `story_<set>.html` path + a 2–3 sentence verdict. Findings must
cite diff rows verbatim ("`activations_amax` gmm_down L5 ends at 39 in RUN_A vs 15
in RUN_B/C, divergence 1.05, gap opens ~step 1950; sister layers within 10%").

---

## Workflow — checkpoints (`weights/`)

You usually start from an `analyze.json` (+ optional `joint.json`). If those JSONs
are missing, submit `weights analyze` / `joint` as GPU producer jobs first (see
"How fresh JSONs get made"). Do not run GPU producers directly in the local
terminal.

**Single checkpoint:**

```bash
python3 -m weights report  <out>/analyze.json <out>/joint.json        # read it
python3 -m weights dashboard <out>/analyze.json <out>/joint.json --out-dir <out>/dashboard
ls -R <out>/dashboard                                                # pick images
python3 -m weights story <out>/story_config.json <out>/dashboard --out <out>/story.html
```

`weights report` emits: inventory by `(role, proj, block)`; **HEADLINES** (worst
tensor per curated metric, direction-aware); per-metric `min/p50/p95/max` +
worst-N; per-expert hot spots. NaN/Inf is terminal — report it. For each
HEADLINE group, dump the per-layer profile (confirming clean sister layers is
part of the finding) and pick the matching dashboard PNG for the story.

**Several runs (or iterations):** use `weights compare` for the cross-run diff,
then build one combined story the same way as the TB multi-run flow — point
`weights story` at the parent dir with `--dashboard-url-prefix .` and pair each
divergence with the same per-run dashboard PNG (e.g. each run's
`02_groups/<role>__<proj>__<block>/<metric>.png`).

```bash
python3 -m weights compare RUN_A/.../analyze.json RUN_B/.../analyze.json \
    RUN_C/.../analyze.json --csv-dir output/<exp>/compare_<set>
# RUN_*/ep0_ba*/analyze.json across iterations → divergence trajectory.
```

`weights report --trajectory ROLE[/PROJ[/BLOCK]]` and DELTAS track *iterations of
one run*; `weights compare` is for *different runs*. Curated metrics (direction):
`n_nan`/`n_inf` [alert], `max_abs`/`std`/`mean_abs`/`kurt_excess` [high], `skew`
[|abs|], `rowcol.{row,col}.max_over_median` [high], `spectral.stable_rank` [low],
`spectral.effective_rank_entropy_topk` [low], `spectral.spectral_norm` [high].

---

## Story config schema

You write the *content* (a small JSON — prose + image refs + verdicts); the story
script owns the *form* (CSS, layout, gallery). All fields optional except
`title`. Image paths are relative to the figures/dashboard folder you pass (or to
`output/<exp>/` when you use `--figures-url-prefix .` for combined reports).
**Every `findings`/`metric_sections`/`comparison` entry should name an image** —
that's the "every point backed by a plot" rule, enforced in the config.

```json
{
  "title": "stability — PapayaSeek-9B / lr_explosion",
  "subtitle": "50B tokens; max_vio blows up at step ~9k",

  "summary": "Verdict + mechanism in 3–5 sentences: what happened, the outcome, and the causal chain (X at step S drives Y, observed as Z=<value>).",

  "comparison": {
    "runs": ["RUN_A", "RUN_B", "RUN_C"],
    "intro": "Three runs, same model; RUN_A is the outlier, gap opens ~step 1k.",
    "image": "compare_set/figures/01_divergence_by_group.png",
    "diffs": [
      {"metric": "activations_amax/...layers.5...gmm_down_input.0", "layer": 5,
       "group": "activation_amax",
       "values": {"RUN_A": 39.2, "RUN_B": 14.9, "RUN_C": 15.8},
       "divergence": 1.05, "note": "RUN_A lone outlier — sisters within 10%"}
    ]
  },

  "metric_sections": [
    {"metric": "max_vio per layer", "direction_note": "warn > 4",
     "prose": "Healthy/flat until step 8k, then L16–17 climb — one sentence.",
     "image": "02_max_vio_heatmap.png", "caption": "max_vio per layer × step"}
  ],

  "findings": [
    {"title": "Layer 17 grad-norm spike → max_vio blowup",
     "verdict": "anomaly",
     "prose": "What it is + what detects it (l2_norm/grad L17 crosses 5× at step 8k) + probable cause (router collapse onto one expert) + what led to it (effective_lr L17 ramps from step 6k, precedes the spike). Anchored to sister layers.",
     "images": [
       {"src": "figures/atlas/grad_norm/...layer_L_....png",
        "caption": "grad-norm per layer: L17 spikes at 8k"},
       {"src": "10_router_dynamics.png", "caption": "router collapses onto 1 expert at 8k"}
     ]}
  ],

  "next_steps": ["Concrete rerun/query, each naming a file or command."]
}
```

**Every plot gets a caption.** An image is either a plain path or
`{"src": "...", "caption": "..."}`; the caption renders under the plot. Write a
**super-short but clear** caption for *every* plot you put in the report — what
the reader is looking at in ≤ ~8 words ("max_vio per layer × step", "RUN_A vs
sisters: gmm_down L5", "grad-norm L17 spikes at 8k"). `overview_image`,
`metric_sections[].image`, every entry in `findings[].images`, and the
`comparison`/`joint` images all take a caption (`overview_caption` /
`image_caption` for the single-image fields).

`weights story` additionally accepts a `joint` block
(`{"prose": ..., "image": "03_joint_alignment.png", "image_caption": "..."}`);
its findings images live under `02_groups/<role>__<proj>__<block>/`.

Per-row in `comparison.diffs`, the worst run is auto-highlighted (direction-aware:
lowest is red for `stable_rank`-style "low = bad" metrics, highest otherwise).
Don't hand-write HTML; don't reference PNGs that don't exist (`ls -R` first);
don't fabricate a `joint` block the report didn't produce.

---

## Diagnosis content — for each finding

Make each finding a small causal story, not a label:

* **What** — the tensor/monitor (`role/proj` + `layer/block`, or `tag` + layer)
  and the value, anchored (vs sister p50 or threshold).
* **What detects it** — the specific metric and the line it crossed
  (`stable_rank` 1.8 vs sister p50 28; `max_vio` > 4 from step 9k).
* **Probable cause** — a *mechanism*: init / layer fusion / scaling (arch), LR
  schedule / momentum (optimizer), seq-len warmup / packing (data), quant
  rounding / kernel eps (kernel). Briefly argued, not a guess list.
* **What leads to it in the data** — the upstream signal and the step/layer where
  it started, from timing (atlas/heatmap), spike clusters, phase changes,
  `temporal` lead/lag, or (multi-run) the divergence trajectory's onset step.
* **Verdict** — `anomaly` (out of range *and* confirmed by profile/trajectory) /
  `expected` (high but normal for the family, or within ±20% of sister p50) /
  `inconclusive` (end with one disambiguating query). No fence-sitting.

---

## Anchors — "is this number bad?"

Rough, architecture-dependent; always cross-check the run's own sister-group p50
and trajectory first.

* `spectral.stable_rank` healthy `[D,D]`: 30–70% of `min(shape)`. Sub-10 on a
  multi-hundred-dim matrix = real rank-collapse.
* `rowcol.{row,col}.max_over_median` healthy 1.5–3.0; >5 concentration; >10 one
  direction dominates.
* `base.kurt_excess` healthy 0–3; >5 heavy-tailed; >50 a single outlier (dead).
* `base.max_abs` on bias/SSM scalars (`dt_bias`, `A_log`, `o_norm`): routinely
  1–10 — don't alarm; alarm on matrix-valued projections.
* `base.n_nan` / `n_inf`: zero is the only acceptable value (terminal).
* TB `max_vio`: spikes vs running min = real instability; flat late = healthy
  convergence.
* TB `clip_share_gate` / `clip_share_up`: any layer crossing 1e-3 means
  sandwich-norm pressure reaches the clip threshold.

---

## Tensor role taxonomy (weights)

Roles from `analyze.classify()`; each record also has `proj`, `layer`, `expert`
(MoE only), `block` (`main` / `mtp` — separate populations, compare within block).

* **Top-level** — `embedding`, `lm_head`, `final_norm` (proj `weight`/`gate_up`/`gate_down`).
* **MTP head** — `mtp` role (`eh_proj`, `enorm`, `hnorm`, `shared_head`, local `embed_tokens`); `block="mtp"`.
* **Attention** — `attn_q/k/v/o/qkv` (Llama-style, rare here); **MLA** single `mla` role, proj `q_a`/`q_b`/`kv_a`/`kv_b` (MLA layernorms → `layernorm`, proj `mla_*`).
* **Linear-attn / Mamba / DeltaNet** — single `linear_attn` role (proj `qkv`, `qkv_conv1d`, `conv1d`, `gate`, `z`, `in_qkvz`, `in_ba`, `ba`, `o`, `out`); 1-D SSM scalars (`A_log`, `dt_bias`, `norm`, `o_norm`) → `ssm_param`.
* **MoE** — `moe_router` (proj `weight`/`bias`); `moe_expert` (one record per expert, `expert=<i>`, proj `gate`/`up`/`down`); `moe_shared_expert`.
* **Dense MLP** — `dense_mlp` (proj `gate`/`up`/`down`).
* **GatedRMSNorm** — `layernorm` role, proj `input_layernorm_*` / `post_attention_layernorm_*` / `post_self_attn_layernorm_*` / `post_feedforward_layernorm_*` in `*_weight`/`*_gate_up`/`*_gate_down`.
* **`unknown`** — catch-all; if any appear in a fresh report, `classify()` needs a regex — flag it in the report.

---

## How fresh JSONs get made (job-submitted producers)

If a checkpoint has no `analyze.json` / `joint.json` yet, create and submit a
producer job instead of running GPU code in the local terminal. The job should
write JSONs to an explicit output directory and leave the checkpoint directory
untouched.

Minimal producer payload for one HF export:

```bash
python3 -m weights analyze HF_DIR --out OUT_DIR/analyze.json --device cuda
python3 -m weights joint HF_DIR --out OUT_DIR/joint.json --device cuda -k 4
```

For several checkpoint iterations under one shared checkpoint-run directory, use
`weights batch` only with exact iteration patterns or explicitly enumerated
iterations; do not use a broad pattern that silently includes extra checkpoints.
If checkpoints are unrelated training runs, submit explicit `analyze` / `joint`
commands per HF directory instead.

`PYTHONPATH` must include `llm-foundry/sber_scripts/analytics/stability`; on O include the
repo root too if imports require it. The job payload is a binary script invocation
(`script_type="binary"`), and the command inside that payload uses `python3`.
After the job finishes, continue with `weights report`, `weights dashboard`, and
`weights story` to produce the required single HTML report.

---

## Cache & fallbacks

TB cache: `~/.cache/instability_tb/{md5}.pkl.gz`, keyed on `(path, mtime_ns,
size)` per event file; auto-invalidates on change. Cold parse of a ~450-event TB
≈ 100s; warm ≈ 0.1s. `from instability.tb_helpers import clear_cache` or
`rm -rf ~/.cache/instability_tb/` to reset.

Ad-hoc cuts the tools don't cover:

```bash
jq '.meta' analyze.json
jq '.tensors | to_entries[0].value' analyze.json      # one full record
jq -r --arg role linear_attn --arg proj ba \
   '.tensors|to_entries[]|.value|select(.role==$role and .proj==$proj)
    |[.layer,.block,.base.max_abs,.spectral.stable_rank,.base.kurt_excess]|@tsv' \
   analyze.json | sort -n
python3 -m instability.heatmap --tb-dir RUN --out-dir output/<exp>/heatmap_<run> \
   --tags moe_load_balance moe_maxvio --threshold 4   # any <tag>/layer_N family
```

---

## Anti-patterns (any = redo before delivering)

* Submitting a cluster job before asking the required pre-flight permission question.
* Running GPU framework producers directly in the local terminal.
* Running framework commands directly in the local terminal instead of via a
  cluster job when cluster execution was permitted.
* Stopping after "cluster not permitted" instead of running the applicable
  local CPU-only framework commands and building the shorter HTML report.
* Using plain `python` inside an O job script; use `python3`.
* Treating an example submitter as the deliverable instead of creating the requested HTML story.
* **No HTML, or analysis dumped in chat.** The deliverable is `story.html`; chat
  is path + 2–3 sentence verdict.
* Leaving ad-hoc submitters, temporary Python scripts, scratch configs, manual
  logs, or other non-framework intermediate files in the repo after delivery.
* **A claim with no plot**, or a "summary"/"health"/traffic-light/gauge plot
  standing in for the data. Show the measured signal.
* **Only the curated 14 figures** — you ignored the atlas and the CSVs, so you
  didn't use all the data.
* **A comparison without paired plots** — "RUN_A looks worse" with A's plot only.
  Show the same plot for every run (side-by-side panel, overlay, or per-run heatmap).
* **Naming an artifact file in the prose** — "see `run_comparison_diffs.csv`",
  "per `max_vio_comparison.csv`", any `.csv`/`.json`/`.png` path. Cite the fact;
  the reader never sees your inputs.
* **Full run names in the text or legends** — use the short run-key aliases.
* **A plot with no caption** — every figure in the report needs a ≤ ~8-word caption.
* **A finding without a cause and a what-led-to-it** — a label, not a diagnosis.
* "Interesting" / "worth watching" / "I'd look at X" — run the query and commit,
  or mark `inconclusive` with a disambiguator.
* Treating bias/SSM scalars (`dt_bias`, `A_log`, `o_norm`) as `max_abs`/`std`
  anomalies — routinely 1–10.
* Calling alignment-collapse from `joint.json` when `alignment` is empty.
* Referencing PNGs that don't exist, or hand-writing HTML instead of using the
  story tool.
