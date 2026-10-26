# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- **Datasets sharing one judge are now judged through a single thread pool.** Judge identity is `(model_name, api_base, api_type, api_key)`; request-body parameters (`system_prompt`, `prompt_template`, `temperature`, `max_tokens`, `regex_pattern`, ...) may differ per dataset within a group. Battles and slow retries of all grouped datasets interleave in one `ThreadPoolExecutor` sized by the group's largest `parallel_api_calls`, so straggler retries of one dataset no longer idle the pool while other datasets wait their turn. `parallel_api_calls` is thereby a per-judge limit, not a per-dataset one.
- **Runner-scoped `JUDGE_SETTINGS` keys (category namespacing).** Each `JudgeRunner` registers its categories as `ds<N>::<category>`, so two datasets that share a category name but use different judge system prompts never collide on the shared arena-hard-auto registry. All registry writes happen before the judging pool starts; worker threads only read. Upstream per-category judge protocols (e.g. `hard_prompt`, `creative_writing`) are still inherited, and pristine upstream entries are never mutated anymore.

### Changed
- **Judging is executed after the per-dataset generation loop** (grouped by judge identity) instead of interleaved with it. For `--phase full` this means all generations happen first, then all judgments; per-dataset artifacts and summaries are unchanged. Input validation for judging now happens before any judging starts (fail-fast).
- **Persisted judgment records always carry the original question category.** Previously strict-retried battles recorded the internal `__strict_retry__` category in `judgment.jsonl`.
- With `judge.fail_on_error: true` and several datasets pooled under one judge, the error is raised only after the whole group finishes AND every dataset of the group has its `judgment.jsonl` and `judgment_summary.jsonl` persisted. As before, the raise aborts the run, so datasets of later judge groups are not judged (groups execute in first-appearance order of the config).

## [0.5.0] - 2026-05-26

### Added
- **Arbitrary config keys are now passed through to the API payload.** Any key in `model` / `baseline` / `judge` that is not in the internal-keys allowlist (`name`, `path`, `artifacts_dir`, `api_base`, `api_key`, `api_type`, `num_retries`, `parallel_api_calls`, `n_parallel`, `system_prompt`, `path_to_sys`, `request_timeout`) is forwarded as-is to the chat-completions endpoint. This enables provider-specific parameters like `reasoning: {effort: "high"}`, `verbosity`, `top_p`, `seed` without code changes. `temperature` and `max_tokens` now flow through the same path instead of being special-cased.
- **`request_timeout` configuration parameter.** Default remains `120` seconds for backward compatibility. Slow thinking-style models (GLM, o1-style, Claude thinking) can set e.g. `request_timeout: 1800` in the config to avoid read timeouts on long generations.
- **New default baselines bundled in `data.tar.zst`:** generations for `claude-opus-4-6-thinking-max`, `gemini-3-1-pro-preview`, `gemini-flash`, `glm-5-1`, `minimax`, `openai-5-4`, `openai-5-4-mini`, `qwen-3-5-122b-a10b` across `sbs-gc3-ultra-vs-alice-ai`, `arena-hard-logs-v3`, `arena-hard-ru`, `ru-llm-arena`, `validator-sbs-pollux`. Pick any of them as `baseline.name` without copying configs or regenerating.

### Fixed
- **OmegaConf containers are normalized to plain Python types before JSON serialization.** Nested config blocks such as `reasoning: {effort: "high"}` no longer break `requests.post(json=payload)` due to `DictConfig` / `ListConfig` not being JSON-serializable.
- **`request_timeout` and `artifacts_dir` are excluded from generation-identity matching.** Both are service parameters: changing them no longer forces a new run directory, and dredd correctly resumes into the existing run that matches the actual generation parameters.

## [0.4.0] - 2026-05-26

### Changed
- **BREAKING (methodology):** Default judge model changed from `gpt-oss-120b` to `minimax-m2.7`, and the default judge system prompt changed from `prompts/judge_arena_hard_v01_ru.txt` to `prompts/judge_minimax_m2_7_sbs.txt`. Judge model and judge prompt are coupled by design and ship as a pair.
- Removed per-dataset `judge.system_prompt` overrides for `ru-llm-arena` and `sbs-gc3-ultra-vs-alice-ai`; the global judge prompt is now used for every dataset.
- Metrics produced with the new judge/prompt pair are **not directly comparable** with historical Dredd runs.

### Added
- New judge prompt `judge_minimax_m2_7_sbs.txt` shipped inside `data.tar.zst` (`data/prompts/`), alongside the other bundled prompts.

### Migration notes (PAM)
- `InferenceManager.MODEL_NAME_PARAMS_MAPPER` in `pretrain-auto-metrics` must include an entry for `minimax-m2.7`; otherwise the judge inference phase will fail to start when Dredd defaults are used. This is delivered in a follow-up PAM change.

## [0.3.0] - 2026-03-27

### Added
- **Automated Ties:** `JudgeRunner` now detects identical candidate and baseline responses and records an automated tie (`[[A=B]]`) without calling the judge API, saving time and tokens.
- **Smart Resume Logic:** The generation phase now detects existing results and only generates answers for missing UIDs, preventing redundant work.
- **Run Versioning:** Changes in model parameters (e.g., temperature) now trigger the creation of a new timestamped directory for generations instead of resuming or overwriting existing ones.
- **Data Robustness:** `load_jsonl` and summary generation now resiliently handle empty lines, whitespace, and malformed JSON records.
- **Comprehensive Testing:** Added a suite of integration and unit tests covering resume logic, parameter identity, automated ties, and data robustness. Total test coverage increased significantly.
- **Deterministic DATA_ROOT:** Unified initialization of data paths across the project for better predictability.
- **Auto-unpacking:** Support for automatic unpacking of `data.tar.zst` on first run.

### Changed
- Refactored `JudgeRunner` counters (`ok_count`, `fail_count`, `precheck_fail_count`) for consistent progress reporting.
- Improved judging skip visibility: added prominent warnings and instructions when judging is skipped due to existing output files.
- `APIModel.generate_answer` now raises `RuntimeError` after exhausting retries, allowing runners to handle failures gracefully per-task.
- Unified `n_parallel` and `parallel_api_calls` into a single configuration parameter.
- Enhanced smart classification in `JudgmentSummary` to correctly handle comparison operators (e.g., `A<B`, `A>B`).

### Fixed
- Fixed bug where identical response texts were treated as "precheck errors" and excluded from the final win-rate calculations.
- Fixed inconsistent path resolution for datasets and artifacts across different environments.
- Fixed logic in `JudgmentSummary` to correctly interpret outcomes from raw judge text when structured score fields are missing.

## [0.2.0] - 2026-03-17

### Added
- `CHANGELOG.md` — this file
- `.gitlab-ci.yml` — CI pipeline with `test:unit`, `test:integration`, `test:cov` jobs
- `load_prompt()` in `src/utils/io.py` — unified function for resolving system prompts
  from either a plain text string or a file path (absolute or relative)
- Project-level path constants block in `dredd.py` (`CONFIGS_DIR`, `DEFAULT_CONFIG`,
  `MERGED_CONFIG`, `ARTIFACTS_DIR`, `ARENAS_DIR`, `GENERATIONS_DIR`, `JUDGMENTS_DIR`)
- Documentation in `README.md` about merged config location and overwrite behavior

### Changed
- `main()` in `dredd.py` refactored: eliminated duplicate config loading, extracted
  `_require_config()` helper, config is now loaded once instead of twice
- `merge_dataset_configs()` now returns the full config (all top-level keys preserved)
  instead of a stripped-down object; the redundant second `OmegaConf.merge` in `run()`
  was removed
- All hardcoded path strings scattered across `dredd.py` replaced with named constants
- `model.system_prompt` and `baseline.system_prompt` now accept both plain text and a
  file path (same behavior as `judge.system_prompt`)
- `judge.py` now imports `load_prompt` from `src/utils/io` instead of defining its own copy
- `configs/default.yaml` and `configs/editable.yaml`: `path_to_sys` replaced with
  `system_prompt`

### Deprecated
- `path_to_sys` config key for `model` and `baseline` — emits a deprecation warning at
  runtime; use `system_prompt` instead

### Removed
- `load_system_prompt()` from `dredd.py` (superseded by `load_prompt` in `src/utils/io.py`)
- `load_prompt_from_file()` from `src/judges/judge.py` (superseded by `load_prompt`)
- `model_conf.system_prompt_text` intermediate field — system prompt is now resolved
  lazily inside `APIModel.generate_answer()`
