#!/usr/bin/env python3
import sys
import argparse
import logging
import os
import time
import json
import subprocess
from datetime import datetime
from pathlib import Path
from omegaconf import OmegaConf, DictConfig, ListConfig
from dotenv import load_dotenv

load_dotenv()

# ==============================================================================
# Project paths
# All default paths are defined here. Override DATA_ROOT via DREDD_DATA_ROOT env var.
# ==============================================================================
PROJECT_ROOT = Path(__file__).parent.resolve()
CONFIGS_DIR         = PROJECT_ROOT / "configs"
DEFAULT_CONFIG      = CONFIGS_DIR / "default.yaml"
MERGED_CONFIG       = CONFIGS_DIR / "merged_configs" / "merged_config.yaml"

# Artifact layout inside a model directory (model.path)
# e.g. <model.path>/artifacts/arenas/<arena>/generations/<run>/generation.jsonl
DEFAULT_ARTIFACTS_DIR = "artifacts"
ARENAS_DIR          = "arenas"
GENERATIONS_DIR     = "generations"
JUDGMENTS_DIR       = "judgments"


def _model_artifacts_root(model_conf: DictConfig) -> Path:
    """Return the artifacts root for a model: <model.path>/<artifacts_dir>/.

    When artifacts_dir is empty string, arenas are written directly into model.path.
    """
    base = resolve_path(model_conf.path)
    artifacts_dir = model_conf.get("artifacts_dir", DEFAULT_ARTIFACTS_DIR)
    if artifacts_dir:
        return base / artifacts_dir
    return base

# Add to sys.path for imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def _init_data_root() -> Path:
    """
    Determines and initializes DATA_ROOT.
    
    1. Reads from DREDD_DATA_ROOT env var, or defaults to PROJECT_ROOT / "data".
    2. If the path points to the project's default 'data' and it's missing, try to unpack the archive.
    3. Verify the resulting path for validity:
       - 'path/arenas' exists -> return path
       - 'path/data/arenas' exists -> return path/data
       - Otherwise: error.
    """
    env_val = os.environ.get("DREDD_DATA_ROOT")
    candidate = Path(env_val) if env_val else (PROJECT_ROOT / "data")
    
    # Unpack logic ONLY for the project-level 'data' directory if it's missing
    if not env_val and not (candidate / "arenas").exists():
        archive = PROJECT_ROOT / "data.tar.zst"
        if archive.exists():
            print(f"Data directory missing at {candidate}. Unpacking {archive}...")
            try:
                subprocess.run(
                    ["tar", "--zstd", "-xf", str(archive), "-C", str(PROJECT_ROOT)],
                    check=True
                )
            except Exception as e:
                print(f"Error unpacking data: {e}")

    # Final validation of the candidate path
    if (candidate / "arenas").exists():
        return candidate
    
    # Handle nested structure (the 'data/data' case)
    if (candidate / "data" / "arenas").exists():
        return candidate / "data"

    # If we reached here, the data directory is invalid or missing
    print(f"Error: Invalid data directory structure at {candidate}")
    print("Expected 'arenas/' directly inside it, or inside a 'data/' subdirectory.")
    return candidate # Fallback to candidate to avoid breaking everything immediately, but it's likely broken

DATA_ROOT = _init_data_root()
print(f"DEBUG: DATA_ROOT initialized at: {DATA_ROOT}")
print(f"DEBUG: Final DATA_ROOT used for benchmark: {DATA_ROOT}")

try:
    from src.models import create_model, APIModel
    from src.generators.runner import GenerationRunner
    from src.judges.judge import Judge
    from src.judges.runner import JudgeRunner, run_judge_runners
    from src.utils.io import load_jsonl, JsonlWriter
    from src.utils.judgment_summary import compute_judgment_summary, format_summary_table, save_judgment_summary_jsonl
except ImportError as e:
    print(f"Error: Could not import project modules. Detail: {e}")
    sys.exit(1)

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

def configure_logging(level_name: str = "INFO") -> int:
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    return level

configure_logging("INFO")
logger = logging.getLogger("dredd")

def resolve_path(path_str: str, source_path: Path = PROJECT_ROOT) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (source_path / path)

def resolve_judgment_path(cand_root: str, arena_name: str, baseline_name: str, artifacts_dir: str = DEFAULT_ARTIFACTS_DIR) -> Path:
    base = resolve_path(cand_root)
    if artifacts_dir:
        base = base / artifacts_dir
    return base / ARENAS_DIR / arena_name / JUDGMENTS_DIR / baseline_name / "judgment.jsonl"

# ---------------------------------------------------------------------------
# NEW: two clean helpers replacing find_existing_generations / setup_model_generation
# ---------------------------------------------------------------------------

def _generation_params(model_conf: DictConfig) -> dict[str, object]:
    """
    Parameters that define generation identity for matching run config.yaml.
    Service fields are excluded.
    """
    excluded_keys = {
        "name",
        "path",
        "api_base",
        "api_key",
        "num_retries",
        "parallel_api_calls",
        "n_parallel",
        "request_timeout",
        "artifacts_dir",
    }
    raw = OmegaConf.to_container(model_conf, resolve=True)
    if not isinstance(raw, dict):
        return {}
    return {k: v for k, v in raw.items() if k not in excluded_keys}


def _find_gen_in_base(gen_base: Path, target_params: dict[str, object]) -> Path | None:
    """
    Generation lookup policy:
    1) If default/generation.jsonl exists -> use it.
    2) Else look for run dirs with generation.jsonl + config.yaml matching target_params.
    """
    if not gen_base.exists():
        return None

    default_file = gen_base / "default" / "generation.jsonl"
    if default_file.exists():
        return default_file

    for run_dir in sorted(gen_base.iterdir(), reverse=True):
        if not run_dir.is_dir() or run_dir.name == "default":
            continue
        gen_file = run_dir / "generation.jsonl"
        cfg_file = run_dir / "config.yaml"
        if not gen_file.exists() or not cfg_file.exists():
            continue
        try:
            existing_cfg = OmegaConf.load(cfg_file)
            existing_params = OmegaConf.to_container(existing_cfg, resolve=True)
            if existing_params == target_params:
                return gen_file
        except Exception as e:
            logger.warning(f"Failed to parse generation config {cfg_file}: {e}")
    return None


def _find_gen_file(model_conf: DictConfig, arena_name: str) -> Path | None:
    """
    Finds generation.jsonl according to matching policy:
    - If model.path is set: search ONLY there.
    - Else: search ONLY in reference models (data/model_data/<name>/...).
    """
    name = model_conf.get("name", "unknown")
    target_params = _generation_params(model_conf)

    if model_conf.get("path"):
        gen_base = _model_artifacts_root(model_conf) / ARENAS_DIR / arena_name / GENERATIONS_DIR
        found = _find_gen_in_base(gen_base, target_params)
        if found:
            logger.info(f"Found generations for '{name}' in model.path: {found}")
        return found

    ref_base = DATA_ROOT / "model_data" / name / ARENAS_DIR / arena_name / GENERATIONS_DIR
    logger.info(f"Searching for reference generations for '{name}' at: {ref_base}")
    found = _find_gen_in_base(ref_base, target_params)
    if found:
        logger.info(f"Found reference generations for '{name}': {found}")
    return found


def _prepare_gen_file(model_conf: DictConfig, arena_name: str, overwrite: bool) -> tuple[Path, bool]:
    """
    For a model with API access: finds an existing generation file to resume,
    or creates a new timestamped run directory.
    Returns (path_to_generation.jsonl, is_new_run).
    """
    if not model_conf.get("path"):
        logger.error(f"model.path is required when api_base is set (model: '{model_conf.get('name')}').")
        sys.exit(1)

    gen_base = _model_artifacts_root(model_conf) / ARENAS_DIR / arena_name / GENERATIONS_DIR

    if not overwrite:
        target_params = _generation_params(model_conf)
        existing = _find_gen_in_base(gen_base, target_params)
        if existing:
            return existing, False

    run_dir = gen_base / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "config.yaml"
    OmegaConf.save(config=OmegaConf.create(_generation_params(model_conf)), f=cfg_path)
    logger.info(f"New generation run: {run_dir}")
    return run_dir / "generation.jsonl", True


def _generate(model_conf: DictConfig, arena_name: str, questions_path: Path, overwrite: bool) -> None:
    """
    Ensures generations exist for model_conf/arena_name.

    Two modes determined by whether model.path is set:
    - model.path is set   → we own this directory; generate (via API) or find existing there.
    - model.path not set  → reference directory (data/model_data/<name>/); read-only, never generate.
    """
    name = model_conf.get("name", "unknown")

    if not model_conf.get("path"):
        # Reference model: read-only, just verify generations exist.
        if _find_gen_file(model_conf, arena_name) is None:
            logger.error(f"Generations for '{name}' in arena '{arena_name}' not found. Check the search path above.")
            sys.exit(1)
        return

    # model.path is set — we own this directory.
    if _model_has_api(model_conf):
        gen_path, is_new_run = _prepare_gen_file(model_conf, arena_name, overwrite)
        if not is_new_run and gen_path.parent.name == "default":
            logger.info(f"Found default generations for '{name}', skipping generation (default is read-only).")
            return
        GenerationRunner(
            create_model(model_conf.name, model_conf),
            str(questions_path),
            str(gen_path.parent),
            overwrite=is_new_run,
            output_file=str(gen_path),
            parallel_api_calls=_resolve_parallel_api_calls(model_conf, default=1),
        ).run()
    else:
        # Has path but no API — look for existing generations inside model.path.
        if _find_gen_file(model_conf, arena_name) is None:
            logger.error(
                f"No generations for '{name}' in {model_conf.path} and no API configured. "
                "Set api_base in config or place generation.jsonl manually."
            )
            sys.exit(1)


def _model_has_api(model_conf: DictConfig) -> bool:
    has_api_base = bool(model_conf.get("api_base"))
    is_ift = model_conf.get("api_type") == "ift"
    has_ift_env = bool(os.environ.get("GIGACHAT_IFT_CLIENT_ID")) and bool(
        os.environ.get("GIGACHAT_IFT_CLIENT_SECRET")
    )
    return has_api_base or (is_ift and has_ift_env)


def _resolve_parallel_api_calls(cfg: DictConfig, default: int = 1) -> int:
    """Read parallel_api_calls from config; return default if missing or invalid."""
    value = cfg.get("parallel_api_calls")
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid parallel_api_calls=%r for %s; using default %s",
            value,
            cfg.get("name", "<unnamed-model>"),
            default,
        )
        return default
    if parsed < 1:
        logger.warning(
            "parallel_api_calls must be >= 1 for %s; got %s, using default %s",
            cfg.get("name", "<unnamed-model>"),
            parsed,
            default,
        )
        return default
    return parsed

def _judge_group_key(judge_conf: DictConfig) -> tuple[str, str, str, str]:
    """Identity of a judge for pooling datasets into one judging run.

    Datasets whose judge configs agree on these keys hit the same endpoint and
    are judged through a single shared thread pool. Everything else
    (system_prompt, prompt_template, temperature, max_tokens, regex_pattern...)
    travels with each request, so it may differ within a group.
    """
    return (
        str(judge_conf.get("model_name", "")),
        str(judge_conf.get("api_base", "")),
        str(judge_conf.get("api_type", "openai")),
        str(judge_conf.get("api_key", "EMPTY")),
    )


def get_questions_path(arena_name: str) -> Path:
    """
    Determines the path to the questions file.
    
    Looks for data/arenas/{arena_name}.jsonl
    
    Args:
        arena_name: Dataset name
        
    Returns:
        Path to the questions file
    """
    questions_path = DATA_ROOT / ARENAS_DIR / f"{arena_name}.jsonl"
    return questions_path

def merge_dataset_configs(config: DictConfig) -> DictConfig:
    """
    Expands each dataset entry with global judge/experiment/model/baseline defaults.

    Dataset-level overrides take priority over global ones.
    All other top-level config keys (e.g. merged_config_path) are preserved as-is.
    """
    datasets = config.get("datasets", [])
    if not datasets:
        return config

    common = {
        "judge": config.get("judge", {}),
        "experiment": config.get("experiment", {}),
        "model": config.get("model", {}),
        "baseline": config.get("baseline", {}),
    }

    merged_datasets = []
    for dataset in datasets:
        base = OmegaConf.create(common)
        merged_datasets.append(OmegaConf.merge(base, dataset))

    return OmegaConf.merge(config, OmegaConf.create({"datasets": merged_datasets}))

def save_merged_config(config: DictConfig, output_file: Path) -> None:
    """
    Saves final configuration after all merges.
    
    Args:
        config: Configuration containing all datasets
        output_file: Path to output file
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        OmegaConf.save(config=config, f=output_file)
        logger.info(f"Saved merged config to: {output_file}")
    except Exception as e:
        logger.error(f"Failed to save merged config: {e}")


def run(config: DictConfig, phase: str = "full") -> None:
    """
    Args:
        config: merged config (default + overrides).
        phase: optional CLI override. "full" | "generation" | "judgement".
               If None, falls back to config experiment.phase.
    """
    logger.info("Starting benchmark run")
    logger.info(f"Selected phase: {phase}")
    
    # Merge common parameters into each dataset
    merged_config = merge_dataset_configs(config)
    
    datasets = merged_config.get("datasets", [])
    if not datasets:
        logger.error("No datasets specified in config")
        sys.exit(1)

    merged_config_file = resolve_path(
        config.get("merged_config_path", str(MERGED_CONFIG)), PROJECT_ROOT
    )
    save_merged_config(merged_config, merged_config_file)
    
    logger.info(f"Processing {len(datasets)} dataset(s)")

    # Final benchmark scores dictionary: arena_name -> summary
    benchmark_summaries = {}

    # Judging jobs collected per dataset; executed after the loop, grouped by
    # judge identity so datasets sharing one judge run through a single pool.
    judging_jobs: list[dict] = []

    # Iterate over each dataset
    for dataset_idx, dataset_conf in enumerate(datasets, 1):
        arena_name = dataset_conf.get("name")

        logger.info(f"\n{'='*60}")
        logger.info(f"Dataset {dataset_idx}/{len(datasets)}: {arena_name}")
        logger.info(f"{'='*60}")

        if phase in ("full", "generation"):
            logger.info("Phase 1: Generation")

            model_conf = dataset_conf.model
            baseline_conf = dataset_conf.baseline
            overwrite_gen = dataset_conf.experiment.get("overwrite_generations", False)

            questions_path = get_questions_path(arena_name)

            if not questions_path.exists():
                logger.error(f"Questions file not found: {questions_path}")
                logger.warning(f"Skipping dataset: {arena_name}")
                continue

            questions = load_jsonl(str(questions_path))
            logger.info(f"Questions: {questions_path} ({len(questions)} items)")

            _generate(model_conf,    arena_name, questions_path, overwrite=overwrite_gen)
            _generate(baseline_conf, arena_name, questions_path, overwrite=overwrite_gen)

            logger.info(f"Generation complete for '{arena_name}'.")

        if phase in ("full", "judgement"):
            logger.info("Phase 2: Judging")
            model_conf = dataset_conf.model
            baseline_conf = dataset_conf.baseline
            judge_conf = dataset_conf.judge

            questions_path = get_questions_path(arena_name)
            if not questions_path.exists():
                logger.error(f"Questions file not found: {questions_path}")
                logger.warning(f"Skipping dataset: {arena_name}")
                continue

            candidate_path = _find_gen_file(model_conf, arena_name)
            if candidate_path is None or not candidate_path.exists():
                logger.error(
                    f"Candidate generations not found for '{model_conf.name}' (dataset: {arena_name}). "
                    "Run with phase='generation' or phase='full' first."
                )
                sys.exit(1)

            baseline_path = _find_gen_file(baseline_conf, arena_name)
            if baseline_path is None or not baseline_path.exists():
                logger.error(
                    f"Baseline generations not found for '{baseline_conf.name}' (dataset: {arena_name}). "
                    "Run with phase='generation' or phase='full' first."
                )
                sys.exit(1)

            judge_obj = Judge(APIModel(judge_conf.model_name, judge_conf), judge_conf, project_root=DATA_ROOT)

            artifacts_dir = model_conf.get("artifacts_dir", DEFAULT_ARTIFACTS_DIR)
            judgment_file = resolve_judgment_path(model_conf.path, arena_name, baseline_conf.name, artifacts_dir=artifacts_dir)
            judgment_file.parent.mkdir(parents=True, exist_ok=True)

            judging_jobs.append({
                "idx": len(judging_jobs),
                "arena_name": arena_name,
                "judge_conf": judge_conf,
                "judge_obj": judge_obj,
                "model_conf": model_conf,
                "baseline_conf": baseline_conf,
                "dataset_conf": dataset_conf,
                "questions_path": questions_path,
                "candidate_path": candidate_path,
                "baseline_path": baseline_path,
                "judgment_file": judgment_file,
            })

    # Execute judging: group jobs by judge identity, one shared pool per group.
    if judging_jobs:
        groups: dict[tuple, list[dict]] = {}
        for job in judging_jobs:
            groups.setdefault(_judge_group_key(job["judge_conf"]), []).append(job)

        summaries_by_idx: dict[int, tuple[str, dict]] = {}
        for group_key, group_jobs in groups.items():
            runners = []
            for job in group_jobs:
                runners.append(JudgeRunner(
                    job["judge_obj"],
                    str(job["questions_path"]),
                    str(job["candidate_path"]),
                    str(job["baseline_path"]),
                    str(job["judgment_file"].parent),
                    candidate_model_name=job["model_conf"].name.replace("/", "_"),
                    baseline_model_name=job["baseline_conf"].name.replace("/", "_"),
                    overwrite=job["dataset_conf"].experiment.get("overwrite_judgments", False),
                    output_file=str(job["judgment_file"]),
                    parallel_api_calls=_resolve_parallel_api_calls(job["judge_conf"], default=1),
                    verbose_errors=job["judge_conf"].get("verbose_errors", False),
                    fail_on_error=job["judge_conf"].get("fail_on_error", False),
                    category_namespace=f"ds{job['idx']}",
                ))

            if len(group_jobs) == 1:
                stats_list = [runners[0].run()]
            else:
                output_paths = [str(job["judgment_file"]) for job in group_jobs]
                if len(set(output_paths)) != len(output_paths):
                    logger.warning(
                        "Duplicate judgment output paths within one judge group; "
                        "the same measurement is likely configured twice."
                    )
                pool_size = max(r.parallel_api_calls for r in runners)
                logger.info(
                    "Shared judge '%s @ %s': judging %s datasets through one pool (parallel_api_calls=%s): %s",
                    group_key[0], group_key[1], len(group_jobs), pool_size,
                    ", ".join(job["arena_name"] for job in group_jobs),
                )
                # Never raises: summaries below must be persisted for every
                # dataset of the group before fail_on_error is honored.
                stats_list = run_judge_runners(runners)

            for job, judge_stats in zip(group_jobs, stats_list):
                arena_name = job["arena_name"]
                judgment_file = job["judgment_file"]
                judge_conf = job["judge_conf"]

                failed_judgments = 0 if not judge_stats else int(judge_stats.get("failed", 0))
                if failed_judgments > 0:
                    logger.warning(f"'{arena_name}' judging finished with {failed_judgments} failures. File: {judgment_file}")
                else:
                    logger.info(f"'{arena_name}' judging complete. Saved to: {judgment_file}")

                regex_pattern = judge_conf.get("regex_pattern", r"\[\[([AB<>=]+)\]\]")
                summary = compute_judgment_summary(judgment_file, regex_pattern=regex_pattern)
                summary["failed_judgments"] = failed_judgments
                summaries_by_idx[job["idx"]] = (arena_name, summary)

                judgment_summary_path = judgment_file.parent / "judgment_summary.jsonl"
                save_judgment_summary_jsonl(
                    summary, judgment_summary_path, arena_name=arena_name, baseline_name=job["baseline_conf"].name,
                )
                logger.info(f"Judgment summary saved to: {judgment_summary_path}")

            # Honor fail_on_error only after every dataset of the group has its
            # judgments and summary persisted.
            if len(group_jobs) > 1:
                flagged_failures = sum(
                    int(stats["failed"])
                    for runner, stats in zip(runners, stats_list)
                    if runner.fail_on_error and int(stats["failed"]) > 0
                )
                if flagged_failures > 0:
                    raise RuntimeError(f"Judging completed with {flagged_failures} failed task(s)")

        # Preserve dataset order in the final table regardless of group order.
        for idx in sorted(summaries_by_idx):
            arena_name, summary = summaries_by_idx[idx]
            benchmark_summaries[arena_name] = summary

    logger.info(f"\n{'='*60}")
    logger.info(f"All datasets processed successfully!")
    logger.info(f"{'='*60}")

    # Output final benchmark scores
    if benchmark_summaries:
        logger.info(format_summary_table(benchmark_summaries))
        logger.info("")

def get_config_keys(cfg: object, prefix: str = "", max_len: int = 50) -> list[str]:
    keys = []
    if isinstance(cfg, (DictConfig, ListConfig)):
        data = OmegaConf.to_container(cfg, resolve=False)
    else:
        data = cfg

    if isinstance(data, dict):
        for k, v in data.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (dict, list)):
                keys.extend(get_config_keys(v, full_key))
            else:
                val_str = str(v).replace("\n", " ")
                if len(val_str) > max_len:
                    val_str = val_str[:max_len-3] + "..."
                keys.append(f"{full_key}={val_str}")
    elif isinstance(data, list):
        keys.append(f"{prefix} (List)")
    else:
        val_str = str(data).replace("\n", " ")
        if len(val_str) > max_len:
            val_str = val_str[:max_len-3] + "..."
        keys.append(f"{prefix}={val_str}")
    return keys

def validate_cli_overrides(cli_conf: DictConfig, base_conf: DictConfig) -> None:
    """
    Validates that all keys from CLI overrides exist in the base config.
    
    Args:
        cli_conf: Configuration from CLI overrides
        base_conf: Base config (after default + editable merge)
        
    Raises:
        ValueError: If keys are found that do not exist in the base config
    """
    if not cli_conf:
        return
    
    # --- TEMPORARY FIX: Handle deprecated experiment.max_questions (Remove in next version) ---
    if "experiment" in cli_conf and "max_questions" in cli_conf.experiment:
        logger.warning("DEPRECATED: 'experiment.max_questions' is no longer supported and will cause an error in future versions. Ignoring it for now.")
        del cli_conf.experiment.max_questions
    if "datasets" in cli_conf:
        for ds in cli_conf.datasets:
            if "experiment" in ds and "max_questions" in ds.experiment:
                logger.warning(f"DEPRECATED: 'datasets[].experiment.max_questions' for {ds.get('name', 'unknown')} is no longer supported. Ignoring it.")
                del ds.experiment.max_questions
    # ---------------------------------------------------------------------------------------

    def get_all_keys(cfg, prefix=""):
        """Recursively collects all keys from config."""
        keys = set()
        if isinstance(cfg, (DictConfig, dict)):
            for k, v in cfg.items():
                full_key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, (DictConfig, dict)):
                    keys.add(full_key)  # Add the key itself
                    keys.update(get_all_keys(v, full_key))
                elif isinstance(v, (ListConfig, list)):
                    keys.add(full_key)
                else:
                    keys.add(full_key)
        return keys
    
    cli_keys = get_all_keys(cli_conf)
    base_keys = get_all_keys(base_conf)
    
    invalid_keys = cli_keys - base_keys
    
    if invalid_keys:
        invalid_keys_sorted = sorted(invalid_keys)
        logger.error("Error: Attempt to set non-existent parameters via CLI:")
        for key in invalid_keys_sorted:
            logger.error(f"  - {key}")
        logger.error("\nAvailable parameters can be viewed via: ./dredd.py --help")
        raise ValueError(f"Invalid CLI parameters: {', '.join(invalid_keys_sorted)}")

def _require_config(path_str: str, base: Path, label: str = "Config") -> tuple[Path, DictConfig]:
    path = resolve_path(path_str, base)
    if not path.exists():
        logger.error(f"{label} file not found: {path}")
        sys.exit(1)
    return path, OmegaConf.load(path)


def main() -> None:
    # Pre-parse just --config so we can load it before building the full parser.
    # This lets us populate --help with the actual config keys.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    pre_args, _ = pre_parser.parse_known_args()

    conf_path, base_conf = _require_config(pre_args.config, CONFIGS_DIR)

    parser = argparse.ArgumentParser(
        description="Dredd: Side-by-Side LLM Evaluation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            f"Available config keys (from {conf_path.name}):\n  "
            + "\n  ".join(get_config_keys(base_conf))
        ),
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="Path to YAML config file")
    parser.add_argument("--editable", default=None,
                        help="Extra YAML config merged on top of --config")
    parser.add_argument("--overrides-json", default=None,
                        help='JSON dict of overrides, e.g. \'{"experiment":{"overwrite_generations":true}}\'')
    parser.add_argument("--phase", choices=["full", "generation", "judgement"], default="full",
                        help="full = generation + judgement (default); or run one phase only")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="Enable DEBUG logging (-v)")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        default=None, help="Explicit log level (overrides -v)")

    args, cli_overrides = parser.parse_known_args()
    configure_logging(args.log_level or ("DEBUG" if args.verbose else "INFO"))

    # Reload config only if the user passed a different --config than pre_args picked up
    if args.config != pre_args.config:
        conf_path, base_conf = _require_config(args.config, CONFIGS_DIR)
    logger.info(f"Base config: {conf_path}")

    # Merge editable config on top
    if args.editable:
        editable_path, editable_conf = _require_config(args.editable, CONFIGS_DIR, label="Editable config")
        logger.info(f"Editable config: {editable_path}")
        base_conf = OmegaConf.merge(base_conf, editable_conf)

    # Merge --overrides-json on top
    if args.overrides_json:
        try:
            overrides_dict = json.loads(args.overrides_json)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in --overrides-json: {e}")
            sys.exit(1)
        json_conf = OmegaConf.create(overrides_dict)
        validate_cli_overrides(json_conf, base_conf)
        base_conf = OmegaConf.merge(base_conf, json_conf)
        logger.info(f"JSON overrides applied: {list(overrides_dict.keys())}")

    # Merge positional key=value overrides (e.g. model.name=foo) on top
    cli_conf = OmegaConf.from_cli(cli_overrides)
    if cli_conf:
        validate_cli_overrides(cli_conf, base_conf)
        logger.info(f"CLI overrides applied: {list(cli_conf.keys())}")
    base_conf = OmegaConf.merge(base_conf, cli_conf)

    run(base_conf, phase=args.phase)

if __name__ == "__main__":
    main()
