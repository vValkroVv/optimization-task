"""Schema and loader for the DeepGEMM warmup config YAML.

Shared by the GPU entrypoint (:mod:`deep_gemm_warmup`) and by the plan builder
(:mod:`torchtitan.ops.warmup.plan`), so it imports only stdlib and ``yaml``: a
submit host without CUDA, ``torch`` or ``deep_gemm`` must be able to load and
validate a config.

Everything the sweep reads unconditionally is validated here, so that a bad
config fails on the submit host instead of on a GPU that a job already waited
for.
"""

from __future__ import annotations

from pathlib import Path

import yaml

# The only section of the warmup YAML this module understands.
DEEP_GEMM_SECTION = "deep_gemm"

# DeepGEMM k_grouped wgrad: the num_stages template parameter changes at this
# threshold (SM90, FP8 1D1D block config 128x152x128, bf16 output).
# max_k <= 127 744 -> num_stages = 5; max_k > 127 744 -> num_stages = 4.
# Warmup must cross this boundary so both kernel variants are compiled.
SM90_STAGES_THRESHOLD = 128_000

# Every GEMM the sweep knows how to warm: the w1/w2 prefix picks the projection
# shape, the suffix picks the sweep (fwd, dgrad-transposed or wgrad).
GEMM_GROUPS = (
    "w1_m_grouped",
    "w1_m_grouped_nt_for_dgrad_transposed",
    "w1_k_grouped",
    "w2_m_grouped",
    "w2_m_grouped_nt_for_dgrad_transposed",
    "w2_k_grouped",
)
# Token buffers, the sweep step and the fp8 scaling blocks DeepGEMM quantizes
# over are all 128 tokens wide, so every per-group token cap must be a multiple.
TOKEN_BLOCK = 128


def load_warmup_config(config_path: str | Path) -> dict:
    """Load and validate the ``deep_gemm`` section of a warmup config YAML.

    Args:
        config_path: Path to the warmup config YAML.

    Returns:
        The validated ``deep_gemm`` section.

    Raises:
        ValueError: The file or the section is missing, or the section is
            incomplete: see :func:`_validate`.
    """
    config_path = Path(config_path)
    if not config_path.is_file():
        raise ValueError(f"Warmup config '{config_path}' does not exist")

    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f) or {}
    if DEEP_GEMM_SECTION not in full_config:
        raise ValueError(
            f"Warmup config '{config_path}' has no '{DEEP_GEMM_SECTION}' section"
        )

    warmup_config = full_config[DEEP_GEMM_SECTION]
    _validate(warmup_config, config_path)
    return warmup_config


def _config_error(config_path: Path, message: str) -> ValueError:
    return ValueError(f"'{DEEP_GEMM_SECTION}' section of '{config_path}': {message}")


def _validate(warmup_config: dict, config_path: Path) -> None:
    """Reject a config the sweep could not run to completion.

    Args:
        warmup_config: The ``deep_gemm`` section to check.
        config_path: Path the section came from, for error messages.

    Raises:
        ValueError: ``cache_dir`` is not an absolute path, ``gemm_groups`` is
            empty or names a group outside :data:`GEMM_GROUPS`, ``max_size`` is
            not a multiple of :data:`TOKEN_BLOCK`, or ``min_size`` exceeds the
            largest token total the sweep can reach.
    """
    cache_dir = warmup_config.get("cache_dir")
    if not isinstance(cache_dir, str) or not Path(cache_dir).is_absolute():
        raise _config_error(
            config_path,
            f"'cache_dir' must be an absolute path to the shared JIT cache, got "
            f"{cache_dir!r}",
        )

    gemm_groups = warmup_config.get("gemm_groups")
    if not isinstance(gemm_groups, list) or not gemm_groups:
        raise _config_error(
            config_path,
            f"'gemm_groups' must list at least one of {list(GEMM_GROUPS)}, got "
            f"{gemm_groups!r}",
        )
    unknown = [group for group in gemm_groups if group not in GEMM_GROUPS]
    if unknown:
        raise _config_error(
            config_path,
            f"'gemm_groups' names unknown GEMM group(s) {unknown}; a group the "
            f"sweep does not recognize would warm nothing. Expected any of "
            f"{list(GEMM_GROUPS)}",
        )

    # The sweeps assert this alignment on the GPU.  min_size is aligned by the
    # sweep itself, so only max_size has to be exact.
    max_size = warmup_config["max_size"]
    if max_size % TOKEN_BLOCK != 0:
        raise _config_error(
            config_path,
            f"'max_size' must be a multiple of {TOKEN_BLOCK}, got {max_size}",
        )
    # min_size is a total over all groups, max_size a per-group cap; past their
    # product the sweep has no token total left and completes with an empty cache.
    max_total = max_size * warmup_config["num_groups"]
    min_size = warmup_config["min_size"]
    if min_size > max_total:
        raise _config_error(
            config_path,
            f"'min_size' ({min_size}) exceeds max_size * num_groups ({max_total}), "
            f"the largest token total the sweep can reach, so nothing would be "
            f"warmed",
        )
