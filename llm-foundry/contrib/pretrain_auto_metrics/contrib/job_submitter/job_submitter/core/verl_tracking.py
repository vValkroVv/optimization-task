from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_PROJECT_NAME = "verl"
_DEFAULT_EXPERIMENT_NAME = "experiment"
_TRACKING_KEYS = (
    "project_name",
    "exp_name",
    "CKPTS_DATA_HOME",
    "CKPTS_DIR",
    "TENSORBOARD_DIR",
    "TB_BASE_DIR",
)
_RUNTIME_ENV_KEYS = ("TENSORBOARD_DIR", "WANDB_DIR")
_SIMPLE_VAR_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
_LOGGER_RE = re.compile(r"trainer\.logger", re.IGNORECASE)


@dataclass(slots=True, frozen=True)
class ShellBracedExpression:
    expression: str
    end_index: int


@dataclass(slots=True, frozen=True)
class VerlScriptTracking:
    """Tracking-relevant facts parsed from a VERL train launch script."""

    variables: dict[str, str]
    uses_tensorboard: bool
    uses_wandb: bool


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    return value


def _normalize_logs_path(path: str) -> str:
    normalized = path.replace("\\", "/").strip()
    if len(normalized) > 2 and normalized[1] == ":":
        normalized = normalized[2:]
    return normalized.rstrip("/")


def _extract_assignment(text: str, name: str) -> Optional[str]:
    """Return the first simple ``KEY=...`` assignment from a shell script."""
    match = re.search(
        rf"^(?:export\s+)?{re.escape(name)}=(.+)$",
        text,
        re.MULTILINE,
    )
    if not match:
        return None
    value = match.group(1).strip()
    if value and value[0] not in "'\"":
        value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
    return value


def _extract_tracking_variables(text: str) -> dict[str, str]:
    raw = {
        key: assignment
        for key in _TRACKING_KEYS
        if (assignment := _extract_assignment(text, key)) is not None
    }
    return _resolve_tracking_variables(raw)


def _find_braced_expression(
    value: str, start_index: int
) -> ShellBracedExpression | None:
    end_index = -1
    depth = 0
    cursor = start_index + 2
    while cursor < len(value):
        if value[cursor : cursor + 2] == "${":
            depth += 1
            cursor += 2
            continue
        if value[cursor] == "}":
            if depth == 0:
                end_index = cursor
                break
            depth -= 1
        cursor += 1

    if end_index == -1:
        return None

    return ShellBracedExpression(
        expression=value[start_index + 2 : end_index],
        end_index=end_index,
    )


def _resolve_braced_expression(
    expression: str,
    variables: dict[str, str],
) -> str:
    if ":-" in expression:
        key, default = expression.split(":-", 1)
        key = key.strip()
        default = _strip_quotes(default.strip())
        current = variables.get(key)
        if current is None or "$" in current or current == key:
            return default
        return current

    key = expression.strip()
    current = variables.get(key)
    if current is not None and "$" not in current:
        return current
    return expression


def _resolve_simple_expression(match: re.Match[str], variables: dict[str, str]) -> str:
    key = match.group(1)
    current = variables.get(key)
    if current is not None and "$" not in current:
        return current
    return match.group(0)


def _expand_shell_var(value: str, variables: dict[str, str]) -> str:
    expanded = value
    parts: list[str] = []
    index = 0
    while index < len(expanded):
        if expanded[index : index + 2] != "${":
            next_dollar = expanded.find("${", index)
            if next_dollar == -1:
                parts.append(expanded[index:])
                break
            parts.append(expanded[index:next_dollar])
            index = next_dollar
            continue

        braced = _find_braced_expression(expanded, index)
        if braced is None:
            parts.append(expanded[index:])
            break

        parts.append(_resolve_braced_expression(braced.expression, variables))
        index = braced.end_index + 1

    expanded = _strip_quotes("".join(parts))
    return _SIMPLE_VAR_RE.sub(
        lambda match: _resolve_simple_expression(match, variables),
        expanded,
    )


def _resolve_tracking_variables(raw_variables: dict[str, str]) -> dict[str, str]:
    resolved = {key: _strip_quotes(value) for key, value in raw_variables.items()}
    for _ in range(len(raw_variables) + 1):
        changed = False
        for key, value in list(resolved.items()):
            new_value = _strip_quotes(_expand_shell_var(value, resolved))
            if new_value != value:
                resolved[key] = new_value
                changed = True
        if not changed:
            break
    return resolved


def _logger_enabled(text: str, backend: str) -> bool:
    for line in text.splitlines():
        if _LOGGER_RE.search(line) and backend in line.lower():
            return True
    return False


def parse_runtime_env(runtime_env_path: str | Path) -> dict[str, str]:
    """Read ``env_vars`` paths from a Ray ``runtime_env.yaml``."""
    path = Path(runtime_env_path)
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    env_vars = data.get("env_vars") or {}
    if not isinstance(env_vars, dict):
        return {}

    resolved: dict[str, str] = {}
    for key in _RUNTIME_ENV_KEYS:
        value = env_vars.get(key)
        if value is None:
            continue
        normalized = _strip_quotes(str(value).strip())
        if normalized:
            resolved[key] = normalized
    return resolved


def parse_verl_train_script(script_path: str | Path) -> VerlScriptTracking:
    """Extract tracking-relevant variables and logger flags from a train script."""
    text = Path(script_path).read_text(encoding="utf-8")
    return VerlScriptTracking(
        variables=_extract_tracking_variables(text),
        uses_tensorboard=_logger_enabled(text, "tensorboard"),
        uses_wandb=_logger_enabled(text, "wandb"),
    )


def _project_experiment_logs_path(
    base_dir: str,
    project_name: str,
    experiment_name: str,
) -> str:
    """Build ``{base_dir}/{project}/{experiment}`` as verl writes per-run logs.

    Both verl's ``_TensorboardAdapter`` (which appends ``/{project}/{experiment}``
    to ``TENSORBOARD_DIR`` before handing it to ``SummaryWriter``; see
    ``verl/utils/tracking.py``) and the wandb run layout place per-run logs under
    this sub-directory, so the daemon must watch the same suffixed path. The
    extra suffix is therefore intentional, not a duplicate.
    """
    return _normalize_logs_path(
        f"{base_dir.rstrip('/')}/{project_name}/{experiment_name}"
    )


def resolve_verl_logs_path(
    script_path: str | Path,
    *,
    runtime_env_path: Optional[str | Path] = None,
    exp_name_override: Optional[str] = None,
    job_logdir: Optional[str] = None,
) -> str:
    """Derive daemon monitoring path for a VERL job.

    Always returns a path. Preference order:

    1. ``CKPTS_DIR`` from the train shell script (typical wandb + checkpoints layout)
    2. TensorBoard path when ``trainer.logger`` includes ``tensorboard``
    3. ``CKPTS_DATA_HOME`` / ``project_name`` / ``exp_name`` from the script
    4. ``TENSORBOARD_DIR`` from ``runtime_env.yaml`` ``env_vars``
    5. ``WANDB_DIR`` from ``runtime_env.yaml`` ``env_vars``
    6. ``job_logdir`` from the submitted job template
    """
    script = parse_verl_train_script(script_path)
    script_vars = script.variables
    runtime_vars = parse_runtime_env(runtime_env_path) if runtime_env_path else {}

    project_name = script_vars.get("project_name", _DEFAULT_PROJECT_NAME)
    experiment_name = (
        exp_name_override or script_vars.get("exp_name") or _DEFAULT_EXPERIMENT_NAME
    )

    ckpts_dir = script_vars.get("CKPTS_DIR")
    if ckpts_dir and "$" not in ckpts_dir:
        return _normalize_logs_path(ckpts_dir)

    tensorboard_dir = script_vars.get("TENSORBOARD_DIR") or runtime_vars.get(
        "TENSORBOARD_DIR"
    )
    if script.uses_tensorboard and tensorboard_dir:
        return _project_experiment_logs_path(
            tensorboard_dir,
            project_name,
            experiment_name,
        )

    if tb_base_dir := script_vars.get("TB_BASE_DIR"):
        # Recipe scripts export TENSORBOARD_DIR="${TB_BASE_DIR}/${project}/${exp}"
        # (see verl_prod recipes), so reconstruct that value here. The extra
        # /project/exp that verl's adapter adds on top is applied by
        # _project_experiment_logs_path below.
        script_tensorboard_dir = (
            f"{tb_base_dir.rstrip('/')}/{project_name}/{experiment_name}"
        )
        return _project_experiment_logs_path(
            script_tensorboard_dir,
            project_name,
            experiment_name,
        )

    ckpts_data_home = script_vars.get("CKPTS_DATA_HOME")
    if ckpts_data_home and "$" not in ckpts_data_home:
        return _normalize_logs_path(
            f"{ckpts_data_home.rstrip('/')}/ckpts/{project_name}/{experiment_name}"
        )

    if tensorboard_dir:
        return _project_experiment_logs_path(
            tensorboard_dir,
            project_name,
            experiment_name,
        )

    wandb_dir = runtime_vars.get("WANDB_DIR")
    if wandb_dir:
        if script.uses_wandb or not script.uses_tensorboard:
            return _project_experiment_logs_path(
                wandb_dir, project_name, experiment_name
            )
        return _normalize_logs_path(wandb_dir)

    if job_logdir and str(job_logdir).strip():
        return _normalize_logs_path(str(job_logdir))

    raise ValueError(
        "No VERL logs path found. Set CKPTS_DIR/TENSORBOARD_DIR/TB_BASE_DIR "
        f"in script={script_path!s}, provide runtime_env_path={runtime_env_path!s}, "
        "or pass job_logdir."
    )


def build_verl_tracking_config(
    script_path: str | Path,
    pam_config_path: Optional[str] = None,
    *,
    runtime_env_path: Optional[str | Path] = None,
    exp_name_override: Optional[str] = None,
    job_logdir: Optional[str] = None,
) -> dict[str, Any]:
    """Build a minimal tracking config dict compatible with ``extract_logs_path``."""
    logs_path: str | None = None
    if pam_config_path:
        pam_config = _load_yaml_mapping(pam_config_path)
        if isinstance(pam_config, dict):
            logs_path = _extract_tracking_path_from_mapping(pam_config)

    if logs_path is None:
        logs_path = resolve_verl_logs_path(
            script_path,
            runtime_env_path=runtime_env_path,
            exp_name_override=exp_name_override,
            job_logdir=job_logdir,
        )

    return {"tensorboard_path": logs_path}


def _load_yaml_mapping(path: str | Path) -> dict[str, Any] | None:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data if isinstance(data, dict) else None


def _extract_tracking_path_from_mapping(config: dict[str, Any]) -> str | None:
    from omegaconf import OmegaConf
    from omegaconf.errors import InterpolationKeyError, InterpolationResolutionError

    try:
        resolved = OmegaConf.to_container(OmegaConf.create(config), resolve=True)
    except (InterpolationKeyError, InterpolationResolutionError) as exc:
        raise ValueError(f"Config interpolation error: {exc}") from exc
    if not isinstance(resolved, dict):
        return None

    tensorboard_path = _nested_get(resolved, "tensorboard_path")
    if tensorboard_path and str(tensorboard_path).strip():
        return _normalize_logs_path(str(tensorboard_path))

    log_dir = _nested_get(resolved, "loggers.tensorboard.log_dir")
    if log_dir and str(log_dir).strip():
        run_name = str(resolved.get("run_name") or _DEFAULT_EXPERIMENT_NAME)
        return _normalize_logs_path(f"{str(log_dir).rstrip('/')}/{run_name}")

    return None


def _nested_get(data: dict[str, Any], dotted_path: str) -> Any:
    value: Any = data
    for key in dotted_path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value
