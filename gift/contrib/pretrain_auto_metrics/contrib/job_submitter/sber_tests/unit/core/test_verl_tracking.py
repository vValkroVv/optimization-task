from pathlib import Path

import pytest

from job_submitter.core.tracking import extract_logs_path
from job_submitter.core.verl_tracking import (
    _expand_shell_var,
    _extract_assignment,
    _extract_tracking_path_from_mapping,
    _logger_enabled,
    _nested_get as _verl_nested_get,
    _normalize_logs_path,
    _resolve_tracking_variables,
    build_verl_tracking_config,
    parse_runtime_env,
    parse_verl_train_script,
    resolve_verl_logs_path,
)
from job_submitter.jobs.verl.train import RayTrainJobVERL


@pytest.mark.unit
class TestVerlTracking:
    def test_extract_ckpts_dir_for_wandb_script(self, tmp_path: Path) -> None:
        script = tmp_path / "train.sh"
        script.write_text(
            "\n".join(
                [
                    "project_name='verl_prod'",
                    "exp_name='debug_run'",
                    'CKPTS_DATA_HOME="/data/rl"',
                    'CKPTS_DIR="${CKPTS_DATA_HOME}/ckpts/${project_name}/${exp_name}"',
                    'trainer.logger=\'["console","wandb"]\' \\',
                ]
            ),
            encoding="utf-8",
        )

        assert resolve_verl_logs_path(script) == "/data/rl/ckpts/verl_prod/debug_run"
        assert extract_logs_path(build_verl_tracking_config(script)) == (
            "/data/rl/ckpts/verl_prod/debug_run"
        )

    def test_extract_tensorboard_path_with_double_suffix(self, tmp_path: Path) -> None:
        script = tmp_path / "train_tb.sh"
        script.write_text(
            "\n".join(
                [
                    "project_name='verl_prod'",
                    "exp_name='debug_run'",
                    'TB_BASE_DIR="/mnt/tb"',
                    'export TENSORBOARD_DIR="${TB_BASE_DIR}/${project_name}/${exp_name}"',
                    "trainer.logger=['console','tensorboard'] \\",
                ]
            ),
            encoding="utf-8",
        )

        assert (
            resolve_verl_logs_path(script)
            == "/mnt/tb/verl_prod/debug_run/verl_prod/debug_run"
        )

    def test_runtime_env_tensorboard_dir(self, tmp_path: Path) -> None:
        script = tmp_path / "train.sh"
        script.write_text(
            "\n".join(
                [
                    "project_name='verl_prod'",
                    "exp_name='debug_run'",
                    "trainer.logger=['console','tensorboard'] \\",
                ]
            ),
            encoding="utf-8",
        )
        runtime_env = tmp_path / "runtime_env.yaml"
        runtime_env.write_text(
            "\n".join(
                [
                    "working_dir: ./",
                    "env_vars:",
                    "  TENSORBOARD_DIR: /mnt/nfs/07-share/coconut/rl/tensorboard",
                ]
            ),
            encoding="utf-8",
        )

        assert (
            resolve_verl_logs_path(
                script,
                runtime_env_path=runtime_env,
            )
            == "/mnt/nfs/07-share/coconut/rl/tensorboard/verl_prod/debug_run"
        )

    def test_runtime_env_wandb_dir_when_script_has_no_ckpts(
        self, tmp_path: Path
    ) -> None:
        script = tmp_path / "train.sh"
        script.write_text(
            "\n".join(
                [
                    "project_name='verl_prod'",
                    "exp_name='debug_run'",
                    'trainer.logger=\'["console","wandb"]\' \\',
                ]
            ),
            encoding="utf-8",
        )
        runtime_env = tmp_path / "runtime_env.yaml"
        runtime_env.write_text(
            "\n".join(
                [
                    "working_dir: ./",
                    "env_vars:",
                    "  WANDB_DIR: /home/jovyan/shares/SR008.fs2/gigachat_checkpoints/rl/ckpts",
                ]
            ),
            encoding="utf-8",
        )

        assert resolve_verl_logs_path(
            script,
            runtime_env_path=runtime_env,
        ) == (
            "/home/jovyan/shares/SR008.fs2/gigachat_checkpoints/rl/ckpts/"
            "verl_prod/debug_run"
        )

    def test_runtime_env_tensorboard_without_logger_still_uses_runtime_dir(
        self, tmp_path: Path
    ) -> None:
        script = tmp_path / "train.sh"
        script.write_text(
            "\n".join(
                [
                    "project_name='verl_prod'",
                    "exp_name='debug_run'",
                    'trainer.logger=\'["console","wandb"]\' \\',
                ]
            ),
            encoding="utf-8",
        )
        runtime_env = tmp_path / "runtime_env.yaml"
        runtime_env.write_text(
            "\n".join(
                [
                    "working_dir: ./",
                    "env_vars:",
                    "  TENSORBOARD_DIR: /mnt/tb",
                ]
            ),
            encoding="utf-8",
        )

        assert (
            resolve_verl_logs_path(
                script,
                runtime_env_path=runtime_env,
                exp_name_override="submit_exp",
            )
            == "/mnt/tb/verl_prod/submit_exp"
        )

    def test_job_logdir_fallback(self, tmp_path: Path) -> None:
        script = tmp_path / "train.sh"
        script.write_text("project_name='p'\nexp_name='e'\n", encoding="utf-8")

        assert (
            resolve_verl_logs_path(
                script,
                job_logdir="/workspace/logs/debug_run",
            )
            == "/workspace/logs/debug_run"
        )

    def test_parse_runtime_env(self, tmp_path: Path) -> None:
        runtime_env = tmp_path / "runtime_env.yaml"
        runtime_env.write_text(
            "\n".join(
                [
                    "env_vars:",
                    "  WANDB_DIR: /wandb/base",
                    "  TENSORBOARD_DIR: /tb/base",
                ]
            ),
            encoding="utf-8",
        )

        assert parse_runtime_env(runtime_env) == {
            "WANDB_DIR": "/wandb/base",
            "TENSORBOARD_DIR": "/tb/base",
        }

    def test_ray_train_job_verl_merged_config(self, tmp_path: Path) -> None:
        script = tmp_path / "train.sh"
        script.write_text(
            "\n".join(
                [
                    "project_name='verl_prod'",
                    "exp_name='from_script'",
                    'CKPTS_DIR="/ckpts/run"',
                ]
            ),
            encoding="utf-8",
        )
        runtime_env = tmp_path / "runtime_env.yaml"
        runtime_env.write_text(
            "\n".join(
                [
                    "working_dir: ./",
                    "env_vars:",
                    "  WANDB_DIR: /ignored/wandb",
                ]
            ),
            encoding="utf-8",
        )

        job = RayTrainJobVERL(
            user_login="u",
            exp_name="debug_run",
            submission_type="cloud",
            region="SR008",
            script_path=str(script),
            runtime_env_config_path=str(runtime_env),
            job_logdir="/workspace/logs",
        )
        from types import SimpleNamespace

        merged = job._tracking_merged_config(
            SimpleNamespace(job_logdir="/workspace/logs")
        )
        assert merged == {"tensorboard_path": "/ckpts/run"}


@pytest.mark.unit
class TestVerlTrackingInternals:
    def test_normalize_logs_path_strips_windows_drive(self) -> None:
        assert _normalize_logs_path("C:\\logs\\run\\") == "/logs/run"

    def test_extract_assignment_with_comment_and_missing(self) -> None:
        text = "CKPTS_DIR=/data/ckpts # keep\n"
        assert _extract_assignment(text, "CKPTS_DIR") == "/data/ckpts"
        assert _extract_assignment(text, "MISSING") is None

    def test_expand_shell_var_branches(self) -> None:
        variables = {
            "project": "verl_prod",
            "exp": "debug",
            "self_ref": "${self_ref}",
        }
        assert _expand_shell_var("${project}/${exp}", variables) == "verl_prod/debug"
        assert _expand_shell_var("$project", variables) == "verl_prod"
        assert _expand_shell_var("${missing:-/fallback}", variables) == "/fallback"
        assert _expand_shell_var("${project:-/fallback}", variables) == "verl_prod"
        assert _expand_shell_var("${self_ref:-fallback}", variables) == "fallback"
        assert _expand_shell_var("${unclosed", variables) == "${unclosed"
        assert _expand_shell_var("$unresolved", variables) == "$unresolved"
        assert _expand_shell_var("$partial", {"partial": "$nested"}) == "$partial"
        assert _expand_shell_var("${missing}", variables) == "missing"
        assert _expand_shell_var("${${x:-a}}", {}) == "a}"

    def test_resolve_tracking_variables_stable_without_changes(self) -> None:
        resolved = _resolve_tracking_variables({"CKPTS_DIR": "/ckpts/run"})
        assert resolved == {"CKPTS_DIR": "/ckpts/run"}

    def test_resolve_tracking_variables_iterates_until_stable(self) -> None:
        resolved = _resolve_tracking_variables(
            {
                "project_name": "verl_prod",
                "exp_name": "debug",
                "CKPTS_DIR": "${CKPTS_DATA_HOME}/ckpts/${project_name}/${exp_name}",
                "CKPTS_DATA_HOME": "/data/rl",
            }
        )
        assert resolved["CKPTS_DIR"] == "/data/rl/ckpts/verl_prod/debug"

    def test_resolve_tracking_variables_runs_second_pass_for_key_order(
        self,
    ) -> None:
        resolved = _resolve_tracking_variables({"c": "${a}", "b": "/end", "a": "${b}"})
        assert resolved == {"c": "a", "b": "/end", "a": "/end"}

    def test_logger_enabled(self) -> None:
        assert _logger_enabled("trainer.logger=['tensorboard']", "tensorboard")
        assert not _logger_enabled("trainer.logger=['wandb']", "tensorboard")

    def test_parse_runtime_env_edge_cases(self, tmp_path: Path) -> None:
        assert parse_runtime_env(tmp_path / "missing.yaml") == {}

        invalid = tmp_path / "invalid.yaml"
        invalid.write_text("env_vars: not-a-dict\n", encoding="utf-8")
        assert parse_runtime_env(invalid) == {}

        empty_value = tmp_path / "empty.yaml"
        empty_value.write_text("env_vars:\n  WANDB_DIR: '   '\n", encoding="utf-8")
        assert parse_runtime_env(empty_value) == {}

    def test_tb_base_dir_branch(self, tmp_path: Path) -> None:
        script = tmp_path / "train.sh"
        script.write_text(
            "\n".join(
                [
                    "project_name='verl_prod'",
                    "exp_name='debug_run'",
                    'TB_BASE_DIR="/mnt/tb"',
                    "trainer.logger=['console','tensorboard'] \\",
                ]
            ),
            encoding="utf-8",
        )

        assert resolve_verl_logs_path(script) == (
            "/mnt/tb/verl_prod/debug_run/verl_prod/debug_run"
        )

    def test_ckpts_data_home_without_ckpts_dir(self, tmp_path: Path) -> None:
        script = tmp_path / "train.sh"
        script.write_text(
            "\n".join(
                [
                    "project_name='verl_prod'",
                    "exp_name='debug_run'",
                    'CKPTS_DATA_HOME="/data/rl"',
                ]
            ),
            encoding="utf-8",
        )

        assert resolve_verl_logs_path(script) == "/data/rl/ckpts/verl_prod/debug_run"

    def test_wandb_dir_with_tensorboard_logger_only(self, tmp_path: Path) -> None:
        script = tmp_path / "train.sh"
        script.write_text(
            "\n".join(
                [
                    "project_name='verl_prod'",
                    "exp_name='debug_run'",
                    "trainer.logger=['console','tensorboard'] \\",
                ]
            ),
            encoding="utf-8",
        )
        runtime_env = tmp_path / "runtime_env.yaml"
        runtime_env.write_text(
            "env_vars:\n  WANDB_DIR: /wandb/base\n",
            encoding="utf-8",
        )

        assert (
            resolve_verl_logs_path(script, runtime_env_path=runtime_env)
            == "/wandb/base"
        )

    def test_raises_when_nothing_configured(self, tmp_path: Path) -> None:
        script = tmp_path / "train.sh"
        script.write_text("echo hello\n", encoding="utf-8")

        with pytest.raises(ValueError, match="No VERL logs path found"):
            resolve_verl_logs_path(script)

    def test_build_verl_tracking_config_from_pam(self, tmp_path: Path) -> None:
        script = tmp_path / "train.sh"
        script.write_text("echo hello\n", encoding="utf-8")
        pam = tmp_path / "pam.yaml"
        pam.write_text("tensorboard_path: /pam/tb\n", encoding="utf-8")

        assert build_verl_tracking_config(script, pam_config_path=str(pam)) == {
            "tensorboard_path": "/pam/tb"
        }

    def test_build_verl_tracking_config_from_pam_log_dir(self, tmp_path: Path) -> None:
        script = tmp_path / "train.sh"
        script.write_text("echo hello\n", encoding="utf-8")
        pam = tmp_path / "pam.yaml"
        pam.write_text(
            "run_name: pam_run\nloggers:\n  tensorboard:\n    log_dir: /pam/tb\n",
            encoding="utf-8",
        )

        assert build_verl_tracking_config(script, pam_config_path=str(pam)) == {
            "tensorboard_path": "/pam/tb/pam_run"
        }

    def test_build_verl_tracking_config_pam_without_tracking_path_falls_back(
        self, tmp_path: Path
    ) -> None:
        script = tmp_path / "train.sh"
        script.write_text('CKPTS_DIR="/ckpts/run"\n', encoding="utf-8")
        pam = tmp_path / "pam.yaml"
        pam.write_text("unrelated: value\n", encoding="utf-8")

        assert build_verl_tracking_config(script, pam_config_path=str(pam)) == {
            "tensorboard_path": "/ckpts/run"
        }

    def test_build_verl_tracking_config_raises_on_pam_interpolation_error(
        self, tmp_path: Path
    ) -> None:
        script = tmp_path / "train.sh"
        script.write_text('CKPTS_DIR="/ckpts/run"\n', encoding="utf-8")
        pam = tmp_path / "pam.yaml"
        pam.write_text("tensorboard_path: ${missing}/tb\n", encoding="utf-8")

        with pytest.raises(ValueError, match="Config interpolation error"):
            build_verl_tracking_config(script, pam_config_path=str(pam))

    def test_build_verl_tracking_config_non_dict_pam_falls_back(
        self, tmp_path: Path
    ) -> None:
        script = tmp_path / "train.sh"
        script.write_text('CKPTS_DIR="/ckpts/run"\n', encoding="utf-8")
        pam = tmp_path / "pam.yaml"
        pam.write_text("null\n", encoding="utf-8")

        assert build_verl_tracking_config(script, pam_config_path=str(pam)) == {
            "tensorboard_path": "/ckpts/run"
        }

    def test_extract_tracking_path_returns_none_when_resolved_config_is_not_dict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "omegaconf.OmegaConf.to_container",
            lambda *_args, **_kwargs: [],
        )

        assert _extract_tracking_path_from_mapping({"tensorboard_path": "/tb"}) is None

    def test_nested_get_returns_none_when_chain_hits_non_dict(self) -> None:
        assert _verl_nested_get({"a": "not-a-dict"}, "a.b") is None

    def test_resolve_tracking_variables_multiple_passes(self) -> None:
        resolved = _resolve_tracking_variables(
            {
                "a": "$b",
                "b": "/final",
                "c": "${a}",
            }
        )
        assert resolved["c"] == "/final"

    def test_resolve_tracking_variables_exhausts_pass_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def always_grow(value: str, variables: dict[str, str]) -> str:
            return value + "!"

        monkeypatch.setattr(
            "job_submitter.core.verl_tracking._expand_shell_var",
            always_grow,
        )

        resolved = _resolve_tracking_variables({"x": "a"})

        assert resolved == {"x": "a!!"}

    def test_parse_verl_train_script_flags(self, tmp_path: Path) -> None:
        script = tmp_path / "train.sh"
        script.write_text(
            "trainer.logger=['console','wandb']\n",
            encoding="utf-8",
        )
        parsed = parse_verl_train_script(script)
        assert parsed.uses_wandb is True
        assert parsed.uses_tensorboard is False
