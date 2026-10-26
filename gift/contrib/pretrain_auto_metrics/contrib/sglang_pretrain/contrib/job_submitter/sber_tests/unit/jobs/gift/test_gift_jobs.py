import json
import shlex
from pathlib import Path

import pytest

from job_submitter.core.backends import CloudBackend
from job_submitter.core.job_templates.slurm_client_lib import _PYTORCH2_SCRIPT
from job_submitter.jobs.gift import train as gift_train_module
from job_submitter.jobs.gift.train import TrainJobGIFT
from job_submitter.jobs.image_registry import load_gift_base_images
from sber_tests.dry_run_helpers import assert_dry_run_submit_contract

TEST_BASE_IMAGES = {
    "o1": "/mnt/nfs/images/torchtitan_test.sqsh",
    "SR008": "cr.ai.cloud.ru/test/torchtitan:test",
}


def _write_images_registry(root: Path) -> None:
    registry = root / "gift_docker" / "images.json"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(json.dumps(TEST_BASE_IMAGES), encoding="utf-8")


# Stands in for the checkout's scripts/config_check.py: accepts an existing
# YAML with `--key value` overrides, rejects a missing file, a bare positional
# or any override naming `bad`, the way the real check rejects unknown keys.
_FAKE_CONFIG_CHECK = """\
import os, sys
path, args = sys.argv[1], sys.argv[2:]
if not os.path.isfile(path):
    print(f"config error: {path}: no such file", file=sys.stderr); sys.exit(1)
if any("bad" in a for a in args):
    print("error:   training.bad: unknown key", file=sys.stderr); sys.exit(1)
values_expected = False
for a in args:
    if a.startswith("--"):
        values_expected = "=" not in a
    elif values_expected:
        values_expected = False
    else:
        print(f"config error: unexpected positional argument {a!r}", file=sys.stderr)
        sys.exit(1)
print(f"{path}: OK")
"""

EXPERIMENT = "configs/experiments/gift_moe_debug.yaml"


def _gift_root(tmp_path: Path) -> Path:
    root = tmp_path / "torchtitan checkout"
    (root / "gift_scripts").mkdir(parents=True)
    (root / "gift_scripts" / "train.py").write_text("pass\n", encoding="utf-8")
    (root / "torchtitan").mkdir()
    (root / "torchtitan" / "train.py").write_text("pass\n", encoding="utf-8")
    (root / "scripts").mkdir()
    (root / "scripts" / "config_check.py").write_text(
        _FAKE_CONFIG_CHECK, encoding="utf-8"
    )
    experiment = root / EXPERIMENT
    experiment.parent.mkdir(parents=True)
    experiment.write_text("config_version: 1\n", encoding="utf-8")
    _write_images_registry(root)
    return root


def _yaml_job(root: Path, **overrides) -> TrainJobGIFT:
    params = dict(
        user_login="autotest",
        exp_name="e",
        submission_type="cloud",
        region="SR008",
        workspace_name="pretrain",
        config_file=EXPERIMENT,
        gift_root_path=root,
    )
    params.update(overrides)
    return TrainJobGIFT(**params)


@pytest.mark.unit
class TestTrainJobGIFT:
    def test_load_gift_base_images_returns_pinned_mapping(self, tmp_path: Path) -> None:
        assert load_gift_base_images(_gift_root(tmp_path)) == TEST_BASE_IMAGES

    @pytest.mark.parametrize(
        "content",
        [["not", "a", "mapping"], {"SR008": ""}, {"SR008": 42}],
    )
    def test_load_gift_base_images_rejects_malformed_registry(
        self, content, tmp_path: Path
    ) -> None:
        registry = tmp_path / "gift_docker" / "images.json"
        registry.parent.mkdir(parents=True)
        registry.write_text(json.dumps(content), encoding="utf-8")

        with pytest.raises(ValueError, match="Malformed"):
            load_gift_base_images(tmp_path)

    def test_resolve_base_image_requires_checkout_registry(
        self, tmp_path: Path
    ) -> None:
        job = TrainJobGIFT(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            config_file=EXPERIMENT,
            gift_root_path=tmp_path,
        )

        with pytest.raises(ValueError, match="images.json"):
            job._resolve_base_image()

    def test_resolve_base_image_rejects_unpinned_region(self, tmp_path: Path) -> None:
        job = TrainJobGIFT(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR004",
            config_file=EXPERIMENT,
            gift_root_path=_gift_root(tmp_path),
        )

        with pytest.raises(ValueError, match="SR004"):
            job._resolve_base_image()

    def test_explicit_base_image_skips_checkout_registry(self, tmp_path: Path) -> None:
        job = TrainJobGIFT(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            config_file=EXPERIMENT,
            base_image="cr.ai.cloud.ru/x/y:z",
            gift_root_path=tmp_path,
        )

        assert job._resolve_base_image() == "cr.ai.cloud.ru/x/y:z"

    def test_script_builds_config_file_command(self, tmp_path: Path) -> None:
        job = _yaml_job(_gift_root(tmp_path), script_args="--training.steps 2")

        script = job._script()

        assert "gift_scripts/train.py" in script.replace("\\", "/")
        assert f" {EXPERIMENT} " in script
        assert "--module" not in script and "--config" not in script
        assert script.endswith("--training.steps 2")

    @pytest.mark.parametrize("script_type", ["pytorch2", "binary"])
    def test_config_file_launch_passes_config_check(
        self, tmp_path: Path, script_type: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        root = _gift_root(tmp_path)
        (root / "scripts" / "config_check.py").write_text(
            _FAKE_CONFIG_CHECK + '\nprint("preflight warning", file=sys.stderr)\n',
            encoding="utf-8",
        )
        job = _yaml_job(root, script_args="--training.steps 2", script_type=script_type)

        job.assert_job(job, job.to_config())
        assert "preflight warning" in caplog.text

    def test_home_relative_config_uses_the_preflight_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _gift_root(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))
        job = _yaml_job(root, config_file=f"~/{root.name}/{EXPERIMENT}")

        assert shlex.split(job._script())[1] == str(job.config_file_path)
        job.assert_job(job, job.to_config())

    def test_config_file_may_be_absolute(self, tmp_path: Path) -> None:
        root = _gift_root(tmp_path)
        job = _yaml_job(root, config_file=root / EXPERIMENT)

        assert job.config_file_path == root / EXPERIMENT
        job.assert_job(job, job.to_config())

    def test_assert_job_rejects_missing_config_file(self, tmp_path: Path) -> None:
        job = _yaml_job(
            _gift_root(tmp_path), config_file="configs/experiments/nope.yaml"
        )

        with pytest.raises(AssertionError, match="config_file"):
            job.assert_job(job, job.to_config())

    def test_assert_job_rejects_config_check_failure(self, tmp_path: Path) -> None:
        job = _yaml_job(_gift_root(tmp_path), script_args="--training.bad 1")

        with pytest.raises(
            AssertionError, match=r"(?s)config-check rejected.*unknown key"
        ):
            job.assert_job(job, job.to_config())

    def test_assert_job_rejects_positional_override(self, tmp_path: Path) -> None:
        job = _yaml_job(
            _gift_root(tmp_path), script_args="other.yaml --training.steps 2"
        )

        with pytest.raises(AssertionError, match="unexpected positional"):
            job.assert_job(job, job.to_config())

    def test_assert_job_requires_torchtitan_train_entrypoint(
        self, tmp_path: Path
    ) -> None:
        root = _gift_root(tmp_path)
        (root / "torchtitan" / "train.py").unlink()
        job = _yaml_job(root)

        with pytest.raises(AssertionError, match="torchtitan/train.py"):
            job.assert_job(job, job.to_config())

    def test_assert_job_requires_config_file(self, tmp_path: Path) -> None:
        job = _yaml_job(_gift_root(tmp_path), config_file="")

        with pytest.raises(AssertionError, match="config_file"):
            job.assert_job(job, job.to_config())

    def test_gift_root_uses_environment_variable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        root = tmp_path / "gift from env"
        monkeypatch.setenv("GIFT_ROOT_PATH", str(root))

        job = TrainJobGIFT(
            user_login="u",
            exp_name="e",
            config_file=EXPERIMENT,
        )

        assert job.gift_root == root.resolve()

    def test_gift_root_defaults_to_vendored_checkout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GIFT_ROOT_PATH", raising=False)
        job = TrainJobGIFT(
            user_login="u",
            exp_name="e",
            config_file=EXPERIMENT,
        )

        assert job.gift_root == Path(gift_train_module.__file__).resolve().parents[5]

    def test_external_libraries_include_checkout_and_streaming(
        self, tmp_path: Path
    ) -> None:
        root = _gift_root(tmp_path)
        job = TrainJobGIFT(
            user_login="u",
            exp_name="e",
            config_file=EXPERIMENT,
            gift_root_path=root,
        )

        assert job._external_code_libraries() == [
            str(root.resolve()),
            str(root.resolve() / "contrib" / "streaming"),
        ]

    def test_cloud_config_uses_gift_image_and_pytorch_launcher(
        self, tmp_path: Path
    ) -> None:
        root = _gift_root(tmp_path)
        cfg = TrainJobGIFT(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            workspace_name="pretrain",
            config_file=EXPERIMENT,
            gift_root_path=root,
        ).to_config()

        assert cfg.base_image == TEST_BASE_IMAGES["SR008"]
        assert cfg.script_type == "pytorch2"
        assert cfg.processes_per_worker == 8
        assert cfg.env_variables["PYTHONPATH"] == ":".join(
            [
                str(root.resolve()),
                str(root.resolve() / "contrib" / "streaming"),
            ]
        )
        assert cfg.env_variables["LOG_RANK"] == "0"
        assert cfg.env_variables["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"

        job_params = object.__new__(CloudBackend)._build_job_params(
            cfg, cfg.env_variables
        )
        assert job_params["type"] == "pytorch2"
        assert job_params["processes_per_worker"] == 8
        assert job_params["script"] == cfg.script

    @pytest.mark.parametrize("n_gpus", [1, 2, 4, 8])
    def test_cloud_process_count_follows_gpu_count(
        self, n_gpus: int, tmp_path: Path
    ) -> None:
        cfg = TrainJobGIFT(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            workspace_name="pretrain",
            config_file=EXPERIMENT,
            gift_root_path=_gift_root(tmp_path),
            n_gpus=n_gpus,
        ).to_config()

        assert cfg.processes_per_worker == n_gpus

    def test_cloud_rejects_process_count_different_from_gpu_count(
        self, tmp_path: Path
    ) -> None:
        job = TrainJobGIFT(
            user_login="autotest",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            workspace_name="pretrain",
            config_file=EXPERIMENT,
            gift_root_path=_gift_root(tmp_path),
            n_gpus=4,
            processes_per_worker=8,
        )
        cfg = job.to_config()

        with pytest.raises(AssertionError, match="processes_per_worker"):
            job.assert_job(job, cfg)

    def test_slurm_config_uses_sqsh_and_one_launcher_task(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        root = _gift_root(tmp_path)
        job = TrainJobGIFT(
            user_login="u",
            exp_name="e",
            submission_type="slurm",
            region="o1",
            queue_name="pretrain",
            config_file=EXPERIMENT,
            gift_root_path=root,
        )
        monkeypatch.setattr(job, "_resolve_slurm_jwt_token", lambda: "token")
        monkeypatch.setattr(
            job, "_resolve_slurmrestd_urls", lambda: ["http://slurm:6820"]
        )

        cfg = job.to_config()

        assert cfg.base_image == TEST_BASE_IMAGES["o1"]
        assert cfg.base_image.endswith(".sqsh")
        assert cfg.script_type == "pytorch2"
        assert cfg.processes_per_worker == 1
        assert cfg.gpu_num == 8

        launcher = _PYTORCH2_SCRIPT.format(script=cfg.script)
        assert "--nproc-per-node=\\$JS_GPUS_PER_NODE" in launcher
        assert "--nnodes=\\$NNODES" in launcher
        assert "gift_scripts/train.py" in launcher.replace("\\", "/")
        job.assert_job(job, cfg)

    def test_extra_env_overrides_gift_defaults(self, tmp_path: Path) -> None:
        root = _gift_root(tmp_path)
        job = TrainJobGIFT(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            workspace_name="pretrain",
            config_file=EXPERIMENT,
            gift_root_path=root,
            extra_env={
                "LOG_RANK": "0,7",
                "PYTORCH_ALLOC_CONF": "expandable_segments:False",
            },
        )

        env = job._build_env()

        assert env["LOG_RANK"] == "0,7"
        assert env["PYTORCH_ALLOC_CONF"] == "expandable_segments:False"

    def test_assert_job_rejects_missing_entrypoint(self, tmp_path: Path) -> None:
        root = tmp_path / "torchtitan"
        root.mkdir()
        _write_images_registry(root)
        job = TrainJobGIFT(
            user_login="autotest",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            workspace_name="pretrain",
            config_file=EXPERIMENT,
            gift_root_path=root,
        )
        cfg = job.to_config()

        with pytest.raises(AssertionError, match="gift_scripts/train.py"):
            job.assert_job(job, cfg)

    def test_assert_job_rejects_malformed_script_args(self, tmp_path: Path) -> None:
        job = TrainJobGIFT(
            user_login="autotest",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            workspace_name="pretrain",
            config_file=EXPERIMENT,
            script_args="--training.steps '",
            gift_root_path=_gift_root(tmp_path),
        )
        cfg = job.to_config()

        with pytest.raises(AssertionError, match="Invalid script_args"):
            job.assert_job(job, cfg)

    def test_submit_dry_run_calls_before_submit_and_skips_backend(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        job = TrainJobGIFT(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            config_file=EXPERIMENT,
            gift_root_path=_gift_root(tmp_path),
            dry_run=True,
        )
        assert_dry_run_submit_contract(monkeypatch=monkeypatch, job=job)
