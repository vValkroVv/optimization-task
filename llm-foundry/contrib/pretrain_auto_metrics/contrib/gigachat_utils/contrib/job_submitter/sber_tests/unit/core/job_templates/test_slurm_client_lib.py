from pathlib import Path

import pytest

from job_submitter.core.job_templates.slurm_client_lib import (
    _memory_to_mib,
    _submit_slurmrestd,
    _time_limit_to_minutes,
    submit_slurm_job,
)
from job_submitter.core.slurm_rest import SlurmRESTError


@pytest.mark.unit
class TestSubmitSlurmJob:
    def test_rejects_invalid_type(self) -> None:
        with pytest.raises(AssertionError):
            submit_slurm_job(
                base_image="img",
                n_workers=1,
                script="echo x",
                job_desc="desc",
                region="o1",
                job_logdir="/tmp/job",
                slurm_logdir="/tmp/slurm",
                type="bad-type",
            )

    def test_rejects_bad_extra_mounts(self) -> None:
        with pytest.raises(AssertionError):
            submit_slurm_job(
                base_image="img",
                n_workers=1,
                script="echo x",
                job_desc="desc",
                region="o1",
                job_logdir="/tmp/job",
                slurm_logdir="/tmp/slurm",
                type="binary",
                extra_mounts="invalid-mount",
            )

    def test_cpu_job_uses_host_template_without_gpu_or_container(
        self,
        monkeypatch,
    ) -> None:
        rendered = {}

        def submit(script_path, job_logdir, slurm_logdir):
            path = Path(script_path)
            rendered["script"] = path.read_text(encoding="utf-8")
            path.unlink()
            return "123"

        monkeypatch.setattr(
            "job_submitter.core.job_templates.slurm_client_lib._submit_sbatch",
            submit,
        )

        job_id = submit_slurm_job(
            base_image=None,
            n_workers=1,
            script="rclone copy /source remote:path",
            job_desc="xfer-request",
            region="o1",
            job_logdir="/tmp/job",
            slurm_logdir="/tmp/slurm",
            type="binary",
            gpu_num=0,
            cpu_only=True,
            partition="transfer",
            cpus_per_task=16,
            memory="32G",
        )

        assert job_id == "123"
        assert "#SBATCH --partition=transfer" in rendered["script"]
        assert "#SBATCH --cpus-per-task=16" in rendered["script"]
        assert "#SBATCH --mem=32G" in rendered["script"]
        assert "--gres" not in rendered["script"]
        assert "--container-image" not in rendered["script"]
        assert "rclone copy /source remote:path" in rendered["script"]

    def test_cpu_job_renders_long_running_options(self, monkeypatch) -> None:
        rendered = {}

        def submit(script_path, job_logdir, slurm_logdir):
            path = Path(script_path)
            rendered["script"] = path.read_text(encoding="utf-8")
            path.unlink()
            return "123"

        monkeypatch.setattr(
            "job_submitter.core.job_templates.slurm_client_lib._submit_sbatch",
            submit,
        )

        submit_slurm_job(
            base_image=None,
            n_workers=1,
            script="python -m auto_transfer.hot.runner",
            job_desc="tb-hot-o1",
            region="o1",
            job_logdir="/tmp/job",
            slurm_logdir="/tmp/slurm",
            type="binary",
            gpu_num=0,
            cpu_only=True,
            partition="transfer",
            time_limit="12:00:00",
            requeue=True,
            comment="/data/tensorboard/run",
        )

        assert "#SBATCH --time=12:00:00" in rendered["script"]
        assert "#SBATCH --requeue" in rendered["script"]
        assert "#SBATCH --comment=/data/tensorboard/run" in rendered["script"]

    def test_cpu_job_default_render_stays_byte_compatible(self, monkeypatch) -> None:
        rendered = {}

        def submit(script_path, job_logdir, slurm_logdir):
            path = Path(script_path)
            rendered["script"] = path.read_text(encoding="utf-8")
            path.unlink()
            return "123"

        monkeypatch.setattr(
            "job_submitter.core.job_templates.slurm_client_lib._submit_sbatch",
            submit,
        )

        submit_slurm_job(
            base_image=None,
            n_workers=1,
            script="rclone copy /source remote:path",
            job_desc="xfer-request",
            region="o1",
            job_logdir="/tmp/job",
            slurm_logdir="/tmp/slurm",
            type="binary",
            gpu_num=0,
            cpu_only=True,
            partition="transfer",
            cpus_per_task=16,
            memory="32G",
        )

        assert rendered["script"] == (
            '#!/usr/bin/env bash\n#SBATCH --job-name="xfer-request"\n'
            "#SBATCH --nodes=1\n#SBATCH --ntasks-per-node=1\n"
            "#SBATCH --partition=transfer\n"
            "#SBATCH --output=/tmp/slurm/slurm-%j.out\n"
            "#SBATCH --error=/tmp/slurm/slurm-%j.err\n\n"
            "#SBATCH --cpus-per-task=16\n\n\n#SBATCH --mem=32G\n\n\n\n\n"
            "set -euo pipefail\numask 0002\n\n\n"
            "rclone copy /source remote:path"
        )


@pytest.mark.unit
class TestMemoryToMib:
    @pytest.mark.parametrize(
        ("memory", "expected"),
        [
            (None, None),
            ("1024", 1024),
            ("512M", 512),
            ("32G", 32768),
            ("1T", 1048576),
            (" 16g ", 16384),
        ],
    )
    def test_converts_supported_values(self, memory, expected) -> None:
        assert _memory_to_mib(memory) == expected

    def test_rejects_unsupported_values(self) -> None:
        with pytest.raises(ValueError, match="Unsupported Slurm memory value"):
            _memory_to_mib("32Gi")


@pytest.mark.unit
class TestTimeLimitToMinutes:
    @pytest.mark.parametrize(
        ("time_limit", "expected"),
        [(None, None), ("00:01:00", 1), ("12:00:00", 720), ("1-00:00:01", 1441)],
    )
    def test_converts_supported_values(self, time_limit, expected) -> None:
        assert _time_limit_to_minutes(time_limit) == expected

    @pytest.mark.parametrize("time_limit", ["12h", "00:60:00", "00:00:60"])
    def test_rejects_unsupported_values(self, time_limit) -> None:
        with pytest.raises(ValueError, match="Unsupported Slurm time limit"):
            _time_limit_to_minutes(time_limit)


@pytest.mark.unit
class TestSubmitSlurmRestd:
    def test_raises_without_urls(self) -> None:
        with pytest.raises(SlurmRESTError):
            _submit_slurmrestd(
                slurmrestd_urls=[],
                slurm_jwt_token="token",
                script="echo x",
                slurm_logdir="/tmp/slurm",
                job_logdir="/tmp/job",
                job_desc="desc",
                n_workers=1,
                gpu_num=8,
                processes_per_worker=1,
            )


@pytest.mark.unit
def test_rest_environment_entries_include_home_and_path(monkeypatch) -> None:
    from job_submitter.core.job_templates import slurm_client_lib

    monkeypatch.setenv("HOME", "/home/tester")
    monkeypatch.setenv("PATH", "/usr/bin")
    entries = slurm_client_lib._rest_environment_entries()
    assert entries[0].startswith("USER=")
    assert "HOME=/home/tester" in entries
    assert "PATH=/usr/bin" in entries


@pytest.mark.unit
def test_rest_environment_entries_skip_missing_values(monkeypatch) -> None:
    from job_submitter.core.job_templates import slurm_client_lib

    monkeypatch.delenv("HOME", raising=False)
    monkeypatch.delenv("PATH", raising=False)
    entries = slurm_client_lib._rest_environment_entries()
    assert len(entries) == 1
    assert entries[0].startswith("USER=")
