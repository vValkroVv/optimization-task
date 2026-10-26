import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from job_submitter.core.slurm_rest import SlurmRESTError

from job_submitter.core.job_templates import slurm_client_lib as scl


@pytest.fixture
def slurm_submit_kwargs() -> dict[str, object]:
    return {
        "base_image": "img",
        "n_workers": 1,
        "script": "echo x",
        "job_desc": "desc",
        "region": "o1",
        "job_logdir": "/tmp/job",
        "slurm_logdir": "/tmp/slurm",
        "type": "binary",
    }


def call_submit(kwargs: dict[str, object], **overrides):
    return scl.submit_slurm_job(**(kwargs | overrides))


@pytest.mark.unit
class TestSubmitSbatch:
    def test_submit_sbatch_success(self, monkeypatch) -> None:
        captured = {}

        def _run(*args, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(stdout="Submitted batch job 123\n", stderr="")

        monkeypatch.setattr(scl.subprocess, "run", _run)

        assert scl._submit_sbatch("/tmp/script.sh", "/tmp/job", "/tmp/slurm") == "123"
        assert captured["env"]["SLURM_UMASK"] == "0002"

    def test_submit_sbatch_raises_on_failure(self, monkeypatch) -> None:
        def _raise(*args, **kwargs):
            raise subprocess.CalledProcessError(
                returncode=1, cmd="sbatch", stderr="boom"
            )

        monkeypatch.setattr(scl.subprocess, "run", _raise)
        with pytest.raises(subprocess.CalledProcessError):
            scl._submit_sbatch("/tmp/script.sh", "/tmp/job", "/tmp/slurm")

    def test_submit_sbatch_with_stderr_still_returns_job_id(self, monkeypatch) -> None:
        monkeypatch.setattr(
            scl.subprocess,
            "run",
            lambda *args, **kwargs: SimpleNamespace(
                stdout="Submitted batch job 123\n", stderr="warning"
            ),
        )
        assert scl._submit_sbatch("/tmp/script.sh", "/tmp/job", "/tmp/slurm") == "123"


@pytest.mark.unit
class TestSubmitSlurmRestdFlow:
    def test_submit_slurmrestd_retries_until_success(self, monkeypatch) -> None:
        calls = {"count": 0}

        class _Resp:
            def __init__(self, code, body=None):
                self.status_code = code
                self.text = "err"
                self._body = body or {}

            def json(self):
                return self._body

        def _post(url, headers, json, timeout):
            calls["count"] += 1
            if calls["count"] == 1:
                return _Resp(500)
            return _Resp(200, {"job_id": "42"})

        monkeypatch.setattr(requests, "post", _post)
        out = scl._submit_slurmrestd(
            slurmrestd_urls=["http://a:6820", "http://b:6820"],
            slurm_jwt_token="token",
            script="echo 1",
            slurm_logdir="/tmp/slurm",
            job_logdir="/tmp/job",
            job_desc="desc",
            n_workers=1,
            gpu_num=8,
            processes_per_worker=1,
        )
        assert out == "42"
        assert calls["count"] == 2

    def test_submit_slurmrestd_raises_when_response_has_no_job_id(
        self, monkeypatch
    ) -> None:
        class _Resp:
            status_code = 200
            text = "ok"

            @staticmethod
            def json():
                return {}

        monkeypatch.setattr(requests, "post", lambda *args, **kwargs: _Resp())
        with pytest.raises(SlurmRESTError):
            scl._submit_slurmrestd(
                slurmrestd_urls=["http://a:6820"],
                slurm_jwt_token="token",
                script="echo 1",
                slurm_logdir="/tmp/slurm",
                job_logdir="/tmp/job",
                job_desc="desc",
                n_workers=1,
                gpu_num=8,
                processes_per_worker=1,
                max_restarts=3,
                nodelist="node001",
            )

    def test_submit_slurmrestd_raises_when_all_urls_fail(self, monkeypatch) -> None:
        class _Resp:
            status_code = 500
            text = "err"

            @staticmethod
            def json():
                return {}

        monkeypatch.setattr(requests, "post", lambda *args, **kwargs: _Resp())
        with pytest.raises(SlurmRESTError):
            scl._submit_slurmrestd(
                slurmrestd_urls=["http://a:6820", "http://b:6820"],
                slurm_jwt_token="token",
                script="echo 1",
                slurm_logdir="/tmp/slurm",
                job_logdir="/tmp/job",
                job_desc="desc",
                n_workers=1,
                gpu_num=8,
                processes_per_worker=1,
            )

    def test_submit_slurmrestd_builds_expected_payload(self, monkeypatch) -> None:
        captured = {}

        class _Resp:
            status_code = 200
            text = "ok"

            @staticmethod
            def json():
                return {"job_id": "77"}

        def _post(url, headers, json, timeout):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            captured["timeout"] = timeout
            return _Resp()

        monkeypatch.setattr(requests, "post", _post)

        out = scl._submit_slurmrestd(
            slurmrestd_urls=["http://a:6820/"],
            slurm_jwt_token="token",
            script="echo train",
            slurm_logdir="/tmp/slurm",
            job_logdir="/tmp/job",
            job_desc="desc",
            n_workers=2,
            gpu_num=4,
            processes_per_worker=2,
            max_restarts=3,
            nodelist="node001",
            exclude_nodes="node[002-003]",
            partition="main",
            time_limit="12:00:00",
            requeue=True,
            comment="/data/tensorboard/run",
        )

        assert out == "77"
        assert captured["url"].endswith("/slurm/v0.0.40/job/submit")
        assert captured["headers"]["X-SLURM-USER-TOKEN"] == "token"
        assert captured["timeout"] == 30
        assert captured["json"]["script"] == "echo train"
        assert captured["json"]["job"]["name"] == "desc"
        assert captured["json"]["job"]["nodes"] == 2
        assert captured["json"]["job"]["tres_per_node"] == "gres/gpu:4"
        assert captured["json"]["job"]["tasks_per_node"] == 2
        assert captured["json"]["job"]["partition"] == "main"
        assert "qos" not in captured["json"]["job"]
        assert captured["json"]["job"]["max_restarts"] == 3
        assert captured["json"]["job"]["requeue"] is True
        assert captured["json"]["job"]["nodelist"] == "node001"
        assert captured["json"]["job"]["excluded_nodes"] == "node[002-003]"
        assert captured["json"]["job"]["time_limit"] == 720
        assert captured["json"]["job"]["comment"] == "/data/tensorboard/run"

    def test_submit_slurmrestd_includes_qos_when_provided(self, monkeypatch) -> None:
        captured = {}

        class _Resp:
            status_code = 200
            text = "ok"

            @staticmethod
            def json():
                return {"job_id": "99"}

        def _post(url, headers, json, timeout):
            captured["json"] = json
            return _Resp()

        monkeypatch.setattr(requests, "post", _post)

        out = scl._submit_slurmrestd(
            slurmrestd_urls=["http://a:6820/"],
            slurm_jwt_token="token",
            script="echo train",
            slurm_logdir="/tmp/slurm",
            job_logdir="/tmp/job",
            job_desc="desc",
            n_workers=2,
            gpu_num=4,
            processes_per_worker=2,
            partition="main",
            qos="alignment",
        )

        assert out == "99"
        assert captured["json"]["job"]["qos"] == "alignment"

    def test_submit_slurmrestd_with_gpu_type_keeps_nodes(self, monkeypatch) -> None:
        captured = {}

        class _Resp:
            status_code = 200
            text = "ok"

            @staticmethod
            def json():
                return {"job_id": "100"}

        def _post(url, headers, json, timeout):
            captured["json"] = json
            return _Resp()

        monkeypatch.setattr(requests, "post", _post)

        out = scl._submit_slurmrestd(
            slurmrestd_urls=["http://a:6820/"],
            slurm_jwt_token="token",
            script="echo train",
            slurm_logdir="/tmp/slurm",
            job_logdir="/tmp/job",
            job_desc="desc",
            n_workers=2,
            gpu_num=4,
            processes_per_worker=2,
            gpu_type="a100",
        )

        assert out == "100"
        assert captured["json"]["job"]["tres_per_node"] == "gres/gpu:a100:4"
        assert captured["json"]["job"]["environment"][0].startswith("USER=")
        assert captured["json"]["job"]["nodes"] == 2
        assert "excluded_nodes" not in captured["json"]["job"]

    def test_submit_slurmrestd_includes_explicit_account(self, monkeypatch) -> None:
        captured = {}

        class _Resp:
            status_code = 200
            text = "ok"

            @staticmethod
            def json():
                return {"job_id": "101"}

        def _post(url, headers, json, timeout):
            captured["json"] = json
            return _Resp()

        monkeypatch.setattr(requests, "post", _post)

        out = scl._submit_slurmrestd(
            slurmrestd_urls=["http://a:6820/"],
            slurm_jwt_token="token",
            script="echo train",
            slurm_logdir="/tmp/slurm",
            job_logdir="/tmp/job",
            job_desc="desc",
            n_workers=2,
            gpu_num=4,
            processes_per_worker=2,
            partition="leads",
            qos="leads",
            account="leads",
            nice=5,
        )

        assert out == "101"
        assert captured["json"]["job"]["partition"] == "leads"
        assert captured["json"]["job"]["qos"] == "leads"
        assert captured["json"]["job"]["account"] == "leads"
        assert captured["json"]["job"]["nice"] == 5

    def test_submit_slurmrestd_omits_account_when_not_given(self, monkeypatch) -> None:
        captured = {}

        class _Resp:
            status_code = 200
            text = "ok"

            @staticmethod
            def json():
                return {"job_id": "102"}

        def _post(url, headers, json, timeout):
            captured["json"] = json
            return _Resp()

        monkeypatch.setattr(requests, "post", _post)

        out = scl._submit_slurmrestd(
            slurmrestd_urls=["http://a:6820/"],
            slurm_jwt_token="token",
            script="echo train",
            slurm_logdir="/tmp/slurm",
            job_logdir="/tmp/job",
            job_desc="desc",
            n_workers=2,
            gpu_num=4,
            processes_per_worker=2,
            partition="",
        )

        assert out == "102"
        assert captured["json"]["job"]["partition"] == ""
        assert "account" not in captured["json"]["job"]
        assert "nice" not in captured["json"]["job"]

    def test_submit_slurmrestd_cpu_only_sets_host_resources(self, monkeypatch) -> None:
        captured = {}

        class _Resp:
            status_code = 200
            text = "ok"

            @staticmethod
            def json():
                return {"job_id": "cpu-1"}

        def _post(url, headers, json, timeout):
            captured["json"] = json
            return _Resp()

        monkeypatch.setattr(requests, "post", _post)

        out = scl._submit_slurmrestd(
            slurmrestd_urls=["http://a:6820/"],
            slurm_jwt_token="token",
            script="rclone copy /source remote:path",
            slurm_logdir="/tmp/slurm",
            job_logdir="/tmp/job",
            job_desc="xfer-request",
            n_workers=1,
            gpu_num=0,
            processes_per_worker=1,
            cpu_only=True,
            partition="transfer",
            cpus_per_task=16,
            memory="32G",
        )

        job = captured["json"]["job"]
        assert out == "cpu-1"
        assert job["standard_output"] == "/tmp/slurm/slurm-%j.out"
        assert job["standard_error"] == "/tmp/slurm/slurm-%j.err"
        assert job["cpus_per_task"] == 16
        assert job["memory_per_node"] == 32768
        assert job["partition"] == "transfer"
        assert "tres_per_node" not in job
        assert "exclusive" not in job
        assert job["environment"][0].startswith("USER=")

    def test_submit_slurmrestd_cpu_only_omits_optional_resources(
        self, monkeypatch
    ) -> None:
        captured = {}

        class _Resp:
            status_code = 200
            text = "ok"

            @staticmethod
            def json():
                return {"job_id": "cpu-2"}

        def _post(url, headers, json, timeout):
            captured["json"] = json
            return _Resp()

        monkeypatch.setattr(requests, "post", _post)

        out = scl._submit_slurmrestd(
            slurmrestd_urls=["http://a:6820/"],
            slurm_jwt_token="token",
            script="echo hi",
            slurm_logdir="/tmp/slurm",
            job_logdir="/tmp/job",
            job_desc="xfer-request",
            n_workers=1,
            gpu_num=0,
            processes_per_worker=1,
            cpu_only=True,
            partition="transfer",
        )

        job = captured["json"]["job"]
        assert out == "cpu-2"
        assert "cpus_per_task" not in job
        assert "memory_per_node" not in job


@pytest.mark.unit
class TestSubmitSlurmJobExtended:
    def test_partition_middle_with_too_many_workers_is_allowed(
        self, slurm_submit_kwargs
    ) -> None:
        out = call_submit(slurm_submit_kwargs, n_workers=33, dry_run=True)
        assert out is None

    def test_submit_slurm_job_api_path(self, monkeypatch, slurm_submit_kwargs) -> None:
        captured = {}

        def _fake_submit_slurmrestd(**kwargs):
            captured.update(kwargs)
            return "api-job"

        monkeypatch.setattr(scl, "_submit_slurmrestd", _fake_submit_slurmrestd)
        out = call_submit(
            slurm_submit_kwargs,
            type="pytorch2",
            slurm_submit_type="api",
            slurmrestd_urls=["http://a:6820"],
            slurm_jwt_token="token",
            exclude_nodes="node001",
            qos="data_high",
            account="data",
            nice=7,
        )
        assert out == "api-job"
        assert captured["slurmrestd_urls"] == ["http://a:6820"]
        assert captured["slurm_jwt_token"] == "token"
        assert captured["job_desc"] == "desc"
        assert captured["exclude_nodes"] == "node001"
        assert captured["qos"] == "data_high"
        assert captured["account"] == "data"
        assert captured["nice"] == 7

    def test_submit_slurm_job_dry_run_returns_none(self, slurm_submit_kwargs) -> None:
        out = call_submit(slurm_submit_kwargs, dry_run=True)
        assert out is None

    def test_submit_slurm_job_custom_region_requires_override(
        self, slurm_submit_kwargs
    ) -> None:
        with pytest.raises(AssertionError):
            call_submit(slurm_submit_kwargs, region="custom", dry_run=True)

    def test_submit_slurm_job_rejects_too_small_max_restarts(
        self, slurm_submit_kwargs
    ) -> None:
        with pytest.raises(AssertionError):
            call_submit(slurm_submit_kwargs, max_restarts=1, dry_run=True)

    def test_submit_slurm_job_accepts_valid_max_restarts(
        self, slurm_submit_kwargs
    ) -> None:
        out = call_submit(slurm_submit_kwargs, max_restarts=2, dry_run=True)
        assert out is None

    def test_submit_slurm_job_handles_empty_mount_entries(
        self, slurm_submit_kwargs
    ) -> None:
        out = call_submit(slurm_submit_kwargs, extra_mounts="a:b,   ,", dry_run=True)
        assert out is None

    @pytest.mark.parametrize(
        ("exclude_nodes", "expected_directive"),
        [
            (None, None),
            ("node[001-002]", "#SBATCH --exclude=node[001-002]"),
        ],
    )
    def test_submit_slurm_job_sbatch_path_renders_excluded_nodes(
        self,
        monkeypatch,
        slurm_submit_kwargs,
        exclude_nodes,
        expected_directive,
    ) -> None:
        captured = {}

        def _submit_sbatch(script_path, *args, **kwargs):
            captured["script"] = Path(script_path).read_text()
            return "sbatch-job"

        monkeypatch.setattr(scl, "_submit_sbatch", _submit_sbatch)
        out = call_submit(
            slurm_submit_kwargs,
            slurm_submit_type="sbatch",
            exclude_nodes=exclude_nodes,
        )

        assert out == "sbatch-job"
        if expected_directive is None:
            assert "#SBATCH --exclude=" not in captured["script"]
        else:
            assert expected_directive in captured["script"]

    @pytest.mark.parametrize(
        ("submit_kwargs", "expected_directives", "absent_directives"),
        [
            (
                {
                    "partition": "data",
                    "qos": "data_high",
                    "account": "data",
                    "nice": 20,
                },
                [
                    "#SBATCH --partition=data\n",
                    "#SBATCH --qos=data_high\n",
                    "#SBATCH --account=data\n",
                    "#SBATCH --nice=20\n",
                ],
                [],
            ),
            (
                {"partition": "middle"},
                ["#SBATCH --partition=middle\n"],
                ["#SBATCH --qos=", "#SBATCH --account=", "#SBATCH --nice="],
            ),
        ],
    )
    def test_submit_slurm_job_sbatch_path_renders_qos_and_account(
        self,
        monkeypatch,
        slurm_submit_kwargs,
        submit_kwargs,
        expected_directives,
        absent_directives,
    ) -> None:
        captured = {}

        def _submit_sbatch(script_path, *args, **kwargs):
            captured["script"] = Path(script_path).read_text()
            return "sbatch-job"

        monkeypatch.setattr(scl, "_submit_sbatch", _submit_sbatch)
        call_submit(slurm_submit_kwargs, slurm_submit_type="sbatch", **submit_kwargs)

        for directive in expected_directives:
            assert directive in captured["script"]
        for directive in absent_directives:
            assert directive not in captured["script"]
