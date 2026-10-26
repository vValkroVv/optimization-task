import subprocess
from types import SimpleNamespace

import pytest

from job_submitter.core.backends import (
    CloudBackend,
    JobBackend,
    SlurmBackend,
    SlurmCpuSubmittedJob,
    SlurmSubmittedJob,
    submit_job_cfg,
)


@pytest.fixture
def fake_job_tracker():
    called = {}

    class FakeJob:
        job_name = None

        def __init__(self, **kwargs):
            called["init"] = kwargs

        def submit(self):
            called["submit"] = True
            self.job_name = "job-123"
            return f'Job "{self.job_name}" created.'

    return called, FakeJob


@pytest.mark.unit
class TestCloudBackend:
    def test_dispatcher_success_skips_client_lib(
        self, monkeypatch, cloud_template, fake_job_tracker
    ) -> None:
        called, FakeJob = fake_job_tracker

        monkeypatch.setattr(
            "job_submitter.core.backends.import_module",
            lambda _: SimpleNamespace(Job=FakeJob),
        )
        monkeypatch.setattr(
            "job_submitter.core.backends.requests.post",
            lambda *a, **kw: SimpleNamespace(
                status_code=200,
                text="ok",
                json=lambda: {"job_name": "dispatched-job-42"},
            ),
        )

        backend = CloudBackend()
        backend.submit(cloud_template)

        assert backend.last_submitted_job is not None
        assert backend.last_submitted_job.job_name == "dispatched-job-42"
        assert "submit" not in called
        assert called["init"]["job_desc"] == cloud_template.job_desc
        assert called["init"]["region"] == cloud_template.region

    def test_tts_workspace_skips_dispatcher(
        self, monkeypatch, cloud_template, fake_job_tracker
    ) -> None:
        called, FakeJob = fake_job_tracker
        monkeypatch.setattr(
            "job_submitter.core.backends.import_module",
            lambda _: SimpleNamespace(Job=FakeJob),
        )

        def _unexpected_dispatch(*_args, **_kwargs):
            raise AssertionError("TTS jobs must not use job-dispatcher")

        monkeypatch.setattr(
            "job_submitter.core.backends.requests.post",
            _unexpected_dispatch,
        )
        cloud_template.env_variables = {"MLS_NTB_WORKSPACE_NAME": "tts"}

        backend = CloudBackend()
        backend.submit(cloud_template)

        assert called["submit"] is True
        assert backend.last_submitted_job is not None
        assert backend.last_submitted_job.job_name == "job-123"

    def test_dispatcher_non_200_falls_back_to_client_lib(
        self, monkeypatch, cloud_template, fake_job_tracker, capsys
    ) -> None:
        called, FakeJob = fake_job_tracker

        monkeypatch.setattr(
            "job_submitter.core.backends.import_module",
            lambda _: SimpleNamespace(Job=FakeJob),
        )
        monkeypatch.setattr(
            "job_submitter.core.backends.requests.post",
            lambda *a, **kw: SimpleNamespace(status_code=502, text="bad gateway"),
        )

        backend = CloudBackend()
        backend.submit(cloud_template)

        assert backend.last_submitted_job is not None
        assert backend.last_submitted_job.job_name == "job-123"
        assert called["submit"] is True
        captured = capsys.readouterr()
        assert "Cloud job id: job-123" in captured.out
        assert called["init"]["job_desc"] == cloud_template.job_desc
        assert called["init"]["region"] == cloud_template.region

    def test_dispatcher_exception_falls_back_to_client_lib(
        self, monkeypatch, cloud_template, fake_job_tracker
    ) -> None:
        called, FakeJob = fake_job_tracker

        monkeypatch.setattr(
            "job_submitter.core.backends.import_module",
            lambda _: SimpleNamespace(Job=FakeJob),
        )

        def raise_connection_error(*a, **kw):
            raise ConnectionError("refused")

        monkeypatch.setattr(
            "job_submitter.core.backends.requests.post",
            raise_connection_error,
        )

        backend = CloudBackend()
        backend.submit(cloud_template)

        assert backend.last_submitted_job is not None
        assert backend.last_submitted_job.job_name == "job-123"
        assert called["submit"] is True
        assert called["init"]["job_desc"] == cloud_template.job_desc
        assert called["init"]["region"] == cloud_template.region

    def test_dispatcher_invalid_payload_falls_back_to_client_lib(
        self, monkeypatch, cloud_template, fake_job_tracker
    ) -> None:
        called, FakeJob = fake_job_tracker

        monkeypatch.setattr(
            "job_submitter.core.backends.import_module",
            lambda _: SimpleNamespace(Job=FakeJob),
        )
        monkeypatch.setattr(
            "job_submitter.core.backends.requests.post",
            lambda *a, **kw: SimpleNamespace(
                status_code=200,
                text="missing job_name",
                json=lambda: {},
            ),
        )

        backend = CloudBackend()
        backend.submit(cloud_template)

        assert backend.last_submitted_job is not None
        assert backend.last_submitted_job.job_name == "job-123"
        assert called["submit"] is True
        assert called["init"]["job_desc"] == cloud_template.job_desc
        assert called["init"]["region"] == cloud_template.region

    def test_dispatcher_url_and_timeout_from_job_env(
        self, monkeypatch, cloud_template, fake_job_tracker
    ) -> None:
        called, FakeJob = fake_job_tracker
        request_capture = {}

        monkeypatch.setattr(
            "job_submitter.core.backends.import_module",
            lambda _: SimpleNamespace(Job=FakeJob),
        )

        def _fake_post(url, *_args, **kwargs):
            request_capture["url"] = url
            request_capture["timeout"] = kwargs.get("timeout")
            return SimpleNamespace(status_code=502, text="bad gateway")

        monkeypatch.setattr(
            "job_submitter.core.backends.requests.post",
            _fake_post,
        )

        cloud_template.env_variables = {
            "JOB_DISPATCHER_BASE_URL": "http://dispatcher.custom:9000",
            "JOB_DISPATCHER_TIMEOUT_SECONDS": "7",
        }

        backend = CloudBackend()
        backend.submit(cloud_template)

        assert request_capture["url"] == "http://dispatcher.custom:9000/jobs/create-job"
        assert request_capture["timeout"] == (5, 7.0)
        assert called["submit"] is True


@pytest.mark.unit
class TestCloudBackendDispatcherPayload:
    def test_dispatcher_payload_preserves_required_null_fields(self) -> None:
        payload = CloudBackend._dispatcher_payload(
            {
                "region": "SR004",
                "allocation_name": None,
                "max_retry": None,
                "queue_name": "shared",
                "internet": True,
                "flags": {},
            }
        )

        assert payload == {
            "region": "SR004",
            "allocation_name": None,
            "max_retry": None,
            "queue_name": "shared",
            "internet": True,
            "flags": {},
        }

    def test_dispatcher_includes_null_allocation_name_for_sr004(
        self, monkeypatch, cloud_template, fake_job_tracker
    ) -> None:
        called, FakeJob = fake_job_tracker
        request_capture = {}

        monkeypatch.setattr(
            "job_submitter.core.backends.import_module",
            lambda _: SimpleNamespace(Job=FakeJob),
        )

        def _fake_post(url, *_args, **kwargs):
            request_capture["json"] = kwargs.get("json")
            return SimpleNamespace(
                status_code=200,
                text="ok",
                json=lambda: {"job_name": "dispatched-job-sr004"},
            )

        monkeypatch.setattr(
            "job_submitter.core.backends.requests.post",
            _fake_post,
        )

        cloud_template.region = "SR004"
        cloud_template.allocation_name = None

        backend = CloudBackend()
        backend.submit(cloud_template)

        assert request_capture["json"]["allocation_name"] is None
        assert request_capture["json"]["max_retry"] is None
        assert backend.last_submitted_job is not None
        assert backend.last_submitted_job.job_name == "dispatched-job-sr004"
        assert "submit" not in called


@pytest.mark.unit
class TestSlurmSubmittedJob:
    def test_job_name_and_logs(self) -> None:
        job = SlurmSubmittedJob(job_id="123")
        assert job.job_name == "123"
        assert job.status() == "Unknown"
        assert job.logs(tail=10, verbose=False) == ""


@pytest.mark.unit
class TestSlurmCpuSubmittedJob:
    def test_status_prefers_active_queue_state(self, monkeypatch) -> None:
        monkeypatch.setattr(
            SlurmCpuSubmittedJob,
            "_run",
            staticmethod(
                lambda command: SimpleNamespace(
                    returncode=0,
                    stdout="RUNNING\n",
                    stderr="",
                )
            ),
        )
        assert SlurmCpuSubmittedJob("123").status() == "RUNNING"

    def test_status_falls_back_to_accounting(self, monkeypatch) -> None:
        results = iter(
            [
                SimpleNamespace(returncode=0, stdout="", stderr=""),
                SimpleNamespace(
                    returncode=0,
                    stdout="header\n123|COMPLETED|0:0\n",
                    stderr="",
                ),
            ]
        )
        monkeypatch.setattr(
            SlurmCpuSubmittedJob,
            "_run",
            staticmethod(lambda command: next(results)),
        )
        assert SlurmCpuSubmittedJob("123").status() == "COMPLETED"

    def test_status_rejects_invalid_job_id(self) -> None:
        with pytest.raises(ValueError, match="Invalid Slurm job ID"):
            SlurmCpuSubmittedJob("unsafe").status()

    def test_status_reports_accounting_failure(self, monkeypatch) -> None:
        results = iter(
            [
                SimpleNamespace(returncode=1, stdout="", stderr=""),
                SimpleNamespace(returncode=1, stdout="", stderr="accounting offline"),
            ]
        )
        monkeypatch.setattr(
            SlurmCpuSubmittedJob,
            "_run",
            staticmethod(lambda command: next(results)),
        )
        with pytest.raises(RuntimeError, match="accounting offline"):
            SlurmCpuSubmittedJob("123").status()

    def test_find_active_job_matches_exact_name(self, monkeypatch) -> None:
        monkeypatch.setattr(
            SlurmCpuSubmittedJob,
            "_run",
            staticmethod(
                lambda command: SimpleNamespace(
                    returncode=0,
                    stdout="bad\n123|other\n456|xfer-job\n",
                    stderr="",
                )
            ),
        )
        assert SlurmCpuSubmittedJob.find_active_job("xfer-job") == "456"

    def test_find_active_job_reports_queue_failure(self, monkeypatch) -> None:
        monkeypatch.setattr(
            SlurmCpuSubmittedJob,
            "_run",
            staticmethod(
                lambda command: SimpleNamespace(returncode=1, stdout="", stderr="")
            ),
        )
        with pytest.raises(RuntimeError, match="active Slurm jobs"):
            SlurmCpuSubmittedJob.find_active_job("xfer-job")

    def test_find_active_job_returns_none_without_exact_match(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            SlurmCpuSubmittedJob,
            "_run",
            staticmethod(
                lambda command: SimpleNamespace(
                    returncode=0,
                    stdout="bad\n123|other\n",
                    stderr="",
                )
            ),
        )
        assert SlurmCpuSubmittedJob.find_active_job("xfer-job") is None

    def test_find_job_returns_active_job(self, monkeypatch) -> None:
        monkeypatch.setattr(
            SlurmCpuSubmittedJob,
            "find_active_job",
            classmethod(lambda cls, job_name: "123"),
        )
        assert SlurmCpuSubmittedJob.find_job("xfer-job") == "123"

    def test_find_job_returns_latest_matching_history_row(self, monkeypatch) -> None:
        monkeypatch.setattr(
            SlurmCpuSubmittedJob,
            "find_active_job",
            classmethod(lambda cls, job_name: None),
        )
        monkeypatch.setattr(
            SlurmCpuSubmittedJob,
            "_run",
            staticmethod(
                lambda command: SimpleNamespace(
                    returncode=0,
                    stdout="bad\n123|other|\n456|xfer-job|\n789|xfer-job|\n",
                    stderr="",
                )
            ),
        )
        assert SlurmCpuSubmittedJob.find_job("xfer-job") == "789"

    def test_find_job_returns_none_without_history(self, monkeypatch) -> None:
        monkeypatch.setattr(
            SlurmCpuSubmittedJob,
            "find_active_job",
            classmethod(lambda cls, job_name: None),
        )
        monkeypatch.setattr(
            SlurmCpuSubmittedJob,
            "_run",
            staticmethod(
                lambda command: SimpleNamespace(returncode=0, stdout="", stderr="")
            ),
        )
        assert SlurmCpuSubmittedJob.find_job("xfer-job") is None

    def test_find_job_reports_accounting_failure(self, monkeypatch) -> None:
        monkeypatch.setattr(
            SlurmCpuSubmittedJob,
            "find_active_job",
            classmethod(lambda cls, job_name: None),
        )
        monkeypatch.setattr(
            SlurmCpuSubmittedJob,
            "_run",
            staticmethod(
                lambda command: SimpleNamespace(
                    returncode=1,
                    stdout="",
                    stderr="accounting offline",
                )
            ),
        )
        with pytest.raises(RuntimeError, match="job history: accounting offline"):
            SlurmCpuSubmittedJob.find_job("xfer-job")

    @pytest.mark.parametrize(
        ("accounting_output", "expected_state", "expected_exit_code"),
        [
            ("123|COMPLETED|1:0\n", "FAILED", "1:0"),
            ("123|||\n", "UNKNOWN", None),
            ("header\n", "UNKNOWN", None),
        ],
    )
    def test_status_interprets_terminal_accounting_rows(
        self,
        monkeypatch,
        accounting_output,
        expected_state,
        expected_exit_code,
    ) -> None:
        results = iter(
            [
                SimpleNamespace(returncode=0, stdout="", stderr=""),
                SimpleNamespace(returncode=0, stdout=accounting_output, stderr=""),
            ]
        )
        monkeypatch.setattr(
            SlurmCpuSubmittedJob,
            "_run",
            staticmethod(lambda command: next(results)),
        )

        status = SlurmCpuSubmittedJob("123").status_details()

        assert status.state == expected_state
        assert status.exit_code == expected_exit_code

    def test_run_invokes_slurm_command_with_timeout(self, monkeypatch) -> None:
        completed = SimpleNamespace(returncode=0, stdout="ok", stderr="")
        invocation = {}

        def run(*args, **kwargs):
            invocation["args"] = args
            invocation["kwargs"] = kwargs
            return completed

        monkeypatch.setattr("job_submitter.core.backends.subprocess.run", run)
        assert SlurmCpuSubmittedJob._run(["squeue"]) is completed
        assert invocation == {
            "args": (["squeue"],),
            "kwargs": {
                "text": True,
                "capture_output": True,
                "timeout": 30.0,
                "check": False,
            },
        }

    @pytest.mark.parametrize(
        "error",
        [
            FileNotFoundError("missing"),
            subprocess.TimeoutExpired("squeue", timeout=30.0),
        ],
    )
    def test_run_wraps_command_errors(self, monkeypatch, error) -> None:
        monkeypatch.setattr(
            "job_submitter.core.backends.subprocess.run",
            lambda *a, **kw: (_ for _ in ()).throw(error),
        )
        with pytest.raises(RuntimeError, match="Cannot run squeue"):
            SlurmCpuSubmittedJob._run(["squeue"])

    @pytest.mark.parametrize(
        ("job", "expected_state", "expected_exit_code"),
        [
            (
                {
                    "job_state": ["COMPLETED"],
                    "exit_code": {"status": 1, "signal": 0},
                },
                "FAILED",
                "1:0",
            ),
            (
                {
                    "state": {"state": "running"},
                    "exit_code": 0,
                },
                "RUNNING",
                "0:0",
            ),
            (
                {
                    "state": {"name": "pending"},
                    "exit_code": {"return_code": 0},
                },
                "PENDING",
                "0:0",
            ),
            ({"job_state": "", "exit_code": ""}, "UNKNOWN", None),
            (
                {"job_state": "FAILED", "exit_code": {"signal": 9}},
                "FAILED",
                None,
            ),
            ({"job_state": "FAILED", "exit_code": "2:0"}, "FAILED", "2:0"),
            (None, "UNKNOWN", None),
        ],
    )
    def test_rest_status_normalizes_job_details(
        self,
        monkeypatch,
        job,
        expected_state,
        expected_exit_code,
    ) -> None:
        class FakeRestClient:
            def __init__(self, **kwargs):
                assert kwargs == {"urls": [], "jwt_token": None}

            def get_job(self, job_id):
                assert job_id == "123"
                return job

        monkeypatch.setattr(
            "job_submitter.core.backends.SlurmRestClient",
            FakeRestClient,
        )

        status = SlurmCpuSubmittedJob(
            "123",
            slurm_submit_type="api",
        ).status_details()

        assert status.state == expected_state
        assert status.exit_code == expected_exit_code

    def test_rest_find_methods_pass_explicit_configuration(self, monkeypatch) -> None:
        calls = []

        class FakeRestClient:
            def __init__(self, **kwargs):
                calls.append(("init", kwargs))

            def find_job(self, job_name):
                calls.append(("find", job_name))
                return "456"

        monkeypatch.setattr(
            "job_submitter.core.backends.SlurmRestClient",
            FakeRestClient,
        )
        kwargs = {
            "slurm_submit_type": "api",
            "slurm_jwt_token": "token",
            "slurmrestd_urls": ["http://slurm:6820"],
        }

        assert SlurmCpuSubmittedJob.find_active_job("xfer", **kwargs) == "456"
        assert SlurmCpuSubmittedJob.find_job("xfer", **kwargs) == "456"
        assert calls == [
            (
                "init",
                {
                    "urls": ["http://slurm:6820"],
                    "jwt_token": "token",
                },
            ),
            ("find", "xfer"),
            (
                "init",
                {
                    "urls": ["http://slurm:6820"],
                    "jwt_token": "token",
                },
            ),
            ("find", "xfer"),
        ]

    def test_rest_cancel_validates_mode_and_job_id(self, monkeypatch) -> None:
        cancelled = []

        class FakeRestClient:
            def __init__(self, **kwargs):
                assert kwargs == {"urls": [], "jwt_token": None}

            def cancel(self, job_id):
                cancelled.append(job_id)

        monkeypatch.setattr(
            "job_submitter.core.backends.SlurmRestClient",
            FakeRestClient,
        )

        SlurmCpuSubmittedJob("123", slurm_submit_type="api").cancel()

        assert cancelled == ["123"]
        with pytest.raises(RuntimeError, match="requires slurm_submit_type"):
            SlurmCpuSubmittedJob("123").cancel()
        with pytest.raises(ValueError, match="Invalid Slurm job ID"):
            SlurmCpuSubmittedJob("unsafe", slurm_submit_type="api").cancel()


@pytest.mark.unit
class TestCloudBackendTimeout:
    def test_invalid_timeout_falls_back_to_default(self) -> None:
        assert (
            CloudBackend._resolve_dispatcher_timeout_seconds(
                {"JOB_DISPATCHER_TIMEOUT_SECONDS": "bad"}
            )
            == CloudBackend.DEFAULT_JOB_DISPATCHER_TIMEOUT_SECONDS
        )
        assert (
            CloudBackend._resolve_dispatcher_timeout_seconds(
                {"JOB_DISPATCHER_TIMEOUT_SECONDS": "-1"}
            )
            == CloudBackend.DEFAULT_JOB_DISPATCHER_TIMEOUT_SECONDS
        )

    def test_fallback_preserves_job_name_set_by_client_lib(
        self, monkeypatch, cloud_template
    ) -> None:
        class FakeJob:
            job_name = None

            def __init__(self, **_kwargs):
                return None

            def submit(self):
                self.job_name = "lm-mpi-job-456"
                return f'Job "{self.job_name}" created.'

        monkeypatch.setattr(
            "job_submitter.core.backends.import_module",
            lambda _: SimpleNamespace(Job=FakeJob),
        )
        monkeypatch.setattr(
            "job_submitter.core.backends.requests.post",
            lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("down")),
        )

        backend = CloudBackend()
        backend.submit(cloud_template)

        assert backend.last_submitted_job is not None
        assert backend.last_submitted_job.job_name == "lm-mpi-job-456"
        assert "created." not in backend.last_submitted_job.job_name

    def test_fallback_does_not_store_handle_when_submit_fails(
        self, monkeypatch, cloud_template
    ) -> None:
        class FakeJob:
            job_name = None

            def __init__(self, **_kwargs):
                return None

            def submit(self):
                return "Error 400: bad request"

        monkeypatch.setattr(
            "job_submitter.core.backends.import_module",
            lambda _: SimpleNamespace(Job=FakeJob),
        )
        monkeypatch.setattr(
            "job_submitter.core.backends.requests.post",
            lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("down")),
        )

        backend = CloudBackend()
        backend.submit(cloud_template)

        assert backend.last_submitted_job is None


@pytest.mark.unit
class TestSlurmBackend:
    def test_calls_submit_slurm_job(self, monkeypatch, slurm_template) -> None:
        called = {}
        slurm_template.exclude_nodes = "node[001-002]"
        slurm_template.qos = "pretrain"
        slurm_template.account = "pretrain"
        slurm_template.nice = 10

        def fake_submit_slurm_job(**kwargs):
            called.update(kwargs)
            return "slurm-123"

        monkeypatch.setattr(
            "job_submitter.core.backends.submit_slurm_job", fake_submit_slurm_job
        )
        backend = SlurmBackend()
        out = backend.submit(slurm_template)

        assert out is None
        assert backend.last_submitted_job is not None
        assert type(backend.last_submitted_job) is SlurmSubmittedJob
        assert backend.last_submitted_job.job_id == "slurm-123"
        assert called["region"] == "o1"
        assert called["partition"] == "middle"
        assert called["script"] == "echo hello"
        assert called["exclude_nodes"] == "node[001-002]"
        assert called["qos"] == "pretrain"
        assert called["account"] == "pretrain"
        assert called["nice"] == 10

    def test_cpu_submit_returns_pollable_job(self, monkeypatch, slurm_template) -> None:
        slurm_template.cpu_only = True
        monkeypatch.setattr(
            "job_submitter.core.backends.submit_slurm_job",
            lambda **kwargs: "123",
        )

        backend = SlurmBackend()
        backend.submit(slurm_template)

        assert isinstance(backend.last_submitted_job, SlurmCpuSubmittedJob)

    def test_submit_without_job_id_clears_last_submitted_job(
        self, monkeypatch, slurm_template
    ) -> None:
        monkeypatch.setattr(
            "job_submitter.core.backends.submit_slurm_job",
            lambda **kwargs: None,
        )

        backend = SlurmBackend()
        backend.submit(slurm_template)

        assert backend.last_submitted_job is None


@pytest.mark.unit
class TestSubmitJobCfg:
    def test_selects_backend_by_submission_type(
        self, monkeypatch, cloud_template, slurm_template
    ) -> None:
        called = {"cloud": 0, "slurm": 0}
        monkeypatch.setattr(
            CloudBackend,
            "submit",
            lambda self, cfg: called.__setitem__("cloud", called["cloud"] + 1),
        )
        monkeypatch.setattr(
            SlurmBackend,
            "submit",
            lambda self, cfg: called.__setitem__("slurm", called["slurm"] + 1),
        )

        assert submit_job_cfg(cloud_template) is None
        assert submit_job_cfg(slurm_template) is None
        assert called == {"cloud": 1, "slurm": 1}

    def test_rejects_unknown_submission_type(self, cloud_template) -> None:
        cloud_template.submission_type = "clodu"

        with pytest.raises(ValueError, match="Unknown submission_type"):
            submit_job_cfg(cloud_template)


@pytest.mark.unit
class TestJobBackendBase:
    def test_cannot_instantiate_abstract_backend(self) -> None:
        with pytest.raises(TypeError):
            JobBackend()

    def test_default_submit_body_returns_none_when_called_via_super(self) -> None:
        class _BackendWithSuperSubmit(JobBackend):
            def submit(self, cfg):
                return super().submit(cfg)

        backend = _BackendWithSuperSubmit()
        assert backend.submit(None) is None


@pytest.mark.unit
class TestNormalizeRestExitCode:
    def _norm(self, value):
        return SlurmCpuSubmittedJob._normalize_rest_exit_code(value)

    def test_none_and_empty_return_none(self) -> None:
        assert self._norm(None) is None
        assert self._norm("") is None

    def test_plain_int_and_string_passthrough(self) -> None:
        assert self._norm(5) == "5:0"
        assert self._norm("0:0") == "0:0"

    def test_v40_success_struct_normalizes_to_zero(self) -> None:
        value = {
            "status": ["SUCCESS"],
            "return_code": {"set": True, "infinite": False, "number": 0},
            "signal": {
                "id": {"set": False, "infinite": False, "number": 0},
                "name": "",
            },
        }
        assert self._norm(value) == "0:0"

    def test_v40_failure_struct_uses_numbers(self) -> None:
        value = {
            "status": ["ERROR"],
            "return_code": {"set": True, "number": 1},
            "signal": {"id": {"set": True, "number": 9}, "name": "KILL"},
        }
        assert self._norm(value) == "1:9"

    def test_success_status_string_without_return_code(self) -> None:
        assert self._norm({"status": "SUCCESS", "return_code": {"set": False}}) == "0:0"

    def test_error_status_without_numbers_joins_names(self) -> None:
        assert self._norm({"status": ["ERROR", None]}) == "ERROR:0"

    def test_unknown_shapes_return_none(self) -> None:
        assert self._norm({"status": True}) is None
        assert self._norm({"return_code": {"set": True, "number": None}}) is None
        assert self._norm({}) is None

    def test_legacy_plain_int_status_and_signal(self) -> None:
        assert self._norm({"status": 1, "signal": 9}) == "1:9"

    def test_no_val_struct_without_set_or_id_is_ignored(self) -> None:
        assert (
            self._norm({"return_code": {"number": 7}, "status": ["ERROR"]}) == "ERROR:0"
        )
