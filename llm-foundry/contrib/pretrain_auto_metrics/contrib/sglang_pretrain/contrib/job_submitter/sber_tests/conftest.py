import sys
import types
from pathlib import Path

import pytest
from job_submitter.core.job_config import JobTemplate

from sber_tests.factories import DummyJob

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import jinja2  # noqa: F401
except ModuleNotFoundError:
    jinja2_stub = types.ModuleType("jinja2")

    class _Template:
        def __init__(self, template: str):
            self.template = template

        def render(self, context):
            return self.template

    jinja2_stub.Template = _Template
    sys.modules["jinja2"] = jinja2_stub


@pytest.fixture(autouse=True)
def stable_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NAMESPACE", "tests")
    monkeypatch.setenv("NB_PREFIX", "/x/y/tests")
    monkeypatch.setenv("WORKSPACE_ID", "ws-tests")
    monkeypatch.setenv("MLS_NTB_WORKSPACE_NAME", "ws-name-tests")
    monkeypatch.setenv("GWAPI_ADDR", "gw.local")
    monkeypatch.setenv("GWAPI_V2_URL", "gw-v2.local")
    monkeypatch.setenv("GWAPI_KEY", "test-key")
    monkeypatch.setenv("CLUSTER_KEY", "SR008")
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "o1")


@pytest.fixture
def cloud_template() -> JobTemplate:
    return JobTemplate(
        submission_type="cloud",
        user_login="tester",
        exp_name="exp",
        job_desc="tester | exp",
        base_image="cr.ai.cloud.ru/repo/image:tag",
        script="echo hello",
        n_workers=1,
        processes_per_worker=1,
        script_type="binary",
        region="SR008",
    )


@pytest.fixture
def slurm_template() -> JobTemplate:
    return JobTemplate(
        submission_type="slurm",
        user_login="tester",
        exp_name="exp",
        job_desc="tester | exp",
        base_image="/tmp/image.sqsh",
        script="echo hello",
        n_workers=1,
        processes_per_worker=1,
        script_type="binary",
        region="o1",
        partition="middle",
    )


@pytest.fixture
def dummy_job() -> DummyJob:
    return DummyJob(
        user_login="tester",
        exp_name="exp",
        region="SR008",
        submission_type="cloud",
        n_gpus=8,
    )


@pytest.fixture
def tmp_yaml(tmp_path: Path):
    def _writer(name: str, content: str) -> Path:
        path = tmp_path / name
        path.write_text(content, encoding="utf-8")
        return path

    return _writer
