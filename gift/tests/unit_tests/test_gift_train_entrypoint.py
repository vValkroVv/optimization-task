import sys
import runpy
from pathlib import Path
from types import ModuleType

import pytest

from gift_scripts import train as train_entrypoint

_SQLITE_S3_QUERY_ROOT = (
    train_entrypoint.REPO_ROOT
    / "contrib"
    / "job_submitter"
    / "contrib"
    / "sqlite_s3_query"
)


def test_gift_train_entrypoint_runs_from_checkout_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    called = {}

    def _main() -> None:
        called["cwd"] = Path.cwd()
        called["sys_path"] = sys.path[0]

    module = ModuleType("torchtitan.train")
    module.main = _main
    monkeypatch.setitem(sys.modules, "torchtitan.train", module)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "configs/experiments/llama3_debug.yaml", "--training.steps", "2"],
    )
    monkeypatch.chdir(tmp_path)

    train_entrypoint.main()

    assert called["cwd"] == train_entrypoint.REPO_ROOT
    assert called["sys_path"] == str(train_entrypoint.REPO_ROOT)


@pytest.mark.parametrize(
    ("script_name", "class_name"),
    [
        ("submit_autometrics_job.py", "AutometricsJobGIFT"),
        (
            "submit_external_api_metrics_job.py",
            "AutometricsJobGIFTExternalAPI",
        ),
    ],
)
def test_gift_metric_submit_examples_import_vendored_jobs(
    script_name: str,
    class_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = sys.path.copy()
    if _SQLITE_S3_QUERY_ROOT.is_dir():
        path.insert(0, str(_SQLITE_S3_QUERY_ROOT))
    monkeypatch.setattr(sys, "path", path)

    namespace = runpy.run_path(
        train_entrypoint.REPO_ROOT / "gift_scripts" / script_name,
        run_name="gift_submit_import_contract",
    )

    assert namespace[class_name].__module__.startswith("job_submitter.jobs.gift.")
