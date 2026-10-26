import sys

import pytest

_JOB_SOURCE = (
    "import json\n"
    "from pathlib import Path\n"
    "class {name}:\n"
    "    def __init__(self, **options):\n"
    "        self.options = options\n"
    "    def submit(self):\n"
    "        root = next(path for path in Path(__file__).resolve().parents if (path / 'scripts').is_dir())\n"
    "        (root / 'submitted.json').write_text(json.dumps(self.options))\n"
)


@pytest.fixture
def foundry(tmp_path, monkeypatch):
    """A llm-foundry checkout whose job classes write the options they were given to submitted.json."""
    root = tmp_path / "llm-foundry"
    (root / "scripts/train").mkdir(parents=True)
    (root / "scripts/train/train.py").write_text("", encoding="utf-8")
    jobs = root / "contrib/job_submitter/job_submitter/jobs"
    (jobs / "llmf").mkdir(parents=True)
    # Regular packages, as in a real checkout: a namespace directory loses to a regular package of the same
    # name further down sys.path, such as a checkout on PYTHONPATH.
    for package in (jobs / "llmf", jobs, jobs.parent, jobs.parent.parent, root / "contrib"):
        (package / "__init__.py").write_text("", encoding="utf-8")
    for module, name in ((jobs / "llmf/train.py", "TrainJobLLMF"), (jobs / "any_script.py", "AnyScriptJob")):
        module.write_text(_JOB_SOURCE.format(name=name), encoding="utf-8")
    monkeypatch.setattr(sys, "path", sys.path.copy())
    modules = sys.modules.copy()
    # A checkout on PYTHONPATH (as in the training container) has imported its own contrib package, which
    # the import of the selected checkout would reuse.
    for name in modules:
        if name == "contrib" or name.startswith("contrib."):
            monkeypatch.delitem(sys.modules, name)
    yield root
    for name in set(sys.modules) - modules.keys():
        if name == "contrib" or name.startswith("contrib."):
            del sys.modules[name]


@pytest.fixture
def site_arguments(foundry, tmp_path):
    return [
        "--llmf-root", str(foundry), "--stand-root", str(tmp_path / "stand"), "--user-login", "researcher",
        "--region", "region-a",
    ]


@pytest.fixture
def common_job_options(tmp_path):
    stand_root = tmp_path.resolve() / "stand"
    return {
        "user_login": "researcher",
        "region": "region-a",
        "job_logdir": str(stand_root / "jobs_logs"),
        "slurm_logdir": str(stand_root / "slurm_logs"),
        "enable_auto_track": False,
        "max_retry": None,
        "extra_env": {},
        "queue_name": "pretrain",
        "priority_class": "high",
    }
