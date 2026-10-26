from __future__ import annotations

from typing import Any

import pytest


def assert_dry_run_submit_contract(
    *,
    monkeypatch: pytest.MonkeyPatch,
    job: Any,
    patch_anonimize: bool = False,
) -> None:
    """Assert common dry-run behavior for submit() implementations."""
    calls = {
        "to_config": 0,
        "assert_job": 0,
        "anonimize": 0,
        "before_submit": 0,
        "submit_config": 0,
        "backend_submit": 0,
    }
    fake_cfg = object()

    def _to_config_stub():
        calls["to_config"] += 1
        return fake_cfg

    def _assert_job_stub(*_args, **_kwargs):
        calls["assert_job"] += 1

    monkeypatch.setattr(job, "to_config", _to_config_stub)
    monkeypatch.setattr(job, "assert_job", _assert_job_stub)

    if patch_anonimize:
        monkeypatch.setattr(
            job,
            "_anonimize_config",
            lambda *_args, **_kwargs: calls.__setitem__(
                "anonimize", calls["anonimize"] + 1
            ),
        )

    monkeypatch.setattr(
        job,
        "_before_submit",
        lambda: calls.__setitem__("before_submit", calls["before_submit"] + 1),
    )
    monkeypatch.setattr(
        "job_submitter.jobs.base.CloudBackend.submit",
        lambda self, _cfg: calls.__setitem__(
            "backend_submit", calls["backend_submit"] + 1
        ),
    )
    monkeypatch.setattr(
        "job_submitter.jobs.base.SlurmBackend.submit",
        lambda self, _cfg: calls.__setitem__(
            "backend_submit", calls["backend_submit"] + 1
        ),
    )

    original_submit_config = job._submit_config

    def _submit_config_stub(cfg):
        calls["submit_config"] += 1
        return original_submit_config(cfg)

    monkeypatch.setattr(job, "_submit_config", _submit_config_stub)

    assert job.submit() is job
    assert calls["to_config"] == 1
    assert calls["assert_job"] == 1
    if patch_anonimize:
        assert calls["anonimize"] == 1
    assert calls["before_submit"] == 1
    assert calls["submit_config"] == 1
    assert calls["backend_submit"] == 0
