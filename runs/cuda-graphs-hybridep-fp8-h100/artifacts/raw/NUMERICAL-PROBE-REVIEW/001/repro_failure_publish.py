"""CPU fault injection at the actual batch_end failure-report boundary."""

import json
import runpy
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
STAGED = ROOT / "runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/NUMERICAL-PROBE-IMPLEMENTATION/001/staged/gift"
subject = runpy.run_path(str(STAGED / "tests/unit_tests/test_campaign_numerical_probe.py"))["probe_module"]


def quota_failure(*args, **kwargs):
    raise OSError("injected ENOSPC/quota refusal")


with tempfile.TemporaryDirectory(dir=HERE, prefix="failure-fixture-") as temporary:
    probe = object.__new__(subject.CampaignNumericalProbe)
    probe.config = SimpleNamespace(mode="observe", updates=1)
    probe._current = {"step": 2, "rng_restored_checked": True, "updated_parameters": {"p": {}}}
    probe._active_writer = SimpleNamespace(add=lambda *args, **kwargs: {})
    probe.output_dir = Path(temporary)
    probe.rank = 0
    probe.budget = SimpleNamespace(reserve=quota_failure)
    probe._native_state = lambda: {}
    probe._save_small_state = quota_failure
    publications = []
    probe._publish_phase = lambda *args, **kwargs: publications.append((args, kwargs))
    state = SimpleNamespace(step=2, loss=torch.tensor(1.0), ntokens_seen=4, global_valid_tokens=torch.tensor(4))
    try:
        probe.batch_end(state)
    except OSError as error:
        assert "injected" in str(error)
    else:
        raise AssertionError("Current failure-status write must escape before phase publication")
    assert not publications
    print(json.dumps({"injected_fault": "native small-state ENOSPC then failure-report budget refusal", "collective_phase_publications": len(publications), "current_handler_escaped_before_collective": True}))
