import subprocess
import sys

import pytest
import torch

from llmfoundry.models.gigaomni.modelling_gigaomni import GigaOmniForCausalLM
from sber_tests.unit.gigaomni_merge_harness import build_merge_inputs


@pytest.mark.unit
@pytest.mark.gpu
def test_cuda_merge_matches_reference_masked_scatter_and_is_contiguous() -> None:
    """A valid CUDA modality merge must equal a hand-derived masked-scatter result."""
    model, state = build_merge_inputs("cuda")
    expected = torch.tensor(
        [[[1.0, 1.0], [0.0, 1.0], [3.0, 3.0], [2.0, 3.0]]],
        device="cuda",
    )

    actual = GigaOmniForCausalLM.merge_modality_embeddings(model, state)

    torch.testing.assert_close(actual, expected)
    assert actual.is_contiguous()


@pytest.mark.unit
@pytest.mark.gpu
def test_cuda_token_count_mismatch_raises_after_synchronize() -> None:
    """A CUDA mismatch must surface from the asynchronous assertion in isolation."""
    script = """
import torch
from llmfoundry.models.gigaomni.modelling_gigaomni import GigaOmniForCausalLM
import runpy

namespace = runpy.run_path("sber_tests/unit/gigaomni_merge_harness.py")
model, state = namespace["build_merge_inputs"]("cuda", modality_rows=1)
GigaOmniForCausalLM.merge_modality_embeddings(model, state)
torch.cuda.synchronize()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "must equal the number of output tokens" in (result.stdout + result.stderr)


@pytest.mark.unit
@pytest.mark.gpu
def test_valid_cuda_merge_has_no_scalar_extraction_operator() -> None:
    """The valid CUDA path must not synchronize through item-like operators."""
    model, state = build_merge_inputs("cuda")

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
    ) as profile:
        GigaOmniForCausalLM.merge_modality_embeddings(model, state)
    torch.cuda.synchronize()

    operator_names = {event.key for event in profile.key_averages()}
    assert "aten::item" not in operator_names
    assert "aten::_local_scalar_dense" not in operator_names
