import pytest
import torch

from llmfoundry.models.gigaomni.modelling_gigaomni import GigaOmniForCausalLM
from sber_tests.unit.gigaomni_merge_harness import build_merge_inputs


@pytest.mark.unit
def test_merge_matches_reference_masked_scatter_and_is_contiguous() -> None:
    """A valid modality merge must equal a hand-derived masked-scatter result."""
    model, state = build_merge_inputs("cpu")
    expected = torch.tensor([[[1.0, 1.0], [0.0, 1.0], [3.0, 3.0], [2.0, 3.0]]])

    actual = GigaOmniForCausalLM.merge_modality_embeddings(model, state)

    torch.testing.assert_close(actual, expected)
    assert actual.is_contiguous()


@pytest.mark.unit
def test_cpu_token_count_mismatch_raises_documented_error() -> None:
    """CPU validation must reject mismatched placeholder and embedding counts."""
    model, state = build_merge_inputs("cpu", modality_rows=1)

    with pytest.raises(
        AssertionError,
        match="must equal the number of output tokens",
    ):
        GigaOmniForCausalLM.merge_modality_embeddings(model, state)
