import pytest

pytestmark = pytest.mark.unit

from sglang.srt.entrypoints.openai.mean_var.decoder import decode_pair


def test_decoder_uses_protocol_floors(protocol):
    protocol["sigma_floor"] = 0.3
    with pytest.raises(ValueError, match="invalid sigma"):
        decode_pair([[1, 0.2], [0, 0.4]], protocol)
    protocol["sigma_floor"] = 0.01
    protocol["probability_floor"] = 0.2
    result = decode_pair([[1000, 0.02], [0, 0.02]], protocol)
    assert result["probabilities"] == pytest.approx({"A>B": 0.75, "B>A": 0.125, "A=B": 0.125})


@pytest.mark.parametrize(
    "outputs,verdict",
    [
        ([[1, 0.1], [0, 0.1]], "A>B"),
        ([[0, 0.1], [1, 0.1]], "B>A"),
        ([[0, 0.1], [0, 0.1]], "A=B"),
    ],
)
def test_expected_class(outputs, verdict, protocol):
    result = decode_pair(outputs, protocol)
    assert result["verdict"] == verdict
    assert sum(result["probabilities"].values()) == pytest.approx(1, abs=2e-7)


def test_original_cached_fp32_probabilities_match_within_roundoff(protocol):
    result = decode_pair([[3.0764319896698, 0.8029559254646301], [2.706117630004883, 0.7036908268928528]], protocol)
    assert result["probabilities"] == pytest.approx({
        "A>B": 0.455522745847702,
        "B>A": 0.3133675456047058,
        "A=B": 0.23110966384410858,
    }, abs=1e-7, rel=0)
    assert result["verdict"] == "A>B"


@pytest.mark.parametrize(
    "outputs",
    [
        [],
        [[1, 0.1]],
        [[1], [2]],
        [[float("nan"), 0.1], [1, 0.1]],
        [[1, 0.09], [1, 0.1]],
        [[float("inf"), 0.1], [1, 0.1]],
    ],
)
def test_invalid_outputs_rejected(outputs, protocol):
    with pytest.raises(ValueError):
        decode_pair(outputs, protocol)
