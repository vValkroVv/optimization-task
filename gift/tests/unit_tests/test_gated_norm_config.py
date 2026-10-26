"""CuTe backend validation without runtime imports."""

from dataclasses import replace

import pytest

from torchtitan.config.schema.models.common.norms.gated_norm import GatedRMSNormConfig


def test_cute_partial_norm_validates_owner_filled_width():
    config = GatedRMSNormConfig(norm_backend="cute")
    assert replace(config, normalized_shape=128).normalized_shape == 128
    with pytest.raises(ValueError, match="positive multiple of 128"):
        replace(config, normalized_shape=96)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"norm_backend": "unknown"}, "must be.*compile.*cute"),
        ({"normalized_shape": 0}, "positive multiple of 128"),
        ({"normalized_shape": -128}, "positive multiple of 128"),
        ({"gate_rank": 8}, "gate_rank=16"),
        ({"eps": 0}, "finite and positive"),
        ({"eps": float("nan")}, "finite and positive"),
        ({"eps": float("inf")}, "finite and positive"),
    ],
)
def test_cute_norm_rejects_invalid_config(overrides, message):
    with pytest.raises(ValueError, match=message):
        GatedRMSNormConfig(**{"norm_backend": "cute", **overrides})
