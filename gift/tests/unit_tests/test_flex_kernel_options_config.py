"""Torch-free decoding of supported scalar FlexAttention kernel options."""

import pytest

from torchtitan.config.decode import decode
from torchtitan.config.errors import ConfigSchemaError
from torchtitan.config.schema.models.common.attention.softmax.kernels.flex import (
    FlexAttentionConfig,
)
from torchtitan.config.validation import ValidationContext


@pytest.mark.parametrize("backend", ["AUTO", "TRITON", "FLASH", "TRITON_DECODE"])
def test_backend_string_decodes_without_runtime_import(backend):
    cfg = decode(FlexAttentionConfig, {"kernel_options": {"BACKEND": backend}})
    assert cfg.kernel_options["BACKEND"] == backend
    assert not list(cfg.validate(ValidationContext(stage="pre_submit")))


def test_scalar_options_keep_their_types():
    options = {"BLOCK_M": 64, "PRESCALE_QK": True, "fwd_FLOAT32_PRECISION": "ieee"}
    cfg = decode(FlexAttentionConfig, {"kernel_options": options})
    assert cfg.kernel_options == options
    assert type(cfg.kernel_options["BLOCK_M"]) is int
    assert type(cfg.kernel_options["PRESCALE_QK"]) is bool


@pytest.mark.parametrize("backend", ["flash", "OTHER", True, 123])
def test_invalid_backend_is_a_config_error(backend):
    with pytest.raises(ConfigSchemaError, match="kernel_options.BACKEND"):
        decode(FlexAttentionConfig, {"kernel_options": {"BACKEND": backend}})


def test_backend_rejects_legacy_selection_knob():
    with pytest.raises(ConfigSchemaError, match="FORCE_USE_FLEX_ATTENTION"):
        decode(
            FlexAttentionConfig,
            {
                "kernel_options": {
                    "BACKEND": "TRITON",
                    "FORCE_USE_FLEX_ATTENTION": False,
                }
            },
        )


@pytest.mark.parametrize("value", [[64], {"nested": 64}, 1.5])
def test_nonscalar_or_unsupported_value_is_rejected(value):
    with pytest.raises(ConfigSchemaError, match="kernel_options"):
        decode(FlexAttentionConfig, {"kernel_options": {"BLOCK_M": value}})
