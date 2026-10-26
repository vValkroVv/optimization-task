from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.model_executor.cuda_graph_config import default_cuda_graph_config
from sglang.srt.server_args import ServerArgs


def _make_args(attention_arch):
    args = ServerArgs(model_path="dummy")
    args.cuda_graph_config = default_cuda_graph_config()
    args._quantization_explicitly_unset = False
    hf_config = SimpleNamespace(
        architectures=["Gigachat35ForCausalLM"],
        get_text_config=lambda: SimpleNamespace(),
    )
    args.get_model_config = lambda: SimpleNamespace(
        hf_config=hf_config,
        hf_text_config=hf_config,
        attention_arch=attention_arch,
    )
    return args


@pytest.mark.unit
@pytest.mark.parametrize(
    "attention_arch,expected_backend",
    [(AttentionArch.MHA, None), (AttentionArch.MLA, "trtllm_mla")],
)
def test_sm100_trtllm_mla_default_requires_mla(attention_arch, expected_backend):
    args = _make_args(attention_arch)

    with patch("sglang.srt.server_args.is_sm100_supported", return_value=True), patch(
        "sglang.srt.server_args.has_fp8_weights_in_checkpoint", return_value=False
    ), patch("sglang.srt.server_args.get_quantization_config", return_value=None):
        args._handle_model_specific_adjustments()

    assert args.use_mla_backend() is (attention_arch == AttentionArch.MLA)
    assert args.attention_backend == expected_backend
