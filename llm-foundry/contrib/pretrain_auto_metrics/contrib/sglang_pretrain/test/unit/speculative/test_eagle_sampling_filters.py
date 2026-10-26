from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.speculative.eagle_utils import eagle_sample


@pytest.mark.parametrize(
    "need_k,need_p", [(False, False), (True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA softmax coverage"
            ),
        ),
    ],
)
def test_eagle_sampling_always_applies_filters(need_k, need_p, device):
    """Verify computes target probs with torch softmax and always runs the
    top-k and top-p renorms, whatever the need_top_k/top_p flags say."""
    sampling = SimpleNamespace(
        is_all_greedy=False,
        acc_additive_penalties=None,
        acc_scaling_penalties=None,
        logit_bias=None,
        temperatures=torch.tensor([[0.5], [1.5]], device=device),
        top_ks=torch.tensor([2, 4], device=device),
        top_ps=torch.tensor([0.8, 1.0], device=device),
        need_top_k_sampling=need_k,
        need_top_p_sampling=need_p,
    )
    batch = SimpleNamespace(
        device=device,
        seq_lens=[1, 1],
        sampling_info=sampling,
        forward_mode=SimpleNamespace(is_idle=lambda: False),
    )
    verify = SimpleNamespace(
        draft_token=torch.zeros(4, dtype=torch.int64, device=device),
        draft_token_num=2,
        max_tree_depth=2,
        retrieve_index=None,
        retrieve_next_token=None,
        retrieve_next_sibling=None,
    )
    logits = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
    calls = []

    def top_k(probs, ks):
        calls.append("k")
        torch.testing.assert_close(ks, torch.tensor([2, 2, 4, 4], device=device))
        return probs * 0.5

    def top_p(probs, ps):
        calls.append("p")
        torch.testing.assert_close(
            ps, torch.tensor([0.8, 0.8, 1.0, 1.0], device=device)
        )
        return probs * 0.25

    def sample(**kwargs):
        expected = (
            torch.softmax(
                logits / sampling.temperatures.repeat_interleave(2, dim=0), dim=-1
            )
            * 0.5
            * 0.25
        )
        torch.testing.assert_close(kwargs["target_probs"], expected.reshape(2, 2, 4))
        kwargs["accept_token_num"].zero_()

    def no_flashinfer_softmax(*args, **kwargs):
        raise AssertionError("verify must not use flashinfer.sampling.softmax")

    args = SimpleNamespace(
        speculative_use_rejection_sampling=False,
        speculative_accept_threshold_single=1.0,
        speculative_accept_threshold_acc=1.0,
    )
    with patch("flashinfer.sampling.softmax", side_effect=no_flashinfer_softmax), patch(
        "sgl_kernel.top_k_renorm_prob", side_effect=top_k
    ), patch("sgl_kernel.top_p_renorm_prob", side_effect=top_p), patch(
        "sgl_kernel.tree_speculative_sampling_target_only", side_effect=sample
    ), patch(
        "sglang.srt.server_args.get_global_server_args", return_value=args
    ), patch(
        "sglang.srt.distributed.get_tp_group",
        return_value=SimpleNamespace(world_size=1),
    ), patch(
        "sglang.srt.layers.dp_attention.is_dp_attention_enabled", return_value=False
    ):
        eagle_sample(verify, batch, SimpleNamespace(next_token_logits=logits))
    assert calls == ["k", "p"]
