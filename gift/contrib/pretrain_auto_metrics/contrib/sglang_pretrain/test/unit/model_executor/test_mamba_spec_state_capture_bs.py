import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.model_executor.model_runner_kv_cache_mixin import (  # noqa: E402
    ModelRunnerKVCacheMixin,
)


class _FakeModelRunner(ModelRunnerKVCacheMixin):
    def __init__(self, **attrs):
        self.__dict__.update(attrs)


class _PoolSizingCaptured(Exception):
    def __init__(self, kwargs):
        self.kwargs = kwargs


def _dp4_server_args():
    return SimpleNamespace(
        enable_dp_attention=True,
        dp_size=4,
        moe_dense_tp_size=None,
        enable_dp_lm_head=False,
        disable_attn_tp_gather=False,
        enable_two_batch_overlap=False,
        cuda_graph_config=SimpleNamespace(
            decode=SimpleNamespace(bs=[1, 2, 4, 8, 16, 32, 64, 128, 256])
        ),
        enable_torch_compile=False,
        torch_compile_max_bs=32,
        speculative_algorithm="EAGLE",
        speculative_eagle_topk=1,
        max_speculative_num_draft_tokens=4,
        page_size=1,
        disaggregation_mode="null",
        max_mamba_cache_size=179,
        enable_memory_saver=False,
        disable_overlap_schedule=False,
        enable_mamba_extra_buffer=lambda: False,
        enable_mamba_extra_buffer_lazy=lambda: False,
        enable_gdn_spec_replay_commit=False,
    )


def _fake_model_runner(server_args):
    return _FakeModelRunner(
        max_running_requests=server_args.max_mamba_cache_size // 3,
        req_to_token_pool=None,
        server_args=server_args,
        device="cpu",
        start_layer=0,
        end_layer=40,
        model_config=SimpleNamespace(context_len=8192),
        mambaish_config=SimpleNamespace(
            mamba2_cache_params=SimpleNamespace(layers=[3, 7, 11])
        ),
    )


class TestMambaSpecStateCoversCaptureBs(unittest.TestCase):
    def test_dp4_odd_cap_spec_state_covers_padded_capture_bs(self):
        from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
            get_batch_sizes_to_capture,
        )

        server_args = _dp4_server_args()
        mr = _fake_model_runner(server_args)
        self.assertEqual(mr.max_running_requests, 59)

        def _record_pool(**kwargs):
            raise _PoolSizingCaptured(kwargs)

        parallel = SimpleNamespace(attn_tp_size=4, attn_cp_size=1)
        with patch(
            "sglang.srt.model_executor.runner.base_cuda_graph_runner.get_parallel",
            return_value=parallel,
        ), patch("sglang.srt.runtime_context.get_parallel", return_value=parallel):
            with patch(
                "sglang.srt.model_executor.model_runner_kv_cache_mixin.HybridReqToTokenPool",
                side_effect=_record_pool,
            ):
                with self.assertRaises(_PoolSizingCaptured) as ctx:
                    mr._init_pools()

            capture_bs, _ = get_batch_sizes_to_capture(
                SimpleNamespace(
                    server_args=server_args,
                    req_to_token_pool=SimpleNamespace(size=mr.max_running_requests),
                )
            )
            plan_spec_rows = mr._mamba_spec_state_size(mr.max_running_requests)

        pool_kwargs = ctx.exception.kwargs
        max_capture_bs = max(capture_bs)
        self.assertEqual(pool_kwargs["size"], 59)
        self.assertEqual(max_capture_bs, 60)

        spec_state_size = pool_kwargs["mamba_spec_state_size"]
        self.assertGreaterEqual(spec_state_size, max_capture_bs)
        self.assertEqual(plan_spec_rows, spec_state_size + 1)

        verify_seat_table = torch.arange(spec_state_size, dtype=torch.int32)
        self.assertEqual(verify_seat_table[:max_capture_bs].shape[0], max_capture_bs)


if __name__ == "__main__":
    unittest.main()
