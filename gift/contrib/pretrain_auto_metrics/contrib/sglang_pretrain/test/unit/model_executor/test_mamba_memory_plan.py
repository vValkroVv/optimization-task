import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.configs.mamba_utils import (  # noqa: E402
    Mamba2CacheParams,
    Mamba2StateShape,
)
from sglang.srt.disaggregation.decode import (  # noqa: E402
    HybridMambaDecodeReqToTokenPool,
)
from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.mem_cache.memory_pool import (  # noqa: E402
    HybridReqToTokenPool,
    get_tensor_size_bytes,
)
from sglang.srt.model_executor.model_runner_kv_cache_mixin import (  # noqa: E402
    ModelRunnerKVCacheMixin,
)

_zeros = torch.zeros


def _meta_zeros(*args, **kwargs):
    kwargs["device"] = "meta"
    return _zeros(*args, **kwargs)


class _FakeModelRunner(ModelRunnerKVCacheMixin):
    def __init__(self, **attrs):
        self.__dict__.update(attrs)


def _cache_params():
    mamba_layers = [0, 1, 2]
    with envs.SGLANG_MAMBA_SSM_DTYPE.override("float32"):
        shape = Mamba2StateShape.create(
            tp_world_size=1,
            intermediate_size=4096,
            n_groups=16,
            num_heads=32,
            head_dim=128,
            state_size=128,
            conv_kernel=4,
        )
        return Mamba2CacheParams(shape=shape, layers=mamba_layers)


def _server_args(replay, extra_buffer, draft_tokens, mode="decode"):
    return SimpleNamespace(
        enable_dp_attention=False,
        dp_size=1,
        moe_dense_tp_size=None,
        enable_dp_lm_head=False,
        disable_attn_tp_gather=False,
        enable_two_batch_overlap=False,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=draft_tokens,
        max_speculative_num_draft_tokens=draft_tokens,
        max_running_requests=32,
        max_mamba_cache_size=None,
        disable_radix_cache=True,
        disaggregation_mode=mode,
        disaggregation_decode_extra_slots=4,
        disable_overlap_schedule=False,
        enable_gdn_spec_replay_commit=replay,
        enable_mamba_extra_buffer=lambda: extra_buffer,
        enable_mamba_extra_buffer_lazy=lambda: False,
    )


def _allocated_bytes(pool):
    cache = pool.mamba_pool.mamba_cache
    return (
        get_tensor_size_bytes(cache.conv)
        + get_tensor_size_bytes(cache.temporal)
        + get_tensor_size_bytes(cache.intermediate_ssm)
        + get_tensor_size_bytes(pool.mamba_pool._intermediate_conv_window_phys)
    )


class TestMambaMemoryPlanMatchesPool(unittest.TestCase):
    def _plan(self, server_args, cache_params, max_num_reqs):
        mr = _FakeModelRunner(
            server_args=server_args,
            mambaish_config=SimpleNamespace(mamba2_cache_params=cache_params),
            hybrid_gdn_config=object(),
            spec_algorithm=SimpleNamespace(is_none=lambda: False),
            pp_size=1,
            dp_size=1,
        )
        parallel = SimpleNamespace(attn_tp_size=1, attn_cp_size=1)
        with patch(
            "sglang.srt.model_executor.runner.base_cuda_graph_runner.get_parallel",
            return_value=parallel,
        ), patch("sglang.srt.runtime_context.get_parallel", return_value=parallel):
            rest_gb = 100.0
            planned = (rest_gb - mr.handle_max_mamba_cache(rest_gb)) * (1 << 30)
            spec_state_size = mr._mamba_spec_state_size(max_num_reqs) - 1
        return planned, spec_state_size

    def _check(self, replay, extra_buffer, draft_tokens):
        cache_params = _cache_params()
        server_args = _server_args(replay, extra_buffer, draft_tokens)
        planned, spec_state_size = self._plan(server_args, cache_params, 32)

        with patch("torch.zeros", _meta_zeros):
            pool = HybridMambaDecodeReqToTokenPool(
                size=32,
                max_context_len=16,
                device="cpu",
                enable_memory_saver=False,
                cache_params=cache_params,
                mamba_layer_ids=cache_params.layers,
                speculative_num_draft_tokens=draft_tokens,
                enable_mamba_extra_buffer=extra_buffer,
                pre_alloc_size=server_args.disaggregation_decode_extra_slots,
                enable_overlap_schedule=True,
                mamba_spec_state_size=spec_state_size,
                mamba_size=server_args.max_mamba_cache_size,
                speculative_eagle_topk=1,
                gdn_spec_replay=replay,
            )
        self.assertAlmostEqual(planned, _allocated_bytes(pool), delta=1024)

    def test_pd_decode_plan_matches_allocation(self):
        for replay in (False, True):
            for extra_buffer in (False, True):
                for draft_tokens in (2, 4):
                    with self.subTest(
                        replay=replay, extra_buffer=extra_buffer, draft=draft_tokens
                    ):
                        self._check(replay, extra_buffer, draft_tokens)

    def test_pd_prefill_req_pool_floor_is_planned(self):
        cache_params = _cache_params()
        server_args = _server_args(False, False, 3, mode="prefill")
        with envs.SGLANG_PD_PREFILL_REQ_POOL_SIZE.override(256):
            planned, _ = self._plan(server_args, cache_params, 256)
        with patch("torch.zeros", _meta_zeros):
            pool = HybridReqToTokenPool(
                size=256,
                mamba_size=32,
                mamba_spec_state_size=256,
                max_context_len=16,
                device="cpu",
                enable_memory_saver=False,
                cache_params=cache_params,
                mamba_layer_ids=cache_params.layers,
                enable_mamba_extra_buffer=False,
                speculative_num_draft_tokens=3,
                speculative_eagle_topk=1,
            )
        self.assertAlmostEqual(planned, _allocated_bytes(pool), delta=1024)


if __name__ == "__main__":
    unittest.main()
