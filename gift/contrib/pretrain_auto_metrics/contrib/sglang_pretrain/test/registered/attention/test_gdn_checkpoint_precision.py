import unittest

import torch

from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_cuda_ci, register_xpu_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")
register_xpu_ci(est_time=900, suite="stage-b-test-1-gpu-xpu")


@unittest.skipIf(
    not (torch.cuda.is_available() or torch.xpu.is_available()),
    "Test requires CUDA or XPU",
)
class TestGDNCheckpointPrecision(unittest.TestCase):
    def test_tracking_preserves_output_with_fp16_inputs_and_bf16_state(self):
        device = get_device()
        q = torch.zeros(1, 513, 1, 64, device=device, dtype=torch.float16)
        q[..., 0] = 1
        k = torch.zeros_like(q)
        k[:, 0, :, 0] = 1
        v = torch.zeros_like(q)
        v[:, 0, :, 0] = 2.005859375
        g = torch.zeros(1, 513, 1, device=device)
        beta = torch.full_like(g, 0.5, dtype=torch.float16)
        initial = torch.zeros(1, 1, 64, 64, device=device, dtype=torch.bfloat16)
        indices = torch.zeros(1, device=device, dtype=torch.int64)
        kernel = TritonGDNKernel()

        def extend(checkpoint_dtype):
            return kernel.extend(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                ssm_states=initial.clone(),
                cache_indices=indices,
                query_start_loc=torch.tensor(
                    [0, 513], device=device, dtype=torch.int32
                ),
                intermediate_state_dtype=checkpoint_dtype,
            )

        cold, _, _ = extend(None)
        tracked, _, checkpoints = extend(torch.bfloat16)
        self.assertEqual(checkpoints.dtype, torch.float32)
        torch.testing.assert_close(tracked, cold, rtol=0, atol=0)

    def test_unaligned_checkpoint_preserves_recurrent_state(self):
        device = get_device()
        for key_dim, value_heads in ((64, 2), (128, 2), (192, 4), (256, 4)):
            with self.subTest(key_dim=key_dim, value_heads=value_heads):
                torch.manual_seed(37128 + key_dim)
                shape = (1, 640, 2, key_dim)
                q = torch.randn(shape, device=device, dtype=torch.bfloat16)
                k = torch.randn_like(q)
                v = torch.randn(
                    1, 640, value_heads, 64, device=device, dtype=torch.bfloat16
                )
                g = torch.full((1, 640, value_heads), -0.1, device=device)
                beta = torch.full_like(g, 0.5, dtype=torch.bfloat16)
                initial = torch.randn(1, value_heads, 64, key_dim, device=device) * 0.01
                indices = torch.zeros(1, device=device, dtype=torch.int64)
                kernel = TritonGDNKernel()

                def extend(start, end, state, checkpoint_dtype=None):
                    return kernel.extend(
                        q=q[:, start:end].contiguous(),
                        k=k[:, start:end].contiguous(),
                        v=v[:, start:end].contiguous(),
                        g=g[:, start:end].contiguous(),
                        beta=beta[:, start:end].contiguous(),
                        ssm_states=state,
                        cache_indices=indices,
                        query_start_loc=torch.tensor(
                            [0, end - start], device=device, dtype=torch.int32
                        ),
                        intermediate_state_dtype=checkpoint_dtype,
                    )

                final512 = initial.clone()
                extend(0, 512, final512, torch.float32)
                _, _, h513 = extend(0, 513, initial.clone(), torch.float32)
                checkpoint512 = h513[:, 8].float()
                torch.testing.assert_close(checkpoint512, final512, rtol=0, atol=0)

                cold_state = initial.clone()
                cold, _, _ = extend(0, 640, cold_state)
                tracked_state = initial.clone()
                tracked, _, _ = extend(0, 640, tracked_state, torch.float32)
                torch.testing.assert_close(tracked, cold, rtol=0, atol=0)
                torch.testing.assert_close(tracked_state, cold_state, rtol=0, atol=0)

                warm_state = checkpoint512.clone()
                warm, _, _ = extend(512, 640, warm_state)
                torch.testing.assert_close(warm, cold[:, 512:], rtol=0, atol=0)
                torch.testing.assert_close(warm_state, cold_state, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
