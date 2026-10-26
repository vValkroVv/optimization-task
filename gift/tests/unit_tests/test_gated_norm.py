import unittest

import torch
import torch.nn as nn

from torchtitan.models.common.norms import GatedRMSNorm
from torchtitan.config.schema.protocols.module import Normal, Ones, Zeros


def _param_init(*, weight_init):
    return {
        "weight": weight_init,
        "gate_up_weight": Normal(std=0.02),
        "gate_down_weight": Normal(std=0.02),
    }


class TestGatedRMSNorm(unittest.TestCase):
    """Tests for the self-gated RMSNorm (GatedRMSNorm)."""

    def test_config_build_param_shapes(self):
        norm = GatedRMSNorm.Config(normalized_shape=32, gate_rank=8).build()
        self.assertIsInstance(norm, GatedRMSNorm)
        self.assertEqual(norm.weight.shape, torch.Size([32]))
        self.assertEqual(norm.gate_up_weight.shape, torch.Size([8, 32]))
        self.assertEqual(norm.gate_down_weight.shape, torch.Size([32, 8]))

    def test_partial_config_cannot_be_built(self):
        # The width is the owner's to fill; without it there is nothing to build.
        with self.assertRaises(ValueError):
            GatedRMSNorm.Config().build()

    def test_defaults(self):
        norm = GatedRMSNorm.Config(normalized_shape=16).build()
        self.assertEqual(norm.gating_weight, 2.0)
        self.assertFalse(norm.zero_centered)

    def test_init_states(self):
        norm = GatedRMSNorm.Config(
            normalized_shape=16, param_init=_param_init(weight_init=Zeros())
        ).build()
        nn.init.ones_(norm.weight)
        norm.init_states()
        self.assertTrue(torch.all(norm.weight == 0))  # zeros init applied
        self.assertTrue(torch.isfinite(norm.gate_up_weight).all())
        self.assertTrue(torch.isfinite(norm.gate_down_weight).all())

    def test_zero_centered_vs_standard_differ(self):
        """Same weights, the (1+w) vs (w) scaling must produce different outputs."""
        torch.manual_seed(0)
        cfg = dict(normalized_shape=32, gate_rank=8)
        std = GatedRMSNorm.Config(
            zero_centered=False,
            param_init=_param_init(weight_init=Normal(std=0.3)),
            **cfg,
        ).build()
        std.init_states()
        zc = GatedRMSNorm.Config(
            zero_centered=True,
            param_init=_param_init(weight_init=Zeros()),
            **cfg,
        ).build()
        zc.init_states()
        # Copy std's params into zc so only the zero_centered flag differs.
        with torch.no_grad():
            zc.weight.copy_(std.weight)
            zc.gate_up_weight.copy_(std.gate_up_weight)
            zc.gate_down_weight.copy_(std.gate_down_weight)
        x = torch.randn(2, 5, 32)
        self.assertFalse(torch.allclose(std(x), zc(x)))

    def test_gradients_flow(self):
        norm = GatedRMSNorm.Config(
            normalized_shape=32, param_init=_param_init(weight_init=Ones())
        ).build()
        norm.init_states()
        x = torch.randn(2, 5, 32, requires_grad=True)
        norm(x).pow(2).sum().backward()
        for p in (norm.weight, norm.gate_up_weight, norm.gate_down_weight):
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
        self.assertIsNotNone(x.grad)


class TestGatedNormBackend(unittest.TestCase):
    def test_backend_config_validation(self):
        self.assertEqual(
            GatedRMSNorm.Config(normalized_shape=128).norm_backend, "compile"
        )
        for overrides, message in (
            ({"norm_backend": "unknown"}, "must be.*compile.*cute"),
            ({"normalized_shape": 96}, "multiple of 128"),
            ({"gate_rank": 8}, "gate_rank=16"),
            ({"eps": 0}, "finite and positive"),
        ):
            with (
                self.subTest(overrides=overrides),
                self.assertRaisesRegex(ValueError, message),
            ):
                GatedRMSNorm.Config(
                    **{"normalized_shape": 128, "norm_backend": "cute", **overrides}
                )

    def test_cute_rejects_row_count_overflow(self):
        with torch.device("meta"):
            norm = GatedRMSNorm.Config(
                normalized_shape=128, norm_backend="cute"
            ).build()
            x = torch.empty(2**31, 128, dtype=torch.bfloat16)
            with self.assertRaisesRegex(ValueError, "at most"):
                norm(x)

    def test_checkpoint_compatibility(self):
        config = dict(normalized_shape=128, param_init=_param_init(weight_init=Ones()))
        original = GatedRMSNorm.Config(**config).build()
        original.init_states()
        cute = GatedRMSNorm.Config(**config, norm_backend="cute").build()
        cute.load_state_dict(original.state_dict(), strict=True)
        self.assertEqual(
            set(cute.state_dict()), {"weight", "gate_up_weight", "gate_down_weight"}
        )
        for name, value in original.state_dict().items():
            torch.testing.assert_close(cute.state_dict()[name], value, rtol=0, atol=0)

    def test_cpu_cute_fails_explicitly(self):
        norm = GatedRMSNorm.Config(normalized_shape=128, norm_backend="cute").build()
        with self.assertRaisesRegex(ValueError, "BF16 CUDA"):
            norm(torch.ones(2, 128, dtype=torch.bfloat16))

    def test_compile_backend_model_forward_backward(self):
        torch.manual_seed(42)
        model = nn.Sequential(
            *[
                GatedRMSNorm.Config(
                    normalized_shape=32,
                    zero_centered=zc,
                    param_init=_param_init(weight_init=Ones()),
                ).build()
                for zc in (False, True)
            ]
        )
        for norm in model:
            norm.init_states()
        x = torch.randn(2, 5, 32, requires_grad=True)
        expected = model(x)
        targets = (x, *model.parameters())
        expected_grads = torch.autograd.grad(expected.sum(), targets)
        actual = torch.compile(model, backend="aot_eager", fullgraph=True)(x)
        actual_grads = torch.autograd.grad(actual.sum(), targets)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)

    def test_cute_model_aot_autograd_without_cuda_runtime(self):
        """Exercise the real registrations through AOTAutograd on meta tensors."""
        with torch.device("meta"):
            model = nn.Sequential(
                *[
                    GatedRMSNorm.Config(
                        normalized_shape=128, norm_backend="cute", zero_centered=zc
                    ).build()
                    for zc in (False, True)
                ]
            )
            compiled = torch.compile(
                model, backend="aot_eager", fullgraph=True, dynamic=True
            )
            for rows in (0, 1, 65, 129):
                x = torch.randn(2, rows, 128, dtype=torch.bfloat16, requires_grad=True)
                y = compiled(x)
                grads = torch.autograd.grad(y.sum(), (x, *model.parameters()))
                self.assertEqual(y.shape, x.shape)
                self.assertEqual(y.dtype, x.dtype)
                for grad, tensor in zip(grads, (x, *model.parameters())):
                    self.assertEqual(grad.shape, tensor.shape)
                    self.assertEqual(grad.dtype, tensor.dtype)


if __name__ == "__main__":
    unittest.main()
