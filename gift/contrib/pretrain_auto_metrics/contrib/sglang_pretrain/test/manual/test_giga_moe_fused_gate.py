"""Numerical equivalence test for sgl_kernel_ext.giga_moe_fused_gate
versus the Python reference biased_grouped_topk_impl. Covers the actual model
configurations: bf16/fp16/fp32 inputs, fp32 bias (default for the GigaChat 3.5
gate), with and without fused-shared-experts (num_fused_shared ∈ {0, 1}),
plus routed_scaling_factor and renormalize.

Run:
    cd /home/askhoroshev/sources/sglang/test
    python -m unittest manual.test_giga_moe_fused_gate -v
"""
import unittest

import torch

from sglang.test.test_utils import CustomTestCase


NUM_EXPERTS = 256
TOPK_ROUTED = 8
ROUTED_SCALING_FACTOR = 2.5
PHANTOM_BASE_ID = NUM_EXPERTS  # phantom expert id when num_fused_shared > 0


@unittest.skipUnless(torch.cuda.is_available(), "CUDA-only kernel")
class TestGigaMoeFusedGate(CustomTestCase):
    """giga_moe_fused_gate kernel vs biased_grouped_topk_impl reference."""

    BATCH_SIZES = [1, 4, 16, 64, 256, 1024]

    @classmethod
    def setUpClass(cls) -> None:
        # Ensure the JIT extension is built and the op is registered.
        import sglang.sgl_kernel_ext as _ext  # noqa: F401
        from sglang.srt.layers.moe.topk import biased_grouped_topk_impl

        cls.kernel = torch.ops.sgl_kernel_ext.giga_moe_fused_gate
        cls.reference = staticmethod(biased_grouped_topk_impl)

    @staticmethod
    def _make_inputs(
        batch_size: int,
        gating_dtype: torch.dtype,
        bias_dtype: torch.dtype,
        seed: int,
    ):
        torch.manual_seed(seed)
        device = "cuda"
        gating = torch.randn(
            batch_size, NUM_EXPERTS, dtype=gating_dtype, device=device
        )
        # Gate bias is small (router learning around 0).
        bias = (
            torch.randn(NUM_EXPERTS, dtype=torch.float32, device=device) * 0.05
        ).to(bias_dtype)
        # `hidden_states` is unused by topk_impl beyond a shape assertion.
        hidden = torch.zeros(
            batch_size, 7168, dtype=gating_dtype, device=device
        )
        return gating, bias, hidden

    def _diff_topk_weights(self, w_k, ids_k, w_r, ids_r):
        """For each row, sort weights by (sorted-by-id) so order doesn't matter;
        return mean / max abs diff over batch. Assumes both have the same id set
        per row — caller checks set equality first."""

        def gather_in_id_order(w, ids):
            order = ids.argsort(dim=-1)
            return torch.gather(w.float(), 1, order)

        if not (w_k.shape == w_r.shape and ids_k.shape == ids_r.shape):
            return None, None
        d = (gather_in_id_order(w_k, ids_k) - gather_in_id_order(w_r, ids_r)).abs()
        return float(d.mean()), float(d.max())

    def _run_case(
        self,
        batch_size: int,
        gating_dtype: torch.dtype,
        bias_dtype: torch.dtype,
        num_fused_shared: int,
        seed: int = 42,
        rsf: float = ROUTED_SCALING_FACTOR,
    ) -> dict:
        gating, bias, hidden = self._make_inputs(
            batch_size, gating_dtype, bias_dtype, seed=seed
        )
        topk_arg = TOPK_ROUTED + num_fused_shared

        # Kernel: (input, bias, topk, num_fused_shared, renormalize, rsf, apply_rsf).
        w_k, ids_k = self.kernel(
            gating, bias, topk_arg, num_fused_shared, True, rsf, False
        )

        # Reference (full Python fallback with the exact fusion semantics).
        w_r, ids_r = self.reference(
            hidden_states=hidden,
            gating_output=gating,
            correction_bias=bias,
            topk=topk_arg,
            renormalize=True,
            num_expert_group=1,
            topk_group=1,
            num_fused_shared_experts=num_fused_shared,
            routed_scaling_factor=rsf,
            apply_routed_scaling_factor_on_output=False,
        )

        # Split into real (first `TOPK_ROUTED`) and phantom (last `num_fused_shared`).
        real_real_ids = TOPK_ROUTED
        real_set_k = [set(ids_k[i, :real_real_ids].tolist()) for i in range(batch_size)]
        real_set_r = [set(ids_r[i, :real_real_ids].tolist()) for i in range(batch_size)]
        rows_real_match = sum(
            1 for sk, sr in zip(real_set_k, real_set_r) if sk == sr
        )

        # Phantom slot check (only meaningful when num_fused_shared > 0).
        if num_fused_shared > 0:
            phantom_ok = bool((ids_r[:, -1] == PHANTOM_BASE_ID).all().item())
            phantom_k = ids_k[:, -1].tolist()
            kernel_phantom_ok = all(p >= PHANTOM_BASE_ID for p in phantom_k)
        else:
            phantom_ok = True
            kernel_phantom_ok = True

        mean_w, max_w = self._diff_topk_weights(w_k, ids_k, w_r, ids_r)

        return {
            "batch": batch_size,
            "input_dtype": gating_dtype,
            "bias_dtype": bias_dtype,
            "fused_shared": num_fused_shared,
            "rsf": rsf,
            "rows_real_match": rows_real_match,
            "rows_total": batch_size,
            "ref_phantom_at_256": phantom_ok,
            "kernel_phantom_ok": kernel_phantom_ok,
            "weights_mean_d": mean_w,
            "weights_max_d": max_w,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Tests
    # ──────────────────────────────────────────────────────────────────────

    def test_no_fusion_bf16_input_fp32_bias(self) -> None:
        """The main model config: bf16 gating, fp32 bias, no fused-shared.
        All routed-real sets and weights must match the Python reference."""
        for bs in self.BATCH_SIZES:
            with self.subTest(batch_size=bs):
                s = self._run_case(
                    bs, torch.bfloat16, torch.float32, num_fused_shared=0
                )
                self.assertEqual(
                    s["rows_real_match"], s["rows_total"],
                    f"row sets differ for bs={bs}: {s}",
                )
                self.assertLess(s["weights_max_d"], 1e-2, str(s))

    def test_fusion_bf16_input_fp32_bias(self) -> None:
        """The actual production config: num_fused_shared=1. Real routed
        experts (first 8 slots) and weights must match, plus the kernel must
        put the phantom expert id (256) in slot 9 and the corresponding weight
        must equal `sum(first 8) / routed_scaling_factor` pre-renorm."""
        for bs in self.BATCH_SIZES:
            with self.subTest(batch_size=bs):
                s = self._run_case(
                    bs, torch.bfloat16, torch.float32, num_fused_shared=1
                )
                self.assertEqual(
                    s["rows_real_match"], s["rows_total"],
                    f"real routed sets differ for bs={bs}: {s}",
                )
                self.assertTrue(
                    s["kernel_phantom_ok"],
                    f"kernel did not place phantom id in last slot: {s}",
                )
                self.assertLess(s["weights_max_d"], 1e-2, str(s))

    def test_dtypes_no_fusion(self) -> None:
        """All 3×3 dtype combinations work, no fusion. Quick sanity."""
        bs = 32
        for in_dt in (torch.float32, torch.bfloat16, torch.float16):
            for b_dt in (torch.float32, torch.bfloat16, torch.float16):
                with self.subTest(input=in_dt, bias=b_dt):
                    s = self._run_case(bs, in_dt, b_dt, num_fused_shared=0)
                    self.assertEqual(s["rows_real_match"], s["rows_total"], str(s))

    def test_rsf_sweep(self) -> None:
        """Sweep routed_scaling_factor × num_fused_shared on fp32 inputs (no
        bf16-sigmoid noise — kernel and reference compute identical scores, so
        we can assert sub-ULP weight equivalence). Verifies that the phantom
        weight `sum_real/rsf` and split renormalize behave correctly across
        scaling factors, both with fusion (N=1) and without (N=0)."""
        bs = 256
        # rsf=1.0 makes phantom weight == sum_real (degenerate but valid).
        # rsf=2.5 is the GigaChat 3.5 production value. 4.0 / 8.0 stress the
        # scaling math further.
        for rsf in (1.0, 2.5, 4.0, 8.0):
            for nfs in (0, 1):
                with self.subTest(rsf=rsf, num_fused_shared=nfs):
                    s = self._run_case(
                        bs, torch.float32, torch.float32,
                        num_fused_shared=nfs, rsf=rsf,
                    )
                    self.assertEqual(
                        s["rows_real_match"], s["rows_total"],
                        f"real routed sets differ: {s}",
                    )
                    self.assertTrue(s["kernel_phantom_ok"], str(s))
                    # fp32 path — expect numerically exact match.
                    self.assertLess(s["weights_max_d"], 1e-5, str(s))


if __name__ == "__main__":
    unittest.main()
