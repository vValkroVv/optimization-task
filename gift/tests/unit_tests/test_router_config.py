import unittest

from torchtitan.config.schema.models.common.moe.const import ScoreFunc
from torchtitan.models.common.moe.router import (
    TokenChoiceTopKRouter,
)
from torchtitan.models.common.nn_modules import Linear


def _make(*, num_experts: int, **kwargs) -> TokenChoiceTopKRouter.Config:
    """Build a router Config; num_experts is derived from the gate width."""
    return TokenChoiceTopKRouter.Config(
        gate=Linear.Config(in_features=8, out_features=num_experts, param_init={}),
        **kwargs,
    )


class TestRouterConfigValidation(unittest.TestCase):
    """Node-limited routing invariants are validated at config-build time
    (TokenChoiceTopKRouter.Config.__post_init__), not deferred to the forward pass.
    """

    def test_no_groups_skips_validation(self):
        # Without num_expert_groups there is nothing to validate; even otherwise
        # "odd" values must construct fine.
        _make(num_experts=8, top_k=2)
        # Direct construction (the path gpt_oss uses) must also early-return.
        TokenChoiceTopKRouter.Config(
            gate=Linear.Config(in_features=8, out_features=7, param_init={}),
            top_k=3,
            score_func=ScoreFunc.SOFTMAX,
        )

    def test_valid_group_configs_pass(self):
        # deepseek/gift 236B-style: 8 groups, 3 limited, 160 experts, top_k 6.
        _make(num_experts=160, top_k=6, num_expert_groups=8, num_limited_groups=3)
        # deepseek/gift 671B-style: 8 groups, 4 limited, 256 experts, top_k 8.
        _make(num_experts=256, top_k=8, num_expert_groups=8, num_limited_groups=4)

    def test_missing_num_limited_groups_raises(self):
        with self.assertRaisesRegex(ValueError, "num_limited_groups must be set"):
            _make(num_experts=160, top_k=6, num_expert_groups=8)

    def test_non_divisible_num_experts_raises(self):
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            _make(num_experts=160, top_k=6, num_expert_groups=7, num_limited_groups=3)

    def test_experts_per_group_too_small_raises(self):
        with self.assertRaisesRegex(ValueError, "experts_per_group"):
            _make(num_experts=8, top_k=2, num_expert_groups=8, num_limited_groups=4)

    def test_top_k_exceeds_surviving_experts_raises(self):
        # 8 limited-out-of-16 groups isn't possible; use a case where the surviving
        # experts (num_limited_groups * experts_per_group) are fewer than top_k.
        with self.assertRaisesRegex(ValueError, "fewer than top_k"):
            _make(num_experts=16, top_k=3, num_expert_groups=8, num_limited_groups=1)


if __name__ == "__main__":
    unittest.main()
