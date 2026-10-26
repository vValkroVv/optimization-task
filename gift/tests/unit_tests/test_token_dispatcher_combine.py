"""``combine()`` returns this rank's own tokens, indexed locally.

The earlier version built a buffer ``sp_size`` times larger and offset token
indices by ``T_local * sp_rank``. That offset assumes the local flat view is a
contiguous slice of the global one, which only holds at ``batch_size == 1``:
``x_BLD.view(T, D)`` is row-major over ``(b, l_local)``, so token ``(b=0, l=1)``
of a ``B=2, L=4, sp=2, sp_rank=1`` shard belongs at global row 3, not 5.

Reproducing that needs ``tp >= 2`` and ``ep >= 2`` at once -- four GPUs. These
tests pin the shape and indexing contract instead, on CPU.
"""

import unittest

import torch

from torchtitan.models.common.moe.dispatch import LocalTokenDispatcher


def _dispatcher() -> LocalTokenDispatcher:
    return LocalTokenDispatcher(LocalTokenDispatcher.Config())


class TestCombineIsLocal(unittest.TestCase):
    def test_output_matches_input_shape(self):
        """Never the global sequence: the caller declares the result sharded."""
        dispatcher = _dispatcher()
        x_TD = torch.zeros(6, 4)
        routed, counts, metadata = dispatcher.dispatch(
            x_TD,
            torch.ones(6, 1),
            torch.zeros(6, 1, dtype=torch.long),
            torch.tensor([6, 0]),
            score_before_experts=False,
        )
        out_TD = dispatcher.combine(torch.ones_like(routed), metadata, x_TD)
        self.assertEqual(out_TD.shape, x_TD.shape)

    def test_token_lands_on_its_own_row(self):
        """Each token's expert output returns to the row it came from."""
        dispatcher = _dispatcher()
        T, D = 6, 4
        x_TD = torch.zeros(T, D)
        expert_ids = torch.zeros(T, 1, dtype=torch.long)
        routed, counts, metadata = dispatcher.dispatch(
            x_TD,
            torch.ones(T, 1),
            expert_ids,
            torch.tensor([T, 0]),
            score_before_experts=False,
        )
        # A distinct value per token, laid out in expert-sorted order.
        marks = torch.arange(1, T + 1, dtype=torch.float32)
        routed_output = (
            marks[metadata.token_indices_experts_sorted_N].unsqueeze(-1).expand(-1, D)
        )
        out_TD = dispatcher.combine(routed_output.contiguous(), metadata, x_TD)
        torch.testing.assert_close(out_TD[:, 0], marks)

    def test_no_sp_coordinates_left(self):
        """The rank's place in the sequence is a placement, not dispatcher state."""
        from torchtitan.models.common.moe.dispatch import (
            AllToAllTokenDispatcher,
            BaseEPTokenDispatcher,
        )

        for cls in (
            LocalTokenDispatcher,
            BaseEPTokenDispatcher,
            AllToAllTokenDispatcher,
        ):
            with self.subTest(cls=cls.__name__):
                self.assertNotIn("sp_size", cls.__init__.__code__.co_names)
                self.assertNotIn("sp_rank", cls.__init__.__code__.co_names)


if __name__ == "__main__":
    unittest.main()
