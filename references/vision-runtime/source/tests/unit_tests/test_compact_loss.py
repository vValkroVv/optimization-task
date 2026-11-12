"""CPU compact LM-head parity; distributed Omni coverage runs in INF jobs."""

import copy
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtitan.components.loss import (
    CompactCELoss,
    GradAccumulator,
    IGNORE_INDEX,
)
from torchtitan.config import CompileConfig


class _ContextDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.context = nn.Linear(5, 5)
        self.norm = nn.LayerNorm(5)
        self.head = nn.Linear(5, 11)

    def forward(self, inputs):
        # Supervised predictions depend on earlier, unsupervised input slots.
        return self.norm(inputs + self.context(inputs).tanh().cumsum(dim=1))


class TestCompactCELoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)

    def _compare(
        self,
        labels,
        *,
        normalization="tokens",
        positions=None,
        packing_mode="none",
        chunks=4,
        compile_config=None,
    ):
        dense = _ContextDecoder()
        compact = copy.deepcopy(dense)
        source = torch.randn(*labels.shape, 5)
        dense_input = source.clone().requires_grad_()
        compact_input = source.clone().requires_grad_()
        loss_fn = CompactCELoss(
            CompactCELoss.Config(num_chunks=chunks, normalization=normalization),
            compile_config=compile_config,
        )
        loss_fn.set_lm_head(compact.head)
        original = labels.clone()
        counts = (
            (labels != IGNORE_INDEX).sum(dim=1)
            if normalization == "samples" and packing_mode == "none"
            else None
        )
        prepared = loss_fn.prepare_targets(
            labels, counts, positions=positions, packing_mode=packing_mode
        )
        torch.testing.assert_close(labels, original)
        count = loss_fn.normalization_count(labels, positions=positions)

        token_loss = F.cross_entropy(
            dense.head(dense(dense_input)).flatten(0, 1),
            labels.flatten(),
            reduction="none",
            ignore_index=IGNORE_INDEX,
        ).reshape_as(labels)
        if normalization == "tokens":
            expected_sum = token_loss.sum()
            expected_count = (labels != IGNORE_INDEX).sum()
        else:
            pieces = []
            for row in range(labels.shape[0]):
                boundaries = [0]
                if positions is not None:
                    boundaries += [
                        col
                        for col in range(1, labels.shape[1])
                        if positions[row, col] == 0
                    ]
                boundaries.append(labels.shape[1])
                for start, stop in zip(boundaries, boundaries[1:]):
                    valid = (labels[row, start:stop] != IGNORE_INDEX).sum()
                    if valid:
                        pieces.append(token_loss[row, start:stop].sum() / valid)
            # Preserve the reference graph even when the entire batch is ignored.
            expected_sum = sum(pieces, token_loss.sum() * 0)
            expected_count = torch.tensor(len(pieces))
        torch.testing.assert_close(count, expected_count)
        denominator = count.clamp_min(1)
        expected = expected_sum / denominator
        expected.backward()

        shapes = []
        hook = compact.head.register_forward_pre_hook(
            lambda module, args: shapes.append(tuple(args[0].shape))
        )
        actual = loss_fn(compact(compact_input), prepared, denominator)
        actual.backward()
        hook.remove()
        self.assertEqual(
            sum(shape[0] * shape[1] for shape in shapes),
            max(1, int((labels != IGNORE_INDEX).sum())),
        )
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(loss_fn.components()["loss_ce"], expected.detach())
        torch.testing.assert_close(compact_input.grad, dense_input.grad)
        for (name, param), (dense_name, dense_param) in zip(
            compact.named_parameters(), dense.named_parameters()
        ):
            self.assertEqual(name, dense_name)
            self.assertIsNotNone(param.grad, name)
            torch.testing.assert_close(param.grad, dense_param.grad, msg=name)
        return prepared, compact_input, compact

    def test_uneven_sparse_counts_preserve_loss_and_context_gradients(self):
        # Five supervised slots exercise an uneven final chunk; the first slot
        # is valid to catch accidentally applying the source model's label shift.
        labels = torch.tensor(
            [[1, -100, 2, 3, -100, -100], [-100, 4, -100, 5, -100, -100]]
        )
        for normalization in ("tokens", "samples"):
            for chunks in (2, 4, 8):
                with self.subTest(normalization=normalization, chunks=chunks):
                    prepared, inputs, _ = self._compare(
                        labels, normalization=normalization, chunks=chunks
                    )
                    self.assertEqual(prepared[0, 1].tolist(), [0, 2, 3, 7, 9])
                    self.assertGreater(int(inputs.grad[0, 1].count_nonzero()), 0)

    def test_packed_sample_mean_excludes_empty_padding_document(self):
        labels = torch.tensor([[2, -100, -100, 4, 6, -100, -100, -100]])
        positions = torch.tensor([[0, 1, 2, 0, 1, 2, 0, 1]], dtype=torch.int32)
        for packing_mode in ("offline", "online"):
            with self.subTest(packing_mode=packing_mode):
                prepared, _, _ = self._compare(
                    labels,
                    normalization="samples",
                    positions=positions,
                    packing_mode=packing_mode,
                )
                self.assertEqual(prepared[0, 2].tolist(), [1, 2, 2])

    def test_loss_compile_preserves_token_and_sample_parity(self):
        config = CompileConfig(enable=True, components=["loss"], backend="eager")
        labels = torch.tensor([[1, -100, 2, -100], [3, 4, 5, -100]])
        for normalization in ("tokens", "samples"):
            with self.subTest(normalization=normalization):
                self._compare(
                    labels, normalization=normalization, compile_config=config
                )

    def test_empty_rank_keeps_head_and_decoder_graph(self):
        labels = torch.full((2, 3), IGNORE_INDEX)
        for normalization in ("tokens", "samples"):
            with self.subTest(normalization=normalization):
                prepared, inputs, model = self._compare(
                    labels, normalization=normalization
                )
                self.assertEqual(prepared.tolist(), [[[-100], [0], [1]]])
                self.assertEqual(int(inputs.grad.count_nonzero()), 0)
                for parameter in model.parameters():
                    self.assertEqual(int(parameter.grad.count_nonzero()), 0)

    def test_accumulation_uses_global_denominator(self):
        for normalization in ("tokens", "samples"):
            with self.subTest(normalization=normalization):
                dense = nn.Linear(5, 11)
                head = copy.deepcopy(dense)
                loss_fn = CompactCELoss(
                    CompactCELoss.Config(num_chunks=4, normalization=normalization)
                )
                loss_fn.set_lm_head(head)
                label_batches = [
                    torch.tensor([[2, -100, 4, -100]]),
                    torch.tensor([[3, 4, 5, -100], [-100, -100, -100, -100]]),
                    torch.full((1, 4), IGNORE_INDEX),
                ]
                denominator = sum(loss_fn.normalization_count(t) for t in label_batches)
                dense_total = torch.tensor(0.0)
                compact_total = torch.tensor(0.0)
                for labels in label_batches:
                    source = torch.randn(*labels.shape, 5)
                    inputs = source.clone().requires_grad_()
                    compact_inputs = source.clone().requires_grad_()
                    per_token = F.cross_entropy(
                        dense(inputs).flatten(0, 1), labels.flatten(), reduction="none"
                    ).reshape_as(labels)
                    if normalization == "samples":
                        per_token = per_token / (
                            (labels != IGNORE_INDEX)
                            .sum(dim=1, keepdim=True)
                            .clamp_min(1)
                        )
                    expected = per_token.sum() / denominator
                    actual = loss_fn(
                        compact_inputs,
                        loss_fn.prepare_targets(
                            labels,
                            (labels != IGNORE_INDEX).sum(dim=1)
                            if normalization == "samples"
                            else None,
                        ),
                        denominator,
                    )
                    expected.backward()
                    actual.backward()
                    dense_total += expected.detach()
                    compact_total += actual.detach()
                    torch.testing.assert_close(compact_inputs.grad, inputs.grad)
                torch.testing.assert_close(compact_total, dense_total)
                torch.testing.assert_close(head.weight.grad, dense.weight.grad)
                torch.testing.assert_close(head.bias.grad, dense.bias.grad)

    def test_validation_and_dense_token_fallback(self):
        head = nn.Linear(5, 11)
        loss_fn = CompactCELoss(CompactCELoss.Config(num_chunks=2))
        loss_fn.set_lm_head(head)
        labels = torch.tensor([[1, -100, 4, -100, 2]])
        hidden = torch.randn(1, 5, 5)
        with torch.no_grad():
            prepared = loss_fn.prepare_targets(labels)
            expected = F.cross_entropy(
                head(hidden).flatten(0, 1), labels.flatten(), reduction="sum"
            )
            torch.testing.assert_close(loss_fn(hidden, prepared), expected)
            torch.testing.assert_close(loss_fn(hidden, labels), expected)
        self.assertIsNone(head.weight.grad)

    def test_rejects_invalid_metadata_without_silent_fallback(self):
        loss_fn = CompactCELoss(CompactCELoss.Config(normalization="samples"))
        labels = torch.ones(1, 4, dtype=torch.long)
        for bad in (labels.float(), labels.flatten(), labels.to("meta")):
            with self.subTest(shape=bad.shape, dtype=bad.dtype, device=bad.device):
                with self.assertRaises(ValueError):
                    loss_fn.prepare_targets(bad)
        with self.assertRaisesRegex(ValueError, "positions"):
            loss_fn.prepare_targets(
                labels, positions=torch.tensor([[0, 1, 3, 4]]), packing_mode="offline"
            )
        with self.assertRaisesRegex(ValueError, "prepared CPU targets"):
            loss_fn(torch.randn(1, 4, 5), labels)


class TestUnevenGradAccumulator(unittest.TestCase):
    def test_short_final_chunk_uses_cumulative_offset(self):
        reference = torch.arange(30).reshape(2, 5, 3).float()
        accumulator = GradAccumulator(reference, num_chunks=2, dtype=torch.float32)
        for chunk in reference.chunk(2, dim=1):
            accumulator.add(chunk)
        torch.testing.assert_close(accumulator.result(), reference)


if __name__ == "__main__":
    unittest.main()
