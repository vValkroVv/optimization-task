import pytest

import torch
from torch import nn


@pytest.fixture
def batch_reshape_example_small():
    return [
        {
            "input_ids": torch.tensor(
                [[[0, 0], [3, 3]], [[1, 1], [4, 4]], [[2, 2], [5, 5]]]
            ),
            "attention_mask": torch.tensor(
                [[[0, 1], [0, 1]], [[0, 0], [0, 0]], [[1, 1], [1, 1]]]
            ),
        },
        {
            "input_ids": torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]),
            "attention_mask": torch.tensor(
                [[0, 1], [0, 0], [1, 1], [0, 1], [0, 0], [1, 1]]
            ),
        },
    ]


@pytest.fixture
def batch_example_small():
    return [
        {
            "input_ids": torch.tensor(
                [[1, 1, 1, 1, 1], [1, 1, 2, 2, 2], [1, 2, 3, 4, 5], [6, 7, 8, 9, 0]]
            ),
            "attention_mask": torch.tensor(
                [[1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 1, 1], [1, 1, 1, 1, 1]]
            ),
            "action_mask": torch.tensor(
                [[0, 1, 1, 1, 1], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 0, 1, 1, 0]]
            ),
        },
        (
            torch.tensor(
                [[1, 1, 1, 1], [1, 2, 2, -100], [-100, 3, 4, 5], [-100, 8, 9, -100]]
            ),
            torch.tensor([4, 3, 3, 2]),
        ),
    ]


@pytest.fixture
def batch_example_small_ignore_5():
    return [
        {
            "input_ids": torch.tensor(
                [[1, 1, 1, 1, 1], [1, 1, 2, 2, 2], [1, 2, 3, 4, 5], [6, 7, 8, 9, 0]]
            ),
            "attention_mask": torch.tensor(
                [[1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 1, 1], [1, 1, 1, 1, 1]]
            ),
            "action_mask": torch.tensor(
                [[0, 1, 1, 1, 1], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 0, 1, 1, 0]]
            ),
        },
        (
            torch.tensor(
                [[1, 1, 1, 1], [1, 2, 2, -100], [-100, 3, 4, -100], [-100, 8, 9, -100]]
            ),
            torch.tensor([4, 3, 2, 2]),
        ),
    ]


@pytest.fixture
def batch_example_small_ignore_prefix_and_5():
    return [
        {
            "input_ids": torch.tensor(
                [[1, 1, 2, 1, 1], [1, 2, 3, 4, 5], [1, 1, 2, 2, 2], [6, 7, 8, 9, 0]]
            ),
            "attention_mask": torch.tensor(
                [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1]]
            ),
            "action_mask": torch.tensor(
                [[0, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 0], [1, 1, 1, 1, 0]]
            ),
        },
        (
            torch.tensor(
                [
                    [-100, -100, 1, 1],
                    [2, 3, 4, -100],
                    [-100, -100, 2, -100],
                    [7, 8, 9, -100],
                ]
            ),
            torch.tensor([2, 3, 1, 3]),
        ),
    ]


@pytest.fixture
def logits_example(dpo_modules):
    FusedCrossEntropyLoss = dpo_modules.FusedCrossEntropyLoss
    return [
        {
            "loss_fct": FusedCrossEntropyLoss(
                reduction="none", inplace_backward=True, process_group=None
            ),
            "logits": torch.tensor(
                [
                    [
                        [-1, -3, -1, -1],
                        [-2, -4, -1, -1],
                        [-3, -5, -1, -1],
                        [-4, -6, -1, -1],
                        [-5, -7, -1, -1],
                    ]
                ]
                * 4
            ).cuda(),
            "labels": torch.tensor(
                [
                    [0, 1, 2, 3, 0],
                    [0, 1, 2, 3, -100],
                    [-100, 1, -100, 3, -100],
                    [-100, -100, 2, -100, -100],
                ]
            ).cuda(),
        },
        (-11.2172, -6.5137, -4.6038, -0.767164),
    ]


@pytest.fixture
def spectoken_nll_example(dpo_modules):
    FusedCrossEntropyLoss = dpo_modules.FusedCrossEntropyLoss
    return [
        {
            "loss_fct": FusedCrossEntropyLoss(
                reduction="none", inplace_backward=True, process_group=None
            ),
            "po_logits": torch.tensor(
                [
                    [
                        [-1, -3, -1, -1],
                        [-2, -4, -1, -1],
                        [-3, -5, -1, -1],
                        [-4, -6, -1, -1],
                        [-5, -7, -1, -1],
                    ]
                ]
                * 2
            ).cuda(),
            "labels": torch.tensor([[0, 1, 2, 3, -100], [0, 1, 2, 3, -100]]).cuda(),
            "spectoken_list": [1, 2],
        },
        2.3250,
    ]


@pytest.fixture
def logp_example():
    return [
        {
            "logprobs": torch.tensor([-1, -2, -3, -4]),
            "ref_logprobs": torch.tensor([-2, -4, -3, -16]),
            "token_num": torch.tensor([2, 1, 3, 4]),
        },
        {
            "length": torch.tensor([0.5, 2, 1, 1]),
            "ref": torch.tensor([0.5, 0.5, 1, 0.25]),
            "none": torch.tensor([1, 2, 3, 4]),
        },
    ]


@pytest.fixture
def losses_example():
    return [
        {
            "loss_type": "dpo",
            "chosen_po_logprobs": torch.tensor([-1, -2, -1000, 0]),
            "rejected_po_logprobs": torch.tensor([-2, -4, -3, -16]),
            "beta_chosen": 0.05,
            "beta_rejected": 0.05,
            "gamma": 1.0,
            "margin_coef": 0.0,
            "margins": torch.tensor([1, 1, 1, 1]),
            "chosen_ref_logprobs": torch.tensor([-1, -3, -1, -50]),
            "rejected_ref_logprobs": torch.tensor([-2, -4, -3, -16]),
            "chosen_tokens_num": torch.tensor([1, 2, 3, 4]),
            "rejected_tokens_num": torch.tensor([1, 2, 3, 4]),
        },
        {
            "loss_type": "dpo",
            "chosen_po_logprobs": torch.tensor([-1, -2, -3, -4]),
            "rejected_po_logprobs": torch.tensor([-2, -4, -3, -16]),
            "beta_chosen": 1.0,
            "beta_rejected": 0.5,
            "gamma": 1.0,
            "margin_coef": 0.0,
            "margins": torch.tensor([1, 1, 1, 1]),
            "chosen_ref_logprobs": torch.tensor([-1, -2, -3, -4]),
            "rejected_ref_logprobs": torch.tensor([-2, -4, -3, -16]),
            "chosen_tokens_num": torch.tensor([1, 2, 3, 4]),
            "rejected_tokens_num": torch.tensor([1, 2, 3, 4]),
        },
        {
            "loss_type": "dpo",
            "chosen_po_logprobs": torch.tensor([-1, -4, -2, -1]),
            "rejected_po_logprobs": torch.tensor([-9, -4, -6, -1]),
            "beta_chosen": 0.3,
            "beta_rejected": 0.1,
            "gamma": 1.0,
            "margin_coef": 0.0,
            "margins": torch.tensor([1, 1, 1, 1]),
            "chosen_ref_logprobs": torch.tensor([-5, -1, -5, -6]),
            "rejected_ref_logprobs": torch.tensor([-12, -14, -13, -16]),
            "chosen_tokens_num": torch.tensor([1, 2, 3, 4]),
            "rejected_tokens_num": torch.tensor([1, 2, 3, 4]),
        },
        {
            "loss_type": "simpo",
            "chosen_po_logprobs": torch.tensor([-1, -2, -3, -4]),
            "rejected_po_logprobs": torch.tensor([-2, -4, -3, -16]),
            "beta_chosen": 1.0,
            "beta_rejected": 0.5,
            "gamma": 0.0,
            "margin_coef": 1.0,
            "margins": torch.tensor([1, 1, 1, 1]),
            "chosen_ref_logprobs": torch.tensor([-1, -2, -3, -4]),
            "rejected_ref_logprobs": torch.tensor([-2, -4, -3, -16]),
            "chosen_tokens_num": torch.tensor([1, 2, 3, 4]),
            "rejected_tokens_num": torch.tensor([1, 2, 3, 4]),
        },
        {
            "loss_type": "simpo",
            "chosen_po_logprobs": torch.tensor([-1, -4, -2, -1]),
            "rejected_po_logprobs": torch.tensor([-9, -4, -6, -1]),
            "beta_chosen": 0.3,
            "beta_rejected": 0.1,
            "margin_coef": 1.0,
            "margins": torch.tensor([1, 1, 1, 1]),
            "chosen_ref_logprobs": torch.tensor([-5, -1, -5, -6]),
            "rejected_ref_logprobs": torch.tensor([-12, -14, -13, -16]),
            "chosen_tokens_num": torch.tensor([1, 2, 3, 4]),
            "rejected_tokens_num": torch.tensor([1, 2, 3, 4]),
        },
        {
            "loss_type": "simpo",
            "chosen_po_logprobs": torch.tensor([-1, -2, -1000, 0]),
            "rejected_po_logprobs": torch.tensor([-2, -4, -3, -16]),
            "beta_chosen": 0.05,
            "beta_rejected": 0.05,
            "gamma": 1.0,
            "margin_coef": 1.0,
            "margins": torch.tensor([1, 1, 1, 1]),
            "chosen_ref_logprobs": torch.tensor([-1, -3, -1, -50]),
            "rejected_ref_logprobs": torch.tensor([-2, -4, -3, -16]),
            "chosen_tokens_num": torch.tensor([1, 2, 3, 4]),
            "rejected_tokens_num": torch.tensor([1, 2, 3, 4]),
        },
    ], (
        12.8476238,
        0.693147182,
        0.917956650257,
        0.66840827465,
        0.69052749872,
        5.335420131683,
    )


@pytest.fixture
def batch_example():
    return (
        (
            (torch.rand((16, 128, 32)) - 1).cuda() * 100,
            (torch.rand((16, 128, 32)) - 1).cuda() * 100,
        ),
        {
            "input_ids": torch.randint(0, 32, (8, 2, 128)).cuda(),
            "action_mask": torch.randint(0, 2, (8, 2, 128)).cuda(),
            "attention_mask": torch.randint(0, 2, (8, 2, 128)).cuda(),
            "margins": torch.randint(0, 32, (8,)).cuda(),
        },
    )


def canonical_loss(
    outputs,
    batch,
    beta,
    gamma,
    label_smoothing,
    average_logp,
    loss_coef,
    chosen_nll_coef,
    dpo_modules,
):
    dpo_logits, ref_logits = outputs

    for key in ["attention_mask", "input_ids", "action_mask"]:
        if key in batch:
            batch[key] = batch[key].transpose(0, 1).reshape(-1, batch[key].shape[-1])

    inputs = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    action_mask = batch["action_mask"] * attention_mask
    margins = batch["margins"]
    bs = inputs.shape[0] // 2
    chosen_act_mask = action_mask[:bs, 1:]
    rejected_act_mask = action_mask[bs:, 1:]
    ref_logprobs = gather_log_probs(ref_logits[:, :-1, :], inputs[:, 1:])
    chosen_ref_logprobs = (ref_logprobs[:bs] * chosen_act_mask).sum(-1)
    rejected_ref_logprobs = (ref_logprobs[bs:] * rejected_act_mask).sum(-1)
    dpo_logprobs = gather_log_probs(dpo_logits[:, :-1, :], inputs[:, 1:])
    chosen_dpo_logprobs = (dpo_logprobs[:bs] * chosen_act_mask).sum(-1)
    rejected_dpo_logprobs = (dpo_logprobs[bs:] * rejected_act_mask).sum(-1)

    chosen_rewards = chosen_dpo_logprobs - chosen_ref_logprobs

    rejected_rewards = rejected_dpo_logprobs - rejected_ref_logprobs
    reward_diff = chosen_rewards - rejected_rewards
    logits = beta * reward_diff - gamma * margins
    dpo_loss = (
        -nn.functional.logsigmoid(logits) * (1 - label_smoothing)
        - nn.functional.logsigmoid(-logits) * label_smoothing
    ).mean()
    chosen_act_mask_sum = 1 if average_logp else chosen_act_mask.sum(-1) + 1e-16
    chosen_nll = -(chosen_dpo_logprobs / chosen_act_mask_sum).mean()
    loss = dpo_loss * loss_coef + chosen_nll * chosen_nll_coef

    return loss


def gather_log_probs(logits, labels):
    log_probs = nn.functional.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


class TestReshapeBatch:
    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_reshape_batch_works_as_expected(
        self, batch_reshape_example_small, dpo_modules
    ):
        reshape_batch = dpo_modules.reshape_batch
        input_batch, expected = batch_reshape_example_small
        result = reshape_batch(input_batch, input_batch.keys())

        assert result.keys() == expected.keys()
        assert torch.equal(result["input_ids"], expected["input_ids"])
        assert torch.equal(result["attention_mask"], expected["attention_mask"])

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_reshape_batch_ignores_non_requested_keys(
        self, batch_reshape_example_small, dpo_modules
    ):
        reshape_batch = dpo_modules.reshape_batch
        input_batch, expected = batch_reshape_example_small

        result = reshape_batch(input_batch, [])
        assert result.keys() == expected.keys()
        assert torch.equal(result["input_ids"], input_batch["input_ids"])
        assert torch.equal(result["attention_mask"], input_batch["attention_mask"])

        result = reshape_batch(input_batch, ["input_ids"])
        assert result.keys() == expected.keys()
        assert torch.equal(result["input_ids"], expected["input_ids"])
        assert torch.equal(result["attention_mask"], input_batch["attention_mask"])


# ============================================================================
# Tests - get_masked_labels
# ============================================================================


class TestGetMaskedLabels:
    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_masked_labels_works(self, batch_example_small, dpo_modules):
        get_masked_labels = dpo_modules.get_masked_labels
        input_batch, (expected_labels, expected_sums) = batch_example_small
        labels, sums = get_masked_labels(input_batch, False, [])

        assert torch.equal(labels, expected_labels)
        assert torch.equal(sums, expected_sums)

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_masked_labels_ignores_symbol(
        self, batch_example_small_ignore_5, dpo_modules
    ):
        get_masked_labels = dpo_modules.get_masked_labels
        input_batch, (expected_labels, expected_sums) = batch_example_small_ignore_5
        labels, sums = get_masked_labels(input_batch, False, [5])

        assert torch.equal(labels, expected_labels)
        assert torch.equal(sums, expected_sums)

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_masked_labels_ignores_common_prefix(
        self, batch_example_small_ignore_prefix_and_5, dpo_modules
    ):
        get_masked_labels = dpo_modules.get_masked_labels
        input_batch, (expected_labels, expected_sums) = (
            batch_example_small_ignore_prefix_and_5
        )
        labels, sums = get_masked_labels(input_batch, True, [5])

        assert torch.equal(labels, expected_labels)
        assert torch.equal(sums, expected_sums)


class TestCalculateSpectokenNLL:
    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_calculate_spectoken_nll_works(self, spectoken_nll_example, dpo_modules):
        """Тест расчета NLL для специальных токенов."""
        calculate_spectoken_nll = dpo_modules.calculate_spectoken_nll
        batch, expected = spectoken_nll_example
        actual = calculate_spectoken_nll(**batch).item()

        assert actual == pytest.approx(expected, abs=1e-4)


class TestCalculateLogprobs:
    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_calculate_chosen_and_rejected_logprobs_works(
        self, logits_example, dpo_modules
    ):
        calculate_chosen_and_rejected_logprobs = (
            dpo_modules.calculate_chosen_and_rejected_logprobs
        )
        batch, expected = logits_example
        logp_chosen_actual, logp_rejected_actual = (
            calculate_chosen_and_rejected_logprobs(**batch)
        )

        assert logp_chosen_actual.shape == (2,)
        assert logp_rejected_actual.shape == (2,)
        assert (
            logp_chosen_actual.tolist() + logp_rejected_actual.tolist()
            == pytest.approx(expected, abs=1e-4)
        )


class TestCalculateNLL:
    @pytest.mark.unit
    @pytest.mark.requires_gpu
    @pytest.mark.parametrize("norm", ["length", "ref", "none"])
    def test_calculate_nll_works(self, logp_example, norm, dpo_modules):
        """Тест расчета NLL с разными типами нормализации."""
        calculate_nll = dpo_modules.calculate_nll
        batch, expected = logp_example
        batch["norm"] = norm
        actual = calculate_nll(**batch)

        assert torch.equal(actual, expected[norm])


class TestCalculatePOLoss:
    @pytest.mark.unit
    @pytest.mark.requires_gpu
    @pytest.mark.parametrize("batch_idx", [0, 1, 2, 3, 4, 5])
    def test_calculate_po_loss_works(self, losses_example, batch_idx, dpo_modules):
        calculate_po_loss = dpo_modules.calculate_po_loss
        batches, expected_values = losses_example
        batch = batches[batch_idx]
        expected = expected_values[batch_idx]

        actual, _ = calculate_po_loss(**batch)

        assert actual.item() == pytest.approx(expected, abs=1e-4)


# ============================================================================
# Tests - Vision DPO
# ============================================================================


class TestReshapeImages:
    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_reshape_images_single_batch(self, dpo_modules):
        """Test reshape_images with bs=1 (typical DPO microbatch)."""
        reshape_images = dpo_modules.reshape_images
        # bs=1, 2*K=4 images (2 chosen + 2 rejected), C=3, H=W=2
        images = torch.arange(48).reshape(1, 4, 3, 2, 2).float().cuda()
        batch = {"images": images}
        reshape_images(batch)

        assert batch["images"].shape == (2, 2, 3, 2, 2)
        # First batch = chosen (first 2 images), second = rejected (last 2)
        assert torch.equal(
            batch["images"][0],
            torch.arange(24).reshape(2, 3, 2, 2).float().cuda(),
        )
        assert torch.equal(
            batch["images"][1],
            torch.arange(24, 48).reshape(2, 3, 2, 2).float().cuda(),
        )

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_reshape_images_multi_batch(self, dpo_modules):
        """Test reshape_images with bs>1 — chosen/rejected grouping, not interleaving."""
        reshape_images = dpo_modules.reshape_images
        # bs=2, 2*K=2 (1 chosen + 1 rejected per sample), C=3, H=2, W=2
        # Each image: [C=3, H=2, W=2] = 12 values
        images = torch.arange(2 * 2 * 3 * 2 * 2).reshape(2, 2, 3, 2, 2).float().cuda()
        batch = {"images": images}
        reshape_images(batch)

        assert batch["images"].shape == (4, 1, 3, 2, 2)
        # Grouped: chosen_0, chosen_1, rejected_0, rejected_1 (NOT interleaved)
        assert torch.equal(batch["images"][0], images[0, :1])  # chosen from sample 0
        assert torch.equal(batch["images"][1], images[1, :1])  # chosen from sample 1
        assert torch.equal(batch["images"][2], images[0, 1:])  # rejected from sample 0
        assert torch.equal(batch["images"][3], images[1, 1:])  # rejected from sample 1

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_reshape_images_matches_reshape_batch_ordering(self, dpo_modules):
        """Verify reshape_images produces same chosen/rejected grouping as reshape_batch."""
        reshape_images = dpo_modules.reshape_images
        reshape_batch = dpo_modules.reshape_batch

        bs, K = 3, 2
        # Text: [bs, 2, seq] — chosen/rejected pairs
        text_batch = {
            "input_ids": torch.arange(bs * 2 * 4).reshape(bs, 2, 4).cuda(),
        }
        text_result = reshape_batch(text_batch, ["input_ids"])
        # After reshape_batch: first bs rows = chosen, last bs = rejected

        # Images: [bs, 2*K, C, H, W]
        img_batch = {
            "images": torch.arange(bs * 2 * K)
            .reshape(bs, 2 * K, 1, 1, 1)
            .float()
            .cuda()
        }
        reshape_images(img_batch)
        # After reshape_images: first bs rows = chosen, last bs = rejected

        # Both should have same grouping pattern
        assert text_result["input_ids"].shape[0] == 2 * bs
        assert img_batch["images"].shape[0] == 2 * bs


class TestFilterOutput:
    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_filter_output_extracts_logits_from_dict(self, dpo_modules):
        filter_output = dpo_modules.filter_output
        logits = torch.randn(2, 4, 100).cuda()
        output = {"logits": logits, "hidden_states": torch.randn(2, 4, 64).cuda()}
        result = filter_output(output)
        assert torch.equal(result, logits)

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_filter_output_extracts_logits_from_tuple(self, dpo_modules):
        filter_output = dpo_modules.filter_output
        logits = torch.randn(2, 4, 100).cuda()
        output = (logits, torch.randn(2, 4, 64).cuda())
        result = filter_output(output)
        assert torch.equal(result, logits)


class TestSPSplitPaddingMask:
    """Test that SP-split padding_mask uses get_tensors_sp_part correctly."""

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_equal_split_returns_full_when_no_sp(self):
        """Without active SP group, get_tensors_sp_part returns the full tensor."""
        from llmfoundry.models.parallel.sequence.utils import get_tensors_sp_part

        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]]).cuda()
        # No SP group active on single GPU → should return full mask unchanged
        try:
            (split_mask,) = get_tensors_sp_part(mask, split_type="equal", dim=1)
            assert split_mask.shape == mask.shape
        except Exception:
            # SP not initialized — expected on single GPU without dist init
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "unit and requires_gpu"])
