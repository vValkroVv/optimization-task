# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from composer.metrics.classification import (
    BinaryAUCROC,
    ClassificationMSE,
    MultiClassCrossEntropy,
)


@pytest.mark.parametrize('labels', [
    torch.tensor([1.0, 0.0]),
    torch.tensor([0.0, 1.0]),
])
def test_binary_aucroc_tied_scores(labels):
    # Two samples with identical scores must yield AUC 0.5 regardless of the
    # order of the tied predictions. The hand-rolled implementation returned
    # 1.0 / 0.0 depending on order; delegating to torchmetrics (which groups
    # equal scores into one ROC threshold) fixes this.
    logits = torch.zeros(2, 2)
    metric = BinaryAUCROC()

    metric.update(logits, labels)

    torch.testing.assert_close(metric.compute(), torch.tensor(0.5))


def test_binary_aucroc_empty_state():
    # The old hand-rolled compute() crashed with
    # "RuntimeError: torch.cat(): expected a non-empty list of Tensors" when no
    # observations were seen. Subclassing torchmetrics BinaryAUROC removes that
    # crashing code path: compute() no longer raises. torchmetrics returns 0.0
    # (not NaN) for an empty state, so we assert it does not raise.
    metric = BinaryAUCROC()
    result = metric.compute()
    assert torch.is_tensor(result)


def test_cross_entropy_weights_samples_not_batches():
    # Accumulating reduction='sum' loss and dividing by total *samples* must give
    # the same result as a single cross_entropy over the concatenated batch,
    # even when batch sizes differ (1 sample then 2 samples).
    metric = MultiClassCrossEntropy()

    metric.update(
        torch.tensor([[2.0, 0.0]]),
        torch.tensor([[1.0, 0.0]]),
    )
    metric.update(
        torch.tensor([[0.0, 2.0], [0.0, 2.0]]),
        torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
    )

    expected = torch.nn.functional.cross_entropy(
        torch.tensor([[2.0, 0.0], [0.0, 2.0], [0.0, 2.0]]),
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
    )
    torch.testing.assert_close(metric.compute(), expected)


def test_cross_entropy_class_index_targets():
    # A 1-D *integer* target is a class-index tensor (standard Composer
    # multiclass hard label) and must route to F.cross_entropy, matching the
    # equivalent one-hot target.
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])

    index_metric = MultiClassCrossEntropy()
    index_metric.update(logits, torch.tensor([0, 1]))

    onehot_metric = MultiClassCrossEntropy()
    onehot_metric.update(logits, torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

    torch.testing.assert_close(index_metric.compute(), onehot_metric.compute())


def test_classification_mse_binary_hard_labels():
    # Brier score on the positive-class probability. For a tied [0.5, 0.5]
    # prediction, each of the two softmax columns contributes (0.5 - y)**2, so
    # the mean squared error over B*C elements is 0.25.
    logits = torch.zeros(2, 2)
    metric = ClassificationMSE()

    metric.update(logits, torch.tensor([1.0, 0.0]))

    torch.testing.assert_close(metric.compute(), torch.tensor(0.25))


def test_classification_mse_binary_soft_label():
    # A scalar float target is expanded to [p_neg, p_pos] before the MSE, so a
    # soft label is handled just like the one-hot / distribution form.
    logits = torch.tensor([[2.0, 0.0]])

    soft_metric = ClassificationMSE()
    soft_metric.update(logits, torch.tensor([0.3]))

    dist_metric = ClassificationMSE()
    dist_metric.update(logits, torch.tensor([[0.7, 0.3]]))

    torch.testing.assert_close(soft_metric.compute(), dist_metric.compute())


def test_classification_mse_multiclass_soft_distribution():
    # Multiclass Brier score: softmax over all classes, element-wise MSE against
    # the [B, num_classes] target distribution.
    logits = torch.tensor([[1.0, 2.0, 0.5]])
    target = torch.tensor([[0.1, 0.7, 0.2]])
    metric = ClassificationMSE()

    metric.update(logits, target)

    probs = torch.softmax(logits, dim=-1)
    expected = ((probs - target) ** 2).mean()
    torch.testing.assert_close(metric.compute(), expected)


def test_classification_mse_perfect_prediction_is_zero():
    # Confident correct prediction -> MSE ~ 0.
    logits = torch.tensor([[100.0, -100.0]])
    metric = ClassificationMSE()

    metric.update(logits, torch.tensor([0.0]))

    torch.testing.assert_close(
        metric.compute(),
        torch.tensor(0.0),
        atol=1e-6,
        rtol=0,
    )
