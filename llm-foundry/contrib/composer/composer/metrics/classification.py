# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Soft-label classification torchmetrics (custom GigaChat addition).

These metrics power the GigaChat sequence-classifier (``gigaclf``) eval loop.
They are kept in composer (rather than in the llm-foundry classifier package)
so the whole metric family lives alongside the other composer metrics and can
be reused by any classification model.

All metrics accept **raw logits** and **soft or hard targets**:

    preds  : Tensor[batch_size, num_classes]  -- raw, pre-softmax scores
    target : Tensor[batch_size]               -- scalar soft label in [0, 1]
             OR Tensor[batch_size, num_classes] -- soft / one-hot distribution

Binary metrics (``Binary*``) obtain the predicted and reference class by
applying a decision ``threshold`` (default 0.5) to the positive-class
probability, so they work seamlessly with soft/scalar labels::

    pred  = (softmax(logits)[:, 1] >= threshold)
    y_bin = (target_pos_prob      >= threshold)

Multiclass metrics (``MultiClass*``) obtain classes by argmax, which is
well-defined for both hard one-hot and soft target distributions::

    pred = argmax(logits,  dim=-1)
    true = argmax(target,  dim=-1)

``BinaryAUCROC`` is threshold-agnostic: it feeds raw positive-class
probabilities to torchmetrics' :class:`~torchmetrics.classification.BinaryAUROC`,
which integrates the ROC curve at ``compute()`` time.

Note on naming: ``SoftBinaryF1Score`` is intentionally *not* named
``BinaryF1Score`` to avoid a collision with the hard-label, sklearn-style
``composer.metrics.nlp.BinaryF1Score``. The two are distinct metrics with
different target semantics.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import MeanSquaredError, Metric
from torchmetrics.classification import BinaryAUROC

log = logging.getLogger(__name__)

__all__ = [
    'BinaryAccuracy',
    'BinaryPrecision',
    'BinaryRecall',
    'SoftBinaryF1Score',
    'BinaryAUCROC',
    'MultiClassCrossEntropy',
    'MultiClassOverallAccuracy',
    'MultiClassWeightedF1',
    'ClassificationMSE',
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_inputs(
    logits: Tensor,
    target: Tensor,
    threshold: float = 0.5,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``(preds, y_bin, pos_probs)`` for binary metrics.

    Args:
        logits (Tensor): ``[B, num_classes]`` raw logits.
        target (Tensor): ``[B]`` soft labels OR ``[B, num_classes]`` soft
            distributions.
        threshold (float): decision threshold applied to the positive-class
            probability to obtain both the predicted and reference binary
            labels. Default: ``0.5``.

    Returns:
        preds (Tensor): LongTensor ``[B]`` -- predicted class (0 or 1).
        y_bin (Tensor): LongTensor ``[B]`` -- binarised ground-truth (0 or 1).
        pos_probs (Tensor): FloatTensor ``[B]`` -- positive-class probability.
    """
    probs = torch.softmax(logits.float(), dim=-1)  # [B, num_classes]
    pos_probs = probs[:, 1]                         # positive-class prob

    preds = (pos_probs >= threshold).long()

    if target.dim() == 2:
        # [B, num_classes] soft distribution -- take p_pos column
        y_soft = target[:, 1].float()
    else:
        y_soft = target.float()

    y_bin = (y_soft >= threshold).long()

    return preds, y_bin, pos_probs


def _get_preds_and_true(
    logits: Tensor,
    target: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return ``(preds, true)`` class indices for multiclass metrics.

    Handles both:
        - ``target`` shape ``[B, num_classes]``: argmax is taken directly.
        - ``target`` shape ``[B]``: scalar soft label (binary compat, only
          valid for ``num_classes == 2``); rounded to 0 or 1.

    Args:
        logits (Tensor): ``[B, num_classes]`` raw logits.
        target (Tensor): ``[B, num_classes]`` soft/one-hot distributions OR
            ``[B]`` scalars.

    Returns:
        preds (Tensor): LongTensor ``[B]`` -- predicted class index.
        true (Tensor): LongTensor ``[B]`` -- ground-truth class index.
    """
    preds = logits.argmax(dim=-1)

    if target.dim() == 1:
        # binary compat: scalar in [0, 1] -> round to 0 or 1
        true = target.round().long()
    else:
        true = target.argmax(dim=-1)

    return preds, true


# ---------------------------------------------------------------------------
# Binary metrics
# ---------------------------------------------------------------------------

class BinaryAccuracy(Metric):
    """Fraction of samples whose predicted class matches the binarised label.

    Adds metric state variables:
        correct (Tensor): Number of correctly classified samples.
        total (Tensor): Total number of samples seen.

    Args:
        threshold (float, optional): Decision threshold on the positive-class
            probability. Default: ``0.5``.
        dist_sync_on_step (bool, optional): Synchronize metric state across
            processes at each forward() before returning the value at the step.
            Default: ``False``.
    """

    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(self, threshold: float = 0.5, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.threshold = threshold
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update the state with new predictions and targets."""
        pred_labels, y_bin, _ = _parse_inputs(preds, target, self.threshold)
        self.correct += (pred_labels == y_bin).sum()
        self.total += y_bin.numel()

    def compute(self) -> Tensor:
        """Aggregate state over all processes and compute the metric."""
        assert isinstance(self.correct, Tensor)
        assert isinstance(self.total, Tensor)
        return self.correct.float() / self.total.clamp(min=1)


class BinaryPrecision(Metric):
    """Precision = TP / (TP + FP) for the positive class.

    Adds metric state variables:
        true_positive (Tensor): Samples correctly predicted as positive.
        false_positive (Tensor): Samples incorrectly predicted as positive.

    Args:
        threshold (float, optional): Decision threshold. Default: ``0.5``.
        dist_sync_on_step (bool, optional): Synchronize metric state across
            processes at each forward(). Default: ``False``.
    """

    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(self, threshold: float = 0.5, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.threshold = threshold
        self.add_state('true_positive', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('false_positive', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update the state with new predictions and targets."""
        pred_labels, y_bin, _ = _parse_inputs(preds, target, self.threshold)
        predicted_positive = pred_labels == 1
        self.true_positive += (predicted_positive & (y_bin == 1)).sum()
        self.false_positive += (predicted_positive & (y_bin == 0)).sum()

    def compute(self) -> Tensor:
        """Aggregate state over all processes and compute the metric."""
        assert isinstance(self.true_positive, Tensor)
        assert isinstance(self.false_positive, Tensor)
        denom = self.true_positive + self.false_positive
        if denom == 0:
            return torch.tensor(0.0, device=denom.device)
        return self.true_positive.float() / denom.float()


class BinaryRecall(Metric):
    """Recall = TP / (TP + FN) for the positive class.

    Adds metric state variables:
        true_positive (Tensor): Samples correctly predicted as positive.
        false_negative (Tensor): Positive samples incorrectly predicted negative.

    Args:
        threshold (float, optional): Decision threshold. Default: ``0.5``.
        dist_sync_on_step (bool, optional): Synchronize metric state across
            processes at each forward(). Default: ``False``.
    """

    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(self, threshold: float = 0.5, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.threshold = threshold
        self.add_state('true_positive', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('false_negative', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update the state with new predictions and targets."""
        pred_labels, y_bin, _ = _parse_inputs(preds, target, self.threshold)
        actual_positive = y_bin == 1
        self.true_positive += (actual_positive & (pred_labels == 1)).sum()
        self.false_negative += (actual_positive & (pred_labels == 0)).sum()

    def compute(self) -> Tensor:
        """Aggregate state over all processes and compute the metric."""
        assert isinstance(self.true_positive, Tensor)
        assert isinstance(self.false_negative, Tensor)
        denom = self.true_positive + self.false_negative
        if denom == 0:
            return torch.tensor(0.0, device=denom.device)
        return self.true_positive.float() / denom.float()


class SoftBinaryF1Score(Metric):
    """F1 = 2*TP / (2*TP + FP + FN) -- harmonic mean of precision and recall.

    Named ``SoftBinaryF1Score`` (not ``BinaryF1Score``) to avoid a name
    collision with :class:`composer.metrics.nlp.BinaryF1Score`; unlike that
    hard-label metric this one accepts soft/scalar targets and applies a
    decision threshold (see the module docstring).

    Adds metric state variables:
        true_positive (Tensor): TP count.
        false_positive (Tensor): FP count.
        false_negative (Tensor): FN count.

    Args:
        threshold (float, optional): Decision threshold. Default: ``0.5``.
        dist_sync_on_step (bool, optional): Synchronize metric state across
            processes at each forward(). Default: ``False``.
    """

    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(self, threshold: float = 0.5, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.threshold = threshold
        self.add_state('true_positive', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('false_positive', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('false_negative', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update the state with new predictions and targets."""
        pred_labels, y_bin, _ = _parse_inputs(preds, target, self.threshold)
        predicted_positive = pred_labels == 1
        actual_positive = y_bin == 1
        self.true_positive += (predicted_positive & actual_positive).sum()
        self.false_positive += (predicted_positive & ~actual_positive).sum()
        self.false_negative += (~predicted_positive & actual_positive).sum()

    def compute(self) -> Tensor:
        """Aggregate state over all processes and compute the metric."""
        assert isinstance(self.true_positive, Tensor)
        assert isinstance(self.false_positive, Tensor)
        assert isinstance(self.false_negative, Tensor)
        denom = 2 * self.true_positive + self.false_positive + self.false_negative
        if denom == 0:
            return torch.tensor(0.0, device=denom.device)
        return (2 * self.true_positive).float() / denom.float()


class BinaryAUCROC(BinaryAUROC):
    """Area Under the ROC Curve for binary classification.

    Thin wrapper over :class:`torchmetrics.classification.BinaryAUROC` that
    adapts our ``(logits, soft/scalar target)`` inputs to the ``(scores,
    binary target)`` interface torchmetrics expects. Delegating to torchmetrics
    gives a correct, rank-based AUROC that handles tied scores properly (equal
    scores are treated as a single ROC threshold) and a maintained distributed
    aggregation, instead of a hand-rolled trapezoidal integration that made the
    result depend on the arbitrary order of tied predictions.

    The metric is threshold-agnostic: it consumes raw positive-class
    probabilities. The ground-truth labels are binarised at ``gt_threshold``
    (default 0.5) so the metric works with soft/scalar labels.

    Args:
        gt_threshold (float, optional): Threshold used to binarise soft
            ground-truth labels. Default: ``0.5``.
        dist_sync_on_step (bool, optional): Synchronize metric state across
            processes at each forward(). Default: ``False``.
    """

    def __init__(self, gt_threshold: float = 0.5, dist_sync_on_step: bool = False):
        # thresholds=None keeps the exact (non-binned) rank-based AUROC.
        super().__init__(thresholds=None, dist_sync_on_step=dist_sync_on_step)
        self.gt_threshold = gt_threshold

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Feed positive-class probabilities and binarised labels to torchmetrics."""
        _, y_bin, pos_probs = _parse_inputs(preds, target, self.gt_threshold)
        # Keep tensors on their original device (GPU). torchmetrics accumulates
        # them as list states and concatenates via all_gather; moving to CPU
        # here would force a CPU collective (gloo), which is absent in
        # NCCL-only environments.
        super().update(pos_probs.detach(), y_bin.detach())


# ---------------------------------------------------------------------------
# Multiclass metrics
# ---------------------------------------------------------------------------

class MultiClassCrossEntropy(Metric):
    """Per-sample mean cross-entropy between logits and soft/hard targets.

    This is the custom classification counterpart to
    :class:`composer.metrics.metrics.CrossEntropy`. Whereas that metric
    averages the loss per *batch* (using ``soft_cross_entropy``), this one
    accumulates a ``reduction='sum'`` loss and divides by the number of
    *samples*, so uneven batch sizes are weighted correctly. It also expands a
    scalar binary target into a ``[p_neg, p_pos]`` distribution, matching the
    training loss used by the GigaChat classifier head.

    Works with:
        - hard one-hot targets    ``[B, num_classes]`` float
        - soft label targets      ``[B, num_classes]`` float
        - class-index targets     ``[B]`` integer  (standard Composer multiclass
          hard labels; routed to ``F.cross_entropy``)
        - scalar binary targets   ``[B]`` float  (GigaChat binary soft label,
          converted to ``[p_neg, p_pos]``)

    Adds metric state variables:
        total_loss (Tensor): Accumulated sum of per-sample CE losses.
        total_samples (Tensor): Total number of samples seen.

    Args:
        dist_sync_on_step (bool, optional): Synchronize metric state across
            processes at each forward(). Default: ``False``.
    """

    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state('total_loss', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total_samples', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update the state with new predictions and targets."""
        target = target.to(preds.device)

        # An integer 1-D target is a class-index tensor (standard Composer
        # multiclass hard label). Dtype is what distinguishes it from the
        # GigaChat binary *soft* label below, which is also 1-D but float.
        if target.dim() == preds.dim() - 1 and not torch.is_floating_point(target):
            loss = F.cross_entropy(preds, target.long(), reduction='sum')
        else:
            target = target.float()
            if target.dim() == 1:
                # Binary soft label: scalar p_pos in [0, 1] -> [p_neg, p_pos].
                target = torch.stack([1 - target, target], dim=1)
            # A 2-D target is already a soft / one-hot [B, num_classes] dist.
            loss = F.cross_entropy(preds, target, reduction='sum')

        self.total_loss += loss
        self.total_samples += preds.shape[0]

    def compute(self) -> Tensor:
        """Aggregate state over all processes and compute the metric."""
        assert isinstance(self.total_loss, Tensor)
        assert isinstance(self.total_samples, Tensor)
        return self.total_loss / self.total_samples.clamp(min=1)


class MultiClassOverallAccuracy(Metric):
    """Top-1 accuracy: fraction of samples where argmax(logits) == argmax(target).

    Adds metric state variables:
        correct (Tensor): Number of correctly classified samples.
        total (Tensor): Total number of samples seen.

    Args:
        dist_sync_on_step (bool, optional): Synchronize metric state across
            processes at each forward(). Default: ``False``.
    """

    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update the state with new predictions and targets."""
        pred_labels, true = _get_preds_and_true(preds, target)
        self.correct += (pred_labels == true).sum()
        self.total += preds.shape[0]

    def compute(self) -> Tensor:
        """Aggregate state over all processes and compute the metric."""
        assert isinstance(self.correct, Tensor)
        assert isinstance(self.total, Tensor)
        return self.correct.float() / self.total.clamp(min=1)


class MultiClassPerClassAccuracy(Metric):
    """Per-class recall: for each class c, fraction of true-c samples predicted c.

    Warning:
        Unlike every other metric in composer, ``compute()`` returns a
        **vector** of shape ``(num_classes,)`` rather than a scalar. Composer's
        trainer logs ``compute()`` output as-is (``trainer.py`` ->
        ``_compute_and_log_metrics``) and does NOT split it into per-class
        keys. ``format_log_data_value`` renders a non-scalar tensor as the
        string ``"Tensor of shape ..."``, so the console/file loggers will not
        show the per-class values (WandB logs it as an array; scalar-only
        loggers such as MLflow may reject it). If you need per-class values
        surfaced individually, register one scalar metric per class instead of
        using this vector metric.

    Adds metric state variables:
        correct (Tensor): ``[num_classes]`` per-class correct prediction counts.
        total (Tensor): ``[num_classes]`` per-class true sample counts.

    Args:
        num_classes (int): Number of output classes. Must match the second
            dimension of ``logits``.
        dist_sync_on_step (bool, optional): Synchronize metric state across
            processes at each forward(). Default: ``False``.
    """

    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(self, num_classes: int, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_classes = num_classes
        self.add_state('correct', default=torch.zeros(num_classes), dist_reduce_fx='sum')
        self.add_state('total', default=torch.zeros(num_classes), dist_reduce_fx='sum')

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update the state with new predictions and targets."""
        pred_labels, true = _get_preds_and_true(preds, target)
        for c in range(self.num_classes):
            mask = true == c
            self.correct[c] += (pred_labels[mask] == c).sum()
            self.total[c] += mask.sum()

    def compute(self) -> Tensor:
        """Aggregate state over all processes and compute the metric.

        Returns a ``(num_classes,)`` tensor; classes with zero true samples
        return 0.0 (zero-safe division).
        """
        assert isinstance(self.correct, Tensor)
        assert isinstance(self.total, Tensor)
        return self.correct / self.total.clamp(min=1)


class MultiClassWeightedF1(Metric):
    """Weighted F1 score across all classes.

    Each class F1 is weighted by its frequency in the true labels so that
    rare classes contribute less than frequent ones -- consistent with
    sklearn's ``f1_score(..., average='weighted')``::

        F1_c = 2 * TP_c / (2 * TP_c + FP_c + FN_c)
        WeightedF1 = sum_c( weight_c * F1_c )
        weight_c = (TP_c + FN_c) / total_true_samples

    Adds metric state variables:
        tp (Tensor): ``[num_classes]`` true positive counts per class.
        fp (Tensor): ``[num_classes]`` false positive counts per class.
        fn (Tensor): ``[num_classes]`` false negative counts per class.

    Args:
        num_classes (int): Number of output classes.
        dist_sync_on_step (bool, optional): Synchronize metric state across
            processes at each forward(). Default: ``False``.
    """

    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(self, num_classes: int, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_classes = num_classes
        self.add_state('tp', default=torch.zeros(num_classes), dist_reduce_fx='sum')
        self.add_state('fp', default=torch.zeros(num_classes), dist_reduce_fx='sum')
        self.add_state('fn', default=torch.zeros(num_classes), dist_reduce_fx='sum')

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update the state with new predictions and targets."""
        pred_labels, true = _get_preds_and_true(preds, target)
        for c in range(self.num_classes):
            pred_c = pred_labels == c
            true_c = true == c
            self.tp[c] += (pred_c & true_c).sum()
            self.fp[c] += (pred_c & ~true_c).sum()
            self.fn[c] += (~pred_c & true_c).sum()

    def compute(self) -> Tensor:
        """Aggregate state over all processes and compute the metric."""
        assert isinstance(self.tp, Tensor)
        assert isinstance(self.fp, Tensor)
        assert isinstance(self.fn, Tensor)
        precision = self.tp / (self.tp + self.fp).clamp(min=1)
        recall = self.tp / (self.tp + self.fn).clamp(min=1)
        f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)

        # weight by true class frequency
        support = self.tp + self.fn
        total = support.sum().clamp(min=1)
        weights = support / total

        return (f1 * weights).sum()


class ClassificationMSE(MeanSquaredError):
    """Brier-style mean squared error between predicted and target probabilities.

    Softmaxes the logits over all classes and computes the element-wise MSE
    against the target probability distribution -- i.e. the (multiclass) Brier
    score. This is well-defined for hard and soft labels alike and handles both
    supported target shapes uniformly:

        - scalar binary target ``[B]`` (float in ``[0, 1]``): expanded to a
          ``[p_neg, p_pos]`` distribution, matching ``MultiClassCrossEntropy``.
        - soft / one-hot distribution ``[B, num_classes]``: used as-is.

    For the binary case this is the standard Brier score on the positive-class
    probability (both softmax columns contribute the identical ``(p - y)**2``).
    Lower is better; the value is in ``[0, 2]`` for softmax outputs.

    Delegates all state handling and distributed reduction to torchmetrics'
    :class:`~torchmetrics.MeanSquaredError`.
    """

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update the state with new predictions and targets."""
        probs = torch.softmax(preds.float(), dim=-1)  # [B, num_classes]

        target = target.to(preds.device).float()
        if target.dim() == 1:
            # Binary soft label: scalar p_pos in [0, 1] -> [p_neg, p_pos].
            target = torch.stack([1 - target, target], dim=1)

        super().update(probs, target)
