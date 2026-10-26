# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""A collection of common torchmetrics."""

from composer.metrics.classification import (BinaryAccuracy, BinaryAUCROC, BinaryPrecision, BinaryRecall,
                                             ClassificationMSE, MultiClassCrossEntropy, MultiClassOverallAccuracy,
                                             MultiClassWeightedF1, SoftBinaryF1Score)
from composer.metrics.map import MAP
from composer.metrics.metrics import CrossEntropy, Dice, LossMetric, MIoU
from composer.metrics.nlp import (BinaryF1Score, InContextLearningCodeEvalAccuracy, 
                                  InContextLearningLMAccuracy, InContextLearningLMCrossEntropy,
                                  InContextLearningLMExpectedCalibrationError,
                                  InContextLearningMCExpectedCalibrationError, InContextLearningMetric,
                                  InContextLearningMultipleChoiceAccuracy, InContextLearningQAAccuracy,
                                  LanguageCrossEntropy, LanguagePerplexity, MaskedAccuracy)

__all__ = [
    'MAP',
    'MIoU',
    'Dice',
    'CrossEntropy',
    'LossMetric',
    'BinaryF1Score',
    'LanguageCrossEntropy',
    'MaskedAccuracy',
    'LanguagePerplexity',
    'InContextLearningLMAccuracy',
    'InContextLearningLMCrossEntropy',
    'InContextLearningMultipleChoiceAccuracy',
    'InContextLearningQAAccuracy',
    'InContextLearningMCExpectedCalibrationError',
    'InContextLearningLMExpectedCalibrationError',
    'InContextLearningMetric',
    'InContextLearningCodeEvalAccuracy',
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

METRIC_DEFAULT_CTORS = {
    'InContextLearningLMAccuracy': InContextLearningLMAccuracy,
    'InContextLearningLMCrossEntropy': InContextLearningLMCrossEntropy,
    'InContextLearningMultipleChoiceAccuracy': InContextLearningMultipleChoiceAccuracy,
    'InContextLearningQAAccuracy': InContextLearningQAAccuracy,
    'InContextLearningCodeEvalAccuracy': InContextLearningCodeEvalAccuracy,
}
