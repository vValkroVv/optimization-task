import sacrebleu

from lm_eval.api.metrics import _sacreformat


def chrf_norm(items):
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    refs, preds = _sacreformat(refs, preds)
    return sacrebleu.corpus_chrf(preds, refs).score / 100.0


def chrf_sent(references, predictions):
    return sacrebleu.sentence_chrf(predictions[0], [references[0]]).score / 100.0
