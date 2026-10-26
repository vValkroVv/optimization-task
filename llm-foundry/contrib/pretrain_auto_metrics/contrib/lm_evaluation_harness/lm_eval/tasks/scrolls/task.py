import os
import re
import string
from abc import abstractmethod
from collections import Counter
from functools import reduce
from pathlib import Path

import numpy as np
import transformers.data.metrics.squad_metrics as squad_metrics
from datasets import Dataset
from transformers import AutoTokenizer

from lm_eval.api.instance import Instance
from lm_eval.api.metrics import mean
from lm_eval.api.task import ConfigurableTask


_CITATION = """
@inproceedings{shaham-etal-2022-scrolls,
    title = "{SCROLLS}: Standardized {C}ompa{R}ison Over Long Language Sequences",
    author = "Shaham, Uri  and
      Segal, Elad  and
      Ivgi, Maor  and
      Efrat, Avia  and
      Yoran, Ori  and
      Haviv, Adi  and
      Gupta, Ankit  and
      Xiong, Wenhan  and
      Geva, Mor  and
      Berant, Jonathan  and
      Levy, Omer",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.823",
    pages = "12007--12021"
}
"""

# SCROLLS is formualted as a sequence-to-sequence task.
# To allow for evaluation of causal models, we'll
# reformualte these with appropriate prompts


def _format_context_length(context_length):
    if context_length % 1024 == 0:
        return f"{context_length // 1024}k"
    return str(context_length)


_PRECOMPUTED_PREFIXES = {
    "contract_nli": "scrolls_contractnli",
    "gov_report": "scrolls_govreport",
    "narrative_qa": "scrolls_narrativeqa",
    "qasper": "scrolls_qasper",
    "qmsum": "scrolls_qmsum",
    "quality": "scrolls_quality",
    "summ_screen_fd": "scrolls_summscreenfd",
}


def _normalize_answer(text):
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _metric_max_over_references(metric_fn, prediction, references):
    return max(metric_fn(prediction, reference) for reference in references)


def _exact_match_score(prediction, reference):
    return _normalize_answer(prediction) == _normalize_answer(reference)


def _compute_exact_match(predictions, references):
    exact_match = sum(
        _metric_max_over_references(_exact_match_score, prediction, reference_list)
        for prediction, reference_list in zip(predictions, references)
    )
    return exact_match / len(predictions)


def _f1_score(prediction, reference):
    prediction_tokens = _normalize_answer(prediction).split()
    reference_tokens = _normalize_answer(reference).split()
    common = Counter(prediction_tokens) & Counter(reference_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(reference_tokens)
    return (2 * precision * recall) / (precision + recall)


def _compute_f1(predictions, references):
    f1 = sum(
        _metric_max_over_references(_f1_score, prediction, reference_list)
        for prediction, reference_list in zip(predictions, references)
    )
    return f1 / len(predictions)


def _rouge_postprocess_text(text):
    import nltk

    return "\n".join(nltk.sent_tokenize(text))


def _compute_rouge(predictions, references):
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(
        rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"],
        use_stemmer=False,
    )
    scores = {key: [] for key in ("rouge1", "rouge2", "rougeL", "rougeLsum")}
    for prediction, reference_list in zip(predictions, references):
        prediction = _rouge_postprocess_text(prediction)
        reference_scores = []
        for reference in reference_list:
            reference_scores.append(
                scorer.score(_rouge_postprocess_text(reference), prediction)
            )
        for key in scores:
            scores[key].append(max(score[key].fmeasure for score in reference_scores))

    metrics = {f"rouge/{key}": sum(value) / len(value) for key, value in scores.items()}
    metrics["rouge/geometric_mean"] = (
        metrics["rouge/rouge1"] * metrics["rouge/rouge2"] * metrics["rouge/rougeL"]
    ) ** (1.0 / 3.0)
    return {key: round(value, 4) for key, value in metrics.items()}


def _substring_match_enabled(metadata=None):
    if os.environ.get("LM_EVAL_SUBSTRING_MATCH", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return True
    value = (metadata or {}).get("substring_match")
    return (
        value.lower() in ("1", "true", "yes", "on")
        if isinstance(value, str)
        else bool(value)
    )


def _substring_match_score(prediction, reference):
    normalized_reference = _normalize_answer(reference)
    return (
        1.0
        if normalized_reference
        and normalized_reference in _normalize_answer(prediction)
        else 0.0
    )


def _compute_substring_match(predictions, references):
    substring_match = sum(
        _metric_max_over_references(_substring_match_score, prediction, reference_list)
        for prediction, reference_list in zip(predictions, references)
    )
    return substring_match / len(predictions)


def _f1_ps(prediction, reference_list):
    return round(_metric_max_over_references(_f1_score, prediction, reference_list), 4)


def _substring_ps(prediction, reference_list):
    return round(
        _metric_max_over_references(_substring_match_score, prediction, reference_list),
        4,
    )


_ROUGE_PS_SCORER = None


def _rouge_ps(prediction, reference_list):
    """Per-sample max-over-references ROUGE f-measures (rouge1/rouge2/rougeL)."""
    global _ROUGE_PS_SCORER
    if _ROUGE_PS_SCORER is None:
        from rouge_score import rouge_scorer

        _ROUGE_PS_SCORER = rouge_scorer.RougeScorer(
            rouge_types=["rouge1", "rouge2", "rougeL"], use_stemmer=False
        )
    prediction = _rouge_postprocess_text(prediction)
    best = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for reference in reference_list:
        score = _ROUGE_PS_SCORER.score(_rouge_postprocess_text(reference), prediction)
        for key in best:
            best[key] = max(best[key], score[key].fmeasure)
    return {key: round(value, 4) for key, value in best.items()}


def _compute_scrolls_metric(metric_name, predictions, references):
    if metric_name == "exact_match":
        return {"exact_match": round(_compute_exact_match(predictions, references), 4)}
    if metric_name == "f1":
        return {"f1": round(_compute_f1(predictions, references), 4)}
    if metric_name == "substring_match":
        return {
            "substring_match": round(
                _compute_substring_match(predictions, references), 4
            )
        }
    if metric_name.startswith("rouge/"):
        return _compute_rouge(predictions, references)
    raise KeyError(f"Unsupported SCROLLS metric: {metric_name}")


def _process_doc_prepended_question(doc):
    # "When a query is given in addition to the raw text (as
    # in QMSum, Qasper, NarrativeQA, QuALITY, and ContractNLI),
    # we prepend it to the text, using two newlines as a natural separator"
    input = doc["input"]
    split = input.find("\n\n")
    return {
        "id": doc["id"],
        "pid": doc["pid"],
        "input": input,
        "outputs": doc["outputs"],
        "question": input[0:split],
        "text": input[split + 2 :],
    }


def _drop_duplicates_in_input(untokenized_dataset):
    # from scrolls/evaluator/dataset_evaluator.py

    indices_to_keep = []
    id_to_idx = {}
    outputs = []
    for i, (id_, output) in enumerate(
        zip(untokenized_dataset["id"], untokenized_dataset["output"])
    ):
        if id_ in id_to_idx:
            outputs[id_to_idx[id_]].append(output)
            continue
        indices_to_keep.append(i)
        id_to_idx[id_] = len(outputs)
        outputs.append([output])
    untokenized_dataset = untokenized_dataset.select(indices_to_keep).flatten_indices()
    untokenized_dataset = untokenized_dataset.remove_columns("output")
    untokenized_dataset = untokenized_dataset.add_column("outputs", outputs)
    return untokenized_dataset


def _num_cpu_cores():
    # https://stackoverflow.com/questions/1006289/how-to-find-out-the-number-of-cpus-using-python/55423170#55423170
    try:
        import psutil

        return psutil.cpu_count(logical=False)
    except ImportError:
        import os

        return len(os.sched_getaffinity(0))


class _SCROLLSTask(ConfigurableTask):
    VERSION = 2
    DATASET_PATH = "tau/scrolls"
    DATASET_NAME = None
    PRUNE_TOKENIZERS = None
    PRUNE_MAX_TOKENS = None
    PRUNE_NUM_PROC = None

    def __init__(self, config=None):
        user_metadata = dict((config or {}).get("metadata") or {})
        user_metadata.setdefault("version", self.VERSION)
        self._precomputed = False
        self._precomputed_lengths = None
        super().__init__(config={"metadata": user_metadata})

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._precomputed:
            return self.dataset["validation"]
        processed_docs = list(map(self._process_doc, self.dataset["train"]))

        # Flatten the list of lists since _process_doc returns a list of one element.
        processed_docs = [item for sublist in processed_docs for item in sublist]
        processed_dict = {
            key: [d[key] for d in processed_docs] for key in processed_docs[0]
        }

        return Dataset.from_dict(processed_dict)

    def validation_docs(self):
        if self._precomputed:
            return self.dataset["validation"]
        processed_docs = list(map(self._process_doc, self.dataset["validation"]))

        # Flatten the list of lists since _process_doc returns a list of one element.
        processed_docs = [item for sublist in processed_docs for item in sublist]
        processed_dict = {
            key: [d[key] for d in processed_docs] for key in processed_docs[0]
        }

        return Dataset.from_dict(processed_dict)

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["input"]

    def download(self, *args, **kwargs):
        if self._load_precomputed_dataset():
            return
        super().download(*args, **kwargs)
        del self.dataset["test"]
        for split in self.dataset:
            self.dataset[split] = _drop_duplicates_in_input(self.dataset[split])
        if self.PRUNE_TOKENIZERS is not None:
            self.prune()

    def _load_precomputed_dataset(self):
        metadata = self.config.metadata or {}
        data_dir = metadata.get("data_dir")
        if data_dir is None:
            return False

        prefix = metadata.get("precomputed_prefix") or _PRECOMPUTED_PREFIXES.get(
            self.DATASET_NAME,
            f"scrolls_{self.DATASET_NAME}",
        )
        lengths = metadata.get("max_seq_lengths") or []
        max_samples_per_length = metadata.get("max_samples_per_length")
        if max_samples_per_length is None:
            max_samples_per_length = metadata.get("num_samples")
        seed = metadata.get("seed", 42)

        search_dirs = [Path(data_dir)]
        if not Path(data_dir).is_absolute():
            search_dirs.append(Path(__file__).parent / data_dir)

        paths = []
        for length in lengths:
            filename = f"{prefix}_{_format_context_length(length)}.jsonl"
            for directory in search_dirs:
                candidate = directory / filename
                if candidate.exists():
                    paths.append((length, candidate))
                    break

        if not paths:
            return False

        import datasets

        per_length = []
        for length, path in paths:
            dataset = datasets.load_dataset(
                "json",
                data_files={"validation": str(path)},
                split="validation",
            )
            if "scrolls_max_length" not in dataset.column_names:
                dataset = dataset.add_column(
                    "scrolls_max_length",
                    [length] * len(dataset),
                )
            if max_samples_per_length is not None:
                dataset = dataset.shuffle(seed=seed)
                dataset = dataset.select(
                    range(min(int(max_samples_per_length), len(dataset)))
                )
            per_length.append(dataset)

        combined = (
            per_length[0]
            if len(per_length) == 1
            else datasets.interleave_datasets(
                per_length,
                stopping_strategy="all_exhausted",
            )
        )
        self.dataset = {"validation": combined}
        self._precomputed = True
        self._precomputed_lengths = [length for length, _ in paths]
        return True

    def _configured_lengths(self):
        if self._precomputed_lengths is not None:
            return self._precomputed_lengths
        return []

    def _length_metric_map(self, metrics):
        lengths = self._configured_lengths()
        if not lengths:
            return metrics
        return {
            (f"{metric}_{length}" if metric else str(length)): value
            for length in lengths
            for metric, value in metrics.items()
        }

    def _with_length_metrics(self, doc, metrics):
        length = doc.get("scrolls_max_length")
        if length is None:
            return metrics
        return {
            (f"{metric}_{int(length)}" if metric else str(int(length))): value
            for metric, value in metrics.items()
        }

    def _get_prune_text(self, sample):
        return self.doc_to_text(self._process_doc(sample)[0])

    def prune(self):
        """Create a pruned version of a SCROLLS task dataset containing only inputs
        that are less than `max_tokens` when tokenized by each tokenizer
        """

        tokenizers = [
            AutoTokenizer.from_pretrained(tokenizer)
            for tokenizer in self.PRUNE_TOKENIZERS
        ]
        cache = {}

        def _filter(sample):
            text = self._get_prune_text(sample)
            cached = cache.get(text, None)
            if cached is None:
                for tokenizer in tokenizers:
                    if len(tokenizer(text).input_ids) > self.PRUNE_MAX_TOKENS:
                        cache[text] = False
                        return False
                cache[text] = True
                return True
            else:
                return cached

        self.dataset = self.dataset.filter(_filter, num_proc=self.PRUNE_NUM_PROC)

    def doc_to_target(self, doc):
        return " " + ", ".join(doc["outputs"])

    def doc_to_text(self, doc):
        return f"{doc['text']}\n\nQuestion: {doc['question']}\nAnswer:"

    def higher_is_better(self):
        metrics = {x: True for x in self._scrolls_metrics().keys()}
        return self._length_metric_map(metrics)

    @abstractmethod
    def _scrolls_metrics(self):
        pass

    def _make_compute_metrics(self, value):
        # Scores are computed per-sample in process_results; the corpus metric is
        # just their mean (equals the old corpus-level f1/rouge computation).
        def compute_metrics(samples):
            return sum(samples) / len(samples) if samples else 0.0

        return compute_metrics

    def aggregation(self):
        metrics = {
            key: self._make_compute_metrics(value)
            for key, value in self._scrolls_metrics().items()
        }
        return self._length_metric_map(metrics)


class _SCROLLSMultipleChoiceTask(_SCROLLSTask):
    def __post_init__(self):
        self.metric = None

    def _scrolls_metrics(self):
        return None

    def aggregation(self):
        return self._length_metric_map(
            {"": mean, "em": mean, "acc": mean, "acc_norm": mean}
        )

    def higher_is_better(self):
        return self._length_metric_map(
            {"": True, "em": True, "acc": True, "acc_norm": True}
        )

    def process_results(self, doc, results):
        gold = doc["gold"]

        lls, _ = zip(*results)
        acc = 1.0 if np.argmax(lls) == gold else 0.0
        completion_len = np.array([float(len(i)) for i in doc["choices"]])
        acc_norm = 1.0 if np.argmax(lls / completion_len) == gold else 0.0

        return self._with_length_metrics(
            doc,
            {
                "": acc_norm,
                "acc": acc,
                "acc_norm": acc_norm,
                "em": acc_norm,
            },
        )

    def construct_requests(
        self, doc, ctx, chat_template=None, apply_chat_template=False, **kwargs
    ):
        request_list = [
            Instance(
                request_type="loglikelihood",
                doc=doc,
                arguments=(ctx, " {}".format(choice))
                if not apply_chat_template
                else (ctx, "{}".format(choice)),
                idx=i,
                **kwargs,
            )
            for i, choice in enumerate(doc["choices"])
        ]
        return request_list


class _SCROLLSSummaryTask(_SCROLLSTask):
    def _process_doc(self, doc):
        return [doc]

    def _scrolls_metrics(self):
        return {
            "rouge1": "rouge/rouge1",
            "rouge2": "rouge/rouge2",
            "rougeL": "rouge/rougeL",
        }

    def process_results(self, doc, results):
        scores = _rouge_ps(results[0], doc["outputs"])
        return self._with_length_metrics(
            doc,
            {
                "rouge1": scores["rouge1"],
                "rouge2": scores["rouge2"],
                "rougeL": scores["rougeL"],
            },
        )

    def construct_requests(
        self, doc, ctx, chat_template=None, apply_chat_template=False, **kwargs
    ):
        return Instance(
            request_type="generate_until",
            doc=doc,
            arguments=(ctx, {"until": ["\n"]}),
            idx=0,
            **kwargs,
        )

    def doc_to_text(self, doc):
        return f"{doc['input']}\n\nQuestion: What is a summary of the preceding text?\nAnswer:"


class Qasper(_SCROLLSTask):
    """A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers
    https://arxiv.org/abs/2105.03011
    """

    DATASET_NAME = "qasper"

    def _process_doc(self, doc):
        doc = _process_doc_prepended_question(doc)
        doc["is_yes_no"] = reduce(
            lambda prev, cur: prev
            and squad_metrics.normalize_answer(cur) in ["yes", "no"],
            doc["outputs"],
            True,
        )
        return [doc]

    def _scrolls_metrics(self):
        return {"f1": "f1", "": "f1"}

    def process_results(self, doc, results):
        if doc["is_yes_no"]:
            prediction = " yes" if results[0] > results[1] else " no"
        elif len(results[0].strip()) == 0:
            prediction = "Unanswerable"
        else:
            prediction = results[0]
        f1 = _f1_ps(prediction, doc["outputs"])
        return self._with_length_metrics(doc, {"f1": f1, "": f1})

    def construct_requests(
        self, doc, ctx, chat_template=None, apply_chat_template=False, **kwargs
    ):
        if doc["is_yes_no"]:
            return [
                Instance(
                    request_type="loglikelihood",
                    doc=doc,
                    arguments=(ctx, " yes")
                    if not apply_chat_template
                    else (ctx, "yes"),
                    idx=0,
                    **kwargs,
                ),
                Instance(
                    request_type="loglikelihood",
                    doc=doc,
                    arguments=(ctx, " no") if not apply_chat_template else (ctx, "no"),
                    idx=1,
                    **kwargs,
                ),
            ]
        else:
            return Instance(
                request_type="generate_until",
                doc=doc,
                arguments=(ctx, {"until": ["\n"]}),
                idx=0,
                **kwargs,
            )


class QuALITY(_SCROLLSMultipleChoiceTask):
    """QuALITY: Question Answering with Long Input Texts, Yes!
    https://arxiv.org/abs/2112.08608
    """

    DATASET_NAME = "quality"
    _multiple_choice_pattern = re.compile(r" *\([A-D]\) *")

    @staticmethod
    def _normalize_answer(text):
        return " ".join(text.split()).strip()

    def _process_doc(self, doc):
        doc = _process_doc_prepended_question(doc)

        split = doc["text"].find("\n\n", doc["text"].find("(D)"))
        choices_text = doc["text"][:split]

        doc["text"] = doc["text"][split:].strip()
        doc["choices"] = [
            QuALITY._normalize_answer(choice)
            for choice in re.split(QuALITY._multiple_choice_pattern, choices_text)[1:]
        ]
        doc["gold"] = doc["choices"].index(QuALITY._normalize_answer(doc["outputs"][0]))

        return [doc]


class NarrativeQA(_SCROLLSTask):
    """The NarrativeQA Reading Comprehension Challenge
    https://arxiv.org/abs/1712.07040
    """

    DATASET_NAME = "narrative_qa"

    def _process_doc(self, doc):
        return [_process_doc_prepended_question(doc)]

    def _scrolls_metrics(self):
        # bare length key (like helmet/ruler); metric is substring or f1 per the
        # flag, so we do not hard-code "f1" in the name.
        metric = (
            "substring_match"
            if _substring_match_enabled(self.config.metadata)
            else "f1"
        )
        return {"": metric}

    def _get_prune_text(self, doc):
        # pruning narrativeqa takes forever -- let's cheat a bit
        # and just cache on the text, not the question, since
        # the dataset is different questions about the same large
        # documents
        return self._process_doc(doc)[0]["text"]

    def process_results(self, doc, results):
        if _substring_match_enabled(self.config.metadata):
            score = _substring_ps(results[0], doc["outputs"])
        else:
            score = _f1_ps(results[0], doc["outputs"])
        return self._with_length_metrics(doc, {"": score})

    def construct_requests(
        self, doc, ctx, chat_template=None, apply_chat_template=False, **kwargs
    ):
        return Instance(
            request_type="generate_until",
            doc=doc,
            arguments=(ctx, {"until": ["\n"]}),
            idx=0,
            **kwargs,
        )


class ContractNLI(_SCROLLSMultipleChoiceTask):
    """ContractNLI: A Dataset for Document-level Natural Language Inference for Contracts
    https://arxiv.org/abs/1712.07040
    """

    DATASET_NAME = "contract_nli"
    CHOICES = ["Not mentioned", "Entailment", "Contradiction"]

    def _process_doc(self, doc):
        doc = _process_doc_prepended_question(doc)
        doc["choices"] = ContractNLI.CHOICES
        doc["gold"] = ContractNLI.CHOICES.index(doc["outputs"][0])
        return [doc]

    def doc_to_text(self, doc):
        return f"{doc['text']}\n\nHypothesis: {doc['question']}\nConclusion:"


class GovReport(_SCROLLSSummaryTask):
    """Efficient Attentions for Long Document Summarization
    https://arxiv.org/abs/2104.02112

    Note: The average length of the reference summaries is ~3,000
    characters, or ~600 tokens as tokenized by GPT-NeoX. For causal models,
    it is recommended to set `max_gen_toks` sufficiently large (e.g. 1024)
    to allow a full summary to be generated.
    """

    DATASET_NAME = "gov_report"


class SummScreenFD(_SCROLLSSummaryTask):
    """SummScreen: A Dataset for Abstractive Screenplay Summarization
    https://arxiv.org/abs/2104.07091
    """

    DATASET_NAME = "summ_screen_fd"


class QMSum(_SCROLLSSummaryTask):
    """QMSum: A New Benchmark for Query-based Multi-domain
    Meeting Summarization

    https://arxiv.org/abs/2104.05938
    """

    DATASET_NAME = "qmsum"

    def _process_doc(self, doc):
        return [_process_doc_prepended_question(doc)]

    def doc_to_text(self, doc):
        return f"{doc['text']}\n\nQuestion: {doc['question']}\nAnswer:"
