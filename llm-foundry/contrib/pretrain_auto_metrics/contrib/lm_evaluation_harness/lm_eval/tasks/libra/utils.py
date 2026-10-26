import importlib
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List

import datasets
import pymorphy3

from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter


normalizer = pymorphy3.MorphAnalyzer()


@register_filter("libra_scorer")
class LibraScorer(Filter):
    def __init__(self, **kwargs):
        self.score_type = kwargs.pop("score_type")
        if self.score_type == "em":
            self.scoring_function = LibraScorer.exact_match_score
        elif self.score_type == "f1":
            self.scoring_function = LibraScorer.f1_score
        elif self.score_type == "count":
            self.scoring_function = LibraScorer.count_score
        else:
            raise ValueError(f"Unknown score type: {self.score_type}")

        super().__init__(**kwargs)

    @staticmethod
    def normalize_answer(sentence: str) -> str:
        """Normalize an input sentence by removing punctuation and converting words to their base (lemmatized) form.
        :param sentence: str
            Input sentence.
        :return: str
            A normalized sentence where:
            - All characters except letters, digits, and underscores are removed.
            - All words are converted to lowercase.
            - Words are lemmatized using `normalizer`.
        :raises ValueError:
            If `sentence` is not a string.
        :example:
        >>> normalize_answer("Hello, world! This is a test sentence.")
        'hello world this is a test sentence'
        """
        sentence = str(sentence)
        new_sentence = []
        for word in sentence.split():
            token = re.sub(r"[^a-zа-яй0-9_]+", "", word.lower())
            token = normalizer.parse(token)[0].normal_form.lower()
            new_sentence.append(token)
        return " ".join(new_sentence)

    @staticmethod
    def exact_match_score(prediction: str, ground_truth: str) -> float:
        result = 0.0
        if LibraScorer.normalize_answer(ground_truth) in LibraScorer.normalize_answer(
            prediction
        ):
            result = 1.0
        return result

    @staticmethod
    def f1_score(prediction: str, ground_truth: str) -> float:
        common = Counter(prediction) & Counter(ground_truth)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction)
        recall = 1.0 * num_same / len(ground_truth)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    @staticmethod
    def count_score(prediction: str, ground_truth: str) -> float:
        numbers = re.findall(r"\d+", prediction)
        right_num = 0
        for number in numbers:
            if str(number) == str(ground_truth):
                right_num += 1
        final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
        return float(final_score)

    def apply(self, results, docs):
        """Aggregates score by 'length' by scoring_function.

        :param reslts: List of predictions.
        :param docs: List of dictionaries containing 'positive_outputs', 'length' and other dataset metadata.
        :return: List of dictionaries with 'length' as keys and average score as values.

        :example:
        >>> results = [
        ...     {"pred_answer": "1", "answers": ["1", "one"], "length": "8p"},
        ...     {"pred_answer": "0", "answers": ["zero", "none"], "length": "8p"},
        ...     {"pred_answer": "one", "answers": ["1", "one"], "length": "16p"}
        ... ]
        >>> aggregate_results(results=results)
        {'8p': 0.5, '16p': 1.0}
        """

        scores = defaultdict(lambda: [0, 0])
        lengths = [doc["length"] for doc in docs]
        answers = [doc["positive_outputs"] for doc in docs]
        preds = [resp[0] for resp in results]
        for length, answers_set, pred in zip(lengths, answers, preds):
            pred = self.normalize_answer(pred)
            answers_set = set([self.normalize_answer(text) for text in answers_set])
            scores[length][1] += 1
            for answer in answers_set:
                metric = self.scoring_function(prediction=pred, ground_truth=answer)
                if metric > 0:
                    scores[length][0] += metric
                    break
        return [
            {
                key + "_" + self.score_type: correct / total
                for key, (correct, total) in scores.items()
            }
        ] * len(results)


def filter_dataset_by_page_lengths(*args, **kwargs) -> Dict[str, datasets.Dataset]:
    """Filter dataset by page lengths for Libra task.

    in CLI metadata --metadata '{"valid_pages": ["8p", "32p"], "dataset_repo_name": "ai-forever/LIBRA"}'
    """
    valid_pages = kwargs.get("valid_pages", [])

    dataset_repo_name = kwargs.get("dataset_repo_name", "ai-forever/LIBRA")
    dataset_name = kwargs.get("dataset_name", None)
    filter_colname = kwargs.get("filter_colname", "length")
    token = kwargs.get("token", None)

    dataset_columns = list(
        datasets.load_dataset(dataset_repo_name, dataset_name, token=token)[
            "test"
        ].features.keys()
    )
    if filter_colname not in dataset_columns:
        raise ValueError(f"Column {filter_colname} not found in dataset {dataset_name}")

    if valid_pages:
        dataset_filtered = datasets.load_dataset(
            dataset_repo_name, dataset_name, token=token
        )["test"].filter(lambda doc: doc.get(filter_colname) in valid_pages)
    else:
        dataset_filtered = datasets.load_dataset(
            dataset_repo_name, dataset_name, token=token
        )["test"]
    return {"test": dataset_filtered}


def process_results(doc: List, results: List[str]) -> Dict:
    """Evaluation results passthrough.

    :return: A dictionary where the key is the input page length ("length") and the value is a mean score for all samples
    """
    return results[0]
