import logging
import re
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Union

import datasets
from transformers import AutoTokenizer


if TYPE_CHECKING:
    import transformers


eval_logger = logging.getLogger(__name__)

DEFAULT_SEQ_LENGTHS = [
    4096,
]
DEFAULT_SHUFFLE_SEED = 42


def _resolve_data_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(__file__).parent / path


def _format_seq_length(seq_length):
    if seq_length % 1024 == 0:
        return f"{seq_length // 1024}k"
    return str(seq_length)


def _prepared_data_paths(
    data_file=None,
    data_files=None,
    data_dir=None,
    dataset_id=None,
    max_seq_lengths=None,
):
    """Resolve one or more prepared JSONL files for a RULER subtask.

    Per-length files written by ``scripts/preprocess_ruler.py`` are preferred
    when ``max_seq_lengths`` is provided; a single ``{dataset_id}.jsonl`` is
    used as a fallback to preserve the previous single-file behaviour.
    """
    if data_file is not None:
        path = _resolve_data_path(data_file)
        return [path] if path.exists() else []
    if data_files is not None and dataset_id is not None and dataset_id in data_files:
        path = _resolve_data_path(data_files[dataset_id])
        return [path] if path.exists() else []
    if data_dir is None or dataset_id is None:
        return []

    data_dir = _resolve_data_path(data_dir)
    paths = []
    for length in max_seq_lengths or []:
        candidate = data_dir / f"{dataset_id}_{_format_seq_length(length)}.jsonl"
        if candidate.exists():
            paths.append(candidate)
    if paths:
        return paths

    for suffix in (".jsonl", ".json"):
        candidate = data_dir / f"{dataset_id}{suffix}"
        if candidate.exists():
            return [candidate]
    return []


def make_ruler_dataset(
    rows,
    data_file=None,
    data_files=None,
    data_dir=None,
    dataset_id=None,
    seed=DEFAULT_SHUFFLE_SEED,
    shuffle=True,
    max_seq_lengths=None,
    max_samples=None,
):
    if seed is None:
        seed = DEFAULT_SHUFFLE_SEED

    paths = _prepared_data_paths(
        data_file=data_file,
        data_files=data_files,
        data_dir=data_dir,
        dataset_id=dataset_id,
        max_seq_lengths=max_seq_lengths,
    )
    if paths:
        if max_samples is None:
            dataset = datasets.load_dataset(
                "json",
                data_files={"test": [str(path) for path in paths]},
                split="test",
            )
        else:
            datasets_per_length = []
            for path in paths:
                dataset = datasets.load_dataset(
                    "json",
                    data_files={"test": str(path)},
                    split="test",
                )
                if shuffle:
                    dataset = dataset.shuffle(seed=seed)
                limit = min(max_samples, len(dataset))
                datasets_per_length.append(dataset.select(range(limit)))
            dataset = datasets.concatenate_datasets(datasets_per_length)
    else:
        dataset = datasets.Dataset.from_list(list(rows), split=datasets.Split.TEST)

    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    return {"test": dataset}


@cache
def get_tokenizer(
    tokenizer=None, pretrained=None, **kwargs
) -> Union["transformers.PreTrainedTokenizer", "transformers.PreTrainedTokenizerFast"]:
    pretrained = tokenizer or pretrained
    assert pretrained, "No tokenizer or pretrained provided."
    eval_logger.info(f"Using tokenizer {pretrained} for synthetic tasks.")
    return AutoTokenizer.from_pretrained(
        pretrained,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )


def postprocess_pred(prediction: list[str]) -> list[str]:
    res = []
    for predict_str in prediction:
        predict_str = predict_str.strip()

        # Remove all non-printable characters
        np_pattern = re.compile(r"[\x00-\x1f]")
        predict_str = np_pattern.sub("\n", predict_str).strip()
        res.append(predict_str)

    return res


def string_match_all(preds: list[str], refs: list[list[str]]) -> float:
    score = sum(
        [
            sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref)
            for pred, ref in zip(preds, refs)
        ]
    ) / len(preds)
    return score


def string_match_part(preds: list[str], refs: list[list[str]]) -> float:
    score = max(
        [
            sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref)
            for pred, ref in zip(preds, refs)
        ]
    ) / len(preds)
    return score


def process_results(doc: dict, results: list[str]) -> dict[str, float]:
    input_len = doc["max_length"]
    pred = postprocess_pred(results)
    score = string_match_all(pred, [doc["outputs"]])
    return {str(input_len): score}


def process_results_part(doc: dict, results: list[str]) -> dict[str, float]:
    input_len = doc["max_length"]
    pred = postprocess_pred(results)
    score = string_match_part(pred, [doc["outputs"]])
    return {str(input_len): score}


def aggregate_metrics(metrics: list[float]) -> float:
    res = [x for x in metrics if x != -1]
    if not res:
        # we don't have any samples with this length
        return -1
    return sum(res) / len(res)
