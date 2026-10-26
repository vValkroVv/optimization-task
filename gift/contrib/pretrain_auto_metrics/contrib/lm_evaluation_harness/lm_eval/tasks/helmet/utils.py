# Adapted from https://github.com/princeton-nlp/HELMET (MIT License).

import hashlib
import json
import logging
import math
import os
import random
import re
import string
import tarfile
from collections import Counter
from functools import cache
from pathlib import Path

import datasets
import numpy as np
from rouge_score import rouge_scorer


HELMET_DATA_DIR = "HELMET_DATA_DIR"
HELMET_DATA_REPO = "princeton-nlp/HELMET"
HELMET_DATA_ARCHIVE = "data.tar.gz"
HELMET_DEFAULT_DATA_DIR = os.path.join(
    os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
    "helmet_data",
)
eval_logger = logging.getLogger(__name__)

# Standard HELMET context lengths (tokens). The same set RULER uses, extended to 8k.
HELMET_STANDARD_LENGTHS = [8192, 16384, 32768, 65536, 131072]

# Length -> controlling count for the file-based HELMET families.
_HELMET_KILT_K = {8192: 50, 16384: 105, 32768: 220, 65536: 440, 131072: 1000}
_HELMET_MSMARCO_K = {8192: 50, 16384: 130, 32768: 285, 65536: 600, 131072: 1000}
_HELMET_ALCE_DOCS = {8192: 30, 16384: 75, 32768: 165, 65536: 345, 131072: 700}
_HELMET_ICL_SHOTS = {
    "icl_trec_coarse": {8192: 400, 16384: 800, 32768: 1600, 65536: 3300, 131072: 6600},
    "icl_trec_fine": {8192: 400, 16384: 800, 32768: 1600, 65536: 3200, 131072: 6400},
    "icl_banking77": {8192: 360, 16384: 720, 32768: 1450, 65536: 2900, 131072: 5900},
    "icl_clinic150": {8192: 440, 16384: 880, 32768: 1750, 65536: 3525, 131072: 7050},
    "icl_nlu": {8192: 510, 16384: 1020, 32768: 2040, 65536: 4080, 131072: 8296},
}
# Tokenizer-budget families: the trailing budget baked into the dataset name is
# length minus a per-family offset (matches the pre-generated per-length yamls).
_HELMET_TOKEN_OFFSET = {
    "narrativeqa": 300,
    "multi_lexsum": 700,
    "infbench_qa_eng": 210,
    "infbench_choice_eng": 210,
    "infbench_sum_eng": 1400,
}
_KILT_DEP = {"kilt_hotpotqa": "dep3"}
_KILT_FILE = {
    "kilt_nq": "nq-dev-multikilt_1000_k{k}_{dep}.jsonl",
    "kilt_triviaqa": "triviaqa-dev-multikilt_1000_k{k}_{dep}.jsonl",
    "kilt_hotpotqa": "hotpotqa-dev-multikilt_1000_k{k}_{dep}.jsonl",
    "kilt_popqa_3": "popqa_test_1000_k{k}_dep6.jsonl",
}
_HELMET_JSON_KV_K = {8192: 105, 16384: 220, 32768: 440, 65536: 900, 131072: 1800}
_HELMET_RULER_NIAH = {
    "ruler_niah_mk_2": "niah_multikey_2",
    "ruler_niah_mk_3": "niah_multikey_3",
    "ruler_niah_mv": "niah_multivalue",
}

# Maps a HELMET dataset family (the ``dataset`` kwarg passed to the loader) to
# the filename prefix used by ``scripts/preprocess_helmet.py`` for the
# corresponding task. Allows loading precomputed per-length JSONL files.
_HELMET_TASK_PREFIX = {
    "alce_asqa": "helmet_cite_asqa",
    "alce_qampari": "helmet_cite_qampari",
    "icl_trec_coarse": "helmet_icl_trec_coarse",
    "icl_trec_fine": "helmet_icl_trec_fine",
    "icl_banking77": "helmet_icl_banking77",
    "icl_clinic150": "helmet_icl_clinic150",
    "icl_nlu": "helmet_icl_nlu",
    "narrativeqa": "helmet_longqa_narrativeqa",
    "infbench_qa_eng": "helmet_longqa_infbench_qa_eng",
    "infbench_choice_eng": "helmet_longqa_infbench_choice_eng",
    "infbench_sum_eng": "helmet_summ_infbench_sum_eng",
    "multi_lexsum": "helmet_summ_multi_lexsum",
    "kilt_nq": "helmet_rag_nq",
    "kilt_triviaqa": "helmet_rag_triviaqa",
    "kilt_hotpotqa": "helmet_rag_hotpotqa",
    "kilt_popqa_3": "helmet_rag_popqa",
    "msmarco_rerank_psg": "helmet_rerank_msmarco",
    "json_kv": "helmet_recall_json_kv",
    "ruler_niah_mk_2": "helmet_recall_niah_multikey_2",
    "ruler_niah_mk_3": "helmet_recall_niah_multikey_3",
    "ruler_niah_mv": "helmet_recall_niah_multivalue",
}


def _variant_for_length(family, length):
    """Resolve the (dataset, test_file) pair for a HELMET family at a length.

    Mirrors the pre-generated per-length task yamls: for file-based families the
    length selects a different data file, for tokenizer-budget families it selects
    a different trailing token budget in the dataset name.
    """
    if family in _HELMET_TOKEN_OFFSET:
        return f"{family}_{length - _HELMET_TOKEN_OFFSET[family]}", ""
    if family in _HELMET_ICL_SHOTS:
        return f"{family}_{_HELMET_ICL_SHOTS[family][length]}shot_balance", ""
    if family in ("alce_asqa", "alce_qampari"):
        return f"{family}_{_HELMET_ALCE_DOCS[length]}", ""
    if family == "msmarco_rerank_psg":
        return (
            family,
            f"data/msmarco/test_reranking_data_k{_HELMET_MSMARCO_K[length]}_dep3.jsonl",
        )
    if family in _KILT_FILE:
        dep = _KILT_DEP.get(family, "dep6")
        return (
            family,
            f"data/kilt/{_KILT_FILE[family].format(k=_HELMET_KILT_K[length], dep=dep)}",
        )
    if family == "json_kv":
        return family, f"data/json_kv/test_k{_HELMET_JSON_KV_K[length]}_dep6.jsonl"
    if family in _HELMET_RULER_NIAH:
        return (
            family,
            f"data/ruler/{_HELMET_RULER_NIAH[family]}/validation_{length}.jsonl",
        )
    raise ValueError(f"Unknown HELMET family for multi-length mode: {family}")


def _format_context_length(context_length):
    if context_length % 1024 == 0:
        return f"{context_length // 1024}k"
    return str(context_length)


def _resolve_precomputed_paths(data_dir, dataset, max_seq_lengths):
    """Find precomputed per-length JSONL files written by
    ``scripts/preprocess_helmet.py`` for the HELMET family ``dataset``.

    Returns an empty list unless ``data_dir`` contains the per-length files for
    every requested length, so a ``data_dir`` pointing at raw HELMET data falls
    through to the normal generation pipeline.
    """
    if data_dir is None:
        return []
    prefix = _HELMET_TASK_PREFIX.get(dataset)
    if prefix is None:
        return []

    lengths = max_seq_lengths or HELMET_STANDARD_LENGTHS
    paths = []
    for length in lengths:
        candidate = Path(data_dir) / f"{prefix}_{_format_context_length(length)}.jsonl"
        if candidate.exists():
            paths.append(candidate)
    return paths


def _load_precomputed_dataset(paths, max_test_samples=None, seed=42):
    per_length = []
    for path in paths:
        dataset = datasets.load_dataset(
            "json",
            data_files={"test": str(path)},
            split="test",
        )
        if max_test_samples is not None:
            dataset = dataset.shuffle(seed=seed)
            dataset = dataset.select(range(min(max_test_samples, len(dataset))))
        per_length.append(dataset)

    combined = (
        per_length[0]
        if len(per_length) == 1
        else datasets.concatenate_datasets(per_length)
    )
    return {"test": combined}


def load_helmet_dataset(
    dataset,
    test_file="",
    demo_file="",
    data_dir=None,
    max_test_samples=None,
    shots=0,
    seed=42,
    tokenizer=None,
    pretrained=None,
    **kwargs,
):
    truncation_tokenizer = tokenizer or pretrained
    max_seq_lengths = kwargs.pop("max_seq_lengths", None)
    num_samples = kwargs.pop("num_samples", None)
    _maybe_enable_substring_match(kwargs)
    if num_samples is not None:
        max_test_samples = num_samples

    precomputed_paths = _resolve_precomputed_paths(data_dir, dataset, max_seq_lengths)
    if precomputed_paths:
        return _load_precomputed_dataset(precomputed_paths, max_test_samples, seed)

    if max_seq_lengths:
        return _load_helmet_multi_length(
            dataset,
            max_seq_lengths,
            test_file,
            demo_file,
            data_dir,
            max_test_samples,
            shots,
            seed,
            truncation_tokenizer,
        )

    data = _load_single_length(
        dataset,
        test_file,
        demo_file,
        data_dir,
        max_test_samples,
        shots,
        seed,
        truncation_tokenizer,
    )
    return {"test": _materialize_prompts(data, dataset)}


def _load_single_length(
    dataset,
    test_file,
    demo_file,
    data_dir,
    max_test_samples,
    shots,
    seed,
    truncation_tokenizer,
):
    if "popqa" in dataset:
        data = _load_qa(
            dataset,
            test_file,
            demo_file,
            data_dir,
            max_test_samples=max_test_samples,
            popularity_threshold=float(dataset.split("_")[-1]),
            shots=shots,
            seed=seed,
        )
    elif any(name in dataset for name in ("nq", "hotpotqa", "triviaqa")):
        data = _load_qa(
            dataset,
            test_file,
            demo_file,
            data_dir,
            max_test_samples=max_test_samples,
            shots=shots,
            seed=seed,
        )
    elif dataset == "json_kv":
        data = _load_json_kv(
            test_file,
            data_dir,
            shots=shots,
            max_test_samples=max_test_samples,
            seed=seed,
        )
    elif "msmarco" in dataset:
        data = _load_msmarco_rerank(
            test_file,
            demo_file,
            data_dir,
            max_test_samples=max_test_samples,
            shots=shots,
            seed=seed,
        )
    elif "ruler" in dataset:
        data = _load_ruler(
            dataset,
            test_file,
            data_dir,
            max_test_samples=max_test_samples,
            seed=seed,
        )
    elif "alce" in dataset:
        data = _load_alce(
            dataset,
            test_file,
            demo_file,
            data_dir,
            shots=shots,
            seed=seed,
            max_test_samples=max_test_samples,
        )
    elif "icl" in dataset:
        data = _load_icl(dataset, max_test_samples=max_test_samples, seed=seed)
    elif "infbench" in dataset:
        data = _load_infbench(
            dataset,
            shots=shots,
            max_test_samples=max_test_samples,
            seed=seed,
            tokenizer=truncation_tokenizer,
        )
    elif "multi_lexsum" in dataset:
        data = _load_multi_lexsum(
            dataset,
            shots=shots,
            max_test_samples=max_test_samples,
            seed=seed,
            tokenizer=truncation_tokenizer,
        )
    elif "narrativeqa" in dataset:
        data = _load_narrativeqa(
            dataset,
            shots=shots,
            max_test_samples=max_test_samples,
            seed=seed,
            tokenizer=truncation_tokenizer,
        )
    else:
        raise ValueError(f"Unknown HELMET dataset: {dataset}")

    return data


def _load_helmet_multi_length(
    family,
    lengths,
    test_file,
    demo_file,
    data_dir,
    max_test_samples,
    shots,
    seed,
    truncation_tokenizer,
):
    """Build one dataset spanning several context lengths for a HELMET family.

    Each requested length is loaded via the existing single-length loader (using
    the variant resolved from the family id), every doc is tagged with its
    ``helmet_max_length``, and the per-length datasets are concatenated, mirroring
    how RULER concatenates ``num_samples`` docs per length into one task.
    """
    per_length_datasets = []
    template = None
    for length in lengths:
        variant_dataset, variant_test_file = _variant_for_length(family, length)
        length_test_file = variant_test_file or test_file
        data = _load_single_length(
            variant_dataset,
            length_test_file,
            demo_file,
            data_dir,
            max_test_samples,
            shots,
            seed,
            truncation_tokenizer,
        )
        data["data"] = data["data"].map(lambda row, L=length: {"helmet_max_length": L})
        template = data
        per_length_datasets.append(data["data"])

    combined = (
        datasets.concatenate_datasets(per_length_datasets)
        if len(per_length_datasets) > 1
        else per_length_datasets[0]
    )
    template = {**template, "data": combined}
    return {"test": _materialize_prompts(template, family)}


def doc_to_text(doc):
    return doc.get("helmet_user_prompt", doc["prompt"])


def doc_to_target(doc):
    answer = doc.get("answer", "")
    if answer == "" and "answers" in doc:
        answer = doc["answers"]
    if isinstance(answer, list):
        first_answer = answer[0] if answer else ""
        if isinstance(first_answer, list):
            return first_answer[0] if first_answer else ""
        return first_answer
    return str(answer)


def process_results(doc, results):
    metrics = _compute_helmet_metrics(doc, results)
    length = doc.get("helmet_max_length")
    if length is None:
        return metrics

    return {str(int(length)): metrics["helmet_score"]}


def aggregate_helmet_score(values):
    values = [value for value in values if value != -1]
    if not values:
        return -1.0
    return sum(values) / len(values)


def _substring_match_enabled():
    return os.environ.get("LM_EVAL_SUBSTRING_MATCH", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _maybe_enable_substring_match(meta):
    """Enable substring match from lm-eval --metadata (per-subtask), e.g. {"substring_match": true}."""
    value = (meta or {}).get("substring_match")
    enabled = (
        value.lower() in ("1", "true", "yes", "on")
        if isinstance(value, str)
        else bool(value)
    )
    if enabled:
        os.environ["LM_EVAL_SUBSTRING_MATCH"] = "1"


def _compute_helmet_metrics(doc, results):
    prediction = results[0]
    metric_type = doc["helmet_metric_type"]

    if metric_type == "ruler_recall":
        answers = _as_list(doc["answer"])
        score = sum(answer.lower() in prediction.lower() for answer in answers) / len(
            answers
        )
        return {"helmet_score": score, "ruler_recall": score}

    if metric_type == "rerank":
        parsed = parse_rankings(prediction)
        metrics = calculate_retrieval_metrics(
            parsed,
            {item[0]: int(item[1]) for item in doc["qrel"]},
            doc["k_values"],
        )
        score = metrics.get("NDCG@10", metrics.get("MRR", 0.0))
        return {
            "helmet_score": score,
            "num_preds": len(parsed),
            **metrics,
        }

    if metric_type == "choice":
        answer = doc["answer"]
        metrics = calculate_metrics(prediction, answer)
        metrics.pop("substring_exact_match", None)

        parsed = parse_output(prediction)
        if parsed is not None:
            parsed_metrics = calculate_metrics(parsed, answer)
            parsed_metrics.pop("substring_exact_match", None)
            metrics = {
                key: max(value, parsed_metrics[key]) for key, value in metrics.items()
            }

        substring_exact_match = answer[1].lower() in prediction.lower()
        metrics["substring_exact_match"] = substring_exact_match
        if substring_exact_match:
            metrics["exact_match"] = True

        metrics["helmet_score"] = metrics["exact_match"]
        return metrics

    if metric_type == "alce_asqa":
        metrics = _process_alce_asqa(doc, prediction)
        metrics["helmet_score"] = metrics["str_em"]
        return metrics

    if metric_type == "alce_qampari":
        metrics = _process_alce_qampari(doc, prediction)
        metrics["helmet_score"] = metrics["qampari_rec_top5"]
        return metrics

    metrics = _default_process_results(doc, prediction)
    if metric_type == "substring_exact_match":
        metrics["helmet_score"] = metrics["substring_exact_match"]
    elif metric_type == "exact_match":
        metrics["helmet_score"] = metrics["exact_match"]
    elif metric_type == "rougeL_f1":
        metrics["helmet_score"] = metrics["rougeL_f1"]
    elif metric_type == "rougeL_recall":
        metrics["helmet_score"] = metrics["rougeL_recall"]
    elif _substring_match_enabled():
        metrics["helmet_score"] = metrics["substring_exact_match"]
    else:
        metrics["helmet_score"] = metrics["f1"]
    return metrics


def _materialize_prompts(data, dataset):
    prompt_template = data["prompt_template"]
    system_template = data["system_template"]
    metric_type = data.get("metric_type", "default")
    k_values = data.get("k_values")

    def add_prompt(example):
        prompt = prompt_template.format(**example)
        gen_prefix = system_template.format(**example)
        user_prompt = prompt
        if gen_prefix and prompt.endswith(gen_prefix):
            user_prompt = prompt[: -len(gen_prefix)]
        else:
            gen_prefix = ""

        output = {
            "prompt": prompt,
            "helmet_user_prompt": user_prompt,
            "helmet_gen_prefix": gen_prefix,
            "helmet_dataset": dataset,
            "helmet_metric_type": metric_type,
            "system_template": system_template,
        }
        if k_values is not None:
            output["k_values"] = k_values
        return output

    return data["data"].map(add_prompt)


def _resolve_file(path, data_dir=None):
    if path is None or path == "":
        return ""

    candidate = Path(path)
    if candidate.is_absolute() and candidate.exists():
        return str(candidate)

    roots = []
    if data_dir:
        roots.append(Path(data_dir))
    if os.environ.get(HELMET_DATA_DIR):
        roots.append(Path(os.environ[HELMET_DATA_DIR]))
    roots.append(Path(HELMET_DEFAULT_DATA_DIR))
    roots.append(Path.cwd())
    roots.append(Path(__file__).parent)

    for root in roots:
        candidate = root / path
        if candidate.exists():
            return str(candidate)

    if _is_helmet_data_path(path):
        _download_helmet_data(_download_root(data_dir), path)
        for root in roots:
            candidate = root / path
            if candidate.exists():
                return str(candidate)

    raise FileNotFoundError(
        f"HELMET data file '{path}' was not found. Download and extract "
        f"https://huggingface.co/datasets/{HELMET_DATA_REPO} {HELMET_DATA_ARCHIVE}, "
        f"then set {HELMET_DATA_DIR} to the extracted directory or pass data_dir."
    )


def _is_helmet_data_path(path):
    return Path(path).parts[:1] == ("data",)


def _download_root(data_dir=None):
    if data_dir:
        return Path(data_dir)
    if os.environ.get(HELMET_DATA_DIR):
        return Path(os.environ[HELMET_DATA_DIR])
    return Path(HELMET_DEFAULT_DATA_DIR)


@cache
def _download_helmet_data(root, expected_path):
    root = Path(root)
    expected_file = root / expected_path
    if expected_file.exists():
        return

    root.mkdir(parents=True, exist_ok=True)
    eval_logger.warning(
        "HELMET data file %s was not found. Downloading %s/%s to %s.",
        expected_path,
        HELMET_DATA_REPO,
        HELMET_DATA_ARCHIVE,
        root,
    )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise FileNotFoundError(
            f"HELMET data file '{expected_path}' was not found and "
            "huggingface_hub is unavailable for automatic download."
        ) from exc

    archive_path = Path(
        hf_hub_download(
            repo_id=HELMET_DATA_REPO,
            repo_type="dataset",
            filename=HELMET_DATA_ARCHIVE,
        )
    )
    _extract_helmet_data(archive_path, root)


def _extract_helmet_data(archive_path, root):
    root = Path(root)
    root_resolved = root.resolve()
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            target = (root / member.name).resolve()
            if target != root_resolved and root_resolved not in target.parents:
                raise RuntimeError(
                    f"Refusing to extract HELMET archive member outside {root}: "
                    f"{member.name}"
                )
        archive.extractall(root)


def _load_json_dataset(path, data_dir=None, field=None):
    resolved_path = _resolve_file(path, data_dir)
    kwargs = {"data_files": resolved_path}
    if field is not None:
        kwargs["field"] = field
    return datasets.load_dataset("json", **kwargs)["train"]


def _drop_duplicates(data, key):
    indices_to_keep = []
    keys = set()
    for i, item in enumerate(data):
        if item[key] in keys:
            continue
        indices_to_keep.append(i)
        keys.add(item[key])
    return data.select(indices_to_keep)


def _load_qa(
    dataset,
    path,
    demo_path,
    data_dir,
    max_test_samples=None,
    popularity_threshold=None,
    shots=0,
    seed=42,
):
    if "nq_bad" in dataset:
        user_template = (
            "Use the given documents to write a concise and short answer to the "
            "question. Only use the information presented in the documents, and "
            "output 'unanswerable' if the question is not valid or cannot be "
            "answered with the given document. Write your answer in the following "
            "format:\nAnswer: [answer]\n\n{demos}{context}\n\nQuestion: {question}"
        )
    else:
        user_template = (
            "Use the given documents to write a concise and short answer to the "
            "question. Write your answer in the following format:\nAnswer: "
            "[answer]\n\n{demos}{context}\n\nQuestion: {question}"
        )
    system_template = "Answer:"

    data = _load_json_dataset(path, data_dir)
    demo_data = _load_json_dataset(demo_path, data_dir)

    if "popqa" in dataset and popularity_threshold is not None:
        data = data.filter(
            lambda item: math.log10(item["s_pop"]) < popularity_threshold
        )
        demo_data = demo_data.filter(
            lambda item: math.log10(item["s_pop"]) < popularity_threshold
        )

    key = "id" if "id" in data.column_names else "question"
    if max_test_samples is not None:
        keys = sorted(set(data[key]))
        selected = set(
            random.Random(seed).sample(keys, min(max_test_samples, len(keys)))
        )
        data = data.filter(lambda item: item[key] in selected)

    demo_template = "{documents}\n\nQuestion: {question}\nAnswer: {answer}"
    passage_template = "Document (Title: {title}): {text}"

    def update(sample):
        demo_text = ""
        demos = demo_data
        if shots > 0:
            if "popqa" in dataset:
                demos = demo_data.filter(lambda item: item[key] != sample[key])

            local_seed = (
                int(hashlib.sha256(str(sample[key]).encode("utf-8")).hexdigest(), 16)
                % 2**31
            )
            demos = _drop_duplicates(demos.shuffle(seed=local_seed), key)
            demos = demos.select(range(min(shots, len(demos))))
            demo_text = (
                "\n\n".join(
                    demo_template.format(
                        **demo,
                        documents="\n\n".join(
                            passage_template.format(**ctx) for ctx in demo["ctxs"]
                        ),
                        answer=demo["answers"][0],
                    )
                    for demo in demos
                )
                + "\n\n"
            )

        passage_text = "\n\n".join(
            passage_template.format(**ctx) for ctx in sample["ctxs"]
        )
        return {
            "demos": demo_text,
            "context": passage_text,
            "answer": sample["answers"],
        }

    return {
        "data": data.map(update),
        "prompt_template": user_template + "\n" + system_template,
        "system_template": system_template,
        "metric_type": "substring_exact_match",
    }


def _load_json_kv(path, data_dir, shots=0, max_test_samples=None, seed=42):
    user_template = (
        "{context}\n\nExtract the value corresponding to the specified key in the "
        "JSON object below.\n\n{demos}Key: {question}"
    )
    system_template = "Corresponding value:"
    data = _load_json_dataset(path, data_dir)
    demo_template = "Key: {key}\nCorresponding value:{value}"

    def update(sample):
        return {
            "demos": "\n\n".join(
                demo_template.format(key=key, value=" " + value)
                for key, value in sample["demos"][:shots]
            )
            + ("\n\n" if shots > 0 else ""),
            "k": sample["num_kvs"],
        }

    data = data.map(update)
    if max_test_samples is not None:
        data = data.shuffle(seed=seed).select(range(min(max_test_samples, len(data))))

    return {
        "data": data,
        "prompt_template": user_template + "\n" + system_template,
        "system_template": system_template,
        "metric_type": "substring_exact_match",
    }


def _load_msmarco_rerank(
    path,
    demo_path,
    data_dir,
    max_test_samples=None,
    shots=0,
    seed=42,
):
    user_template = (
        "You are provided with a list of documents, each indicated by their ID. "
        "Rank each document based on their relevance to the question in "
        "descending order from most relelvant to least relevant texts. Include "
        "all documents in the rankings. Write your answer using the unique IDs, "
        "with the following format:\nRanking: ID3 > ID1 > ID2\n\n{demos}"
        "{context}\n\nQuery: {question}"
    )
    system_template = "Ranking:"
    data = _load_json_dataset(path, data_dir)
    demos = _load_json_dataset(demo_path, data_dir)

    if max_test_samples is not None:
        key = "qid" if "qid" in data.column_names else "query"
        keys = sorted(set(data[key]))
        selected = set(
            random.Random(seed).sample(keys, min(max_test_samples, len(keys)))
        )
        data = data.filter(lambda item: item[key] in selected)

    k_values = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
    k_values = [k for k in k_values if k <= len(data[0]["ctxs"])]

    demo_filtered = False
    if len(demos) > 2 * len(data):
        qids = set(data["qid"])
        demos = demos.filter(lambda item: item["qid"] not in qids)
        demo_filtered = True

    def update(sample):
        passage_template = (
            "[ID: {id}] Document (Title: {title}): {text}"
            if "title" in sample["ctxs"][0]
            else "[ID: {id}] Document: {text}"
        )
        passage_text = "\n\n".join(
            passage_template.format(**ctx) for ctx in sample["ctxs"]
        )
        gold_ranking = " > ".join(
            ctx["id"]
            for ctx in sorted(
                sample["ctxs"], key=lambda ctx: ctx["label"], reverse=True
            )
        )
        demo_text = ""

        if shots > 0:
            local_demos = demos
            if not demo_filtered:
                local_demos = demos.filter(lambda item: item["qid"] != sample["qid"])
            local_seed = (
                int(hashlib.sha256(sample["qid"].encode("utf-8")).hexdigest(), 16)
                % 2**31
            )
            local_demos = _drop_duplicates(local_demos.shuffle(seed=local_seed), "qid")
            local_demos = local_demos.select(range(min(shots, len(local_demos))))

            demo_ids = set()
            for demo in local_demos:
                if demo["qid"] in demo_ids or len(demo_ids) >= shots:
                    continue
                demo_ids.add(demo["qid"])
                ids = sorted(demo["ctxs"], key=lambda ctx: ctx["label"], reverse=True)
                ranking = " > ".join(ctx["id"] for ctx in ids)
                demo_text += (
                    "\n\n".join(passage_template.format(**ctx) for ctx in demo["ctxs"])
                    + f"\n\nQuery: {demo['query']}\nRanking: {ranking}\n\n"
                )

        return {
            "context": passage_text,
            "question": sample["query"],
            "demos": demo_text,
            "answer": gold_ranking,
            "qrel": [[ctx["id"], str(ctx["label"])] for ctx in sample["ctxs"]],
        }

    return {
        "data": data.map(update, remove_columns=["query", "ctxs"]),
        "prompt_template": user_template + "\n" + system_template,
        "system_template": system_template,
        "metric_type": "rerank",
        "k_values": k_values,
    }


def _load_ruler(dataset, path, data_dir, max_test_samples=None, seed=42):
    data = _load_json_dataset(path, data_dir)

    if "niah_mv" in dataset or "niah_mq" in dataset:
        user_template = (
            "Some special magic {type_needle_v} are hidden within the following "
            "text. Make sure to memorize it. I will quiz you about the "
            "{type_needle_v} afterwards.\n{context}\nWhat are all the special "
            "magic {type_needle_v} for {query} mentioned in the provided text?"
        )
        system_template = (
            "The special magic {type_needle_v} for {query} mentioned in the "
            "provided text are"
        )
    elif "niah" in dataset:
        user_template = (
            "A special magic {type_needle_v} is hidden within the following text. "
            "Make sure to memorize it. I will quiz you about the {type_needle_v} "
            "afterwards.\n{context}\nWhat is the special magic {type_needle_v} "
            "for {query} mentioned in the provided text?"
        )
        system_template = (
            "The special magic {type_needle_v} for {query} mentioned in the "
            "provided text is"
        )
    elif "vt" in dataset:
        user_template = (
            "{example}Memorize and track the chain(s) of variable assignment "
            "hidden in the following text.\n\n{context}\nQuestion: Find all "
            "variables that are assigned the value {query} in the text above."
        )
        system_template = (
            "Answer: According to the chain(s) of variable assignment in the text "
            "above, {num_v} variables are assigned the value {query}, they are:"
        )
    elif "cwe" in dataset:
        user_template = (
            "{example}Below is a numbered list of words. In these words, some "
            "appear more often than others. Memorize the ones that appear most "
            "often.\n{context}\nQuestion: What are the 10 most common words in "
            "the above list?"
        )
        system_template = (
            "Answer: The top 10 words that appear most often in the list are:"
        )
    elif "fwe" in dataset:
        user_template = (
            "Read the following coded text and track the frequency of each coded "
            "word. Find the three most frequently appeared coded words.\n{context}"
            "\nQuestion: Do not provide any explanation. Please ignore the dots "
            "'....'. What are the three most frequently appeared words in the "
            "above coded text?"
        )
        system_template = (
            "Answer: According to the coded text above, the three most frequently "
            "appeared words are:"
        )
    elif "qa" in dataset:
        user_template = (
            "Answer the question based on the given documents. Only give me the "
            "answer and do not output any other words.\n\nThe following are given "
            "documents.\n\n{context}\n\nAnswer the question based on the given "
            "documents. Only give me the answer and do not output any other "
            "words.\n\nQuestion: {question}"
        )
        system_template = "Answer:"
    else:
        raise ValueError(f"Unknown HELMET RULER dataset: {dataset}")

    def process_example(example):
        return {
            "question": example["query"]
            if "query" in example
            else example.get("question", ""),
            "example": example["example"] + "\n\n"
            if "example" in example and example["example"] != ""
            else "",
            "answer": example["answer"] if "answer" in example else example["outputs"],
        }

    data = data.map(process_example)
    if max_test_samples is not None:
        data = data.shuffle(seed=seed).select(range(min(len(data), max_test_samples)))

    metric_type = "default" if "qa" in dataset else "ruler_recall"
    return {
        "data": data,
        "prompt_template": user_template + "\n" + system_template,
        "system_template": system_template,
        "metric_type": metric_type,
    }


def _load_alce(
    dataset, path, demo_path, data_dir, shots=0, seed=42, max_test_samples=None
):
    with open(_resolve_file(demo_path, data_dir), encoding="utf-8") as handle:
        demos = json.load(handle)

    instruction = demos["instruction"]
    demo_prompt = demos["demo_prompt"]
    doc_prompt = demos["doc_prompt"]

    user_template = "{demo_text}{instruction}\n\nQuestion: {question}\n\n{context}"
    system_template = "Answer:"
    data = _load_json_dataset(path, data_dir)
    num_docs = int(dataset.split("_")[-1])
    rng = random.Random(seed)

    def preprocess_example(example):
        context = "\n\n".join(
            doc_prompt.format(**doc, ID=index + 1)
            for index, doc in enumerate(example["docs"][:num_docs])
        )
        demo_text = "\n\n\n".join(
            demo_prompt.format(
                **demo,
                instruction=instruction,
                context="\n\n".join(
                    doc_prompt.format(**doc, ID=index + 1)
                    for index, doc in enumerate(demo["docs"])
                ),
            )
            for demo in rng.sample(demos["demos"], shots)
        )
        if shots > 0:
            demo_text += "\n\n\n"
        return {
            "context": context,
            "demo_text": demo_text,
            "instruction": instruction,
        }

    data = data.map(preprocess_example)
    if max_test_samples is not None:
        data = data.shuffle(seed=seed).select(range(min(max_test_samples, len(data))))

    return {
        "data": data,
        "prompt_template": user_template + "\n\n" + system_template,
        "system_template": system_template,
        "metric_type": "alce_qampari" if "qampari" in dataset else "alce_asqa",
    }


def _load_icl(dataset, max_test_samples=None, seed=42):
    shot = int(dataset.split("shot")[0].split("_")[-1])

    if "trec_fine" in dataset.lower():
        train_data = datasets.load_dataset("CogComp/trec", trust_remote_code=True)[
            "train"
        ]
        test_data = datasets.load_dataset("CogComp/trec", trust_remote_code=True)[
            "test"
        ]
        id2label = train_data.features["fine_label"].names
        text_field = "text"
        label_field = "fine_label"
        num_labels = 50
    elif "trec_coarse" in dataset.lower():
        train_data = datasets.load_dataset("CogComp/trec", trust_remote_code=True)[
            "train"
        ]
        test_data = datasets.load_dataset("CogComp/trec", trust_remote_code=True)[
            "test"
        ]
        id2label = train_data.features["coarse_label"].names
        text_field = "text"
        label_field = "coarse_label"
        num_labels = 6
    elif "banking77" in dataset.lower():
        train_data = datasets.load_dataset("PolyAI/banking77", trust_remote_code=True)[
            "train"
        ]
        test_data = datasets.load_dataset("PolyAI/banking77", trust_remote_code=True)[
            "test"
        ]
        id2label = train_data.features["label"].names
        id2label = {i: id2label[i] for i in range(len(id2label))}
        text_field = "text"
        label_field = "label"
        num_labels = 77
    elif "clinic150" in dataset.lower():
        train_data = datasets.load_dataset("clinc/clinc_oos", "plus")["train"]
        test_data = datasets.load_dataset("clinc/clinc_oos", "plus")["validation"]
        id2label = train_data.features["intent"].names
        text_field = "text"
        label_field = "intent"
        num_labels = 151
    elif "nlu" in dataset.lower():
        all_data = datasets.load_dataset(
            "xingkunliuxtracta/nlu_evaluation_data", trust_remote_code=True
        )["train"]
        id2label = all_data.features["label"].names
        all_data = all_data.train_test_split(test_size=0.1, seed=seed)
        train_data = all_data["train"]
        test_data = all_data["test"]
        text_field = "text"
        label_field = "label"
        num_labels = 68
    else:
        raise ValueError(f"Unknown HELMET ICL dataset: {dataset}")

    def balance_labels(data, shots, local_seed):
        rand = random.Random(local_seed)
        label_mapping = {item[label_field]: [] for item in data}
        for item in data:
            label_mapping[item[label_field]].append(item)

        num_rounds = math.ceil(shots / len(label_mapping))
        selected = [[] for _ in range(num_rounds)]
        for samples in label_mapping.values():
            indices = rand.sample(range(len(samples)), num_rounds % len(samples))
            while len(indices) < num_rounds:
                indices += rand.sample(
                    range(len(samples)), min(num_rounds - len(indices), len(samples))
                )
            for round_index, sample_index in enumerate(indices):
                selected[round_index].append(samples[sample_index])

        for round_items in selected:
            rand.shuffle(round_items)
        return [item for round_items in selected for item in round_items][:shots]

    if max_test_samples is not None and len(test_data) > max_test_samples:
        test_data = datasets.Dataset.from_list(
            balance_labels(test_data.shuffle(seed=seed), max_test_samples, seed)
        )

    item_template = "{text}\nlabel: {label}"
    user_template = (
        "Use the provided mapping from the text to label to assign a label to the text. "
        'Only output "label: {{label}}" and nothing else. \n\n{context}\n\n{question}'
    )
    system_template = "label:"

    def preprocess(sample):
        local_seed = (
            int(hashlib.sha256(sample[text_field].encode("utf-8")).hexdigest(), 16)
            + seed
        ) % 2**31
        rng = np.random.default_rng(local_seed)
        if "balance" in dataset:
            demos = balance_labels(train_data, shot, local_seed)
        else:
            demos = []
            while len(demos) < shot:
                demos.extend(
                    rng.choice(
                        train_data,
                        min(len(train_data), shot - len(demos)),
                        replace=False,
                    )
                )

        if "natural_label" in dataset:
            label_mapping = [id2label[i] for i in range(num_labels)]
        else:
            label_mapping = list(range(num_labels))
            random.Random(local_seed).shuffle(label_mapping)

        context = "\n\n".join(
            item_template.format(
                text=selected[text_field],
                label=str(label_mapping[int(selected[label_field])]),
            )
            for selected in demos
        )
        return {
            "context": context,
            "question": sample[text_field],
            "answer": str(label_mapping[int(sample[label_field])]),
        }

    return {
        "data": test_data.map(preprocess),
        "prompt_template": user_template + "\n" + system_template,
        "system_template": system_template,
        "metric_type": "exact_match",
    }


_INFBENCH_SPLIT_BY_TASK = {
    "qa_eng": "longbook_qa_eng",
    "choice_eng": "longbook_choice_eng",
    "sum_eng": "longbook_sum_eng",
}


def _load_infbench(dataset, shots=0, max_test_samples=None, seed=42, tokenizer=None):
    features = datasets.Features(
        {
            "id": datasets.Value("int64"),
            "context": datasets.Value("string"),
            "input": datasets.Value("string"),
            "answer": datasets.Sequence(datasets.Value("string")),
            "options": datasets.Sequence(datasets.Value("string")),
        }
    )

    split_name = next(
        (split for key, split in _INFBENCH_SPLIT_BY_TASK.items() if key in dataset),
        None,
    )
    if split_name is None:
        raise ValueError(f"Unknown HELMET InfiniteBench dataset: {dataset}")

    from huggingface_hub import hf_hub_download

    infbench_jsonl = hf_hub_download(
        repo_id="xinrongzhang2022/InfiniteBench",
        repo_type="dataset",
        filename=f"{split_name}.jsonl",
    )
    data = datasets.load_dataset(
        "json",
        data_files={split_name: infbench_jsonl},
        split=split_name,
        features=features,
    )

    metric_type = "default"
    if "qa_eng" in dataset:
        user_template = (
            "You are given a story and a question. Answer the question as concisely "
            "as you can, using a single phrase if possible.\n\n{demo}{context}\n\n"
            "Question: {question}"
        )
        system_template = "Answer:"
        metric_type = "rougeL_f1"
    elif "choice_eng" in dataset:
        user_template = (
            "You are given a story and a question with multiple choices. Choose "
            "the best answer from the options provided. Only one of the following "
            "options is correct, output the answer using one single letter (A, B, "
            "C, or D). Don't say anything else.\n\n{demo}{context}\n\nQuestion: "
            "{question}\nOptions:\n{options}"
        )
        system_template = "Answer:"
        metric_type = "choice"
    elif "sum_eng" in dataset:
        user_template = (
            "You are given a book and you are tasked to summarize it. Write a "
            "summary of about 1000 to 1200 words. Only write about the plot and "
            "characters of the story. Do not discuss the themes or background of "
            "the book. Do not provide any analysis or commentary.\n\n{demo}"
            "{context}\n\nNow summarize the book."
        )
        system_template = "Summary:"
        metric_type = "rougeL_f1"
    else:
        raise ValueError(f"Unknown HELMET InfiniteBench dataset: {dataset}")

    def process_example(example):
        update = {"question": example["input"], "demo": ""}
        if "choice" in dataset:
            options = "A. {}\nB. {}\nC. {}\nD. {}".format(*example["options"])
            answer = chr(ord("A") + example["options"].index(example["answer"][0]))
            update["options"] = options
            update["answer"] = [answer, f"{answer}. {example['answer'][0]}"]
        return update

    data = data.map(process_example)

    def add_demos(example):
        demos = (
            data.filter(lambda item: item["id"] != example["id"])
            .shuffle(seed=seed)
            .select(range(shots))
        )
        if "qa_eng" in dataset:
            template = "[story text]\nQuestion: {question}\nAnswer: {answer[0]}"
            demo = "\n\n".join(template.format(**item) for item in demos)
        elif "choice_eng" in dataset:
            template = (
                "[story text]\nQuestion: {question}\nOptions:\n{options}\n"
                "Answer: {answer[0]}"
            )
            demo = "\n\n".join(template.format(**item) for item in demos)
        else:
            demo = "\n\n".join(
                f"[story text]\nSummary: {item['answer'][0].strip()}" for item in demos
            )
        return {"demo": f"For example:\n\n{demo}\n\nNow, read the following story:\n\n"}

    if shots > 0:
        data = data.map(add_demos)

    data = _filter_length(data, 65536, "context", tokenizer=tokenizer)
    data = _truncate_by_tokenizer(dataset, data, tokenizer=tokenizer)

    if max_test_samples is not None:
        data = data.shuffle(seed=seed).select(range(min(len(data), max_test_samples)))

    return {
        "data": data,
        "prompt_template": user_template + "\n\n" + system_template,
        "system_template": system_template,
        "metric_type": metric_type,
    }


def _load_multi_lexsum(
    dataset, shots=0, max_test_samples=None, seed=42, tokenizer=None
):
    all_data = datasets.load_dataset(
        "allenai/multi_lexsum", name="v20230518", trust_remote_code=True
    )
    all_data = all_data.filter(lambda item: item["summary/short"] is not None)
    train_data = all_data["train"]

    user_template = (
        "You are given the legal documents in a civil rights lawsuit, and you are "
        "tasked to summarize the case. Write a concise summary of one paragraph "
        "(200 to 250 words). The summary should contain a short description of "
        "the background, the parties involved, and the outcomes of the case.\n\n"
        "{demo}Legal documents:\n{context}\n\nNow please summarize the case."
    )
    system_template = "Summary:"

    def preprocess(example):
        demo = ""
        if shots > 0:
            demo = (
                "Example summaries:\n\n"
                + "\n\n".join(
                    "Summary: {}".format(item["summary/short"])
                    for item in train_data.shuffle(seed=seed).select(range(shots))
                )
                + "\n\nNow, write a summary of the following legal documents.\n"
            )
        return {
            "context": "\n\n".join(example["sources"]),
            "demo": demo,
            "answer": example["summary/short"],
            "question": "",
        }

    data = all_data["validation"].map(preprocess)
    data = _filter_length(data, 65536, "context", tokenizer=tokenizer)
    data = _truncate_by_tokenizer(dataset, data, tokenizer=tokenizer)
    if max_test_samples is not None and len(data) > max_test_samples:
        data = data.shuffle(seed=seed).select(range(max_test_samples))

    return {
        "data": data,
        "prompt_template": user_template + "\n\n" + system_template,
        "system_template": system_template,
        "metric_type": "rougeL_f1",
    }


def _load_narrativeqa(dataset, shots=0, max_test_samples=None, seed=42, tokenizer=None):
    all_data = datasets.load_dataset("narrativeqa")
    train_data = all_data["train"]
    data = all_data["test"].shuffle(seed=seed)
    truncation_tokenizer = _load_truncation_tokenizer(tokenizer=tokenizer)
    data = data.map(
        lambda item: {
            "input_length": len(
                truncation_tokenizer(item["document"]["text"])["input_ids"]
            )
        }
    )
    data = data.filter(lambda item: item["input_length"] > 131072)
    data = data.remove_columns("input_length")

    user_template = (
        "You are given a story, which can be either a novel or a movie script, and "
        "a question. Answer the question as concisely as you can, using a single "
        "phrase if possible.\n\n{demo}{context}\n\nQuestion: {question}"
    )
    system_template = "Answer:"

    def preprocess(example):
        demo = ""
        if shots > 0:
            demo = (
                "For example:\n\n"
                + "\n\n".join(
                    "Question: {question}\nAnswer: {answer}".format(
                        question=item["question"]["text"],
                        answer=item["answers"][0]["text"],
                    )
                    for item in train_data.shuffle(seed=seed).select(range(shots))
                )
                + "\n\nNow, use the following story to answer the question:\n\n"
            )
        return {
            "context": example["document"]["text"],
            "question": example["question"]["text"],
            "answer": [answer["text"] for answer in example["answers"]],
            "demo": demo,
        }

    data = data.map(preprocess, remove_columns=["document", "answers"])
    data = _filter_length(data, 131072, "context", tokenizer=tokenizer)
    data = _truncate_by_tokenizer(dataset, data, tokenizer=tokenizer)
    if max_test_samples is not None:
        data = data.select(range(min(max_test_samples, len(data))))

    return {
        "data": data,
        "prompt_template": user_template + "\n" + system_template,
        "system_template": system_template,
        "metric_type": "rougeL_f1",
    }


def _default_process_results(doc, prediction):
    answer = doc["answer"]
    metrics = calculate_metrics(prediction, answer)
    parsed = parse_output(prediction, doc.get("system_template", "Answer:"))
    if parsed is not None:
        parsed_metrics = calculate_metrics(parsed, answer)
        metrics = {
            key: max(value, parsed_metrics[key]) for key, value in metrics.items()
        }
    return metrics


@cache
def _load_truncation_tokenizer(tokenizer=None, pretrained=None):
    from transformers import AutoTokenizer

    tokenizer_name = tokenizer or pretrained
    assert tokenizer_name, "No tokenizer or pretrained provided for HELMET truncation."
    eval_logger.info(f"Using tokenizer {tokenizer_name} for HELMET truncation.")
    return AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )


def _truncate_by_tokenizer(
    dataset,
    data,
    tokenizer=None,
    postfix_text=" ... [the rest of the text is omitted]",
):
    max_length = int(dataset.split("_")[-1])
    tokenizer = _load_truncation_tokenizer(tokenizer=tokenizer)
    separator_length = len(tokenizer(postfix_text)["input_ids"])

    def truncate(sample):
        tokens = tokenizer(sample["context"], return_offsets_mapping=True)
        if len(tokens["input_ids"]) > max_length:
            sample["context"] = (
                sample["context"][
                    : tokens["offset_mapping"][max_length - separator_length][1]
                ]
                + postfix_text
            )
        return sample

    return data.map(truncate)


def _filter_length(data, min_length, key, tokenizer=None):
    tokenizer = _load_truncation_tokenizer(tokenizer=tokenizer)
    return data.filter(
        lambda item: len(tokenizer(item[key])["input_ids"]) >= min_length
    )


def _process_alce_asqa(doc, prediction):
    prediction = remove_citations(re.sub(r"\n+", " ", prediction))
    metrics = {"length": len(prediction.split())}
    metrics["str_em"], metrics["str_hit"] = _str_em_for_doc(doc, prediction)
    metrics["rougeLsum"] = _rouge_for_doc(doc, prediction)
    return metrics


def _process_alce_qampari(doc, prediction):
    prediction = remove_citations(re.sub(r"\n+", " ", prediction))
    return {
        "length": len(prediction.split()),
        **_qampari_f1_for_doc(doc, prediction),
    }


def normalize_answer(text):
    def remove_articles(value):
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value):
        return " ".join(value.split())

    def remove_punc(value):
        exclude = set(string.punctuation)
        return "".join(char for char in value if char not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def remove_citations(text):
    return (
        re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", text))
        .replace(" |", "")
        .replace("]", "")
    )


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    if (
        normalized_prediction in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return 0.0
    if (
        normalized_ground_truth in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return 0.0

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def substring_exact_match_score(prediction, ground_truth):
    return normalize_answer(ground_truth) in normalize_answer(prediction)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    return max(
        metric_fn(prediction, ground_truth) for ground_truth in _as_list(ground_truths)
    )


def calculate_metrics(prediction, answers):
    scorer = rouge_scorer.RougeScorer(["rougeL", "rougeLsum"], use_stemmer=True)
    answers = _as_list(answers)
    rouges = [scorer.score(target=answer, prediction=prediction) for answer in answers]
    return {
        "exact_match": metric_max_over_ground_truths(
            exact_match_score, prediction, answers
        ),
        "f1": metric_max_over_ground_truths(f1_score, prediction, answers),
        "substring_exact_match": metric_max_over_ground_truths(
            substring_exact_match_score, prediction, answers
        ),
        "rougeL_f1": max(score["rougeL"].fmeasure for score in rouges),
        "rougeL_recall": max(score["rougeL"].recall for score in rouges),
        "rougeLsum_f1": max(score["rougeLsum"].fmeasure for score in rouges),
        "rougeLsum_recall": max(score["rougeLsum"].recall for score in rouges),
    }


def parse_output(output, prefix="Answer:"):
    patterns = [
        re.compile(f"(?:{re.escape(prefix)})(.*)(?:\n|$)", flags=re.IGNORECASE),
        re.compile(r"(?:^)(.*)(?:\n|$)"),
    ]
    for pattern in patterns:
        match = pattern.search(output)
        if match is not None:
            return re.sub(
                f"^{re.escape(prefix)}", "", match[1].strip(), flags=re.IGNORECASE
            ).strip()
    return None


def parse_rankings(output):
    output = re.sub(r"[\[\]:]", "", output)
    output = output.lower().replace("id", "")
    matches = re.finditer(r"(\d+)(?:\s*>\s*(\d+))*", output)
    longest = ""
    for match in matches:
        if len(match.group(0)) > len(longest):
            longest = match.group(0)

    if longest:
        rankings = [num.strip() for num in longest.split(">") if num.strip().isdigit()]
    else:
        rankings = [output]

    results = {}
    for i, rank in enumerate(rankings):
        if rank not in results:
            results[rank] = len(rankings) - i
    return results


def calculate_retrieval_metrics(results, qrels, k_values):
    ranked = [
        doc_id
        for doc_id, _ in sorted(results.items(), key=lambda item: item[1], reverse=True)
    ]
    relevant = {doc_id: rel for doc_id, rel in qrels.items() if rel > 0}
    output = {}
    total_relevant = len(relevant)

    for k in k_values:
        top_k = ranked[:k]
        gains = [qrels.get(doc_id, 0) for doc_id in top_k]
        dcg = _dcg(gains)
        ideal = _dcg(sorted(qrels.values(), reverse=True)[:k])
        retrieved_relevant = sum(1 for doc_id in top_k if doc_id in relevant)
        output[f"NDCG@{k}"] = 0.0 if ideal == 0 else dcg / ideal
        output[f"MAP@{k}"] = _average_precision_at_k(top_k, relevant, k)
        output[f"Recall@{k}"] = (
            0.0 if total_relevant == 0 else retrieved_relevant / total_relevant
        )
        output[f"P@{k}"] = retrieved_relevant / k

    output["MRR"] = 0.0
    for index, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            output["MRR"] = 1 / index
            break
    return output


def _dcg(gains):
    return sum((2**gain - 1) / math.log2(index + 2) for index, gain in enumerate(gains))


def _average_precision_at_k(ranked, relevant, k):
    if not relevant:
        return 0.0
    score = 0.0
    hits = 0
    for index, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in relevant:
            hits += 1
            score += hits / index
    return score / len(relevant)


def _str_em_for_doc(doc, prediction):
    if "qa_pairs" not in doc or doc["qa_pairs"] is None:
        return 0.0, 0.0

    scores = [
        any(
            normalize_answer(answer) in normalize_answer(prediction)
            for answer in pair["short_answers"]
        )
        for pair in doc["qa_pairs"]
    ]
    score = float(np.mean(scores))
    return score, float(score == 1.0)


def _rouge_for_doc(doc, prediction):
    scorer = rouge_scorer.RougeScorer(["rougeLsum"], use_stemmer=True)
    if "annotations" in doc and doc["annotations"] is not None:
        references = [
            annotation["long_answer"] for annotation in doc["annotations"][:2]
        ]
    else:
        references = _as_list(doc["answer"])
    return max(
        scorer.score(reference.lower(), prediction.lower())["rougeLsum"].fmeasure
        for reference in references
    )


def _qampari_f1_for_doc(doc, prediction):
    answer_groups = doc.get("answers", doc.get("answer"))
    if answer_groups is None:
        return {
            "num_preds": 0.0,
            "qampari_prec": 0.0,
            "qampari_rec": 0.0,
            "qampari_rec_top5": 0.0,
            "qampari_f1": 0.0,
            "qampari_f1_top5": 0.0,
        }

    answers = [
        [normalize_answer(answer) for answer in _as_list(answer_group)]
        for answer_group in answer_groups
    ]
    flat_answers = {answer for answer_group in answers for answer in answer_group}
    predictions = [
        normalize_answer(item)
        for item in re.split(r",|;", prediction)
        if normalize_answer(item)
    ]
    if not predictions:
        return {
            "num_preds": 0.0,
            "qampari_prec": 0.0,
            "qampari_rec": 0.0,
            "qampari_rec_top5": 0.0,
            "qampari_f1": 0.0,
            "qampari_f1_top5": 0.0,
        }

    num_matched_predictions = sum(
        prediction in flat_answers for prediction in predictions
    )
    num_matched_answers = sum(
        any(answer in predictions for answer in answer_group)
        for answer_group in answers
    )
    precision = num_matched_predictions / len(predictions)
    recall = num_matched_answers / len(answers) if answers else 0.0
    recall_top5 = min(5, num_matched_answers) / min(5, len(answers)) if answers else 0.0
    f1 = (
        0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )
    f1_top5 = (
        0.0
        if precision + recall_top5 == 0
        else 2 * precision * recall_top5 / (precision + recall_top5)
    )
    return {
        "num_preds": len(predictions),
        "qampari_prec": precision,
        "qampari_rec": recall,
        "qampari_rec_top5": recall_top5,
        "qampari_f1": f1,
        "qampari_f1_top5": f1_top5,
    }


def _as_list(value):
    if isinstance(value, str):
        return [value]
    if len(value) > 0 and isinstance(value[0], list):
        return [item for sublist in value for item in sublist]
    return list(value)
