import json
import logging
import re
from difflib import SequenceMatcher

import tiktoken


eval_logger = logging.getLogger(__name__)

# Pattern to strip thinking tags (e.g. Qwen3 <think>...</think>)
THINK_TAG_PATTERN = re.compile(r"^<think>.*?</think>\s*", re.DOTALL)
ENCODING = tiktoken.get_encoding("o200k_base")


def doc_to_text(doc):
    """Parse JSON-encoded chat messages and flatten to text."""
    messages = json.loads(doc["prompt"])
    parts = []
    for msg in messages:
        role = msg["role"].capitalize()
        parts.append(f"{role}: {msg['content']}")
    return "\n\n".join(parts)


def process_results(doc, results):
    """Grade using SequenceMatcher ratio with random_string_to_prepend prefix check."""
    response = results[0]
    response = (response or "").strip()
    # Strip thinking tags (e.g. <think>...</think>) from reasoning models
    response = THINK_TAG_PATTERN.sub("", response).strip()
    answer = doc["answer"]
    random_string = doc["random_string_to_prepend"]

    if not response.startswith(random_string):
        return {"mrcr_score": 0.0}

    response_clean = response.removeprefix(random_string)
    answer_clean = answer.removeprefix(random_string)
    score = float(SequenceMatcher(None, response_clean, answer_clean).ratio())
    return {"mrcr_score": score}


def filter_2needle(dataset):
    return dataset.filter(lambda x: x["n_needles"] == 2)


def filter_4needle(dataset):
    return dataset.filter(lambda x: x["n_needles"] == 4)


def filter_8needle(dataset):
    return dataset.filter(lambda x: x["n_needles"] == 8)


CONTEXT_BINS = {
    "8k": (4096, 8192),
    "16k": (8192, 16384),
    "32k": (16384, 32768),
    "64k": (32768, 65536),
    "128k": (65536, 131072),
}


def _token_count(doc):
    messages = json.loads(doc["prompt"])
    return sum(
        len(ENCODING.encode(message["content"])) for message in messages
    ) + len(ENCODING.encode(doc["answer"]))


def _is_in_context_bin(doc, context_key):
    lo, hi = CONTEXT_BINS[context_key]
    token_count = _token_count(doc)
    if context_key == "8k":
        return lo <= token_count <= hi
    return lo < token_count <= hi


def _make_combined_filter(n_needles, context_key):
    return lambda dataset: dataset.filter(
        lambda doc: doc["n_needles"] == n_needles
        and _is_in_context_bin(doc, context_key)
    )


# Combined needle × context filters
filter_2needle_8k = _make_combined_filter(2, "8k")
filter_2needle_16k = _make_combined_filter(2, "16k")
filter_2needle_32k = _make_combined_filter(2, "32k")
filter_2needle_64k = _make_combined_filter(2, "64k")
filter_2needle_128k = _make_combined_filter(2, "128k")

filter_4needle_8k = _make_combined_filter(4, "8k")
filter_4needle_16k = _make_combined_filter(4, "16k")
filter_4needle_32k = _make_combined_filter(4, "32k")
filter_4needle_64k = _make_combined_filter(4, "64k")
filter_4needle_128k = _make_combined_filter(4, "128k")

filter_8needle_8k = _make_combined_filter(8, "8k")
filter_8needle_16k = _make_combined_filter(8, "16k")
filter_8needle_32k = _make_combined_filter(8, "32k")
filter_8needle_64k = _make_combined_filter(8, "64k")
filter_8needle_128k = _make_combined_filter(8, "128k")
