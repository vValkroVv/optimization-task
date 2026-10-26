import os
import re
import string
import unicodedata
from collections import Counter
from pathlib import Path


DATASET_PATH = "xinrongzhang2022/InfiniteBench"
INFINITEBENCH_CONTEXT_LENGTHS = [4096, 8192, 16384, 32768, 65536, 131072]


def _format_context_length(context_length):
    if context_length % 1024 == 0:
        return f"{context_length // 1024}k"
    return str(context_length)


def _resolve_precomputed_paths(data_dir, data_name, max_seq_lengths):
    if data_dir is None:
        return []
    prefix = f"infinitebench_{data_name}"

    search_dirs = [Path(data_dir)]
    if not Path(data_dir).is_absolute():
        search_dirs.append(Path(__file__).parent / data_dir)

    lengths = max_seq_lengths or []
    paths = []
    for length in lengths:
        filename = f"{prefix}_{_format_context_length(length)}.jsonl"
        for directory in search_dirs:
            candidate = directory / filename
            if candidate.exists():
                paths.append((length, candidate))
                break
    return paths


def _load_precomputed_dataset(paths):
    import datasets

    per_length = []
    for length, path in paths:
        dataset = datasets.load_dataset(
            "json",
            data_files={"test": str(path)},
            split="test",
        )
        if "infinitebench_max_length" not in dataset.column_names:
            dataset = dataset.add_column(
                "infinitebench_max_length",
                [length] * len(dataset),
            )
        per_length.append(dataset)

    if len(per_length) == 1:
        return per_length[0]
    return datasets.concatenate_datasets(per_length)


def load_infinitebench_dataset(
    data_name: str,
    dataset_path: str = DATASET_PATH,
    data_dir: str = None,
    max_seq_lengths: list = None,
    **kwargs: object,
) -> dict[str, object]:
    import datasets

    _maybe_enable_substring_match(kwargs)

    precomputed_paths = _resolve_precomputed_paths(data_dir, data_name, max_seq_lengths)
    if precomputed_paths:
        dataset = _load_precomputed_dataset(precomputed_paths)
    else:
        dataset = datasets.load_dataset(
            dataset_path,
            split=data_name,
            features=_features(),
        )
    if "infinitebench_task" in dataset.column_names:
        return {"test": dataset}
    return {
        "test": dataset.add_column(
            "infinitebench_task",
            [data_name] * len(dataset),
        )
    }


def doc_to_text(doc: dict) -> str:
    task = doc["infinitebench_task"]
    context = doc["context"]
    question = doc["input"]

    if task == "passkey":
        return (
            "There is an important info hidden inside a lot of irrelevant text. "
            "Find it and memorize it. I will quiz you about the important "
            f"information.\n\n{context}\n\n{question}\n\nThe pass key is"
        )
    if task == "number_string":
        return (
            "There is an important info hidden inside a lot of irrelevant text. "
            "Find it. I will quiz you about the important information there.\n\n"
            f"{context}\n\n{question}\n\nThe sequence of digits is"
        )
    if task == "kv_retrieval":
        return (
            "Extract the value corresponding to the specified key in the JSON "
            f"object below.\n\n{context}\n\n{question}"
        )
    if task == "longbook_sum_eng":
        return f"Summarize the book below.\n\n{context}\n\nSummary:"
    if task == "longbook_choice_eng":
        return (
            f"Read the book and answer the question.\n\n{context}\n\n"
            f"Question: {question}\n{_format_options(doc)}\n\n"
            "The letter of the correct answer is"
        )
    if task == "longbook_qa_eng":
        return (
            "Read the book and answer the question. Be very concise in your "
            f"answer.\n\n{context}\n\nQuestion: {question}\nAnswer:"
        )
    if task == "longbook_qa_chn":
        return (
            "\u9605\u8bfb\u4ee5\u4e0b\u4e66\u7c4d\u7136\u540e\u56de\u7b54"
            f"\u95ee\u9898\u3002\n\n{context}\n\n"
            f"\u95ee\u9898\uff1a{question}\n\u7b54\u6848\uff1a"
        )
    if task == "math_find":
        return _math_find_prompt(context, question)
    if task == "math_calc":
        return (
            "Let us calculate the intermediate values of an expression.\n\n"
            "Expression: 1 + 3 + 4\nValues: [1, 4, 8]\n\n"
            "Expression: 8 - 3 + 2 - 4\nValues: [8, 5, 7, 3]\n\n"
            f"Expression: {context}\nValues:"
        )
    if task == "code_run":
        func_call = _code_call(question)
        func = func_call.split("(")[0]
        return (
            f"There is a function called {func} in the following Python code.\n\n"
            f"{context}\n\nPlease compute the exact value of {func_call}. "
            f"The value of {func_call} is"
        )
    if task == "code_debug":
        return (
            "Following is a Python code where exactly one of the functions/methods "
            f"has a deliberate error that makes it crash.\n\n{context}\n\n"
            f"Options:\n{_format_options(doc)}\n\nThe correct option is:"
        )
    if task == "longdialogue_qa_eng":
        return (
            "Below is a dialogue script where one random occurrence of a character "
            'name is replaced with "$$MASK$$", and you should try to guess who '
            f"that character is.\n\n{context}\n\nThe name that has been replaced "
            "with $$MASK$$ is likely"
        )

    raise ValueError(f"Unknown InfiniteBench task: {task}")


def doc_to_target(doc: dict) -> str:
    task = doc["infinitebench_task"]
    if task in ("code_debug", "longbook_choice_eng"):
        label = _choice_label(doc)
        return label[1] if len(label) > 1 else _first_answer(doc)
    return _first_answer(doc)


def doc_to_choice(doc: dict) -> list[str]:
    return ["A", "B", "C", "D"]


def process_results(doc: dict, results: list[str]) -> dict[str, float]:
    prediction = results[0]
    task = doc["infinitebench_task"]

    if task == "kv_retrieval":
        score = _score_kv_retrieval(prediction, _first_answer(doc))
    elif task == "passkey":
        score = _score_first_int(prediction, _first_answer(doc))
    elif task == "number_string":
        score = _score_first_int(prediction, _first_answer(doc))
    elif task == "code_run":
        score = _score_code_run(prediction, _first_answer(doc))
    elif task == "code_debug":
        score = _score_code_debug(prediction, _choice_label(doc))
    elif task == "longdialogue_qa_eng":
        score = _score_longdialogue(prediction, _answers(doc))
    elif task == "longbook_choice_eng":
        score = _score_longbook_choice(prediction, _choice_label(doc))
    elif task == "longbook_qa_eng":
        score = (
            _qa_substring_score(prediction, _answers(doc))
            if _substring_match_enabled()
            else _qa_f1_score(prediction, _answers(doc))
        )
    elif task == "longbook_sum_eng":
        score = _rouge_lsum_score(prediction, _first_answer(doc))
    elif task == "longbook_qa_chn":
        score = _qa_f1_zh_score(prediction, _answers(doc))
    elif task == "math_find":
        score = _score_math_find(prediction, _first_answer(doc))
    elif task == "math_calc":
        score = _score_math_calc(prediction, _answers(doc))
    else:
        raise ValueError(f"Unknown InfiniteBench task: {task}")

    return _format_metric(doc, float(score))


def process_multiple_choice_results(
    doc: dict,
    results: list[tuple[float, bool]],
) -> dict[str, float]:
    choices = doc_to_choice(doc)
    if len(results) != len(choices):
        raise ValueError("Expected one loglikelihood result per answer choice.")

    target = doc_to_target(doc)
    if target not in choices:
        return _format_metric(doc, 0.0)

    lls, _ = zip(*results)
    prediction = max(range(len(choices)), key=lambda index: lls[index])
    return _format_metric(doc, float(choices[prediction] == target))


def _format_metric(doc: dict, score: float) -> dict[str, float]:
    length = doc.get("infinitebench_max_length")
    if length is None:
        return {"infinitebench_score": score}
    return {f"infinitebench_score_{int(length)}": score}


def _features() -> object:
    import datasets

    return datasets.Features(
        {
            "id": datasets.Value("int64"),
            "context": datasets.Value("string"),
            "input": datasets.Value("string"),
            "answer": datasets.Sequence(datasets.Value("string")),
            "options": datasets.Sequence(datasets.Value("string")),
        }
    )


def _answers(doc: dict) -> list[str]:
    answer = doc.get("answer", [])
    if isinstance(answer, str):
        return [answer]
    return [str(item) for item in answer]


def _first_answer(doc: dict) -> str:
    answers = _answers(doc)
    return answers[0] if answers else ""


def _choice_label(doc: dict) -> list[str]:
    answers = _answers(doc)
    if len(answers) == 2 and answers[1] in "ABCD":
        return answers

    options = [str(option) for option in doc.get("options", [])]
    if answers and answers[0] in options:
        return [answers[0], chr(ord("A") + options.index(answers[0]))]
    return answers


def _format_options(doc: dict) -> str:
    options = [str(option) for option in doc.get("options", [])]
    if len(options) != 4:
        raise ValueError("InfiniteBench multiple-choice tasks require four options.")
    return "\n".join(f"{letter}. {option}" for letter, option in zip("ABCD", options))


def _math_find_prompt(context: str, question: str) -> str:
    match = re.findall(r"The .+ of", question)
    if not match:
        raise ValueError(f"Cannot find the target number in {question}")
    target_number = match[0].lower()[:-3]
    prefix = f"What is {target_number} in the following list?"
    return f"{prefix}\n\n{context}\n\n{question}"


def _code_call(question: str) -> str:
    match = re.findall(r"func_[0-9]+\(\-?[0-9]+\)", question)
    if not match:
        raise ValueError(f"Cannot find a function call in {question}")
    return match[0]


def _normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def remove_punc(value: str) -> str:
        return "".join(char for char in value if char not in string.punctuation)

    return " ".join(remove_articles(remove_punc(text.lower())).split())


def _normalize_zh_answer(text: str) -> str:
    normalized = []
    for char in text.lower():
        if char.isspace() or unicodedata.category(char).startswith("P"):
            continue
        normalized.append(char)
    return "".join(normalized)


def _f1_score(prediction: list[str], ground_truth: list[str]) -> float:
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction)
    recall = num_same / len(ground_truth)
    return 2 * precision * recall / (precision + recall)


def _substring_match_enabled() -> bool:
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


def _qa_substring_score(prediction: str, answers: list[str]) -> float:
    """RULER-style: 1.0 if any normalized gold answer is a substring of the prediction."""
    normalized_prediction = _normalize_answer(prediction)
    for answer in answers:
        normalized_answer = _normalize_answer(answer)
        if normalized_answer and normalized_answer in normalized_prediction:
            return 1.0
    return 0.0


def _qa_f1_score(prediction: str, answers: list[str]) -> float:
    score = 0.0
    prediction_tokens = _normalize_answer(prediction).split()
    for answer in answers:
        answer_tokens = _normalize_answer(answer).split()
        score = max(score, _f1_score(prediction_tokens, answer_tokens))
    return score


def _qa_f1_zh_score(prediction: str, answers: list[str]) -> float:
    score = 0.0
    prediction_tokens = list(_normalize_zh_answer(prediction))
    for answer in answers:
        answer_tokens = list(_normalize_zh_answer(answer))
        score = max(score, _f1_score(prediction_tokens, answer_tokens))
    return score


def _rouge_lsum_score(prediction: str, answer: str) -> float:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeLsum"], use_stemmer=False)
    return scorer.score(target=answer, prediction=prediction)["rougeLsum"].fmeasure


def _score_kv_retrieval(prediction: str, answer: str) -> bool:
    for char in ["\n", ":", '"', "'", ".", ",", "?", "!", "{", "}"]:
        prediction = prediction.replace(char, " ")
    return answer in prediction.split()


def _score_first_int(prediction: str, answer: str) -> bool:
    for item in re.split("[^0-9]", prediction):
        if item:
            return item == answer
    return False


def _score_code_run(prediction: str, answer: str) -> bool:
    prediction = prediction.strip()
    for char in ["\n", ".", "`", "'", '"', ":"]:
        prediction = prediction.replace(char, " ")
    words = prediction.split()
    if not words:
        return False
    try:
        return int(words[-1]) == int(answer)
    except ValueError:
        return False


def _score_code_debug(prediction: str, label: list[str]) -> bool:
    if len(label) < 2:
        return False

    prediction = prediction.strip()
    function_name = label[0]
    label_choice = label[1]
    match = re.search(r"\b[A-J]\b(?!.*\b[A-J]\b)", prediction)
    if match and match.group(0) == label_choice:
        return True

    for char in ["\n", "`", "'", '"', "-", "*", "Option", "option"]:
        prediction = prediction.replace(char, " ")
    while "  " in prediction:
        prediction = prediction.replace("  ", " ")
    if prediction.startswith(label_choice) or prediction.startswith(function_name):
        return True

    for prefix in ["answer is:", "is:", "answer:", "correct option is:"]:
        idx = prediction.find(prefix)
        if idx == -1:
            continue
        if len(prediction) < idx + len(prefix) + 1:
            return False
        rest = prediction[idx + len(prefix) + 1 :]
        return rest.startswith(label_choice) or rest.startswith(function_name)
    return False


def _score_math_find(prediction: str, answer: str) -> bool:
    match = re.search(r"\d+\.\d+|\d+", prediction)
    if match is None:
        return False

    predicted = match.group(0).strip()
    try:
        if "." in answer:
            return float(predicted) == float(answer)
        return int(predicted) == int(answer)
    except ValueError:
        return predicted == answer


def _score_longdialogue(prediction: str, answers: list[str]) -> bool:
    prediction = prediction.strip().upper()
    return any(answer.upper() in prediction for answer in answers)


def _score_longbook_choice(prediction: str, label: list[str]) -> bool:
    prediction = prediction.strip()
    match = re.search(r"\b[A-D]\b(?!.*\b[A-D]\b)", prediction)
    if match and match.group(0) in label:
        return True
    if not prediction:
        return False
    if prediction[0] in "ABCD":
        return prediction[0] in label
    if prediction in label:
        return True

    for char in ["\n", '"', "'", ".", ",", "?", "!", "{", "}"]:
        prediction = prediction.replace(char, " ")
    while "  " in prediction:
        prediction = prediction.replace("  ", " ")
    for prefix in ["answer is:", "answer:", "answer is", "option is"]:
        idx = prediction.find(prefix)
        if idx == -1:
            continue
        if len(prediction) < idx + len(prefix) + 1:
            return False
        rest = prediction[idx + len(prefix) + 1 :]
        return any(rest.startswith(item) for item in label)

    for word in prediction.split():
        if word in "ABCD":
            return word in label
    return False


def _score_math_calc(prediction: str, answers: list[str]) -> float:
    prediction_numbers = [
        int(item) for item in re.split("[^0-9]", prediction) if item != ""
    ]
    answer_numbers = _answer_numbers(answers)
    if not answer_numbers:
        return 0.0

    count = 0
    for index, answer in enumerate(answer_numbers):
        if index >= len(prediction_numbers):
            break
        if answer != prediction_numbers[index]:
            break
        count += 1
    return count / len(answer_numbers)


def _answer_numbers(answers: list[str]) -> list[int]:
    numbers = []
    for answer in answers:
        for match in re.findall(r"-?\d+", answer):
            numbers.append(int(match))
    return numbers
