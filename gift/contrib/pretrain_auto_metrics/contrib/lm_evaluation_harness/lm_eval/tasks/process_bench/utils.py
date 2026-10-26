import re
from typing import Dict, List

import datasets


def load_dataset(**kwargs):
    split = kwargs.get("split", "gsm8k")
    dataset = datasets.load_dataset("Qwen/ProcessBench", split=split)
    return {"train": dataset}


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        problem = doc["problem"]
        steps = doc["steps"]
        label = int(doc["label"])
        
        steps_text = "\n\n".join([f"<paragraph_{i}>\n{step}" for i, step in enumerate(steps)])
        
        content = f"""The following is a math problem and a solution (split into paragraphs, enclosed with tags and indexed from 0):

[Math Problem]

{problem}

[Solution]

{steps_text}

Your task is to review and critique the solution paragraph by paragraph. Once you identify an error in a paragraph, return the index of the paragraph where the earliest error occurs. Otherwise, return the index of -1 (which typically denotes "not found").

Please put your final answer (i.e., the index) in \\boxed{{}}."""
        
        return {
            "content": content,
            "answer": label,
        }

    return dataset.map(_process_doc)


def extract_last_boxed_integer(response: str) -> int:
    regex_boxed = re.compile(r"\\boxed\{([^}]*)\}")
    matches = regex_boxed.findall(response)
    
    if not matches:
        return None
    
    parsed = matches[-1].strip()
    
    try:
        return int(parsed)
    except ValueError:
        return None


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    if not results:
        return {"exact_match": 0}
    
    response = results[0]
    predicted_answer = extract_last_boxed_integer(response)
    expected_answer = doc["answer"]
    
    if predicted_answer is not None and predicted_answer == expected_answer:
        return {"exact_match": 1}
    else:
        return {"exact_match": 0}

