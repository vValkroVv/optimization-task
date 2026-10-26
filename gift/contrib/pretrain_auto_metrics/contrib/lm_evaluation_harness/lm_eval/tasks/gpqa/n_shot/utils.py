import random
import re

import datasets


GPQA_CHOICE_SEED = 0


def preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def _ordered_choices(doc, rng: random.Random):
    labeled_choices = [
        ("incorrect_1", preprocess(doc["Incorrect Answer 1"])),
        ("incorrect_2", preprocess(doc["Incorrect Answer 2"])),
        ("incorrect_3", preprocess(doc["Incorrect Answer 3"])),
        ("correct", preprocess(doc["Correct Answer"])),
    ]
    rng.shuffle(labeled_choices)
    return labeled_choices


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    rng = random.Random(GPQA_CHOICE_SEED)

    def _process_doc(doc):
        ordered = _ordered_choices(doc, rng)
        choices = [choice for _label, choice in ordered]
        correct_answer_index = [label for label, _choice in ordered].index("correct")

        out_doc = {
            "choice1": choices[0],
            "choice2": choices[1],
            "choice3": choices[2],
            "choice4": choices[3],
            "answer": f"({chr(65 + correct_answer_index)})",
        }
        return out_doc

    return dataset.map(_process_doc)
