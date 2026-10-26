# noqa
"""
Take in a YAML, and output all "other" splits with this YAML
"""

import argparse
import logging
from pathlib import Path

from tqdm import tqdm

from utils import SUBJECTS


eval_logger = logging.getLogger(__name__)


CATEGORIES = {
    "abstract_algebra": "stem",
    "anatomy": "stem",
    "astronomy": "stem",
    "business_ethics": "other",
    "clinical_knowledge": "other",
    "college_biology": "stem",
    "college_chemistry": "stem",
    "college_computer_science": "stem",
    "college_mathematics": "stem",
    "college_medicine": "other",
    "college_physics": "stem",
    "computer_security": "stem",
    "conceptual_physics": "stem",
    "econometrics": "social_sciences",
    "electrical_engineering": "stem",
    "elementary_mathematics": "stem",
    "formal_logic": "humanities",
    "global_facts": "other",
    "high_school_biology": "stem",
    "high_school_chemistry": "stem",
    "high_school_computer_science": "stem",
    "high_school_european_history": "humanities",
    "high_school_geography": "social_sciences",
    "high_school_government_and_politics": "social_sciences",
    "high_school_macroeconomics": "social_sciences",
    "high_school_mathematics": "stem",
    "high_school_microeconomics": "social_sciences",
    "high_school_physics": "stem",
    "high_school_psychology": "social_sciences",
    "high_school_statistics": "stem",
    "high_school_us_history": "humanities",
    "high_school_world_history": "humanities",
    "human_aging": "other",
    "human_sexuality": "social_sciences",
    "international_law": "humanities",
    "jurisprudence": "humanities",
    "logical_fallacies": "humanities",
    "machine_learning": "stem",
    "management": "other",
    "marketing": "other",
    "medical_genetics": "other",
    "miscellaneous": "other",
    "moral_disputes": "humanities",
    "moral_scenarios": "humanities",
    "nutrition": "other",
    "philosophy": "humanities",
    "prehistory": "humanities",
    "professional_accounting": "other",
    "professional_law": "humanities",
    "professional_medicine": "other",
    "professional_psychology": "social_sciences",
    "public_relations": "social_sciences",
    "security_studies": "social_sciences",
    "sociology": "social_sciences",
    "us_foreign_policy": "social_sciences",
    "virology": "other",
    "world_religions": "humanities",
}

AGGREGATE_METRIC_LIST = "\n".join(
    [
        "aggregate_metric_list:",
        "  - metric: acc",
        "    weight_by_size: True",
        "  - metric: nll",
        "    weight_by_size: True",
        "  - metric: nll_cond",
        "    weight_by_size: True",
    ]
)


def quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


def subject_yaml(base_yaml: str, name: str, subject: str, category: str) -> str:
    description = (
        f"Ниже приведены вопросы с множественным выбором (с ответами) по {subject}\n\n"
    )
    return "\n".join(
        [
            f"\"include\": {quote(base_yaml)}",
            f"process_docs: !function utils.process_{name}",
            f"\"tag\": {quote(f'rummlu_{category}_tasks')}",
            f"\"task\": {quote(f'rummlu_{name}')}",
            f"\"task_alias\": {quote(subject)}",
            f"\"description\": {quote(description)}",
        ]
    )


def category_yaml(category: str) -> str:
    return "\n".join(
        [
            f"group: rummlu_{category}",
            f"group_alias: {category}",
            "task:",
            f"  - rummlu_{category}_tasks",
            AGGREGATE_METRIC_LIST,
            "metadata:",
            "  version: 2",
        ]
    )


def group_yaml(categories: list[str]) -> str:
    return "\n".join(
        ["group: rummlu", "task:"]
        + [f"  - rummlu_{category}" for category in categories]
        + [AGGREGATE_METRIC_LIST, "metadata:", "  version: 2"]
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_yaml_path", default="_default_template_yaml")
    parser.add_argument("--save_prefix_path", default="rummlu")
    return parser.parse_args()


def write_file(path: Path, content: str) -> None:
    path.write_text(content + "\n", encoding="utf-8")


if __name__ == "__main__":
    args = parse_args()

    base_yaml_name = Path(args.base_yaml_path).name

    ALL_CATEGORIES = []
    for name, subject in tqdm(SUBJECTS.items()):
        category = CATEGORIES[name]
        if category not in ALL_CATEGORIES:
            ALL_CATEGORIES.append(category)

        file_save_path = Path(f"{args.save_prefix_path}_{name}.yaml")
        eval_logger.info(f"Saving yaml for subset {name} to {file_save_path}")
        write_file(file_save_path, subject_yaml(base_yaml_name, name, subject, category))

    for category in ALL_CATEGORIES:
        file_save_path = Path(f"_{args.save_prefix_path}_{category}.yaml")
        eval_logger.info(f"Saving group config to {file_save_path}")
        write_file(file_save_path, category_yaml(category))

    file_save_path = Path(f"_{args.save_prefix_path}.yaml")
    eval_logger.info(f"Saving benchmark config to {file_save_path}")
    write_file(file_save_path, group_yaml(ALL_CATEGORIES))
