"""Генератор посабджектных конфигов mmlu_cloze.

Соответствие «сабджект → категория», `dataset_name`, `task_alias` и текст
`description` берутся из существующих файлов `lm_eval/tasks/mmlu/default/mmlu_*.yaml`,
чтобы cloze-семейство не расходилось с MCQ-семейством при обновлении апстрима.
Из `description` удаляется подстрока `multiple choice ` — единственное отличие
текста от исходного.

Запуск из корня репозитория, без аргументов:
    python lm_eval/tasks/mmlu_cloze/_generate_configs.py
"""

import json
from pathlib import Path

import yaml


SOURCE_DIR = Path("lm_eval/tasks/mmlu/default")
TARGET_DIR = Path("lm_eval/tasks/mmlu_cloze")
MCQ_MARKER = "multiple choice "


def main() -> None:
    sources = sorted(SOURCE_DIR.glob("mmlu_*.yaml"))
    if len(sources) != 57:
        raise SystemExit(
            f"ожидалось 57 исходных конфигов в {SOURCE_DIR}, найдено {len(sources)}"
        )

    for source in sources:
        config = yaml.safe_load(source.read_text(encoding="utf-8"))
        description = config["description"]
        if MCQ_MARKER not in description:
            raise SystemExit(
                f"{source}: в description нет подстроки {MCQ_MARKER!r}, "
                "апстримные файлы отличаются от ожидаемых"
            )

        subject = config["dataset_name"]
        category = config["tag"].removeprefix("mmlu_").removesuffix("_tasks")
        target = TARGET_DIR / f"mmlu_cloze_{subject}.yaml"
        lines = [
            ("dataset_name", subject),
            ("description", description.replace(MCQ_MARKER, "")),
            ("tag", f"mmlu_cloze_{category}_tasks"),
            ("include", "_cloze_template_yaml"),
            ("task", f"mmlu_cloze_{subject}"),
            ("task_alias", config["task_alias"]),
        ]
        target.write_text(
            "".join(f"{json.dumps(k)}: {json.dumps(v)}\n" for k, v in lines),
            encoding="utf-8",
        )

    print(f"записано {len(sources)} конфигов в {TARGET_DIR}")


if __name__ == "__main__":
    main()
