# Миграция структуры данных

## Новая структура

Все файлы `question.jsonl` теперь хранятся в плоской структуре в `data/arenas/`:

```
data/
├── arenas/
│   ├── arena-hard-v0.1.jsonl
│   ├── arena-hard-v2.0.jsonl
│   ├── mt_bench.jsonl
│   ├── mt_bench-ru.jsonl
│   ├── arena-general-v0.1.jsonl
│   └── ...
└── model_data/
    └── ...
```

## Миграция

### 1. Просмотр изменений (dry-run)

```bash
python3 scripts/migrate_questions.py --dry-run
```

### 2. Выполнение миграции

```bash
python3 scripts/migrate_questions.py
```

Скрипт:
- Найдет все `question.jsonl` файлы в `data/arenas/`
- Переместит их в `data/arenas/{arena_name}.jsonl`
- Сохранит все данные

## Использование в конфигах

### Старый способ (все еще работает):

```yaml
dataset:
  name: "arena-hard-v0.1"
  question_file: "data/arenas/arena-hard-auto/arena-hard-v0.1/question.jsonl"
```

### Новый способ (рекомендуется):

```yaml
dataset:
  name: "arena-hard-v0.1"  # Файл автоматически ищется как data/arenas/arena-hard-v0.1.jsonl
  question_file: null
```

Если `question_file: null` или не указан, система автоматически ищет `data/arenas/{name}.jsonl`.

## Обратная совместимость

Код поддерживает старую структуру для обратной совместимости. Если файл не найден в новой структуре, будет поиск в старой.

