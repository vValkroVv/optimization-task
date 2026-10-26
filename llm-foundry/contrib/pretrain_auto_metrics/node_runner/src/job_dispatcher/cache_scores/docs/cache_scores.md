# cache_scores: сжатый файл скоров для bootstrap-CI

После прогона OBM PAM кладёт в `metrics_dir` файл `cache_scores.jsonl.zst`: одна строка на
сэмпл, только поля, по которым можно посчитать скор. Сырые кеши весят сотни МБ из-за
промптов и генераций, этот файл - сотни КБ.

Далее с его помощью `job-dispatcher` (JD) считает доверительные интервалы

## Пример строки cache_scores.jsonl.zst

```json
{"benchmark": "arxiv_math", "metric_name": "accuracy", "uid": "17", "matched": [true, false]}
```

`benchmark` - имя файла `cache/<benchmark>.jsonl`.

`metric_name` - по нему JD выбирает, как считать скор сэмпла для доверительных
интервалов. Это имя основной метрики бенчмарка, берётся из поля `metric.name` в строке
этого бенчмарка в `all_metrics.jsonl`, вот как она выглядит:

```json
{"name": "aime_2026", "metric": {"name": "accuracy/max@32", "value": 0.27,
 "extra_metrics": [{"name": "accuracy/avg@32", "value": 0.04}]}, "size": 30}
```

В `cache_scores` идёт только `metric.name`.

JD считает скор по `metric_name`, а не по имени бенчмарка: все бенчмарки с одинаковым
`metric_name` считаются одинаково.

| `metric_name` | Что JD берёт за скор сэмпла |
|---|---|
| `accuracy` | `matched`: bool или число -> 0/1, список -> доля верных |
| `accuracy/max@32` | 1, если хоть одна из попыток верна |
| `mean` | поле `score` |
| `f1` | поле `f1_score` |
| любая другая | пропуск, сэмпл не участвует в интервалах |

Копируются только те поля, которые перечислены в
[`OBM_CACHE_SCORES_FIELDS`](../obm.py).

Ограниченный список полей нужен ради размера: JD качает этот файл целиком на каждый
расчёт доверительных интервалов. Полные файлы с кешами лежат рядом на S3 в папке `cache/`.

## Что происходит после прогона

Всё в `after_run` (`evaluators/obm.py`): собрали файл -> залили `metrics_dir` на S3 ->
отправили JD путь `cache_s3_path`.

При повторном замере старые бенчмарки не теряются: билдер скачивает файл, уже лежащий на
S3 по тому же пути, и мерджит его с новым по бенчмаркам. Пересчитанные бенчмарки
заменяются новыми строками, отсутствующие в текущем `cache/` переносятся из старого файла.

## Добавление нового бенчмарка на примере OBM

1. Поле, в котором лежит скор, есть в [`OBM_CACHE_SCORES_FIELDS`](../obm.py); если нет -
   добавьте его.
2. `metric_name` бенчмарка есть в таблице выше - тогда JD уже умеет его считать и делать
   ничего не нужно. Если такого `metric_name` в таблице нет, правила для него в JD тоже
   нет: заведите его, иначе интервалов по этому бенчмарку не будет.

## Добавление доверительных интервалов для шутера

1. `cache_scores/<shooter>.py` - подкласс `CacheScoresBuilder` с двумя методами:

   ```python
   class MyCacheScoresBuilder(CacheScoresBuilder):
       def __init__(self, metrics_dir):
           # merge_key - поле, по которому сливаемся со старым файлом; без него слияния нет
           super().__init__(CacheScoresSpec(raw_fields=MY_FIELDS, merge_key="benchmark"))
           self.metrics_dir = Path(metrics_dir)

       def iter_source_files(self):  # какие файлы читать
           return sorted((self.metrics_dir / "cache").glob("*.jsonl"))

       def derived_fields(self, source_file):  # что добавить в каждую строку
           return {"benchmark": source_file.stem}
   ```

   Слияние со старым файлом включается через `existing_rows()` - в OBM его подставляет
   `existing_rows_loader`, который тянет объект с S3.

2. В эвалуаторе вызовите `builder.build(Path(self.metrics_dir))` в `after_run` (пример -
   `_build_cache_scores` в `evaluators/obm.py`) и передайте путь в JD.

3. Заведите правила скоринга в JD