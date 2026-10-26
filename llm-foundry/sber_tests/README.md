# How to Тесты

## Юнит-тесты

Юнит-тесты проверяют корректность работы отдельных компонентов системы в изоляции.

### Шаблон для юнит-тестов

**ВАЖНО**: Перед написанием нового юнит-теста ознакомьтесь с шаблоном:
```
sber_tests/unit/TEMPLATE_UNIT_TEST.py
```

Шаблон содержит:
- Примеры CPU и GPU тестов
- Правильное использование маркеров (@pytest.mark.unit, @pytest.mark.cpu, @pytest.mark.gpu)
- Использование общих фикстур из test_utils/test_fixtures.py
- Использование helpers из test_utils/test_helpers.py
- Примеры параметризации тестов
- Инструкции по запуску

### Структура юнит-тестов

```
sber_tests/unit/
├── TEMPLATE_UNIT_TEST.py    # ШАБЛОН - смотреть перед написанием теста
├── cpu/                     # CPU тесты
│   └── core/
│       ├── test_config_matrix.py
│       └── test_modality.py
└── gpu/
    ├── conftest.py             # сессионный seed, фикстуры из fixtures/
    ├── fixtures/
    │   ├── imports.py          # unified_module, dpo_modules, mla_modules
    │   ├── data_dpo.py         # dataclass DPOModules
    │   └── data_mla.py         # dataclass MLAModules
    ├── test_import.py
    ├── test_mla.py
    ├── test_mla_w_unified_module.py
    └── test_dpo.py
```

### Каталог `unit/gpu`

**`conftest.py`**

- для всей сессии `autouse`-фикстура задаёт `torch.manual_seed(42)` и при наличии CUDA — `torch.cuda.manual_seed_all(42)`.
- через `from .fixtures.imports import *` в область pytest попадают фикстуры из `fixtures/imports.py`.

**Фикстуры (`fixtures/imports.py`)**

| Фикстура | Назначение |
|----------|------------|
| `unified_module` | сессионная фабрика `(module_path, name) -> объект` на базе `importlib.import_module`. Может быть использован внутри любого теста без дополнительных фикстур. |
| `*_modules` | замороженный dataclass `**Modules` и необходимые импортируемые модули. Требует отдельного написания новой фикстуры под каждый новый тест. |

Типы `**Modules` объявлены в `fixtures/data_*.py`.

### Запуск юнит-тестов

```bash
# Все юнит-тесты
pytest sber_tests/unit/ -v -m unit

# Только CPU тесты
pytest sber_tests/unit/ -v -m "unit and cpu"

# Только GPU тесты
pytest sber_tests/unit/ -v -m "unit and gpu"

# С покрытием
pytest sber_tests/unit/ --cov=sber_tests.test_local_core
```

## Интеграционные тесты

Интеграционные тесты проверяют, что end-to-end пайплайн обучения на определённом
конфиге корректно отрабатывает.

Для проверки корректности тестовые прогоны сравниваются с golden прогонами, которые
заранее создаются и хранятся на диске. Это важно держать в уме, еслы вы хотите
создавать новые тест кейсы.

## Быстрый старт

### Запуск интеграционных тестов

Базовый запуск всех интеграционных тестов:
```bash
pytest sber_tests/integration/test_local_jobs.py -s -vvv
```

Запуск конкретной модальности:
```bash
pytest sber_tests/integration/test_local_jobs.py::TestVision -s -vvv
```

### Переменные окружения для интеграционных тестов

Поддерживается параллельный запуск тестов, для этого необходимо выставить переменную окружения

```bash
export TESTS_PARALLEL=1
```

Чтобы артефакты запусков не занимали слишком много места на диске, рекомендуется запускать с переменной окружения

```bash
export CLEAR_RUN_DIR=1
```

На данный момент некоторые модальности нестабильны при запуске на H100 с flash-attention v3. Рекомендуется выставить переменную окружения 

```bash
export USE_FA3=0
```


## Архитектура тестов

Все интеграционные тесты являются обычными pytest функциями, которые на вход
принимают `test_utils.entries.TestEntry`. Для параметризации этих функций
легче всего использовать декоратор `parametrize_with_config_matrix`.
Сами тестовые классы наследуются от `BaseJobTest`, который включает стандартные проверки:

- `test_model_completion` — успешное завершение обучения
- `test_model_checkpoints_exist` — наличие чекпоинтов
- `test_model_logs_exist` — наличие логов TensorBoard
- `test_model_approximate_weights_similarity` — сравнение весов с golden-чекпоинтом
- `test_logits` — сравнение логитов с референсными значениями

### Goldenы

Некоторые тесты требуют наличия golden-чекпоинтов для сравнения. Golden-джобы создаются вручную и хранятся в `ARTIFACT_DIR/golden/<modality>/<version>/`.
Для обновления golden-чекпоинтов установите или вручную укажите корректную версию голденов в sber_tests/test_local_core/version_manager.py
в переменной new_version, а также ARE_WE_UPDATING_GOLDENS = TRUE в sber_tests/test_local_core/modality.py

При запуске тестов система запросит новую версию. Формат версии: `YYYY-MM-DD_HH-MM-SS`

Некоторые модальности, подготовленные при `TESTS_PARALLEL=0`, могут быть нестабильны при `TESTS_PARALLEL=1` и наоборот. Голдены необходимо подготавливать при той переменной `TESTS_PARALLEL`, при которой они будут запускаться.

## Создание новых тестов
### 1. Простой тест

```python
from test_local_core.matrix import ConfigMatrix, test_config

@parametrize_with_config_matrix(
    **test_config(
        matrix=ConfigMatrix(
            icl_type=["icl-default.yaml"],
            fsdp_type=["fsdp.yaml", "gigafsdp.yaml"],
            tp_size=["tp-1.yaml", "tp-2.yaml"],
        ),
        test_name="TestMyFeature"
    )
)
class TestMyFeature(BaseJobTest):
    pass
```

### 2. Тест с кастомными проверками

```python
@parametrize_with_config_matrix(
    **test_config(
        matrix=ConfigMatrix(
            sft_type=["base_sft.yaml"],
            fsdp_type=["gigafsdp.yaml"],
        ),
        test_name="TestCustom",
        modality="SFT"
    )
)
class TestCustom(BaseJobTest):
    def test_custom_metric(self, entry: TestEntry):
        # Ваша кастомная проверка
        log_path = Path(entry.config.save_folder).parent / "stderr.txt"
        with open(log_path, "r") as f:
            content = f.read()
            assert "expected_pattern" in content
```

### 3. Исключение комбинаций

```python
@parametrize_with_config_matrix(
    **test_config(
        matrix=ConfigMatrix(
            fsdp_type=["fsdp.yaml", "gigafsdp.yaml"],
            tp_size=["tp-1.yaml", "tp-2.yaml"],
        ),
        exclude_subsets=[
            ("fsdp.yaml", "tp-2.yaml"),  # FSDP + TP-2 не поддерживается
        ],
        test_name="TestWithExclusions"
    )
)
class TestWithExclusions(BaseJobTest):
    pass
```

## Конфигурация тестов

### ConfigMatrix

`ConfigMatrix` — основной инструмент для создания матриц конфигураций. Он автоматически добавляет значения по умолчанию и поддерживает предустановленные паттерны.

#### Поля по умолчанию:
```python
DEFAULTS = {
    "base": ["base_gigar.yaml"],
    "fsdp_type": ["gigafsdp.yaml"],
    "tp_size": ["tp-1.yaml"],
    "sp_size": ["sp-1.yaml"],
}
```

#### GPU mapping по умолчанию:
```python
GPU_DEFAULTS = {
    "gigafsdp.yaml": 2,
    "fsdp.yaml": 2,
    "tp-2.yaml": 2,
    "sp-2.yaml": 2,
}
```

### Checkpoint cache pattern

Паттерн для кэширования чекпоинтов генерируется автоматически на основе полей матрицы:

```python
# По умолчанию исключаются поля icl_type
checkpoint_pattern = "{base},{fsdp_type},{tp_size},{sp_size}"

# Можно переопределить вручную:
test_config(
    matrix=ConfigMatrix(...),
    checkpoint_pattern="{base},{custom_field},{tp_size}"
)
```

### Параметры test_config

```python
test_config(
    matrix=ConfigMatrix(...),           # Матрица конфигураций
    test_name="TestName",               # Имя теста
    gpu_mapping={"tp-2.yaml": 2},      # GPU множители для конкретных конфигов
    exclude_subsets=[...],              # Исключаемые комбинации
    modality="SFT",                     # Модальность для группировки golden-джобов
    checkpoint_pattern="{...}",         # Кастомный паттерн чекпоинтов
    start_from_checkpoint=False,        # Продолжить с чекпоинта
    n_gpus=8,                          # Фиксированное число GPU
)
```

### Готовые пресеты ConfigMatrix

Для популярных сценариев можно создать конфигурацию:

```python
# Ring Attention
ConfigMatrix.ring_attention()
```
и использовать её, вместо прописывания конфигурации матрицы.

Для этого необходим в sber_tests/test_local_core/matrix.py определить @classmethod в ConfigMatrix и использовать его
```python
@classmethod
def ring_attention(cls):
    return cls(
        icl_type=["icl-default.yaml"],
        attention_type=["ring_attn.yaml"],
        tp_size=["tp-1.yaml", "tp-2.yaml"],
        sp_size=["sp-2.yaml"],
    )
```

Пример использования:
```python
@parametrize_with_config_matrix(
    **test_config(
        matrix=ConfigMatrix.vision(),
        test_name="TestVision",
        modality="VISION"
    )
)
class TestVision(BaseJobTest):
    pass
```         

## Модальности

### Добавление новой модальности

1. **Зарегистрируйте модальность** в `test_local_core/modality.py`:

```python
MODALITIES = {
    "GOLDEN": Modality(...),
    "MY_MODALITY": Modality(
        name="MY_MODALITY",
        version=DEFAULT_VERSION,
        allow_update=os.getenv("ALLOW_TO_UPDATE_MY_MODALITY", "false").lower() == "true"
    ),
}
```

2. **Добавьте маппинг в argfilter** в `test_utils/argfilter.py`:

```python
def get_modality_mapping():
    return {
        'CORE': [...],
        'MY_MODALITY': ['TestMyModality'],
    }
```

3. **Создайте тестовый класс**:

```python
@parametrize_with_config_matrix(
    **test_config(
        matrix=ConfigMatrix(...),
        test_name="TestMyModality",
        modality="MY_MODALITY"
    )
)
class TestMyModality(BaseJobTest):
    pass
```

4. **Добавьте модальность по тому же принципу в автотесты в** .gitlab-ci.yml

### Управление нестабильными тестами

Для нестабильных тестов используйте декоратор `@xfail_if_unstable`:

```python
class TestMyModality(BaseJobTest):
    @xfail_if_unstable("my_modality")
    def test_logits(self, entry: TestEntry):
        return super().test_logits(entry)
```

Включить xfail на модальность можно через переменную окружения:
```bash
export XFAIL_MY_MODALITY=1
```
не забудьте добавить xfail в .yml файл.

## Отладка конфигураций

Для проверки итоговой конфигурации теста используйте функцию `inspect`, лежащую в sber_tests/test_local_core/matrix.py:

```python
from test_local_core.matrix import inspect, test_config, ConfigMatrix

result = test_config(
    matrix=ConfigMatrix(
        fsdp_type=["fsdp.yaml", "gigafsdp.yaml"],
        tp_size=["tp-1.yaml", "tp-2.yaml"],
    ),
    test_name="TestDebug",
    exclude_subsets=[("fsdp.yaml", "tp-2.yaml")]
)

inspect(result)
```

Вывод:
```
============================================================
TEST CONFIGURATION
============================================================

Test Name: TestDebug

Matrix Fields:
  base                : ['base_gigar.yaml']
  fsdp_type           : ['fsdp.yaml', 'gigafsdp.yaml']
  tp_size             : ['tp-1.yaml', 'tp-2.yaml']
  sp_size             : ['sp-1.yaml']

Total Combinations: 4

Checkpoint Pattern: {base},{fsdp_type},{tp_size},{sp_size}

GPU Mapping:
  fsdp.yaml           : 2 GPUs
  gigafsdp.yaml       : 2 GPUs
  tp-2.yaml           : 2 GPUs

Excluded Combinations:
  - ('fsdp.yaml', 'tp-2.yaml')

============================================================
```

