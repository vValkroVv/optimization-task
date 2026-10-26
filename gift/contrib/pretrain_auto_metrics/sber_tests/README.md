# sber_tests

Тесты организованы по структуре исходного пакета `node_runner`:

- `unit/` - тесты модулей верхнего уровня из `node_runner/src` (`arg_parser`, `logger`, `config_manager`, `convertation_manager`, `subprocess_*`, ...).
- `unit/inference` и `unit/inference/sglang` - тесты менеджера инференса и `sglang`-подсистемы.
- `unit/metrics` и `unit/metrics/evaluators` - менеджер метрик, tensorboard и evaluators.
- `unit/obm_proxy` - тесты proxy-менеджера и парсинга proxy-ответов.
- `test_support/` - служебные test helpers и контекстные утилиты.
- `test_data/` - тестовые данные и моки (конфиги, mock metrics, mock checkpoints).

## Как запускать

Локально:

```bash
pytest -m unit \
  --strict-markers \
  --durations=10 \
  --cov=node_runner \
  --cov-branch \
  --cov-report=term-missing:skip-covered \
  sber_tests
```

CI-CD:

```bash
pytest -m unit \
  --strict-markers \
  --cov=node_runner \
  --cov-branch \
  --cov-report=term-missing:skip-covered \
  --cov-report=xml:coverage.xml \
  --junitxml=pytest-report.xml \
  --cov-fail-under=100 \
  sber_tests
```

E2E гоняются только ночным пайплайном на `dev` (джоба на кластере, см. [карту CI/CD](../docs/ci_pipeline_map.md#2-ночной-пайплайн-на-dev)). Команда внутри джобы:

```bash
pytest -m e2e --strict-markers --maxfail=1 -q -rA \
  --junitxml=pytest-e2e.xml \
  sber_tests/e2e
```


## Как добавлять новые тесты

1. Найди модуль в `node_runner/src`.
2. Создай зеркальный тест в соответствующей подпапке `sber_tests/unit/...` по аналогии с `node_runner/src/...`.
3. Используй фикстуры из `sber_tests/conftest.py` и хелперы из `sber_tests/test_support/`.
4. Ставь `pytest.mark.unit` (через `pytestmark` на уровне модуля).
5. Группируй тесты по логике в `Test...`-классы (например `TestParseArguments`, `TestObmProxyManager`).
6. Предпочитай unit-тесты без внешних сетей и сервисов; внешние вызовы мокай через `monkeypatch`.
