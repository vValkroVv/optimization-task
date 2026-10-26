# sber_tests

Тесты организованы по структуре исходного пакета `job_submitter`:

- `unit/asserts` - проверки валидаторов и assertor-классов.
- `unit/core` - тесты конфиг-лоадера, backend-слоя и шаблонов.
- `unit/jobs` - тесты job-классов, включая `llmf` и `verl`.

## Как запускать

Локально:

```bash
pytest -m unit \
  --strict-markers \
  --durations=10 \
  --cov=job_submitter \
  --cov-branch \
  --cov-report=term-missing:skip-covered \
  sber_tests
```

CI-CD:

```bash
pytest -m unit \
  --strict-markers \
  --cov=job_submitter \
  --cov-report=term-missing:skip-covered \
  --cov-report=xml:coverage.xml \
  --junitxml=pytest-report.xml \
  --cov-fail-under=80 \
  --cov-branch \
  sber_tests
```

## Как добавлять новые тесты

1. Найди пакет/модуль в `job_submitter`.
2. Создай зеркальный файл в `sber_tests/unit/...`.
3. Используй фикстуры из `sber_tests/conftest.py` и фабрики из `sber_tests/factories.py`.
4. Группируй тесты в классы по смысловым блокам, например `TestBaseJobResolvers`, `TestCloudBackend`.
5. Предпочитай unit-тесты без внешних сетей и реальных кластеров; все внешние вызовы мокай через `monkeypatch`.
