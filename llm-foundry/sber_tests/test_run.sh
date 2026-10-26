#!/bin/bash

# Директория для сохранения логов
export LOG_DIR="test_logs_golden_runs/$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p $LOG_DIR

ROOT_DIR="$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")"
echo $ROOT_DIR
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/contrib/composer:${ROOT_DIR}/contrib/streaming:${ROOT_DIR}/contrib/gigafsdp:${ROOT_DIR}/contrib/gigachat-utils/convertation/"

python -c 'import streaming; streaming.base.util.clean_stale_shared_memory()'
pkill -9 -f python

# Запуск тестов и сохранение логов в файл
export TESTS_PARALLEL=1
export CLEAR_RUN_DIR=1
export USE_FA3=0

LOG_FILE="$LOG_DIR/test_run_$i.log"
echo "Запуск тестов №$i..."
pytest integration/test_local_jobs.py -s -v > $LOG_FILE 2>&1

unset CLEAR_RUN_DIR
unset USE_FA3

echo "Все тесты завершены. Логи сохранены в директорию $LOG_DIR."
