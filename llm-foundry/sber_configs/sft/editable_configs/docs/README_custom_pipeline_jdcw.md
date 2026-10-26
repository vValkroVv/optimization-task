# Custom Pipeline JobDispatcherCallbackWeb

## Описание

Кастомная конфигурация пайплайна для job-dispatcher позволяет создавать гибкие многоэтапные процессы обработки чекпоинтов.

**ВАЖНО:** При добавлении кастомного пайплайна, основной (jobs) будет проигнорирован!

## Обязательные поля для каждой джобы

- `script`: команда/скрипт для выполнения
- `job_desc`: описание джобы для логирования

## Автоматические подстановки

- `{CHECKPOINT_PATH}`: полный путь до чекпоинта (например: /path/to/model/ep0_ba1500/ep0-ba1500-tp_rank0)
- `{CHECKPOINT_DIR}`: директория чекпоинта (например: /path/to/model/ep0_ba1500)
- `{WANDB_ENTITY}`: entity из текущего wandb run
- `{WANDB_PROJECT}`: project из текущего wandb run
- `{WANDB_RUN_ID}`: run ID из текущего wandb run

## Дополнительные поля (опциональные)

- `base_image`: Docker образ для запуска джобы (default: cr.ai.cloud.ru/.../job-dispatcher:v1.1)
- `llmf_path`: путь до llm-foundry (для конвертации, автоматически добавляет в PYTHONPATH)
- `workspace`: рабочее пространство (wm-train, giga-vision, giga-audio, giga-nlp-2)
- `instance_type`: тип инстанса (переопределяет дефолтный для региона)
- `job_type`: тип джобы (default: binary)
- `n_workers`: количество воркеров (default: 1)
- `pytorch_use_env`: использовать pytorch окружение (default: true)
- `processes_per_worker`: процессов на воркер (default: 1)
- `max_retry`: максимальное количество перезапусков (default: 10)
- `env_variables`: дополнительные переменные окружения (dict, будут добавлены к дефолтным / перезаписаны)
- `proceed_on_failure`: продолжать пайплайн при неуспешном создании/выполнении джобы (true/false, default: false)

## Управление флагом proceed_on_failure

- **Уровень пайплайна**: применяется ко всем стейджам по умолчанию
- **Уровень стейджа**: переопределяет настройку пайплайна для всех джоб в стейдже
- **Уровень джобы**: переопределяет настройки стейджа для конкретной джобы
- **По умолчанию**: false (остановка пайплайна при неуспешном создании/выполнении джобы)

## Автоматический мониторинг Child Jobs

- После успешного завершения джобы система проверяет логи на наличие child jobs
- Паттерн поиска: 'Job "job_name" created' в логах родительской джобы
- Child джобы автоматически отслеживаются с наследованием proceed_on_failure от родителя
- Пример: джоба "sam" может создать child джобу "sam - child"
- Child джобы учитываются при принятии решения о продолжении пайплайна

## Структура пайплайна

```yaml
custom_pipeline:
  proceed_on_failure: false                    # Глобальная настройка для всего пайплайна. По умолчанию false

  stages:                                      # Список стейджей (выполняются последовательно)

    # Stage 1: Базовый стейдж без переопределения
    - jobs:
        - script: "command1"
          job_desc: "description1"

    # Stage 2: Некритичный стейдж
    - proceed_on_failure: true                 # Переопределение для всех джоб в стейдже
      jobs:                                    # Джобы внутри стейджа (параллельно)

        - script: "command3"
          job_desc: "description3"             # Унаследует proceed_on_failure: true

        - script: "command4"
          job_desc: "description4"
          proceed_on_failure: false            # Переопределение на уровне джобы. Если именно эта джоба завершится неуспешно, весь пайплайн остановится!

    # Stage 3: Критический стейдж
    - proceed_on_failure: false                # Остановка пайплайна при ошибке в любой из джоб в рамках стейджа
      jobs:
        - script: "command5"
          job_desc: "critical_job"
```

## Полный пример конфигурации

```yaml
custom_pipeline:
  proceed_on_failure: false # На уровне всего пайплайна
  stages:
    # Stage 1: Convertation
    - jobs:
      - script: >-
          python /home/jovyan/ramakarov/llm-foundry/contrib/gigachat-utils/convertation/map-fsdp-to-fsdp.py
          --input-checkpoint-path {CHECKPOINT_PATH}
          --mappers map_gigafsdp_to_hf
          --input-tp-size 1
          --tokenizer-path /home/jovyan/models/gigar/gigachat4-lite-9b/32k_context_rope_theta_2800000/gigachat4-lite-9b-32k_context-rope_theta_2800000-ba10000/artifacts/tokenizer
          --output-checkpoint-path {CHECKPOINT_DIR}/hf_chkpt
        base_image: cr.ai.cloud.ru/19d74875-55e7-4f05-92d7-5bff1b43b8a0/llm-foundry-train:2025_02_03_commit277c3722
        job_desc: "Job_dispatcher_convertation {CHECKPOINT_PATH}"
        llmf_path: "/home/jovyan/ramakarov/llm-foundry" # используется во внутренней логике подстановки PYTHONPATH для работы со свежим image конвертации. Можно убрать поля llmf_path и base_image, тогда будет использоваться старый образ

    # Stage 2: SAM Evaluation
    - jobs:
      - script: >-
          python /workspace/ramakarov/service-for-automatic-measurement/submit_measurement.py
          --user_name ramakarov
          --model_class sft
          --size lite
          --pass_judge
          --pass_obb
          --chatmlv3
          --wandb_entity {WANDB_ENTITY}
          --wandb_project {WANDB_PROJECT}
          --wandb_run_id {WANDB_RUN_ID}
          --model_path {CHECKPOINT_DIR}
          --artifacts_alongside
        job_desc: sam
        proceed_on_failure: true # На уровне отдельной джобы в стейдже

    # Stage 3: S3 Sync
    - proceed_on_failure: false # На уровне стейджа
      jobs:
        - script: >-
            python /workspace/dabrazhnik/job-dispatcher-metrics/gigachat-utils/job-dispatcher-utils/auto_s3.py
            "{CHECKPOINT_DIR}"
            --target_root_dir gigachat_checkpoints/nfs2
            --bucket s3://gigaeye-pretrain-checkpoints
            --sleep_interval 180
            --signal_dir /home/jovyan/alert-signals/s3-upload
          job_desc: "S3 sync for {CHECKPOINT_PATH}"
```

## Примечания

- Дефолтный workspace: wm-train
- Можно указывать любые дополнительные поля для гибкости
- Всегда проверяйте конфигурацию перед запуском!