### Что делает этот MR?

> Добавьте **краткое** описание того, чего этот MR должен достичь, что в нем реализовано и зачем это нужно.
> Если изменение связано с upstream TorchTitan, приложите ссылки на upstream PR/commit/issue.

### Чеклист перед началом работы

- [ ] Заголовок MR оформлен в формате `[{modules}] {type}: {description}`.
  - `{modules}` могут включать:
    - `trainer`
    - `cfg`
    - `distributed`
    - `fsdp`
    - `tp`
    - `pp`
    - `cp`
    - `ep`
    - `model`
    - `data`
    - `optim`
    - `scheduler`
    - `ckpt`
    - `metrics`
    - `eval`
    - `perf`
    - `ci`
    - `build`
    - `doc`
    - `test`
    - `recipe`
    - `launch`
    - `misc`
  - Если MR затрагивает несколько модулей, разделяйте их через `,`, например: `[distributed, ckpt, doc]`.
  - `{type}` должен быть одним из значений, принимаемых CI-проверкой
    `validate:merge_request:title` (`MR_TITLE_PATTERN` в
    [.gitlab-ci.yml](/.gitlab-ci.yml)):
    - `feat`
    - `fix`
    - `chore`
    - `docs`
    - `style`
    - `refactor`
    - `test`
    - `perf`
    - `ci`
    - `build`
    - `revert`
  - Если MR ломает API, CLI, config schema, checkpoint compatibility или публичную сигнатуру, добавьте `[BREAKING]` в начало заголовка через пробел перед скобкой модулей (иначе CI-регекс отклонит заголовок).
  - Пример: `[BREAKING] [distributed, ckpt] feat: sharded optimizer resume path`

### Риски и совместимость

#### Затронутые области

- [ ] Training loop
- [ ] Config / registry / launch
- [ ] Distributed: FSDP / TP / PP / CP / EP
- [ ] Checkpoint save/load/resume
- [ ] Model definitions
- [ ] Optimizer / scheduler
- [ ] Dataloader / tokenizer
- [ ] Metrics / logging / profiling
- [ ] Eval
- [ ] CI / packaging
- [ ] Runtime behavior не меняется

### Тестирование

> Для изменений, которые нельзя проверить через CI, проведите проверку экспериментами и приложите результаты: графики обучения, метрики, ссылку на тикет/отчет, W&B/TensorBoard, логи запуска и т.д.

### API и пример использования

> Если MR меняет API/CLI/YAML config, опишите изменения и добавьте пример использования.
> Если изменений нет, укажите `N/A`.

Изменения API/CLI/YAML config:

- N/A

Пример использования:

- N/A

### Дизайн и изменения в коде

> Если MR сложный, опишите архитектуру на высоком уровне и перечислите конкретные изменения.

- ...

### Чеклист перед отправкой на ревью

> [!IMPORTANT]
> Проверьте все пункты ниже перед тем, как запрашивать ревью. Если пункт неприменим, отметьте его и объясните почему.

- [ ] Указать себя в assignee MR.
- [ ] Указать главного reviewer.
- [ ] CI/CD pipeline проходит или known failures описаны.
- [ ] В разделе «Тестирование» подробно описана проверка изменений.
- [ ] Добавлены или обновлены unit/integration/e2e tests. Если это невозможно, причина описана.
- [ ] Public API / CLI / config changes отражены в документации или примерах.
- [ ] Проведено авторевью ботом и все комментарии поправлены.
- [ ] Если менялись submodules/contrib dependencies:
  - [ ] submodule commit обновлен осознанно;
  - [ ] dependency уже смержена в соответствующем contrib repo;
  - [ ] указана ссылка на contrib MR/commit.
- [ ] Follow-up задачи заведены или перечислены ниже.

### Follow-ups

- ...
