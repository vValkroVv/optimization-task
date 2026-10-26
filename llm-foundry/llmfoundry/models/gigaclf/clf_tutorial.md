# Гайд по тренировке LLM-based классификаторов

---

## 1. Установка репозитория

Скачиваем `llm-foundry`:

```bash
git clone https://gitlab.ai.cloud.ru/gigallms/llm-foundry.git --recursive <адрес вашей папки>
```

Не забываем обновлять зависимости:

```bash
git pull --recurse-submodules && git submodule update --init --recursive
```

Переходим на ветку классификаторов (до мержа MR):

```bash
git checkout clf_v4_integrate
```

После мержа MR отдельная ветка не нужна — запускать можно из обычного репозитория (`main`) сразу после `clone` и обновления сабмодулей.

В случае конфликтов (или дальнейших проблем с запуском скриптов) можно использовать жёсткий сброс:

```bash
git checkout clf_v4_integrate
git fetch origin
git reset --hard origin/clf_v4_integrate
```

---

## 2. Подготовка данных

Пайплайн работает как для бинарной, так и для многоклассовой классификации.

---

### 2.1 Бинарная классификация

Если у вас уже готовы данные в формате `.mds` по схеме `{'input_ids', 'label'}` (токены, лейблы) — этот пункт можно пропустить.

Иначе воспользуйтесь скриптом `llm-foundry/scripts/data_prep/convert_clf_dataset_json_mds.py` — он ищет в `--input_dir` все `*.jsonl` файлы и конвертирует их в формат `.mds`.

В качестве `label` можно использовать:

- **Hard labels** — бинарные значения `0` или `1`
- **Soft labels** — вещественные значения в диапазоне `[0.0, 1.0]`

При запуске скрипта из актуального образа (или посылке джобы) дополнительные зависимости не требуются.

> **Образ:** актуальный базовый образ берётся из job-submitter — реестр `LLMF_BASE_IMAGES` в `contrib/job_submitter/job_submitter/jobs/image_registry.py` (по регионам).

Конвертацию в регионе SR008 на Jupyter Notebook можно запустить после активации окружения:
```bash
conda activate llm-foundry
```

**Пример запуска скрипта** (со случайным фиксированным сплитом на train и test):
```bash
python convert_clf_dataset_json_mds.py \
    --input_dir <папка с *.jsonl> \
    --output_dir <папка с *.mds> \
    --tokenizer .../models/Qwen3-0.6B \
    --text_field text \
    --label_field scaled_score \
    --max_length 4096 \
    --num_workers 8 \
    --train_ratio 0.9 \
    --seed 42
```

При необходимости скрипт можно изменить для добавления произвольной обработки токенов или лейблов:
```python
# Пример кастомной трансформации лейблов:
# raw_score = float(item.get(label_field, 0.0))
# label = min(raw_score * 1.2, 1.0)
```

---

### 2.2 Многоклассовая классификация

Если у вас уже готовы данные в формате `.mds` по схеме `{'input_ids', 'label'}` (токены, лейблы) — этот пункт можно пропустить.

В отличие от бинарного случая, для нескольких классов передаётся вектор лейблов в формате `[0., 0., 1., ..., 0., 0.]` dtype `float32`.

Иначе используйте скрипт `llm-foundry/scripts/data_prep/convert_clf_multilabels_json_mds.py` — он ищет в `--input_dir` все `*.jsonl` файлы и конвертирует их в формат `.mds`.

В качестве `label` можно использовать как **hard labels**, так и **soft labels**. Вся логика преобразования лейблов заключена в функции `score_to_onehot` вышеупомянутого скрипта. Она работает как с векторами, так и с числами (`num_classes` выставляется внутри функции).

**Примеры:**
```python
score_to_onehot(3)                                        # -> [0., 0., 1., 0., 0.]
score_to_onehot(2, zero_based=True)                       # -> [0., 0., 1., 0., 0.]
score_to_onehot([0., 1., 0., 0., 0.])                    # -> [0., 1., 0., 0., 0.]  (pass-through)
score_to_onehot([0., 1., 0.], num_classes=5)             # -> [0., 1., 0., 0., 0.]  (padded)
score_to_onehot([0., 1., 0., 0., 0., 0.], num_classes=5) # -> [0., 1., 0., 0., 0.]  (truncated)
```

Скрипт можно модифицировать для кастомных преобразований под свои задачи.

При запуске скрипта из актуального образа дополнительные зависимости не требуются.

> **Образ:** актуальный базовый образ берётся из job-submitter — реестр `LLMF_BASE_IMAGES` в `contrib/job_submitter/job_submitter/jobs/image_registry.py` (по регионам).

Конвертацию в регионе SR008 на Jupyter Notebook можно запустить после активации окружения:
```bash
conda activate llm-foundry
```

Пример запуска скрипта аналогичен бинарной классификации — отличается только логика преобразования лейблов.

---

## 3. Запуск обучения

Обучение запускается стандартным способом — либо через скрипт, либо через submit job.

**Через скрипт:** нужна среда с актуальным образом (см. реестр `LLMF_BASE_IMAGES` в job-submitter). Сам скрипт находится по адресу `llm-foundry/sber_scripts/infinite_jobs/run.sh` — в нём выставляются параметры и указывается ссылка на `.yaml` конфигурацию.

**Через submit job:** заполняется шаблон `llm-foundry/sber_scripts/submit_template.py` со ссылкой на `.yaml` конфиг. [Инструкции по заполнению шаблона](https://gitlab.ai.cloud.ru/gigallms/llm-foundry/-/tree/main/sber_scripts?ref_type=heads)

> **Важно:** перед отправкой не забываем выполнить `client_lib.update()`.

### Отправка обучения через job-submitter (рекомендуется)

Вместо кастомного submit-скрипта запуск классификатора отправляется через `job-submitter` — унифицированный сервис сабмита. Он сам резолвит регион, базовый образ (из реестра `LLMF_BASE_IMAGES`), allocation и env, а также **валидирует train-конфиг до сабмита**. Для classifier-трена используется класс `TrainJobLLMF` (`job_submitter.jobs.llmf.train`), который собирает команду `python scripts/train/train.py <config>` и добавляет в `PYTHONPATH` `llm-foundry` + сабмодули (`gigafsdp`, `composer`, `streaming`, `flash-linear-attention`).

Готовый скрипт запуска лежит в `llm-foundry/sber_scripts/jobs/examples/train_clf.py`. В нём достаточно поправить путь до конфига (`CONFIG_PATH`) и параметры запуска:

```python
TrainJobLLMF(
    user_login="nestroev",
    exp_name="mr_clf_exp",
    path_to_config=str(CONFIG_PATH),   # ваш gigaclf-конфиг
    region="SR008",
    n_workers=1,
    n_gpus=8,
    queue_name="shared",       # shared-очередь допускает только priority_class="low";
    priority_class="low",      # для medium/high используйте pretrain-ml / pretrain-data
    enable_auto_track=False,   # True -> уведомления в Mattermost
    dry_run=True,              # сначала валидация без реального сабмита
).submit()
```

Запуск (из окружения отправки, где стоят `client_lib` и `job_submitter`):
```bash
conda activate <ваше окружение отправки>
cd <адрес llm-foundry>
python sber_scripts/jobs/examples/train_clf.py     # dry_run=True -> проверка конфига
```
Если валидация прошла — поменяйте `dry_run=True` на `False` и запустите снова для реальной отправки на кластер.

> **Важно:** `TrainJobLLMF` определяет корень `llm-foundry` относительно **сабмодуля в репозитории** (`contrib/job_submitter`), поэтому используйте копию job-submitter именно из `contrib/`, а не отдельный клон. Другие примеры джоб (autometrics, infinite и т.д.) — в `sber_scripts/jobs/examples/`.

> Обучение выполняется в облаке на выбранном образе — GPU-стек на стороне ноутбука не задействуется, окружение отправки нужно лишь для сборки конфига и отправки джобы.

### Конфигурация `.yaml`

Конфиг для параметров тренировки во многом совпадает с конфигом тренировки моделей класса `gigar`. Ключевые **отличия**:

**1. Класс модели:**
```yaml
model:
  name: gigar_classifier
```

**2. Загрузчики данных** (ссылки обязательно должны указывать на `.mds`-данные вместе с лейблами):
```yaml
train_loader:
  name: text_classification

eval_loader:
  name: text_classification
```

**3. Параметр количества классов:**
```yaml
num_labels: 5
```

Параметр также продублирован для модели и загрузчиков данных.

В остальном можно добавлять параметры аналогично обычному обучению.

В качестве стартовой модели можно указывать как модели класса `gigar_classifier`, так и `gigar` (то есть можно стартовать с большинства LLM). Это касается и старта с чекпоинтов — если старт произошёл с модели `gigar`, в процессе обучения она автоматически переходит в класс `gigar_classifier`.

Поддерживаются как llama-подобные, так и `gigar`-подобные плотные (dense) LLM. В основном обучение проводилось на моделях **Qwen3-0.6B** и **Qwen3-8B**.

Дефолтные шаблонные конфиги лежат в подпапке `llm-foundry/sber_configs/gigaclf/`:

- `gigaclf_binary_config.yaml` — бинарная классификация (`num_labels: 2`);
- `gigaclf_multiclass_config.yaml` — многоклассовая (`num_labels > 2`).

Конфиги шаблонные — пути (модель, данные, `save_folder`) нужно заполнить под свой эксперимент. `eos_token_id` / `bos_token_id` заданы один раз (привязаны к семейству Qwen) и переиспользуются через интерполяцию.

---

## 4. Конвертация в HuggingFace-формат

После успешного обучения модели появляется шардированный чекпоинт с весами. Стандартный маппер не видит добавленные модули (например, классификационную голову), поэтому требуется отдельный маппер.

Скачиваем mappers:

```bash
git clone https://gitlab.ai.cloud.ru/gigallms/gigachat-utils.git
cd gigachat-utils
git checkout classifier_mapping
```

Документация по работе с mappers: https://gitlab.ai.cloud.ru/gigallms/gigachat-utils/-/tree/master/convertation?ref_type=heads

Выставляем нужные параметры в `gigachat-utils/convertation/scripts/submit_convertation_job.py`.

В `.yaml` конфиге `gigachat-utils/convertation/default.yaml` добавляем директории чекпоинта и выходной HF-модели. Нам нужен кастомный маппер `map_gigafsdp_to_hf_clf` в поле `mappers`.

Запускаем скрипт отправки джобы аналогично предыдущему разделу.

> **После получения HF-директории:** если планируется инференс через `sglang`, необходимо вручную поменять в `config.json` полученной модели значение параметра `"v_head_dim"` с `null` на `128` (или другое число, соответствующее размерности вашей модели).

---

## 5. Serving (инференс)

В данном разделе разбираем два способа инференса — через HuggingFace и через `sglang` (быстрее и предпочтительнее).
Рассматриваем бинарные классификаторы, для нескольких классов логика аналогичная.

В связи с архитектурой классификатора (см. раздел 6) требуется подавать токены с **left padding**, чтобы токен `<eos>` всегда оказывался последним в последовательности (если не используется специализированный dataloader). Также вместо генерации токенов необходимо извлекать вероятности или эмбеддинги.

---

### 5.1 Инференс через HuggingFace

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from tqdm import tqdm

model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

all_preds = []
all_probs = []  # вероятность класса 1

for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
    batch_texts = texts[i : i + batch_size]

    # 1. Токенизируем БЕЗ специальных токенов и с запасом места под <eos>
    inputs = tokenizer(
        batch_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length - 1,   # оставляем слот под <eos>
        add_special_tokens=False,
    )

    # 2. Добавляем токен <eos> в конец каждой последовательности
    eos_tensor = torch.full(
        (inputs["input_ids"].size(0), 1),
        tokenizer.eos_token_id,
        dtype=torch.long,
    )
    inputs["input_ids"] = torch.cat([inputs["input_ids"], eos_tensor], dim=1)

    # 3. Обновляем маску внимания: добавляем 1 для нового токена <eos>
    inputs["attention_mask"] = torch.cat(
        [inputs["attention_mask"], torch.ones_like(eos_tensor)],
        dim=1,
    )

    # 4. Переносим на устройство и делаем forward pass
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)

    all_preds.extend(preds.cpu().tolist())
    all_probs.extend(probs[:, 1].cpu().tolist())  # вероятность класса 1
```

---

### 5.2 Инференс через sglang

на примере бинарного классификатора

Можно использовать актуальный образ (см. реестр `LLMF_BASE_IMAGES` в job-submitter), установив `sglang` поверх:

```bash
pip install sglang
```

Разворачиваем сервер:

```bash
python -m sglang.launch_server \
  --model-path .../models/8B_Qwen_4Gb_classifier_with_logits_v3_hf \
  --port 30000 \
  --host 0.0.0.0 \
  --is-embedding \
  --tp 1 \
  --dp 8
```

Логика запроса к серверу:

```python
import json
import requests
from transformers import AutoTokenizer

model_path = "<путь к модели>"
file_path = "*.jsonl"
server_url = "http://localhost:30000/v1/classify"

tokenizer = AutoTokenizer.from_pretrained(model_path)
max_len = 4096 - 2  # оставляем запас в 1 токен + 1 слот под <eos>

with open(file_path, "r") as f:
    first_line = f.readline()
sample = json.loads(first_line)

original_text = sample["text"]
true_label = sample["score"]

# Усекаем текст и добавляем <eos> в конец
tokens = tokenizer.encode(original_text, add_special_tokens=False, max_length=max_len)
truncated_text = tokenizer.decode(tokens, skip_special_tokens=True) + tokenizer.eos_token

# Отправляем запрос
response = requests.post(server_url, json={"model": "default", "input": [truncated_text]})
response.raise_for_status()
data = response.json()

# Извлекаем предсказание (первый элемент data['data'])
pred_raw = data["data"][0]["label"]   # например, "LABEL_0"
```

После этого вычисляем и записываем нужные метрики.

---

## 6*. Архитектура LLM-based классификатора

Архитектурно мы наследуем класс `LlamaForCausalLM` и вместо проекции внутреннего состояния на словарь берём классификационную голову, которая извлекает координаты последнего токена `<eos>`:

```python
# Классификационная голова: hidden_size -> num_labels
self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

# hidden_states shape: [batch_size, seq_len, hidden_size]
hidden_states = outputs[0]

# Поскольку батч выровнен и заканчивается на <eos>, берём последний токен
pooled_logits = hidden_states[:, -1, :]

# Классификационная голова: logits shape: [batch_size, num_labels]
logits = self.score(pooled_logits)
```


Лосс считается через `nn.CrossEntropyLoss()` — для soft/hard labels и любом количестве классов.

Большинство фич, связанных с генерацией текста, было удалено. Помимо самой модели, были также внесены изменения в `composer`, `dataloader` (datacollator с left padding), `streaming class` и другие компоненты.

Вся классификаторная логика вынесена в отдельный пакет `llm-foundry/llmfoundry/models/gigaclf/` (моделинг — `modelling_gigaclf.py`), чтобы не пересекаться с плотной моделью `gigar/modelling_gigar.py`.

На место LLM подойдёт любая базовая модель. **Очень важно** сдвигать токен `<eos>` в конец предложения — поэтому в тренировочном конфиге он явно указывается для семейства моделей Qwen.

В качестве классификационных метрик, помимо функции потерь `nn.CrossEntropyLoss()`, были добавлены soft-label метрики (принимают мягкие/скалярные таргеты, в отличие от одноимённых метрик в `composer`). Они живут в submodule `composer` — `composer/metrics/classification.py` (импортируются как `from composer.metrics.classification import ...`):

- **CrossEntropy по soft-таргетам:** `MultiClassCrossEntropy` (совпадает с train-лоссом;
  для бинарного случая скалярный таргет разворачивается в `[p_neg, p_pos]`) —
  см. `composer/metrics/classification.py`

- **Бинарная классификация:** `BinaryAccuracy`, `BinaryPrecision`, `BinaryRecall`,
  `SoftBinaryF1Score`, `BinaryAUCROC` —
  см. `composer/metrics/classification.py`
  (`SoftBinaryF1Score` назван с префиксом `Soft`, чтобы не конфликтовать с
  `composer.metrics.nlp.BinaryF1Score`).

- **Многоклассовая классификация:** `MultiClassCrossEntropy`, `MultiClassOverallAccuracy`,
  `MultiClassWeightedF1` —
  см. `composer/metrics/classification.py`

В этом файле можно добавлять свои кастомные метрики и включать их в evals в
`llm-foundry/llmfoundry/models/gigaclf/modelling_gigaclf.py`.

---