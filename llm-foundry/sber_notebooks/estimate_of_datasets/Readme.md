## 1. Начало работы
На данный момент (30.05.25) работает джоба с развернутым GigaChat-Max на sr004 `lm-mpi-job-56c5855f-ed20-4f79-a2e5-1b012193572a`

<details>
<summary>Если джобы с GigaChat нет</summary>
<br>
Если джобы с GigaChat нет, то нужно произвести следующую последовательность действий:

1) Скачать репозиторий
```
git clone --recurse-submodules https://gitlab.ai.cloud.ru/gigallms/service-for-automatic-measurement.git
```
2) Зайти в проект
```
cd service-for-automatic-measurement

```
3) Выполнить скрипт `deploy_model.py`, где `MODEL_PATH=/home/jovyan/smirnov_da/merging/models/base_models/Max-28.2`

```
python deploy_model.py --model_path MODEL_PATH --tokenizer_path MODEL_PATH --user_name USER_NAME
```
4) Зайти в созданную джобу `ssh-master JOB_ID`
```
ssh-master () {
   job_id=$1
   ssh user@${job_id}-mpimaster-0.${job_id} -p 2222 -i /home/jovyan/.ssh/ai0001071-03990
}
```
5) Внутри джобы активировать окружение
```
conda activate /home/jovyan/.mlspace/envs/llm-python-server
```

Убедиться, что всё работает правильно, можно с помощью скрипта:

```
curl --location 'http://127.0.0.1:8081/chat' \
--header 'Content-Type: application/json' \
--data '{
    "messages": [
        {
            "role": "user",
            "content": "как дела"
        }
    ],
    "model": "llm05-30-2025_step1"
}'
```

6) Внутри ноутбука замените старый id джобы на новый в переменной `JOB_ID`.

</details>


## 2. Запуск ноутбука

Откройте Jupyter Notebook (notebook.ipynb) и выполните следующие действия:
* Скопируйте ноутбук к себе в рабочую папку.
* Выберите датасет.
* Заполните поле USERNAME.
* Запустите остальные ячейки.

*Примечание*: Возможно, потребуется установить дополнительные зависимости.

## 3. Оценка текста

Вам будет предложено оценить текст по шкале от 0 до 2, где:
* 0 — плохой текст,
* 1 — удовлетворительный,
* 2 — отличный текст.

Также необходимо оставить структурированный комментарий, указав:
* Плюсы текста,
* Минусы текста.

**Важно!** Перед сохранением оценки и комментария убедитесь, что:

1) Получены ответы от моделей,
2) В ячейке статуса отображается "Ожидание оценки пользователя".

Если сохранить ответ, не дожидаясь завершения работы моделей, программа может работать некорректно.
