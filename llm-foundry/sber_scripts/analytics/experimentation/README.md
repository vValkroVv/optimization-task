# Эксперименты обучения

Эта папка содержит простой шаблон для воспроизводимых pretrain-экспериментов.
Новый эксперимент начинается с копирования `example_exp/`.

В скопированной папке:

- `baseline_patch.yaml` описывает изменение, общее для всех запусков;
- `submit_exp.py` выбирает baseline из `sber_configs/baseline_registry/`, задаёт
  инфраструктуру и перечисляет отдельные запуски через `Setup`.

Каждый `Setup` превращается в одну `TrainJobLLMF`. Промежуточные training YAML не
создаются: patch и параметры setup передаются в `train.py` как OmegaConf overrides.
Если один параметр указан и в patch, и в setup, для этой job используется значение
из setup. Значение `null` из patch передаётся как `null`.

Использование:

```bash
cp -R sber_scripts/analytics/experimentation/example_exp \
  sber_scripts/analytics/experimentation/my_exp
# отредактировать my_exp/baseline_patch.yaml и my_exp/submit_exp.py
python sber_scripts/analytics/experimentation/my_exp/submit_exp.py
```

Перед запуском укажите настоящий baseline, имя эксперимента, пользователя и нужные
`Setup`. Ожидается, что baseline будет выбран из `sber_configs/baseline_registry/`;
корректность путей остаётся ответственностью автора эксперимента.
