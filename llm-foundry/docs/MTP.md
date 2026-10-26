# Multi-Token Prediction (MTP)

## Теория

Пусть базовая модель выдаёт скрытые представления $\mathbf{h}^0_i \in \mathbb{R}^d$ для токена $t_i$.
Последовательный MTP добавляет $D$ «глубин» предсказаний: $k=1,\dots,D$.
На каждой глубине $k$ мы смешиваем прошлое представление $\mathbf{h}^{k-1}_i$ и embedding целевого токена $t_{i+k}$,
после чего пропускаем через **собственный** трансформер-блок $\mathrm{TRM}_k$ (который может быть **MoE** или **Dense**)
и используем **общую** выходную голову $\mathrm{OutHead}(\cdot)$.

### Смешивание скрытого состояния и target-эмбеддинга
  $$
    \mathbf{h'}^{\,k}_i
    =
    M_k
    \left[
    \mathrm{RMSNorm}\!\big(\mathbf{h}^{k-1}_i\big)\;;\;
    \mathrm{RMSNorm}\!\big(\mathrm{Emb}(t_{i+k})\big)
    \right],
    \tag{1}
  $$
где $M_k \in \mathbb{R}^{d\times 2d}$ — проекция, $[\cdot;\cdot]$ — конкатенация по признаковой оси.
При $k=1$ берём $\mathbf{h}^{0}_i$ из **основной** модели.

### Трансформер-блок на глубине $k$ (MoE **или** Dense)
  $$
    \mathbf{h}^{k}_{1:T-k}
    =
    \mathrm{TRM}_k\!\big(\mathbf{h'}^{\,k}_{1:T-k}\big),
    \tag{2}
  $$
где $T$ — длина входной последовательности. В коде это соответствует циклическому сдвигу
входных эмбеддингов на один шаг на каждой глубине ($k$) с маскированием хвостового токена.

### Выходная голова
  $$
    P^{k}_{i+k+1}
    =
    \mathrm{OutHead}\!\big(\mathbf{h}^{k}_i\big)
    \in \mathbb{R}^{V},
    \tag{3}
  $$
где $V$ — размер словаря. Вероятности вычисляются как $\mathrm{Softmax}(\cdot)$ над логитами выходной головы.

### Обучающая цель
На глубине $k$ оптимизируем кросс-энтропию по валидным позициям (последний токен замаскирован $\texttt{ignore\_index}$):
  $$
    \mathcal{L}^{k}_{\mathrm{MTP}}
    =
    -\frac{1}{T}\sum_{i=2+k}^{T+1}\log P^{k}_{i}\!\big[t_i\big].
    \tag{4}
  $$
Итоговый MTP-лосс добавляется к основному:
  $$
    \mathcal{L}_{\mathrm{total}}
    =
    \mathcal{L}_{\mathrm{main}}
    +
    \underbrace{\frac{\lambda}{D}\sum_{k=1}^{D}\mathcal{L}^{k}_{\mathrm{MTP}}}_{\mathcal{L}_{\mathrm{MTP}}}\,,
    \quad
    \lambda=\texttt{mtp\_loss\_weight},\; D=\texttt{mtp\_predictor\_num}.
    \tag{5}
  $$

---

## Код

- `MTPAttnAdapter`: реализует (1) — RMSNorm двух входов, конкатенация, линейная проекция $M_k$,
  поддержка TP/async-TP. Может выполняться **внутри** `self_attn` (для MoE) или **до** `input_layernorm` (для Dense).
- `LlamaDecoderMTPLayer`: оборачивает `self_attn` через адаптер и содержит свой `decoder_layer`:
  - **MoE-вариант**: `BLOCK_CLASS_REGISTRY["LlamaMixDecoderLayer"]`, `exec_inside_self_attn=True`.
  - **Dense-вариант**: `BLOCK_CLASS_REGISTRY["LlamaDecoderLayer"]`, `exec_inside_self_attn=False`.
- `LlamaDecoderMTPBlock`: создаёт $D$ слоёв `LlamaDecoderMTPLayer`, на каждом шаге
  делает `input_embeds = roll(input_embeds, -1)` и обнуляет хвост,
  затем применяет post-norm из `mtp_norms`.

Инференс отключён: MTP используется только в обучении.

---

## Практика

```yaml
model:
  config_overrides:
    use_mtp: true
    mtp_predictor_num: 2     # D: число глубин MTP
    mtp_loss_weight: 0.3     # λ: вес MTP-части лосса
    mtp_loss_detach: false   # true: не пускать MTP-градиент в decoder/embeddings
```

- Выбор **MoE** vs **Dense** задаётся типом блока в конфиге (`BLOCK_CLASS_REGISTRY`).
- При `mtp_loss_detach: true` hidden state основной модели и token embeddings
  передаются в MTP-ветку через `detach()`. Параметры самой MTP-ветки продолжают
  обучаться; общий `lm_head` остаётся общим.
