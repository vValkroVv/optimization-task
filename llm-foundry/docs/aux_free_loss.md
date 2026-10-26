# Aux‑Free Load Balancing (bias update for MoE routers)

## Теория

**Идея:** вместо классического auxiliary loss на распределение токенов по экспертам мы онлайн‑обновляем **bias** у gate так, чтобы недогруженные эксперты получали бонус, а перегруженные — штраф. Aux-free не меняет целевую функцию — градиенты считаются только по основному лоссу, а балансировка делается через обновление bias.


### Обозначения

Пусть для MoE‑слоя с $E$ экспертами и каждого эксперта $e \in \{1,\dots,E\}$:
- $T_e$ — число токенов, ушедших в эксперта $e$ за шаг (или окно шагов), **после** all‑reduce по девайсам.
- Средняя нагрузка:
  $$
  \overline{T} \equiv \frac{1}{E} \sum_{e=1}^{E} T_e
  $$
- Максимальная нагрузка и ожидаемая нагрузка:
  $$
  T_{\max} \equiv \max_{e} T_e, \qquad T_{\mathrm{exp}} \equiv \overline{T}
  $$
- Максимальное отклонение (max violation):
  $$
  \mathrm{max\_vio} \equiv \frac{T_{\max}-T_{\mathrm{exp}}}{T_{\mathrm{exp}}+\varepsilon}
  $$
- $\gamma$ — скорость обновления (update rate).
- $b_e$ — bias эксперта $e$. Обновление:
  $$
  b_e \leftarrow b_e + \Delta_e
  $$

### Стратегии обновления

Индикатор события обозначим как $\mathbb{1}[\cdot] \in \{0,1\}$.

1. **baseline**
   $$
   \Delta_e \;=\; \gamma \cdot \mathrm{sign}\!\big(\,\overline{T} - T_e\,\big)
   $$

2. **indicator**
   $$
   \Delta_e \;=\; \gamma \cdot \mathbb{1}\!\big[\,T_e < \overline{T}\,\big]
   $$

3. **threshold** (c зазором $\beta \ge 0$)
   $$
   \Delta_e \;=\; \gamma \cdot \Big(\,
   \mathbb{1}[\,T_e < \overline{T}\,] \;-\;
   \mathbb{1}[\,T_e > \overline{T}\,(1+\beta)\,]
   \Big)
   $$

4. **adaptive** (threshold + адаптивный масштаб)
   $$
   s \;=\; \mathrm{clip}\!\big(1 + \alpha \cdot \mathrm{max\_vio},\; 1,\; S_{\max}\big), \qquad
   \Delta_e \;=\; \gamma \cdot s \cdot \Big(\,
   \mathbb{1}[\,T_e < \overline{T}\,] \;-\;
   \mathbb{1}[\,T_e > \overline{T}\,(1+\beta)\,]
   \Big)
   $$

---

## Практика

### Конфигурация модели

```yaml
model:
  config_overrides:
    aux_loss_free: true
    moe_router_bias_update_rate: 0.001      # γ
    aux_loss_free_strategy: adaptive         # baseline|indicator|threshold|adaptive
    adaptive_beta: 0.05                      # β
    adaptive_alpha: 1.0                      # α
    adaptive_scale: 5.0                      # S_max
```

### Метрики

- $\mathrm{max\_vio}$ по слоям: `moe_maxvio/layer_i`
- Доли недо/перегруженных: `moe_under_frac/layer_i`, `moe_over_frac/layer_i`
- Статистики bias: `moe_bias_{min,max,mean,std}/layer_i` и `..._updated_...`
- Адаптивный масштаб: `moe_scale/layer_i`
- Роутинг‑метрики мониторинга: `old_moe_routing_{max_vio,max_load,min_load}/layer_i`, `moe_routing_top_k_*`, и т.д.
