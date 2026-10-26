# Baseline registry

`baseline_registry/` содержит полные конфиги обучения, сгруппированные по семейству
модели. Версия рецепта и backend входят в имя файла:

```text
baseline_registry/
├── giga/
│   ├── l8-1_5b-a-0_3b-4_0-papaya.yaml
│   └── l16-11_3b-a-1_1b-4_0-papaya.yaml
├── mhc/
│   └── {3b-3_0b-a-0_6b,9b-9_2b-a-1_7b,27b-27_0b-a-4_1b}-4_0-papaya.yaml
└── diff/
    ├── 3_5-3_0.yaml
    └── 4_0-3_5.yaml
```

Файлы в `diff/` — частичные YAML-патчи между соседними версиями. Они рекурсивно
накладываются на полный baseline через OmegaConf; списки заменяются целиком, а
`null` становится значением `null`. Исторические полные конфиги 3.0/3.5 в registry
не хранятся; патчи описывают изменения рецепта, а новые эксперименты выбирают
существующий полный baseline 4.0.

Для нового эксперимента скопируйте шаблон
[`sber_scripts/analytics/experimentation/example_exp/`](../sber_scripts/analytics/experimentation/example_exp/),
выберите полный baseline и опишите общий патч в `baseline_patch.yaml`. Параметры
конкретного запуска задаются в `Setup.config_overrides`; они имеют приоритет над
общим patch.

# Model params


## gigar 1b
```

model_path: /home/jovyan/models/llama2-7b-fp16-TheBloke-extended_tok-cc_10k
model:
	config_overrides:
		hidden_size: 2048
		intermediate_size: 5504
		num_hidden_layers: 16
		num_attention_heads: 16
		num_key_value_heads: 16
```


## gigar 3b

```
model_path: /home/jovyan/models/llama2-7b-fp16-TheBloke-extended_tok-cc_10k
model:
	config_overrides:
	    hidden_size: 2560
	    intermediate_size: 6880
	    num_hidden_layers: 32
	    num_attention_heads: 20
	    num_key_value_heads: 20
```


## gigar 7b

```
model_path: /home/jovyan/models/llama2-7b-fp16-TheBloke-extended_tok-cc_10k
```


## gigamix 3b

```
model_path: /home/jovyan/models/llama2-7b-fp16-TheBloke-extended_tok-cc_10k
model:
  name: giga_mix_causal_lm
  init_device: meta
  pretrained: false
  pretrained_model_name_or_path: /home/jovyan/ndermolaev/models/Qwen2.5-3B-digits-simple-remapped/
  config_overrides:
    vocab_size: 128256
    max_position_embeddings: 4096
    rope_theta: 10000.0
    rope_scaling: null
    eos_token_id: 2
    tie_word_embeddings: false
    hidden_size: 1280
    dense_intermediate_size: 896
    num_hidden_layers: 12
    attention_type: LlamaLatentAttention
    num_attention_heads: 16
    num_key_value_heads: 16
    attention_bias: false
    attention_dropout: 0.0
    attention_hidden_size: null
    hidden_act: silu
    initializer_range: 0.006
    rms_norm_eps: 1.0e-20
    apply_torch_compile_to_projections: false
    use_custom_rotary_kernel: false
    qk_nope_head_dim: 128
    qk_rope_head_dim: 64
    v_head_dim: 128
    kv_lora_rank: 512
    q_lora_rank: 1536
    num_krope_heads: 1
    moe_impl: DeepseekGMMMoeBlock
    first_k_dense_replace: 1
    intermediate_size: 896
    n_shared_experts: 2
    num_routed_experts: 64
    num_experts_per_tok: 6
    norm_topk_prob: true
    aux_loss_free: true
    router_aux_loss_coef: 0.0001
    moe_router_bias_update_rate: 0.001
    aux_loss_free_strategy: adaptive
    drop_by_group_capacity: true
    moe_router_num_groups: null
    moe_router_group_topk: null
    scoring_func: sigmoid
    moe_router_routed_scaling_factor: 1
    use_cache: false
    tp_size: 1
    use_liger: false
    fused_mlp: false
    fused_mlp_checkpoint_lvl: 3
    activation_checkpoint_layers_num: null
    loss_inplace_backward: true
    delete_logits: true
    use_mtp: false
    mtp_predictor_num: 1
    mtp_loss_weight: 0.3
    init_type: deepseek
    deepseek_init_std: 0.006
```


## Training debug section

All train configs can now include an optional `debug:` block for short validation runs.

Keep `debug.use_debug: false` in committed configs. Turn it on only when you need a short smoke run instead of full training.

When to use it:
- cluster / launcher smoke: validate distributed init, one forward-backward step, and clean shutdown;
- data wiring smoke: validate one real train stream without paying the cost of the full dataset mix;
- metric wiring smoke: keep only one small ICL task and skip the heavier eval paths.

Behavior notes:
- `debug.startup_speedups` is applied first;
- `debug.overrides` is merged after the speedups;
- top-level `overrides` still wins after that;
- if the base config already has `train_loader.dataset.epoch_size`, prefer overriding `epoch_size` for dummy smoke runs instead of adding `dataset_size`.

Supported shape:

```yaml
debug:
  use_debug: false
  startup_speedups:
    use_one_train_dataloader: true
    train_stream_name: null
    disable_runtime_benchmark_callbacks: true
    disable_eval_loader: true
    icl_tasks_metrics: null
    disable_all_evaluators: false
    load_checkpoints: false
    fast_model_init: true
  overrides:
    max_duration: 1ba
    save_interval: 10ba
    console_log_interval: 1ba
    global_train_batch_size: 4
    device_train_microbatch_size: 1
    device_eval_batch_size: 1
    train_loader:
      num_workers: 0
      dataset:
        use_dummy: true
        epoch_size: 8
```

### Example 1: Dummy one-batch startup smoke

Use when you only need to validate launch, model init, dataloader construction, one train batch, and clean exit.

```yaml
debug:
  use_debug: true
  startup_speedups:
    use_one_train_dataloader: true
    disable_runtime_benchmark_callbacks: true
    disable_eval_loader: true
    icl_tasks_metrics: null
    load_checkpoints: false
    fast_model_init: true
  overrides:
    max_duration: 1ba
    save_interval: 10ba
    console_log_interval: 1ba
    global_train_batch_size: 4
    device_train_microbatch_size: 1
    device_eval_batch_size: 1
    train_loader:
      num_workers: 0
      dataset:
        use_dummy: true
        epoch_size: 8
```

### Example 2: One real stream smoke

Use when dummy data is not enough and you need to validate reading one specific real dataset stream with minimal startup cost.

```yaml
debug:
  use_debug: true
  startup_speedups:
    use_one_train_dataloader: true
    train_stream_name: starcoder2_filtered_vocab_rarity
    disable_runtime_benchmark_callbacks: true
    disable_eval_loader: true
    icl_tasks_metrics: null
    load_checkpoints: false
    fast_model_init: true
  overrides:
    max_duration: 5ba
    console_log_interval: 1ba
    global_train_batch_size: 8
    device_train_microbatch_size: 1
    device_eval_batch_size: 1
    train_loader:
      num_workers: 0
```

Do not set `train_loader.dataset.use_dummy` in this profile, otherwise the stream selection will not exercise the real dataset path.

### Example 3: Keep one ICL task, skip the rest

Use when you want a small metric sanity check without starting the full eval loader stack.

```yaml
debug:
  use_debug: true
  startup_speedups:
    use_one_train_dataloader: true
    disable_runtime_benchmark_callbacks: true
    disable_eval_loader: true
    icl_tasks_metrics: ['babymmlu']
    load_checkpoints: false
    fast_model_init: true
  overrides:
    max_duration: 2ba
    console_log_interval: 1ba
    global_train_batch_size: 4
    device_train_microbatch_size: 1
    device_eval_batch_size: 1
    train_loader:
      num_workers: 0
```

### Validated remote smoke on 2 nodes x 2 GPUs

The repo now includes a tracked launcher for the exact short distributed smoke that was used to validate the debug path on a real `sber_config`:

`sber_scripts/infinite_jobs/default_gigar_debug_smoke.sh`

Run it through the required remote wrapper:

```bash
python remote_run.py "cd <repo-root> && bash sber_scripts/infinite_jobs/default_gigar_debug_smoke.sh"
```

`<repo-root>` here means this repository root on the shared NFS mount.

What this launcher validates:
- real distributed startup on `2 nodes x 2 GPUs`;
- `sber_configs/default_gigar_config.yaml` parsing with `debug.use_debug=true`;
- model init, train loader build, trainer build, one train batch, checkpoint save, and clean shutdown.
- repeated runs without manual checkpoint cleanup, because the launcher stamps `run_name`, log dir, tensorboard dir, and save dir with a timestamp.

Why it has a few explicit overrides:
- the base config under test is still `sber_configs/default_gigar_config.yaml`;
- the original `model_path` from that config is not present on the current test cluster, so the launcher points to a compatible local checkpoint via `MODEL_PATH`;
- that fallback checkpoint also needs `model.config_overrides.gated_attention=false` and `model.config_overrides.layernorm_type=pre` to match the available architecture.

Last successful run:
- launcher log root: `tmp/inf-job-logs/default-gigar-debug-smoke-20260429-170401/`
- rank-0 log: `tmp/inf-job-logs/default-gigar-debug-smoke-20260429-170401/nojobid/1/rank.0/stdout`
- key lines:
  - `Initializing model...`
  - `Building train loader...`
  - `Building trainer...`
  - `Train loss/train/total: 11.6942`
  - `1 checkpoints saved to .../tmp/checkpoints/default-gigar-debug-smoke-20260429-170401/...`
  - `Done.`

This smoke proves that the debug section works on a real multi-node launch with a real `sber_config`. It does not claim that the original production `model_path` from `default_gigar_config.yaml` exists on every cluster.

# Validation loss (NLL) in TensorBoard

By default a run logs only train-side loss (`loss/train/total`, `loss/train/loss_ce`). Validation NLL is opt-in and requires a held-out dataset, which the default configs do not ship.

## What you get

Composer never calls `model.loss()` during eval — it only updates metrics — so there is no `loss/eval/*` tag and never will be. The validation NLL is the `LanguageCrossEntropy` metric evaluated on `eval_loader`, logged as:

- `metrics/eval/LanguageCrossEntropy` — mean per-token NLL in **nats** (padding and `-100` labels excluded);
- `metrics/eval/LanguagePerplexity` — `exp()` of the above.

Both land on the same step axis as `loss/train/*`, so they can be overlaid in TensorBoard with a regex like `loss/train/loss_ce|metrics/eval/LanguageCrossEntropy`.

## How to enable it

1. Uncomment the `eval_loader` template in your `sber_config` and point it at a held-out split.
2. Keep `metric_names: [LanguageCrossEntropy, LanguagePerplexity]`. It is **not** optional: without it every val metric of the model is attached to this evaluator, and the ICL metrics crash on a text batch because they require `continuation_indices`.
3. Keep `shuffle: false` and `drop_last: false` so the same batches are scored at every eval point — otherwise the curve is noisy for no reason and checkpoints are not comparable.
4. Set `eval_interval` and cap `eval_subset_num_batches` (the default `-1` means the whole eval set).
5. Set `eval_before_train: regular` if you want a point at step 0. Do not add the deprecated `eval_first` key — its mere presence, even when set to `false`, forces `eval_before_train` to `skip`.

## Cost

Eval runs outside `self.training`, so the liger fusion of `lm_head` + cross-entropy is disabled and `delete_logits` does not apply: logits are materialized in full, `device_eval_batch_size x max_seq_len x vocab_size`. With a 128k vocabulary this is the reason these metrics were disabled in the first place. Start with `device_eval_batch_size: 1-2` and a few tens of batches.

## The held-out data itself

Out of scope for this section, but load-bearing: the eval split must not overlap with any `train_loader` stream. Slicing "the tail" off a train stream does not give you that — the streams are shuffled (`shuffle_algo: py1b`), so the same samples can still be drawn during training, and the resulting number is contaminated and reads optimistically. If you knowingly point `eval_loader` at train data as a temporary signal, say so wherever the graph is shown.

## Debug profile

`debug.startup_speedups.disable_eval_loader` defaults to `true` and drops `eval_loader` — that is intentional, since the point of the debug profile is a fast startup. Set it to `false` in the smoke you use to verify a validation setup, otherwise no eval will run and `eval_interval` will be forced to `0`.
