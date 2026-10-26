# Recipe migration inventory

At main commit `ab0904812b`, the 14 `config_registry.py` files contain **113 public
recipe functions**. The tables
cover all 113 entry points; several were renamed and several have only a model
preset. A preset alone does not replace a recipe's data, callbacks or parallelism.

This is file-level coverage, not proof of equivalent settings, numerical parity
or GPU correctness. Unmigrated recipes are explicit gaps, not declarations that
the recipes are obsolete. Runtime imports are not needed to enumerate the source
functions. Source paths below are relative to `torchtitan/` at the pinned commit.

Experiment files live in `configs/experiments/`; model presets live in
`configs/presets/models/`. The inventory tracks historical recipe replacements;
additional YAML-only experiments may exist without a historical counterpart.
Update the relevant row when migrating another historical recipe. List the
current experiment files with `git ls-files 'configs/experiments/*.yaml'`.

## YAML migration requirements

YAML recipes replace the Python registry entry points. Use an experiment YAML
path with `torchrun -m torchtitan.train` and pass overrides as `--section.field
value`. Runtime preparation now belongs to the model's `prepare_config` hook;
tools must not call the removed config method `update_from_config`.

RoPE selection is explicit: every supplied `rope:` mapping needs a `type`, such
as `type: complex` or `type: cos_sin`. There is no family default. An experiment
may inherit this key from a preset, but replacing the whole mapping with
`!replace` must include it. This is a breaking change to earlier YAML drafts in
this MR; files from those drafts need migration before use.

The non-blocking HybridEP experiment uses FP8 only for dense projections and
BF16 for experts. Its filename includes `dense_fp8` to make that distinction
visible. Full FP8 experts with fixed-capacity HybridEP are rejected during
pre-submit validation because their gradient path is unsupported.

## GIFT: models/gift_moe

| Historical entry point | YAML status |
| --- | --- |
| `gift_moe_debugmodel` | Experiment: `gift_moe_debug.yaml` |
| `gift_moe_debugmodel_tiny` | No equivalent experiment migrated. |
| `gift_moe_debugmodel_frozen` | `gift_moe_debug_frozen.yaml`: freeze the first two layers through the `freeze` converter. |
| `gift_moe_debugmodel_tiny_frozen` | No equivalent experiment migrated. |
| `gift_moe_debugmodel_hybridep` | No equivalent experiment migrated. |
| `gift_moe_debugmodel_deepep` | Experiment: `gift_moe_debug_deepep.yaml` |
| `gift_moe_debugmodel_deepep_fp8` | Experiment: `gift_moe_debug_deepep_fp8.yaml` |
| `gift_moe_debugmodel_hybridep_fp8` | Experiment: `gift_moe_debug_hybridep_fp8.yaml` |
| `gift_moe_debugmodel_hybridep_fp8_compile` | Experiment: `gift_moe_debug_hybridep_fp8_compile.yaml` |
| `gift_moe_debugmodel_fp8_dense` | No equivalent experiment migrated. |
| `gift_moe_debugmodel_fp8_grouped_deepep` | No equivalent experiment migrated. |
| `gift_moe_debugmodel_fp8_pp` | No equivalent experiment migrated. |
| `gift_moe_debugmodel_fp8_pp_interleaved` | No equivalent experiment migrated. |
| `gift_moe_debugmodel_flex_attn` | No equivalent experiment migrated. |
| `gift_moe_debugmodel_flex_attn_ep` | No equivalent experiment migrated. |
| `gift_moe_debugmodel_varlen` | No equivalent experiment migrated. |
| `gift_moe_debugmodel_ep` | No equivalent experiment migrated. |
| `gift_3_5_debugmodel` | Experiment: `gift_3_5_debug.yaml` |
| `gift_3_5_debugmodel_varlen` | Experiment: `gift_3_5_debug_varlen.yaml` |
| `gift_3_5_debugmodel_ep` | No equivalent experiment migrated. |
| `gift_3_5_3b` | Model preset only: `gift_moe/gift_3_5/3b.yaml`; training recipe not migrated. |
| `gift_3_5_700b` | Experiment: `gift_3_5_700b.yaml` |
| `gift_4_debugmodel` | Experiment: `gift_4_debug.yaml` |
| `gift_4_3b` | Model preset only: `gift_moe/gift_4/3b.yaml`; training recipe not migrated. |
| `gift_4_700b` | Experiment: `gift_4_700b.yaml` |
| `gift_moe_debugmodel_mds` | Experiment: `gift_moe_debug_mds.yaml` |
| `gift_3_5_debugmodel_mds` | Experiment: `gift_3_5_debug_mds.yaml` |
| `gift_moe_streaming_debugmodel` | Experiment: `gift_moe_streaming_debug.yaml` |
| `gift_moe_16b` | Model preset only: `gift_moe/16b.yaml`; training recipe not migrated. |
| `gift_moe_671b` | Model preset only: `gift_moe/671b.yaml`; training recipe not migrated. |
| `gift_moe_3b` | Model preset only: `gift_moe/3b.yaml`; training recipe not migrated. |
| `gift_moe_3b_monitored` | No equivalent experiment migrated. |
| `gift_moe_debugmodel_monitored` | Experiment: `gift_moe_debug_monitored.yaml` |
| `gift_moe_3b_hybridep_fp8_compile` | Experiment: `gift_moe_3b_hybridep_fp8_compile.yaml` |
| `gift_moe_3b_hybridep_fp8_compile_pp` | Experiment: `gift_moe_3b_hybridep_fp8_compile_pp.yaml` |
| `gift_moe_3b_hybridep_fp8_compile_hsdp` | Experiment: `gift_moe_3b_hybridep_fp8_compile_hsdp.yaml` |
| `gift_moe_3b_hybridep_fp8_compile_tp` | Experiment: `gift_moe_3b_hybridep_fp8_compile_tp.yaml` |
| `gift_moe_3b_hybridep_fp8_compile_non_blocking` | `gift_moe_3b_hybridep_dense_fp8_compile_non_blocking.yaml`: dense FP8, BF16 experts; full FP8 parity is not supported. |

## experiments/ft/llama3

| Historical entry point | YAML status |
| --- | --- |
| `llama3_ft_debugmodel` | Experiment: `ft_llama3_debug.yaml` |

## experiments/graph_trainer/deepseek_v3

| Historical entry point | YAML status |
| --- | --- |
| `graph_trainer_deepseek_v3_debugmodel` | Experiment: `graph_deepseek_v3_debug.yaml` |
| `graph_trainer_deepseek_v3_debugmodel_ep` | No corresponding YAML experiment migrated. |
| `graph_trainer_deepseek_v3_debugmodel_hybridep` | Experiment: `graph_deepseek_v3_debug_hybridep.yaml` |
| `graph_trainer_deepseek_v3_debugmodel_flex_attn` | Experiment: `graph_deepseek_v3_debug_flex_attn.yaml` |
| `graph_trainer_deepseek_v3_debugmodel_flex_attn_ep` | No corresponding YAML experiment migrated. |
| `graph_trainer_deepseek_v3_16b` | No corresponding YAML experiment migrated. |
| `graph_trainer_deepseek_v3_16b_sdpa` | No corresponding YAML experiment migrated. |
| `graph_trainer_deepseek_v3_671b` | No corresponding YAML experiment migrated. |
| `deepseek_v3_debugmodel_ep_ce_loss` | No corresponding YAML experiment migrated. |

## experiments/graph_trainer/llama3

| Historical entry point | YAML status |
| --- | --- |
| `graph_trainer_llama3_debugmodel` | Experiment: `graph_llama3_debug.yaml` |
| `graph_trainer_llama3_debugmodel_flex_attn` | Experiment: `graph_llama3_debug_flex_attn.yaml` |
| `graph_trainer_llama3_8b` | Experiment: `graph_llama3_8b.yaml` |
| `graph_trainer_llama3_70b` | No corresponding YAML experiment migrated. |
| `graph_trainer_llama3_405b` | No corresponding YAML experiment migrated. |

## experiments/graph_trainer/qwen3

| Historical entry point | YAML status |
| --- | --- |
| `graph_trainer_qwen3_debugmodel` | Experiment: `graph_qwen3_debug.yaml` |
| `graph_trainer_qwen3_debugmodel_flex_attn` | No corresponding YAML experiment migrated. |
| `graph_trainer_qwen3_debugmodel_moe` | Experiment: `graph_qwen3_moe_debug.yaml` |
| `graph_trainer_qwen3_debugmodel_moe_ep` | No corresponding YAML experiment migrated. |
| `graph_trainer_qwen3_14b` | No corresponding YAML experiment migrated. |
| `qwen3_debugmodel_ce_loss` | No corresponding YAML experiment migrated. |
| `qwen3_moe_debug_ep_ce_loss` | No corresponding YAML experiment migrated. |

## experiments/rl

| Historical entry point | YAML status |
| --- | --- |
| `rl_grpo_qwen3_0_6b` | Experiment: `rl_grpo_qwen3_0_6b.yaml` |
| `rl_grpo_qwen3_1_7b` | Experiment: `rl_grpo_qwen3_1_7b.yaml` |
| `rl_grpo_qwen3_14b` | Experiment: `rl_grpo_qwen3_14b.yaml` |
| `rl_grpo_qwen3_0_6b_batch_invariant` | Experiment: `rl_grpo_qwen3_0_6b_batch_invariant.yaml` |

## experiments/transformers_modeling_backend

| Historical entry point | YAML status |
| --- | --- |
| `transformers_modeling_backend_debugmodel` | No corresponding YAML experiment migrated. |
| `transformers_modeling_backend_full` | No corresponding YAML experiment migrated. |

## models/deepseek_v3

| Historical entry point | YAML status |
| --- | --- |
| `deepseek_v3_debugmodel` | Experiment: `deepseek_v3_debug.yaml` |
| `deepseek_v3_debugmodel_hybridep` | Experiment: `deepseek_v3_debug_hybridep.yaml` |
| `deepseek_v3_debugmodel_flex_attn` | Experiment: `deepseek_v3_debug_flex_attn.yaml` |
| `deepseek_v3_debugmodel_flex_attn_ep` | No corresponding YAML experiment migrated. |
| `deepseek_v3_16b` | Model preset only: `deepseek_v3/16b.yaml`; training recipe not migrated. |
| `deepseek_v3_671b` | Model preset only: `deepseek_v3/671b.yaml`; training recipe not migrated. |

## models/flux

| Historical entry point | YAML status |
| --- | --- |
| `flux_debugmodel` | Experiment: `flux_debug.yaml` |
| `flux_dev` | Model preset only: `flux/dev.yaml`; training recipe not migrated. |
| `flux_schnell` | Model preset only: `flux/schnell.yaml`; training recipe not migrated. |

## models/gpt_oss

| Historical entry point | YAML status |
| --- | --- |
| `gpt_oss_debugmodel` | Experiment: `gpt_oss_debug.yaml` |
| `gpt_oss_debugmodel_ep` | No corresponding YAML experiment migrated. |
| `gpt_oss_20b` | Model preset only: `gpt_oss/20b.yaml`; training recipe not migrated. |
| `gpt_oss_120b` | Model preset only: `gpt_oss/120b.yaml`; training recipe not migrated. |

## models/llama3

| Historical entry point | YAML status |
| --- | --- |
| `llama3_debugmodel` | Experiment: `llama3_debug.yaml` |
| `llama3_debugmodel_fused_qkv` | Model preset only: `llama3/debugmodel_fused_qkv.yaml`; training recipe not migrated. |
| `llama3_debugmodel_flex_attn` | Experiment: `llama3_debug_flex_attn.yaml` |
| `llama3_debugmodel_varlen_attn` | Experiment: `llama3_debug_varlen_attn.yaml` |
| `llama3_debugmodel_fa3` | No corresponding YAML experiment migrated. |
| `llama3_debugmodel_opt_in_bwd` | No corresponding YAML experiment migrated. |
| `llama3_debugmodel_float8` | Experiment: `llama3_debug_float8.yaml` |
| `llama3_debugmodel_ce_loss` | Experiment: `llama3_debug_ce_loss.yaml` |
| `llama3_8b` | Experiment: `llama3_8b.yaml` |
| `llama3_70b` | Model preset only: `llama3/70b.yaml`; training recipe not migrated. |
| `llama3_405b` | Model preset only: `llama3/405b.yaml`; training recipe not migrated. |
| `sft_debugmodel` | Experiment: `sft_llama3_debug.yaml` |

## models/llama4

| Historical entry point | YAML status |
| --- | --- |
| `llama4_debugmodel` | Experiment: `llama4_debug.yaml` |
| `llama4_debugmodel_ep` | No corresponding YAML experiment migrated. |
| `llama4_debugmodel_fp8` | Experiment: `llama4_debug_fp8.yaml` |
| `llama4_17bx128e` | Model preset only: `llama4/17bx128e.yaml`; training recipe not migrated. |
| `llama4_17bx16e` | Model preset only: `llama4/17bx16e.yaml`; training recipe not migrated. |

## models/qwen3

| Historical entry point | YAML status |
| --- | --- |
| `qwen3_debugmodel` | Experiment: `qwen3_debug.yaml` |
| `qwen3_debugmodel_param_groups` | Experiment: `qwen3_debug_param_groups.yaml` |
| `qwen3_debugmodel_flex` | No corresponding YAML experiment migrated. |
| `qwen3_debugmodel_flex_flash` | No corresponding YAML experiment migrated. |
| `qwen3_0_6b` | Model preset only: `qwen3/0_6b.yaml`; training recipe not migrated. |
| `qwen3_1_7b` | Model preset only: `qwen3/1_7b.yaml`; training recipe not migrated. |
| `qwen3_14b` | Model preset only: `qwen3/14b.yaml`; training recipe not migrated. |
| `qwen3_32b` | Model preset only: `qwen3/32b.yaml`; training recipe not migrated. |
| `qwen3_debugmodel_fused_qkv` | Experiment: `qwen3_debug_fused_qkv.yaml` |
| `qwen3_moe_debug` | Experiment: `qwen3_moe_debug.yaml` |
| `qwen3_moe_debug_ep` | No corresponding YAML experiment migrated. |
| `sft_qwen3_8b_math` | Experiment: `sft_qwen3_8b_math.yaml` |

## models/qwen3_vl

| Historical entry point | YAML status |
| --- | --- |
| `qwen3_vl_debugmodel` | Experiment: `qwen3_vl_debug.yaml` |
| `qwen3_vl_debugmodel_moe` | Experiment: `qwen3_vl_moe_debug.yaml` |
| `qwen3_vl_2b` | Model preset only: `qwen3_vl/2b.yaml`; training recipe not migrated. |
| `qwen3_vl_8b` | Model preset only: `qwen3_vl/8b.yaml`; training recipe not migrated. |
| `qwen3_vl_30b_a3b` | Model preset only: `qwen3_vl/30b_a3b.yaml`; training recipe not migrated. |
