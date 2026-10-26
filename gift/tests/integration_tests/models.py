# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from tests.integration_tests import OverrideDefinitions


def build_model_tests_list() -> list[OverrideDefinitions]:
    """
    Build a dictionary of model parallelism test configurations.
    This test suite is aimed at testing the model parallelism of torchtitan, and will broadly cover
    all the supported model parallelism patterns on all the supported models.

    Returns:
        A dictionary where each key is a model name and value is a list of OverrideDefinitions
    """
    model_tests = [
        # Integration Test Cases for DeepSeek V3
        OverrideDefinitions(
            [
                [
                    "--config-file configs/experiments/deepseek_v3_debug.yaml",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.expert_parallel_degree 2",
                    "--compile.enable",
                ],
            ],
            "DeepSeek V3 FSDP+EP+compile",
            "deepseek_v3_fsdp+ep+compile",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--config-file configs/experiments/deepseek_v3_debug.yaml",
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.pipeline_parallel_schedule Interleaved1F1B",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.expert_parallel_degree 4",
                ],
            ],
            "DeepSeek V3 PP+FSDP+TP+EP",
            "deepseek_v3_pp+fsdp+tp+ep",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--config-file configs/experiments/deepseek_v3_debug.yaml",
                    "--parallelism.data_parallel_replicate_degree 2",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.expert_parallel_degree 2",
                ],
            ],
            "DeepSeek V3 HSDP+EP",
            "deepseek_v3_hsdp+ep",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--config-file configs/experiments/deepseek_v3_debug_flex_attn.yaml",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.pipeline_parallel_schedule Interleaved1F1B",
                    "--parallelism.expert_parallel_degree 4",
                    "--activation_checkpoint.mode 'selective'",
                ],
            ],
            "DeepSeek V3 Flex+PP+FSDP+EP+SACOP",
            "deepseek_v3_flex+pp+fsdp+ep+sacop",
            ngpu=8,
        ),
        # Integration Test Cases for gift_moe (DeepSeek-V3-style MoE clone)
        OverrideDefinitions(
            [
                [
                    "--config-file configs/experiments/gift_moe_debug.yaml",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.expert_parallel_degree 2",
                    "--compile.enable",
                ],
            ],
            "gift_moe FSDP+EP+compile",
            "gift_moe_fsdp+ep+compile",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--config-file configs/experiments/gift_moe_debug_mds.yaml",
                    "--parallelism.data_parallel_replicate_degree 2",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.expert_parallel_degree 2",
                ],
            ],
            "gift_moe HSDP+EP (streaming loader)",
            "gift_moe_hsdp+ep",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--config-file configs/experiments/gift_moe_debug_monitored.yaml",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--training.steps 12",
                    "--checkpoint.enable",
                    "--checkpoint.interval 10",
                ],
            ],
            "gift_moe monitored callbacks",
            "gift_moe_monitored_callbacks",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--config-file configs/experiments/gift_moe_debug_monitored.yaml",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--compile.enable",
                    "--activation_checkpoint.mode full",
                    "--training.steps 12",
                ],
            ],
            "gift_moe monitored callbacks, compiled blocks (step 10 runs eager)",
            "gift_moe_monitored_callbacks_compile",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--config-file configs/experiments/gift_moe_debug_monitored.yaml",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.pipeline_parallel_schedule Interleaved1F1B",
                    "--compile.enable",
                    "--checkpoint.enable",
                    "--checkpoint.interval 9",
                    f"--training.steps {steps}",
                ]
                for steps in (9, 12)
            ],
            "gift_moe monitored callbacks, compile + PP resume on logging step 10",
            "gift_moe_monitored_callbacks_compile_pp",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--config-file configs/experiments/gift_moe_streaming_debug.yaml",
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.pipeline_parallel_schedule Interleaved1F1B",
                    "--checkpoint.enable",
                ],
                [
                    "--config-file configs/experiments/gift_moe_streaming_debug.yaml",
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.pipeline_parallel_schedule Interleaved1F1B",
                    "--checkpoint.enable",
                    "--training.steps 20",
                ],
            ],
            "gift_moe streaming loader PP2+FSDP save/resume",
            "gift_moe_streaming_pp2_resume",
            ngpu=4,
        ),
        # Integration Test Cases for GIFT 3.5
        OverrideDefinitions(
            [
                [
                    "--config-file configs/experiments/gift_3_5_debug.yaml",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.expert_parallel_degree 2",
                    "--compile.enable",
                ],
            ],
            "GIFT 3.5 FSDP+EP+compile (MTP)",
            "gift_3_5_fsdp+ep+compile",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--config-file configs/experiments/gift_3_5_debug.yaml",
                    "--parallelism.data_parallel_replicate_degree 2",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.expert_parallel_degree 2",
                    "--activation_checkpoint.mode full",
                ],
            ],
            "GIFT 3.5 HSDP+EP+full AC (MTP)",
            "gift_3_5_hsdp+ep+full_ac",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--config-file configs/experiments/gift_3_5_debug_mds.yaml",
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.pipeline_parallel_schedule Interleaved1F1B",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.expert_parallel_degree 2",
                ],
            ],
            "GIFT 3.5 PP+FSDP+EP (MTP, streaming loader)",
            "gift_3_5_pp+fsdp+ep",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--config-file configs/experiments/gift_3_5_debug.yaml",
                    "--parallelism.pipeline_parallel_degree 4",
                    "--parallelism.pipeline_parallel_schedule 1F1B",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.expert_parallel_degree 2",
                ],
            ],
            "GIFT 3.5 PP(4 stages)+FSDP+EP (MTP)",
            "gift_3_5_pp4+fsdp+ep",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--config-file configs/experiments/gift_3_5_debug.yaml",
                    "--parallelism.pipeline_parallel_degree 4",
                    "--parallelism.pipeline_parallel_balancing flops",
                    "--parallelism.pipeline_parallel_schedule 1F1B",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.expert_parallel_degree 2",
                ],
            ],
            "GIFT 3.5 PP(4 stages, FLOPs-balanced)+FSDP+EP (MTP)",
            "gift_3_5_pp4+fsdp+ep+flops",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--config-file configs/experiments/gift_3_5_debug.yaml",
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.pipeline_parallel_schedule ZBVZeroBubble",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--activation_checkpoint.mode none",
                ],
            ],
            "GIFT 3.5 PP zero-bubble (V-shaped)+FSDP (MTP)",
            "gift_3_5_pp_zbv+fsdp",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--config-file configs/experiments/gift_3_5_debug_varlen.yaml",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.expert_parallel_degree 2",
                ],
            ],
            "GIFT 3.5 varlen (document-packed GDN + varlen MLA) FSDP+EP (MTP)",
            "gift_3_5_varlen_fsdp+ep",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--config-file configs/experiments/gift_3_5_debug_mds.yaml",
                    "--checkpoint.enable",
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.expert_parallel_degree 2",
                ],
                [
                    "--config-file configs/experiments/gift_3_5_debug_mds.yaml",
                    "--checkpoint.enable",
                    "--training.steps 20",
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.expert_parallel_degree 2",
                ],
            ],
            "GIFT 3.5 PP+FSDP+EP save/load resume checkpoint (MTP, streaming loader)",
            "gift_3_5_pp+fsdp+ep_checkpoint",
            ngpu=4,
        ),
        # Integration Test Cases for GIFT 4 (GIFT 3.5 + soft-capped LM head and
        # projection hidden z-loss). PP is the interesting one: the soft-capped
        # lm_head and the MTP heads land on the same last stage.
        OverrideDefinitions(
            [
                [
                    "--config-file configs/experiments/gift_4_debug.yaml",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.expert_parallel_degree 2",
                    "--compile.enable",
                ],
            ],
            "GIFT 4 FSDP+EP+compile (MTP)",
            "gift_4_fsdp+ep+compile",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--config-file configs/experiments/gift_4_debug.yaml",
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.pipeline_parallel_schedule Interleaved1F1B",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.expert_parallel_degree 2",
                ],
            ],
            "GIFT 4 PP+FSDP+EP (MTP)",
            "gift_4_pp+fsdp+ep",
            ngpu=4,
        ),
        # Integration Test Cases for Qwen3 dense and MoE model
        OverrideDefinitions(
            [
                [
                    "--config-file configs/experiments/qwen3_debug_param_groups.yaml",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                ],
            ],
            "Qwen3 FSDP+TP (param groups)",
            "qwen3_fsdp+tp_param_groups",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--config-file configs/experiments/qwen3_debug.yaml",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.enable_sequence_parallel false",
                ],
                [
                    "--config-file configs/experiments/qwen3_debug_fused_qkv.yaml",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                ],
            ],
            "Qwen3 FSDP+TP (SP disabled + fused QKV)",
            "qwen3_fsdp+tp_no_sp_fused_qkv",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--config-file configs/experiments/qwen3_debug.yaml",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.context_parallel_degree 2",
                ],
            ],
            "Qwen3 FSDP+TP+CP",
            "qwen3_fsdp+tp+cp",
            ngpu=8,
        ),
        # Integration Test Cases for Llama 4
        # TODO: re-enable compile after fixing
        # https://github.com/pytorch/torchtitan/issues/2771
        OverrideDefinitions(
            [
                [
                    "--config-file configs/experiments/llama4_debug.yaml",
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.pipeline_parallel_schedule Interleaved1F1B",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.expert_parallel_degree 4",
                    # "--compile.enable",
                ],
            ],
            "Llama 4 PP+FSDP+TP+EP+compile",
            "llama4_pp+fsdp+tp+ep+compile",
            ngpu=8,
        ),
        # Integration Test Cases for Qwen3-VL
        OverrideDefinitions(
            [
                [
                    "--config-file configs/experiments/qwen3_vl_moe_debug.yaml",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.expert_parallel_degree 4",
                ],
            ],
            "Qwen3-VL MoE FSDP+TP+EP",
            "qwen3_vl_moe_fsdp+tp+ep",
            ngpu=8,
        ),
        # Integration Test Cases for gpt-oss
        # TODO: re-enable compile after fixing
        # https://github.com/pytorch/torchtitan/issues/3409
        OverrideDefinitions(
            [
                [
                    "--config-file configs/experiments/gpt_oss_debug.yaml",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.expert_parallel_degree 4",
                    # "--compile.enable",
                ],
            ],
            "Gpt-oss FSDP+TP+EP+compile",
            "gpt_oss_fsdp+tp+ep+compile",
            ngpu=8,
        ),
    ]

    return model_tests
