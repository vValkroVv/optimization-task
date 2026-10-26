# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

from tests.integration_tests import OverrideDefinitions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_h100_tests_list() -> list[OverrideDefinitions]:
    """
    key is the config file name and value is a list of OverrideDefinitions
    that is used to generate variations of integration tests based on the
    same root config file.
    """
    integration_tests_flavors = [
        *[
            OverrideDefinitions(
                [[f"--backend {backend}"]],
                f"{backend} token-drop outputs and gradients against reference",
                f"token_drop_numerics_{backend}",
                ngpu=2,
                skip_rocm_test=True,
                timeout=600,
                test_module="tests.integration_tests.token_drop_numerics",
            )
            for backend in ("deepep", "hybridep")
        ],
        OverrideDefinitions(
            [
                [
                    "--compile.enable",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.enable_async_tensor_parallel",
                ],
            ],
            "2D async TP compile",
            "2d_asynctp_compile",
        ),
        OverrideDefinitions(
            [
                [
                    "--config-file configs/experiments/llama3_debug_float8.yaml",
                ],
            ],
            "Float8 test",
            "float8",
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.full_dtensor",
                    "--parallelism.enable_fsdp_symm_mem",
                ],
            ],
            "FSDP symmetric memory",
            "fsdp_symm_mem",
            ngpu=2,
        ),
        OverrideDefinitions(
            [
                [
                    "--config-file configs/experiments/llama3_debug_float8.yaml",
                    "--compile.enable",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.data_parallel_replicate_degree 2",
                    "--parallelism.context_parallel_degree 2",
                ]
            ],
            "HSDP+CP+torch.compile+Float8",
            "hsdp+cp+compile+float8",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--config-file configs/experiments/deepseek_v3_debug_hybridep.yaml",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.expert_parallel_degree 2",
                    "--compile.enable",
                    "--compile.components model,loss",
                ],
            ],
            "DeepSeek V3 FSDP+HybridEP+compile",
            "deepseek_v3_fsdp+hybridep+compile",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--config-file configs/experiments/gift_moe_debug_deepep.yaml",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.expert_parallel_degree 2",
                ],
            ],
            "gift_moe FSDP+DeepEP",
            "gift_moe_fsdp+deepep",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--config-file configs/experiments/gift_moe_debug_deepep_token_drop.yaml",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.expert_parallel_degree 2",
                    "--training.steps 10",
                ]
            ],
            "gift_moe FSDP+DeepEP token drop",
            "gift_moe_fsdp+deepep+token_drop",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--config-file configs/experiments/gift_moe_debug_deepep_token_drop_fp8.yaml",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.expert_parallel_degree 2",
                    "--training.steps 10",
                ]
            ],
            "gift_moe FSDP+DeepEP token drop+fp8",
            "gift_moe_fsdp+deepep+token_drop+fp8",
            ngpu=4,
            env={"DG_CACHE_FORCE_SKIP": "1"},
        ),
        OverrideDefinitions(
            [
                [
                    "--config-file configs/experiments/gift_moe_debug_hybridep_token_drop.yaml",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.expert_parallel_degree 2",
                    "--training.steps 10",
                ]
            ],
            "gift_moe FSDP+HybridEP token drop conditional",
            "gift_moe_fsdp+hybridep+token_drop+conditional",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--config-file configs/experiments/gift_moe_debug_hybridep_token_drop_branchless.yaml",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.expert_parallel_degree 2",
                    "--training.steps 10",
                ]
            ],
            "gift_moe FSDP+HybridEP token drop branchless",
            "gift_moe_fsdp+hybridep+token_drop+branchless",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--config-file configs/experiments/gift_moe_debug_hybridep_token_drop_branchless.yaml",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.expert_parallel_degree 2",
                    "--compile.enable",
                    "--compile.components model,loss",
                    "--training.steps 10",
                ]
            ],
            "gift_moe FSDP+HybridEP token drop branchless+compile",
            "gift_moe_fsdp+hybridep+token_drop+branchless+compile",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--config-file configs/experiments/gift_moe_debug_hybridep_token_drop.yaml",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.expert_parallel_degree 2",
                    "--compile.enable",
                    "--compile.components model,loss",
                    "--training.steps 10",
                ]
            ],
            "gift_moe FSDP+HybridEP token drop conditional+compile",
            "gift_moe_fsdp+hybridep+token_drop+conditional+compile",
            ngpu=4,
        ),
        # fp8 over an EP backend had no integration coverage: the float8 rows
        # above are dense-only, and the HybridEP row is bf16.
        #
        # DG_CACHE_FORCE_SKIP: DeepGEMM refuses a JIT cache miss rather than
        # compiling silently, and these rows hit shapes no warmup covers.
        # Production warms the cache instead -- see torchtitan/ops/warmup.
        OverrideDefinitions(
            [
                [
                    "--config-file configs/experiments/gift_moe_debug_hybridep_fp8.yaml",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.expert_parallel_degree 2",
                ],
            ],
            "gift_moe FSDP+HybridEP+fp8",
            "gift_moe_fsdp+hybridep+fp8",
            ngpu=4,
            env={"DG_CACHE_FORCE_SKIP": "1"},
        ),
        OverrideDefinitions(
            [
                [
                    "--config-file "
                    "configs/experiments/gift_moe_debug_hybridep_fp8_compile.yaml",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.expert_parallel_degree 2",
                ],
            ],
            "gift_moe FSDP+HybridEP+fp8+compile",
            "gift_moe_fsdp+hybridep+fp8+compile",
            ngpu=4,
            env={"DG_CACHE_FORCE_SKIP": "1"},
        ),
        OverrideDefinitions(
            [
                [
                    "--config-file configs/experiments/gift_moe_debug_deepep_fp8.yaml",
                    "--parallelism.data_parallel_shard_degree 8",
                    "--parallelism.expert_parallel_degree 4",
                ],
            ],
            "gift_moe FSDP+DeepEP+fp8 dense and grouped",
            "gift_moe_fsdp+deepep+fp8",
            ngpu=8,
            env={"DG_CACHE_FORCE_SKIP": "1"},
        ),
        OverrideDefinitions(
            [
                [
                    "--config-file configs/experiments/gift_moe_debug_deepep.yaml",
                    "--parallelism.fsdp_backend gfsdp2",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.expert_parallel_degree 2",
                ],
            ],
            "gift_moe gfsdp2+DeepEP",
            "gift_moe_gfsdp2+deepep",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--config-file configs/experiments/gift_moe_debug_deepep.yaml",
                    "--parallelism.fsdp_backend gfsdp2",
                    "--parallelism.fsdp_reshard_after_forward never",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.expert_parallel_degree 2",
                ],
            ],
            "gift_moe gfsdp2+DeepEP, parameters held after forward",
            "gift_moe_gfsdp2+deepep+no_reshard",
            ngpu=4,
        ),
    ]
    return integration_tests_flavors
