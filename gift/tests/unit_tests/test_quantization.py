# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from tests.unit_tests.config_helpers import load_experiment_config
from torchtitan.components.quantization.utils import apply_float8, has_quantization
from torchtitan.models.common.nn_modules import Linear
from torchtitan.ops.float8.cuda_gemm import FP8Linear


def _model_config(experiment: str):
    """Load *experiment* and apply the fp8 selection the trainer would apply."""
    config = load_experiment_config(experiment)
    model_config = config.model_spec.model
    apply_float8(
        model_config,
        config.fp8,
        tensor_parallel_degree=config.parallelism.tensor_parallel_degree,
    )
    return model_config


def test_no_float8_by_default():
    model_config = _model_config("llama3_debug.yaml")
    assert not has_quantization(model_config)
    # Every Linear.Config stays a plain Linear.Config.
    for _fqn, lc, _parent, _attr in model_config.traverse(Linear.Config):
        assert not isinstance(lc, FP8Linear.Config)


def test_unmatched_float8_selector_warns(caplog):
    config = load_experiment_config("llama3_debug.yaml")
    config.fp8.dense_include_linear_layers = "wq_typo"
    apply_float8(config.model_spec.model, config.fp8, tensor_parallel_degree=1)
    assert "wq_typo" in caplog.text and "matched no" in caplog.text
    assert not has_quantization(config.model_spec.model)


def test_float8_applied_from_fp8_config():
    """FP8Config.include_linear_layers swaps matching Linear.Config for FP8Linear."""
    model_config = _model_config("llama3_debug_float8.yaml")
    assert has_quantization(model_config)
    converted = [
        fqn
        for fqn, lc, _parent, _attr in model_config.traverse(Linear.Config)
        if isinstance(lc, FP8Linear.Config)
    ]
    assert converted, "expected at least one Linear.Config swapped to FP8Linear.Config"


def test_float8_respects_the_include_regex():
    """Layers outside include_linear_layers keep their plain Linear.Config."""
    model_config = _model_config("llama3_debug_float8.yaml")
    # llama3_debug_float8.yaml includes only .*\.(w1|w2|w3|wo)$, so the
    # attention q/k/v projections and the lm_head must be untouched.
    for fqn, lc, _parent, _attr in model_config.traverse(Linear.Config):
        if fqn.split(".")[-1] not in ("w1", "w2", "w3", "wo"):
            assert not isinstance(lc, FP8Linear.Config), fqn
