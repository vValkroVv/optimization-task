import ast
import concurrent.futures
import logging
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Optional, Tuple, Union

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).parents[3]
_LOCAL = _ROOT / "python/sglang/srt/models/gigachat35.py"
_MODULE_NAME = "test_gigachat35_subset"
_TARGET_SYMBOLS = {
    "_BaseGatedNormTorchCompile",
    "Gigachat35GatedNorm",
    "Gigachat35GatedNormMixin",
    "Gigachat35GatedNormTorchCompile",
    "Gigachat35MoE",
    "Gigachat35RMSNorm",
    "Gigachat35ZeroCenteredRMSNorm",
    "Gigachat35ZeroCenteredGatedNorm",
    "Gigachat35ZeroCenteredGatedNormTorchCompile",
    "Gigachat35GatedNormTritonFused",
    "resolve_gated_norm_class",
    "resolve_norm_class",
    "build_norm",
    "is_gigachat35_dsa_attention_layer",
    "Gigachat35AttentionGQA",
    "Gigachat35AttentionMLA",
    "Gigachat35DecoderLayer",
    "Gigachat35GatedDeltaNet",
    "Gigachat35HybridGDNWeightLoaderMixin",
    "Gigachat35PassthroughNorm",
    "Gigachat35MlpPrepNorm",
    "Gigachat35TopK",
}
_TARGET_ASSIGNMENTS = {"_GATED_NORM_IMPL_MAP"}


def _install_namespace_package(name: str, path: Path) -> None:
    module = ModuleType(name)
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    sys.modules[name] = module


# gigachat35.py pulls in sglang.srt.distributed (orjson, NCCL, ...) at import time.
# Fake the heavyweight package __init__ files so the leaf dependency modules below
# can be imported directly without triggering the full sglang runtime.
sys.path.insert(0, str(_ROOT / "python"))
_install_namespace_package("sglang", _ROOT / "python/sglang")
_install_namespace_package("sglang.srt", _ROOT / "python/sglang/srt")
_install_namespace_package("sglang.srt.models", _ROOT / "python/sglang/srt/models")

import sglang.srt.configs.hybrid_gdn as _hybrid_gdn  # noqa: E402
from sglang.srt.configs.hybrid_gdn import (  # noqa: E402
    get_gigachat35_mtp_attention_layer_id,
    get_hybrid_gdn_config_view,
    gigachat35_mtp_uses_linear_attention,
    is_gigachat35_linear_attention_layer,
    is_linear_attention_layer as default_is_linear_attention_layer,
)
from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.layers import layernorm as _layernorm  # noqa: E402
from sglang.srt.layers.layernorm import (  # noqa: E402
    GemmaRMSNorm,
    RMSNorm,
    _compiled_gated_norm,
    _compiled_zero_centered_gated_norm,
)
from sglang.srt.layers.moe.topk import (  # noqa: E402
    StandardTopKOutput,
    TopKOutputChecker,
    TopKOutputFormat,
)
from sglang.srt.layers.quantization.fp8_utils import quant_weight_ue8m0  # noqa: E402
from sglang.srt.layers.utils import MultiPlatformOp, get_layer_id  # noqa: E402
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (  # noqa: E402
    DeepseekV2WeightLoaderMixin,
    NextNDisabledConfig,
    NextNEnabledConfig,
)
from sglang.srt.models.deepseek_common.utils import (  # noqa: E402
    enable_nextn_moe_bf16_cast_to_fp8,
)
from sglang.srt.models.qwen3_next import (  # noqa: E402
    Qwen3GatedDeltaNet,
    _apply_sigmoid_gate_fp32,
)
from sglang.srt.configs.model_config import (  # noqa: E402
    parse_dense_attention_layers,
)
from sglang.srt.utils import add_prefix  # noqa: E402

if _layernorm._is_triton_available:
    _TARGET_SYMBOLS.add("Gigachat35ZeroCenteredGatedNormTriton")


class _StubBackend:
    def __init__(
        self,
        *,
        triton: bool = False,
        flashinfer_trtllm: bool = False,
        flashinfer_mxfp4: bool = False,
    ) -> None:
        self._triton = triton
        self._flashinfer_trtllm = flashinfer_trtllm
        self._flashinfer_mxfp4 = flashinfer_mxfp4

    def is_triton_kernels(self) -> bool:
        return self._triton

    def is_flashinfer_trtllm(self) -> bool:
        return self._flashinfer_trtllm

    def is_flashinfer_mxfp4(self) -> bool:
        return self._flashinfer_mxfp4


class _RunnerBackend:
    def __init__(self, value: str):
        self.value = value

    def is_auto(self) -> bool:
        return self.value == "auto"

    def is_triton(self) -> bool:
        return self.value == "triton"

    def is_triton_llmf(self) -> bool:
        return self.value == "triton_llmf"

    def is_triton_kernels(self) -> bool:
        return self.value == "triton_kernel"

    def is_deep_gemm(self) -> bool:
        return self.value == "deep_gemm"

    def is_cutlass(self) -> bool:
        return self.value == "cutlass"

    def is_flashinfer_trtllm(self) -> bool:
        return self.value == "flashinfer_trtllm"


class _FakeQuantConfig:
    def __init__(self, name: str, **attrs):
        self._name = name
        for key, value in attrs.items():
            setattr(self, key, value)

    def get_name(self) -> str:
        return self._name


class _FakeDeepseekV2MLP:
    def __init__(self):
        self.swiglu_limit = 0.0


class _FakeDeepseekV2MoE:
    def __init__(
        self,
        config,
        layer_id,
        quant_config=None,
        prefix="",
        alt_stream=None,
        is_nextn=False,
        is_deepseek_v4=False,
        dsa_enable_prefill_cp=False,
        mla_enable_prefill_cp=False,
    ):
        self.config = config
        self.layer_id = layer_id
        self.quant_config = quant_config
        self.prefix = prefix
        self.alt_stream = alt_stream
        self.is_nextn = is_nextn
        self.shared_experts = getattr(config, "_test_shared_experts", None)
        self.experts = getattr(config, "_test_experts", None)
        self.num_fused_shared_experts = getattr(config, "num_fused_shared_experts", 0)
        self.topk = getattr(config, "_test_topk", None)


class _FakeDeepseekV2AttentionMLA(nn.Module):
    def __init__(
        self,
        config,
        hidden_size,
        num_heads,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        q_lora_rank,
        kv_lora_rank,
        **kwargs,
    ):
        super().__init__()
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        self.num_local_heads = num_heads
        self.v_head_dim = v_head_dim
        self.use_dsa = True
        self.indexer = object()
        self.skip_topk = False
        self.next_skip_topk = False
        self.attn_mqa = SimpleNamespace()


def _target_writes_name(target: ast.expr) -> bool:
    if isinstance(target, ast.Name):
        return target.id in _TARGET_ASSIGNMENTS
    if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
        return target.value.id in _TARGET_ASSIGNMENTS
    return False


def _assignment_writes_target(node: ast.AST) -> bool:
    if isinstance(node, ast.AnnAssign):
        return _target_writes_name(node.target)
    if isinstance(node, ast.Assign):
        return any(_target_writes_name(target) for target in node.targets)
    return False


def default_gigachat35_mtp_is_dsa(config):
    """Stand-in for configs.model_config.gigachat35_mtp_is_dsa (importing the real
    one would drag in the full sglang runtime). ModelConfig resolves nextn_is_dsa
    into a concrete bool on the config; this just reads it (index_topk None = dense)."""
    if getattr(config, "index_topk", None) is None:
        return False
    return bool(getattr(config, "nextn_is_dsa", None))


class _FakeLinear:
    def __init__(self, in_features: int, out_features: int, **kwargs) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs
        self.weight = torch.randn(out_features, in_features)

    def __call__(self, x):
        return F.linear(x, self.weight.to(x.dtype)), None


class _FakeQwen3Attention:
    def __init__(self, **kwargs) -> None:
        self.init_kwargs = kwargs
        self.total_num_heads = kwargs["num_heads"]
        self.num_heads = kwargs["num_heads"]
        self.num_kv_heads = kwargs["num_kv_heads"]
        self.head_dim = kwargs["head_dim"]
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv_proj = _FakeLinear(
            kwargs["hidden_size"], self.q_size + 2 * self.kv_size
        )
        self.o_proj = _FakeLinear(self.q_size, kwargs["hidden_size"])
        self.rotary_emb = lambda positions, q, k: (q, k)
        self.attn = None
        self.q_norm = "q_norm"
        self.k_norm = "k_norm"


class _FakeDeepseekV2DecoderLayer(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.init_kwargs = kwargs
        self.self_attn = "mla"
        self.mlp = None
        self.layer_communicator = SimpleNamespace(
            qkv_latent_func="latent", _context=SimpleNamespace()
        )


class _FakeFusedMoE:
    @staticmethod
    def make_expert_params_mapping(**kwargs):
        return []

    @staticmethod
    def make_expert_input_scale_params_mapping(**kwargs):
        return []


def _fake_maybe_executor_submit(
    *, executor, futures, use_async, func, func_args=(), func_kwargs=None
):
    func(*func_args, **(func_kwargs or {}))


def _fake_default_weight_loader(param, loaded_weight):
    param.default_loaded.append(loaded_weight)


# Globals the gigachat35.py source expects. The AST subset we exec below references
# these as free variables, so we pre-populate them — substituting lightweight
# stand-ins for the runtime-only pieces (the MoE runner backend, tqdm).
_GIGACHAT35_GLOBALS = {
    "torch": torch,
    "F": F,
    "nn": nn,
    "Optional": Optional,
    "Tuple": Tuple,
    "Union": Union,
    "PretrainedConfig": object,
    "GemmaRMSNorm": GemmaRMSNorm,
    "RMSNorm": RMSNorm,
    "MultiPlatformOp": MultiPlatformOp,
    "Qwen3GatedDeltaNet": Qwen3GatedDeltaNet,
    "_apply_sigmoid_gate_fp32": _apply_sigmoid_gate_fp32,
    "Qwen3Attention": _FakeQwen3Attention,
    "ColumnParallelLinear": _FakeLinear,
    "FusedMoE": _FakeFusedMoE,
    "add_prefix": add_prefix,
    "get_parallel": lambda: SimpleNamespace(attn_tp_rank=0, attn_tp_size=1),
    "concurrent": concurrent,
    "should_async_load": lambda weight: False,
    "maybe_executor_submit": _fake_maybe_executor_submit,
    "default_weight_loader": _fake_default_weight_loader,
    "log_info_on_rank0": lambda logger, msg: None,
    "NVFP4_CKPT_FP8_ATTN_QUANT_MODULES": ["q_b_proj"],
    "_is_npu": False,
    "DeepseekV2WeightLoaderMixin": DeepseekV2WeightLoaderMixin,
    "NextNDisabledConfig": NextNDisabledConfig,
    "NextNEnabledConfig": NextNEnabledConfig,
    "StandardTopKOutput": StandardTopKOutput,
    "TopKOutputChecker": TopKOutputChecker,
    "TopKOutputFormat": TopKOutputFormat,
    "get_hybrid_gdn_config_view": get_hybrid_gdn_config_view,
    "envs": envs,
    "quant_weight_ue8m0": quant_weight_ue8m0,
    "enable_nextn_moe_bf16_cast_to_fp8": enable_nextn_moe_bf16_cast_to_fp8,
    "get_layer_id": get_layer_id,
    "is_linear_attention_layer": default_is_linear_attention_layer,
    "get_gigachat35_mtp_attention_layer_id": get_gigachat35_mtp_attention_layer_id,
    "gigachat35_mtp_uses_linear_attention": gigachat35_mtp_uses_linear_attention,
    "is_gigachat35_linear_attention_layer": is_gigachat35_linear_attention_layer,
    "gigachat35_mtp_is_dsa": default_gigachat35_mtp_is_dsa,
    "parse_dense_attention_layers": parse_dense_attention_layers,
    "get_global_server_args": lambda: SimpleNamespace(gated_norm_impl="native"),
    "get_moe_runner_backend": lambda: _StubBackend(),
    "get_moe_a2a_backend": lambda: SimpleNamespace(
        is_deepep=lambda: False,
        is_mooncake=lambda: False,
    ),
    "deep_gemm_wrapper": SimpleNamespace(ENABLE_JIT_DEEPGEMM=False),
    "deepseek_v2": SimpleNamespace(
        DeepseekV2AttentionMLA=_FakeDeepseekV2AttentionMLA,
        DeepseekV2MoE=_FakeDeepseekV2MoE,
        DeepseekV2MLP=_FakeDeepseekV2MLP,
        DeepseekV2DecoderLayer=_FakeDeepseekV2DecoderLayer,
    ),
    "tqdm": SimpleNamespace(tqdm=lambda seq, desc=None: seq),
    "logger": logging.getLogger(_MODULE_NAME),
    "_GATED_NORM_LOW_RANK": 16,
    "_is_triton_available": _layernorm._is_triton_available,
    "_compiled_gated_norm": _compiled_gated_norm,
    "_compiled_zero_centered_gated_norm": _compiled_zero_centered_gated_norm,
    "triton_zero_centered_gated_norm": getattr(
        _layernorm,
        "triton_zero_centered_gated_norm",
        None,
    ),
}

if _layernorm._is_triton_available:
    _GIGACHAT35_GLOBALS["_gigachat35_fused_gated_norm"] = getattr(
        _layernorm,
        "_gigachat35_fused_gated_norm",
        None,
    )


def _load_gigachat35_subset() -> ModuleType:
    """Compile just the _TARGET_SYMBOLS definitions from gigachat35.py.

    Importing the module normally would run its top-level
    ``from sglang.srt.distributed import ...``, dragging in the full runtime.
    Instead we parse the source, keep only the defs we test, and exec them
    against _GIGACHAT35_GLOBALS.
    """
    tree = ast.parse(_LOCAL.read_text(encoding="utf-8"), filename=str(_LOCAL))
    selected = []
    selected_names = set()
    selected_assignments = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if node.name in _TARGET_SYMBOLS:
                selected.append(node)
                selected_names.add(node.name)
        elif _assignment_writes_target(node):
            selected.append(node)
            selected_assignments.add("_GATED_NORM_IMPL_MAP")
        elif isinstance(node, ast.If):
            nested_names = {
                child.name
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.ClassDef))
                and child.name in _TARGET_SYMBOLS
            }
            nested_assignments = {
                "_GATED_NORM_IMPL_MAP"
                for child in node.body + node.orelse
                if _assignment_writes_target(child)
            }
            if nested_names or nested_assignments:
                selected.append(node)
                selected_names.update(nested_names)
                selected_assignments.update(nested_assignments)

    missing = _TARGET_SYMBOLS - selected_names
    if missing:
        raise AssertionError(f"gigachat35.py is missing expected symbols: {sorted(missing)}")
    missing_assignments = _TARGET_ASSIGNMENTS - selected_assignments
    if missing_assignments:
        raise AssertionError(
            f"gigachat35.py is missing expected assignments: {sorted(missing_assignments)}"
        )

    subset = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                *selected,
            ],
            type_ignores=[],
        )
    )
    module = ModuleType(_MODULE_NAME)
    module.__dict__.update(_GIGACHAT35_GLOBALS)
    exec(compile(subset, str(_LOCAL), "exec"), module.__dict__)
    return module


_gigachat35 = _load_gigachat35_subset()

# Only the symbols defined in gigachat35.py need extracting; everything else is
# already a module-level name from the imports above.
Gigachat35AttentionGQA = _gigachat35.Gigachat35AttentionGQA
Gigachat35AttentionMLA = _gigachat35.Gigachat35AttentionMLA
Gigachat35DecoderLayer = _gigachat35.Gigachat35DecoderLayer
Gigachat35GatedDeltaNet = _gigachat35.Gigachat35GatedDeltaNet
Gigachat35GatedNorm = _gigachat35.Gigachat35GatedNorm
Gigachat35GatedNormTorchCompile = _gigachat35.Gigachat35GatedNormTorchCompile
Gigachat35HybridGDNWeightLoaderMixin = _gigachat35.Gigachat35HybridGDNWeightLoaderMixin
Gigachat35MlpPrepNorm = _gigachat35.Gigachat35MlpPrepNorm
Gigachat35MoE = _gigachat35.Gigachat35MoE
Gigachat35PassthroughNorm = _gigachat35.Gigachat35PassthroughNorm
Gigachat35RMSNorm = _gigachat35.Gigachat35RMSNorm
Gigachat35TopK = _gigachat35.Gigachat35TopK
Gigachat35ZeroCenteredGatedNorm = _gigachat35.Gigachat35ZeroCenteredGatedNorm
Gigachat35ZeroCenteredGatedNormTorchCompile = (
    _gigachat35.Gigachat35ZeroCenteredGatedNormTorchCompile
)
Gigachat35ZeroCenteredGatedNormTriton = getattr(
    _gigachat35,
    "Gigachat35ZeroCenteredGatedNormTriton",
    None,
)
Gigachat35ZeroCenteredRMSNorm = _gigachat35.Gigachat35ZeroCenteredRMSNorm
build_norm = _gigachat35.build_norm
is_gigachat35_dsa_attention_layer = _gigachat35.is_gigachat35_dsa_attention_layer
resolve_norm_class = _gigachat35.resolve_norm_class

HIDDEN = 8
BF16_ASSERT_CLOSE_ATOL = 2e-2
BF16_ASSERT_CLOSE_RTOL = 2e-2
DEFAULT_ASSERT_CLOSE_ATOL = 1e-6
DEFAULT_ASSERT_CLOSE_RTOL = 1e-6
FP8_ROUNDTRIP_ATOL = 8e-2
FP8_ROUNDTRIP_RTOL = 8e-2
BLOCK_FP8_MOE_CLIPPING_ATOL = 5e-1
BLOCK_FP8_MOE_CLIPPING_RTOL = 8e-2


def _cfg(**kwargs):
    return SimpleNamespace(**kwargs)


class _AttrDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    if actual.dtype == torch.bfloat16 or expected.dtype == torch.bfloat16:
        torch.testing.assert_close(
            actual,
            expected,
            atol=BF16_ASSERT_CLOSE_ATOL,
            rtol=BF16_ASSERT_CLOSE_RTOL,
        )
    else:
        torch.testing.assert_close(
            actual,
            expected,
            atol=DEFAULT_ASSERT_CLOSE_ATOL,
            rtol=DEFAULT_ASSERT_CLOSE_RTOL,
        )


def _manual_gemma_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    output = x.float()
    variance = output.pow(2).mean(dim=-1, keepdim=True)
    output = output * torch.rsqrt(variance + eps)
    output = output * (1.0 + weight.float())
    return output.to(dtype=x.dtype)


def _manual_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    output = x.float()
    variance = output.pow(2).mean(dim=-1, keepdim=True)
    output = output * torch.rsqrt(variance + eps)
    output = output * weight.float()
    return output.to(dtype=x.dtype)


def _manual_gigachat35_gated_norm(
    norm,
    x: torch.Tensor,
    residual: torch.Tensor | None = None,
    post_residual_addition: torch.Tensor | None = None,
    *,
    zero_centered: bool,
):
    if residual is None:
        merged = x
        residual_out = None
    else:
        merged = x + residual
        if post_residual_addition is not None:
            merged = merged + post_residual_addition
        residual_out = merged.clone()

    norm_fn = _manual_gemma_rms_norm if zero_centered else _manual_rms_norm
    normed = norm_fn(merged, norm.weight, norm.variance_epsilon)
    gate_hidden = F.linear(normed.float(), norm.gate_up_projection.weight.float())
    gate_hidden = F.silu(gate_hidden)
    gate_hidden = F.linear(gate_hidden, norm.gate_down_projection.weight.float())
    output = (
        normed.float()
        * norm.layernorm_gating_weight
        * torch.sigmoid(gate_hidden)
    ).to(dtype=x.dtype)

    if residual_out is None:
        return output
    return output, residual_out


def _manual_gigachat35_zero_centered_gated_norm(
    norm: Gigachat35ZeroCenteredGatedNorm,
    x: torch.Tensor,
    residual: torch.Tensor | None = None,
    post_residual_addition: torch.Tensor | None = None,
):
    return _manual_gigachat35_gated_norm(
        norm,
        x,
        residual=residual,
        post_residual_addition=post_residual_addition,
        zero_centered=True,
    )


def _manual_swiglu_clamp(
    gate_up: torch.Tensor,
    limit: float,
    probs: torch.Tensor | None = None,
) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    output = F.silu(gate.float().clamp(max=limit)) * up.float().clamp(
        min=-limit,
        max=limit,
    )
    if probs is not None:
        output = output * probs.reshape(-1, 1).float()
    return output


def _repeat_block_scales(
    scales: torch.Tensor,
    rows: int,
    cols: int,
    block_shape: list[int],
) -> torch.Tensor:
    block_n, block_k = block_shape
    return scales.repeat_interleave(block_n, dim=-2).repeat_interleave(
        block_k,
        dim=-1,
    )[..., :rows, :cols]


def _dequant_block_fp8_weight(
    weight: torch.Tensor,
    scales: torch.Tensor,
    block_shape: list[int],
) -> torch.Tensor:
    expanded_scales = _repeat_block_scales(
        scales,
        rows=weight.shape[-2],
        cols=weight.shape[-1],
        block_shape=block_shape,
    )
    return weight.float() * expanded_scales


def _manual_block_fp8_clipped_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    block_shape: list[int],
    swiglu_limit: float,
) -> torch.Tensor:
    w1_dequant = _dequant_block_fp8_weight(w1, w1_scale, block_shape)
    w2_dequant = _dequant_block_fp8_weight(w2, w2_scale, block_shape)
    output = torch.zeros(
        hidden_states.shape[0],
        topk_ids.shape[1],
        w2.shape[1],
        device=hidden_states.device,
        dtype=torch.float32,
    )

    for token_idx in range(hidden_states.shape[0]):
        token = hidden_states[token_idx].float()
        for topk_idx in range(topk_ids.shape[1]):
            expert_idx = int(topk_ids[token_idx, topk_idx].item())
            gate_up = token @ w1_dequant[expert_idx].transpose(0, 1)
            activated = _manual_swiglu_clamp(
                gate_up.unsqueeze(0),
                swiglu_limit,
            ).squeeze(0)
            expert_output = activated @ w2_dequant[expert_idx].transpose(0, 1)
            output[token_idx, topk_idx] = (
                expert_output * topk_weights[token_idx, topk_idx].float()
            )

    return output.sum(dim=1).to(dtype=hidden_states.dtype)


def _make_block_fp8_moe_case(
    device: str,
    *,
    tokens: int = 4,
    hidden: int = 512,
    intermediate: int = 256,
    experts: int = 2,
):
    # DeepSeekV3/Gigachat35-style MoE has moe_intermediate_size=2048; with
    # TP/EP 8 this gives a 256-wide local down input. Hidden=512 keeps the
    # test small while exercising multi-block [128, 128] fp8 scales.
    block_shape = [128, 128]
    block_n, block_k = block_shape
    dtype = torch.bfloat16

    hidden_states = torch.linspace(
        -0.2,
        0.2,
        steps=tokens * hidden,
        device=device,
        dtype=torch.float32,
    ).reshape(tokens, hidden).to(dtype)
    w1 = (
        torch.linspace(
            -0.75,
            0.75,
            steps=experts * 2 * intermediate * hidden,
            device=device,
            dtype=torch.float32,
        ).reshape(experts, 2 * intermediate, hidden)
        * 0.1
    ).to(torch.float8_e4m3fn)
    w2 = (
        torch.linspace(
            0.6,
            -0.6,
            steps=experts * hidden * intermediate,
            device=device,
            dtype=torch.float32,
        ).reshape(experts, hidden, intermediate)
        * 0.1
    ).to(torch.float8_e4m3fn)

    w1_scale_shape = (
        experts,
        (2 * intermediate + block_n - 1) // block_n,
        (hidden + block_k - 1) // block_k,
    )
    w2_scale_shape = (
        experts,
        (hidden + block_n - 1) // block_n,
        (intermediate + block_k - 1) // block_k,
    )
    w1_scale = torch.linspace(
        0.875,
        1.125,
        steps=w1_scale_shape[0] * w1_scale_shape[1] * w1_scale_shape[2],
        device=device,
        dtype=torch.float32,
    ).reshape(w1_scale_shape)
    w2_scale = torch.linspace(
        1.125,
        0.875,
        steps=w2_scale_shape[0] * w2_scale_shape[1] * w2_scale_shape[2],
        device=device,
        dtype=torch.float32,
    ).reshape(w2_scale_shape)

    topk_weights = torch.linspace(0.5, 1.0, steps=tokens, device=device).reshape(
        tokens,
        1,
    ).to(dtype)
    topk_ids = (
        torch.arange(tokens, device=device, dtype=torch.long).reshape(tokens, 1) % experts
    )
    topk_output = StandardTopKOutput(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        router_logits=torch.zeros(tokens, experts, device=device, dtype=dtype),
    )
    return hidden_states, w1, w2, topk_output, w1_scale, w2_scale, block_shape


class _Affine(nn.Module):
    def __init__(self, scale: float, bias: float):
        super().__init__()
        self.scale = scale
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.bias


def _make_uninitialized_gated_delta_net() -> Gigachat35GatedDeltaNet:
    model = Gigachat35GatedDeltaNet.__new__(Gigachat35GatedDeltaNet)
    nn.Module.__init__(model)
    return model


@pytest.mark.unit
class TestResolveNormClass:
    def test_llamarmsnorm(self):
        assert resolve_norm_class(_cfg(norm_type="LlamaRMSNorm")) is Gigachat35RMSNorm

    def test_zero_centered_rms(self):
        assert (
            resolve_norm_class(_cfg(norm_type="ZeroCenteredRMSNorm"))
            is Gigachat35ZeroCenteredRMSNorm
        )

    def test_zero_centered_gated(self):
        assert (
            resolve_norm_class(_cfg(norm_type="ZeroCenteredGatedNorm"))
            is Gigachat35ZeroCenteredGatedNorm
        )

    def test_gated_norm_impl_map_uses_production_classes(self):
        impl_map = _gigachat35._GATED_NORM_IMPL_MAP

        assert impl_map["native"] is Gigachat35ZeroCenteredGatedNorm
        assert impl_map["torch_compile"] is Gigachat35ZeroCenteredGatedNormTorchCompile
        if Gigachat35ZeroCenteredGatedNormTriton is None:
            assert "triton" not in impl_map
        else:
            assert impl_map["triton"] is Gigachat35ZeroCenteredGatedNormTriton

    def test_zero_centered_gated_torch_compile(self, monkeypatch):
        monkeypatch.setattr(
            _gigachat35,
            "get_global_server_args",
            lambda: SimpleNamespace(gated_norm_impl="torch_compile"),
        )
        assert (
            resolve_norm_class(_cfg(norm_type="ZeroCenteredGatedNorm"))
            is Gigachat35ZeroCenteredGatedNormTorchCompile
        )

    def test_zero_centered_gated_triton(self, monkeypatch):
        if Gigachat35ZeroCenteredGatedNormTriton is None:
            pytest.skip("Triton gated norm implementation is not available")
        monkeypatch.setattr(
            _gigachat35,
            "get_global_server_args",
            lambda: SimpleNamespace(gated_norm_impl="triton"),
        )
        assert (
            resolve_norm_class(_cfg(norm_type="ZeroCenteredGatedNorm"))
            is Gigachat35ZeroCenteredGatedNormTriton
        )

    def test_gated_native(self, monkeypatch):
        monkeypatch.setattr(
            _gigachat35,
            "get_global_server_args",
            lambda: SimpleNamespace(gated_norm_impl="native"),
        )
        assert resolve_norm_class(_cfg(norm_type="GatedNorm")) is Gigachat35GatedNorm

    def test_gated_torch_compile(self, monkeypatch):
        monkeypatch.setattr(
            _gigachat35,
            "get_global_server_args",
            lambda: SimpleNamespace(gated_norm_impl="torch_compile"),
        )
        assert (
            resolve_norm_class(_cfg(norm_type="GatedNorm"))
            is Gigachat35GatedNormTorchCompile
        )

    def test_gated_triton_falls_back_to_torch_compile(self, monkeypatch):
        monkeypatch.setattr(
            _gigachat35,
            "get_global_server_args",
            lambda: SimpleNamespace(gated_norm_impl="triton"),
        )
        assert (
            resolve_norm_class(_cfg(norm_type="GatedNorm"))
            is Gigachat35GatedNormTorchCompile
        )

    def test_gated_triton_fused_uses_plain_scale_kernel_class(self, monkeypatch):
        fused_cls = getattr(_gigachat35, "Gigachat35GatedNormTritonFused", None)
        if fused_cls is None:
            pytest.skip("triton_fused gated norm implementation is not available")
        monkeypatch.setattr(
            _gigachat35,
            "get_global_server_args",
            lambda: SimpleNamespace(gated_norm_impl="triton_fused"),
        )
        prev = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            assert resolve_norm_class(_cfg(norm_type="GatedNorm")) is fused_cls
            # Same bf16-only rule as the zero-centered kernel: any other dtype runs torch_compile.
            torch.set_default_dtype(torch.float32)
            assert (
                resolve_norm_class(_cfg(norm_type="GatedNorm"))
                is Gigachat35GatedNormTorchCompile
            )
        finally:
            torch.set_default_dtype(prev)

    def test_missing_attr_defaults_to_rms(self):
        assert resolve_norm_class(_cfg()) is Gigachat35RMSNorm

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unsupported norm_type"):
            resolve_norm_class(_cfg(norm_type="Bogus"))


@pytest.mark.unit
class TestBuildNorm:
    def test_builds_rms_norm(self):
        norm = build_norm(
            _cfg(norm_type="LlamaRMSNorm", rms_norm_eps=1e-5),
            hidden_size=HIDDEN,
        )
        assert isinstance(norm, Gigachat35RMSNorm)
        assert norm.variance_epsilon == pytest.approx(1e-5)

    def test_builds_zero_centered_rms_norm(self):
        norm = build_norm(
            _cfg(norm_type="ZeroCenteredRMSNorm", rms_norm_eps=1e-4),
            hidden_size=HIDDEN,
        )
        assert isinstance(norm, Gigachat35ZeroCenteredRMSNorm)
        assert norm.variance_epsilon == pytest.approx(1e-4)

    def test_builds_zero_centered_gated_norm_with_default_weight(self):
        norm = build_norm(
            _cfg(norm_type="ZeroCenteredGatedNorm", rms_norm_eps=1e-6),
            hidden_size=HIDDEN,
        )
        assert isinstance(norm, Gigachat35ZeroCenteredGatedNorm)
        assert norm.layernorm_gating_weight == pytest.approx(2.0)

    def test_builds_zero_centered_gated_norm_with_custom_weight(self):
        norm = build_norm(
            _cfg(
                norm_type="ZeroCenteredGatedNorm",
                rms_norm_eps=1e-6,
                layernorm_gating_weight=1.25,
            ),
            hidden_size=HIDDEN,
        )
        assert isinstance(norm, Gigachat35ZeroCenteredGatedNorm)
        assert norm.layernorm_gating_weight == pytest.approx(1.25)

    def test_builds_gated_norm_with_custom_weight(self):
        norm = build_norm(
            _cfg(
                norm_type="GatedNorm",
                rms_norm_eps=1e-6,
                layernorm_gating_weight=1.75,
            ),
            hidden_size=HIDDEN,
        )
        assert isinstance(norm, Gigachat35GatedNorm)
        assert norm.layernorm_gating_weight == pytest.approx(1.75)

    def test_builds_gated_norm_triton_fused_with_custom_weight(self, monkeypatch):
        fused_cls = getattr(_gigachat35, "Gigachat35GatedNormTritonFused", None)
        if fused_cls is None:
            pytest.skip("triton_fused gated norm implementation is not available")
        monkeypatch.setattr(
            _gigachat35,
            "get_global_server_args",
            lambda: SimpleNamespace(gated_norm_impl="triton_fused"),
        )
        prev = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            norm = build_norm(
                _cfg(
                    norm_type="GatedNorm",
                    rms_norm_eps=1e-6,
                    layernorm_gating_weight=1.75,
                ),
                hidden_size=HIDDEN,
            )
        finally:
            torch.set_default_dtype(prev)
        assert isinstance(norm, fused_cls)
        assert norm.layernorm_gating_weight == pytest.approx(1.75)


@pytest.mark.unit
class TestGetMtpAttentionLayerId:
    def test_normal(self):
        assert get_gigachat35_mtp_attention_layer_id(_cfg(num_hidden_layers=28)) == 27

    def test_single_layer(self):
        assert get_gigachat35_mtp_attention_layer_id(_cfg(num_hidden_layers=1)) == 0

    def test_zero(self):
        assert get_gigachat35_mtp_attention_layer_id(_cfg(num_hidden_layers=0)) == 0

    def test_missing_attr(self):
        assert get_gigachat35_mtp_attention_layer_id(_cfg()) == 0


@pytest.mark.unit
class TestIsGigachat35LinearAttentionLayer:
    def test_layer_types_linear(self):
        cfg = _cfg(layer_types=["linear_attention", "full_attention"])
        assert is_gigachat35_linear_attention_layer(cfg, 0) is True

    def test_layer_types_full(self):
        cfg = _cfg(layer_types=["linear_attention", "full_attention"])
        assert is_gigachat35_linear_attention_layer(cfg, 1) is False

    def test_layer_types_have_priority_over_other_config_fields(self):
        cfg = _cfg(
            layer_types=["full_attention"],
            linear_layer_ids=[0],
            full_attention_layer_ids=[],
        )
        assert is_gigachat35_linear_attention_layer(cfg, 0) is False

    def test_layer_types_out_of_range_falls_through(self):
        cfg = _cfg(layer_types=["linear_attention"], linear_layer_ids=[5])
        assert is_gigachat35_linear_attention_layer(cfg, 5) is True

    def test_linear_layer_ids_in(self):
        cfg = _cfg(linear_layer_ids=[0, 2, 4])
        assert is_gigachat35_linear_attention_layer(cfg, 2) is True

    def test_linear_layer_ids_not_in(self):
        cfg = _cfg(linear_layer_ids=[0, 2, 4])
        assert is_gigachat35_linear_attention_layer(cfg, 3) is False

    def test_full_attention_layer_ids_not_in(self):
        cfg = _cfg(full_attention_layer_ids=[1, 3])
        assert is_gigachat35_linear_attention_layer(cfg, 0) is True

    def test_full_attention_layer_ids_in(self):
        cfg = _cfg(full_attention_layer_ids=[1, 3])
        assert is_gigachat35_linear_attention_layer(cfg, 1) is False

    def test_linear_attention_type_not_in_full(self):
        cfg = _cfg(linear_attention_type="gdn", full_attention_layers=[2, 5])
        assert is_gigachat35_linear_attention_layer(cfg, 0) is True

    def test_linear_attention_type_in_full(self):
        cfg = _cfg(linear_attention_type="gdn", full_attention_layers=[2, 5])
        assert is_gigachat35_linear_attention_layer(cfg, 2) is False

    def test_linear_attention_type_without_full_attention_layers(self):
        cfg = _cfg(linear_attention_type="gdn")
        assert is_gigachat35_linear_attention_layer(cfg, 7) is True

    def test_delegates_to_hybrid_gdn_helper_for_in_range_layers(self, monkeypatch):
        seen = []

        def fake_is_linear_attention_layer(config, layer_id):
            seen.append((config, layer_id))
            return True

        monkeypatch.setattr(
            _hybrid_gdn,
            "is_linear_attention_layer",
            fake_is_linear_attention_layer,
        )
        cfg = _cfg(num_hidden_layers=4)
        assert is_gigachat35_linear_attention_layer(cfg, 2) is True
        assert seen == [(cfg, 2)]

    def test_no_attrs_out_of_range(self):
        assert is_gigachat35_linear_attention_layer(_cfg(num_hidden_layers=4), 10) is False


@pytest.mark.unit
class TestIsGigachat35DsaAttentionLayer:
    def test_selects_exact_dense_attention_layers(self):
        cfg = _cfg(
            num_hidden_layers=12,
            attention_type="LlamaLatentSparseAttention",
            index_topk=8,
            linear_attention_type="Qwen3NextGatedDeltaNet",
            full_attention_layers=[3, 7, 11],
            dense_attention_layers=[7],
        )

        assert is_gigachat35_dsa_attention_layer(cfg, 3) is True
        assert is_gigachat35_dsa_attention_layer(cfg, 7) is False
        assert is_gigachat35_dsa_attention_layer(cfg, 11) is True
        assert is_gigachat35_dsa_attention_layer(cfg, 0) is False

    def test_non_hybrid_explicit_layers(self):
        cfg = _cfg(
            num_hidden_layers=4,
            attention_type="LlamaLatentSparseAttention",
            index_topk=8,
            linear_attention_type=None,
            dense_attention_layers=[1],
        )

        assert is_gigachat35_dsa_attention_layer(cfg, 0) is True
        assert is_gigachat35_dsa_attention_layer(cfg, 1) is False
        assert is_gigachat35_dsa_attention_layer(cfg, 2) is True

    def test_without_indexer_is_dense(self):
        cfg = _cfg(
            num_hidden_layers=4,
            attention_type="LlamaLatentSparseAttention",
            dense_attention_layers=[1],
        )
        assert is_gigachat35_dsa_attention_layer(cfg, 3) is False

    @pytest.mark.parametrize("value", [[-1], [True], [1.5], "1", ()])
    def test_rejects_invalid_dense_attention_layers(self, value):
        cfg = _cfg(
            num_hidden_layers=4,
            attention_type="LlamaLatentSparseAttention",
            index_topk=8,
            linear_attention_type=None,
            dense_attention_layers=value,
        )
        with pytest.raises(ValueError, match="dense_attention_layers"):
            is_gigachat35_dsa_attention_layer(cfg, 0)

    def test_rejects_dense_layer_outside_full_attention_layers(self):
        cfg = _cfg(
            num_hidden_layers=4,
            attention_type="LlamaLatentSparseAttention",
            index_topk=8,
            linear_attention_type="Qwen3NextGatedDeltaNet",
            full_attention_layers=[1, 3],
            dense_attention_layers=[2],
        )
        with pytest.raises(ValueError, match="subset of full_attention_layers"):
            is_gigachat35_dsa_attention_layer(cfg, 0)


@pytest.mark.unit
class TestGigachat35AttentionMlaDenseLayers:
    @staticmethod
    def _build(layer_id, *, is_nextn=False, nextn_is_dsa=None):
        config = _cfg(
            num_hidden_layers=12,
            attention_type="LlamaLatentSparseAttention",
            index_topk=8,
            linear_attention_type="Qwen3NextGatedDeltaNet",
            full_attention_layers=[3, 7, 11],
            dense_attention_layers=[7],
            norm_type="LlamaRMSNorm",
            use_mla_scaling_factor=False,
            gated_attention=False,
            # MTP DSA-ness is resolved into nextn_is_dsa by ModelConfig; the layer
            # reads that concrete value (there is no structural guess from base layers).
            nextn_is_dsa=nextn_is_dsa,
        )
        return Gigachat35AttentionMLA(
            config=config,
            hidden_size=16,
            num_heads=2,
            qk_nope_head_dim=4,
            qk_rope_head_dim=2,
            v_head_dim=4,
            q_lora_rank=8,
            kv_lora_rank=8,
            layer_id=layer_id,
            is_nextn=is_nextn,
        )

    def test_explicit_dense_layer_removes_indexer(self):
        attention = self._build(7)

        assert attention.use_dsa is False
        assert not hasattr(attention, "indexer")
        assert attention.skip_topk is None
        assert attention.next_skip_topk is None

    def test_other_full_attention_layer_keeps_indexer(self):
        attention = self._build(3)

        assert attention.use_dsa is True
        assert hasattr(attention, "indexer")

    def test_mtp_dsa_follows_resolved_nextn_is_dsa(self):
        attention = self._build(0, is_nextn=True, nextn_is_dsa=True)

        assert attention.use_dsa is True
        assert hasattr(attention, "indexer")

    def test_mtp_dense_follows_resolved_nextn_is_dsa(self):
        attention = self._build(0, is_nextn=True, nextn_is_dsa=False)

        assert attention.use_dsa is False
        assert not hasattr(attention, "indexer")
_ANCHOR_FULL = [1, 3, 5, 7, 9, 11]
_ANCHOR_LINEAR = [0, 2, 4, 6, 8, 10]


def _hybrid_cfg(full_attention_layers, **kwargs):
    return _cfg(
        num_hidden_layers=12,
        linear_attention_type="Qwen3NextGatedDeltaNet",
        full_attention_layers=full_attention_layers,
        **kwargs,
    )


@pytest.mark.unit
class TestGigachat35MtpUsesLinearAttention:
    def test_flag_true_overrides_full_attention_anchor(self):
        cfg = _hybrid_cfg(_ANCHOR_FULL, nextn_is_linear=True)
        assert gigachat35_mtp_uses_linear_attention(cfg) is True

    def test_flag_false_overrides_linear_anchor(self):
        cfg = _hybrid_cfg(_ANCHOR_LINEAR, nextn_is_linear=False)
        assert gigachat35_mtp_uses_linear_attention(cfg) is False

    def test_flag_true_without_linear_attention_type_is_noop(self):
        cfg = _cfg(
            num_hidden_layers=12,
            linear_attention_type=None,
            nextn_is_linear=True,
        )
        assert gigachat35_mtp_uses_linear_attention(cfg) is False

    def test_legacy_checkpoint_falls_back_to_linear_anchor(self):
        assert gigachat35_mtp_uses_linear_attention(_hybrid_cfg(_ANCHOR_LINEAR)) is True

    def test_legacy_checkpoint_falls_back_to_full_anchor(self):
        assert gigachat35_mtp_uses_linear_attention(_hybrid_cfg(_ANCHOR_FULL)) is False

    def test_non_hybrid_model_is_never_linear(self):
        cfg = _cfg(num_hidden_layers=12, linear_attention_type=None)
        assert gigachat35_mtp_uses_linear_attention(cfg) is False


@pytest.fixture
def passthrough():
    return Gigachat35PassthroughNorm()


@pytest.mark.unit
class TestGigachat35PassthroughNorm:
    def test_no_residual_returns_x_x(self, passthrough):
        x = torch.randn(4, HIDDEN)
        out, residual = passthrough(x)
        assert torch.equal(out, x)
        assert torch.equal(residual, x)

    def test_with_residual(self, passthrough):
        x = torch.randn(4, HIDDEN)
        residual = torch.randn(4, HIDDEN)
        out, merged = passthrough(x, residual=residual)
        expected = x + residual
        assert torch.equal(out, expected)
        assert torch.equal(merged, expected)

    def test_post_residual_addition_without_residual(self, passthrough):
        x = torch.randn(4, HIDDEN)
        post = torch.randn(4, HIDDEN)
        out, residual = passthrough(x, post_residual_addition=post)
        expected = x + post
        assert torch.equal(out, expected)
        assert torch.equal(residual, expected)

    def test_post_residual_then_residual(self, passthrough):
        x = torch.randn(4, HIDDEN)
        residual = torch.randn(4, HIDDEN)
        post = torch.randn(4, HIDDEN)
        out, merged = passthrough(
            x,
            residual=residual,
            post_residual_addition=post,
        )
        expected = x + post + residual
        assert torch.equal(out, expected)
        assert torch.equal(merged, expected)


@pytest.mark.unit
class TestGigachat35MlpPrepNorm:
    def test_returns_merged_tensor_when_no_layernorms_are_present(self):
        norm = Gigachat35MlpPrepNorm(pre_layernorm=None, post_layernorm=None)
        x = torch.randn(3, HIDDEN)
        residual = torch.randn(3, HIDDEN)
        post = torch.randn(3, HIDDEN)
        out, merged = norm(x, residual=residual, post_residual_addition=post)
        expected = x + post + residual
        assert torch.equal(out, expected)
        assert torch.equal(merged, expected)

    def test_applies_post_then_add_then_residual_then_pre(self):
        norm = Gigachat35MlpPrepNorm(
            pre_layernorm=_Affine(scale=2.0, bias=-1.0),
            post_layernorm=_Affine(scale=-0.5, bias=0.25),
        )
        x = torch.tensor([[1.0, -2.0, 3.0, -4.0]], dtype=torch.float32)
        residual = torch.tensor([[0.2, 0.4, 0.6, 0.8]], dtype=torch.float32)
        post = torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float32)

        out, merged = norm(x, residual=residual, post_residual_addition=post)

        post_normed = x * -0.5 + 0.25
        expected_merged = post_normed + post + residual
        expected_out = expected_merged * 2.0 - 1.0

        _assert_close(out, expected_out)
        _assert_close(merged, expected_merged)

    def test_applies_pre_layernorm_to_x_when_residual_is_missing(self):
        norm = Gigachat35MlpPrepNorm(
            pre_layernorm=_Affine(scale=0.5, bias=3.0),
            post_layernorm=None,
        )
        x = torch.tensor([[1.5, -1.5]], dtype=torch.float32)
        out, merged = norm(x)
        _assert_close(merged, x)
        _assert_close(out, x * 0.5 + 3.0)


@pytest.fixture
def gated_norm():
    norm = Gigachat35ZeroCenteredGatedNorm(
        hidden_size=HIDDEN,
        eps=1e-6,
        layernorm_gating_weight=2.0,
    )
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(-0.2, 0.2, steps=HIDDEN))
        norm.gate_up_projection.weight.copy_(
            torch.linspace(-0.3, 0.3, steps=16 * HIDDEN).reshape(16, HIDDEN)
        )
        norm.gate_down_projection.weight.copy_(
            torch.linspace(-0.4, 0.4, steps=HIDDEN * 16).reshape(HIDDEN, 16)
        )
    return norm


def _fill_gated_norm_weights(norm) -> None:
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.75, 1.25, steps=HIDDEN))
        norm.gate_up_projection.weight.copy_(
            torch.linspace(-0.3, 0.3, steps=16 * HIDDEN).reshape(16, HIDDEN)
        )
        norm.gate_down_projection.weight.copy_(
            torch.linspace(-0.4, 0.4, steps=HIDDEN * 16).reshape(HIDDEN, 16)
        )


def _replace_parameter(module: nn.Module, name: str, value: torch.Tensor) -> None:
    setattr(
        module,
        name,
        nn.Parameter(value.detach().clone(), requires_grad=False),
    )


def _make_gated_norm_parity_module(
    norm_cls,
    *,
    zero_centered: bool,
    precision: str,
    device: str,
    kernel_weight_dtype: torch.dtype | None = None,
):
    if precision == "fp8" and getattr(torch, "float8_e4m3fn", None) is None:
        pytest.skip("torch.float8_e4m3fn is not available")

    norm = norm_cls(
        hidden_size=HIDDEN,
        eps=1e-6,
        layernorm_gating_weight=1.375,
    )
    norm_weight = (
        torch.linspace(-0.25, 0.25, steps=HIDDEN)
        if zero_centered
        else torch.linspace(0.75, 1.25, steps=HIDDEN)
    )
    gate_up_weight = torch.linspace(-0.3125, 0.3125, steps=16 * HIDDEN).reshape(
        16,
        HIDDEN,
    )
    gate_down_weight = torch.linspace(-0.4375, 0.4375, steps=HIDDEN * 16).reshape(
        HIDDEN,
        16,
    )

    gate_dtype = torch.bfloat16
    norm_dtype = torch.bfloat16
    if precision == "fp8":
        gate_dtype = torch.float8_e4m3fn
        # Non-ZC standard RMSNorm multiplies by weight directly, so keep the
        # scale in bf16 there and exercise fp8 storage on the gate projections.
        if zero_centered:
            norm_dtype = torch.float8_e4m3fn

    norm_weight = norm_weight.to(device=device, dtype=norm_dtype)
    gate_up_weight = gate_up_weight.to(device=device, dtype=gate_dtype)
    gate_down_weight = gate_down_weight.to(device=device, dtype=gate_dtype)
    if kernel_weight_dtype is not None:
        norm_weight = norm_weight.to(dtype=kernel_weight_dtype)
        gate_up_weight = gate_up_weight.to(dtype=kernel_weight_dtype)
        gate_down_weight = gate_down_weight.to(dtype=kernel_weight_dtype)

    _replace_parameter(norm, "weight", norm_weight)
    _replace_parameter(norm.gate_up_projection, "weight", gate_up_weight)
    _replace_parameter(norm.gate_down_projection, "weight", gate_down_weight)
    return norm


def _make_gated_norm_parity_inputs(device: str):
    x = torch.linspace(
        -1.5,
        1.5,
        steps=3 * HIDDEN,
        device=device,
        dtype=torch.float32,
    ).reshape(3, HIDDEN)
    residual = torch.linspace(
        0.75,
        -0.75,
        steps=3 * HIDDEN,
        device=device,
        dtype=torch.float32,
    ).reshape(3, HIDDEN)
    post = torch.linspace(
        -0.25,
        0.25,
        steps=3 * HIDDEN,
        device=device,
        dtype=torch.float32,
    ).reshape(3, HIDDEN)
    return x.to(torch.bfloat16), residual.to(torch.bfloat16), post.to(torch.bfloat16)


def _assert_gated_norm_parity(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    precision: str,
) -> None:
    if precision == "fp8":
        torch.testing.assert_close(
            actual,
            expected,
            atol=FP8_ROUNDTRIP_ATOL,
            rtol=FP8_ROUNDTRIP_RTOL,
        )
    else:
        _assert_close(actual, expected)


@pytest.fixture
def non_zero_centered_gated_norm():
    norm = Gigachat35GatedNorm(
        hidden_size=HIDDEN,
        eps=1e-6,
        layernorm_gating_weight=1.5,
    )
    _fill_gated_norm_weights(norm)
    return norm


@pytest.mark.unit
class TestGigachat35ZeroCenteredGatedNorm:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_forward_native_matches_manual_formula(self, gated_norm, dtype):
        x = torch.linspace(-1.5, 1.5, steps=2 * HIDDEN, dtype=torch.float32).reshape(2, HIDDEN)
        x = x.to(dtype=dtype)

        actual = gated_norm.forward_native(x)
        expected = _manual_gigachat35_zero_centered_gated_norm(gated_norm, x)

        assert actual.dtype == dtype
        _assert_close(actual, expected)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_forward_native_with_residual_and_post_addition_matches_manual_formula(
        self,
        gated_norm,
        dtype,
    ):
        x = torch.linspace(-2.0, 2.0, steps=2 * HIDDEN, dtype=torch.float32).reshape(2, HIDDEN)
        residual = torch.linspace(0.5, -0.5, steps=2 * HIDDEN, dtype=torch.float32).reshape(2, HIDDEN)
        post = torch.full((2, HIDDEN), 0.25, dtype=torch.float32)

        x = x.to(dtype=dtype)
        residual = residual.to(dtype=dtype)
        post = post.to(dtype=dtype)

        actual_out, actual_residual = gated_norm.forward_native(
            x,
            residual=residual,
            post_residual_addition=post,
        )
        expected_out, expected_residual = _manual_gigachat35_zero_centered_gated_norm(
            gated_norm,
            x,
            residual=residual,
            post_residual_addition=post,
        )

        assert actual_out.dtype == dtype
        assert actual_residual.dtype == dtype
        _assert_close(actual_out, expected_out)
        _assert_close(actual_residual, expected_residual)

    def test_post_residual_addition_is_ignored_without_residual(self, gated_norm):
        x = torch.randn(2, HIDDEN, dtype=torch.float32)
        post = torch.randn(2, HIDDEN, dtype=torch.float32)
        without_post = gated_norm.forward_native(x)
        with_post = gated_norm.forward_native(x, post_residual_addition=post)
        _assert_close(without_post, with_post)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_zero_gate_weights_reduce_to_zero_centered_rms_norm(self, gated_norm, dtype):
        with torch.no_grad():
            nn.init.zeros_(gated_norm.gate_up_projection.weight)
            nn.init.zeros_(gated_norm.gate_down_projection.weight)

        x = torch.randn(4, HIDDEN, dtype=dtype)
        actual = gated_norm.forward_native(x)
        expected = _manual_gemma_rms_norm(
            x,
            gated_norm.weight,
            gated_norm.variance_epsilon,
        )

        _assert_close(actual, expected)

    def test_forward_cpu_applies_gate_to_parent_output(self):
        # forward_cpu's job is to wrap the parent (GemmaRMSNorm) cpu kernel with
        # the gigachat3.5 gate. Verify exactly that — comparing against
        # forward_native instead would just test sglang's cpu/native kernel
        # parity, which is unrelated to gigachat3.5.
        torch.manual_seed(0)
        norm = Gigachat35ZeroCenteredGatedNorm(
            hidden_size=HIDDEN, eps=1e-6, layernorm_gating_weight=2.0
        )
        x = torch.randn(3, HIDDEN, dtype=torch.float32)

        actual = norm.forward_cpu(x)
        parent_out = Gigachat35ZeroCenteredRMSNorm.forward_cpu(norm, x)
        expected = norm._apply_gate(parent_out)

        assert actual.dtype == x.dtype
        assert actual.shape == x.shape
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected)

        x_bf16 = torch.randn(3, HIDDEN, dtype=torch.bfloat16)
        out_bf16 = norm.forward_cpu(x_bf16)
        assert out_bf16.dtype == torch.bfloat16
        assert out_bf16.shape == x_bf16.shape
        assert torch.isfinite(out_bf16).all()

    @pytest.mark.parametrize(
        "method_name,parent_output",
        [
            ("forward_cuda", torch.full((3, HIDDEN), 0.5, dtype=torch.float32)),
            ("forward_hip", torch.full((3, HIDDEN), -0.75, dtype=torch.float32)),
            ("forward_npu", torch.full((3, HIDDEN), 1.25, dtype=torch.float32)),
            ("forward_xpu", torch.full((3, HIDDEN), 0.125, dtype=torch.float32)),
        ],
    )
    def test_platform_wrappers_apply_gate_to_parent_output(
        self,
        gated_norm,
        method_name,
        parent_output,
        monkeypatch,
    ):
        x = torch.randn(3, HIDDEN, dtype=torch.float32)

        def fake_parent_forward(self, x, residual=None, post_residual_addition=None):
            return parent_output.clone()

        monkeypatch.setattr(
            Gigachat35ZeroCenteredRMSNorm,
            method_name,
            fake_parent_forward,
        )

        actual = getattr(gated_norm, method_name)(x)
        expected = gated_norm._apply_gate(parent_output)
        _assert_close(actual, expected)

    def test_platform_wrapper_preserves_residual_tuple(self, gated_norm, monkeypatch):
        x = torch.randn(3, HIDDEN, dtype=torch.float32)
        residual = torch.randn(3, HIDDEN, dtype=torch.float32)
        parent_normed = torch.full((3, HIDDEN), 0.25, dtype=torch.float32)
        parent_residual = torch.full((3, HIDDEN), -0.5, dtype=torch.float32)

        def fake_parent_forward(self, x, residual=None, post_residual_addition=None):
            return parent_normed.clone(), parent_residual.clone()

        monkeypatch.setattr(
            Gigachat35ZeroCenteredRMSNorm,
            "forward_cuda",
            fake_parent_forward,
        )

        actual_out, actual_residual = gated_norm.forward_cuda(x, residual=residual)
        expected_out = gated_norm._apply_gate(parent_normed)
        _assert_close(actual_out, expected_out)
        _assert_close(actual_residual, parent_residual)

    def test_large_inputs_stay_finite(self, gated_norm):
        x = torch.tensor(
            [
                [1.0e4, -1.0e4, 8.0e3, -8.0e3, 6.0e3, -6.0e3, 4.0e3, -4.0e3],
                [-9.0e3, 9.0e3, -7.0e3, 7.0e3, -5.0e3, 5.0e3, -3.0e3, 3.0e3],
            ],
            dtype=torch.bfloat16,
        )
        out = gated_norm.forward_native(x)
        assert torch.isfinite(out).all()


@pytest.mark.unit
class TestGigachat35GatedNorm:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_forward_native_matches_manual_formula(
        self,
        non_zero_centered_gated_norm,
        dtype,
    ):
        x = torch.linspace(-1.25, 1.25, steps=2 * HIDDEN, dtype=torch.float32).reshape(
            2,
            HIDDEN,
        )
        x = x.to(dtype=dtype)

        actual = non_zero_centered_gated_norm.forward_native(x)
        expected = _manual_gigachat35_gated_norm(
            non_zero_centered_gated_norm,
            x,
            zero_centered=False,
        )

        assert actual.dtype == dtype
        _assert_close(actual, expected)

    def test_forward_native_with_residual_matches_manual_formula(
        self,
        non_zero_centered_gated_norm,
    ):
        x = torch.linspace(-2.0, 2.0, steps=2 * HIDDEN, dtype=torch.float32).reshape(
            2,
            HIDDEN,
        )
        residual = torch.linspace(0.5, -0.5, steps=2 * HIDDEN, dtype=torch.float32).reshape(
            2,
            HIDDEN,
        )
        post = torch.full((2, HIDDEN), -0.125, dtype=torch.float32)

        actual_out, actual_residual = non_zero_centered_gated_norm.forward_native(
            x,
            residual=residual,
            post_residual_addition=post,
        )
        expected_out, expected_residual = _manual_gigachat35_gated_norm(
            non_zero_centered_gated_norm,
            x,
            residual=residual,
            post_residual_addition=post,
            zero_centered=False,
        )

        _assert_close(actual_out, expected_out)
        _assert_close(actual_residual, expected_residual)


@pytest.mark.unit
class TestGigachat35GatedNormTorchCompile:
    @pytest.mark.parametrize(
        "norm_cls,zero_centered",
        [
            (Gigachat35ZeroCenteredGatedNormTorchCompile, True),
            (Gigachat35GatedNormTorchCompile, False),
        ],
    )
    def test_forward_matches_manual_formula(self, norm_cls, zero_centered):
        norm = norm_cls(hidden_size=HIDDEN, eps=1e-6, layernorm_gating_weight=1.25)
        _fill_gated_norm_weights(norm)
        if zero_centered:
            with torch.no_grad():
                norm.weight.sub_(1.0)

        x = torch.linspace(-1.0, 1.0, steps=3 * HIDDEN, dtype=torch.float32).reshape(
            3,
            HIDDEN,
        )

        actual = norm.forward_native(x)
        expected = _manual_gigachat35_gated_norm(
            norm,
            x,
            zero_centered=zero_centered,
        )

        _assert_close(actual, expected)

    def test_forward_with_residual_returns_merged_residual(self):
        norm = Gigachat35GatedNormTorchCompile(
            hidden_size=HIDDEN,
            eps=1e-6,
            layernorm_gating_weight=1.25,
        )
        _fill_gated_norm_weights(norm)
        x = torch.randn(2, HIDDEN, dtype=torch.float32)
        residual = torch.randn(2, HIDDEN, dtype=torch.float32)
        post = torch.randn(2, HIDDEN, dtype=torch.float32)

        actual_out, actual_residual = norm.forward_native(
            x,
            residual=residual,
            post_residual_addition=post,
        )
        expected_out, expected_residual = _manual_gigachat35_gated_norm(
            norm,
            x,
            residual=residual,
            post_residual_addition=post,
            zero_centered=False,
        )

        _assert_close(actual_out, expected_out)
        _assert_close(actual_residual, expected_residual)


@pytest.mark.unit
class TestGigachat35GatedNormImplementationParity:
    @pytest.mark.parametrize(
        "reference_cls,impl_name,impl_cls,zero_centered",
        [
            pytest.param(
                Gigachat35ZeroCenteredGatedNorm,
                "native",
                Gigachat35ZeroCenteredGatedNorm,
                True,
                id="zc-native",
            ),
            pytest.param(
                Gigachat35ZeroCenteredGatedNorm,
                "torch_compile",
                Gigachat35ZeroCenteredGatedNormTorchCompile,
                True,
                id="zc-torch_compile",
            ),
            pytest.param(
                Gigachat35ZeroCenteredGatedNorm,
                "triton",
                Gigachat35ZeroCenteredGatedNormTriton,
                True,
                id="zc-triton",
            ),
            pytest.param(
                Gigachat35GatedNorm,
                "native",
                Gigachat35GatedNorm,
                False,
                id="non-zc-native",
            ),
            pytest.param(
                Gigachat35GatedNorm,
                "torch_compile",
                Gigachat35GatedNormTorchCompile,
                False,
                id="non-zc-torch_compile",
            ),
        ],
    )
    @pytest.mark.parametrize("precision", ["bf16", "fp8"])
    def test_impl_matches_standard_native_with_residual(
        self,
        reference_cls,
        impl_name,
        impl_cls,
        zero_centered,
        precision,
    ):
        if impl_cls is None:
            pytest.skip("Triton gated norm implementation is not available")
        if impl_name == "triton":
            if not torch.cuda.is_available():
                pytest.skip("CUDA is required for the Triton gated norm kernel")
            pytest.importorskip("triton")
            device = "cuda"
        elif precision == "fp8":
            if not torch.cuda.is_available():
                pytest.skip("CUDA is required for fp8 gated norm parity")
            device = "cuda"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        reference = _make_gated_norm_parity_module(
            reference_cls,
            zero_centered=zero_centered,
            precision=precision,
            device=device,
        )
        # The Triton kernel requires activation and gate weights to have the same
        # dtype. Keep fp8 quantization noise in this case by round-tripping the
        # weights through fp8, then pass the dequantized bf16 values to the kernel.
        kernel_weight_dtype = (
            torch.bfloat16
            if impl_name == "triton" and precision == "fp8"
            else None
        )
        actual_norm = _make_gated_norm_parity_module(
            impl_cls,
            zero_centered=zero_centered,
            precision=precision,
            device=device,
            kernel_weight_dtype=kernel_weight_dtype,
        )
        x, residual, post = _make_gated_norm_parity_inputs(device)

        actual_forward = (
            actual_norm.forward_cuda
            if impl_name == "triton"
            else actual_norm.forward_native
        )
        actual_out, actual_residual = actual_forward(
            x,
            residual=residual,
            post_residual_addition=post,
        )
        expected_out, expected_residual = reference.forward_native(
            x,
            residual=residual,
            post_residual_addition=post,
        )
        if device == "cuda":
            torch.cuda.synchronize()

        assert actual_out.dtype == torch.bfloat16
        assert actual_residual.dtype == torch.bfloat16
        _assert_gated_norm_parity(actual_out, expected_out, precision=precision)
        _assert_gated_norm_parity(
            actual_residual,
            expected_residual,
            precision=precision,
        )


@pytest.mark.unit
class TestGigachat35ZeroCenteredGatedNormTriton:
    def test_forward_cuda_matches_manual_formula_with_residual(self):
        if Gigachat35ZeroCenteredGatedNormTriton is None:
            pytest.skip("Triton gated norm implementation is not available")
        if not torch.cuda.is_available():
            pytest.skip("CUDA is required for the Triton gated norm kernel")
        pytest.importorskip("triton")

        norm = Gigachat35ZeroCenteredGatedNormTriton(
            hidden_size=HIDDEN,
            eps=1e-6,
            layernorm_gating_weight=1.25,
        )
        norm.weight = nn.Parameter(
            norm.weight.detach().to(device="cuda", dtype=torch.bfloat16)
        )
        norm.gate_up_projection.to(device="cuda", dtype=torch.bfloat16)
        norm.gate_down_projection.to(device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            norm.weight.copy_(
                torch.linspace(-0.25, 0.25, steps=HIDDEN, device="cuda")
            )
            norm.gate_up_projection.weight.copy_(
                torch.linspace(
                    -0.3,
                    0.3,
                    steps=16 * HIDDEN,
                    device="cuda",
                ).reshape(16, HIDDEN)
            )
            norm.gate_down_projection.weight.copy_(
                torch.linspace(
                    -0.4,
                    0.4,
                    steps=HIDDEN * 16,
                    device="cuda",
                ).reshape(HIDDEN, 16)
            )

        x = torch.linspace(
            -1.0,
            1.0,
            steps=2 * HIDDEN,
            device="cuda",
            dtype=torch.float32,
        ).reshape(2, HIDDEN).to(torch.bfloat16)
        residual = torch.linspace(
            0.5,
            -0.5,
            steps=2 * HIDDEN,
            device="cuda",
            dtype=torch.float32,
        ).reshape(2, HIDDEN).to(torch.bfloat16)
        post = torch.full((2, HIDDEN), 0.125, device="cuda", dtype=torch.bfloat16)

        actual_out, actual_residual = norm.forward_cuda(
            x,
            residual=residual,
            post_residual_addition=post,
        )
        torch.cuda.synchronize()

        expected_out, expected_residual = _manual_gigachat35_gated_norm(
            norm,
            x,
            residual=residual,
            post_residual_addition=post,
            zero_centered=True,
        )

        assert actual_out.dtype == torch.bfloat16
        assert actual_residual.dtype == torch.bfloat16
        _assert_close(actual_out, expected_out)
        _assert_close(actual_residual, expected_residual)

    def test_forward_cuda_handles_large_non_contiguous_hidden(self):
        if Gigachat35ZeroCenteredGatedNormTriton is None:
            pytest.skip("Triton gated norm implementation is not available")
        if not torch.cuda.is_available():
            pytest.skip("CUDA is required for the Triton gated norm kernel")
        pytest.importorskip("triton")

        hidden_size = 4352
        norm = Gigachat35ZeroCenteredGatedNormTriton(
            hidden_size=hidden_size,
            eps=1e-6,
            layernorm_gating_weight=1.125,
        )
        norm.weight = nn.Parameter(
            norm.weight.detach().to(device="cuda", dtype=torch.bfloat16)
        )
        norm.gate_up_projection.to(device="cuda", dtype=torch.bfloat16)
        norm.gate_down_projection.to(device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            norm.weight.copy_(
                torch.linspace(-0.125, 0.125, steps=hidden_size, device="cuda")
            )
            norm.gate_up_projection.weight.copy_(
                torch.linspace(
                    -0.0625,
                    0.0625,
                    steps=16 * hidden_size,
                    device="cuda",
                ).reshape(16, hidden_size)
            )
            norm.gate_down_projection.weight.copy_(
                torch.linspace(
                    -0.09375,
                    0.09375,
                    steps=hidden_size * 16,
                    device="cuda",
                ).reshape(hidden_size, 16)
            )

        base = torch.linspace(
            -1.0,
            1.0,
            steps=2 * hidden_size * 2,
            device="cuda",
            dtype=torch.float32,
        ).reshape(2, hidden_size * 2).to(torch.bfloat16)
        x = base[:, ::2]
        assert not x.is_contiguous()

        actual = norm.forward_cuda(x)
        torch.cuda.synchronize()

        expected = _manual_gigachat35_gated_norm(
            norm,
            x,
            zero_centered=True,
        )
        assert actual.dtype == torch.bfloat16
        _assert_close(actual, expected)


@pytest.mark.unit
class TestGigachat35GatedNormTritonFused:
    """The plain-scale fused kernel (norm_type=GatedNorm) is the zero-centered kernel with ``w`` for ``1 + w``.

    ``ZERO_CENTERED`` is a constexpr of both Triton stages, so the two flavours must be
    bit-exact on ``w`` vs ``w - 1`` whenever ``w - 1`` is exact in bf16.
    """

    FUSED_HIDDEN = 256

    @classmethod
    def _pair(cls):
        plain_cls = getattr(_gigachat35, "Gigachat35GatedNormTritonFused", None)
        zc_cls = getattr(_gigachat35, "Gigachat35ZeroCenteredGatedNormTritonFused", None)
        if plain_cls is None or zc_cls is None:
            pytest.skip("triton_fused gated norm implementation is not available")
        if not torch.cuda.is_available():
            pytest.skip("CUDA is required for the triton_fused gated norm kernel")
        pytest.importorskip("triton")
        torch.manual_seed(0)
        h = cls.FUSED_HIDDEN
        plain = plain_cls(hidden_size=h, eps=1e-6, layernorm_gating_weight=1.375)
        zc = zc_cls(hidden_size=h, eps=1e-6, layernorm_gating_weight=1.375)
        assert torch.equal(plain.weight.detach(), torch.ones(h)), "GatedNorm weight must init to ones"
        with torch.no_grad():
            # w - 1 exact in bf16, so scale = w (plain) and scale = 1 + (w - 1) (zero-centered) are the same numbers.
            w = 1.0 + torch.randint(-8, 9, (h,)).float() / 16.0
            plain.weight.copy_(w)
            zc.weight.copy_(w - 1.0)
            up = torch.randn(16, h) * 0.1
            down = torch.randn(h, 16) * 0.1
            for norm in (plain, zc):
                norm.gate_up_projection.weight.copy_(up)
                norm.gate_down_projection.weight.copy_(down)
        return plain.to("cuda").to(torch.bfloat16), zc.to("cuda").to(torch.bfloat16)

    @pytest.mark.parametrize("with_residual", [False, True], ids=["plain", "residual"])
    def test_plain_scale_kernel_is_bit_exact_with_zero_centered_on_shifted_weight(self, with_residual):
        plain, zc = self._pair()
        rows, h = 37, self.FUSED_HIDDEN
        x = (torch.randn(rows, h, device="cuda") * 1.5).to(torch.bfloat16)
        residual = torch.randn(rows, h, device="cuda").to(torch.bfloat16) if with_residual else None

        got = plain.forward_cuda(x.clone(), residual=None if residual is None else residual.clone())
        ref = zc.forward_cuda(x.clone(), residual=None if residual is None else residual.clone())
        torch.cuda.synchronize()
        if with_residual:
            assert torch.equal(got[1], ref[1])
            got, ref = got[0], ref[0]
        assert torch.equal(got, ref), (
            f"plain-scale kernel differs from the zero-centered kernel on w-1: "
            f"max |diff| = {(got.float() - ref.float()).abs().max().item()}"
        )

    def test_plain_scale_kernel_matches_the_manual_gated_norm_formula(self):
        plain, _ = self._pair()
        rows, h = 37, self.FUSED_HIDDEN
        x = (torch.randn(rows, h, device="cuda") * 1.5).to(torch.bfloat16)
        got = plain.forward_cuda(x.clone())
        expected = _manual_gigachat35_gated_norm(plain, x, zero_centered=False)
        torch.cuda.synchronize()
        _assert_close(got, expected)
        # And the flag is live: the same weights read as zero-centered give a different answer.
        wrong = _manual_gigachat35_gated_norm(plain, x, zero_centered=True)
        assert not torch.allclose(got.float(), wrong.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.unit
class TestGigachat35ZeroCenteredGatedNormTritonFusedResidualRounding:
    """The fused residual add must normalize the value it stores as the new residual.

    gate_up rounds ``x + residual`` to the stream dtype before it feeds ``u`` and
    ``sum(x^2)``, so the fused call normalizes the same bf16 sum that gate_down reads
    back and that the trainer (residual added outside the norm) sees.
    """

    # The fused kernel needs H % 128 == 0, so it cannot share the HIDDEN=8 parity fixtures.
    FUSED_HIDDEN = 256

    @classmethod
    def _fused_norm(cls):
        fused_cls = getattr(
            _gigachat35, "Gigachat35ZeroCenteredGatedNormTritonFused", None
        )
        if fused_cls is None:
            pytest.skip("triton_fused gated norm implementation is not available")
        if not torch.cuda.is_available():
            pytest.skip("CUDA is required for the triton_fused gated norm kernel")
        pytest.importorskip("triton")
        torch.manual_seed(0)
        norm = fused_cls(
            hidden_size=cls.FUSED_HIDDEN, eps=1e-6, layernorm_gating_weight=1.375
        )
        with torch.no_grad():
            norm.weight.copy_(torch.randn(cls.FUSED_HIDDEN) * 0.1)
        return norm.to("cuda").to(torch.bfloat16)

    @pytest.mark.parametrize(
        "with_post", [False, True], ids=["residual", "residual+post"]
    )
    def test_fused_residual_call_is_bit_exact_with_the_unfused_call_on_the_stored_sum(
        self, with_post
    ):
        norm = self._fused_norm()
        rows, hidden = 37, self.FUSED_HIDDEN
        x = (torch.randn(rows, hidden, device="cuda") * 1.5).to(torch.bfloat16)
        residual = torch.randn(rows, hidden, device="cuda").to(torch.bfloat16)
        post = (
            (torch.randn(rows, hidden, device="cuda") * 0.25).to(torch.bfloat16)
            if with_post
            else None
        )

        out_fused, residual_out = norm.forward_cuda(
            x.clone(), residual=residual.clone(), post_residual_addition=post
        )
        stored_sum = residual + post if post is not None else residual
        stored_sum = (x + stored_sum).to(torch.bfloat16)
        out_unfused = norm.forward_cuda(stored_sum)
        torch.cuda.synchronize()

        assert torch.equal(residual_out, stored_sum)
        assert torch.equal(out_fused, out_unfused), (
            "fused residual path normalizes a different value than it stores: "
            f"max |diff| = {(out_fused.float() - out_unfused.float()).abs().max().item()}"
        )


@pytest.mark.unit
class TestSwigluLimitClipping:
    @pytest.mark.parametrize(
        "backend",
        ["triton", "triton_llmf", "deep_gemm"],
    )
    def test_fp8_backend_guard_allows_backends_that_propagate_clamp(
        self,
        monkeypatch,
        backend,
    ):
        monkeypatch.setattr(
            _gigachat35,
            "get_moe_runner_backend",
            lambda: _RunnerBackend(backend),
        )

        Gigachat35MoE._assert_fp8_clipping_supported(
            _FakeQuantConfig("fp8"),
            swiglu_limit=1.0,
        )

    @pytest.mark.parametrize(
        "backend",
        ["triton_kernel", "cutlass", "flashinfer_trtllm"],
    )
    def test_fp8_backend_guard_rejects_backend_that_drops_clamp(
        self,
        monkeypatch,
        backend,
    ):
        monkeypatch.setattr(
            _gigachat35,
            "get_moe_runner_backend",
            lambda: _RunnerBackend(backend),
        )

        with pytest.raises(AssertionError, match=backend):
            Gigachat35MoE._assert_fp8_clipping_supported(
                _FakeQuantConfig("fp8"),
                swiglu_limit=1.0,
            )

    def test_fp8_backend_guard_detects_compressed_tensors_stub_with_fp8_dtype(
        self,
        monkeypatch,
    ):
        monkeypatch.setattr(
            _gigachat35,
            "get_moe_runner_backend",
            lambda: _RunnerBackend("flashinfer_cutlass"),
        )

        # The production autometrics checkpoint uses quant_method="fp8" and
        # creates Fp8Config. This stub only covers the guard's duck-typed
        # compatibility branch; it is not a full CompressedTensorsConfig test.
        with pytest.raises(AssertionError, match="flashinfer_cutlass"):
            Gigachat35MoE._assert_fp8_clipping_supported(
                _FakeQuantConfig("compressed_tensors", weight_dtype="fp8_e4m3"),
                swiglu_limit=1.0,
            )

    def test_fp8_backend_guard_ignores_bf16_quant_config(self, monkeypatch):
        monkeypatch.setattr(
            _gigachat35,
            "get_moe_runner_backend",
            lambda: _RunnerBackend("flashinfer_trtllm"),
        )

        Gigachat35MoE._assert_fp8_clipping_supported(
            _FakeQuantConfig("bf16"),
            swiglu_limit=1.0,
        )

    def test_fp8_backend_guard_resolves_auto_to_triton(self, monkeypatch):
        monkeypatch.setattr(
            _gigachat35,
            "get_moe_runner_backend",
            lambda: _RunnerBackend("auto"),
        )
        monkeypatch.setattr(
            _gigachat35.deep_gemm_wrapper,
            "ENABLE_JIT_DEEPGEMM",
            False,
        )

        Gigachat35MoE._assert_fp8_clipping_supported(
            _FakeQuantConfig("fp8"),
            swiglu_limit=1.0,
        )

    def test_fp8_backend_guard_resolves_auto_to_deep_gemm(self, monkeypatch):
        monkeypatch.setattr(
            _gigachat35,
            "get_moe_runner_backend",
            lambda: _RunnerBackend("auto"),
        )
        monkeypatch.setattr(
            _gigachat35.deep_gemm_wrapper,
            "ENABLE_JIT_DEEPGEMM",
            True,
        )
        monkeypatch.setattr(
            _gigachat35,
            "get_moe_a2a_backend",
            lambda: SimpleNamespace(
                is_deepep=lambda: True,
                is_mooncake=lambda: False,
            ),
        )

        Gigachat35MoE._assert_fp8_clipping_supported(
            _FakeQuantConfig("fp8"),
            swiglu_limit=1.0,
        )

    def test_moe_init_threads_swiglu_limit_to_shared_and_routed_experts(self):
        shared_experts = _FakeDeepseekV2MLP()
        routed_experts = SimpleNamespace(
            moe_runner_config=SimpleNamespace(gemm1_clamp_limit=None),
        )

        moe = Gigachat35MoE(
            _cfg(
                swiglu_limit=1.75,
                use_shared_expert_sigmoid=False,
                _test_shared_experts=shared_experts,
                _test_experts=routed_experts,
            ),
            layer_id=0,
            quant_config=_FakeQuantConfig("bf16"),
        )

        assert moe.shared_experts is shared_experts
        assert moe.experts is routed_experts
        assert shared_experts.swiglu_limit == pytest.approx(1.75)
        assert routed_experts.moe_runner_config.gemm1_clamp_limit == pytest.approx(
            1.75
        )

    def test_moe_runner_fused_triton_path_threads_clamp_to_fused_experts(
        self,
        monkeypatch,
    ):
        pytest.importorskip("triton")
        import importlib

        fused_moe_module = importlib.import_module(
            "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe"
        )
        runner_module = importlib.import_module(
            "sglang.srt.layers.moe.moe_runner.runner"
        )
        triton_runner_module = importlib.import_module(
            "sglang.srt.layers.moe.moe_runner.triton"
        )
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardDispatchOutput,
        )
        from sglang.srt.layers.moe.utils import MoeRunnerBackend

        monkeypatch.setattr(
            runner_module,
            "get_moe_a2a_backend",
            lambda: SimpleNamespace(value="none"),
        )

        calls = []

        def fake_fused_experts(
            hidden_states,
            w1,
            w2,
            topk_output,
            moe_runner_config,
            **kwargs,
        ):
            calls.append(
                {
                    "hidden_states": hidden_states,
                    "topk_output": topk_output,
                    "moe_runner_config": moe_runner_config,
                    "kwargs": kwargs,
                }
            )
            return hidden_states + 2

        monkeypatch.setattr(fused_moe_module, "fused_experts", fake_fused_experts)

        hidden_states = torch.ones(2, HIDDEN, dtype=torch.bfloat16)
        topk_output = StandardTopKOutput(
            topk_weights=torch.ones(2, 1, dtype=torch.bfloat16),
            topk_ids=torch.zeros(2, 1, dtype=torch.long),
            router_logits=torch.zeros(2, 1, dtype=torch.bfloat16),
        )
        dispatch_output = StandardDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_output=topk_output,
        )
        runner_config = MoeRunnerConfig(
            num_experts=1,
            num_local_experts=1,
            hidden_size=HIDDEN,
            intermediate_size_per_partition=HIDDEN,
            top_k=1,
            gemm1_clamp_limit=0.875,
            inplace=False,
        )
        quant_info = triton_runner_module.TritonMoeQuantInfo(
            w13_weight=torch.empty(1, 2 * HIDDEN, HIDDEN),
            w2_weight=torch.empty(1, HIDDEN, HIDDEN),
            use_fp8_w8a8=True,
            w13_scale=torch.ones(1, 1, 1),
            w2_scale=torch.ones(1, 1, 1),
            block_shape=[128, 128],
        )

        output = runner_module.MoeRunner(
            MoeRunnerBackend.TRITON,
            runner_config,
        ).run(dispatch_output, quant_info)

        assert len(calls) == 1
        assert calls[0]["hidden_states"] is hidden_states
        assert calls[0]["topk_output"] is topk_output
        assert calls[0]["moe_runner_config"] is runner_config
        assert runner_config.gemm1_clamp_limit == pytest.approx(0.875)
        assert calls[0]["kwargs"]["use_fp8_w8a8"] is True
        assert calls[0]["kwargs"]["block_shape"] == [128, 128]
        _assert_close(output.hidden_states, hidden_states + 2)

    @pytest.mark.parametrize("backend", ["triton", "triton_llmf"])
    def test_triton_runner_block_fp8_with_clipping_matches_bf16_reference(
        self,
        monkeypatch,
        backend,
    ):
        if not torch.cuda.is_available():
            pytest.skip("CUDA is required for the Triton block-fp8 MoE kernel")
        pytest.importorskip("triton")
        capability = torch.cuda.get_device_capability()
        if capability[0] < 9 and capability != (8, 9):
            pytest.skip("fp8 fused MoE requires SM89 or SM90+")

        import importlib

        triton_runner_module = importlib.import_module(
            "sglang.srt.layers.moe.moe_runner.triton"
        )
        fused_moe_config_module = importlib.import_module(
            "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config"
        )
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig

        fused_moe_module = importlib.import_module(
            "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe"
        )
        monkeypatch.setattr(
            fused_moe_module,
            "get_moe_runner_backend",
            lambda: _RunnerBackend(backend),
        )
        monkeypatch.setattr(
            fused_moe_config_module,
            "get_global_server_args",
            lambda: SimpleNamespace(enable_deterministic_inference=False),
        )
        monkeypatch.setattr(
            fused_moe_module,
            "get_global_server_args",
            lambda: SimpleNamespace(
                enable_deterministic_inference=False,
                enable_fused_moe_sum_all_reduce=False,
            ),
        )

        (
            hidden_states,
            w1,
            w2,
            topk_output,
            w1_scale,
            w2_scale,
            block_shape,
        ) = _make_block_fp8_moe_case("cuda")
        limit = 0.75

        runner_config = MoeRunnerConfig(
            num_experts=w1.shape[0],
            num_local_experts=w1.shape[0],
            hidden_size=hidden_states.shape[1],
            intermediate_size_per_partition=w2.shape[2],
            top_k=topk_output.topk_ids.shape[1],
            gemm1_clamp_limit=limit,
            inplace=False,
        )
        quant_info = triton_runner_module.TritonMoeQuantInfo(
            w13_weight=w1,
            w2_weight=w2,
            use_fp8_w8a8=True,
            w13_scale=w1_scale,
            w2_scale=w2_scale,
            block_shape=block_shape,
        )
        dispatch_output = SimpleNamespace(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )
        running_state = {}
        runner_input = triton_runner_module.pre_permute_standard_to_triton(
            dispatch_output,
            quant_info,
            runner_config,
            running_state,
        )
        actual = triton_runner_module.TritonRunnerCore(runner_config).run(
            runner_input,
            quant_info,
            running_state,
        ).hidden_states
        torch.cuda.synchronize()

        expected = _manual_block_fp8_clipped_moe(
            hidden_states,
            w1,
            w2,
            topk_output.topk_weights,
            topk_output.topk_ids,
            w1_scale,
            w2_scale,
            block_shape,
            swiglu_limit=limit,
        )
        relative_error = (
            (actual.float() - expected.float()).abs().mean()
            / expected.float().abs().mean().clamp_min(1e-6)
        )
        assert relative_error < BLOCK_FP8_MOE_CLIPPING_RTOL
        torch.testing.assert_close(
            actual,
            expected,
            atol=BLOCK_FP8_MOE_CLIPPING_ATOL,
            rtol=BLOCK_FP8_MOE_CLIPPING_RTOL,
        )

    def test_deep_gemm_fp8_activation_quant_matches_dequantized_bf16_reference(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA is required for the fp8 activation quantizer")

        from sglang.srt.layers.activation import _swiglu_silu_clamp_mul
        from sglang.srt.layers.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8,
        )

        total_tokens = 4
        hidden = 128
        limit = 0.875
        gate_up = torch.linspace(
            -2.5,
            2.5,
            steps=total_tokens * hidden * 2,
            device="cuda",
            dtype=torch.float32,
        ).reshape(total_tokens, hidden * 2).to(torch.bfloat16)
        activated = torch.empty(
            total_tokens,
            hidden,
            device="cuda",
            dtype=torch.bfloat16,
        )

        _swiglu_silu_clamp_mul(gate_up, limit, out=activated)
        quantized, scales = sglang_per_token_group_quant_fp8(activated, hidden)
        torch.cuda.synchronize()

        reference = _manual_swiglu_clamp(gate_up, limit).to(torch.bfloat16)
        _assert_close(activated, reference)

        dequantized = quantized.float() * scales
        torch.testing.assert_close(
            dequantized,
            reference.float(),
            atol=FP8_ROUNDTRIP_ATOL,
            rtol=FP8_ROUNDTRIP_RTOL,
        )

    def test_deep_gemm_contiguous_runner_threads_clipped_activation(
        self,
        monkeypatch,
    ):
        import importlib

        activation_module = importlib.import_module("sglang.srt.layers.activation")
        deep_gemm_module = importlib.import_module(
            "sglang.srt.layers.moe.moe_runner.deep_gemm"
        )
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig

        all_tokens = 3
        hidden_size = 8
        gate_up_size = 256
        limit = 0.875
        m_indices = torch.tensor([0, 1, 1], dtype=torch.int32)
        hidden_states = torch.arange(
            all_tokens * hidden_size,
            dtype=torch.float32,
        ).reshape(all_tokens, hidden_size).to(torch.bfloat16)
        hidden_states_scale = torch.ones(all_tokens, 1, dtype=torch.float32)
        gate_up = torch.linspace(
            -2.0,
            2.0,
            steps=all_tokens * gate_up_size,
            dtype=torch.float32,
        ).reshape(all_tokens, gate_up_size).to(torch.bfloat16)
        expected_down_input = _manual_swiglu_clamp(gate_up, limit).to(torch.bfloat16)

        gemm_calls = []
        activation_calls = []
        quant_calls = []
        aligned_scales = []

        def fake_align_scale(scale):
            aligned_scales.append(scale)
            return scale

        def fake_activation(x, gemm1_limit, out):
            activation_calls.append((x.detach().clone(), gemm1_limit, tuple(out.shape)))
            out.copy_(_manual_swiglu_clamp(x, gemm1_limit).to(out.dtype))

        def fake_quant_fp8(x, block_size, **kwargs):
            quant_calls.append(
                {
                    "x": x.detach().clone(),
                    "block_size": block_size,
                    "kwargs": kwargs,
                }
            )
            return x.detach().clone(), torch.ones(x.shape[0], 1, dtype=torch.float32)

        def fake_grouped_gemm(lhs, weights, out, call_m_indices, **kwargs):
            gemm_calls.append(
                {
                    "lhs": lhs,
                    "weights": weights,
                    "m_indices": call_m_indices,
                    "out_shape": tuple(out.shape),
                }
            )
            if len(gemm_calls) == 1:
                out.copy_(gate_up)
            else:
                down_input_fp8, _ = lhs
                _assert_close(down_input_fp8, expected_down_input)
                out.copy_(down_input_fp8[:, :hidden_size].to(out.dtype))

        fake_ep_kernels_module = ModuleType("sglang.srt.layers.moe.ep_moe.kernels")
        fake_ep_kernels_module.tma_align_input_scale = fake_align_scale
        fake_fp8_kernel_module = ModuleType("sglang.srt.layers.quantization.fp8_kernel")
        fake_fp8_kernel_module.sglang_per_token_group_quant_fp8 = fake_quant_fp8

        monkeypatch.setitem(
            sys.modules,
            "sglang.srt.layers.moe.ep_moe.kernels",
            fake_ep_kernels_module,
        )
        monkeypatch.setitem(
            sys.modules,
            "sglang.srt.layers.quantization.fp8_kernel",
            fake_fp8_kernel_module,
        )
        monkeypatch.setattr(deep_gemm_module, "dispose_tensor", lambda tensor: None)
        monkeypatch.setattr(
            deep_gemm_module.deep_gemm_wrapper,
            "DEEPGEMM_SCALE_UE8M0",
            False,
        )
        monkeypatch.setattr(
            deep_gemm_module.deep_gemm_wrapper,
            "grouped_gemm_nt_f8f8bf16_contig",
            fake_grouped_gemm,
        )
        monkeypatch.setattr(activation_module, "_swiglu_silu_clamp_mul", fake_activation)

        runner = deep_gemm_module.DeepGemmRunnerCore(
            MoeRunnerConfig(gemm1_clamp_limit=limit)
        )
        runner_input = deep_gemm_module.DeepGemmRunnerInput(
            hidden_states=hidden_states,
            hidden_states_scale=hidden_states_scale,
            use_masked_gemm=False,
            m_indices=m_indices,
        )
        quant_info = deep_gemm_module.DeepGemmMoeQuantInfo(
            w13_weight=torch.empty(2, gate_up_size, hidden_size),
            w2_weight=torch.empty(2, hidden_size, gate_up_size // 2),
            use_fp8=True,
            w13_scale=torch.ones(2, 2, 1),
            w2_scale=torch.ones(2, 1, 1),
            block_shape=[128, 128],
        )
        output = runner.run(
            runner_input,
            quant_info,
            {
                "all_tokens": all_tokens,
                "hidden_states_device": torch.device("cpu"),
                "hidden_states_dtype": torch.bfloat16,
                "hidden_states_shape": tuple(hidden_states.shape),
            },
        )

        assert len(gemm_calls) == 2
        assert len(aligned_scales) == 2
        assert len(activation_calls) == 1
        activation_input, activation_limit, activation_out_shape = activation_calls[0]
        _assert_close(activation_input, gate_up)
        assert activation_limit == pytest.approx(limit)
        assert activation_out_shape == (all_tokens, gate_up_size // 2)

        assert len(quant_calls) == 1
        _assert_close(quant_calls[0]["x"], expected_down_input)
        assert quant_calls[0]["block_size"] == 128
        assert quant_calls[0]["kwargs"] == {
            "column_major_scales": False,
            "scale_tma_aligned": False,
            "scale_ue8m0": False,
        }
        assert gemm_calls[0]["m_indices"] is m_indices
        assert gemm_calls[1]["m_indices"] is m_indices
        _assert_close(output.hidden_states, expected_down_input[:, :hidden_size])

    def test_deep_gemm_masked_path_rejects_clipping(self):
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
        from sglang.srt.layers.moe.moe_runner.deep_gemm import DeepGemmRunnerCore

        runner = DeepGemmRunnerCore(MoeRunnerConfig(gemm1_clamp_limit=1.0))

        with pytest.raises(AssertionError, match="masked gemm path"):
            runner._run_masked_gemm(None, None, {})

    def test_fused_swiglu_quant_llmf_matches_dequantized_bf16_reference(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA is required for the Triton fused fp8 clipping kernel")
        pytest.importorskip("triton")

        from sglang.srt.layers.moe.fused_moe_triton.fused_swiglu_quant_llmf import (
            fused_swiglu_quant_llmf,
        )

        total_tokens = 5
        hidden = 256
        group_size = 128
        limit = 1.25
        gate_up = torch.linspace(
            -3.0,
            3.0,
            steps=total_tokens * hidden * 2,
            device="cuda",
            dtype=torch.float32,
        ).reshape(total_tokens, hidden * 2)
        gate_up = gate_up.to(torch.bfloat16)
        probs = torch.linspace(0.25, 1.0, steps=total_tokens, device="cuda").to(
            torch.bfloat16
        )

        fp8_data, fp8_scales = fused_swiglu_quant_llmf(
            gate_up,
            swiglu_limit=limit,
            probs=probs,
            group_size=group_size,
            row_block=2,
        )
        torch.cuda.synchronize()

        reference = _manual_swiglu_clamp(gate_up, limit, probs=probs)
        reference_groups = reference.reshape(total_tokens, hidden // group_size, group_size)
        amax = reference_groups.abs().amax(dim=2)
        expected_scales = torch.pow(
            2.0,
            torch.ceil(torch.log2(torch.clamp(amax / 448.0, min=1e-30))),
        )
        expected_fp8 = (reference_groups / expected_scales.unsqueeze(-1)).to(
            torch.float8_e4m3fn
        ).reshape(total_tokens, hidden)

        torch.testing.assert_close(fp8_scales, expected_scales)
        assert torch.equal(fp8_data, expected_fp8)

        expanded_scales = fp8_scales.repeat_interleave(group_size, dim=1)
        dequantized = fp8_data.float() * expanded_scales
        torch.testing.assert_close(
            dequantized,
            reference,
            atol=FP8_ROUNDTRIP_ATOL,
            rtol=FP8_ROUNDTRIP_RTOL,
        )


def _make_loader(
    *,
    config=None,
    model=None,
    quant_config=None,
    draft_model_idx: int = 0,
):
    loader = Gigachat35HybridGDNWeightLoaderMixin()
    loader.config = config or _cfg(num_hidden_layers=4, n_routed_experts=1)
    loader.model = model or SimpleNamespace(
        start_layer=0,
        end_layer=getattr(loader.config, "num_hidden_layers", 0),
    )
    loader.quant_config = quant_config
    loader.draft_model_idx = draft_model_idx
    return loader


def _patch_base_post_load_weights(monkeypatch):
    calls = []

    def fake_post_load_weights(self, is_nextn=False, weight_names=None):
        normalized_weight_names = None if weight_names is None else list(weight_names)
        calls.append((is_nextn, normalized_weight_names))
        return {
            "is_nextn": is_nextn,
            "weight_names": normalized_weight_names,
        }

    monkeypatch.setattr(
        DeepseekV2WeightLoaderMixin,
        "post_load_weights",
        fake_post_load_weights,
    )
    return calls


def _patch_base_quant_weights(monkeypatch):
    calls = []

    def fake_maybe_quant_weights_to_fp8_ue8m0(
        self,
        weights,
        attn_quant_modules,
        nextn_conf,
    ):
        normalized_weights = list(weights)
        normalized_attn_modules = list(attn_quant_modules)
        calls.append((normalized_weights, normalized_attn_modules, nextn_conf))
        return "delegated"

    monkeypatch.setattr(
        DeepseekV2WeightLoaderMixin,
        "_maybe_quant_weights_to_fp8_ue8m0",
        fake_maybe_quant_weights_to_fp8_ue8m0,
    )
    return calls


class _FakeTopK:
    def __init__(self, output, output_format):
        self.output = output
        self.topk_config = SimpleNamespace(output_format=output_format)
        self.calls = []
        self.empty_calls = []

    def empty_topk_output(self, device):
        self.empty_calls.append(device)
        return ("empty", device)

    def __call__(self, hidden_states, router_logits, **kwargs):
        self.calls.append((hidden_states.clone(), router_logits.clone(), kwargs))
        return self.output


class _FakeSharedExpertSigmoid:
    def __init__(self, gate: torch.Tensor):
        self.gate = gate
        self.calls = []

    def __call__(self, hidden_states):
        self.calls.append(hidden_states.clone())
        return self.gate, None


@pytest.mark.unit
class TestGigachat35GatedDeltaNet:
    def test_op_prepare_moves_hidden_states_to_attn_slot(self):
        model = _make_uninitialized_gated_delta_net()
        model.forward = lambda hidden_states, forward_batch: hidden_states + 2.0

        state = _AttrDict(
            hidden_states_after_comm_pre_attn=torch.tensor([[1.0, 2.0]]),
            forward_batch="forward-batch",
        )

        model.op_prepare(state)

        assert "hidden_states_after_comm_pre_attn" not in state
        assert torch.equal(state.hidden_states_after_attn, torch.tensor([[3.0, 4.0]]))

    def test_op_core_is_noop(self):
        model = _make_uninitialized_gated_delta_net()
        state = _AttrDict(example=1)
        assert model.op_core(state) is None
        assert state.example == 1


@pytest.mark.unit
class TestGigachat35HybridGDNWeightLoaderMixin:
    def test_get_hybrid_gdn_config_view_delegates(self, monkeypatch):
        cfg = _cfg(name="cfg")
        sentinel = object()
        loader = _make_loader(config=cfg)

        monkeypatch.setattr(
            _gigachat35,
            "get_hybrid_gdn_config_view",
            lambda config: sentinel if config is cfg else None,
        )

        assert loader._get_hybrid_gdn_config_view() is sentinel

    def test_initialize_nextn_conf_returns_disabled_config(self):
        loader = _make_loader()
        conf = loader._initialize_nextn_conf(False)
        assert isinstance(conf, NextNDisabledConfig)

    def test_initialize_nextn_conf_requires_num_nextn_predict_layers(self):
        loader = _make_loader(config=_cfg(num_hidden_layers=4))
        with pytest.raises(ValueError, match="num_nextn_predict_layers"):
            loader._initialize_nextn_conf(True)

    def test_initialize_nextn_conf_single_layer_uses_zero_layer_id(self):
        loader = _make_loader(
            config=_cfg(num_hidden_layers=1, num_nextn_predict_layers=1),
        )
        conf = loader._initialize_nextn_conf(True)
        assert isinstance(conf, NextNEnabledConfig)
        assert conf.nextn_layer_id == 0
        assert conf.nextn_layer_prefix == "model.layers.0"
        assert conf.nextn_spec_weight_names == [
            "shared_head.norm",
            "eh_proj",
            "enorm",
            "hnorm",
        ]

    def test_initialize_nextn_conf_uses_draft_model_index(self):
        # 3 heads, not 1: asking for head 2 of a 1-head checkpoint is now rejected.
        loader = _make_loader(
            config=_cfg(num_hidden_layers=5, num_nextn_predict_layers=3),
            draft_model_idx=2,
        )
        conf = loader._initialize_nextn_conf(True)
        assert conf.nextn_layer_id == 7
        assert conf.nextn_layer_prefix == "model.layers.7"

    def test_initialize_nextn_conf_rejects_out_of_range_draft_model_index(self):
        loader = _make_loader(
            config=_cfg(num_hidden_layers=5, num_nextn_predict_layers=2),
            draft_model_idx=2,
        )
        with pytest.raises(ValueError, match="draft_model_idx"):
            loader._initialize_nextn_conf(True)

    def test_initialize_nextn_conf_rejects_non_positive_head_count(self):
        loader = _make_loader(
            config=_cfg(num_hidden_layers=5, num_nextn_predict_layers=0),
        )
        with pytest.raises(ValueError, match="must be positive"):
            loader._initialize_nextn_conf(True)

    @pytest.mark.parametrize(
        "draft_model_idx,expected_layer_id", [(0, 40), (1, 41), (2, 42)]
    )
    def test_deepseek_loader_selects_one_appended_layer(
        self, draft_model_idx, expected_layer_id
    ):
        loader = DeepseekV2WeightLoaderMixin()
        loader.config = _cfg(num_hidden_layers=40, num_nextn_predict_layers=3)
        loader.draft_model_idx = draft_model_idx
        conf = loader._initialize_nextn_conf(True)
        assert conf.num_nextn_layers == 1
        assert conf.nextn_layer_id == expected_layer_id
        assert conf.nextn_layer_prefix == f"model.layers.{expected_layer_id}"

    def test_deepseek_loader_rejects_idx_beyond_checkpoint_heads(self):
        loader = DeepseekV2WeightLoaderMixin()
        loader.config = _cfg(num_hidden_layers=40, num_nextn_predict_layers=2)
        loader.draft_model_idx = 2
        with pytest.raises(ValueError, match="draft_model_idx"):
            loader._initialize_nextn_conf(True)

    def test_deepseek_loader_without_draft_model_idx_uses_first_head(self):
        loader = DeepseekV2WeightLoaderMixin()
        loader.config = _cfg(num_hidden_layers=40, num_nextn_predict_layers=3)
        conf = loader._initialize_nextn_conf(True)
        assert conf.nextn_layer_id == 40

    def test_post_load_weights_skips_super_for_linear_nextn_attention(self, monkeypatch):
        loader = _make_loader(config=_cfg(num_hidden_layers=4))
        calls = _patch_base_post_load_weights(monkeypatch)

        monkeypatch.setattr(
            _gigachat35,
            "gigachat35_mtp_uses_linear_attention",
            lambda config: True,
        )

        result = loader.post_load_weights(is_nextn=True, weight_names=["weight"])

        assert result is None
        assert calls == []

    def test_post_load_weights_delegates_for_non_linear_nextn_attention(self, monkeypatch):
        loader = _make_loader(config=_cfg(num_hidden_layers=4))
        calls = _patch_base_post_load_weights(monkeypatch)

        monkeypatch.setattr(
            _gigachat35,
            "gigachat35_mtp_uses_linear_attention",
            lambda config: False,
        )

        result = loader.post_load_weights(is_nextn=True, weight_names=["weight"])

        assert result == {"is_nextn": True, "weight_names": ["weight"]}
        assert calls == [(True, ["weight"])]

    def test_post_load_weights_delegates_when_all_layers_are_full_attention(
        self,
        monkeypatch,
    ):
        loader = _make_loader(
            config=_cfg(num_hidden_layers=3),
            model=SimpleNamespace(start_layer=0, end_layer=3),
        )
        calls = _patch_base_post_load_weights(monkeypatch)

        monkeypatch.setattr(_gigachat35, "get_hybrid_gdn_config_view", lambda config: None)
        monkeypatch.setattr(
            _gigachat35,
            "is_gigachat35_linear_attention_layer",
            lambda config, layer_id: False,
        )

        result = loader.post_load_weights(is_nextn=False, weight_names=["w0", "w1"])

        assert result == {"is_nextn": False, "weight_names": ["w0", "w1"]}
        assert calls == [(False, ["w0", "w1"])]

    def test_post_load_weights_filters_full_attention_layers_without_weight_names(
        self,
        monkeypatch,
    ):
        loader = _make_loader(
            config=_cfg(num_hidden_layers=5),
            model=SimpleNamespace(start_layer=1, end_layer=4),
        )
        calls = _patch_base_post_load_weights(monkeypatch)

        monkeypatch.setattr(
            _gigachat35,
            "get_hybrid_gdn_config_view",
            lambda config: SimpleNamespace(full_attention_layer_ids=[0, 2, 4]),
        )

        result = loader.post_load_weights(is_nextn=False, weight_names=None)

        assert result == {
            "is_nextn": False,
            "weight_names": ["model.layers.2.self_attn.kv_b_proj.weight"],
        }
        assert calls == [(False, ["model.layers.2.self_attn.kv_b_proj.weight"])]

    def test_post_load_weights_filters_provided_weight_names(self, monkeypatch):
        loader = _make_loader(
            config=_cfg(num_hidden_layers=5),
            model=SimpleNamespace(start_layer=0, end_layer=5),
        )
        calls = _patch_base_post_load_weights(monkeypatch)

        monkeypatch.setattr(
            _gigachat35,
            "get_hybrid_gdn_config_view",
            lambda config: SimpleNamespace(full_attention_layer_ids=[1, 3]),
        )

        result = loader.post_load_weights(
            is_nextn=False,
            weight_names=[
                "model.layers.0.self_attn.kv_b_proj.weight",
                "model.layers.1.self_attn.kv_b_proj.weight",
                "other.weight",
                "model.layers.3.self_attn.o_proj.weight",
            ],
        )

        assert result == {
            "is_nextn": False,
            "weight_names": [
                "model.layers.1.self_attn.kv_b_proj.weight",
                "model.layers.3.self_attn.o_proj.weight",
            ],
        }
        assert calls == [
            (
                False,
                [
                    "model.layers.1.self_attn.kv_b_proj.weight",
                    "model.layers.3.self_attn.o_proj.weight",
                ],
            )
        ]

    def test_maybe_quant_weights_delegates_for_non_hybrid_disabled_config(
        self,
        monkeypatch,
    ):
        loader = _make_loader(config=_cfg(num_hidden_layers=4))
        calls = _patch_base_quant_weights(monkeypatch)
        monkeypatch.setattr(_gigachat35, "get_hybrid_gdn_config_view", lambda config: None)

        weights = [("example.weight", torch.tensor([1.0]))]
        result = loader._maybe_quant_weights_to_fp8_ue8m0(
            weights,
            ["q_proj"],
            NextNDisabledConfig(),
        )

        assert result == "delegated"
        assert len(calls) == 1
        assert calls[0][0] == weights
        assert calls[0][1] == ["q_proj"]
        assert isinstance(calls[0][2], NextNDisabledConfig)

    def test_maybe_quant_weights_quantizes_nextn_attention_weights(self, monkeypatch):
        loader = _make_loader(config=_cfg(num_hidden_layers=4, n_routed_experts=1))
        nextn_conf = NextNEnabledConfig(
            num_nextn_layers=1,
            nextn_layer_id=4,
            nextn_layer_prefix="model.layers.4",
            nextn_spec_weight_names=[],
        )

        monkeypatch.setattr(
            _gigachat35.envs.SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN,
            "get",
            lambda: True,
        )
        monkeypatch.setattr(
            _gigachat35,
            "gigachat35_mtp_uses_linear_attention",
            lambda config: False,
        )
        monkeypatch.setattr(
            _gigachat35,
            "quant_weight_ue8m0",
            lambda weight, weight_block_size: (
                weight + 10,
                torch.tensor(weight_block_size),
            ),
        )

        weights = [
            ("model.layers.4.self_attn.q_proj.weight", torch.tensor([1.0])),
            ("other.weight", torch.tensor([2.0])),
        ]
        result = dict(
            loader._maybe_quant_weights_to_fp8_ue8m0(weights, ["q_proj"], nextn_conf)
        )

        assert torch.equal(result["model.layers.4.self_attn.q_proj.weight"], torch.tensor([11.0]))
        assert torch.equal(
            result["model.layers.4.self_attn.q_proj.weight_scale_inv"],
            torch.tensor([128, 128]),
        )
        assert torch.equal(result["other.weight"], torch.tensor([2.0]))

    def test_maybe_quant_weights_quantizes_nextn_moe_weights_and_marks_them(
        self,
        monkeypatch,
    ):
        loader = _make_loader(config=_cfg(num_hidden_layers=4, n_routed_experts=1))
        marks = []
        nextn_conf = NextNEnabledConfig(
            num_nextn_layers=1,
            nextn_layer_id=4,
            nextn_layer_prefix="model.layers.4",
            nextn_spec_weight_names=[],
        )

        monkeypatch.setattr(
            _gigachat35.envs.SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN,
            "get",
            lambda: False,
        )
        monkeypatch.setattr(
            _gigachat35,
            "enable_nextn_moe_bf16_cast_to_fp8",
            lambda config, quant_config: True,
        )
        monkeypatch.setattr(
            _gigachat35,
            "quant_weight_ue8m0",
            lambda weight, weight_block_size: (
                weight * 2,
                torch.tensor(weight_block_size),
            ),
        )
        monkeypatch.setattr(
            loader,
            "_mark_nextn_moe_weights_as_ue8m0",
            lambda: marks.append(True),
        )

        weights = [
            ("model.layers.4.mlp.shared_experts.gate_proj.weight", torch.tensor([1.0])),
            ("model.layers.4.mlp.shared_experts.up_proj.weight", torch.tensor([2.0])),
            ("model.layers.4.mlp.shared_experts.down_proj.weight", torch.tensor([3.0])),
            ("model.layers.4.mlp.experts.0.gate_proj.weight", torch.tensor([4.0])),
            ("model.layers.4.mlp.experts.0.up_proj.weight", torch.tensor([5.0])),
            ("model.layers.4.mlp.experts.0.down_proj.weight", torch.tensor([6.0])),
        ]
        result = dict(
            loader._maybe_quant_weights_to_fp8_ue8m0(weights, ["q_proj"], nextn_conf)
        )

        assert marks == [True]
        for name, original in weights:
            assert torch.equal(result[name], original * 2)
            assert f"{name}_scale_inv" in result

    def test_maybe_quant_weights_quantizes_hybrid_full_attention_layers(
        self,
        monkeypatch,
    ):
        loader = _make_loader(config=_cfg(num_hidden_layers=4, n_routed_experts=1))

        monkeypatch.setattr(
            _gigachat35,
            "get_hybrid_gdn_config_view",
            lambda config: SimpleNamespace(full_attention_layer_ids=[1, 3]),
        )
        monkeypatch.setattr(
            _gigachat35.envs.SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN,
            "get",
            lambda: True,
        )
        monkeypatch.setattr(
            _gigachat35,
            "quant_weight_ue8m0",
            lambda weight, weight_block_size: (
                weight + 5,
                torch.tensor(weight_block_size),
            ),
        )

        weights = [
            ("model.layers.1.self_attn.q_proj.weight", torch.tensor([1.0])),
            ("model.layers.2.self_attn.q_proj.weight", torch.tensor([2.0])),
            ("model.layers.3.self_attn.q_proj.weight", torch.tensor([3.0])),
        ]
        result = dict(
            loader._maybe_quant_weights_to_fp8_ue8m0(
                weights,
                ["q_proj"],
                NextNDisabledConfig(),
            )
        )

        assert torch.equal(result["model.layers.1.self_attn.q_proj.weight"], torch.tensor([6.0]))
        assert torch.equal(result["model.layers.2.self_attn.q_proj.weight"], torch.tensor([2.0]))
        assert torch.equal(result["model.layers.3.self_attn.q_proj.weight"], torch.tensor([8.0]))


@pytest.mark.unit
class TestGigachat35TopK:
    def test_topk_config_property_delegates(self):
        topk = _FakeTopK(
            output=SimpleNamespace(),
            output_format=TopKOutputFormat.STANDARD,
        )
        wrapper = Gigachat35TopK(
            topk=topk,
            shared_expert_sigmoid=_FakeSharedExpertSigmoid(torch.zeros(1, 1)),
            num_fused_shared_experts=1,
        )
        assert wrapper.topk_config is topk.topk_config

    def test_supports_fused_shared_expert_sigmoid_uses_explicit_output_format(self):
        topk = _FakeTopK(output=SimpleNamespace(), output_format=TopKOutputFormat.STANDARD)
        wrapper = Gigachat35TopK(
            topk=topk,
            shared_expert_sigmoid=_FakeSharedExpertSigmoid(torch.zeros(1, 1)),
            num_fused_shared_experts=1,
        )
        assert wrapper.supports_fused_shared_expert_sigmoid() is True

    def test_supports_fused_shared_expert_sigmoid_uses_backend_fallback(self, monkeypatch):
        topk = _FakeTopK(output=SimpleNamespace(), output_format=None)
        wrapper = Gigachat35TopK(
            topk=topk,
            shared_expert_sigmoid=_FakeSharedExpertSigmoid(torch.zeros(1, 1)),
            num_fused_shared_experts=1,
        )

        monkeypatch.setattr(
            _gigachat35,
            "get_moe_runner_backend",
            lambda: _StubBackend(triton=True),
        )
        assert wrapper.supports_fused_shared_expert_sigmoid() is False

        monkeypatch.setattr(
            _gigachat35,
            "get_moe_runner_backend",
            lambda: _StubBackend(),
        )
        assert wrapper.supports_fused_shared_expert_sigmoid() is True

    def test_supports_fused_shared_expert_sigmoid_flashinfer_backend(self, monkeypatch):
        topk = _FakeTopK(output=SimpleNamespace(), output_format=None)
        wrapper = Gigachat35TopK(
            topk=topk,
            shared_expert_sigmoid=_FakeSharedExpertSigmoid(torch.zeros(1, 1)),
            num_fused_shared_experts=1,
        )

        monkeypatch.setattr(
            _gigachat35,
            "get_moe_runner_backend",
            lambda: _StubBackend(flashinfer_trtllm=True),
        )
        assert wrapper.supports_fused_shared_expert_sigmoid() is False

        monkeypatch.setattr(
            _gigachat35,
            "get_moe_runner_backend",
            lambda: _StubBackend(flashinfer_mxfp4=True),
        )
        assert wrapper.supports_fused_shared_expert_sigmoid() is False

    def test_empty_topk_output_delegates(self):
        topk = _FakeTopK(output=SimpleNamespace(), output_format=TopKOutputFormat.STANDARD)
        wrapper = Gigachat35TopK(
            topk=topk,
            shared_expert_sigmoid=_FakeSharedExpertSigmoid(torch.zeros(1, 1)),
            num_fused_shared_experts=1,
        )
        device = torch.device("cpu")
        assert wrapper.empty_topk_output(device) == ("empty", device)
        assert topk.empty_calls == [device]

    def test_apply_shared_expert_sigmoid_returns_input_for_empty_hidden_states(self):
        topk_output = StandardTopKOutput(
            topk_weights=torch.ones(0, 2),
            topk_ids=torch.zeros(0, 2, dtype=torch.long),
            router_logits=torch.zeros(0, 2),
        )
        wrapper = Gigachat35TopK(
            topk=_FakeTopK(topk_output, TopKOutputFormat.STANDARD),
            shared_expert_sigmoid=_FakeSharedExpertSigmoid(torch.zeros(0, 1)),
            num_fused_shared_experts=1,
        )
        assert wrapper._apply_shared_expert_sigmoid(torch.empty(0, HIDDEN), topk_output) is topk_output

    def test_apply_shared_expert_sigmoid_returns_input_for_non_standard_output(self):
        topk_output = SimpleNamespace(topk_weights=torch.ones(2, 2))
        wrapper = Gigachat35TopK(
            topk=_FakeTopK(topk_output, TopKOutputFormat.STANDARD),
            shared_expert_sigmoid=_FakeSharedExpertSigmoid(torch.zeros(2, 1)),
            num_fused_shared_experts=1,
        )
        hidden_states = torch.randn(2, HIDDEN)
        assert wrapper._apply_shared_expert_sigmoid(hidden_states, topk_output) is topk_output

    def test_apply_shared_expert_sigmoid_scales_last_weights(self):
        topk_output = StandardTopKOutput(
            topk_weights=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            topk_ids=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            router_logits=torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
        )
        gate = torch.tensor([[0.0], [1.0]])
        wrapper = Gigachat35TopK(
            topk=_FakeTopK(topk_output, TopKOutputFormat.STANDARD),
            shared_expert_sigmoid=_FakeSharedExpertSigmoid(gate),
            num_fused_shared_experts=1,
        )

        hidden_states = torch.randn(2, HIDDEN)
        result = wrapper._apply_shared_expert_sigmoid(hidden_states, topk_output)

        expected = topk_output.topk_weights.clone()
        expected[:, -1:] *= torch.sigmoid(gate.float())

        assert isinstance(result, StandardTopKOutput)
        _assert_close(result.topk_weights, expected)
        assert torch.equal(result.topk_ids, topk_output.topk_ids)
        _assert_close(result.router_logits, topk_output.router_logits)
        _assert_close(topk_output.topk_weights, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    def test_forward_calls_topk_and_applies_shared_expert_sigmoid(self):
        topk_output = StandardTopKOutput(
            topk_weights=torch.tensor([[1.0, 2.0]]),
            topk_ids=torch.tensor([[0, 1]], dtype=torch.long),
            router_logits=torch.tensor([[0.1, 0.2]]),
        )
        topk = _FakeTopK(topk_output, TopKOutputFormat.STANDARD)
        wrapper = Gigachat35TopK(
            topk=topk,
            shared_expert_sigmoid=_FakeSharedExpertSigmoid(torch.tensor([[0.0]])),
            num_fused_shared_experts=1,
        )

        hidden_states = torch.randn(1, HIDDEN)
        router_logits = torch.randn(1, 2)
        result = wrapper(hidden_states, router_logits, sample_arg="value")

        assert len(topk.calls) == 1
        called_hidden_states, called_router_logits, called_kwargs = topk.calls[0]
        _assert_close(called_hidden_states, hidden_states)
        _assert_close(called_router_logits, router_logits)
        assert called_kwargs == {"sample_arg": "value"}
        assert isinstance(result, StandardTopKOutput)


def _gqa_cfg(**overrides):
    cfg = _cfg(
        hidden_size=HIDDEN,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        rope_theta=100000.0,
        rope_scaling=None,
        max_position_embeddings=4096,
        rms_norm_eps=1e-6,
        attention_bias=False,
        gated_attention=True,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


@pytest.mark.unit
class TestGigachat35AttentionGQA:
    def test_init_passes_config_to_qwen3_attention_and_disables_qk_norm(self):
        attn = Gigachat35AttentionGQA(
            _gqa_cfg(),
            layer_id=3,
            quant_config="quant",
            prefix="layers.3.self_attn",
            alt_stream="stream",
        )

        assert attn.init_kwargs == {
            "hidden_size": HIDDEN,
            "num_heads": 4,
            "num_kv_heads": 2,
            "layer_id": 3,
            "rope_theta": 100000.0,
            "rope_scaling": None,
            "head_dim": 2,
            "max_position_embeddings": 4096,
            "quant_config": "quant",
            "rms_norm_eps": 1e-6,
            "attention_bias": False,
            "prefix": "layers.3.self_attn",
            "alt_stream": "stream",
        }
        assert attn.q_norm is None
        assert attn.k_norm is None
        assert attn.gated_attention is True
        assert attn.gate_proj.in_features == HIDDEN
        assert attn.gate_proj.out_features == 4 * 2
        assert attn.gate_proj.kwargs == {
            "bias": False,
            "quant_config": "quant",
            "prefix": "layers.3.self_attn.gate_proj",
            "tp_rank": 0,
            "tp_size": 1,
        }

    def test_init_without_gated_attention_has_no_gate_proj(self):
        attn = Gigachat35AttentionGQA(_gqa_cfg(gated_attention=False), layer_id=0)

        assert attn.gated_attention is False
        assert not hasattr(attn, "gate_proj")

    def test_init_rejects_apply_qk_norm(self):
        with pytest.raises(ValueError, match="apply_qk_norm"):
            Gigachat35AttentionGQA(_gqa_cfg(apply_qk_norm=True), layer_id=0)

    @pytest.mark.parametrize(
        "rope_parameters",
        [
            {"rope_theta": 5000.0, "rope_type": "default"},
            {"rope_theta": 5000.0, "rope_type": "yarn", "factor": 4.0},
        ],
    )
    def test_init_prefers_rope_parameters(self, rope_parameters):
        cfg = _gqa_cfg(rope_parameters=rope_parameters)
        del cfg.rope_theta
        del cfg.rope_scaling

        attn = Gigachat35AttentionGQA(cfg, layer_id=0)

        assert attn.init_kwargs["rope_theta"] == 5000.0
        assert attn.init_kwargs["rope_scaling"] is rope_parameters

    @pytest.mark.parametrize("rope_parameters", [None, {}, {"rope_type": "default"}])
    def test_init_falls_back_to_legacy_rope_fields(self, rope_parameters):
        rope_scaling = {"rope_type": "yarn", "factor": 2.0}
        cfg = _gqa_cfg(rope_parameters=rope_parameters, rope_scaling=rope_scaling)

        attn = Gigachat35AttentionGQA(cfg, layer_id=0)

        assert attn.init_kwargs["rope_theta"] == 100000.0
        assert attn.init_kwargs["rope_scaling"] is rope_scaling

    def test_forward_prepare_native_splits_qkv_and_applies_rope(self):
        attn = Gigachat35AttentionGQA(_gqa_cfg(), layer_id=0)
        rope_calls = []

        def rotary_emb(positions, q, k):
            rope_calls.append(positions)
            return q + 1.0, k + 2.0

        attn.rotary_emb = rotary_emb
        hidden_states = torch.randn(3, HIDDEN)
        positions = torch.arange(3)

        q, k, v = attn.forward_prepare_native(positions, hidden_states)

        qkv, _ = attn.qkv_proj(hidden_states)
        expected_q, expected_k, expected_v = qkv.split(
            [attn.q_size, attn.kv_size, attn.kv_size], dim=-1
        )
        assert len(rope_calls) == 1
        assert rope_calls[0] is positions
        _assert_close(q, expected_q + 1.0)
        _assert_close(k, expected_k + 2.0)
        _assert_close(v, expected_v)

    def test_forward_applies_fp32_sigmoid_gate_before_o_proj(self):
        attn = Gigachat35AttentionGQA(_gqa_cfg(), layer_id=0)
        hidden_states = torch.randn(3, HIDDEN, dtype=torch.bfloat16)
        attn_output = torch.randn(3, attn.q_size, dtype=torch.bfloat16)
        attn_calls = []

        def fake_attn(q, k, v, forward_batch):
            attn_calls.append((q, k, v, forward_batch))
            return attn_output

        attn.attn = fake_attn

        output = attn.forward(torch.arange(3), hidden_states, "forward-batch")

        gate_values, _ = attn.gate_proj(hidden_states)
        gated = (attn_output.float() * torch.sigmoid(gate_values.float())).to(
            torch.bfloat16
        )
        expected, _ = attn.o_proj(gated)
        assert len(attn_calls) == 1
        assert attn_calls[0][3] == "forward-batch"
        assert attn_calls[0][0].shape == (3, attn.q_size)
        assert attn_calls[0][1].shape == (3, attn.kv_size)
        assert attn_calls[0][2].shape == (3, attn.kv_size)
        assert output.dtype == torch.bfloat16
        assert torch.equal(output, expected)

    def test_forward_without_gate_projects_attention_output_directly(self):
        attn = Gigachat35AttentionGQA(_gqa_cfg(gated_attention=False), layer_id=0)
        attn_output = torch.randn(3, attn.q_size)
        attn.attn = lambda q, k, v, forward_batch: attn_output

        output = attn.forward(torch.arange(3), torch.randn(3, HIDDEN), None)

        expected, _ = attn.o_proj(attn_output)
        assert torch.equal(output, expected)


class _RecordingParam:
    def __init__(self):
        self.loads = []

    def weight_loader(self, param, loaded_weight, *shard_id):
        assert param is self
        self.loads.append((loaded_weight, *shard_id))


def _dense_loader_cfg(**overrides):
    cfg = _cfg(
        num_hidden_layers=4,
        num_nextn_predict_layers=1,
        attention_type="LlamaAttention",
        linear_attention_type="Qwen3NextGatedDeltaNet",
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        n_routed_experts=0,
        q_lora_rank=None,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _make_dense_loader(*, model, param_names, draft_model_idx: int = 0):
    loader = _make_loader(
        config=_dense_loader_cfg(), model=model, draft_model_idx=draft_model_idx
    )
    loader.num_fused_shared_experts = 0
    loader.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    loader._maybe_quant_weights_to_fp8_ue8m0 = (
        lambda weights, attn_quant_modules, nextn_conf: weights
    )
    params = {name: _RecordingParam() for name in param_names}
    loader.named_parameters = lambda: params.items()
    return loader, params


def _weight(value: float) -> torch.Tensor:
    return torch.full((1,), value)


@pytest.mark.unit
class TestGigachat35DenseWeightLoading:
    def test_post_load_weights_returns_early_for_llama_attention(self, monkeypatch):
        loader = _make_loader(
            config=_cfg(num_hidden_layers=4, attention_type="LlamaAttention")
        )
        calls = _patch_base_post_load_weights(monkeypatch)

        assert loader.post_load_weights(is_nextn=False, weight_names=["w"]) is None
        assert loader.post_load_weights(is_nextn=True, weight_names=["w"]) is None
        assert calls == []

    def test_do_load_weights_maps_gqa_projections_into_qkv_proj(self, caplog):
        loader, params = _make_dense_loader(
            model=SimpleNamespace(start_layer=0, end_layer=5),
            param_names=[
                "model.layers.3.self_attn.qkv_proj.weight",
                "model.layers.3.self_attn.o_proj.weight",
                "model.layers.3.self_attn.gate_proj.weight",
                "model.layers.3.mlp.gate_up_proj.weight",
                "model.layers.0.self_attn.in_proj_qkvz.weight",
                "model.layers.0.self_attn.out_proj.weight",
                "model.norm.weight",
            ],
        )
        q, k, v, o, gate, mlp_gate, in_proj, out_proj, norm, mtp_q = (
            _weight(float(i)) for i in range(10)
        )
        weights = [
            ("model.layers.3.self_attn.q_proj.weight", q),
            ("model.layers.3.self_attn.k_proj.weight", k),
            ("model.layers.3.self_attn.v_proj.weight", v),
            ("model.layers.3.self_attn.o_proj.weight", o),
            ("model.layers.3.self_attn.gate_proj.weight", gate),
            ("model.layers.3.self_attn.rotary_emb.inv_freq", _weight(-1.0)),
            ("model.layers.3.mlp.gate_proj.weight", mlp_gate),
            ("model.layers.0.self_attn.in_proj_qkvz.weight", in_proj),
            ("model.layers.0.self_attn.out_proj.weight", out_proj),
            ("model.norm.weight", norm),
            ("model.layers.4.self_attn.q_proj.weight", mtp_q),
        ]

        with caplog.at_level(logging.WARNING, logger=_MODULE_NAME):
            loader.do_load_weights(weights, is_nextn=False)

        assert caplog.records == []
        assert params["model.layers.3.self_attn.qkv_proj.weight"].loads == [
            (q, "q"),
            (k, "k"),
            (v, "v"),
        ]
        assert params["model.layers.3.self_attn.o_proj.weight"].loads == [(o,)]
        assert params["model.layers.3.self_attn.gate_proj.weight"].loads == [(gate,)]
        assert params["model.layers.3.mlp.gate_up_proj.weight"].loads == [(mlp_gate, 0)]
        assert params["model.layers.0.self_attn.in_proj_qkvz.weight"].loads == [
            (in_proj,)
        ]
        assert params["model.layers.0.self_attn.out_proj.weight"].loads == [(out_proj,)]
        assert params["model.norm.weight"].loads == [(norm,)]

    def test_do_load_weights_nextn_maps_mtp_layer_into_decoder(self, caplog):
        loader, params = _make_dense_loader(
            model=SimpleNamespace(),
            param_names=[
                "model.decoder.self_attn.qkv_proj.weight",
                "model.decoder.self_attn.o_proj.weight",
                "model.decoder.self_attn.gate_proj.weight",
                "model.decoder.mlp.gate_up_proj.weight",
                "model.enorm.weight",
                "model.hnorm.weight",
                "model.eh_proj.weight",
                "model.shared_head.norm.weight",
            ],
        )
        q, k, v, o, gate, mlp_up, enorm, hnorm, eh_proj, head_norm, main_q = (
            _weight(float(i)) for i in range(11)
        )
        weights = [
            ("model.layers.4.self_attn.q_proj.weight", q),
            ("model.layers.4.self_attn.k_proj.weight", k),
            ("model.layers.4.self_attn.v_proj.weight", v),
            ("model.layers.4.self_attn.o_proj.weight", o),
            ("model.layers.4.self_attn.gate_proj.weight", gate),
            ("model.layers.4.mlp.up_proj.weight", mlp_up),
            ("model.layers.4.enorm.weight", enorm),
            ("model.layers.4.hnorm.weight", hnorm),
            ("model.layers.4.eh_proj.weight", eh_proj),
            ("model.layers.4.shared_head.norm.weight", head_norm),
            ("model.layers.4.shared_head.head.weight", _weight(-1.0)),
            ("model.layers.4.embed_tokens.weight", _weight(-2.0)),
            ("model.layers.3.self_attn.q_proj.weight", main_q),
        ]

        with caplog.at_level(logging.WARNING, logger=_MODULE_NAME):
            loader.do_load_weights(weights, is_nextn=True)

        assert caplog.records == []
        assert params["model.decoder.self_attn.qkv_proj.weight"].loads == [
            (q, "q"),
            (k, "k"),
            (v, "v"),
        ]
        assert params["model.decoder.self_attn.o_proj.weight"].loads == [(o,)]
        assert params["model.decoder.self_attn.gate_proj.weight"].loads == [(gate,)]
        assert params["model.decoder.mlp.gate_up_proj.weight"].loads == [(mlp_up, 1)]
        assert params["model.enorm.weight"].loads == [(enorm,)]
        assert params["model.hnorm.weight"].loads == [(hnorm,)]
        assert params["model.eh_proj.weight"].loads == [(eh_proj,)]
        assert params["model.shared_head.norm.weight"].loads == [(head_norm,)]

    def test_maybe_quant_weights_skips_attention_modules_for_llama_attention(
        self, monkeypatch
    ):
        loader = _make_loader(config=_dense_loader_cfg())
        monkeypatch.setattr(
            _gigachat35.envs.SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN, "get", lambda: True
        )
        monkeypatch.setattr(
            _gigachat35,
            "get_hybrid_gdn_config_view",
            lambda config: SimpleNamespace(full_attention_layer_ids=[3]),
        )
        weights = [("model.layers.3.self_attn.qkv_proj.weight", _weight(1.0))]

        result = loader._maybe_quant_weights_to_fp8_ue8m0(
            weights, ["q_b_proj"], NextNDisabledConfig()
        )

        assert result is weights


class _FakeAttentionModule:
    def __init__(self, **kwargs) -> None:
        self.init_kwargs = kwargs


def _decoder_cfg(**overrides):
    cfg = _cfg(
        num_hidden_layers=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        attention_type="LlamaAttention",
        linear_attention_type="Qwen3NextGatedDeltaNet",
        hidden_size=HIDDEN,
        rms_norm_eps=1e-6,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


@pytest.mark.unit
class TestGigachat35DecoderLayerAttentionSelection:
    @pytest.fixture(autouse=True)
    def _fake_attention_classes(self, monkeypatch):
        monkeypatch.setattr(_gigachat35, "Gigachat35AttentionGQA", _FakeAttentionModule)
        monkeypatch.setattr(
            _gigachat35, "Gigachat35GatedDeltaNet", _FakeAttentionModule
        )

    def test_full_attention_layer_uses_gqa(self):
        layer = Gigachat35DecoderLayer(
            _decoder_cfg(),
            layer_id=3,
            quant_config="quant",
            prefix="model.layers.3",
            alt_stream="stream",
        )

        assert layer.use_linear_attn is False
        assert layer.self_attn.init_kwargs["layer_id"] == 3
        assert layer.self_attn.init_kwargs["quant_config"] == "quant"
        assert layer.self_attn.init_kwargs["alt_stream"] == "stream"
        assert layer.self_attn.init_kwargs["prefix"] == "model.layers.3.self_attn"
        assert layer.layer_communicator.qkv_latent_func is None

    def test_linear_attention_layer_uses_gated_delta_net(self):
        layer = Gigachat35DecoderLayer(
            _decoder_cfg(), layer_id=0, prefix="model.layers.0"
        )

        assert layer.use_linear_attn is True
        assert layer.self_attn.init_kwargs["layer_id"] == 0
        assert layer.self_attn.init_kwargs["prefix"] == "model.layers.0.self_attn"
        assert layer.layer_communicator.qkv_latent_func is None

    def test_nextn_layer_follows_last_main_layer_attention_type(self):
        layer = Gigachat35DecoderLayer(
            _decoder_cfg(), layer_id=0, is_nextn=True, prefix="model.decoder"
        )

        assert layer.use_linear_attn is False
        assert layer.self_attn.init_kwargs["layer_id"] == 0
        assert layer.self_attn.init_kwargs["prefix"] == "model.decoder.self_attn"
        assert layer.layer_communicator.qkv_latent_func is None

    def test_non_llama_full_attention_layer_keeps_mla(self):
        cfg = _decoder_cfg()
        del cfg.attention_type

        layer = Gigachat35DecoderLayer(cfg, layer_id=3, prefix="model.layers.3")

        assert layer.use_linear_attn is False
        assert layer.self_attn == "mla"
        assert layer.layer_communicator.qkv_latent_func == "latent"


@pytest.mark.unit
class TestCompiledGatedNormGatePathIsFp32:
    """After the actor-compatible norm round, the compiled gate MLP must stay fp32.

    Both GEMMs, silu and sigmoid run in fp32 with one round to the stream dtype
    at the end, so the result must match the fp32 reference bit-exactly.
    """

    @staticmethod
    def _actor_norm_output(x, weight, eps, *, zero_centered):
        normed = x.float()
        normed = normed * torch.rsqrt(normed.pow(2).mean(-1, keepdim=True) + eps)
        scale = (1.0 + weight.float()) if zero_centered else weight.float()
        # Actor and rollout both materialize the norm result in the stream dtype
        # before the gate. This is not the fp32 bottleneck contract under test.
        return (normed * scale).to(x.dtype).float()

    @staticmethod
    def _fp32_gate_reference(normed, wu, wd, gating, stream_dtype):
        gate_hidden = F.linear(normed, wu.float())
        gate_hidden = F.silu(gate_hidden)
        gate_hidden = F.linear(gate_hidden, wd.float())
        return (normed * gating * torch.sigmoid(gate_hidden)).to(stream_dtype)

    @staticmethod
    def _bf16_rounded_gate_reference(normed, wu, wd, gating, stream_dtype):
        gate_hidden = F.linear(normed, wu.float()).to(stream_dtype).float()
        gate_hidden = F.silu(gate_hidden).to(stream_dtype).float()
        gate_hidden = F.linear(gate_hidden, wd.float()).to(stream_dtype).float()
        gate = torch.sigmoid(gate_hidden).to(stream_dtype).float()
        return (normed * gating * gate).to(stream_dtype)

    @pytest.mark.parametrize("zero_centered", [True, False])
    def test_bf16_stream_gate_is_bit_exact_with_the_fp32_reference(self, zero_centered):
        torch.manual_seed(0)
        hidden = 128
        x = (torch.randn(17, hidden) * 0.5).to(torch.bfloat16)
        weight = (torch.randn(hidden) * 0.1).to(torch.bfloat16)
        gate_up = (torch.randn(16, hidden) * 0.1).to(torch.bfloat16)
        gate_down = (torch.randn(hidden, 16) * 0.1).to(torch.bfloat16)
        eps, gating = 1e-6, 1.375

        compiled_fn = (
            _compiled_zero_centered_gated_norm if zero_centered else _compiled_gated_norm
        )
        # Compare the impl itself, not inductor's schedule for it: the contract
        # under test is which dtype each op runs in.
        with torch.compiler.set_stance("force_eager"):
            actual = compiled_fn(
                x=x,
                weight=weight,
                gate_up_weight=gate_up,
                gate_down_weight=gate_down,
                eps=eps,
                layernorm_gating_weight=gating,
            )
        normed = self._actor_norm_output(
            x, weight, eps, zero_centered=zero_centered
        )
        expected = self._fp32_gate_reference(
            normed, gate_up, gate_down, gating, x.dtype
        )
        bf16_rounded_gate = self._bf16_rounded_gate_reference(
            normed, gate_up, gate_down, gating, x.dtype
        )

        assert actual.dtype == torch.bfloat16
        assert not torch.equal(expected, bf16_rounded_gate), (
            "test fixture is not sensitive to bf16 round-trips inside the gate MLP"
        )
        assert torch.equal(actual, expected), (
            "compiled gate path diverges from the actor-compatible fp32 gate reference: "
            f"max |diff| = {(actual.float() - expected.float()).abs().max().item()}"
        )
