from __future__ import annotations

import concurrent.futures
import logging
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import tqdm
from torch import nn
from transformers import PretrainedConfig

import sglang.srt.models.deepseek_v2 as deepseek_v2
from sglang.srt.layers.attention.dsa.utils import dsa_use_prefill_cp
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.models.deepseek_common.attention_forward_methods import (
    forward_mla as _fwd_mla,
)
from sglang.srt.configs.hybrid_gdn import (
    get_gigachat35_mtp_attention_layer_id,
    get_hybrid_gdn_config_view,
    gigachat35_mtp_uses_linear_attention,
    is_gigachat35_linear_attention_layer,
)
from sglang.srt.configs.model_config import (
    gigachat35_mtp_is_dsa,
    parse_dense_attention_layers,
)
from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size
from sglang.srt.environ import envs
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.layernorm import (
    GemmaRMSNorm,
    RMSNorm,
    _compiled_zero_centered_gated_norm,
    _compiled_gated_norm,
    _is_triton_available,
)
if _is_triton_available:
    from sglang.srt.layers.layernorm import (
        _gigachat35_fused_gated_norm,
        triton_zero_centered_gated_norm,
    )

from sglang.srt.layers.utils import MultiPlatformOp
from sglang.srt.layers.linear import ColumnParallelLinear, ReplicatedLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import get_moe_runner_backend
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.layers.moe.topk import (
    StandardTopKOutput,
    TopK,
    TopKOutputChecker,
    TopKOutputFormat,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8_utils import quant_weight_ue8m0
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.layers.attention.dsa.dequant_k_cache import dequantize_k_cache_paged
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_token_to_kv_pool,
)
from sglang.srt.model_loader.utils import maybe_executor_submit, should_async_load
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    DeepseekV2WeightLoaderMixin,
    NVFP4_CKPT_FP8_ATTN_QUANT_MODULES,
    NextNDisabledConfig,
    NextNEnabledConfig,
)
from sglang.srt.models.deepseek_common.utils import (
    _is_npu,
    enable_nextn_moe_bf16_cast_to_fp8,
)
from sglang.srt.models.deepseek_nextn import DeepseekModelNextN
from sglang.srt.models.kimi_linear import KimiDeltaAttention
from sglang.srt.models.qwen3 import Qwen3Attention
from sglang.srt.models.qwen3_next import Qwen3GatedDeltaNet, _apply_sigmoid_gate_fp32
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import get_global_server_args
from sglang.srt.state_capturer.indexer_topk import maybe_capture_indexer_topk
from sglang.srt.utils import BumpAllocator, add_prefix, log_info_on_rank0

logger = logging.getLogger(__name__)

_GATED_NORM_LOW_RANK = 16


class Gigachat35GatedNormMixin:
    """Mixin that adds gating (gate MLP bottleneck → sigmoid) to any RMSNorm.

    Subclasses must also inherit from an RMSNorm base (``Gigachat35RMSNorm``
    or ``Gigachat35ZeroCenteredRMSNorm``) that provides ``forward_native``,
    ``forward_cuda``, etc.
    """

    def _init_gate(
        self,
        hidden_size: int,
        layernorm_gating_weight: float = 2.0,
    ) -> None:
        self.r = _GATED_NORM_LOW_RANK
        self.layernorm_gating_weight = layernorm_gating_weight
        self.gate_up_projection = nn.Linear(hidden_size, self.r, bias=False)
        self.gate_down_projection = nn.Linear(self.r, hidden_size, bias=False)

    def _apply_gate(self, x: torch.Tensor) -> torch.Tensor:
        gate_hidden = F.linear(x.float(), self.gate_up_projection.weight.float())
        gate_hidden = F.silu(gate_hidden)
        gate_hidden = F.linear(gate_hidden, self.gate_down_projection.weight.float())
        return (
            x.float()
            * self.layernorm_gating_weight
            * torch.sigmoid(gate_hidden)
        ).to(x.dtype)

    def _finish_forward(self, norm_output):
        if isinstance(norm_output, tuple):
            x, residual = norm_output
            return self._apply_gate(x), residual
        return self._apply_gate(norm_output)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for platform in ("native", "cuda", "npu", "hip", "cpu", "xpu"):
            fwd_name = f"forward_{platform}"
            if fwd_name not in cls.__dict__:
                def _make_forward(plat):
                    def _forward(self, x, residual=None, post_residual_addition=None):
                        parent_fwd = getattr(super(cls, self), f"forward_{plat}")
                        return self._finish_forward(
                            parent_fwd(x, residual, post_residual_addition)
                        )
                    return _forward
                setattr(cls, fwd_name, _make_forward(platform))


class Gigachat35RMSNorm(RMSNorm):
    def __init__(self, hidden_size: int, eps: float = 1e-6, **kwargs) -> None:
        super().__init__(hidden_size, eps=eps)

class Gigachat35ZeroCenteredRMSNorm(GemmaRMSNorm):
    def __init__(self, hidden_size: int, eps: float = 1e-6, **kwargs) -> None:
        super().__init__(hidden_size, eps=eps)

class Gigachat35GatedNorm(Gigachat35GatedNormMixin, Gigachat35RMSNorm):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        layernorm_gating_weight: float = 2.0,
        **kwargs,
    ) -> None:
        super().__init__(hidden_size, eps=eps, **kwargs)
        self._init_gate(hidden_size, layernorm_gating_weight)


class Gigachat35ZeroCenteredGatedNorm(
    Gigachat35GatedNormMixin,
    Gigachat35ZeroCenteredRMSNorm,
):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        layernorm_gating_weight: float = 2.0,
        **kwargs,
    ) -> None:
        super().__init__(hidden_size, eps=eps, **kwargs)
        self._init_gate(hidden_size, layernorm_gating_weight)


class _BaseGatedNormTorchCompile(MultiPlatformOp):
    """Shared base for fused ``torch.compile``'d gated RMSNorm classes.

    Subclasses override ``_get_compiled_fn`` to return the compiled function
    and ``_make_weight`` to return the weight parameter with correct init.
    """

    @staticmethod
    def _get_compiled_fn() -> callable:
        raise NotImplementedError

    @staticmethod
    def _make_weight(hidden_size: int) -> nn.Parameter:
        raise NotImplementedError

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        layernorm_gating_weight: float = 2.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.variance_epsilon = eps
        self.r = _GATED_NORM_LOW_RANK
        self.weight = self._make_weight(hidden_size)
        self.gate_up_projection = nn.Linear(hidden_size, self.r, bias=False)
        self.gate_down_projection = nn.Linear(self.r, hidden_size, bias=False)
        self.layernorm_gating_weight = layernorm_gating_weight

    def _apply(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor],
        post_residual_addition: Optional[torch.Tensor],
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if residual is not None:
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            x = x + residual
            residual = x

        with torch.autocast(device_type=x.device.type, dtype=torch.float32):
            if not torch.compiler.is_compiling():
                torch._dynamo.mark_dynamic(x, 0)
                torch._dynamo.decorators.mark_unbacked(x, 0)
                for _d in range(1, x.dim() - 1):  # exclude last (hidden) dim: it's specialized to 7168
                    torch._dynamo.mark_dynamic(x, _d)
            out = self._get_compiled_fn()(
                x=x,
                weight=self.weight,
                gate_up_weight=self.gate_up_projection.weight,
                gate_down_weight=self.gate_down_projection.weight,
                eps=self.variance_epsilon,
                layernorm_gating_weight=self.layernorm_gating_weight,
            )

        return out if residual is None else (out, residual)

    def forward_native(self, x, residual=None, post_residual_addition=None):
        return self._apply(x, residual, post_residual_addition)

    def forward_cuda(self, x, residual=None, post_residual_addition=None):
        return self._apply(x, residual, post_residual_addition)

    def forward_npu(self, x, residual=None, post_residual_addition=None):
        return self._apply(x, residual, post_residual_addition)

    def forward_hip(self, x, residual=None, post_residual_addition=None):
        return self._apply(x, residual, post_residual_addition)

    def forward_cpu(self, x, residual=None, post_residual_addition=None):
        return self._apply(x, residual, post_residual_addition)

    def forward_xpu(self, x, residual=None, post_residual_addition=None):
        return self._apply(x, residual, post_residual_addition)

    def extra_repr(self) -> str:
        return (
            f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}, "
            f"layernorm_gating_weight={self.layernorm_gating_weight}, "
            f"r={self.r}"
        )


class Gigachat35ZeroCenteredGatedNormTorchCompile(_BaseGatedNormTorchCompile):
    """Fused ``torch.compile``'d zero-centered gated RMSNorm for Gigachat35."""

    @staticmethod
    def _make_weight(hidden_size: int) -> nn.Parameter:
        return nn.Parameter(torch.zeros(hidden_size))

    @staticmethod
    def _get_compiled_fn():
        return _compiled_zero_centered_gated_norm


if _is_triton_available:

    class Gigachat35ZeroCenteredGatedNormTriton(Gigachat35ZeroCenteredGatedNormTorchCompile):
        """Fused Triton zero-centered gated RMSNorm kernel for Gigachat35.

        Inherits all forward methods from ``Gigachat35ZeroCenteredGatedNormTorchCompile``
        and only overrides ``_apply`` to dispatch to the Triton kernel.
        """

        def _apply(self, x, residual, post_residual_addition):
            if residual is not None:
                if post_residual_addition is not None:
                    residual = residual + post_residual_addition
                x = x + residual
                residual = x

            out = triton_zero_centered_gated_norm(
                x=x,
                weight=self.weight,
                gate_up_weight=self.gate_up_projection.weight,
                gate_down_weight=self.gate_down_projection.weight,
                eps=self.variance_epsilon,
                layernorm_gating_weight=self.layernorm_gating_weight,
            )

            return out if residual is None else (out, residual)

    class Gigachat35ZeroCenteredGatedNormTritonFused(
        Gigachat35GatedNormMixin,
        Gigachat35ZeroCenteredRMSNorm,
    ):
        def __init__(
            self,
            hidden_size: int,
            eps: float = 1e-6,
            layernorm_gating_weight: float = 2.0,
            **kwargs,
        ) -> None:
            super().__init__(hidden_size, eps=eps, **kwargs)
            self._init_gate(hidden_size, layernorm_gating_weight)

        def forward_cuda(
            self,
            x: torch.Tensor,
            residual: Optional[torch.Tensor] = None,
            post_residual_addition: Optional[torch.Tensor] = None,
        ):
            return _gigachat35_fused_gated_norm(
                x,
                residual,
                post_residual_addition,
                self.weight.data,
                self.gate_up_projection.weight,
                self.gate_down_projection.weight,
                self.variance_epsilon,
                self.layernorm_gating_weight,
            )

    class Gigachat35GatedNormTritonFused(
        Gigachat35GatedNormMixin,
        Gigachat35RMSNorm,
    ):
        """V3 fused kernel for the plain-scale ``GatedNorm`` (``weight`` instead of ``1 + weight``).

        HF ``norm_type=GatedNorm`` (the 37B checkpoints) is the same gate MLP over a plain
        RMSNorm: a ones-initialised ``weight`` instead of ``1 + weight`` with zeros. Same
        three launches as the zero-centered class; the scale is a constexpr of the kernel.
        """

        def __init__(
            self,
            hidden_size: int,
            eps: float = 1e-6,
            layernorm_gating_weight: float = 2.0,
            **kwargs,
        ) -> None:
            super().__init__(hidden_size, eps=eps, **kwargs)
            self._init_gate(hidden_size, layernorm_gating_weight)

        def forward_cuda(
            self,
            x: torch.Tensor,
            residual: Optional[torch.Tensor] = None,
            post_residual_addition: Optional[torch.Tensor] = None,
        ):
            return _gigachat35_fused_gated_norm(
                x,
                residual,
                post_residual_addition,
                self.weight.data,
                self.gate_up_projection.weight,
                self.gate_down_projection.weight,
                self.variance_epsilon,
                self.layernorm_gating_weight,
                zero_centered=False,
            )


class Gigachat35GatedNormTorchCompile(_BaseGatedNormTorchCompile):
    """Fused ``torch.compile``'d gated RMSNorm for Gigachat35 (non-zero-centered)."""

    @staticmethod
    def _make_weight(hidden_size: int) -> nn.Parameter:
        return nn.Parameter(torch.ones(hidden_size))

    @staticmethod
    def _get_compiled_fn():
        return _compiled_gated_norm


_GATED_NORM_IMPL_MAP: Dict[str, type] = {
    "native": Gigachat35ZeroCenteredGatedNorm,
    "torch_compile": Gigachat35ZeroCenteredGatedNormTorchCompile,
}
if _is_triton_available:
    _GATED_NORM_IMPL_MAP["triton"] = Gigachat35ZeroCenteredGatedNormTriton
    _GATED_NORM_IMPL_MAP["triton_fused"] = Gigachat35ZeroCenteredGatedNormTritonFused


def resolve_gated_norm_class(gated_norm_impl: str) -> type:
    """Return the gated-RMSNorm class for a given ``gated_norm_impl`` value.

    Raises ``ValueError`` on an unknown impl so config typos surface immediately.
    """
    try:
        return _GATED_NORM_IMPL_MAP[gated_norm_impl]
    except KeyError:
        available = ", ".join(sorted(_GATED_NORM_IMPL_MAP))
        raise ValueError(
            f"Unknown gated_norm_impl={gated_norm_impl!r}. Available: {available}."
        )


def resolve_norm_class(config: PretrainedConfig) -> type[nn.Module]:
    norm_type = getattr(config, "norm_type", "LlamaRMSNorm")
    if norm_type == "LlamaRMSNorm":
        return Gigachat35RMSNorm
    if norm_type == "ZeroCenteredRMSNorm":
        return Gigachat35ZeroCenteredRMSNorm
    if norm_type == "ZeroCenteredGatedNorm":
        gated_norm_impl = getattr(
            get_global_server_args(), "gated_norm_impl", "native"
        )
        if (
            gated_norm_impl == "triton_fused"
            and torch.get_default_dtype() != torch.bfloat16
        ):
            logger.warning(
                "triton_fused gated_norm_impl requires a bfloat16 model, "
                f"got {torch.get_default_dtype()}. Falling back to torch_compile."
            )
            gated_norm_impl = "torch_compile"
        return resolve_gated_norm_class(gated_norm_impl)
    if norm_type == "GatedNorm":
        gated_norm_impl = getattr(
            get_global_server_args(), "gated_norm_impl", "native"
        )
        if gated_norm_impl == "triton_fused":
            if not _is_triton_available:
                raise ValueError(
                    "triton_fused gated_norm_impl requires triton, which is not available."
                )
            if torch.get_default_dtype() != torch.bfloat16:
                logger.warning(
                    "triton_fused gated_norm_impl requires a bfloat16 model, "
                    f"got {torch.get_default_dtype()}. Falling back to torch_compile."
                )
                return Gigachat35GatedNormTorchCompile
            return Gigachat35GatedNormTritonFused
        if gated_norm_impl == "triton":
            logger.warning(
                "triton gated_norm_impl is not supported for non-zero-centered "
                "GatedNorm. Falling back to torch_compile."
            )
            return Gigachat35GatedNormTorchCompile
        if gated_norm_impl == "torch_compile":
            return Gigachat35GatedNormTorchCompile
        return Gigachat35GatedNorm
    raise ValueError(f"Unsupported norm_type for Gigachat35 SGLang runtime: {norm_type}")


def build_norm(config: PretrainedConfig, hidden_size: int) -> nn.Module:
    norm_class = resolve_norm_class(config)
    kwargs = {"eps": config.rms_norm_eps}
    # Every gated norm, whichever impl serves it, derives from one of these two bases.
    if issubclass(norm_class, (Gigachat35GatedNormMixin, _BaseGatedNormTorchCompile)):
        kwargs["layernorm_gating_weight"] = getattr(config, "layernorm_gating_weight", 2.0)
    return norm_class(hidden_size, **kwargs)


def is_gigachat35_dsa_attention_layer(
    config: PretrainedConfig,
    layer_id: int,
) -> bool:
    """Return whether an MLA layer uses the sparse DSA indexer."""
    # List shape / range validation is shared with the DeepSeek path.
    dense_attention_layers = parse_dense_attention_layers(config)
    if (
        dense_attention_layers
        and getattr(config, "attention_type", None) != "LlamaLatentSparseAttention"
    ):
        raise ValueError(
            "dense_attention_layers is supported only with "
            "attention_type='LlamaLatentSparseAttention'."
        )

    num_hidden_layers = getattr(config, "num_hidden_layers", 0)
    linear_attention_type = getattr(config, "linear_attention_type", None)
    full_attention_layer_ids = (
        set(getattr(config, "full_attention_layers", []))
        if linear_attention_type is not None
        else set(range(num_hidden_layers))
    )
    non_full_dense_layer_ids = dense_attention_layers - full_attention_layer_ids
    if non_full_dense_layer_ids:
        raise ValueError(
            "dense_attention_layers must be a subset of full_attention_layers; "
            f"invalid indices: {sorted(non_full_dense_layer_ids)}."
        )
    if getattr(config, "index_topk", None) is None:
        return False
    return (
        layer_id in full_attention_layer_ids
        and layer_id not in dense_attention_layers
    )


class Gigachat35GatedDeltaNet(Qwen3GatedDeltaNet):
    def op_prepare(self, state):
        state.hidden_states_after_attn = self(
            hidden_states=state.pop("hidden_states_after_comm_pre_attn"),
            forward_batch=state.forward_batch,
        )

    def op_core(self, state):
        pass


def _prepare_kimi_delta_attention_config(config: PretrainedConfig) -> None:
    if config.linear_key_head_dim != config.linear_value_head_dim:
        raise ValueError(
            "KimiDeltaAttention expects converted weights/config with "
            "linear_key_head_dim == linear_value_head_dim."
        )
    if config.linear_num_value_heads % config.linear_num_key_heads != 0:
        raise ValueError(
            "KimiDeltaAttention expects linear_num_value_heads to be divisible "
            "by linear_num_key_heads."
        )
    config.linear_attn_config = {
        "head_dim": config.linear_value_head_dim,
        "key_head_dim": config.linear_key_head_dim,
        "value_head_dim": config.linear_value_head_dim,
        "num_heads": config.linear_num_value_heads,
        "num_key_heads": config.linear_num_key_heads,
        "num_value_heads": config.linear_num_value_heads,
        "short_conv_kernel_size": config.linear_conv_kernel_dim,
    }
    dtype = getattr(config, "dtype", getattr(config, "torch_dtype", torch.bfloat16))
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    config.dtype = dtype


class Gigachat35KimiDeltaAttention(KimiDeltaAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        return super().forward(
            hidden_states=hidden_states,
            positions=None,
            forward_batch=forward_batch,
            zero_allocator=None,
        )

    def op_prepare(self, state):
        state.hidden_states_after_attn = self(
            hidden_states=state.pop("hidden_states_after_comm_pre_attn"),
            forward_batch=state.forward_batch,
        )

    def op_core(self, state):
        pass


class Gigachat35HybridGDNWeightLoaderMixin(DeepseekV2WeightLoaderMixin):
    def _get_hybrid_gdn_config_view(self):
        return get_hybrid_gdn_config_view(self.config)

    def _is_removed_dense_indexer_weight(
        self,
        name: str,
        layer_id: Optional[int],
    ) -> bool:
        if layer_id is None or ".self_attn.indexer." not in name:
            return False
        # Two-class split: a main dense full-attn layer's indexer is removed.
        if layer_id in (getattr(self.config, "dense_attention_layers", None) or []):
            return True
        # A dense MTP/NextN layer (nextn_is_dsa resolved False) ignores its
        # checkpoint indexer -- drop those weights instead of failing to place them.
        num_hidden_layers = getattr(self.config, "num_hidden_layers", None)
        return (
            num_hidden_layers is not None
            and layer_id >= num_hidden_layers
            and not gigachat35_mtp_is_dsa(self.config)
        )

    def do_load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        is_nextn: bool = False,
    ):
        nextn_conf = self._initialize_nextn_conf(is_nextn)

        weights = self._maybe_quant_weights_to_fp8_ue8m0(
            weights, NVFP4_CKPT_FP8_ATTN_QUANT_MODULES, nextn_conf
        )

        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        gqa_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        kimi_delta_params_mapping = [
            ("fused_qkvbfg_a_proj", "q_proj", 0),
            ("fused_qkvbfg_a_proj", "k_proj", 1),
            ("fused_qkvbfg_a_proj", "v_proj", 2),
            ("fused_qkvbfg_a_proj", "b_proj", 3),
            ("fused_qkvbfg_a_proj", "f_a_proj", 4),
            ("fused_qkvbfg_a_proj", "g_a_proj", 5),
            ("fused_fg_b_proj", "f_b_proj", 0),
            ("fused_fg_b_proj", "g_b_proj", 1),
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts + self.num_fused_shared_experts,
        )
        if self.quant_config and self.quant_config.get_name() == "w4afp8":
            expert_params_mapping += FusedMoE.make_expert_input_scale_params_mapping(
                num_experts=self.config.n_routed_experts
            )

        fuse_qkv_a_proj = hasattr(self.config, "q_lora_rank") and (
            self.config.q_lora_rank is not None
        )
        cached_a_proj = {} if fuse_qkv_a_proj else None

        if self.num_fused_shared_experts > 0:
            assert self.num_fused_shared_experts == 1
            log_info_on_rank0(logger, "Shared experts fusion optimization enabled.")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            params_dict = dict(self.named_parameters())
            weight_names = []
            for name, loaded_weight in weights:
                use_async_loading = should_async_load(loaded_weight)
                layer_id = get_layer_id(name)
                if self._is_removed_dense_indexer_weight(name, layer_id):
                    continue
                if (
                    layer_id is not None
                    and hasattr(self.model, "start_layer")
                    and (
                        layer_id < self.model.start_layer
                        or layer_id >= self.model.end_layer
                    )
                ):
                    continue
                if self.num_fused_shared_experts > 0 and "mlp.shared_experts" in name:
                    name = name.replace(
                        "mlp.shared_experts",
                        f"mlp.experts.{self.config.n_routed_experts}",
                    )

                weight_names.append(name)

                match nextn_conf:
                    case NextNEnabledConfig(
                        nextn_layer_prefix=layer_prefix,
                        nextn_spec_weight_names=spec_weight_names,
                    ):
                        if not name.startswith(layer_prefix):
                            continue

                        if "shared_head.head" in name or "embed_tokens" in name:
                            continue

                        if any(s in name for s in spec_weight_names):
                            name = name.replace(layer_prefix, "model")
                        else:
                            name = name.replace(layer_prefix, "model.decoder")
                    case NextNDisabledConfig():
                        if hasattr(self.config, "num_nextn_predict_layers"):
                            num_nextn_layers = self.config.num_nextn_predict_layers
                            if num_nextn_layers > 0 and name.startswith("model.layers"):
                                name_list = name.split(".")
                                if (
                                    len(name_list) >= 3
                                    and int(name_list[2])
                                    >= self.config.num_hidden_layers
                                ):
                                    continue

                if "rotary_emb.inv_freq" in name:
                    continue

                if (
                    layer_id is not None
                    and getattr(self.config, "linear_attention_type", None)
                    == "KimiDeltaAttention"
                    and is_gigachat35_linear_attention_layer(self.config, layer_id)
                    and ".self_attn." in name
                ):
                    mapped_kimi_delta_param = False
                    for param_name, weight_name, shard_id in kimi_delta_params_mapping:
                        weight_key = f".{weight_name}."
                        if weight_key not in name:
                            continue
                        mapped_name = name.replace(
                            weight_key, f".{param_name}.", 1
                        )
                        if mapped_name not in params_dict:
                            continue
                        param = params_dict[mapped_name]
                        weight_loader = param.weight_loader
                        maybe_executor_submit(
                            executor=executor,
                            futures=futures,
                            use_async=use_async_loading,
                            func=weight_loader,
                            func_args=(param, loaded_weight, shard_id),
                        )
                        mapped_kimi_delta_param = True
                        break
                    if mapped_kimi_delta_param:
                        continue

                if is_nextn:
                    is_softmax_attn_layer = not gigachat35_mtp_uses_linear_attention(
                        self.config
                    )
                else:
                    is_softmax_attn_layer = (
                        layer_id is not None
                        and not is_gigachat35_linear_attention_layer(
                            self.config, layer_id
                        )
                    )
                if (
                    getattr(self.config, "attention_type", None) == "LlamaAttention"
                    and is_softmax_attn_layer
                    and ".self_attn." in name
                ):
                    mapped_gqa_param = False
                    for param_name, weight_name, shard_id in gqa_params_mapping:
                        weight_key = f".{weight_name}."
                        if weight_key not in name:
                            continue
                        mapped_name = name.replace(weight_key, f".{param_name}.", 1)
                        if mapped_name not in params_dict:
                            continue
                        param = params_dict[mapped_name]
                        weight_loader = param.weight_loader
                        maybe_executor_submit(
                            executor=executor,
                            futures=futures,
                            use_async=use_async_loading,
                            func=weight_loader,
                            func_args=(param, loaded_weight, shard_id),
                        )
                        mapped_gqa_param = True
                        break
                    if mapped_gqa_param:
                        continue

                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if ".mlp." not in name:
                        continue
                    if weight_name not in name:
                        continue
                    if _is_npu:
                        name = name.replace("weight_packed", "weight")
                    if ("mlp.experts." in name) and name not in params_dict:
                        continue
                    name = name.replace(weight_name, param_name)
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    maybe_executor_submit(
                        executor=executor,
                        futures=futures,
                        use_async=use_async_loading,
                        func=weight_loader,
                        func_args=(param, loaded_weight, shard_id),
                    )
                    break
                else:
                    for mapping in expert_params_mapping:
                        param_name, weight_name, expert_id, shard_id = mapping
                        if weight_name not in name:
                            continue
                        if _is_npu:
                            name = name.replace("weight_packed", "weight")
                        name = name.replace(weight_name, param_name)
                        if name not in params_dict:
                            continue
                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        maybe_executor_submit(
                            executor=executor,
                            futures=futures,
                            use_async=use_async_loading,
                            func=weight_loader,
                            func_args=(
                                param,
                                loaded_weight,
                                name,
                            ),
                            func_kwargs={
                                "shard_id": shard_id,
                                "expert_id": expert_id,
                            },
                        )
                        break
                    else:
                        if name.endswith(".bias") and name not in params_dict:
                            continue
                        if ".embed_tokens." in name and not self.pp_group.is_first_rank:
                            continue
                        if (
                            name.startswith("model.norm.")
                            and not self.pp_group.is_last_rank
                        ):
                            continue
                        if fuse_qkv_a_proj and (
                            "q_a_proj" in name or "kv_a_proj_with_mqa" in name
                        ):
                            cached_a_proj[name] = loaded_weight
                            q_a_proj_name = (
                                name
                                if "q_a_proj" in name
                                else name.replace("kv_a_proj_with_mqa", "q_a_proj")
                            )
                            kv_a_proj_name = (
                                name
                                if "kv_a_proj_with_mqa" in name
                                else name.replace("q_a_proj", "kv_a_proj_with_mqa")
                            )

                            if (
                                q_a_proj_name in cached_a_proj
                                and kv_a_proj_name in cached_a_proj
                            ):
                                q_a_proj_weight = cached_a_proj[q_a_proj_name]
                                kv_a_proj_weight = cached_a_proj[kv_a_proj_name]

                                if q_a_proj_weight.shape == torch.Size(
                                    []
                                ) and kv_a_proj_weight.shape == torch.Size([]):
                                    fused_weight = q_a_proj_weight
                                else:
                                    cat_dim = 0
                                    if self.quant_config is not None and (
                                        self.quant_config.get_name() == "awq"
                                        or self.quant_config.get_name() == "awq_marlin"
                                        or self.quant_config.get_name() == "moe_wna16"
                                    ):
                                        cat_dim = 1

                                    fused_weight = torch.cat(
                                        [q_a_proj_weight, kv_a_proj_weight], dim=cat_dim
                                    )

                                param_name = (
                                    name.replace(
                                        "q_a_proj", "fused_qkv_a_proj_with_mqa"
                                    )
                                    if "q_a_proj" in name
                                    else name.replace(
                                        "kv_a_proj_with_mqa",
                                        "fused_qkv_a_proj_with_mqa",
                                    )
                                )
                                param = params_dict[param_name]

                                weight_loader = getattr(
                                    param, "weight_loader", default_weight_loader
                                )
                                maybe_executor_submit(
                                    executor=executor,
                                    futures=futures,
                                    use_async=use_async_loading,
                                    func=weight_loader,
                                    func_args=(param, fused_weight),
                                )
                                cached_a_proj.pop(q_a_proj_name)
                                cached_a_proj.pop(kv_a_proj_name)
                        else:
                            if (
                                "k_scale" in name or "v_scale" in name
                            ) and name not in params_dict:
                                for scale in ["k_scale", "v_scale"]:
                                    if scale in name:
                                        name = name.replace(
                                            f"{scale[0]}_proj", "attn_mqa"
                                        )
                                        break
                            if name not in params_dict:
                                logger.warning(f"{name} not found in params_dict.")
                                continue
                            param = params_dict[name]
                            weight_loader = getattr(
                                param, "weight_loader", default_weight_loader
                            )
                            maybe_executor_submit(
                                executor=executor,
                                futures=futures,
                                use_async=use_async_loading,
                                func=weight_loader,
                                func_args=(param, loaded_weight),
                            )

            for future in concurrent.futures.as_completed(futures):
                future.result()

        self.post_load_weights(is_nextn=is_nextn, weight_names=weight_names)

    def post_load_weights(
        self,
        is_nextn: bool = False,
        weight_names: Optional[Iterable[str]] = None,
    ) -> None:
        if getattr(self.config, "attention_type", None) == "LlamaAttention":
            return

        if is_nextn:
            if gigachat35_mtp_uses_linear_attention(self.config):
                # MTP decoder uses linear attention — no kv_b_proj weight absorption needed.
                return
            return super().post_load_weights(
                is_nextn=is_nextn,
                weight_names=weight_names,
            )

        hybrid_gdn_config = self._get_hybrid_gdn_config_view()
        if hybrid_gdn_config is not None:
            full_attention_layer_ids = set(hybrid_gdn_config.full_attention_layer_ids)
        else:
            num_layers = self.config.num_hidden_layers
            full_attention_layer_ids = {
                lid
                for lid in range(num_layers)
                if not is_gigachat35_linear_attention_layer(self.config, lid)
            }
            if len(full_attention_layer_ids) == num_layers:
                return super().post_load_weights(
                    is_nextn=is_nextn,
                    weight_names=weight_names,
                )
        if weight_names is None:
            full_attention_weight_names = [
                f"model.layers.{layer_id}.self_attn.kv_b_proj.weight"
                for layer_id in full_attention_layer_ids
                if self.model.start_layer <= layer_id < self.model.end_layer
            ]
        else:
            full_attention_weight_names = [
                name
                for name in weight_names
                if (layer_id := get_layer_id(name)) in full_attention_layer_ids
            ]
        return super().post_load_weights(
            is_nextn=False,
            weight_names=full_attention_weight_names,
        )

    def _maybe_quant_weights_to_fp8_ue8m0(
        self,
        weights,
        attn_quant_modules,
        nextn_conf,
    ):
        weight_block_size = [128, 128]
        partial_names = []
        if getattr(self.config, "attention_type", None) == "LlamaAttention":
            attn_quant_modules = []

        match nextn_conf:
            case NextNEnabledConfig(nextn_layer_id=layer_id):
                if (
                    envs.SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN.get()
                    and not gigachat35_mtp_uses_linear_attention(self.config)
                ):
                    for stem in attn_quant_modules:
                        partial_names.append(
                            f"model.layers.{layer_id}.self_attn.{stem}"
                        )

                if enable_nextn_moe_bf16_cast_to_fp8(self.config, self.quant_config):
                    expert_sub_names = ["shared_experts"] + [
                        f"experts.{i}" for i in range(self.config.n_routed_experts)
                    ]
                    for expert_sub_name in expert_sub_names:
                        for stem in ["gate_proj", "up_proj", "down_proj"]:
                            partial_names.append(
                                f"model.layers.{layer_id}.mlp.{expert_sub_name}.{stem}"
                            )

            case NextNDisabledConfig():
                hybrid_gdn_config = self._get_hybrid_gdn_config_view()
                if hybrid_gdn_config is None:
                    return super()._maybe_quant_weights_to_fp8_ue8m0(
                        weights,
                        attn_quant_modules,
                        nextn_conf,
                    )

                if envs.SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN.get():
                    for layer_id in hybrid_gdn_config.full_attention_layer_ids:
                        for stem in attn_quant_modules:
                            partial_names.append(
                                f"model.layers.{layer_id}.self_attn.{stem}"
                            )

        if not partial_names:
            return weights

        weights_dict = dict(weights)

        for partial_name in tqdm.tqdm(partial_names, desc="quant weights to fp8 ue8m0"):
            original_weight = weights_dict[f"{partial_name}.weight"]
            out_w, out_s = quant_weight_ue8m0(
                original_weight, weight_block_size=weight_block_size
            )
            weights_dict[f"{partial_name}.weight"] = out_w
            weights_dict[f"{partial_name}.weight_scale_inv"] = out_s

        if isinstance(
            nextn_conf, NextNEnabledConfig
        ) and enable_nextn_moe_bf16_cast_to_fp8(self.config, self.quant_config):
            self._mark_nextn_moe_weights_as_ue8m0()

        return list(weights_dict.items())


class Gigachat35PassthroughNorm(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ):
        if post_residual_addition is not None:
            x = x + post_residual_addition
        if residual is None:
            return x, x
        merged = x + residual
        return merged, merged

    forward_native = forward
    forward_cuda = forward
    forward_hip = forward
    forward_npu = forward
    forward_cpu = forward
    forward_xpu = forward


class Gigachat35MlpPrepNorm(nn.Module):
    def __init__(
        self,
        pre_layernorm: Optional[nn.Module],
        post_layernorm: Optional[nn.Module],
    ) -> None:
        super().__init__()
        self.pre_layernorm = pre_layernorm
        self.post_layernorm = post_layernorm

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ):
        if self.post_layernorm is not None:
            x = self.post_layernorm(x)
        if post_residual_addition is not None:
            x = x + post_residual_addition

        if residual is None:
            merged = x
        else:
            merged = x + residual

        if self.pre_layernorm is None:
            return merged, merged
        return self.pre_layernorm(merged), merged


class Gigachat35TopK(nn.Module):
    def __init__(
        self,
        topk: TopK,
        shared_expert_sigmoid: ReplicatedLinear,
        num_fused_shared_experts: int,
    ) -> None:
        super().__init__()
        self.topk = topk
        self.shared_expert_sigmoid = shared_expert_sigmoid
        self.num_fused_shared_experts = num_fused_shared_experts

    @property
    def topk_config(self):
        return self.topk.topk_config

    def supports_fused_shared_expert_sigmoid(self) -> bool:
        output_format = self.topk.topk_config.output_format
        if output_format is None:
            backend = get_moe_runner_backend()
            if backend.is_triton_kernels():
                output_format = TopKOutputFormat.TRITON_KERNEL
            elif backend.is_flashinfer_trtllm() or backend.is_flashinfer_mxfp4():
                output_format = TopKOutputFormat.BYPASSED
            else:
                output_format = TopKOutputFormat.STANDARD
        return output_format == TopKOutputFormat.STANDARD

    def empty_topk_output(self, device: torch.device):
        return self.topk.empty_topk_output(device)

    def _apply_shared_expert_sigmoid(
        self, hidden_states: torch.Tensor, topk_output
    ):
        if hidden_states.shape[0] == 0 or self.num_fused_shared_experts == 0:
            return topk_output
        if not TopKOutputChecker.format_is_standard(topk_output):
            return topk_output

        gate, _ = self.shared_expert_sigmoid(hidden_states)
        gate = torch.sigmoid(gate.float())
        topk_weights = topk_output.topk_weights.clone()
        topk_weights[:, -self.num_fused_shared_experts :] *= gate
        return StandardTopKOutput(
            topk_weights=topk_weights,
            topk_ids=topk_output.topk_ids,
            router_logits=topk_output.router_logits,
        )

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor, **kwargs):
        topk_output = self.topk(
            hidden_states=hidden_states,
            router_logits=router_logits,
            **kwargs,
        )
        return self._apply_shared_expert_sigmoid(hidden_states, topk_output)


class Gigachat35MoE(deepseek_v2.DeepseekV2MoE):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        is_nextn: bool = False,
        is_deepseek_v4: bool = False,
        dsa_enable_prefill_cp: bool = False,
        mla_enable_prefill_cp: bool = False,
    ) -> None:
        super().__init__(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=prefix,
            alt_stream=alt_stream,
            is_nextn=is_nextn,
            is_deepseek_v4=is_deepseek_v4,
            dsa_enable_prefill_cp=dsa_enable_prefill_cp,
            mla_enable_prefill_cp=mla_enable_prefill_cp,
        )
        # Hard SwiGLU clip: shared MLP via swiglu_limit, MoE via gemm1_clamp_limit.
        # Fused FP8 expert kernels drop the clamp -> hard-fail below.
        swiglu_limit = float(getattr(config, "swiglu_limit", 0.0))
        if swiglu_limit > 0:
            self._assert_fp8_clipping_supported(quant_config, swiglu_limit)
            if isinstance(getattr(self, "shared_experts", None), deepseek_v2.DeepseekV2MLP):
                self.shared_experts.swiglu_limit = swiglu_limit
            experts = getattr(self, "experts", None)
            if experts is not None and hasattr(experts, "moe_runner_config"):
                experts.moe_runner_config.gemm1_clamp_limit = swiglu_limit

        self.use_shared_expert_sigmoid = bool(getattr(config, "use_shared_expert_sigmoid", False))
        if self.use_shared_expert_sigmoid:
            self.shared_expert_sigmoid = ReplicatedLinear(
                config.hidden_size,
                1,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("shared_expert_sigmoid", prefix),
            )
            if self.num_fused_shared_experts > 0:
                wrapped_topk = Gigachat35TopK(
                    topk=self.topk,
                    shared_expert_sigmoid=self.shared_expert_sigmoid,
                    num_fused_shared_experts=self.num_fused_shared_experts,
                )
                if wrapped_topk.supports_fused_shared_expert_sigmoid():
                    self.topk = wrapped_topk

    _SUPPORTED_CLAMP_BACKENDS = {"triton", "triton_llmf", "deep_gemm"}

    @staticmethod
    def _resolve_moe_runner_backend() -> str:
        """Resolve the effective MoE runner backend, folding AUTO the same way
        Fp8MoEMethod.create_moe_runner() does."""
        backend = get_moe_runner_backend()
        if not backend.is_auto():
            return backend.value
        # Replicate Fp8MoEMethod.is_deepgemm_moe_runner_backend_enabled()
        if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and (
            get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake()
        ):
            return "deep_gemm"
        return "triton"

    @staticmethod
    def _assert_fp8_clipping_supported(
        quant_config: Optional[QuantizationConfig], swiglu_limit: float
    ) -> None:
        """Hard-fail if `swiglu_limit > 0` is requested with a MoE runner
        backend that does NOT propagate `gemm1_clamp_limit` into the expert
        activation function.

        Supported fp8 backends (clamping is wired):
          - triton / triton_llmf
          - deep_gemm (contiguous path; masked path asserts at runtime)

        Unsupported backends (clamping silently no-ops):
          - triton_kernel (bf16-only in Fp8MoEMethod)
          - cutlass (not fully debugged for fp8+clamp yet)
          - flashinfer_trtllm / flashinfer_cutlass / flashinfer_cutedsl /
            flashinfer_mxfp4 / marlin / auto (when not resolved above)."""
        if swiglu_limit <= 0 or quant_config is None:
            return
        name = quant_config.get_name()
        is_fp8 = name in {"fp8", "modelopt_fp8"}
        if not is_fp8 and name == "compressed_tensors":
            for attr in ("weight_dtype", "act_dtype", "activation_dtype"):
                value = getattr(quant_config, attr, None)
                if value is not None and "fp8" in str(value).lower():
                    is_fp8 = True
                    break
        if not is_fp8:
            return
        backend = Gigachat35MoE._resolve_moe_runner_backend()
        if backend in Gigachat35MoE._SUPPORTED_CLAMP_BACKENDS:
            return
        raise AssertionError(
            f"Gigachat35MoE: swiglu_limit={swiglu_limit} is set, but the "
            f"MoE runner backend is '{backend}' (resolved from "
            f"server_args.moe_runner_backend). SwiGLU activation clipping "
            f"(swiglu_limit > 0) is only supported for backends: "
            f"{sorted(Gigachat35MoE._SUPPORTED_CLAMP_BACKENDS)}. "
            f"Set server_args.moe_runner_backend to one of the supported "
            f"values, or use a bf16 checkpoint."
        )

    def _apply_shared_expert_sigmoid(
        self,
        hidden_states: torch.Tensor,
        shared_output: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if (
            shared_output is None
            or not self.use_shared_expert_sigmoid
            or not hasattr(self, "shared_expert_sigmoid")
        ):
            return shared_output
        gate, _ = self.shared_expert_sigmoid(hidden_states)
        return (shared_output.float() * torch.sigmoid(gate.float())).to(shared_output.dtype)

    def _forward_shared_experts(
        self, hidden_states, gemm_output_zero_allocator: BumpAllocator = None
    ):
        shared_output = super()._forward_shared_experts(hidden_states, gemm_output_zero_allocator)
        return self._apply_shared_expert_sigmoid(hidden_states, shared_output)

    def op_shared_experts(self, state):
        hidden_states_mlp_input = state.pop("hidden_states_mlp_input")
        if (self.num_fused_shared_experts == 0) and deepseek_v2.is_non_idle_and_non_empty(
            state.forward_batch.forward_mode, hidden_states_mlp_input
        ):
            shared_output = self.shared_experts(hidden_states_mlp_input)
            state.shared_output = self._apply_shared_expert_sigmoid(hidden_states_mlp_input, shared_output)
        else:
            state.shared_output = None

    def forward_cpu(
        self,
        hidden_states: torch.Tensor,
        should_allreduce_fusion: bool = False,
    ) -> torch.Tensor:
        if not self.use_shared_expert_sigmoid:
            return super().forward_cpu(hidden_states, should_allreduce_fusion)

        router_logits = self.gate(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )
        final_hidden_states *= self.routed_scaling_factor

        shared_output = self._forward_shared_experts(hidden_states)
        if shared_output is not None:
            final_hidden_states += shared_output
        if self.tp_size > 1 and not should_allreduce_fusion:
            final_hidden_states = deepseek_v2.tensor_model_parallel_all_reduce(final_hidden_states)
        return final_hidden_states


class Gigachat35AttentionGQA(Qwen3Attention):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        if getattr(config, "apply_qk_norm", False):
            raise ValueError(
                "apply_qk_norm is not supported with attention_type='LlamaAttention'"
            )
        if getattr(config, "llama_4_scaling", None) is not None:
            raise ValueError(
                "llama_4_scaling is not supported with attention_type='LlamaAttention'"
            )
        rope_parameters = getattr(config, "rope_parameters", None)
        if rope_parameters and "rope_theta" in rope_parameters:
            rope_theta = rope_parameters["rope_theta"]
            rope_scaling = rope_parameters
        else:
            rope_theta = config.rope_theta
            rope_scaling = getattr(config, "rope_scaling", None)
        super().__init__(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            quant_config=quant_config,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=config.attention_bias,
            prefix=prefix,
            alt_stream=alt_stream,
        )
        self.q_norm = None
        self.k_norm = None
        self.gated_attention = config.gated_attention
        if self.gated_attention:
            self.gate_proj = ColumnParallelLinear(
                config.hidden_size,
                self.total_num_heads * self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("gate_proj", prefix),
                tp_rank=get_parallel().attn_tp_rank,
                tp_size=get_parallel().attn_tp_size,
            )

    def forward_prepare_native(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        return q, k, v

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        q, k, v = self.forward_prepare_native(positions, hidden_states)
        attn_output = self.attn(q, k, v, forward_batch)
        if self.gated_attention:
            gate_values, _ = self.gate_proj(hidden_states)
            attn_output = _apply_sigmoid_gate_fp32(attn_output, gate_values)
        output, _ = self.o_proj(attn_output)
        return output


class Gigachat35AttentionMLA(deepseek_v2.DeepseekV2AttentionMLA):
    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: Optional[int],
        kv_lora_rank: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        layer_id: Optional[int] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        skip_rope: bool = False,
        is_nextn: bool = False,
        dsa_enable_prefill_cp: bool = False,
        mla_enable_prefill_cp: bool = False,
    ) -> None:
        super().__init__(
            config=config,
            hidden_size=hidden_size,
            num_heads=num_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            reduce_results=reduce_results,
            layer_id=layer_id,
            prefix=prefix,
            alt_stream=alt_stream,
            skip_rope=skip_rope,
            is_nextn=is_nextn,
            dsa_enable_prefill_cp=dsa_enable_prefill_cp,
            mla_enable_prefill_cp=mla_enable_prefill_cp,
        )
        # The MTP/NextN layer's DSA-ness follows the resolved MTP mode, not the
        # parent model; every other layer keeps its per-layer decision.
        if is_nextn:
            self.use_dsa = gigachat35_mtp_is_dsa(config)
        else:
            self.use_dsa = is_gigachat35_dsa_attention_layer(config, layer_id)
        if not self.use_dsa:
            if hasattr(self, "indexer"):
                del self.indexer
            self.skip_topk = None
            self.next_skip_topk = None
        self._norm_type = getattr(config, "norm_type", "LlamaRMSNorm")
        self._needs_custom_norm_prepare = self._norm_type != "LlamaRMSNorm"
        self.use_mla_scaling_factor = bool(
            getattr(config, "use_mla_scaling_factor", False)
        )
        self._needs_custom_prepare = (
            self._needs_custom_norm_prepare or self.use_mla_scaling_factor
        )
        if self._needs_custom_norm_prepare:
            if hasattr(self, "q_a_layernorm") and self.q_lora_rank is not None:
                self.q_a_layernorm = build_norm(config, self.q_lora_rank)
            self.kv_a_layernorm = build_norm(config, self.kv_lora_rank)
        q_hidden_dim = hidden_size if self.q_lora_rank is None else self.q_lora_rank
        kv_hidden_dim = hidden_size if self.kv_lora_rank is None else self.kv_lora_rank
        if self.use_mla_scaling_factor:
            self.alpha_q = (hidden_size / q_hidden_dim) ** 0.5
            self.alpha_kv = (hidden_size / kv_hidden_dim) ** 0.5
        else:
            self.alpha_q = 1.0
            self.alpha_kv = 1.0
        self.gated_attention = getattr(config, "gated_attention", False)
        if self.gated_attention:
            from sglang.srt.runtime_context import get_parallel

            attn_tp_rank = get_parallel().attn_tp_rank
            attn_tp_size = get_parallel().attn_tp_size
            self.gate_proj = ColumnParallelLinear(
                hidden_size,
                self.num_heads * self.v_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("gate_proj", prefix),
                tp_rank=attn_tp_rank,
                tp_size=attn_tp_size,
            )

    def _indexer_kwargs(self, config) -> dict:
        # GigaChat 3.5 indexer: rotate_activation off unless enabled in config;
        # key-norm is a plain RMSNorm (indexer_norm_type, default LlamaRMSNorm).
        indexer_norm_type = getattr(config, "indexer_norm_type", None) or "LlamaRMSNorm"
        if indexer_norm_type != "LlamaRMSNorm":
            raise ValueError(
                f"Unsupported indexer_norm_type {indexer_norm_type!r} for Gigachat35 "
                "DSA; llm-foundry only trains LlamaRMSNorm indexer keys"
            )
        return {
            "rotate_activation_enabled": bool(
                getattr(config, "indexer_rotate_activation_enabled", False)
            ),
            "k_norm": Gigachat35RMSNorm(
                deepseek_v2.get_dsa_index_head_dim(config),
                eps=getattr(config, "indexer_norm_eps", None) or 1e-6,
            ),
        }

    def _indexer_topk_or_reuse(
        self,
        hidden_states: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        prev_topk_indices: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        # skip_topk layers carry no indexer weights, so they reuse the carried
        # topk instead of running the indexer.
        if not self.skip_topk or (self.is_nextn and prev_topk_indices is None):
            return self.indexer(
                x=hidden_states,
                q_lora=q_lora,
                positions=positions,
                forward_batch=forward_batch,
                layer_id=self.layer_id,
            )
        return maybe_capture_indexer_topk(self.layer_id, prev_topk_indices)

    def _apply_q_a_layernorm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.q_a_layernorm(hidden_states)
        if self.alpha_q != 1.0:
            hidden_states = hidden_states * self.alpha_q
        return hidden_states

    def _apply_kv_a_layernorm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.kv_a_layernorm(hidden_states)
        if self.alpha_kv != 1.0:
            hidden_states = hidden_states * self.alpha_kv
        return hidden_states

    def _fuse_rope_for_trtllm_mla(self, forward_batch: ForwardBatch) -> bool:
        return self.use_dsa and super()._fuse_rope_for_trtllm_mla(forward_batch)

    def _skip_rope_for_dsa_tilelang_fused(self) -> bool:
        return self.use_dsa and super()._skip_rope_for_dsa_tilelang_fused()

    def _maybe_apply_gate(self, hidden_states, attn_output: torch.Tensor) -> torch.Tensor:
        if not self.gated_attention:
            return attn_output

        gate_values, _ = self.gate_proj(hidden_states)
        gate_values = torch.sigmoid(gate_values)
        if attn_output.ndim == 3:
            gate_values = gate_values.view(
                gate_values.shape[0], self.num_local_heads, self.v_head_dim
            )
        return attn_output * gate_values

    def dispatch_attn_forward_method(self, forward_batch: ForwardBatch):
        # Dense (indexer-removed) full-attn layers prefill as MHA, not absorbed
        # MLA (~20x slower on large tiles). Decode still uses absorbed MLA.
        if not self.use_dsa:
            from sglang.srt.models.deepseek_common.attention_backend_handler import (
                AttentionBackendRegistry,
                _support_mha_one_shot,
            )

            self.current_attention_backend = "fa3"
            method = AttentionBackendRegistry.get_handler("fa3")(self, forward_batch)
            # fp8 prefix hit must not use absorbed MLA (dequantizes the whole KV
            # buffer); prefer MHA.
            if (
                method == deepseek_v2.AttnForwardMethod.MLA
                and self.kv_cache_dtype == "fp8_e4m3"
                and forward_batch.forward_mode.is_extend_without_speculative()
                and not is_in_tc_piecewise_cuda_graph()
            ):
                if _support_mha_one_shot(self, forward_batch, "fa3"):
                    method = deepseek_v2.AttnForwardMethod.MHA_ONE_SHOT
                elif not getattr(self, "disable_chunked_prefix_cache", True):
                    method = deepseek_v2.AttnForwardMethod.MHA_CHUNKED_KV
            # MHA prefix paths need the dense backend to support chunked KV;
            # fall back to absorbed MLA when it cannot (enable_chunk_kv False).
            if (
                method
                in (
                    deepseek_v2.AttnForwardMethod.MHA_ONE_SHOT,
                    deepseek_v2.AttnForwardMethod.MHA_CHUNKED_KV,
                )
                and forward_batch.extend_prefix_lens_cpu is not None
                and any(forward_batch.extend_prefix_lens_cpu)
                and self._dense_layer_chunk_kv_disabled()
            ):
                method = deepseek_v2.AttnForwardMethod.MLA
            return method
        attn_forward_method = super().dispatch_attn_forward_method(forward_batch)
        if (
            attn_forward_method == deepseek_v2.AttnForwardMethod.MLA_FUSED_ROPE_CPU
            and self._norm_type in {"GatedNorm", "ZeroCenteredGatedNorm"}
        ):
            raise AssertionError(
                f"{self._norm_type} is not supported with MLA_FUSED_ROPE_CPU: "
                "the Intel AMX fused MLA path skips q/kv norm gate projections."
            )
        if (
            self._needs_custom_prepare
            and attn_forward_method == deepseek_v2.AttnForwardMethod.MLA_FUSED_ROPE_ROCM
        ):
            return deepseek_v2.AttnForwardMethod.MLA
        return attn_forward_method

    def _dense_layer_chunk_kv_disabled(self) -> bool:
        """True iff this dense layer's real backend has enable_chunk_kv False.
        False when the backend has no such attribute (native-MHA fa3)."""
        backend = get_attn_backend()
        backend = getattr(backend, "primary", backend)  # unwrap TBO
        for getter in ("_full_attn_backend_for", "_backend_for"):
            resolve = getattr(backend, getter, None)
            if resolve is not None:
                backend = resolve(None, layer_id=self.attn_mha.layer_id)
                break
        return getattr(backend, "enable_chunk_kv", None) is False

    def _get_mla_kv_buffer(
        self,
        kv_indices: torch.Tensor,
        dst_dtype: torch.dtype,
        forward_batch: ForwardBatch,
    ):
        # fp8 KV stores the latent in DSA paged-FP8 format (dim_quant=656), which
        # the base triton fetch can't read; dequantize it here for MHA prefill.
        if self.use_dsa and self.kv_cache_dtype == "fp8_e4m3":
            kv_cache_fp8 = get_token_to_kv_pool().get_key_buffer(self.attn_mha.layer_id)
            kv_latent_bf16 = dequantize_k_cache_paged(kv_cache_fp8, kv_indices)
            kv_a = kv_latent_bf16[:, :, : self.kv_lora_rank].squeeze(1).contiguous()
            k_pe = kv_latent_bf16[:, :, self.kv_lora_rank :]
            return kv_a, k_pe
        return super()._get_mla_kv_buffer(kv_indices, dst_dtype, forward_batch)

    def forward_normal_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        if not self._needs_custom_prepare:
            inner_state = super().forward_normal_prepare(
                positions, hidden_states, forward_batch, zero_allocator
            )
            return (*inner_state, hidden_states)

        if self.q_lora_rank is not None:
            q, latent_cache = (
                deepseek_v2.get_attn_tp_context()
                .fetch_qkv_latent()
                .split(
                    [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                    dim=-1,
                )
            )

            if self.use_dsa:
                q_lora = self._apply_q_a_layernorm(q)
                q = self.q_b_proj(q_lora)[0].view(
                    -1, self.num_local_heads, self.qk_head_dim
                )
                _ = self.indexer(
                    x=hidden_states,
                    q_lora=q_lora,
                    positions=positions,
                    forward_batch=forward_batch,
                    layer_id=self.layer_id,
                    return_indices=False,
                )
            else:
                q = self._apply_q_a_layernorm(q)
                q = self.q_b_proj(q)[0].view(
                    -1, self.num_local_heads, self.qk_head_dim
                )
        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
            if self.use_dsa:
                _ = self.indexer(
                    x=hidden_states,
                    q_lora=hidden_states,
                    positions=positions,
                    forward_batch=forward_batch,
                    layer_id=self.layer_id,
                    return_indices=False,
                )

        _, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        kv_a, _ = latent_cache.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        latent_cache = latent_cache.unsqueeze(1)
        kv_a = self._apply_kv_a_layernorm(kv_a)

        k_pe = latent_cache[:, :, self.kv_lora_rank :]
        if self.rotary_emb is not None:
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        q[..., self.qk_nope_head_dim :] = q_pe

        self._set_mla_kv_buffer(latent_cache, kv_a, k_pe, forward_batch)
        if forward_batch.mha_one_shot and sum(forward_batch.extend_prefix_lens_cpu) != 0:
            if (
                self.use_dsa
                and self.kv_cache_dtype == "fp8_e4m3"
                and (
                    not deepseek_v2.get_global_server_args().dsa_decode_backend
                    == "trtllm"
                    or not deepseek_v2.get_global_server_args().dsa_prefill_backend
                    == "trtllm"
                )
            ):
                kv_a, k_pe = self._get_mla_kv_buffer_from_fp8_for_dsa(forward_batch)
            else:
                kv_a, k_pe = self._get_mla_kv_buffer(
                    forward_batch.fetch_mha_one_shot_kv_indices(),
                    q.dtype,
                    forward_batch,
                )

        kv = self.kv_b_proj(kv_a)[0]
        kv = kv.view(
            -1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope = kv[..., : self.qk_nope_head_dim]
        v = kv[..., self.qk_nope_head_dim :]
        k = self._concat_and_cast_mha_k(k_nope, k_pe, forward_batch)
        inner_state = (q, k, v, forward_batch)
        return (*inner_state, hidden_states)

    def forward_normal_core(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        forward_batch: ForwardBatch,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        attn_output = self.attn_mha(q, k, v, forward_batch, save_kv_cache=False)
        attn_output = attn_output.reshape(-1, self.num_local_heads * self.v_head_dim)
        attn_output = self._maybe_apply_gate(hidden_states, attn_output)
        output, _ = self.o_proj(attn_output)
        return output

    def forward_normal_chunked_kv_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        return self.forward_normal_prepare(
            positions, hidden_states, forward_batch, zero_allocator
        )

    def forward_normal_chunked_kv_core(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        forward_batch: ForwardBatch,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        has_extend_prefix = forward_batch.extend_prefix_lens_cpu is not None and any(
            forward_batch.extend_prefix_lens_cpu
        )
        if has_extend_prefix and forward_batch.num_prefix_chunks is None:
            forward_batch.prepare_chunked_prefix_cache_info(q.device)
            if hasattr(get_attn_backend(), "init_mha_chunk_metadata"):
                get_attn_backend().init_mha_chunk_metadata(forward_batch)

        forward_batch.mha_return_lse = has_extend_prefix
        forward_batch.set_attn_attend_prefix_cache(False)
        attn_output = self.attn_mha(q, k, v, forward_batch, save_kv_cache=False)

        # Backends that implement neither `mha_return_lse` nor
        # `attn_attend_prefix_cache` (Triton) return a bare tensor from a plain
        # extend over prefix+new KV, so their result is already complete and there
        # is no per-chunk lse to merge.
        if has_extend_prefix and isinstance(attn_output, tuple):
            attn_output, lse = attn_output
            forward_batch.set_attn_attend_prefix_cache(True)
            attn_output = self._chunked_prefix_attn_mha(
                q=q,
                accum_output=attn_output,
                accum_lse=lse,
                forward_batch=forward_batch,
            )

        attn_output = attn_output.reshape(-1, self.num_local_heads * self.v_head_dim)
        attn_output = self._maybe_apply_gate(hidden_states, attn_output)
        output, _ = self.o_proj(attn_output)
        return output

    def forward_normal_one_shot_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        forward_batch.mha_one_shot = True
        return self.forward_normal_prepare(
            positions, hidden_states, forward_batch, zero_allocator
        )

    def forward_normal_one_shot_core(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        forward_batch: ForwardBatch,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        has_extend_prefix = any(forward_batch.extend_prefix_lens_cpu)
        if has_extend_prefix and forward_batch.num_prefix_chunks is None:
            forward_batch.num_prefix_chunks = 0
            if hasattr(get_attn_backend(), "init_mha_chunk_metadata"):
                get_attn_backend().init_mha_chunk_metadata(forward_batch)
        forward_batch.mha_return_lse = False
        forward_batch.set_attn_attend_prefix_cache(False)
        return self.forward_normal_core(q, k, v, forward_batch, hidden_states)

    def forward_absorb_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
        llama_4_scaling: Optional[torch.Tensor] = None,
        prev_topk_indices: Optional[torch.Tensor] = None,
    ):
        if not self._needs_custom_prepare:
            inner_state = super().forward_absorb_prepare(
                positions,
                hidden_states,
                forward_batch,
                zero_allocator,
                llama_4_scaling,
                prev_topk_indices,
            )
            return (*inner_state, hidden_states)

        from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode

        q_lora = None
        topk_indices = None
        if self.q_lora_rank is not None:
            q, latent_cache = (
                deepseek_v2.get_attn_tp_context()
                .fetch_qkv_latent()
                .split(
                    [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                    dim=-1,
                )
            )
            k_nope = latent_cache[..., : self.kv_lora_rank]

            if self.alt_stream is not None and get_is_capture_mode():
                current_stream = torch.cuda.current_stream()
                self.alt_stream.wait_stream(current_stream)
                q = self._apply_q_a_layernorm(q)
                with torch.cuda.stream(self.alt_stream):
                    k_nope = self._apply_kv_a_layernorm(k_nope)
                current_stream.wait_stream(self.alt_stream)
            else:
                q = self._apply_q_a_layernorm(q)
                k_nope = self._apply_kv_a_layernorm(k_nope)

            if self.use_dsa:
                q_lora = q

            if (
                self.alt_stream is not None
                and get_is_capture_mode()
                and forward_batch.forward_mode.is_decode_or_idle()
                and q_lora is not None
            ):
                current_stream = torch.cuda.current_stream()
                self.alt_stream.wait_stream(current_stream)
                with torch.cuda.stream(self.alt_stream):
                    k_nope = k_nope.unsqueeze(1)
                    q = self.q_b_proj(q)[0].view(
                        -1, self.num_local_heads, self.qk_head_dim
                    )
                topk_indices = self._indexer_topk_or_reuse(
                    hidden_states, q_lora, positions, forward_batch, prev_topk_indices
                )
                current_stream.wait_stream(self.alt_stream)
            else:
                k_nope = k_nope.unsqueeze(1)
                q = self.q_b_proj(q)[0].view(
                    -1, self.num_local_heads, self.qk_head_dim
                )
                if q_lora is not None:
                    topk_indices = self._indexer_topk_or_reuse(
                        hidden_states, q_lora, positions, forward_batch, prev_topk_indices
                    )
        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
            k_nope = latent_cache[..., : self.kv_lora_rank]
            k_nope = self._apply_kv_a_layernorm(k_nope).unsqueeze(1)
            if self.use_dsa:
                topk_indices = self._indexer_topk_or_reuse(
                    hidden_states, hidden_states, positions, forward_batch, prev_topk_indices
                )

        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        k_pe = latent_cache[..., self.kv_lora_rank :].unsqueeze(1)

        if self.use_deep_gemm_bmm:
            q_nope_val, q_nope_scale, masked_m, expected_m, aligned_m = (
                _fwd_mla.per_token_group_quant_mla_deep_gemm_masked_fp8(
                    q_nope.transpose(0, 1)
                )
            )
            q_nope_out = q_nope.new_empty(
                (self.num_local_heads, aligned_m, self.kv_lora_rank)
            )
            deepseek_v2.deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
                (q_nope_val, q_nope_scale),
                (self.w_kc, self.w_scale_k),
                q_nope_out,
                masked_m,
                expected_m,
            )
            q_nope_out = q_nope_out[:, :expected_m, :]
        elif deepseek_v2._is_hip:
            if deepseek_v2._use_aiter_gfx95 and self.w_kc.dtype == torch.uint8:
                x = q_nope.transpose(0, 1)
                q_nope_out = torch.empty(
                    x.shape[0],
                    x.shape[1],
                    self.w_kc.shape[2],
                    device=x.device,
                    dtype=torch.bfloat16,
                )
                deepseek_v2.batched_gemm_afp4wfp4_pre_quant(
                    x,
                    self.w_kc.transpose(-2, -1),
                    self.w_scale_k.transpose(-2, -1),
                    torch.bfloat16,
                    q_nope_out,
                )
            else:
                if (
                    deepseek_v2._use_aiter_gfx95
                    and self.w_kc.dtype == torch.float8_e4m3fn
                ):
                    q_nope_out = deepseek_v2.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
                        X=q_nope,
                        WQ=self.w_kc.transpose(-1, -2),
                        w_scale=self.w_scale,
                        group_size=128,
                        YQ=None,
                        transpose_bm=False,
                        transpose_bm_in=True,
                        dtype=torch.bfloat16,
                    )
                else:
                    q_nope_out = torch.bmm(
                        q_nope.to(torch.bfloat16).transpose(0, 1),
                        self.w_kc.to(torch.bfloat16) * self.w_scale,
                    )
        elif self.w_kc.dtype == torch.float8_e4m3fn:
            q_nope_val, q_nope_scale = _fwd_mla.per_tensor_quant_mla_fp8(
                q_nope.transpose(0, 1),
                (
                    torch.zeros((1,), dtype=torch.float32, device=q_nope.device)
                    if _fwd_mla._is_cublas_ge_129
                    else zero_allocator.allocate(1)
                ),
            )
            q_nope_out = _fwd_mla.bmm_fp8(
                q_nope_val,
                self.w_kc,
                q_nope_scale,
                self.w_scale,
                torch.bfloat16,
            )
        else:
            q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)

        q_nope_out = q_nope_out.transpose(0, 1)

        if (
            self.rotary_emb is not None
            and (not self._fuse_rope_for_trtllm_mla(forward_batch))
            and (
                not deepseek_v2._use_aiter
                or not deepseek_v2._is_gfx95_supported
                or self.use_dsa
            )
        ):
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)

        if dsa_use_prefill_cp(forward_batch):
            k_nope, k_pe = self.rebuild_cp_kv_cache(
                latent_cache, forward_batch, k_nope, k_pe
            )

        inner_state = (
            q_pe,
            k_pe,
            q_nope_out,
            k_nope,
            forward_batch,
            zero_allocator,
            positions,
            topk_indices,
            llama_4_scaling,
        )
        return (*inner_state, hidden_states)

    def forward_absorb_core(
        self,
        q_pe,
        k_pe,
        q_nope_out,
        k_nope,
        forward_batch,
        zero_allocator,
        positions,
        topk_indices,
        llama_4_scaling,
        hidden_states,
    ):
        save_kv_cache = True

        if (
            self.current_attention_backend
            in _fwd_mla.FORWARD_ABSORB_CORE_ATTENTION_BACKENDS
        ):
            extra_args = {}
            if self._fuse_rope_for_trtllm_mla(forward_batch):
                extra_args = {
                    "cos_sin_cache": self.rotary_emb.cos_sin_cache,
                    "is_neox": self.rotary_emb.is_neox_style,
                    "llama_4_scaling": llama_4_scaling,
                }

            attn_output = self.attn_mqa(
                q_nope_out,
                k_nope,
                k_nope,
                forward_batch,
                q_rope=q_pe,
                k_rope=k_pe,
                **extra_args,
                **(dict(topk_indices=topk_indices) if topk_indices is not None else {}),
            )
        else:
            if deepseek_v2._use_aiter_gfx95:
                cos = self.rotary_emb.cos_cache
                sin = self.rotary_emb.sin_cache

                kv_cache_dtype = (
                    _fwd_mla.fp8_dtype
                    if self.kv_cache_dtype == "fp8_e4m3"
                    else q_nope_out.dtype
                )

                q, _, _, k = deepseek_v2.fused_qk_rope_cat_and_cache_mla(
                    q_nope_out,
                    q_pe,
                    k_nope,
                    k_pe,
                    forward_batch.token_to_kv_pool.get_key_buffer(
                        self.attn_mqa.layer_id
                    ),
                    forward_batch.out_cache_loc,
                    positions,
                    cos,
                    sin,
                    self.attn_mqa.k_scale,
                    self.rotary_emb.is_neox_style,
                    q_out_dtype=kv_cache_dtype,
                )

                save_kv_cache = False
            else:
                q = torch.cat([q_nope_out, q_pe], dim=-1)
                k = torch.cat([k_nope, k_pe], dim=-1)

            if llama_4_scaling is not None:
                q *= llama_4_scaling

            attn_output = self.attn_mqa(
                q,
                k,
                k_nope,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **(dict(topk_indices=topk_indices) if topk_indices is not None else {}),
            )
        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

        if self.use_deep_gemm_bmm:
            attn_output_val, attn_output_scale, masked_m, expected_m, aligned_m = (
                _fwd_mla.per_token_group_quant_mla_deep_gemm_masked_fp8(
                    attn_output.transpose(0, 1)
                )
            )
            attn_bmm_output = attn_output.new_empty(
                (self.num_local_heads, aligned_m, self.v_head_dim)
            )
            deepseek_v2.deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
                (attn_output_val, attn_output_scale),
                (self.w_vc, self.w_scale_v),
                attn_bmm_output,
                masked_m,
                expected_m,
            )
            attn_bmm_output = attn_bmm_output[:, :expected_m, :].transpose(0, 1)
            attn_bmm_output = attn_bmm_output.flatten(1, 2)
            attn_bmm_output = self._maybe_apply_gate(hidden_states, attn_bmm_output)
        elif deepseek_v2._is_hip:
            if deepseek_v2._use_aiter_gfx95 and self.w_vc.dtype == torch.uint8:
                x = attn_output.transpose(0, 1)
                attn_bmm_output = torch.empty(
                    x.shape[0],
                    x.shape[1],
                    self.w_vc.shape[2],
                    device=x.device,
                    dtype=torch.bfloat16,
                )
                deepseek_v2.batched_gemm_afp4wfp4_pre_quant(
                    x,
                    self.w_vc.transpose(-2, -1),
                    self.w_scale_v.transpose(-2, -1),
                    torch.bfloat16,
                    attn_bmm_output,
                )
            else:
                if deepseek_v2._use_aiter_gfx95 and self.w_kc.dtype == torch.float8_e4m3fn:
                    attn_bmm_output = deepseek_v2.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
                        X=attn_output,
                        WQ=self.w_vc.transpose(-1, -2),
                        w_scale=self.w_scale,
                        group_size=128,
                        YQ=None,
                        transpose_bm=False,
                        transpose_bm_in=True,
                        dtype=torch.bfloat16,
                    )
                else:
                    attn_bmm_output = torch.bmm(
                        attn_output.to(torch.bfloat16).transpose(0, 1),
                        self.w_vc.to(torch.bfloat16) * self.w_scale,
                    )

            attn_bmm_output = attn_bmm_output.transpose(0, 1)
            attn_bmm_output = self._maybe_apply_gate(hidden_states, attn_bmm_output)
            if self.o_proj.weight.dtype == torch.uint8:
                attn_bmm_output = deepseek_v2.fused_flatten_mxfp4_quant(attn_bmm_output)
            elif self.o_proj.weight.dtype == torch.float8_e4m3fn:
                attn_bmm_output = deepseek_v2.fused_flatten_fp8_group_quant(
                    attn_bmm_output,
                    group_size=128,
                    dtype_quant=torch.float8_e4m3fn,
                )
            else:
                attn_bmm_output = attn_bmm_output.flatten(1, 2)
        elif self.w_vc.dtype == torch.float8_e4m3fn:
            attn_output_val, attn_output_scale = _fwd_mla.per_tensor_quant_mla_fp8(
                attn_output.transpose(0, 1),
                (
                    torch.zeros((1,), dtype=torch.float32, device=attn_output.device)
                    if _fwd_mla._is_cublas_ge_129
                    else zero_allocator.allocate(1)
                ),
            )
            attn_bmm_output = _fwd_mla.bmm_fp8(
                attn_output_val,
                self.w_vc,
                attn_output_scale,
                self.w_scale,
                torch.bfloat16,
            )
            attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
            attn_bmm_output = self._maybe_apply_gate(hidden_states, attn_bmm_output)
        else:
            if is_in_tc_piecewise_cuda_graph():
                attn_bmm_output = (
                    torch.bmm(attn_output.transpose(0, 1), self.w_vc)
                    .transpose(0, 1)
                    .flatten(1, 2)
                )
            else:
                attn_bmm_output = torch.empty(
                    (attn_output.shape[0], self.num_local_heads * self.v_head_dim),
                    dtype=attn_output.dtype,
                    device=attn_output.device,
                )
                torch.bmm(
                    attn_output.transpose(0, 1),
                    self.w_vc,
                    out=attn_bmm_output.view(
                        -1, self.num_local_heads, self.v_head_dim
                    ).transpose(0, 1),
                )
            attn_bmm_output = self._maybe_apply_gate(hidden_states, attn_bmm_output)
        output, _ = self.o_proj(attn_bmm_output)

        return output

    def forward_absorb_fused_mla_rope_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        inner_state = super().forward_absorb_fused_mla_rope_prepare(
            positions, hidden_states, forward_batch, zero_allocator
        )
        return (*inner_state, hidden_states)

    def forward_absorb_fused_mla_rope_core(
        self,
        q_input,
        key_cache_buf,
        val_cache_buf,
        attn_output,
        kv_indptr,
        kv_indices,
        k_pe_output,
        cos_sin_cache,
        positions,
        attn_logits,
        num_kv_split,
        sm_scale,
        enable_rope_fusion,
        k_input,
        forward_batch,
        zero_allocator,
        hidden_states,
    ):
        from sglang.srt.layers.attention.triton_ops.rocm_mla_decode_rope import (
            decode_attention_fwd_grouped_rope,
        )

        decode_attention_fwd_grouped_rope(
            q_input,
            key_cache_buf,
            val_cache_buf,
            attn_output,
            kv_indptr,
            kv_indices,
            k_pe_output,
            self.kv_lora_rank,
            self.rotary_emb.rotary_dim,
            cos_sin_cache,
            positions,
            attn_logits,
            num_kv_split,
            sm_scale,
            logit_cap=self.attn_mqa.logit_cap,
            use_rope=enable_rope_fusion,
            is_neox_style=self.rotary_emb.is_neox_style,
        )

        if enable_rope_fusion:
            k_input[..., self.kv_lora_rank :] = k_pe_output
            forward_batch.token_to_kv_pool.set_kv_buffer(
                self.attn_mqa, forward_batch.out_cache_loc, k_input, None
            )

        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

        if deepseek_v2._is_hip:
            attn_bmm_output = torch.bmm(
                attn_output.to(torch.bfloat16).transpose(0, 1),
                self.w_vc.to(torch.bfloat16) * self.w_scale,
            )
        elif self.w_vc.dtype == torch.float8_e4m3fn:
            attn_output_val, attn_output_scale = _fwd_mla.per_tensor_quant_mla_fp8(
                attn_output.transpose(0, 1),
                zero_allocator.allocate(1),
                dtype=torch.float8_e4m3fn,
            )
            attn_bmm_output = _fwd_mla.bmm_fp8(
                attn_output_val,
                self.w_vc,
                attn_output_scale,
                self.w_scale,
                torch.bfloat16,
            )
        else:
            attn_bmm_output = torch.bmm(attn_output.transpose(0, 1), self.w_vc)
        attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
        attn_bmm_output = self._maybe_apply_gate(hidden_states, attn_bmm_output)
        output, _ = self.o_proj(attn_bmm_output)

        return output

    def forward_absorb_fused_mla_rope_cpu_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        inner_state = super().forward_absorb_fused_mla_rope_cpu_prepare(
            positions, hidden_states, forward_batch, zero_allocator
        )
        return (*inner_state, hidden_states)

    def forward_absorb_fused_mla_rope_cpu_core(
        self,
        q_input,
        k_input,
        v_input,
        forward_batch,
        zero_allocator,
        hidden_states,
    ):
        assert self.q_lora_rank is not None and deepseek_v2.use_intel_amx_backend(
            self
        ), "forward_absorb_fused_mla_rope_cpu_core requires q_lora_rank is not None and use_intel_amx_backend"

        attn_output = self.attn_mqa(q_input, k_input, v_input, forward_batch)
        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

        batch_size = self.w_vc.size(0)
        output_size = self.w_vc.size(1)
        num_tokens = attn_output.size(0)
        output = torch.empty([num_tokens, int(batch_size * output_size)], dtype=attn_output.dtype)
        attn_bmm_output = output.view([num_tokens, batch_size, output_size]).transpose_(0, 1)
        torch.ops.sgl_kernel.bmm_cpu(
            attn_bmm_output,
            attn_output.transpose(0, 1),
            self.w_vc,
            True,
            None,
        )
        output = self._maybe_apply_gate(hidden_states, output)
        output, _ = self.o_proj(output)

        return output


class Gigachat35DecoderLayer(deepseek_v2.DeepseekV2DecoderLayer):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        moe_quant_config_override: Optional[QuantizationConfig] = None,
        is_nextn: bool = False,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        dsa_enable_prefill_cp: bool = False,
        mla_enable_prefill_cp: bool = False,
    ) -> None:
        super().__init__(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            moe_quant_config_override=moe_quant_config_override,
            is_nextn=is_nextn,
            prefix=prefix,
            alt_stream=alt_stream,
            dsa_enable_prefill_cp=dsa_enable_prefill_cp,
            mla_enable_prefill_cp=mla_enable_prefill_cp,
        )
        self.use_linear_attn = (
            gigachat35_mtp_uses_linear_attention(config)
            if is_nextn
            else is_gigachat35_linear_attention_layer(config, layer_id)
        )
        self.use_gqa_attn = (
            not self.use_linear_attn
            and getattr(config, "attention_type", None) == "LlamaAttention"
        )
        if self.use_linear_attn:
            linear_attention_type = getattr(
                config, "linear_attention_type", "Qwen3NextGatedDeltaNet"
            )
            if linear_attention_type == "Qwen3NextGatedDeltaNet":
                self.self_attn = Gigachat35GatedDeltaNet(
                    config=config,
                    layer_id=layer_id,
                    quant_config=quant_config,
                    alt_stream=alt_stream,
                    prefix=add_prefix("self_attn", prefix),
                )
            elif linear_attention_type == "KimiDeltaAttention":
                _prepare_kimi_delta_attention_config(config)
                self.self_attn = Gigachat35KimiDeltaAttention(
                    layer_idx=layer_id,
                    hidden_size=config.hidden_size,
                    config=config,
                    quant_config=quant_config,
                    rms_norm_eps=config.rms_norm_eps,
                    prefix=add_prefix("self_attn", prefix),
                )
            else:
                raise ValueError(
                    f"Unsupported linear_attention_type for Gigachat35: "
                    f"{linear_attention_type!r}"
                )
            self.layer_communicator.qkv_latent_func = None
        elif self.use_gqa_attn:
            self.self_attn = Gigachat35AttentionGQA(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                alt_stream=alt_stream,
                prefix=add_prefix("self_attn", prefix),
            )
            self.layer_communicator.qkv_latent_func = None

        norm_class = resolve_norm_class(config)
        layernorm_type = getattr(config, "layernorm_type", "pre")
        self._use_pre_layernorm = layernorm_type in {"pre", "pre_post"}
        self._use_post_layernorm = layernorm_type in {"post", "pre_post"}

        if self._use_pre_layernorm:
            self.input_layernorm = build_norm(config, config.hidden_size)
            self.post_attention_layernorm = build_norm(config, config.hidden_size)

        self._attn_prepare_layernorm = (
            self.input_layernorm if self._use_pre_layernorm else Gigachat35PassthroughNorm()
        )

        self.post_self_attn_layernorm = None
        self.post_feedforward_layernorm = None
        if self._use_post_layernorm:
            self.post_self_attn_layernorm = build_norm(config, config.hidden_size)
            self.post_feedforward_layernorm = build_norm(config, config.hidden_size)

        self._mlp_prepare_layernorm = Gigachat35MlpPrepNorm(
            pre_layernorm=self.post_attention_layernorm if self._use_pre_layernorm else None,
            post_layernorm=self.post_self_attn_layernorm,
        )
        if self._use_post_layernorm:
            # post_self_attn_layernorm is nonlinear, so under dp-attention it must
            # see the fully reduced attn output before the residual add — the
            # partial-sum trick in _gather_hidden_states_and_residual is invalid.
            self.layer_communicator.force_layernorm_before_dp_gather = True
            self.layer_communicator._context.force_layernorm_before_dp_gather = True

        # Thread DeepSeek-style hard SwiGLU activation clipping into the dense MLP
        # (when this is a dense layer). Routed and shared experts in MoE layers are
        # patched in Gigachat35MoE.__init__ (shared via `swiglu_limit` attr, routed and
        # fused-shared via FusedMoE.moe_runner_config.gemm1_clamp_limit).
        swiglu_limit = float(getattr(config, "swiglu_limit", 0.0))
        if swiglu_limit > 0 and isinstance(self.mlp, deepseek_v2.DeepseekV2MLP):
            self.mlp.swiglu_limit = swiglu_limit

    def _get_quant_format(self) -> str:
        if (
            deepseek_v2._is_gfx95_supported
            and getattr(self.self_attn, "fused_qkv_a_proj_with_mqa", None) is not None
            and getattr(self.self_attn.fused_qkv_a_proj_with_mqa, "weight", None) is not None
            and self.self_attn.fused_qkv_a_proj_with_mqa.weight.dtype == torch.uint8
        ):
            return "mxfp4"
        if (
            deepseek_v2._is_gfx95_supported
            and getattr(self.self_attn, "fused_qkv_a_proj_with_mqa", None) is not None
            and getattr(self.self_attn.fused_qkv_a_proj_with_mqa, "weight", None) is not None
            and self.self_attn.fused_qkv_a_proj_with_mqa.weight.dtype
            == getattr(torch, "float8_e4m3fn", None)
        ):
            return "fp8"
        return ""

    def _prepare_attn(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        quant_format: str,
    ):
        original = self.layer_communicator.input_layernorm
        self.layer_communicator.input_layernorm = self._attn_prepare_layernorm
        try:
            return self.layer_communicator.prepare_attn(
                hidden_states,
                residual,
                forward_batch,
                quant_format,
            )
        finally:
            self.layer_communicator.input_layernorm = original

    def _prepare_mlp(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
    ):
        original = self.layer_communicator.post_attention_layernorm
        self.layer_communicator.post_attention_layernorm = self._mlp_prepare_layernorm
        try:
            return self.layer_communicator.prepare_mlp(
                hidden_states,
                residual,
                forward_batch,
            )
        finally:
            self.layer_communicator.post_attention_layernorm = original

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        zero_allocator: BumpAllocator,
        gemm_output_zero_allocator: BumpAllocator = None,
        llama_4_scaling: Optional[torch.Tensor] = None,
        prev_topk_indices: Optional[torch.Tensor] = None,
        captured_last_layer_outputs: Optional[list] = None,
    ) -> torch.Tensor:
        from sglang.srt.layers.communicator import ScatterMode
        from sglang.srt.layers.dp_attention import (
            get_attention_tp_rank,
            get_attention_tp_size,
            is_dp_attention_enabled,
        )
        from sglang.srt.models.nemotron_h_utils import get_real_num_tokens

        if is_dp_attention_enabled():
            _rows = hidden_states.shape[0]
            _shards, _rank = 1, 0
            if self.layer_scatter_modes.layer_input_mode is ScatterMode.SCATTERED:
                _shards, _rank = get_attention_tp_size(), get_attention_tp_rank()
            _real_tokens = get_real_num_tokens(
                hidden_states, forward_batch, _rows * _shards
            )
            _real_tokens = min(max(_real_tokens - _rows * _rank, 0), _rows)
            if _real_tokens < _rows:
                hidden_states[_real_tokens:].zero_()
                if residual is not None:
                    residual[_real_tokens:].zero_()

        quant_format = self._get_quant_format() if self._use_pre_layernorm else ""
        hidden_states, residual = self._prepare_attn(
            hidden_states,
            residual,
            forward_batch,
            quant_format,
        )

        if self.use_linear_attn:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
        elif self.use_gqa_attn:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
        else:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                zero_allocator=zero_allocator,
                llama_4_scaling=llama_4_scaling,
            )

        hidden_states, residual = self._prepare_mlp(
            hidden_states,
            residual,
            forward_batch,
        )

        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )
        if isinstance(self.mlp, deepseek_v2.DeepseekV2MLP):
            gemm_output_zero_allocator = None

        hidden_states = self.mlp(
            hidden_states,
            forward_batch,
            should_allreduce_fusion=False,
            use_reduce_scatter=use_reduce_scatter,
            gemm_output_zero_allocator=gemm_output_zero_allocator,
        )

        # post_feedforward_layernorm is nonlinear and must see the SUMMED mlp
        # output; with use_reduce_scatter the summation of the tp-partial outputs
        # happens inside postprocess_layer (reduce-scatter), so the norm moves
        # after it.
        if self.post_feedforward_layernorm is not None and not use_reduce_scatter:
            hidden_states = self.post_feedforward_layernorm(hidden_states)

        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states,
            residual,
            forward_batch,
        )

        if self.post_feedforward_layernorm is not None and use_reduce_scatter:
            hidden_states = self.post_feedforward_layernorm(hidden_states)
        return hidden_states, residual, None

    def op_comm_prepare_attn(
        self,
        state,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        zero_allocator: BumpAllocator,
        tbo_subbatch_index: Optional[int] = None,
    ):
        quant_format = self._get_quant_format() if self._use_pre_layernorm else ""
        state.hidden_states_after_comm_pre_attn, state.residual_after_input_ln = (
            self._prepare_attn(
                hidden_states,
                residual,
                forward_batch,
                quant_format,
            )
        )
        state.update(
            dict(
                forward_batch=forward_batch,
                positions=positions,
                zero_allocator=zero_allocator,
                tbo_subbatch_index=tbo_subbatch_index,
            )
        )

    def op_comm_prepare_mlp(self, state):
        state.hidden_states_mlp_input, state.residual_after_comm_pre_mlp = (
            self._prepare_mlp(
                state.pop("hidden_states_after_attn"),
                state.pop("residual_after_input_ln"),
                state.forward_batch,
            )
        )

    def op_mlp(self, state):
        hidden_states = state.pop("hidden_states_mlp_input")
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            state.forward_batch
        )
        if not (
            deepseek_v2.enable_moe_dense_fully_dp()
            and (not self.is_layer_sparse)
            and hidden_states.shape[0] == 0
        ):
            state.hidden_states_mlp_output = self.mlp(
                hidden_states,
                state.forward_batch,
                should_allreduce_fusion=False,
                use_reduce_scatter=use_reduce_scatter,
            )
        else:
            state.hidden_states_mlp_output = hidden_states

        # Same constraint as forward(): with use_reduce_scatter the nonlinear
        # post_feedforward_layernorm moves after postprocess_layer.
        if self.post_feedforward_layernorm is not None and not use_reduce_scatter:
            state.hidden_states_mlp_output = self.post_feedforward_layernorm(
                state.hidden_states_mlp_output
            )

    def op_comm_postprocess_layer(self, state):
        hidden_states, residual = self.layer_communicator.postprocess_layer(
            state.pop("hidden_states_mlp_output"),
            state.pop("residual_after_comm_pre_mlp"),
            state.forward_batch,
        )
        if self.post_feedforward_layernorm is not None and (
            self.layer_communicator.should_use_reduce_scatter(state.forward_batch)
        ):
            hidden_states = self.post_feedforward_layernorm(hidden_states)

        output = dict(
            positions=state.positions,
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=state.forward_batch,
            zero_allocator=state.zero_allocator,
            tbo_subbatch_index=state.tbo_subbatch_index,
        )

        state.clear(
            expect_keys={
                "positions",
                "forward_batch",
                "zero_allocator",
                "tbo_subbatch_index",
            }
        )
        return output


class Gigachat35ForCausalLM(
    Gigachat35HybridGDNWeightLoaderMixin,
    deepseek_v2.DeepseekV3ForCausalLM,
):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        norm_class = resolve_norm_class(config)
        original_attention_cls = deepseek_v2.DeepseekV2AttentionMLA
        original_decoder_layer_cls = deepseek_v2.DeepseekV2DecoderLayer
        original_moe_cls = deepseek_v2.DeepseekV2MoE
        original_rmsnorm_cls = deepseek_v2.RMSNorm
        deepseek_v2.DeepseekV2AttentionMLA = Gigachat35AttentionMLA
        deepseek_v2.DeepseekV2DecoderLayer = Gigachat35DecoderLayer
        deepseek_v2.DeepseekV2MoE = Gigachat35MoE
        if norm_class is not RMSNorm:
            deepseek_v2.RMSNorm = norm_class
        try:
            super().__init__(config, quant_config=quant_config, prefix=prefix)
            self.model.norm = build_norm(config, config.hidden_size)
        finally:
            deepseek_v2.DeepseekV2AttentionMLA = original_attention_cls
            deepseek_v2.DeepseekV2DecoderLayer = original_decoder_layer_cls
            deepseek_v2.DeepseekV2MoE = original_moe_cls
            deepseek_v2.RMSNorm = original_rmsnorm_cls

    def determine_num_fused_shared_experts(
        self, architecture: str = "Gigachat35ForCausalLM"
    ):
        return super().determine_num_fused_shared_experts(architecture=architecture)

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        if not config.n_routed_experts:
            return None
        return super().get_model_config_for_expert_location(config)


class Gigachat35ModelNextN(DeepseekModelNextN):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        draft_model_idx: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        # Module reference needed to monkey-patch DeepseekV2DecoderLayer inside
        # deepseek_nextn before super().__init__() runs. The module-level import
        # of DeepseekModelNextN is insufficient for this purpose.
        import sglang.srt.models.deepseek_nextn as _nextn_module

        original_nextn_cls = _nextn_module.DeepseekV2DecoderLayer
        original_attention_cls = deepseek_v2.DeepseekV2AttentionMLA
        original_moe_cls = deepseek_v2.DeepseekV2MoE
        original_rmsnorm_cls = deepseek_v2.RMSNorm

        _nextn_module.DeepseekV2DecoderLayer = Gigachat35DecoderLayer
        deepseek_v2.DeepseekV2AttentionMLA = Gigachat35AttentionMLA
        deepseek_v2.DeepseekV2MoE = Gigachat35MoE
        norm_class = resolve_norm_class(config)
        if norm_class is not RMSNorm:
            deepseek_v2.RMSNorm = norm_class
        try:
            super().__init__(
                config,
                quant_config,
                draft_model_idx=draft_model_idx,
                prefix=prefix,
            )
        finally:
            _nextn_module.DeepseekV2DecoderLayer = original_nextn_cls
            deepseek_v2.DeepseekV2AttentionMLA = original_attention_cls
            deepseek_v2.DeepseekV2MoE = original_moe_cls
            deepseek_v2.RMSNorm = original_rmsnorm_cls

        self.enorm = build_norm(config, config.hidden_size)
        self.hnorm = build_norm(config, config.hidden_size)
        self.shared_head.norm = build_norm(config, config.hidden_size)


class Gigachat35ForCausalLMNextN(Gigachat35ForCausalLM):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        draft_model_idx: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        # Bypass DeepseekV3ForCausalLMNextN.__init__ to swap in Gigachat35ModelNextN.
        # use_dsa follows the MTP layer's own resolution; DSA context-parallel
        # prefill stays off.
        nn.Module.__init__(self)
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.quant_config = quant_config
        self.pp_group = get_pp_group()
        self.draft_model_idx = draft_model_idx or 0
        self.determine_num_fused_shared_experts("Gigachat35ForCausalLMNextN")
        self.use_dsa = gigachat35_mtp_is_dsa(config)
        self.dsa_enable_prefill_cp = False
        self.cp_rank = None
        self.cp_size = None

        self.model = Gigachat35ModelNextN(
            config,
            quant_config,
            draft_model_idx=self.draft_model_idx,
            prefix=add_prefix("model", prefix),
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("model.shared_head.head", prefix),
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        # NSA context-parallel prefill is not supported for Gigachat35
        # (self.nsa_enable_prefill_cp is always False), so the NSA CP guard
        # present in DeepseekV3ForCausalLMNextN.forward is intentionally omitted.
        hidden_states = self.model(input_ids, positions, forward_batch)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        self.do_load_weights(weights, is_nextn=True)


EntryClass = [Gigachat35ForCausalLM, Gigachat35ForCausalLMNextN]
