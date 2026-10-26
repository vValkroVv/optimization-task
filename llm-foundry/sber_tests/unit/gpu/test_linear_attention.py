"""
Unit tests for linear attention architectures.

Tests single-GPU (against HF reference when available) and Sequence Parallel (SP)
implementations for linear attention architectures.

To add a new architecture:
1. Add architecture config via register_architecture()
2. Optionally add HF reference hooks for single-GPU reference tests

Skip architectures by setting `skip_sp`, `skip_single_gpu`, or `skip_vs_hf` in the config.
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import PretrainedConfig

from .sp_tests_base import run_sp_test_impl, tensor_diff_stats

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*flash-attn.*")


def seed_all(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _assert_sim_close(
    a: torch.Tensor,
    b: torch.Tensor,
    name: str,
    atol: float = 1e-2,
) -> None:
    assert a.shape == b.shape, f"{name}: shape mismatch: {a.shape} vs {b.shape}"

    stats = tensor_diff_stats(a, b, atol=atol)

    if not stats.close:
        raise AssertionError(
            (
                f"{name}: tensors are not similar.\n"
                f"  sim_diff = {stats.sim_diff:.6e} (threshold = {atol})\n"
                f"  max_abs_diff = {stats.max_abs_diff:.6e}\n"
                f"  max_rel_diff = {stats.max_rel_diff:.6e}"
            )
        )


@dataclass
class ArchitectureConfig:
    name: str
    attn_class: Type
    config_class: Type[PretrainedConfig]
    default_config: Dict[str, Any] = field(default_factory=dict)
    skip_sp: bool = False
    skip_single_gpu: bool = False
    skip_vs_hf: bool = False
    skip_reason: str = ""
    skip_vs_hf_reason: str = ""
    hf_reference_class: Optional[Type] = None
    hf_config_class: Optional[Type[PretrainedConfig]] = None
    hf_reference_factory: Optional[Callable[[PretrainedConfig], torch.nn.Module]] = None
    build_hf_test_configs_fn: Optional[
        Callable[..., tuple[PretrainedConfig, PretrainedConfig]]
    ] = None
    copy_weights_fn: Optional[Callable] = None
    assert_hf_grads_fn: Optional[
        Callable[[torch.nn.Module, torch.nn.Module, float], None]
    ] = None
    sp_world_size: int = 2
    supports_short_conv: bool = False


ARCHITECTURE_CONFIGS: Dict[str, ArchitectureConfig] = {}
ARCHITECTURE_NAMES = ["Qwen3NextGatedDeltaNet", "KimiDeltaAttention"]


class Qwen3NextGDNMockConfig(PretrainedConfig):
    # torch.multiprocessing.spawn requires top-level classes so configs stay picklable.
    def __init__(
        self,
        hidden_size: int = 2048,
        linear_key_head_dim: int = 128,
        linear_value_head_dim: int = 128,
        linear_conv_kernel_dim: int = 4,
        linear_num_key_heads: int = 16,
        linear_num_value_heads: int = 16,
        linear_use_short_conv: bool = False,
        linear_sp_implementation: str = "all2all",
        linear_gating_type: str = "gated_rmsnorm",
        linear_sigmoid_gate_scale: float = 1.0,
        linear_attn_o_norm_eps: Optional[float] = None,
        linear_conv_init_std: float = 0.006,
        linear_use_legacy_qkvz_layout: bool = False,
        rms_norm_eps: float = 1e-5,
        init_device: str = "cuda",
        **kwargs: dict[str, Any],
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.linear_use_short_conv = linear_use_short_conv
        self.linear_sp_implementation = linear_sp_implementation
        self.linear_gating_type = linear_gating_type
        self.linear_sigmoid_gate_scale = linear_sigmoid_gate_scale
        self.linear_attn_o_norm_eps = linear_attn_o_norm_eps
        self.linear_conv_init_std = linear_conv_init_std
        self.linear_use_legacy_qkvz_layout = linear_use_legacy_qkvz_layout
        self.rms_norm_eps = rms_norm_eps
        self.init_device = init_device


class KimiDeltaMockConfig(PretrainedConfig):
    def __init__(
        self,
        hidden_size: int = 1024,
        linear_key_head_dim: int = 128,
        linear_value_head_dim: int = 128,
        linear_conv_kernel_dim: int = 4,
        linear_num_key_heads: int = 8,
        linear_num_value_heads: int = 8,
        linear_use_short_conv: bool = False,
        linear_sp_implementation: str = "all2all",
        linear_gating_type: str = "gated_rmsnorm",
        linear_sigmoid_gate_scale: float = 1.0,
        linear_attn_o_norm_eps: Optional[float] = None,
        linear_conv_init_std: float = 0.006,
        linear_use_legacy_qkvz_layout: bool = False,
        rms_norm_eps: float = 1e-5,
        init_device: str = "cuda",
        **kwargs: dict[str, Any],
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.linear_use_short_conv = linear_use_short_conv
        self.linear_sp_implementation = linear_sp_implementation
        self.linear_gating_type = linear_gating_type
        self.linear_sigmoid_gate_scale = linear_sigmoid_gate_scale
        self.linear_attn_o_norm_eps = linear_attn_o_norm_eps
        self.linear_conv_init_std = linear_conv_init_std
        self.linear_use_legacy_qkvz_layout = linear_use_legacy_qkvz_layout
        self.rms_norm_eps = rms_norm_eps
        self.init_device = init_device


def register_architecture(config: ArchitectureConfig):
    ARCHITECTURE_CONFIGS[config.name] = config


def _split_qwen3next_qkvz_weight(
    qkvz_weight: torch.Tensor,
    num_key_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    value_heads_per_key_head = num_value_heads // num_key_heads
    value_dim_per_key_head = value_heads_per_key_head * value_head_dim

    qkvz_weight = qkvz_weight.reshape(
        num_key_heads,
        2 * key_head_dim + 2 * value_dim_per_key_head,
        qkvz_weight.size(-1),
    )
    q, k, v, z = torch.split(
        qkvz_weight,
        [
            key_head_dim,
            key_head_dim,
            value_dim_per_key_head,
            value_dim_per_key_head,
        ],
        dim=1,
    )
    qkv_weight = torch.cat((q, k, v), dim=1).reshape(-1, qkvz_weight.size(-1))
    z_weight = z.reshape(-1, qkvz_weight.size(-1))
    return qkv_weight, z_weight


def _pack_qwen3next_qkvz_weight(
    qkv_weight: torch.Tensor,
    z_weight: torch.Tensor,
    num_key_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
) -> torch.Tensor:
    value_heads_per_key_head = num_value_heads // num_key_heads
    value_dim_per_key_head = value_heads_per_key_head * value_head_dim

    qkv_weight = qkv_weight.reshape(
        num_key_heads,
        2 * key_head_dim + value_dim_per_key_head,
        qkv_weight.size(-1),
    )
    z_weight = z_weight.reshape(
        num_key_heads,
        value_dim_per_key_head,
        z_weight.size(-1),
    )
    return torch.cat((qkv_weight, z_weight), dim=1).reshape(-1, qkv_weight.size(-1))


def _interleaved_qkv_conv_to_conv_order(
    weight: torch.Tensor,
    num_key_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
) -> torch.Tensor:
    """Reorder training's `qkv_conv1d.weight` (interleaved per-head
    [h0: q, k, v; h1: ...; ...]) into HF `conv1d.weight` conv order
    [q_all; k_all; v_all]. Works for any tail shape (weight `[C, 1, K]` or
    bias `[C]`).
    """
    value_heads_per_key_head = num_value_heads // num_key_heads
    value_dim_per_key_head = value_heads_per_key_head * value_head_dim
    per_head = 2 * key_head_dim + value_dim_per_key_head
    tail = weight.shape[1:]
    ph = weight.reshape(num_key_heads, per_head, *tail)
    q = ph[:, :key_head_dim].reshape(num_key_heads * key_head_dim, *tail)
    k = ph[:, key_head_dim:2 * key_head_dim].reshape(
        num_key_heads * key_head_dim, *tail
    )
    v = ph[:, 2 * key_head_dim:].reshape(
        num_key_heads * value_dim_per_key_head, *tail
    )
    return torch.cat((q, k, v), dim=0)


def _conv_order_to_interleaved_qkv_conv(
    weight: torch.Tensor,
    num_key_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
) -> torch.Tensor:
    """Inverse of `_interleaved_qkv_conv_to_conv_order`."""
    value_heads_per_key_head = num_value_heads // num_key_heads
    value_dim_per_key_head = value_heads_per_key_head * value_head_dim
    key_dim = num_key_heads * key_head_dim
    value_dim = num_key_heads * value_dim_per_key_head
    tail = weight.shape[1:]
    q, k, v = torch.split(weight, [key_dim, key_dim, value_dim], dim=0)
    q = q.reshape(num_key_heads, key_head_dim, *tail)
    k = k.reshape(num_key_heads, key_head_dim, *tail)
    v = v.reshape(num_key_heads, value_dim_per_key_head, *tail)
    return torch.cat((q, k, v), dim=1).reshape(-1, *tail)


def _copy_gated_norm_weight(
    source_norm: torch.nn.Module,
    target_norm: torch.nn.Module,
) -> None:
    source_weight = source_norm.weight.data
    if getattr(source_norm, "zero_centered", False):
        source_weight = source_weight + 1

    if getattr(target_norm, "zero_centered", False):
        source_weight = source_weight - 1

    target_norm.weight.data.copy_(source_weight)


def _copy_weights_qwen3next_gdn(
    base_module: torch.nn.Module,
    hf_attention_module: torch.nn.Module,
):
    with torch.no_grad():
        hf_attention_module.dt_bias.data.copy_(base_module.dt_bias.data)
        hf_attention_module.A_log.data.copy_(base_module.A_log.data)

        if hasattr(base_module, "qkvz_proj"):
            hf_attention_module.in_proj_qkvz.weight.data.copy_(
                base_module.qkvz_proj.weight.data
            )
        else:
            hf_attention_module.in_proj_qkvz.weight.data.copy_(
                _pack_qwen3next_qkvz_weight(
                    base_module.qkv_proj.weight.data,
                    base_module.z_proj.weight.data,
                    base_module.num_heads,
                    base_module.num_v_heads,
                    base_module.head_dim,
                    base_module.head_v_dim,
                )
            )

        hf_attention_module.in_proj_ba.weight.data.copy_(
            base_module.ba_proj.weight.data
        )

        if hasattr(base_module, "qkv_conv1d"):
            hf_attention_module.conv1d.weight.data.copy_(
                _interleaved_qkv_conv_to_conv_order(
                    base_module.qkv_conv1d.weight.data,
                    base_module.num_heads,
                    base_module.num_v_heads,
                    base_module.head_dim,
                    base_module.head_v_dim,
                )
            )
        elif hasattr(base_module, "conv1d"):
            hf_attention_module.conv1d.weight.data.copy_(base_module.conv1d.weight.data)
        else:
            hf_attention_module.conv1d.weight.data.copy_(
                torch.cat(
                    [
                        base_module.q_conv1d.weight.data,
                        base_module.k_conv1d.weight.data,
                        base_module.v_conv1d.weight.data,
                    ],
                    dim=0,
                )
            )

        hf_attention_module.out_proj.weight.data.copy_(base_module.o_proj.weight.data)
        _copy_gated_norm_weight(base_module.o_norm, hf_attention_module.norm)


def _assert_hf_grads_qwen3next_gdn(
    base_attention: torch.nn.Module,
    hf_attention: torch.nn.Module,
    atol: float,
):
    _assert_sim_close(
        base_attention.dt_bias.grad,
        hf_attention.dt_bias.grad,
        "dt_bias.grad",
        atol=atol,
    )
    _assert_sim_close(
        base_attention.A_log.grad,
        hf_attention.A_log.grad,
        "A_log.grad",
        atol=atol,
    )

    if hasattr(base_attention, "qkvz_proj"):
        _assert_sim_close(
            base_attention.qkvz_proj.weight.grad,
            hf_attention.in_proj_qkvz.weight.grad,
            "qkvz_proj.weight.grad",
            atol=atol,
        )
    else:
        hf_qkv_grad, hf_z_grad = _split_qwen3next_qkvz_weight(
            hf_attention.in_proj_qkvz.weight.grad,
            base_attention.num_heads,
            base_attention.num_v_heads,
            base_attention.head_dim,
            base_attention.head_v_dim,
        )
        _assert_sim_close(
            base_attention.qkv_proj.weight.grad,
            hf_qkv_grad,
            "qkv_proj.weight.grad",
            atol=atol,
        )
        _assert_sim_close(
            base_attention.z_proj.weight.grad,
            hf_z_grad,
            "z_proj.weight.grad",
            atol=atol,
        )

    _assert_sim_close(
        base_attention.ba_proj.weight.grad,
        hf_attention.in_proj_ba.weight.grad,
        "ba_proj.weight.grad",
        atol=atol,
    )

    has_split_short_convs = all(
        hasattr(base_attention, attr) for attr in ("q_conv1d", "k_conv1d", "v_conv1d")
    )

    if hasattr(base_attention, "qkv_conv1d"):
        base_qkv_conv1d_grad = base_attention.qkv_conv1d.weight.grad
        hf_conv1d_grad_as_interleaved = _conv_order_to_interleaved_qkv_conv(
            hf_attention.conv1d.weight.grad,
            base_attention.num_heads,
            base_attention.num_v_heads,
            base_attention.head_dim,
            base_attention.head_v_dim,
        )
        _assert_sim_close(
            base_qkv_conv1d_grad,
            hf_conv1d_grad_as_interleaved,
            "qkv_conv1d.weight.grad",
            atol=atol,
        )
    elif has_split_short_convs:
        base_q_grad = base_attention.q_conv1d.weight.grad
        base_k_grad = base_attention.k_conv1d.weight.grad
        base_v_grad = base_attention.v_conv1d.weight.grad

        hf_q_grad, hf_k_grad, hf_v_grad = torch.split(
            hf_attention.conv1d.weight.grad,
            split_size_or_sections=[
                base_attention.key_dim,
                base_attention.key_dim,
                base_attention.value_dim,
            ],
            dim=0,
        )

        _assert_sim_close(base_q_grad, hf_q_grad, "q_conv1d.weight.grad", atol=atol)
        _assert_sim_close(base_k_grad, hf_k_grad, "k_conv1d.weight.grad", atol=atol)
        _assert_sim_close(base_v_grad, hf_v_grad, "v_conv1d.weight.grad", atol=atol)
    elif hasattr(base_attention, "conv1d"):
        base_conv1d_grad = base_attention.conv1d.weight.grad
        hf_conv1d_grad = hf_attention.conv1d.weight.grad
        _assert_sim_close(
            base_conv1d_grad, hf_conv1d_grad, "conv1d.weight.grad", atol=atol
        )
    else:
        raise AssertionError("Unknown Qwen3NextGatedDeltaNet convolution layout")

    _assert_sim_close(
        base_attention.o_proj.weight.grad,
        hf_attention.out_proj.weight.grad,
        "o_proj.weight.grad",
        atol=atol,
    )

    _assert_sim_close(
        base_attention.o_norm.weight.grad,
        hf_attention.norm.weight.grad,
        "o_norm.weight.grad",
        atol=atol,
    )


def _get_qwen3next_gdn_config() -> Optional[ArchitectureConfig]:
    try:
        from llmfoundry.models.layers.attention import Qwen3NextGatedDeltaNet
    except ImportError as e:
        if "megablocks" in str(e) or "Please make sure to pip install" in str(e):
            return None
        raise

    try:
        from transformers.models.qwen3_next.configuration_qwen3_next import (
            Qwen3NextConfig,
        )
        from transformers.models.qwen3_next.modeling_qwen3_next import (
            Qwen3NextGatedDeltaNet as HFQwen3NextGatedDeltaNet,
        )

        hf_reference_class = HFQwen3NextGatedDeltaNet
        hf_config_class = Qwen3NextConfig
    except ImportError:
        hf_reference_class = None
        hf_config_class = None

    def _build_qwen3next_gdn_hf_configs(
        hidden_size: int,
        num_heads: int,
        num_value_heads_factor: int,
        head_dim: int,
        value_head_dim: int,
        use_short_conv: bool,
    ) -> tuple[PretrainedConfig, PretrainedConfig]:
        from llmfoundry.models.giga_mix.configuration_giga_mix import GigaMixConfig

        config = GigaMixConfig(
            hidden_size=hidden_size,
            linear_attention_type="Qwen3NextGatedDeltaNet",
            linear_conv_kernel_dim=4,
            linear_key_head_dim=head_dim,
            linear_value_head_dim=value_head_dim,
            linear_num_key_heads=num_heads,
            linear_num_value_heads=int(num_heads * num_value_heads_factor),
            linear_use_short_conv=use_short_conv,
        )

        if hf_config_class is None:
            raise RuntimeError("Qwen3Next HF config class is unavailable")

        hf_config = hf_config_class(
            hidden_size=hidden_size,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=head_dim,
            linear_value_head_dim=value_head_dim,
            linear_num_key_heads=num_heads,
            linear_num_value_heads=int(num_heads * num_value_heads_factor),
            dtype=torch.bfloat16,
        )
        return config, hf_config

    def _build_qwen3next_gdn_hf_reference(
        hf_config: PretrainedConfig,
    ) -> torch.nn.Module:
        if hf_reference_class is None:
            raise RuntimeError("Qwen3Next HF reference class is unavailable")
        return hf_reference_class(config=hf_config, layer_idx=0)

    hf_available = hf_reference_class is not None and hf_config_class is not None

    return ArchitectureConfig(
        name="Qwen3NextGatedDeltaNet",
        attn_class=Qwen3NextGatedDeltaNet,
        config_class=Qwen3NextGDNMockConfig,
        skip_vs_hf=not hf_available,
        skip_vs_hf_reason=(
            ""
            if hf_available
            else "HuggingFace Qwen3Next reference is not available in installed transformers"
        ),
        # TODO: fix sp tests and unset this
        skip_sp=True,
        hf_reference_class=hf_reference_class,
        hf_config_class=hf_config_class,
        hf_reference_factory=(
            _build_qwen3next_gdn_hf_reference
            if hf_reference_class is not None
            else None
        ),
        build_hf_test_configs_fn=(
            _build_qwen3next_gdn_hf_configs if hf_config_class is not None else None
        ),
        copy_weights_fn=_copy_weights_qwen3next_gdn,
        assert_hf_grads_fn=_assert_hf_grads_qwen3next_gdn,
        supports_short_conv=True,
    )


def _get_kimi_delta_config() -> Optional[ArchitectureConfig]:
    try:
        from llmfoundry.models.layers.attention import KimiDeltaAttention
    except ImportError as e:
        if "megablocks" in str(e) or "Please make sure to pip install" in str(e):
            return None
        raise

    return ArchitectureConfig(
        name="KimiDeltaAttention",
        attn_class=KimiDeltaAttention,
        config_class=KimiDeltaMockConfig,
        skip_vs_hf=True,
        skip_vs_hf_reason="HuggingFace Kimi reference is not available in installed transformers",
        # TODO: fix sp tests and unset this
        skip_sp=True,
        # TODO: refactor and update kimi, then unset this
        skip_single_gpu=True,
        supports_short_conv=False,
    )


def get_architecture_config(name: str) -> ArchitectureConfig:
    if name not in ARCHITECTURE_CONFIGS:
        if name == "Qwen3NextGatedDeltaNet":
            config = _get_qwen3next_gdn_config()
            if config is not None:
                register_architecture(config)
        elif name == "KimiDeltaAttention":
            config = _get_kimi_delta_config()
            if config is not None:
                register_architecture(config)
        else:
            raise ValueError(f"Unknown architecture: {name}")
    return ARCHITECTURE_CONFIGS[name]


def get_architecture_test_params() -> List[str]:
    return ARCHITECTURE_NAMES


def _make_mock_config(
    config_class: Type[PretrainedConfig],
    hidden_size: int,
    linear_num_key_heads: int,
    linear_num_value_heads: int,
    linear_key_head_dim: int,
    linear_value_head_dim: int,
    linear_use_short_conv: bool = False,
    linear_gating_type: str = "gated_rmsnorm",
    linear_sigmoid_gate_scale: float = 1.0,
    linear_attn_o_norm_eps: Optional[float] = None,
    linear_conv_init_std: float = 0.006,
    linear_use_legacy_qkvz_layout: bool = False,
    rms_norm_eps: float = 1e-6,
    **extra_kwargs: dict[str, Any],
) -> PretrainedConfig:
    return config_class(
        hidden_size=hidden_size,
        linear_key_head_dim=linear_key_head_dim,
        linear_value_head_dim=linear_value_head_dim,
        linear_num_key_heads=linear_num_key_heads,
        linear_num_value_heads=linear_num_value_heads,
        linear_use_short_conv=linear_use_short_conv,
        linear_sp_implementation="all2all",
        linear_gating_type=linear_gating_type,
        linear_sigmoid_gate_scale=linear_sigmoid_gate_scale,
        linear_attn_o_norm_eps=linear_attn_o_norm_eps,
        linear_conv_init_std=linear_conv_init_std,
        linear_use_legacy_qkvz_layout=linear_use_legacy_qkvz_layout,
        rms_norm_eps=rms_norm_eps,
        init_device="cuda",
        **extra_kwargs,
    )


def run_sp_test(
    rank: int,
    world_size: int,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    linear_num_key_heads: int,
    linear_num_value_heads: int,
    key_value_heads_dim: int,
    linear_use_short_conv: bool,
    arch_name: str,
    use_cu_seqlens: bool,
):
    arch_config = get_architecture_config(arch_name)
    try:
        config = _make_mock_config(
            arch_config.config_class,
            hidden_size=hidden_size,
            linear_num_key_heads=linear_num_key_heads,
            linear_num_value_heads=linear_num_value_heads,
            linear_key_head_dim=key_value_heads_dim,
            linear_value_head_dim=key_value_heads_dim,
            linear_use_short_conv=linear_use_short_conv,
        )
        run_sp_test_impl(
            arch_config.attn_class,
            rank,
            world_size,
            batch_size,
            seq_len,
            config,
            use_cu_seqlens=use_cu_seqlens,
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run_single_gpu_test_with_hf_reference(
    attn_class: Type,
    config: PretrainedConfig,
    hf_config: PretrainedConfig,
    hf_reference_factory: Callable[[PretrainedConfig], torch.nn.Module],
    copy_weights_fn: Callable,
    assert_hf_grads_fn: Callable[[torch.nn.Module, torch.nn.Module, float], None],
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    atol: float = 1e-2,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    seed_all(1)

    base_attention = attn_class(config)
    base_attention.reset_parameters()
    base_attention = base_attention.to(dtype).to(device)

    hf_attention = hf_reference_factory(hf_config)
    copy_weights_fn(base_attention, hf_attention)
    hf_attention = hf_attention.to(dtype).to(device)

    hidden_states = torch.randn(batch_size, seq_len, hidden_size).to(dtype).to(device)

    base_out, _, _ = base_attention(hidden_states)
    hf_out = hf_attention(hidden_states)

    _assert_sim_close(base_out, hf_out, "attention_output", atol=atol)
    assert not torch.isnan(base_out).any(), "attention_output contains NaNs"
    assert not torch.isinf(base_out).any(), "attention_output contains Infs"

    base_loss = base_out.mean()
    hf_loss = hf_out.mean()

    base_loss.backward()
    hf_loss.backward()

    assert_hf_grads_fn(base_attention, hf_attention, atol)

    for name, param in list(base_attention.named_parameters()) + list(
        hf_attention.named_parameters()
    ):
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"{name}.grad contains NaNs"
            assert not torch.isinf(param.grad).any(), f"{name}.grad contains Infs"


def run_single_gpu_test_basic(
    attn_class: Type,
    config: PretrainedConfig,
    batch_size: int,
    seq_len: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    seed_all(1)

    model = attn_class(config).to(device).to(dtype)
    model.reset_parameters()
    model.train()

    hidden_states = torch.randn(
        batch_size, seq_len, config.hidden_size, device=device, dtype=dtype
    )

    output, _, _ = model(hidden_states)

    assert not torch.isnan(output).any(), "Output contains NaNs"
    assert not torch.isinf(output).any(), "Output contains Infs"

    loss = output.mean()
    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"{name}.grad contains NaNs"
            assert not torch.isinf(param.grad).any(), f"{name}.grad contains Infs"


class TestLinearAttentionSP:
    """
    Sequence Parallel tests for linear attention architectures.

    Tests compare outputs and gradients between SP=1 (baseline) and SP=N.
    """

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    @pytest.mark.slow
    @pytest.mark.parametrize("arch_name", get_architecture_test_params())
    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("seq_len", [4096])
    @pytest.mark.parametrize("hidden_size", [4096])
    @pytest.mark.parametrize("linear_num_key_heads", [32])
    # TODO: find bug in HV == H, and add test case
    @pytest.mark.parametrize("linear_num_value_heads", [64])
    @pytest.mark.parametrize("key_value_heads_dim", [128])
    @pytest.mark.parametrize("use_cu_seqlens", [False, True])
    def test_linear_attention_sp(
        self,
        arch_name: str,
        batch_size: int,
        seq_len: int,
        hidden_size: int,
        linear_num_key_heads: int,
        linear_num_value_heads: int,
        key_value_heads_dim: int,
        use_cu_seqlens: bool,
    ):
        arch_config = get_architecture_config(arch_name)
        if arch_config.skip_sp:
            pytest.skip(f"SP test skipped for {arch_name}: {arch_config.skip_reason}")

        linear_use_short_conv = arch_config.supports_short_conv

        world_size = arch_config.sp_world_size
        if torch.cuda.device_count() < world_size:
            pytest.skip(
                f"Test requires {world_size} GPUs, found {torch.cuda.device_count()}"
            )

        mp.spawn(
            run_sp_test,
            args=(
                world_size,
                batch_size,
                seq_len,
                hidden_size,
                linear_num_key_heads,
                linear_num_value_heads,
                key_value_heads_dim,
                linear_use_short_conv,
                arch_name,
                use_cu_seqlens,
            ),
            nprocs=world_size,
            join=True,
        )


class TestLinearAttentionSingleGPU:
    """
    Single-GPU tests for linear attention architectures.

    Tests against HuggingFace reference implementation when available.
    """

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    @pytest.mark.slow
    @pytest.mark.parametrize("arch_name", get_architecture_test_params())
    @pytest.mark.parametrize("batch_size", [1])
    @pytest.mark.parametrize("seq_len", [4096])
    @pytest.mark.parametrize("hidden_size", [4096])
    @pytest.mark.parametrize("num_heads", [32])
    # TODO: find bug in HV == H, and add test case
    @pytest.mark.parametrize("num_value_heads_factor", [2])
    @pytest.mark.parametrize("head_dim", [128])
    @pytest.mark.parametrize("value_head_dim", [128])
    def test_linear_attention_vs_hf_reference(
        self,
        arch_name: str,
        batch_size: int,
        seq_len: int,
        hidden_size: int,
        num_heads: int,
        num_value_heads_factor: int,
        head_dim: int,
        value_head_dim: int,
    ):
        arch_config = get_architecture_config(arch_name)

        if arch_config.skip_vs_hf:
            pytest.skip(
                f"HF reference test skipped for {arch_name}: {arch_config.skip_vs_hf_reason}"
            )

        assert arch_config.hf_reference_factory is not None, "Badly configured HF test"
        assert arch_config.build_hf_test_configs_fn is not None, "Badly configured HF test"
        assert arch_config.copy_weights_fn is not None, "Badly configured HF test"
        assert arch_config.assert_hf_grads_fn is not None, "Badly configured HF test"

        config, hf_config = arch_config.build_hf_test_configs_fn(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_value_heads_factor=num_value_heads_factor,
            head_dim=head_dim,
            value_head_dim=value_head_dim,
            use_short_conv=arch_config.supports_short_conv,
        )

        run_single_gpu_test_with_hf_reference(
            attn_class=arch_config.attn_class,
            config=config,
            hf_config=hf_config,
            hf_reference_factory=arch_config.hf_reference_factory,
            copy_weights_fn=arch_config.copy_weights_fn,
            assert_hf_grads_fn=arch_config.assert_hf_grads_fn,
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_size=hidden_size,
        )

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    @pytest.mark.parametrize("arch_name", get_architecture_test_params())
    @pytest.mark.parametrize("batch_size", [1])
    @pytest.mark.parametrize("seq_len", [1024])
    @pytest.mark.parametrize("hidden_size", [2048])
    @pytest.mark.parametrize("num_heads", [16])
    # TODO: find bug in HV == H, and add test case
    @pytest.mark.parametrize("num_value_heads_factor", [2])
    @pytest.mark.parametrize("head_dim", [128])
    @pytest.mark.parametrize("value_head_dim", [128])
    def test_single_gpu_forward_backward(
        self,
        arch_name: str,
        batch_size: int,
        seq_len: int,
        hidden_size: int,
        num_heads: int,
        num_value_heads_factor: int,
        head_dim: int,
        value_head_dim: int,
    ):
        arch_config = get_architecture_config(arch_name)
        if arch_config.skip_single_gpu:
            pytest.skip(
                f"Single-GPU test skipped for {arch_name}: {arch_config.skip_reason}"
            )

        config = _make_mock_config(
            arch_config.config_class,
            hidden_size=hidden_size,
            linear_num_key_heads=num_heads,
            linear_num_value_heads=num_heads * num_value_heads_factor,
            linear_key_head_dim=head_dim,
            linear_value_head_dim=value_head_dim,
            linear_use_short_conv=arch_config.supports_short_conv,
        )

        run_single_gpu_test_basic(
            attn_class=arch_config.attn_class,
            config=config,
            batch_size=batch_size,
            seq_len=seq_len,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "unit and requires_gpu"])
