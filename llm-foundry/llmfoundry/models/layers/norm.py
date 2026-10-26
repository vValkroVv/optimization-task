# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn.functional as F
from types import ModuleType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union
from flash_attn.ops.rms_norm import dropout_add_rms_norm

import torch._inductor.config as config

if TYPE_CHECKING:
    from transformers import PretrainedConfig

config.max_autotune_gemm_backends = "ATEN,TRITON,NVGEMM"
config.max_autotune_conv_backends = "ATEN,TRITON,NVGEMM"

# avoid recompiles
torch.fx.experimental._config.use_duck_shape = False
torch._dynamo.config.recompile_limit = 64


def _cast_if_autocast_enabled(tensor: torch.Tensor) -> torch.Tensor:
    if torch.is_autocast_enabled():
        if tensor.device.type == "cuda":
            dtype = torch.get_autocast_gpu_dtype()
        elif tensor.device.type == "cpu":
            dtype = torch.get_autocast_cpu_dtype()
        else:
            raise NotImplementedError()
        return tensor.to(dtype=dtype)
    return tensor


def _output_scale_tensor(
    output_scale: Optional[Union[float, torch.Tensor]],
    reference: torch.Tensor,
) -> Optional[torch.Tensor]:
    if output_scale is None:
        return None
    if isinstance(output_scale, torch.Tensor):
        return output_scale.to(device=reference.device, dtype=reference.dtype)
    return torch.tensor(
        float(output_scale),
        device=reference.device,
        dtype=reference.dtype,
    )


def _scale_weight(
    weight: Optional[torch.Tensor],
    output_scale: Optional[Union[float, torch.Tensor]],
) -> Optional[torch.Tensor]:
    if weight is None:
        return None
    scale = _output_scale_tensor(output_scale, weight)
    if scale is None:
        return weight
    return weight * scale


def _output_scale_or_one(
    output_scale: Optional[Union[float, torch.Tensor]],
    reference: torch.Tensor,
) -> torch.Tensor:
    scale = _output_scale_tensor(output_scale, reference)
    if scale is None:
        return torch.ones((), device=reference.device, dtype=reference.dtype)
    return scale


@torch.compile(mode="max-autotune-no-cudagraphs", dynamic=True)
def _compiled_zero_centered_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    output_scale: torch.Tensor,
) -> torch.Tensor:
    output = x.float()
    output = output * torch.rsqrt(output.pow(2).mean(-1, keepdim=True) + eps)
    output = output * ((1.0 + weight.float()) * output_scale.float())
    return output.to(dtype=x.dtype)


@torch.compile(mode="max-autotune-no-cudagraphs", dynamic=True)
def _compiled_gated_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_down_weight: torch.Tensor,
    eps: float,
    layernorm_gating_weight: float,
    output_scale: torch.Tensor,
) -> torch.Tensor:
    output = x.float()
    output = output * torch.rsqrt(output.pow(2).mean(-1, keepdim=True) + eps)
    output = (output * weight.float()).to(dtype=x.dtype)
    gate_hidden = F.linear(output, gate_up_weight.to(dtype=x.dtype))
    gate_hidden = F.silu(gate_hidden.float()).to(dtype=x.dtype)
    gate_hidden = F.linear(gate_hidden, gate_down_weight.to(dtype=x.dtype))
    gate = torch.sigmoid(gate_hidden.float()).to(dtype=x.dtype)
    return (
        output * float(layernorm_gating_weight) * gate * output_scale.to(dtype=x.dtype)
    ).to(dtype=x.dtype)


@torch.compile(mode="max-autotune-no-cudagraphs", dynamic=True)
def _compiled_zero_centered_gated_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_down_weight: torch.Tensor,
    eps: float,
    layernorm_gating_weight: float,
    output_scale: torch.Tensor,
) -> torch.Tensor:
    output = x.float()
    output = output * torch.rsqrt(output.pow(2).mean(-1, keepdim=True) + eps)
    output = (output * (1.0 + weight.float())).to(dtype=x.dtype)
    gate_hidden = F.linear(output, gate_up_weight.to(dtype=x.dtype))
    gate_hidden = F.silu(gate_hidden.float()).to(dtype=x.dtype)
    gate_hidden = F.linear(gate_hidden, gate_down_weight.to(dtype=x.dtype))
    gate = torch.sigmoid(gate_hidden.float()).to(dtype=x.dtype)
    return (
        output * float(layernorm_gating_weight) * gate * output_scale.to(dtype=x.dtype)
    ).to(dtype=x.dtype)


def rms_norm(
    x: torch.Tensor, weight: Optional[torch.Tensor] = None, eps: float = 1e-5
) -> torch.Tensor:
    output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        return output * weight
    return output


class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        normalized_shape: Union[int, List[int], torch.Size],
        eps: float = 1e-5,
        weight: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        if weight:
            self.weight = torch.nn.Parameter(
                torch.empty(normalized_shape, dtype=dtype, device=device)
            )
        else:
            self.register_parameter("weight", None)

        if device != "meta":
            torch.nn.init.ones_(self.weight)

    @classmethod
    def from_config(
        cls, config: "PretrainedConfig", input_dim: Optional[int] = None
    ) -> "RMSNorm":
        """Alternative constructor that takes a config object."""
        return cls(
            normalized_shape=input_dim or config.hidden_size,
            eps=getattr(config, "rms_norm_eps", 1e-6),
            device=getattr(config, "init_device", None),
        )

    def forward(
        self,
        x: torch.Tensor,
        output_scale: Optional[Union[float, torch.Tensor]] = None,
    ) -> torch.Tensor:
        weight = _scale_weight(self.weight, output_scale)
        output = rms_norm(x.float(), weight, self.eps).to(dtype=x.dtype)
        if weight is None and output_scale is not None:
            output = output * _output_scale_or_one(output_scale, x).to(dtype=x.dtype)
        return output


class LlamaRMSNorm(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any,
    ) -> None:
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()

        self.weight = torch.nn.Parameter(torch.empty(hidden_size, device=device))
        self.variance_epsilon = eps

        if hidden_size <= 8192:
            self.rmsnorm_fn = dropout_add_rms_norm
        else:
            self.rmsnorm_fn = torch.compile(self.compute_rmsnorm, fullgraph=True)

        if device != "meta":
            torch.nn.init.ones_(self.weight)

    @classmethod
    def from_config(
        cls, config: "PretrainedConfig", input_dim: Optional[int] = None
    ) -> "LlamaRMSNorm":
        """Alternative constructor that takes a config object."""
        return cls(
            hidden_size=input_dim or config.hidden_size,
            eps=getattr(config, "rms_norm_eps", 1e-6),
            device=getattr(config, "init_device", None),
        )

    @staticmethod
    def compute_rmsnorm(
        x: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> torch.Tensor:
        def _norm(x: torch.Tensor, eps: float) -> torch.Tensor:
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

        output = _norm(x.float(), eps).type_as(x)
        return output * weight

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_scale: Optional[Union[float, torch.Tensor]] = None,
    ) -> torch.Tensor:
        downcast_hidden_states = _cast_if_autocast_enabled(hidden_states)
        downcast_weight: Optional[torch.Tensor] = (
            _cast_if_autocast_enabled(self.weight)
            if self.weight is not None
            else self.weight
        )
        downcast_weight = _scale_weight(downcast_weight, output_scale)

        if self.rmsnorm_fn is dropout_add_rms_norm:
            return self.rmsnorm_fn(
                x0=downcast_hidden_states,
                residual=None,
                weight=downcast_weight,
                bias=None,
                dropout_p=0.0,
                epsilon=self.variance_epsilon,
                rowscale=None,
                layerscale=None,
            )
        else:
            return self.rmsnorm_fn(
                x=downcast_hidden_states,
                weight=downcast_weight,
                eps=self.variance_epsilon,
            )


# NOTE: Original from Qwen3NextRMSNorm
class ZeroCenteredRMSNorm(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.eps = eps
        self.weight = torch.nn.Parameter(torch.zeros(hidden_size, device=device))

    @classmethod
    def from_config(
        cls, config: "PretrainedConfig", input_dim: Optional[int] = None
    ) -> "ZeroCenteredRMSNorm":
        """Alternative constructor that takes a config object."""
        return cls(
            hidden_size=input_dim or config.hidden_size,
            eps=getattr(config, "rms_norm_eps", 1e-6),
            device=getattr(config, "init_device", None),
        )

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    @staticmethod
    def compute_zero_centered_rmsnorm(
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        output_scale: torch.Tensor,
    ) -> torch.Tensor:
        return _compiled_zero_centered_rmsnorm(
            x=x,
            weight=weight,
            eps=eps,
            output_scale=output_scale,
        )

    def forward(
        self,
        x: torch.Tensor,
        output_scale: Optional[Union[float, torch.Tensor]] = None,
    ) -> torch.Tensor:
        with torch.autocast(
            device_type=x.device.type,
            dtype=torch.float32,
        ):
            output = self.compute_zero_centered_rmsnorm(
                x=x,
                weight=self.weight,
                eps=self.eps,
                output_scale=_output_scale_or_one(output_scale, self.weight),
            )
        return output

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


NORM_BACKENDS = ("compile", "cute")
CUTE_HIDDEN_MULTIPLE = 128


def _load_cute_gated_norm() -> ModuleType:
    """Import the fused CuTe GatedNorm package lazily: it pulls in nvidia-cutlass-dsl, which
    is only needed when a gated norm runs with norm_backend="cute"."""
    try:
        from llmfoundry.models.ops.cute_kernels import gated_norm
    except ImportError as exc:
        raise ImportError(
            'norm_backend="cute" needs nvidia-cutlass-dsl and apache-tvm-ffi '
            "(llmfoundry.models.ops.cute_kernels.gated_norm)."
        ) from exc
    return gated_norm


class GatedNorm(torch.nn.Module):
    """RMSNorm with a rank-16 sigmoid gate; `norm_backend` picks the implementation, the
    parameters and the math are the same:

      "compile"  the torch.compile'd reference (`_compiled_gated_norm`)
      "cute"     the fused CuTe DSL kernels for Hopper
                 (llmfoundry.models.ops.cute_kernels.gated_norm): bf16 x read directly, every
                 reduction in fp32; needs hidden_size % 128 == 0. Inputs the kernels do not
                 take (x that is not a bf16 CUDA tensor) run the "compile" path instead, so
                 CPU / fp32 / meta-device code keeps working.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        layernorm_gating_weight: float = 2.0,
        device: Optional[Union[str, torch.device]] = None,
        norm_backend: str = "compile",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if norm_backend not in NORM_BACKENDS:
            raise ValueError(
                f"norm_backend must be one of {NORM_BACKENDS}, got {norm_backend!r}."
            )
        if norm_backend == "cute":
            if hidden_size % CUTE_HIDDEN_MULTIPLE:
                raise ValueError(
                    'norm_backend="cute" needs hidden_size to be a multiple of '
                    f"{CUTE_HIDDEN_MULTIPLE}, got {hidden_size}."
                )
            _load_cute_gated_norm()  # fail at model build, not at the first forward
        self.norm_backend = norm_backend
        self.hidden_size = hidden_size

        self.r = 16  # TODO: fix magic constant
        self.weight = torch.nn.Parameter(torch.zeros(hidden_size, device=device))
        self.gate_up_projection = torch.nn.Linear(
            hidden_size, self.r, bias=False, device=device
        )
        self.gate_down_projection = torch.nn.Linear(
            self.r, hidden_size, bias=False, device=device
        )
        self.variance_epsilon = eps
        self.layernorm_gating_weight = layernorm_gating_weight

    @classmethod
    def from_config(
        cls, config: "PretrainedConfig", input_dim: Optional[int] = None
    ) -> "GatedNorm":
        """Alternative constructor that takes a config object."""
        return cls(
            hidden_size=input_dim or config.hidden_size,
            eps=getattr(config, "rms_norm_eps", 1e-6),
            layernorm_gating_weight=getattr(config, "layernorm_gating_weight", 2.0),
            device=getattr(config, "init_device", None),
            norm_backend=getattr(config, "norm_backend", "compile"),
        )

    @staticmethod
    def compute_norm(
        x: torch.Tensor,
        weight: torch.Tensor,
        gate_up_weight: torch.Tensor,
        gate_down_weight: torch.Tensor,
        eps: float,
        layernorm_gating_weight: float,
        output_scale: torch.Tensor,
    ) -> torch.Tensor:
        for dim in range(x.ndim - 1):
            torch._dynamo.mark_dynamic(x, dim)
        if x.ndim > 1:
            torch._dynamo.decorators.mark_unbacked(x, 0)
        return _compiled_gated_norm(
            x=x,
            weight=weight,
            gate_up_weight=gate_up_weight,
            gate_down_weight=gate_down_weight,
            eps=eps,
            layernorm_gating_weight=layernorm_gating_weight,
            output_scale=output_scale,
        )

    def _cute_weight(self) -> torch.Tensor:
        """Per-channel scale as the CuTe kernel takes it (fp32)."""
        return self.weight.float()

    def _cute_forward(
        self,
        x: torch.Tensor,
        output_scale: Optional[Union[float, torch.Tensor]],
    ) -> torch.Tensor:
        output = _load_cute_gated_norm().gated_norm(
            x,
            self._cute_weight(),
            self.gate_up_projection.weight,
            self.gate_down_projection.weight,
            self.variance_epsilon,
            self.layernorm_gating_weight,
        )
        scale = _output_scale_tensor(output_scale, output)
        if scale is not None:
            output = output * scale
        return output

    def precompile(self) -> None:
        """norm_backend="cute": compile every kernel variant this layer can hit, for any
        number of tokens (the only M-dependent compile is the F2 chunking, a handful of
        variants per hidden size). Call once after the parameters are on the GPU in their
        final dtype, so varlen batches and sequence-length warmup never stall on a compile
        mid-run. A no-op for the "compile" backend.
        """
        if self.norm_backend != "cute":
            return
        _load_cute_gated_norm().gated_norm_precompile(
            self._cute_weight(),
            self.gate_up_projection.weight,
            self.gate_down_projection.weight,
            self.variance_epsilon,
            self.layernorm_gating_weight,
        )

    def forward(
        self,
        x: torch.Tensor,
        output_scale: Optional[Union[float, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if self.norm_backend == "cute":
            assert x.is_cuda, (
                f'norm_backend="cute" requires a CUDA input tensor, got device={x.device}.'
            )
            assert x.dtype == torch.bfloat16, (
                f'norm_backend="cute" requires torch.bfloat16 input, got dtype={x.dtype}.'
            )
            return self._cute_forward(x, output_scale)
        with torch.autocast(
            device_type=x.device.type,
            dtype=torch.float32,
        ):
            output = self.compute_norm(
                x=x,
                weight=self.weight,
                gate_up_weight=self.gate_up_projection.weight,
                gate_down_weight=self.gate_down_projection.weight,
                eps=self.variance_epsilon,
                layernorm_gating_weight=self.layernorm_gating_weight,
                output_scale=_output_scale_or_one(output_scale, self.weight),
            )

        return output

    def extra_repr(self) -> str:
        return (
            f"({self.hidden_size},), eps={self.variance_epsilon}, "
            f"backend={self.norm_backend}"
        )


class ZeroCenteredGatedNorm(GatedNorm):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        layernorm_gating_weight: float = 2.0,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            eps=eps,
            layernorm_gating_weight=layernorm_gating_weight,
            device=device,
            **kwargs,
        )

    @staticmethod
    def compute_norm(
        x: torch.Tensor,
        weight: torch.Tensor,
        gate_up_weight: torch.Tensor,
        gate_down_weight: torch.Tensor,
        eps: float,
        layernorm_gating_weight: float,
        output_scale: torch.Tensor,
    ) -> torch.Tensor:
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.decorators.mark_unbacked(x, 0)
        torch._dynamo.mark_dynamic(x, 1)
        return _compiled_zero_centered_gated_norm(
            x=x,
            weight=weight,
            gate_up_weight=gate_up_weight,
            gate_down_weight=gate_down_weight,
            eps=eps,
            layernorm_gating_weight=layernorm_gating_weight,
            output_scale=output_scale,
        )

    def _cute_weight(self) -> torch.Tensor:
        return 1.0 + self.weight.float()  # the `1 + weight` parameterization, in fp32


NORM_CLASS_REGISTRY: Dict[str, Type[torch.nn.Module]] = {
    "rmsnorm": RMSNorm,
    "LlamaRMSNorm": LlamaRMSNorm,
    "ZeroCenteredRMSNorm": ZeroCenteredRMSNorm,
    "ZeroCenteredGatedNorm": ZeroCenteredGatedNorm,
    "GatedNorm": GatedNorm,
}

ZERO_CENTERED_NORM_CLASSES = (ZeroCenteredRMSNorm, ZeroCenteredGatedNorm)

GATED_NORM_CLASSES = (ZeroCenteredGatedNorm, GatedNorm)


def resolve_norm_class(norm_type: str) -> Type[torch.nn.Module]:
    if not isinstance(norm_type, str):
        raise TypeError(
            f"`norm_type` must be a string, got {type(norm_type).__name__}."
        )

    raw_norm_type = norm_type
    norm_type = norm_type.strip()
    if not norm_type:
        raise ValueError("`norm_type` must be a non-empty string.")

    norm_class = NORM_CLASS_REGISTRY.get(norm_type)
    if norm_class is None:
        available_norms = ", ".join(sorted(NORM_CLASS_REGISTRY.keys()))
        raise ValueError(
            f"Unknown norm type `{raw_norm_type}`. Available values: {available_norms}."
        )
    return norm_class
