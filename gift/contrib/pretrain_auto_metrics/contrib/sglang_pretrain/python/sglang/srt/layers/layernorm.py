# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Fused operators for normalization layers."""

import functools
import logging
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

_is_triton_available = False
try:
    import triton
    import triton.language as tl
    _is_triton_available = True
except ImportError:
    pass

from sglang.srt.batch_invariant_ops import (
    is_batch_invariant_mode_enabled,
    rms_norm_batch_invariant,
)
from sglang.srt.environ import envs
from sglang.srt.layers.utils import MultiPlatformOp
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
    is_cuda,
    is_flashinfer_available,
    is_hip,
    is_musa,
    is_npu,
    is_xpu,
)

_is_cuda = is_cuda()
_is_flashinfer_available = is_flashinfer_available()
_is_hip = is_hip()
_is_musa = is_musa()
_is_npu = is_npu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
_is_xpu = is_xpu()
_flashinfer_layernorm_available = False

if _is_cuda or _is_xpu or _is_musa:
    if _is_flashinfer_available:
        try:
            import flashinfer.norm

            from sglang.srt.utils.custom_op import register_custom_op

            def _layernorm_fake_impl(
                input: torch.Tensor,
                gamma: torch.Tensor,
                beta: torch.Tensor,
                eps: float = 1e-6,
            ) -> torch.Tensor:
                return torch.empty_like(input)

            @register_custom_op(fake_impl=_layernorm_fake_impl)
            def layernorm(
                input: torch.Tensor,
                gamma: torch.Tensor,
                beta: torch.Tensor,
                eps: float = 1e-6,
            ) -> torch.Tensor:
                return flashinfer.norm.layernorm(input, gamma, beta, eps)

            _flashinfer_layernorm_available = True
        except (ImportError, AttributeError):
            _flashinfer_layernorm_available = False
    else:
        _flashinfer_layernorm_available = False

    from sgl_kernel import (
        fused_add_rmsnorm,
        gemma_fused_add_rmsnorm,
        gemma_rmsnorm,
        rmsnorm,
    )
_has_aiter_layer_norm = False
_has_vllm_rms_norm = False
if _use_aiter:
    from aiter import layernorm2d_fwd as layer_norm
    from aiter import rmsnorm2d_fwd as rms_norm
    from aiter import rmsnorm2d_fwd_with_add as fused_add_rms_norm

    _has_aiter_layer_norm = True  # aiter provides the layer_norm functions
    _has_vllm_rms_norm = True  # aiter provides the rms_norm functions
elif _is_hip:
    try:
        from vllm._custom_ops import fused_add_rms_norm, rms_norm

        _has_vllm_rms_norm = True
    except ImportError:
        # Fallback: vllm not available, will use forward_native
        _has_vllm_rms_norm = False

if _is_cuda:
    # HF-semantics RMSNorm kernel (JIT-compiled).  Used when `cast_x_before_out_mul=True`
    # (the transformers backend path) to produce outputs that are numerically identical
    # to HuggingFace `LlamaRMSNorm`: the cast from fp32 to the activation dtype happens
    # BEFORE the weight multiply, so the multiply is done in the narrow dtype.
    _jit_rmsnorm_hf_available = False
    try:
        from sglang.jit_kernel.rmsnorm_hf import (
            is_supported_rmsnorm_hf_hidden_size,
        )
        from sglang.jit_kernel.rmsnorm_hf import rmsnorm_hf as _jit_rmsnorm_hf

        _jit_rmsnorm_hf_available = True
    except ImportError:

        def is_supported_rmsnorm_hf_hidden_size(d: int) -> bool:
            return False

        _jit_rmsnorm_hf = None

    from sglang.jit_kernel.norm import fused_add_rmsnorm as _jit_fused_add_rmsnorm
    from sglang.jit_kernel.norm import (
        is_supported_jit_fused_add_rmsnorm_hidden_size,
    )


logger = logging.getLogger(__name__)

if _is_npu:
    import torch_npu
    from sgl_kernel_npu.norm.add_rmsnorm_bias import add_gemma_rms_norm


def _forward_with_allreduce_fusion(
    norm_module,
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
    post_residual_addition: Optional[torch.Tensor],
    weight: torch.Tensor,
    use_attn_tp_group: bool = True,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Shared allreduce-fused RMSNorm logic usable by any norm."""
    if residual is not None:
        from sglang.srt.distributed import (
            tensor_model_parallel_all_reduce,
            tensor_model_parallel_fused_allreduce_rmsnorm,
        )
        from sglang.srt.layers.flashinfer_comm_fusion import (
            flashinfer_allreduce_residual_rmsnorm,
        )

        if use_attn_tp_group:
            world_size = get_parallel().attn_tp_size
        else:
            if get_parallel().moe_ep_size > 1:
                world_size = get_parallel().moe_ep_size
            else:
                world_size = get_parallel().moe_tp_size

        if world_size > 1:
            if post_residual_addition is not None:
                residual = residual + post_residual_addition

            # Prefer AITER fused AR+RMSNorm when enabled on AMD.
            if _use_aiter:
                fused_result = tensor_model_parallel_fused_allreduce_rmsnorm(
                    x, residual, weight, norm_module.variance_epsilon
                )
                if fused_result is not None:
                    return fused_result
            else:
                fused_result = flashinfer_allreduce_residual_rmsnorm(
                    input_tensor=x,
                    residual=residual,
                    weight=weight,
                    eps=norm_module.variance_epsilon,
                    max_token_num=max(x.shape[0], 2048),
                    use_attn_tp_group=use_attn_tp_group,
                )
                if fused_result[0] is not None:
                    return fused_result

            # For AITER route, preserve correctness when fused path is unavailable.
            if _use_aiter and get_global_server_args().enable_aiter_allreduce_fusion:
                x = tensor_model_parallel_all_reduce(x)
                return norm_module.forward(x, residual, None)

    return norm_module.forward(x, residual, post_residual_addition)


class RMSNorm(MultiPlatformOp):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: Optional[int] = None,
        cast_x_before_out_mul: bool = False,
        fp32_residual: bool = False,
        has_weight: bool = True,
        weight_dtype: Optional = None,
        override_orig_dtype: Optional = None,
        x_pad_to_multiple: int = 0,
    ) -> None:
        super().__init__()
        self.has_weight = has_weight
        self.cast_x_before_out_mul = cast_x_before_out_mul
        self.fp32_residual = fp32_residual
        self.override_orig_dtype = override_orig_dtype
        if self.has_weight:
            self.weight = nn.Parameter(torch.ones(hidden_size, dtype=weight_dtype))
        else:
            self.weight = torch.ones(hidden_size, dtype=weight_dtype)
        self.variance_epsilon = eps
        self.hidden_size = hidden_size
        self.variance_size_override = (
            None if var_hidden_size == hidden_size else var_hidden_size
        )
        # When > 0, fuse a zero-pad of the last dim out to a multiple of
        # this value into the rmsnorm kernel via aiter's
        # `fused_add_rmsnorm_pad` Triton kernel. The padded output has
        # shape (M, ceil(N/x_pad_to_multiple)*x_pad_to_multiple); the
        # residual_out stays at the original (M, N) shape.

        if _use_aiter:
            self.x_pad_to_multiple = x_pad_to_multiple
            self._fused_pad_kernel = None

            if x_pad_to_multiple > 0:
                try:
                    from aiter.ops.triton.fused_add_rmsnorm_pad import (
                        fused_add_rmsnorm_pad as _fused_add_rmsnorm_pad,
                    )

                    self._fused_pad_kernel = _fused_add_rmsnorm_pad
                except ImportError:
                    self._fused_pad_kernel = None
            self._forward_method = self.forward_aiter

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if x.numel() == 0:
            if residual is not None:
                if post_residual_addition is not None:
                    residual = residual + post_residual_addition
                return x, residual
            return x
        # sgl_kernel rmsnorm requires 2D input; reshape higher-rank tensors
        needs_reshape = x.dim() != 2 and residual is None
        if needs_reshape:
            original_shape = x.shape
            x = x.contiguous().reshape(-1, original_shape[-1])
        if self.variance_size_override is not None:
            return self.forward_native(x, residual, post_residual_addition)
        if is_batch_invariant_mode_enabled():
            if (
                residual is not None
                or self.cast_x_before_out_mul
                or get_global_server_args().rl_on_policy_target == "fsdp"
            ):
                return self.forward_native(x, residual, post_residual_addition)
            return rms_norm_batch_invariant(
                x,
                self.weight.data,
                self.variance_epsilon,
            )
        if self.cast_x_before_out_mul and residual is None:
            # Use HF-semantics kernel (cast to dtype before weight multiply).
            if (
                _jit_rmsnorm_hf_available
                and x.dtype in (torch.float16, torch.bfloat16)
                and self.weight.data.dtype == x.dtype
                and is_supported_rmsnorm_hf_hidden_size(x.shape[-1])
            ):
                out = _jit_rmsnorm_hf(
                    x.contiguous(), self.weight.data, self.variance_epsilon
                )
            else:
                # Fallback: pure-Python HF semantics (already implemented in forward_native).
                out = self.forward_native(x, None, None)
            if needs_reshape:
                out = out.reshape(original_shape)
            return out
        if residual is not None:
            if self.cast_x_before_out_mul:
                if (
                    x.dtype in (torch.float16, torch.bfloat16)
                    and self.weight.data.dtype == x.dtype
                    and (
                        post_residual_addition is None
                        or post_residual_addition.dtype == x.dtype
                    )
                    and is_supported_jit_fused_add_rmsnorm_hidden_size(x.shape[-1])
                ):
                    if post_residual_addition is not None:
                        residual = residual + post_residual_addition
                    _jit_fused_add_rmsnorm(
                        x,
                        residual,
                        self.weight.data,
                        self.variance_epsilon,
                        cast_x_before_out_mul=self.cast_x_before_out_mul,
                    )
                    return x, residual
                return self.forward_native(x, residual, post_residual_addition)
            # TODO: Ideally we want to have (hidden_states+residual)+post_residual_addition.
            # but right now we can only have hidden_states+(residual+post_residual_addition).
            # (hidden_states+residual)+post_residual_addition != hidden_states+(residual+post_residual_addition),
            # we probably need to add another parameter to fused_add_rmsnorm
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            fused_add_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)
            return x, residual
        out = rmsnorm(x, self.weight.data, self.variance_epsilon)
        if needs_reshape:
            out = out.reshape(original_shape)
        return out

    def forward_npu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if residual is not None:
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            out, _, residual_out = torch_npu.npu_add_rms_norm(
                residual, x, self.weight.data, self.variance_epsilon
            )
            return out, residual_out
        return torch_npu.npu_rms_norm(x, self.weight.data, self.variance_epsilon)[0]

    def forward_aiter(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Fix dsv4 dp attenton issue
        # the symptom is torch.AcceleratorError: HIP error: invalid configuration argument
        if x.shape[0] == 0:
            if residual is not None:
                return x, residual
            return x
        # Aiter's RMSNorm kernels expect 2D contiguous inputs. Keep the
        # already-safe layout as a zero-copy path, and only normalize strided or
        # higher-rank views such as Q/K slices from packed QKV projections.
        needs_reshape = x.dim() != 2 and residual is None
        if needs_reshape:
            original_shape = x.shape
            x = x.contiguous().reshape(-1, original_shape[-1])
        elif not x.is_contiguous():
            x = x.contiguous()
        # Fused (add +) rmsnorm + zero-pad path. Triggered when caller
        # constructed RMSNorm with x_pad_to_multiple > 0. Output last
        # dim is padded up; residual_out stays at original width. Used
        # by callers (e.g. GPT-OSS MXFP4 MoE) whose immediate consumer
        # needs a padded hidden_size — folding the pad in here removes a
        # separate launch.
        if self._fused_pad_kernel is not None and self.x_pad_to_multiple > 0:
            if post_residual_addition is not None and residual is not None:
                residual = residual + post_residual_addition
            return self._fused_pad_kernel(
                x,
                self.weight.data,
                self.variance_epsilon,
                residual,
                self.x_pad_to_multiple,
            )
        if residual is not None:
            residual_out = torch.empty_like(x)
            output = torch.empty_like(x)
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            fused_add_rms_norm(
                output,
                x,
                residual,
                residual_out,
                self.weight.data,
                self.variance_epsilon,
            )
            return output, residual_out
        output = rms_norm(x, self.weight.data, self.variance_epsilon)
        if needs_reshape:
            output = output.reshape(original_shape)
        return output

    def forward_hip(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Fallback to native implementation if vllm is not available
        if not _has_vllm_rms_norm:
            return self.forward_native(x, residual, post_residual_addition)

        if not x.is_contiguous():
            # NOTE: Remove this if aiter kernel supports discontinuous input
            x = x.contiguous()
        if residual is not None:
            out = torch.empty_like(x)
            residual_out = torch.empty_like(x)
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            fused_add_rms_norm(
                out, x, residual_out, residual, self.weight.data, self.variance_epsilon
            )
            return out, residual_out
        out = torch.empty_like(x)
        rms_norm(out, x, self.weight.data, self.variance_epsilon)
        return out

    def forward_musa(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if check_cuda_graph_backend(Phase.PREFILL, Backend.TC_PIECEWISE):
            return self.forward_native(x, residual, post_residual_addition)

        if not x.is_contiguous():
            x = x.contiguous()

        if residual is not None:
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            fused_add_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)
            return x, residual

        out = nn.functional.rms_norm(
            x, (self.hidden_size,), self.weight.data, self.variance_epsilon
        )
        return out

    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if not x.is_contiguous():
            x = x.contiguous()
        orig_dtype = self.override_orig_dtype or x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            if post_residual_addition is not None:
                x = x + post_residual_addition.to(torch.float32)
            if self.fp32_residual:
                residual = x.clone()
            else:
                residual = x.to(orig_dtype)

        hidden_size = x.shape[-1]
        if hidden_size != self.hidden_size:
            raise ValueError(
                "Expected hidden_size to be "
                f"{self.hidden_size}, but found: {hidden_size}"
            )

        if self.variance_size_override is None:
            x_var = x
        else:
            if hidden_size < self.variance_size_override:
                raise ValueError(
                    "Expected hidden_size to be at least "
                    f"{self.variance_size_override}, but found: {hidden_size}"
                )

            x_var = x[..., : self.variance_size_override]

        variance = x_var.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)

        if self.cast_x_before_out_mul:
            x = self.weight * x.to(orig_dtype)
        else:
            x = (x * self.weight).to(orig_dtype)

        if residual is None:
            return x
        else:
            return x, residual

    def forward_cpu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if _is_cpu_amx_available:
            if residual is not None:
                if post_residual_addition is not None:
                    residual = residual + post_residual_addition
                torch.ops.sgl_kernel.fused_add_rmsnorm_cpu(
                    x, residual, self.weight.data, self.variance_epsilon
                )
                return x, residual
            return torch.ops.sgl_kernel.rmsnorm_cpu(
                x, self.weight.data, self.variance_epsilon
            )
        else:
            return self.forward_native(x, residual, post_residual_addition)

    def forward_xpu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.variance_size_override is not None:
            return self.forward_native(x, residual, post_residual_addition)
        if is_batch_invariant_mode_enabled():
            if (
                residual is not None
                or get_global_server_args().rl_on_policy_target == "fsdp"
            ):
                return self.forward_native(x, residual, post_residual_addition)
            return rms_norm_batch_invariant(
                x,
                self.weight.data,
                self.variance_epsilon,
            )
        if residual is not None:
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            fused_add_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)
            return x, residual
        out = rmsnorm(x, self.weight.data, self.variance_epsilon)
        return out

    def forward_with_allreduce_fusion(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
        use_attn_tp_group: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward with allreduce fusion, prioritizing flashinfer fused operations."""
        return _forward_with_allreduce_fusion(
            self, x, residual, post_residual_addition, self.weight, use_attn_tp_group
        )


class LayerNorm(MultiPlatformOp):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.elementwise_affine = elementwise_affine
        self.use_bias = bias
        self.dtype = dtype

        self.bias = nn.Parameter(torch.zeros(hidden_size, dtype=self.dtype))
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=self.dtype))

    def forward_cuda(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if (
            _flashinfer_layernorm_available
            and x.dtype == torch.bfloat16
            and self.dtype == torch.float32
        ):
            return layernorm(x, self.weight, self.bias, self.variance_epsilon)
        else:
            return self.forward_native(x)

    def forward_native(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        weight = self.weight if self.elementwise_affine else None
        bias = self.bias if self.use_bias else None
        orig_dtype = x.dtype
        x = x.to(self.dtype)
        return F.layer_norm(
            x,
            (self.hidden_size,),
            weight=weight,
            bias=bias,
            eps=self.variance_epsilon,
        ).to(orig_dtype)

    def forward_hip(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if (
            _has_aiter_layer_norm
            and x.dtype in (torch.bfloat16, torch.float16)
            and x.dtype == self.dtype
        ):
            orig_shape = x.shape
            x = x.reshape(-1, self.hidden_size)
            return layer_norm(x, self.weight, self.bias, self.variance_epsilon).view(
                orig_shape
            )
        else:
            return self.forward_native(x)

    def forward_npu(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_native(x)

    def forward_cpu(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if _is_cpu_amx_available:
            bias_data = self.bias.data if self.use_bias else None
            return torch.ops.sgl_kernel.layernorm_cpu(
                x, self.weight.data, bias_data, self.variance_epsilon
            )
        else:
            return self.forward_native(x)


class GemmaRMSNorm(MultiPlatformOp):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
        self.register_buffer(
            "gemma_weight", torch.ones_like(self.weight), persistent=False
        )
        # (Chen-0210) Gemma weight = standard_weight + 1. Precompute once.
        # If TRTLLM allreduce fusion ever provides gemma-style norm
        # natively, this can be removed.
        self.weight.weight_loader = self._weight_loader

    def _weight_loader(self, param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        assert param.size() == loaded_weight.size()
        param.data.copy_(loaded_weight)
        # Keep storage stable for CUDA graphs or fused paths that capture this buffer.
        torch.add(param.data, 1.0, out=self.gemma_weight)

    def _forward_impl(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # flashinfer's gemma_rmsnorm CUDA kernel rejects 0-row inputs with
        # 'invalid configuration argument'. Short-circuit for DP-attn idle ranks.
        if x.numel() == 0:
            return (x, residual) if residual is not None else x
        needs_reshape = x.dim() != 2 and residual is None
        if needs_reshape:
            original_shape = x.shape
            x = x.contiguous().reshape(-1, original_shape[-1])
        if residual is not None:
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            gemma_fused_add_rmsnorm(
                x, residual, self.weight.data, self.variance_epsilon
            )
            return x, residual
        out = gemma_rmsnorm(x, self.weight.data, self.variance_epsilon)
        if needs_reshape:
            out = out.reshape(original_shape)
        return out

    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        orig_dtype = x.dtype
        if residual is not None:
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            x = x + residual
            residual = x

        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x * (1.0 + self.weight.float())
        x = x.to(orig_dtype)
        return x if residual is None else (x, residual)

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self._forward_impl(x, residual, post_residual_addition)

    def forward_hip(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if not _has_vllm_rms_norm:
            return self.forward_native(x, residual, post_residual_addition)

        w = self.gemma_weight
        if _use_aiter:
            # aiter API: rms_norm(input, weight, eps) -> output
            #            fused_add_rms_norm(output, input, residual, residual_out, weight, eps)
            if residual is not None:
                output = torch.empty_like(x)
                residual_out = torch.empty_like(x)
                if post_residual_addition is not None:
                    residual = residual + post_residual_addition
                fused_add_rms_norm(
                    output, x, residual, residual_out, w, self.variance_epsilon
                )
                return output, residual_out
            return rms_norm(x, w, self.variance_epsilon)
        else:
            # vllm API: rms_norm(out, input, weight, eps) -> None (in-place)
            #           fused_add_rms_norm(out, input, residual_out, residual, weight, eps)
            if not x.is_contiguous():
                x = x.contiguous()
            if residual is not None:
                out = torch.empty_like(x)
                residual_out = torch.empty_like(x)
                if post_residual_addition is not None:
                    residual = residual + post_residual_addition
                fused_add_rms_norm(
                    out, x, residual_out, residual, w, self.variance_epsilon
                )
                return out, residual_out
            out = torch.empty_like(x)
            rms_norm(out, x, w, self.variance_epsilon)
            return out

    def forward_cpu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if _is_cpu_amx_available:
            if residual is not None:
                if post_residual_addition is not None:
                    residual = residual + post_residual_addition
                torch.ops.sgl_kernel.gemma_fused_add_rmsnorm_cpu(
                    x, residual, self.weight.data, self.variance_epsilon
                )
                return x, residual
            return torch.ops.sgl_kernel.gemma_rmsnorm_cpu(
                x, self.weight.data, self.variance_epsilon
            )
        return self.forward_native(x, residual, post_residual_addition)

    def forward_npu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if envs.SGLANG_NPU_FORWARD_NATIVE_GEMMA_RMS_NORM.get():
            return self.forward_native(x, residual)
        if residual is not None:
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            norm_out, residual = add_gemma_rms_norm(
                x, self.weight, residual, self.variance_epsilon
            )
            return norm_out, residual

        x, _ = torch_npu.npu_gemma_rms_norm(x, self.weight, self.variance_epsilon)
        return x

    def forward_xpu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self._forward_impl(x, residual, post_residual_addition)

    def forward_with_allreduce_fusion(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
        use_attn_tp_group: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward with allreduce fusion; uses 1 + weight for fused kernels."""
        return _forward_with_allreduce_fusion(
            self,
            x,
            residual,
            post_residual_addition,
            self.gemma_weight,
            use_attn_tp_group=True,
        )


class Gemma3RMSNorm(MultiPlatformOp):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))
        # Re-dispatch

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward_native(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma3 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def forward_cpu(self, x):
        if _is_cpu_amx_available and x.stride(-1) == 1:
            return torch.ops.sgl_kernel.gemma3_rmsnorm_cpu(x, self.weight, self.eps)
        return self.forward_native(x)

    def forward_cuda(self, x):
        return self.forward_native(x)

    def forward_npu(self, x):
        output, _ = torch_npu.npu_gemma_rms_norm(x, self.weight, self.eps)
        return output

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class Gemma4RMSNorm(MultiPlatformOp):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        scale_shift: float = 0.0,
        with_scale: bool = True,
    ):
        super().__init__()
        self.with_scale = with_scale
        if self.with_scale:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_buffer("weight", torch.ones(dim), persistent=False)
        self.eps = eps
        self.scale_shift = scale_shift

    def __repr__(self):
        dim = self.weight.shape[0]
        return (
            f"{self.__class__.__name__}(dim={dim}, eps={self.eps}, "
            f"with_scale={self.with_scale}, scale_shift={self.scale_shift})"
        )

    def _norm(self, x):
        mean_squared = x.pow(2).mean(-1, keepdim=True) + self.eps
        return x * torch.pow(mean_squared, -0.5)

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        normed_output = self._norm(x.float())
        if self.with_scale:
            normed_output = normed_output * (self.weight.float() + self.scale_shift)
        return normed_output.type_as(x)

    def forward_cpu(self, x: torch.Tensor) -> torch.Tensor:
        if _is_cpu_amx_available:
            return torch.ops.sgl_kernel.gemma4_rmsnorm_cpu(
                x, self.weight.data, self.eps, self.scale_shift, self.with_scale
            )
        return self.forward_native(x)

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        needs_reshape = x.dim() != 2
        if needs_reshape:
            original_shape = x.shape
            x = x.contiguous().reshape(-1, original_shape[-1])
        if self.with_scale and self.scale_shift == 1.0:
            # gemma_rmsnorm: norm(x) * (1 + weight)
            out = gemma_rmsnorm(x, self.weight.data, self.eps)
        else:
            # rmsnorm: norm(x) * weight
            # with_scale=False -> weight is ones -> norm(x) * 1 = norm(x)
            # scale_shift=0.0 -> standard RMSNorm without +1 shift
            out = rmsnorm(x, self.weight.data, self.eps)
        if needs_reshape:
            out = out.reshape(original_shape)
        return out

    def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        if self.with_scale and self.scale_shift == 1.0:
            out = gemma_rmsnorm(x, self.weight.data, self.eps)
        else:
            out = rmsnorm(x, self.weight.data, self.eps)
        return out

    def forward_hip(self, x: torch.Tensor) -> torch.Tensor:
        # sgl_kernel's gemma_rmsnorm is not available on ROCm;
        # delegate to the pure-PyTorch implementation.
        return self.forward_native(x)


def _maybe_contiguous(x):
    return x if x is None or x.is_contiguous() else x.contiguous()


if _is_triton_available:

    @triton.jit
    def _silu(x):
        return x * tl.sigmoid(x)

    @triton.jit
    def _zc_rmsnorm_fwd_kernel(
        x_ptr,
        norm_w_ptr,
        xnorm_ptr,
        M,
        D,
        stride_xm,
        stride_xn,
        stride_xnorm_m,
        stride_xnorm_n,
        eps,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        mask_m = offs_m < M

        sumsq = tl.zeros((BLOCK_M,), dtype=tl.float32)

        for d0 in range(0, D, BLOCK_D):
            cols = d0 + offs_d
            mask = mask_m[:, None] & (cols[None, :] < D)

            x = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + cols[None, :] * stride_xn,
                mask=mask,
                other=0.0,
            ).to(tl.float32)

            sumsq += tl.sum(x * x, axis=1)

        rstd = tl.rsqrt(sumsq / D + eps)

        for d0 in range(0, D, BLOCK_D):
            cols = d0 + offs_d
            mask = mask_m[:, None] & (cols[None, :] < D)

            x = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + cols[None, :] * stride_xn,
                mask=mask,
                other=0.0,
            ).to(tl.float32)

            norm_w = tl.load(
                norm_w_ptr + cols,
                mask=cols < D,
                other=0.0,
            ).to(tl.float32)

            x_norm = x * rstd[:, None] * (norm_w[None, :] + 1.0)

            tl.store(
                xnorm_ptr + offs_m[:, None] * stride_xnorm_m + cols[None, :] * stride_xnorm_n,
                x_norm.to(xnorm_ptr.dtype.element_ty),
                mask=mask,
            )

    def _cfg_zc_rmsnorm_fwd(d: int):
        block_m = 2
        block_d = min(triton.next_power_of_2(d), 4096)
        num_warps = 4
        num_stages = 4
        return block_m, block_d, num_warps, num_stages

    def zc_rmsnorm_fwd(
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ):
        assert x.is_cuda and weight.is_cuda
        x = _maybe_contiguous(x)
        weight = _maybe_contiguous(weight)

        assert x.ndim == 2 and weight.ndim == 1
        assert x.shape[1] == weight.shape[0]
        assert x.stride(-1) == 1
        assert weight.stride(-1) == 1

        m, d = x.shape
        x_norm = torch.empty_like(x)

        block_m, block_d, num_warps, num_stages = _cfg_zc_rmsnorm_fwd(d)
        grid = (triton.cdiv(m, block_m),)

        _zc_rmsnorm_fwd_kernel[grid](
            x,
            weight,
            x_norm,
            m,
            d,
            x.stride(0),
            x.stride(1),
            x_norm.stride(0),
            x_norm.stride(1),
            eps,
            BLOCK_M=block_m,
            BLOCK_D=block_d,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return x_norm

    def zero_centered_gated_norm_fwd(
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ):
        start_shape = x.shape
        x2 = x.reshape(-1, x.shape[-1])
        x_norm = zc_rmsnorm_fwd(x2, weight, eps)
        return x_norm.view(start_shape)

    @triton.jit
    def _fused_silu_sigmoid_forward_small_h_kernel(
        out_ptr,
        x_ptr,
        w1_ptr,
        w2_ptr,
        token_size,
        inp_h_size,
        out_h_size_1,
        out_h_size_2,
        stride_om,
        stride_on,
        stride_xm,
        stride_xk,
        stride_w1h,
        stride_w1k,
        stride_w2n,
        stride_w2h,
        layernorm_gating_weight,
        BLOCK_TOKEN_SIZE: tl.constexpr,
        BLOCK_INP_H_SIZE: tl.constexpr,
        BLOCK_OUT_H_SIZE_2: tl.constexpr,
        BLOCK_OUT_H_SIZE_1: tl.constexpr,
    ):
        pid_token = tl.program_id(axis=0)
        offs_token = pid_token * BLOCK_TOKEN_SIZE + tl.arange(0, BLOCK_TOKEN_SIZE)
        offs_h_size_1 = tl.arange(0, BLOCK_OUT_H_SIZE_1)

        acc1 = tl.zeros((BLOCK_TOKEN_SIZE, BLOCK_OUT_H_SIZE_1), dtype=tl.float32)

        for k0 in range(0, inp_h_size, BLOCK_INP_H_SIZE):
            offs_inp_size = k0 + tl.arange(0, BLOCK_INP_H_SIZE)

            x_ptrs = x_ptr + offs_token[:, None] * stride_xm + offs_inp_size[None, :] * stride_xk
            w1_ptrs = w1_ptr + offs_h_size_1[None, :] * stride_w1h + offs_inp_size[:, None] * stride_w1k

            x = tl.load(
                x_ptrs,
                mask=(offs_token[:, None] < token_size) & (offs_inp_size[None, :] < inp_h_size),
                other=0.0,
            )
            w1 = tl.load(
                w1_ptrs,
                mask=(offs_h_size_1[None, :] < out_h_size_1) & (offs_inp_size[:, None] < inp_h_size),
                other=0.0,
            )

            acc1 = tl.dot(x, w1, acc1)

        hid = _silu(acc1).to(x_ptr.dtype.element_ty)

        num_pid_n = tl.cdiv(out_h_size_2, BLOCK_OUT_H_SIZE_2)
        for pid_n in range(0, num_pid_n):
            offs_h_size_2 = pid_n * BLOCK_OUT_H_SIZE_2 + tl.arange(0, BLOCK_OUT_H_SIZE_2)

            w2_ptrs = w2_ptr + offs_h_size_2[None, :] * stride_w2n + offs_h_size_1[:, None] * stride_w2h
            w2 = tl.load(
                w2_ptrs,
                mask=(offs_h_size_2[None, :] < out_h_size_2) & (offs_h_size_1[:, None] < out_h_size_1),
                other=0.0,
            )

            pre2 = tl.dot(hid, w2)
            gate = tl.sigmoid(pre2.to(tl.float32)).to(x_ptr.dtype.element_ty)

            x_gate_ptrs = x_ptr + offs_token[:, None] * stride_xm + offs_h_size_2[None, :] * stride_xk
            x_gate = tl.load(
                x_gate_ptrs,
                mask=(offs_token[:, None] < token_size) & (offs_h_size_2[None, :] < inp_h_size),
                other=0.0,
            )

            out = layernorm_gating_weight * x_gate * gate
            out_ptrs = out_ptr + offs_token[:, None] * stride_om + offs_h_size_2[None, :] * stride_on

            tl.store(
                out_ptrs,
                out.to(out_ptr.dtype.element_ty),
                mask=(offs_token[:, None] < token_size) & (offs_h_size_2[None, :] < out_h_size_2),
            )

    def _cfg_fused_forward():
        block_m = 16
        block_k = 128
        block_n = 128
        num_warps = 4
        num_stages = 3
        return block_m, block_k, block_n, num_warps, num_stages

    def triton_fused_linear_silu_linear_sigmoid(
        x: torch.Tensor,
        weight1: torch.Tensor,
        weight2: torch.Tensor,
        layernorm_gating_weight: float = 2.0,
    ):
        assert x.is_cuda and weight1.is_cuda and weight2.is_cuda
        assert x.dtype == weight1.dtype == weight2.dtype
        assert x.dtype in (torch.float16, torch.bfloat16, torch.float32)

        assert weight1.ndim == 2 and weight2.ndim == 2
        assert weight1.shape[0] in (16, 32)
        assert weight2.shape[1] == weight1.shape[0]
        assert x.shape[-1] == weight1.shape[1]
        assert weight2.shape[0] == x.shape[-1], (
            "weight2.shape[0] must equal x.shape[-1] because the kernel computes "
            "layernorm_gating_weight * x * sigmoid(...) elementwise."
        )

        batch_shape, inp_h_size = x.shape[:-1], x.shape[-1]
        token_size = batch_shape.numel()
        out_h_size_1 = weight1.shape[0]
        out_h_size_2 = weight2.shape[0]

        x2 = x.reshape(token_size, inp_h_size)
        w1 = _maybe_contiguous(weight1)
        w2 = _maybe_contiguous(weight2)

        output = torch.empty((token_size, out_h_size_2), device=x.device, dtype=x.dtype)

        (
            block_token_size,
            block_inp_h_size,
            block_out_h_size_2,
            num_warps,
            num_stages,
        ) = _cfg_fused_forward()

        grid = (triton.cdiv(token_size, block_token_size),)

        _fused_silu_sigmoid_forward_small_h_kernel[grid](
            output,
            x2,
            w1,
            w2,
            token_size,
            inp_h_size,
            out_h_size_1,
            out_h_size_2,
            output.stride(0),
            output.stride(1),
            x2.stride(0),
            x2.stride(1),
            w1.stride(0),
            w1.stride(1),
            w2.stride(0),
            w2.stride(1),
            layernorm_gating_weight,
            BLOCK_TOKEN_SIZE=block_token_size,
            BLOCK_INP_H_SIZE=block_inp_h_size,
            BLOCK_OUT_H_SIZE_1=out_h_size_1,
            BLOCK_OUT_H_SIZE_2=block_out_h_size_2,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        return output.reshape(*batch_shape, out_h_size_2)

    def triton_zero_centered_gated_norm(
        x: torch.Tensor,
        weight: torch.Tensor,
        gate_up_weight: torch.Tensor,
        gate_down_weight: torch.Tensor,
        eps: float,
        layernorm_gating_weight: float = 2.0,
    ):
        x_norm = zero_centered_gated_norm_fwd(x, weight, eps)
        return triton_fused_linear_silu_linear_sigmoid(
            x_norm,
            gate_up_weight,
            gate_down_weight,
            layernorm_gating_weight,
        )

    # --- Fully fused GigaChat3.5 gated-norm pipeline ------------------------------
    # Three stages: Triton gate_up -> sgl_kernel_ext split-K reduce -> Triton
    # gate_down. The RMSNorm scalar factors out of the gate_up GEMM because
    # u_h = x_h * (1 + weight_h) is rms-independent, so gate_up accumulates the
    # split-K partials A_k = sum_h W_up[k,h] * u_h and S = sum_h x_h^2 without
    # materializing the normalized activation; the reduce applies
    # rms = rsqrt(S/H + eps) and the gate prologue g = silu(rms * A).
    from triton.language.extra.cuda import gdc_wait

    from sglang.jit_kernel.utils import is_arch_support_pdl

    # H is split into CHUNK_H-wide pieces; NSPLIT = H // CHUNK_H programs, one
    # 128-column chunk each. NSPLIT is derived per hidden size, not hardcoded.
    _GATED_NORM_CHUNK_H = 128
    _GATED_NORM_BLOCK_B = 32

    @triton.jit
    def _gated_norm_gate_up_kernel(
        x_ptr, res_ptr, w_ptr, w_up_ptr, pA_ptr, pS_ptr, xsum_ptr, B, stride_xb,
        H: tl.constexpr, NSPLIT: tl.constexpr, BOTTLE: tl.constexpr,
        BLOCK_B: tl.constexpr, BLOCK_H: tl.constexpr,
        HAS_RESIDUAL: tl.constexpr, USE_PDL: tl.constexpr,
        ZERO_CENTERED: tl.constexpr = True,
    ):
        ps = tl.program_id(0)
        pj = tl.program_id(1)
        b = pj * BLOCK_B + tl.arange(0, BLOCK_B)
        k = tl.arange(0, BOTTLE)
        b_mask = b < B
        if USE_PDL:
            gdc_wait()
        chunk: tl.constexpr = H // NSPLIT
        h_start = ps * chunk
        A = tl.zeros([BLOCK_B, BOTTLE], tl.float32)
        S = tl.zeros([BLOCK_B], tl.float32)
        for h0 in range(0, chunk, BLOCK_H):
            h = h_start + h0 + tl.arange(0, BLOCK_H)
            # x may be a strided view (row stride != H, e.g. a q_a/kv_a slice);
            # its last dim is contiguous so the h-vector load stays coalesced.
            xt = tl.load(x_ptr + b[:, None] * stride_xb + h[None, :],
                         mask=b_mask[:, None], other=0.0).to(tl.float32)
            if HAS_RESIDUAL:
                # Fuse the residual add: x_sum = x + residual. residual / x_sum are
                # contiguous (row stride == H). This split-K program owns h-chunk
                # [ps*chunk, ...) x batch-block pj, so every (b, h) is written once.
                hs_offs = b[:, None] * H + h[None, :]
                xt += tl.load(res_ptr + hs_offs, mask=b_mask[:, None], other=0.0).to(tl.float32)
                # Normalize the stream-dtype sum, not the fp32 one: it is what the new residual holds and the trainer sees.
                xs = xt.to(xsum_ptr.dtype.element_ty)
                tl.store(xsum_ptr + hs_offs, xs, mask=b_mask[:, None])
                xt = xs.to(tl.float32)
            wt = tl.load(w_ptr + h).to(tl.float32)
            if ZERO_CENTERED:
                wt = 1.0 + wt
            u = (xt * wt[None, :]).to(tl.bfloat16)
            w_up = tl.load(w_up_ptr + k[:, None] * H + h[None, :])
            A += tl.dot(u, tl.trans(w_up))
            S += tl.sum(xt * xt, axis=1)
        tl.store(pA_ptr + (ps * B + b[:, None]) * BOTTLE + k[None, :], A, mask=b_mask[:, None])
        tl.store(pS_ptr + ps * B + b, S, mask=b_mask)

    @triton.jit
    def _gated_norm_gate_down_kernel(
        x_ptr, w_ptr, w_down_ptr, g_ptr, rms_ptr, out_ptr, GATING_WEIGHT: tl.constexpr, B, stride_xb,
        H: tl.constexpr, GD: tl.constexpr, BOTTLE: tl.constexpr,
        BLOCK_B: tl.constexpr, BLOCK_H: tl.constexpr, USE_PDL: tl.constexpr,
        ZERO_CENTERED: tl.constexpr = True,
    ):
        pd = tl.program_id(0)
        pj = tl.program_id(1)
        b = pj * BLOCK_B + tl.arange(0, BLOCK_B)
        k = tl.arange(0, BOTTLE)
        b_mask = b < B
        if USE_PDL:
            gdc_wait()
        g = tl.load(g_ptr + b[:, None] * BOTTLE + k[None, :], mask=b_mask[:, None], other=0.0)
        rms = tl.load(rms_ptr + b, mask=b_mask, other=0.0)
        chunk: tl.constexpr = H // GD
        h_start = pd * chunk
        for h0 in range(0, chunk, BLOCK_H):
            h = h_start + h0 + tl.arange(0, BLOCK_H)
            w_down = tl.load(w_down_ptr + h[:, None] * BOTTLE + k[None, :])
            raw = tl.dot(g, tl.trans(w_down))
            xt = tl.load(x_ptr + b[:, None] * stride_xb + h[None, :], mask=b_mask[:, None], other=0.0).to(tl.float32)
            wt = tl.load(w_ptr + h).to(tl.float32)
            if ZERO_CENTERED:
                wt = 1.0 + wt
            y = xt * wt[None, :] * rms[:, None]
            out = y * (GATING_WEIGHT * tl.sigmoid(raw))
            tl.store(out_ptr + b[:, None] * H + h[None, :], out.to(out_ptr.dtype.element_ty), mask=b_mask[:, None])

    def _gigachat35_fused_gated_norm(
        x, residual, post_residual_addition, weight, w_up, w_down, eps, gating_weight,
        zero_centered: bool = True,
    ):
        """Fused RMSNorm + gate. Returns the gated output, or ``(out, residual)`` when
        a residual is supplied (residual is updated in place, matching the canonical
        fused_add_rmsnorm contract).

        ``zero_centered`` selects the norm scale: ``1 + weight`` (ZeroCenteredGatedNorm)
        or ``weight`` (GatedNorm). It is a constexpr of both Triton stages (gate_up
        operand ``u = bf16(scale * x)``, gate_down output ``y = x * scale * rms``); the
        reduce stage never sees the weight."""
        if x.numel() == 0:
            # 0-row input (e.g. DP-attn idle rank): nothing to compute; mirror the
            # base norm's empty-input contract.
            return (x, residual) if residual is not None else x
        assert x.dim() == 2, "gigachat35 fused gated-norm expects a 2D [tokens, H] input"
        H = x.shape[1]
        # One split-K program per CHUNK_H-wide piece of H, so H must divide evenly.
        assert (
            H % _GATED_NORM_CHUNK_H == 0
        ), f"gigachat35 fused gated-norm needs H % {_GATED_NORM_CHUNK_H} == 0, got H={H}"
        NSPLIT = H // _GATED_NORM_CHUNK_H
        # The reduce launches ceil(NSPLIT/8) warps per token; that block
        # (ceil(NSPLIT/8) * 32 threads) must fit within 1024 -> NSPLIT <= 256.
        assert NSPLIT <= 256, (
            f"gigachat35 fused gated-norm needs NSPLIT <= 256 (reduce block <= 1024), "
            f"got NSPLIT={NSPLIT} (H={H})"
        )

        import sglang.sgl_kernel_ext  # noqa: F401  (registers torch.ops.sgl_kernel_ext.*)

        has_residual = residual is not None
        # gate_up/gate_down index x by row and pass its row stride, so a strided
        # view is fine (e.g. q_a/kv_a_layernorm receive a .split() slice of the
        # fused QKV projection). Only the last (H) dim must be unit-stride for the
        # coalesced h-vector loads.
        assert x.stride(-1) == 1, "gigachat35 fused gated-norm requires x with a unit-stride last dim"
        if has_residual:
            assert (
                residual.is_contiguous()
            ), "gigachat35 fused gated-norm requires contiguous residual"
            # Fold post_residual_addition into the residual up front, exactly like
            # the other norms (residual = residual + post); gate_up then fuses only
            # the residual add.
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            # gate_up reads x (+ residual) and writes the sum back INTO `residual`
            # in place, matching the canonical fused_add_rmsnorm contract -- return
            # the same residual tensor, now holding x + residual. No separate sum
            # buffer is allocated; gate_down re-reads it as the norm input. Safe in
            # place: each split-K program reads its own residual tile before
            # overwriting it.
            x_read = x
            res_t = residual
            x_sum = residual  # in-place accumulator == returned new residual
        else:
            x_read = x
            res_t = x
            x_sum = x

        B = x_sum.shape[0]
        pA = torch.empty(NSPLIT, B, 16, device=x_sum.device, dtype=torch.float32)
        pS = torch.empty(NSPLIT, B, device=x_sum.device, dtype=torch.float32)
        out = torch.empty(B, H, device=x_sum.device, dtype=x_sum.dtype)

        pdl = is_arch_support_pdl()
        grid = (NSPLIT, triton.cdiv(B, _GATED_NORM_BLOCK_B))
        _gated_norm_gate_up_kernel[grid](
            x_read, res_t, weight, w_up, pA, pS, x_sum, B, x_read.stride(0),
            H=H, NSPLIT=NSPLIT, BOTTLE=16,
            BLOCK_B=_GATED_NORM_BLOCK_B, BLOCK_H=_GATED_NORM_CHUNK_H,
            HAS_RESIDUAL=has_residual, USE_PDL=pdl, ZERO_CENTERED=zero_centered,
            num_warps=4, launch_pdl=pdl,
        )
        g, rms = torch.ops.sgl_kernel_ext.gigachat35_gated_norm_reduce(pA, pS, H, eps)
        _gated_norm_gate_down_kernel[grid](
            x_sum, weight, w_down, g, rms, out, gating_weight, B, x_sum.stride(0),
            H=H, GD=NSPLIT, BOTTLE=16,
            BLOCK_B=_GATED_NORM_BLOCK_B, BLOCK_H=_GATED_NORM_CHUNK_H, USE_PDL=pdl,
            ZERO_CENTERED=zero_centered, num_warps=4, launch_pdl=pdl,
        )
        return (out, x_sum) if has_residual else out

# default mode (no max-autotune): with mark_dynamic this compiles ONCE, fast; the
# gated norm is <2% of compute so exhaustive autotune is pure capture-time overhead
# for inference. Heavy ops (MLA, MoE GEMMs) tune via sgl_kernel/triton independently.
@torch.compile(dynamic=True)
def _compiled_gated_norm_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_down_weight: torch.Tensor,
    eps: float,
    layernorm_gating_weight: float,
    *,
    zero_centered: bool = True,
):
    output = x.float()
    output = output * torch.rsqrt(output.pow(2).mean(-1, keepdim=True) + eps)
    if zero_centered:
        output = (output * (1.0 + weight.float())).to(dtype=x.dtype)
    else:
        output = (output * weight.float()).to(dtype=x.dtype)
    normed = output.float()
    gate_hidden = F.linear(normed, gate_up_weight.float())
    gate_hidden = F.silu(gate_hidden)
    gate_hidden = F.linear(gate_hidden, gate_down_weight.float())
    gate = torch.sigmoid(gate_hidden)
    return (normed * layernorm_gating_weight * gate).to(dtype=x.dtype)

_compiled_zero_centered_gated_norm = functools.partial(
    _compiled_gated_norm_impl, zero_centered=True
)
_compiled_gated_norm = functools.partial(
    _compiled_gated_norm_impl, zero_centered=False
)
