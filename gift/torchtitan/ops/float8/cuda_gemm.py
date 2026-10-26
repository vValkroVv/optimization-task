from typing import TypeAlias

import torch
from torch import nn

from torchtitan.config.schema.ops.float8.cuda_gemm import FP8LinearConfig
from torchtitan.protocols.module import Module

from .fp8_tensor import (
    Float8BlockwiseQTensor,
    Float8BlockwiseQTensorMetadata,
    Float8OperandTuple,
    clear_tensor_data,
    is_fp8_operand_tuple,
    is_quantized_tensor,
)
from .fp8_utils import (
    pad_tensor_if_needed,
    unpad_tensor_if_needed,
)

GemmOperand = Float8BlockwiseQTensor | Float8OperandTuple


def prepare_for_saving(
    *tensors: torch.Tensor | Float8BlockwiseQTensor | Float8OperandTuple,
):
    tensor_list, tensor_objects_list = [], []
    for tensor in tensors:
        if isinstance(tensor, Float8BlockwiseQTensor):
            t, t_obj = tensor.prepare_for_saving()
            tensor_list.extend(t)
            tensor_objects_list.append(t_obj)
        elif is_fp8_operand_tuple(tensor):
            tensor_list.extend(tensor[:2])
            tensor_objects_list.append(tensor[2:])
        else:
            tensor_list.append(tensor)
            tensor_objects_list.append(None)
    return tensor_list, tensor_objects_list


def restore_from_saved(
    tensor_objects: list[Float8BlockwiseQTensorMetadata | tuple[int, int] | None],
    saved_tensors: list[torch.Tensor | None],
):
    restored = []
    for obj in tensor_objects:
        if isinstance(obj, Float8BlockwiseQTensorMetadata):
            restored.append(obj.restore(saved_tensors[0], saved_tensors[1]))
            saved_tensors = saved_tensors[2:]
        elif isinstance(obj, tuple) and len(obj) == 2:
            layout, block_scaling_dim = obj
            data, scale = saved_tensors[0], saved_tensors[1]
            restored.append((data, scale, layout, block_scaling_dim))
            saved_tensors = saved_tensors[2:]
        else:
            restored.append(saved_tensors[0])
            saved_tensors = saved_tensors[1:]
    return restored


@torch._dynamo.allow_in_graph
class FP8Gemm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor | Float8BlockwiseQTensor,
        weight: torch.Tensor,
        dtype: torch.dtype,
        act_requires_grad: bool,
        inp_orig_shape: tuple[int, ...],
        residual: torch.Tensor | None = None,
        is_recompute: bool = False,
        do_unpad_out: bool = True,
    ) -> torch.Tensor:
        """Forward autograd function

        Args:
            inp (torch.Tensor | Float8BlockwiseQTensor): input tensor
            weight (torch.Tensor): weight tensor
            dtype (torch.dtype): data type for GEMM output
            act_requires_grad (bool): requires grad for input tensor
            inp_orig_shape (tuple[int, ...]): unpadded shape of ``inp``; the
                output is unpadded to ``(*inp_orig_shape[:-1], out_features)``
            residual (torch.Tensor | None): residual connection tensor
            is_recompute (bool): requantize fp8 weights on backward
            do_unpad_out (bool): do unpad gemm output if needed
        Returns:
            Torch.Tensor: forward output tensor
        """
        from ._cudacherry import fp8_gemm, quantize_blk128x128, quantize_vec128

        ctx.inp_shape = tuple(inp_orig_shape)
        ctx.inp_cur_shape = inp.shape
        ctx.w_shape = weight.shape
        ctx.needs_wgrad = weight.requires_grad
        out_features, _in_features = ctx.w_shape
        ctx.out_shape = (*ctx.inp_shape[:-1], out_features)

        ctx.do_unpad_out = do_unpad_out
        ctx.has_residual = residual is not None

        # get quantized inputs
        inp_is_quantized = is_quantized_tensor(inp)
        if inp_is_quantized:
            inp_quantized = inp
        else:
            inp = pad_tensor_if_needed(inp)
            inp_quantized = quantize_vec128(inp, pack=True)

        # get quantized weights
        weight = pad_tensor_if_needed(weight)

        if not act_requires_grad:
            del inp

        weight_quantized: GemmOperand = quantize_blk128x128(weight, pack=False)

        # TODO (veveselov): diverge if fuse residual, skip now
        if ctx.has_residual:
            # Accumulate into the residual buffer: y = residual + x * w^T.
            # TODO: this mutates ``out`` through the cudacherry op; under
            # torch.compile that needs an out-variant schema on the op.
            gemm_out = (
                residual.contiguous() if not residual.is_contiguous() else residual
            )
            if is_recompute:
                gemm_out = gemm_out.clone()
            gemm_out = pad_tensor_if_needed(gemm_out)
        else:
            # Let the kernel allocate ``(*inp.shape[:-1], out_features)`` itself:
            # the same buffer as before, but the op stays functional (no ``out=``
            # mutation), which is what torch.compile requires.
            gemm_out = None

        # Forward GEMM
        # Note: y = x * w^T
        gemm_out = fp8_gemm(
            inp_quantized,
            weight_quantized,
            out=gemm_out,
            out_dtype=dtype,
            accumulate=ctx.has_residual,
            use_split_accumulator=True,
        )

        if ctx.do_unpad_out:
            gemm_out = unpad_tensor_if_needed(gemm_out, ctx.out_shape)

        if ctx.needs_wgrad:
            args = (inp_quantized, weight)
        else:
            args = (weight,)
        tensors_to_save, tensor_objects = prepare_for_saving(*args)
        ctx.save_for_backward(*tensors_to_save)
        ctx.tensor_objects = tensor_objects
        ctx.dtype = dtype
        ctx.act_requires_grad = act_requires_grad
        return gemm_out

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor | Float8BlockwiseQTensor
    ) -> tuple[torch.Tensor | None, ...]:
        """Backward autograd function

        Args:
            grad_output (torch.Tensor | Float8BlockwiseQTensor): gradients tensor
        Returns:
            tuple[torch.Tensor | None, ...]: gradients tensors for input,
                                                   weights and bias
        """
        from ._cudacherry import (
            fp8_gemm,
            quantize_blk128x128,
            quantize_vec128,
            scaling_aware_fp8_transpose,
        )

        needs_wgrad = ctx.needs_wgrad
        if needs_wgrad:
            inp_quantized, weight = restore_from_saved(
                ctx.tensor_objects, ctx.saved_tensors
            )
        else:
            (weight,) = restore_from_saved(ctx.tensor_objects, ctx.saved_tensors)
        ctx.tensor_objects = None

        weight = pad_tensor_if_needed(weight)
        weight_quantized = quantize_blk128x128(
            weight, transpose_output=True, pack=False
        )

        grad_is_quantized = is_quantized_tensor(grad_output)
        if grad_is_quantized:
            grad_output_quantized = grad_output
        else:
            grad_output = pad_tensor_if_needed(grad_output)
            grad_output_quantized = quantize_vec128(grad_output, pack=False)
        # dgrad GEMM
        # Note: dx = dy * w
        dgrad = None
        if ctx.act_requires_grad:
            dgrad = fp8_gemm(
                grad_output_quantized,
                weight_quantized,
                out_dtype=ctx.dtype,
                use_split_accumulator=True,
            )
            dgrad = unpad_tensor_if_needed(dgrad, ctx.inp_cur_shape)

        clear_tensor_data(weight_quantized)
        # wgrad GEMM (skipped entirely when weights are frozen)
        # Note: dw = dy^T * x
        wgrad = None
        if needs_wgrad:
            inp_quantized = scaling_aware_fp8_transpose(
                inp_quantized,
                pack=False,
            )
            grad_output_quantized = scaling_aware_fp8_transpose(
                grad_output_quantized,
                pack=False,
            )
            wgrad = fp8_gemm(
                inp_quantized,
                grad_output_quantized,
                out_dtype=ctx.dtype,
                use_split_accumulator=True,
            )
            wgrad = unpad_tensor_if_needed(wgrad, ctx.w_shape)
            clear_tensor_data(inp_quantized, grad_output_quantized)
        else:
            clear_tensor_data(grad_output_quantized)

        return (
            dgrad,
            wgrad,
            None,  # dtype
            None,  # act_requires_grad
            None,  # inp_orig_shape
            grad_output if ctx.has_residual else None,
            None,  # is_recompute
            None,  # do_unpad_out
        )


class FP8Linear(nn.Linear, Module):
    # pyrefly: ignore [bad-override]
    Config: TypeAlias = FP8LinearConfig

    def __init__(self, config: "FP8Linear.Config"):
        if config.bias:
            raise ValueError(
                "FP8Linear does not support bias; set bias=False on the "
                f"Linear config (in={config.in_features}, out={config.out_features})."
            )
        self.in_features, self.out_features = config.require_features()
        self._is_recompute = False
        super().__init__(
            self.in_features,
            self.out_features,
            False,  # bias
        )

    def set_is_recompute_state(self, v: bool):
        self._is_recompute = v

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}"

    def _apply_gemm(
        self,
        inp: torch.Tensor | Float8BlockwiseQTensor,
        weight: torch.Tensor,
        residual: torch.Tensor | None,
        is_recompute: bool,
        do_unpad_out: bool,
    ):
        """Apply GEMM operation

        Args:
            inp (torch.Tensor | Float8BlockwiseQTensor): input tensor
            weight (torch.Tensor): weight tensor
            residual (torch.Tensor | None): residual connection tensor
            is_recompute (bool): requantize fp8 weights on backward
            do_unpad_out (bool): do unpad gemm out if needed
        """
        act_dtype = inp.dtype if isinstance(inp, torch.Tensor) else torch.bfloat16
        if is_quantized_tensor(inp):
            act_requires_grad = True
        else:
            act_requires_grad = inp.requires_grad
        # Read here, not inside FP8Gemm.forward: under torch.compile this runs
        # while Dynamo traces, so an ``orig_shape`` a caller assigned on the
        # wrapper is still visible. FP8Gemm.forward runs later, at AOTAutograd
        # trace time, on a wrapper rebuilt from ``__tensor_flatten__``.
        orig_shape = getattr(inp, "orig_shape", inp.shape)

        return FP8Gemm.apply(
            inp,
            weight,
            act_dtype,
            act_requires_grad,
            tuple(orig_shape),
            residual,
            is_recompute,
            do_unpad_out,
        )

    def forward(
        self,
        input: torch.Tensor | Float8BlockwiseQTensor,  # noqa: A002
        residual: torch.Tensor | None = None,
        do_unpad_out: bool = True,
    ):
        assert self.weight is not None, "weights is not initialized."
        return self._apply_gemm(
            input, self.weight, residual, self._is_recompute, do_unpad_out
        )

    def quantize_input(self, inp: torch.Tensor) -> Float8BlockwiseQTensor:
        """Quantize input outside the GEMM

        Args:
            inp (torch.Tensor): input tensor

        Returns:
            Float8BlockwiseQTensor: quantized input tensor
        """
        from ._cudacherry import quantize_vec128

        padded_inp = pad_tensor_if_needed(inp)
        quantized_inp = quantize_vec128(padded_inp, pack=True)
        quantized_inp.orig_shape = inp.shape
        return quantized_inp

    @classmethod
    def swap_module(
        cls,
        module: nn.Linear,
        *,
        config: "FP8Linear.Config | None" = None,
    ) -> "FP8Linear":
        """Create an FP8 linear by reusing a bias-free linear weight.

        When ``config`` is provided, build through it so config-carried
        state such as initializers, sharding, and hidden-z-loss hooks is
        attached to the replacement module.
        """
        if module.bias is not None:
            raise ValueError("FP8Linear does not support bias")

        if config is None:
            config = cls.Config(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=False,
            )
        elif (config.in_features, config.out_features) != (
            module.in_features,
            module.out_features,
        ):
            raise ValueError(
                "FP8Linear config dimensions do not match the source module"
            )

        # Build on the source device, including meta, before reusing the
        # existing parameter. Config.build() also attaches config state.
        with torch.device(module.weight.device):
            new_module = config.build()
        new_module.train(module.training)

        # Keep the original parameter so conversion does not allocate or copy
        # the weight. This also handles meta-device modules.
        new_module._parameters["weight"] = module.weight
        return new_module
