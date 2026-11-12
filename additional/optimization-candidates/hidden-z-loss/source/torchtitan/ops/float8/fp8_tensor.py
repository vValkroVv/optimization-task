# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Blockwise-quantized float8 tensor wrapper.

``Float8BlockwiseQTensor`` packs DeepSeek-style blockwise fp8 quantized data
(1x128 rowwise activations / 128x128 blockwise weights) together with its
float32 scales. It is a ``_make_wrapper_subclass`` tensor that reports
``dtype=orig_dtype`` (e.g. bf16). That dtype report is load-bearing: it lets
the wrapper travel between ``torch.autograd.Function`` boundaries as a
regular-looking bf16 tensor, so autograd threads plain bf16 gradients through
fp8-carrying edges. Raw fp8-dtype tensors must never be differentiable edges —
autograd coerces incoming gradients to the edge's dtype, which would silently
quantize gradients to e4m3 without scales.

Only ``size`` / ``view`` / ``detach`` are supported through
``__torch_dispatch__``; any other op must unpack ``_data`` / ``_scale``
explicitly (fp8 grouped GEMM, DeepEP fp8 dispatch, permutation kernels).

torch.compile
-------------
The wrapper implements the traceable-wrapper-subclass protocol
(``__tensor_flatten__`` / ``__tensor_unflatten__``): Dynamo carries it as a
graph value and AOTAutograd desugars it into ``_data`` / ``_scale`` plus the
allocation-free ``Float8BlockwiseQTensorMetadata``. Without the protocol
Dynamo falls back to bytecode-tracing ``__new__`` and stops at
``_make_wrapper_subclass``, which it cannot trace. The autograd functions that
build or consume the wrapper (``FP8Gemm``, ``SharedActivationQuantize``,
cudacherry's ``SwiGLUQuantize``) are ``torch._dynamo.allow_in_graph``: their
Python runs once at AOTAutograd trace time on FakeTensors instead of being
bytecode-traced, which requires every kernel they launch to be a
``torch.ops.cudacherry.*`` custom op with a Meta kernel.

The wrapper must stay *inside* one compiled graph (per-block compilation
guarantees that for the dense path). As a graph output AOTAutograd would
assume its gradient is the same subclass, but the gradient of a quantized
activation is a plain ``orig_dtype`` tensor.

The wrapper is not a DTensor and carries no sharding information: create it
after a module boundary has redistributed its input (``Module._redistribute_inputs``
would treat it as a plain local tensor), i.e. inside ``forward`` rather than in
a pre-hook, and keep fp8 gradients (the fused SwiGLU backward) flowing straight
into ``FP8Gemm.backward`` with no ``Redistribute`` node in between.
"""

import enum
import functools
from dataclasses import dataclass

import torch

aten = torch.ops.aten


class InputLayoutType(enum.IntEnum):
    ROWWISE = 0
    COLUMNWISE = 1


# Unpacked ``(data, scale, layout, block_scaling_dim)`` form returned by
# quantization helpers that do not wrap their outputs.
Float8OperandTuple = tuple[
    torch.Tensor,
    torch.Tensor,
    InputLayoutType | int,
    int,
]


@dataclass(frozen=True, slots=True)
class Float8BlockwiseQTensorMetadata:
    """Allocation-free metadata needed to rebuild an fp8 wrapper.

    Also the ``__tensor_flatten__`` context: it must stay a plain, comparable
    value because Dynamo guards on it and AOTAutograd embeds it in the graph.
    """

    orig_dtype: torch.dtype
    orig_shape: tuple[int, ...]
    input_layout_type: InputLayoutType
    block_scaling_dim: int
    scale_block_first: bool

    def restore(
        self,
        data: torch.Tensor,
        scale: torch.Tensor,
    ) -> "Float8BlockwiseQTensor":
        restored = Float8BlockwiseQTensor(
            data,
            scale,
            self.orig_dtype,
            self.input_layout_type,
            self.block_scaling_dim,
            scale_block_first=self.scale_block_first,
        )
        restored.orig_shape = self.orig_shape
        return restored


@functools.cache
def _empty_tensor(device: torch.device) -> torch.Tensor:
    """Get a tensor with no entries and no data on ``device``."""
    return torch.empty(0, device=device)


def _release_storage(t: torch.Tensor) -> None:
    """Point ``t`` at an empty tensor so its storage can be freed."""
    t.data = _empty_tensor(t.device)


def is_fp8_operand_tuple(tensor: object) -> bool:
    """True for the unpacked ``(data, scale, layout, block_scaling_dim)`` form."""
    return isinstance(tensor, tuple) and len(tensor) == 4


def is_quantized_tensor(tensor: object) -> bool:
    """True for the packed ``Float8BlockwiseQTensor`` wrapper form."""
    return isinstance(tensor, Float8BlockwiseQTensor)


def clear_tensor_data(*tensors: object) -> None:
    """Deallocate tensor storage early by replacing ``.data`` with an empty tensor.

    Used to free large intermediates (e.g. quantized activations) before the
    end of their Python scope. Must be used carefully: any live alias keeps
    reading freed storage semantics.

    No-op under torch.compile: the compiled graph owns intermediate lifetimes
    and the ``.data`` swap is a Python-level trick tracing cannot see. Running
    it at trace time would also cache a FakeTensor in ``_empty_tensor``.
    """
    if torch.compiler.is_compiling():
        return
    for t in tensors:
        if t is not None:
            if isinstance(t, Float8BlockwiseQTensor):
                t.clear()
            elif isinstance(t, tuple):
                _release_storage(t[0])
                _release_storage(t[1])
            else:
                _release_storage(t)  # pyrefly: ignore [bad-argument-type]
            del t


class Float8BlockwiseQTensor(torch.Tensor):
    _data: torch.Tensor
    _scale: torch.Tensor
    _orig_dtype: torch.dtype
    _fp8_dtype: torch.dtype  # dtype of _data; _orig_dtype is what it dequantizes to
    orig_shape: tuple
    _input_layout_type: InputLayoutType  # rowwise or columnwise
    _block_scaling_dim: int  # 1 for tile (1x128), 2 for block (128x128)
    _scale_block_first: bool  # last two scale dimensions are (blocks, rows)

    __slots__ = [
        "_data",
        "_scale",
        "_orig_dtype",
        "_fp8_dtype",
        "orig_shape",
        "_input_layout_type",
        "_block_scaling_dim",
        "_scale_block_first",
    ]

    def __new__(
        cls,
        data: torch.Tensor,
        scale: torch.Tensor,
        orig_dtype: torch.dtype,
        input_layout_type: InputLayoutType = InputLayoutType.ROWWISE,
        block_scaling_dim: int = 1,
        *,
        scale_block_first: bool,
    ):
        assert orig_dtype.itemsize > 1, (
            "The bit depth for original dtype must be greater than 8 bits"
        )
        self = torch.Tensor._make_wrapper_subclass(
            cls,
            data.size(),
            strides=data.stride(),
            storage_offset=data.storage_offset(),
            dtype=orig_dtype,
            layout=data.layout,
            requires_grad=data.requires_grad,
            device=data.device,
        )

        self._data = data
        self._scale = scale
        self._orig_dtype = orig_dtype
        self.orig_shape = self.shape
        self._fp8_dtype = data.dtype
        self._input_layout_type = input_layout_type
        self._block_scaling_dim = block_scaling_dim
        self._scale_block_first = scale_block_first

        return self

    def __repr__(self):
        return (
            "Float8BlockwiseQTensor("
            f"fp8_dtype={self._fp8_dtype}, "
            f"data_shape={tuple(self.shape)}, "
            f"orig_shape={tuple(self.orig_shape)}, "
            f"scale_shape={tuple(self._scale.shape)}, "
            f"input_layout_type={self._input_layout_type}, "
            f"block_scaling_dim={self._block_scaling_dim}, "
            f"scale_block_first={self._scale_block_first}"
            ")"
        )

    # Traceable-wrapper-subclass protocol (torch.compile / AOTAutograd), and the
    # one place the wrapper is taken apart: ``prepare_for_saving`` derives from
    # it, so the two cannot disagree on what the metadata carries.
    def __tensor_flatten__(
        self,
    ) -> tuple[list[str], Float8BlockwiseQTensorMetadata]:
        return ["_data", "_scale"], Float8BlockwiseQTensorMetadata(
            orig_dtype=self._orig_dtype,
            orig_shape=tuple(self.orig_shape),
            input_layout_type=self._input_layout_type,
            block_scaling_dim=self._block_scaling_dim,
            scale_block_first=self._scale_block_first,
        )

    @staticmethod
    def __tensor_unflatten__(
        inner_tensors: dict[str, torch.Tensor],
        meta: Float8BlockwiseQTensorMetadata,
        outer_size,
        outer_stride,
    ) -> "Float8BlockwiseQTensor":
        # The wrapper mirrors ``_data``'s size/stride, so ``outer_size`` /
        # ``outer_stride`` carry no extra information here.
        return meta.restore(inner_tensors["_data"], inner_tensors["_scale"])

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func == aten.size.default:
            tensor = args[0]
            return tensor._data.size()

        elif func == aten.view.default:
            tensor, new_shape = args[0], args[1]
            if tuple(new_shape) == tuple(tensor._data.shape):
                new_data = aten.view.default(tensor._data, *args[1:], **kwargs)
                viewed = cls(
                    new_data,
                    tensor._scale,
                    tensor._orig_dtype,
                    tensor._input_layout_type,
                    tensor._block_scaling_dim,
                    scale_block_first=tensor._scale_block_first,
                )
                viewed.orig_shape = tensor.orig_shape
                return viewed
            else:
                raise NotImplementedError(
                    f"Float8BlockwiseQTensor does not support view operation "
                    f"with new shape {new_shape}"
                )

        elif func == aten.clone.default:
            # Dynamo's autograd.Function HOP calls ``.contiguous()`` on every incoming gradient.
            tensor = args[0]
            cloned = cls(
                aten.clone.default(tensor._data, **kwargs),
                aten.clone.default(tensor._scale, **kwargs),
                tensor._orig_dtype,
                tensor._input_layout_type,
                tensor._block_scaling_dim,
                scale_block_first=tensor._scale_block_first,
            )
            cloned.orig_shape = tensor.orig_shape
            return cloned

        elif func == aten.detach.default:
            tensor = args[0]
            detached = cls(
                tensor._data,
                tensor._scale,
                tensor._orig_dtype,
                tensor._input_layout_type,
                tensor._block_scaling_dim,
                scale_block_first=tensor._scale_block_first,
            )
            detached.orig_shape = tensor.orig_shape
            detached.requires_grad = False
            return detached
        elif func == aten.sum.default:
            tensor = args[0]
            if tensor._data.numel() != 0:
                raise NotImplementedError(
                    "Float8BlockwiseQTensor only supports sum for empty tensors"
                )
            dtype = kwargs.get("dtype") or tensor._orig_dtype
            return tensor._data.new_zeros((), dtype=dtype)
        else:
            raise NotImplementedError(
                f"Float8BlockwiseQTensor does not support {func} operation"
            )

    def prepare_for_saving(
        self,
    ) -> tuple[list[torch.Tensor], Float8BlockwiseQTensorMetadata]:
        """Split into raw tensors for ``save_for_backward`` and wrapper metadata.

        The tensors have to reach ``ctx.save_for_backward`` unwrapped so autograd
        can version-check them; the metadata rides on ``ctx`` and allocates
        nothing. Rebuild with :meth:`Float8BlockwiseQTensorMetadata.restore`.
        """
        attrs, meta = self.__tensor_flatten__()
        return [getattr(self, name) for name in attrs], meta

    def clear(self):
        """Release the underlying data and scale storage.

        No-op under torch.compile; see :func:`clear_tensor_data`.
        """
        if torch.compiler.is_compiling():
            return
        _release_storage(self._data)
        _release_storage(self._scale)


def bf16_grad_edge(rows: int, cols: int, *, device: torch.device) -> torch.Tensor:
    """An edge for an fp8 payload whose gradient is bf16."""
    return torch.zeros((), device=device, dtype=torch.bfloat16).expand(rows, cols)


def fp8_grad_edge(
    rows: int, cols: int, *, device: torch.device, block_size: int = 128
) -> "Float8BlockwiseQTensor":
    """An edge for a bf16 payload whose gradient is blockwise fp8."""
    data = torch.zeros((), device=device, dtype=torch.float8_e4m3fn).expand(rows, cols)
    scale = torch.zeros((), device=device, dtype=torch.float32).expand(
        rows, cols // block_size
    )
    return Float8BlockwiseQTensor(
        data,
        scale,
        torch.bfloat16,
        InputLayoutType.ROWWISE,
        1,
        scale_block_first=False,
    )
