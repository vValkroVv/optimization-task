import enum
import functools
import typing as tp

import torch

aten = torch.ops.aten


class InputLayoutType(enum.IntEnum):
    ROWWISE = 0
    COLUMNWISE = 1


# Returned by quantize_* / scaling_aware_fp8_transpose when pack=False.
Float8OperandTuple = tuple[
    torch.Tensor,
    torch.Tensor,
    tp.Union[InputLayoutType, int],
    int,
]


@functools.lru_cache(maxsize=None)
def _empty_tensor() -> torch.Tensor:
    """Get tensor with no entries and no data"""
    return torch.Tensor().cuda()


def is_fp8_operand_tuple(tensor: object) -> bool:
    """True for the unpacked ``(data, scale, layout, block_scaling_dim)`` form."""
    return isinstance(tensor, tuple) and len(tensor) == 4


def is_quantized_tensor(tensor: object) -> bool:
    """True for the packed wrapper form.

    Lives here rather than in utils/fp8_utils.py so that this module mirrors
    ``cudacherry.tensor.fp8_tensor``.

    THAT MIRRORING IS NOT ENFORCED BY ANYTHING. The two modules are deliberate
    duplicates -- cudacherry ships its own copy so it can be used in projects
    that have no llm-foundry -- and they are collapsed into one class at runtime
    by ``CUDACHERRY_FP8_TENSOR_MODULE`` (see ops/float8/_cudacherry.py). That
    only stays correct while the two class bodies agree, and the check that
    compared them was removed. If you edit Float8BlockwiseQTensor, its
    ``__slots__``, or the values of InputLayoutType, port the change to
    cudacherry in the same breath. The enum values in particular cross into C++
    as plain ints: renumbering them on one side swaps rowwise and columnwise
    with no error anywhere.
    """
    return isinstance(tensor, Float8BlockwiseQTensor)


def clear_tensor_data(*tensors: object) -> None:
    """
    Trick to deallocate tensor memory when delete operation does not
    release the tensor due to PyTorch override.

    Must be used carefully.
    """
    for t in tensors:
        if t is not None:
            if isinstance(t, Float8BlockwiseQTensor):
                t.clear()
            elif isinstance(t, tuple):
                t[0].data = _empty_tensor()
                t[1].data = _empty_tensor()
            else:
                t.data = _empty_tensor()
            del t


class Float8BlockwiseQTensor(torch.Tensor):
    _data: torch.Tensor
    _scale: torch.Tensor
    _orig_dtype: torch.dtype
    _fp8_dtype: torch.dtype  # dtype of _data; _orig_dtype is what it dequantizes to
    orig_shape: tuple
    _input_layout_type: InputLayoutType  # rowwise or columnwise
    _block_scaling_dim: int  # 1 for tile, 2 for block

    __slots__ = [
        "_data",
        "_scale",
        "_orig_dtype",
        "_fp8_dtype",
        "orig_shape",
        "_input_layout_type",
        "_block_scaling_dim",
    ]

    def __new__(
        cls,
        data: torch.Tensor,
        scale: torch.Tensor,
        orig_dtype: torch.dtype,
        input_layout_type: InputLayoutType = InputLayoutType.ROWWISE,
        block_scaling_dim: int = 1,
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

        return self

    def __repr__(self):
        return (
            "Float8BlockwiseQTensor("
            f"fp8_dtype={self._fp8_dtype}, "
            f"data_shape={tuple(self.shape)}, "
            f"orig_shape={tuple(self.orig_shape)}, "
            f"scale_shape={tuple(self._scale.shape)}, "
            f"input_layout_type={self._input_layout_type}, "
            f"block_scaling_dim={self._block_scaling_dim}"
            ")"
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
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
                )
                viewed.orig_shape = tensor.orig_shape
                return viewed
            else:
                raise NotImplementedError(
                    f"Float8BlockwiseQTensor does not support view operation with new shape {new_shape}"
                )

        elif func == torch.ops.aten.detach.default:
            tensor = args[0]
            detached = cls(
                tensor._data,
                tensor._scale,
                tensor._orig_dtype,
                tensor._input_layout_type,
                tensor._block_scaling_dim,
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

            kwargs = kwargs or {}
            dtype = kwargs.get("dtype") or tensor._orig_dtype
            return torch.zeros((), device=tensor._data.device, dtype=dtype)

        else:
            raise NotImplementedError(
                f"Float8BlockwiseQTensor does not support {func} operation"
            )

    def metadata_copy(self) -> "Float8BlockwiseQTensor":
        # Keep only wrapper metadata here; ctx.save_for_backward owns storage lifetime.
        copied = type(self)(
            torch.empty((0,), device=self._data.device, dtype=self._data.dtype),
            torch.empty((0,), device=self._scale.device, dtype=self._scale.dtype),
            self._orig_dtype,
            self._input_layout_type,
            self._block_scaling_dim,
        )
        copied.orig_shape = self.orig_shape
        return copied

    def prepare_for_saving(
        self,
    ) -> tuple[list[torch.Tensor], "Float8BlockwiseQTensor"]:
        """
        Prepare the tensor base for saving for backward
        """
        tensors = [
            self._data,
            self._scale,
        ]
        return tensors, self.metadata_copy()

    def restore_from_saved(self, tensors: list[tp.Optional[torch.Tensor]]):
        self._data = tensors[0]
        self._scale = tensors[1]
        return tensors[2:]

    def clear(self):
        self._data.data = _empty_tensor()
        self._scale.data = _empty_tensor()
