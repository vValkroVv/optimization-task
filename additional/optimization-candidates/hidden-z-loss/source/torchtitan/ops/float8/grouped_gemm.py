import torch
from torch.autograd.profiler import record_function as record_fn
from torch.fx.experimental.symbolic_shapes import guard_or_false

from torchtitan.ops.float8.deep_gemm_handle import _DeepGemmHandle
from torchtitan.ops.float8.fp8_tensor import (
    Float8BlockwiseQTensor,
    clear_tensor_data,
    fp8_grad_edge,
)
from torchtitan.ops.float8.triton_kernels.block_transpose import (
    transpose_last_two_dims_fn,
)
from torchtitan.ops.float8.triton_kernels.row2col_deepgemm_quantization import (
    row2col_requantization_deepgemm_layout_fn,
)
from torchtitan.ops.float8.triton_kernels.row_quantization import (
    row_quantization_1x128_fn,
)

_cached_handle: _DeepGemmHandle | None = None


class _EmptyGroupGemm(torch.autograd.Function):
    """``_GroupedGemm``'s contract for a rank whose token groups are empty."""

    @staticmethod
    def forward(
        ctx,
        grad_edge: torch.Tensor | None,
        tensor_groups: torch.Tensor | Float8BlockwiseQTensor,
        weights: Float8BlockwiseQTensor,
        fp8_grad_output: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        ctx.use_edges = grad_edge is not None
        ctx.needs_wgrad = weights.requires_grad
        ctx.weight_shape = weights.shape
        ctx.device = weights.device
        ctx.inp_shape = (
            tensor_groups._data.shape
            if isinstance(tensor_groups, Float8BlockwiseQTensor)
            else tensor_groups.shape
        )
        out = torch.empty(
            (ctx.inp_shape[0], int(weights.shape[1])),
            dtype=torch.bfloat16,
            device=weights.device,
        )
        if not fp8_grad_output:
            return out
        ctx.mark_non_differentiable(out)
        return out, fp8_grad_edge(out.shape[0], out.shape[1], device=out.device)

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output, grad_fp8_edge=None):
        dgrad = torch.zeros(ctx.inp_shape, dtype=torch.bfloat16, device=ctx.device)
        wgrad = (
            torch.zeros(ctx.weight_shape, dtype=torch.bfloat16, device=ctx.device)
            if ctx.needs_wgrad
            else None
        )
        # Gradients for: grad_edge, tensor_groups, weights, fp8_grad_output
        return (
            dgrad if ctx.use_edges else None,
            None if ctx.use_edges else dgrad,
            wgrad,
            None,
        )


def _get_deep_gemm_handle(num_sms: int | None = None) -> _DeepGemmHandle:
    global _cached_handle
    if _cached_handle is None:
        _cached_handle = _DeepGemmHandle(num_sms=num_sms)
    return _cached_handle


class _GroupedGemm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        grad_edge: torch.Tensor | None,
        tensor_groups: torch.Tensor,
        list_groups_sizes_padded: list[int] | None,
        weights: Float8BlockwiseQTensor,
        m_indices: torch.Tensor,
        tensor_groups_sizes_padded: torch.Tensor,
        tensor_groups_sizes_cpu: torch.Tensor | None,
        triton_row2col: bool,
        fp8_grad_output: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        ctx.use_edges = grad_edge is not None
        ctx.fp8_grad_output = fp8_grad_output
        del grad_edge  # only its gradient matters
        ctx.device = tensor_groups.device
        ctx.weight_shape = weights.shape

        with record_fn("grouped_gemm::fwd"):
            ctx.is_quantized_input = isinstance(tensor_groups, Float8BlockwiseQTensor)
            if not ctx.is_quantized_input:
                with record_fn("grouped_gemm::fwd::row_quantization_1x128_fn"):
                    tensor_groups_quantized = row_quantization_1x128_fn(
                        tensor_groups.contiguous()
                    )
            else:
                tensor_groups_quantized = (tensor_groups._data, tensor_groups._scale)

            weights_quantized = (weights._data, weights._scale)
            ctx.m_indices = m_indices

            with record_fn("grouped_gemm::fwd::m_grouped_fp8_gemm"):
                _, int_dim, _ = weights_quantized[0].shape
                out = torch.empty(
                    [tensor_groups.shape[0], int_dim],
                    dtype=torch.bfloat16,
                    device=ctx.device,
                )

                tensor_groups_quantized = (
                    tensor_groups_quantized[0].view(torch.float8_e4m3fn).contiguous(),
                    tensor_groups_quantized[1],
                )

                _get_deep_gemm_handle().m_grouped_fp8_gemm_nt_contiguous(
                    tensor_groups_quantized, weights_quantized, out, m_indices
                )

        ctx.needs_wgrad = weights.requires_grad
        saved = [weights._data, weights._scale]
        if ctx.needs_wgrad:
            saved.extend(tensor_groups_quantized)
        ctx.save_for_backward(*saved)
        ctx.list_groups_sizes = list_groups_sizes_padded
        ctx.tensor_groups_sizes = tensor_groups_sizes_padded
        ctx.tensor_groups_sizes_cpu = tensor_groups_sizes_cpu
        ctx.triton_row2col = triton_row2col

        if fp8_grad_output:
            ctx.mark_non_differentiable(out)
            return out, fp8_grad_edge(out.shape[0], out.shape[1], device=out.device)
        return out

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor | Float8BlockwiseQTensor,
        grad_fp8_edge: Float8BlockwiseQTensor | None = None,
    ) -> tuple[torch.Tensor | None, ...]:
        with record_fn("grouped_gemm::bwd"):
            if ctx.fp8_grad_output:
                # The value slot took no gradient; the packed edge carries it.
                assert grad_fp8_edge is not None
                grad_output = grad_fp8_edge
            needs_wgrad = ctx.needs_wgrad
            w_data, w_scale, *tensor_groups_quantized = ctx.saved_tensors
            tensor_groups_quantized = tuple(tensor_groups_quantized)

            # DGRAD
            is_quantized_grad_output = isinstance(grad_output, Float8BlockwiseQTensor)
            if is_quantized_grad_output:
                assert isinstance(grad_output, Float8BlockwiseQTensor)
                assert grad_output._data.shape[0] == grad_output._scale.shape[0]
                assert grad_output._data.shape[1] // 128 == grad_output._scale.shape[1]
                grad_output_quantized = (
                    grad_output._data.view(torch.float8_e4m3fn),
                    grad_output._scale,
                )
            else:
                with record_fn("grouped_gemm::bwd::row_quantization_1x128_fn"):
                    grad_output_quantized = row_quantization_1x128_fn(
                        grad_output.contiguous()
                    )

            assert (
                grad_output_quantized[0].shape[0] == grad_output_quantized[1].shape[0]
            )
            assert (
                grad_output_quantized[0].shape[1] // 128
                == grad_output_quantized[1].shape[1]
            )

            # weights quantization for dgrad
            with record_fn("grouped_gemm::bwd::transpose_weights_fn"):
                weights_quantized = (
                    transpose_last_two_dims_fn(w_data),
                    transpose_last_two_dims_fn(w_scale),
                )

            # dgrad = grad_output x weights
            with record_fn("grouped_gemm::bwd::dgrad_m_grouped_fp8_gemm"):
                dgrad = torch.empty(
                    [grad_output_quantized[0].shape[0], w_data.shape[-1]],
                    dtype=torch.bfloat16,
                    device=ctx.device,
                )

                _get_deep_gemm_handle().m_grouped_fp8_gemm_nt_contiguous(
                    grad_output_quantized, weights_quantized, dgrad, ctx.m_indices
                )

                clear_tensor_data(*weights_quantized)

            # WGRAD (skipped entirely when weights are frozen)
            wgrad = None
            if needs_wgrad:
                with record_fn(
                    "grouped_gemm::bwd::row2col_requantization_deepgemm_layout_fn"
                ):
                    if ctx.triton_row2col:
                        row2col_fn = row2col_requantization_deepgemm_layout_fn
                    else:
                        from torchtitan.ops.float8._cudacherry import (
                            requantize_rowwise_to_columnwise as row2col_fn,
                        )

                    tensor_groups_quantized_col = row2col_fn(
                        tensor_groups_quantized[0],
                        ctx.m_indices,
                        ctx.tensor_groups_sizes,
                        tensor_groups_quantized[1],
                    )
                    clear_tensor_data(*tensor_groups_quantized)

                    grad_output_quantized_col = row2col_fn(
                        grad_output_quantized[0],
                        ctx.m_indices,
                        ctx.tensor_groups_sizes,
                        grad_output_quantized[1],
                    )

                    clear_tensor_data(*grad_output_quantized)

                # TODO: replace with torch.empty
                wgrad = torch.zeros(
                    ctx.weight_shape, dtype=torch.bfloat16, device=ctx.device
                )

                grad_output_quantized_col = (
                    grad_output_quantized_col[0].view(torch.float8_e4m3fn),
                    grad_output_quantized_col[1].mT,
                )
                tensor_groups_quantized_col = (
                    tensor_groups_quantized_col[0].view(torch.float8_e4m3fn),
                    tensor_groups_quantized_col[1].mT,
                )

                _get_deep_gemm_handle().k_grouped_fp8_gemm_contiguous(
                    a=grad_output_quantized_col,
                    b=tensor_groups_quantized_col,
                    d=wgrad,
                    ks=ctx.list_groups_sizes,
                    ks_cpu=ctx.tensor_groups_sizes_cpu,
                    ks_tensor=ctx.tensor_groups_sizes,
                )

                clear_tensor_data(*grad_output_quantized_col)
                clear_tensor_data(*tensor_groups_quantized_col)
            else:
                clear_tensor_data(*grad_output_quantized)

        return (
            dgrad if ctx.use_edges else None,  # grad_edge
            None if ctx.use_edges else dgrad,  # tensor_groups
            None,  # list_groups_sizes_padded
            wgrad,  # weights
            None,  # m_indices
            None,  # tensor_groups_sizes_padded
            None,  # tensor_groups_sizes_cpu
            None,  # triton_row2col
            None,  # fp8_grad_output
        )


class GroupedGemmFp8Wrapper:
    def __init__(self, num_sms: int | None = None) -> None:
        super().__init__()
        _get_deep_gemm_handle(num_sms=num_sms)

    def group_gemm(
        self,
        tensor_groups: torch.Tensor,
        tensor_groups_sizes: list[int] | None,
        weights: Float8BlockwiseQTensor,
        m_indices: torch.Tensor | None = None,
        tensor_groups_sizes_padded: torch.Tensor | None = None,
        tensor_groups_sizes_cpu: torch.Tensor | None = None,
        triton_row2col: bool = False,
        grad_edge: torch.Tensor | None = None,
        fp8_grad_output: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Grouped fp8 GEMM over padded, expert-major token groups.

        Returns ``(output, packed_edge)``; the edge is ``None`` unless
        ``fp8_grad_output`` asked for one, so the caller reads the same pair
        either way.
        """
        payload = (
            tensor_groups._data
            if isinstance(tensor_groups, Float8BlockwiseQTensor)
            else tensor_groups
        )
        # ``guard_or_false``: the row count is data-dependent under compile, and
        # a plain ``== 0`` would guard on it. False is the safe answer -- the
        # traced graph then takes the general path, whose kernels skip an empty
        # grid on their own.
        if guard_or_false(payload.numel() == 0):
            empty = _EmptyGroupGemm.apply(
                grad_edge, tensor_groups, weights, fp8_grad_output
            )
            return empty if fp8_grad_output else (empty, None)

        out = _GroupedGemm.apply(
            grad_edge,
            tensor_groups,
            tensor_groups_sizes,
            weights,
            m_indices,
            tensor_groups_sizes_padded,
            tensor_groups_sizes_cpu,
            triton_row2col,
            fp8_grad_output,
        )
        return out if fp8_grad_output else (out, None)
