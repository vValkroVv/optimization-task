import logging
import math
import typing as tp
from itertools import chain
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from transformers import PreTrainedModel

from einops import rearrange

try:
    from flash_attn import flash_attn_varlen_func  # type: ignore[import-not-found]
    from flash_attn.bert_padding import pad_input, unpad_input  # type: ignore[import-not-found]

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    flash_attn_varlen_func = None
    pad_input = None
    unpad_input = None
    FLASH_ATTN_AVAILABLE = False

try:
    from composer.utils import dist as composer_dist
    from llmfoundry.models.parallel.sequence.mappings import (
        gather_from_sequence_parallel_region,
    )
except ImportError:
    # HF / inference-only loading has no composer; encoder SP is training-only.
    composer_dist = None
    gather_from_sequence_parallel_region = None

from ...base_component import BaseOmniComponent
from .configuration_conformer import ConformerConfig


class StackingSubsampler(nn.Module):
    def __init__(self, subsampling_factor: int, feat_in: int, feat_out: int) -> None:
        super().__init__()

        self.feat_in = feat_in
        self.feat_out = feat_out
        self.subsampling_factor = subsampling_factor

        self.proj = nn.Linear(in_features=self.subsampling_factor * feat_in, out_features=feat_out,
                              bias=False)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tp.Tuple[torch.tensor, ...]:
        # https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/submodules/subsampling.py#L25
        b, t, h = x.size()

        mask = torch.arange(t, device=x.device).unsqueeze(0).unsqueeze(-1).expand(b, t, h)
        mask = mask >= lengths.view(b, 1, 1)
        x = x.clone()
        x[mask] = 0

        pad_size = (self.subsampling_factor - (t % self.subsampling_factor)) % self.subsampling_factor
        x = torch.nn.functional.pad(x, (0, 0, 0, pad_size))
        _, t, _ = x.size()
        x = torch.reshape(x, (b, t // self.subsampling_factor, h * self.subsampling_factor))
        lengths = torch.div(lengths + pad_size + self.subsampling_factor - 1, self.subsampling_factor, rounding_mode='floor')

        x = self.proj(x)

        return x, lengths


class ConvSubsampling(nn.Module):
    def __init__(
        self,
        subsampling_factor: int,
        feat_in: int,
        feat_out: int,
        conv_channels: int,
        subsampling: tp.Optional[str] = None,
    ) -> None:
        super().__init__()
        self._conv_channels = conv_channels
        self._feat_in = feat_in
        self._feat_out = feat_out
        self._subsampling_factor = subsampling_factor
        self._sampling_num = int(math.log(subsampling_factor, 2))

        self.activation = nn.ReLU(inplace=True)

        assert subsampling in ["striding", "striding_conv1d"]
        self.subsampling = subsampling

        layers = []
        if subsampling == 'striding':
            in_channels = 1

            self._stride = 2
            self._kernel_size = 3
            self._ceil_mode = False
            self._left_padding = (self._kernel_size - 1) // 2
            self._right_padding = (self._kernel_size - 1) // 2

            for i in range(self._sampling_num):
                layers.append(
                    torch.nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=conv_channels,
                        kernel_size=self._kernel_size,
                        stride=self._stride,
                        padding=(self._left_padding, self._right_padding),
                    )
                )
                layers.append(self.activation)
                in_channels = conv_channels

            in_length = torch.tensor(feat_in, dtype=torch.float)
            out_length = self.calc_length(
                lengths=in_length,
                all_paddings=self._left_padding + self._right_padding,
                kernel_size=self._kernel_size,
                stride=self._stride,
                ceil_mode=self._ceil_mode,
                repeat_num=self._sampling_num,
            )
            self.out = torch.nn.Linear(conv_channels * int(out_length), feat_out)
            self.conv2d_subsampling = True

        elif subsampling == 'striding_conv1d':
            in_channels = feat_in

            self._stride = 2
            self._kernel_size = 5
            self._ceil_mode = False

            self._left_padding = (self._kernel_size - 1) // 2
            self._right_padding = (self._kernel_size - 1) // 2
            self._max_cache_len = 0

            for i in range(self._sampling_num):
                layers.append(
                    torch.nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=feat_out if self._sampling_num == i + 1 else conv_channels,
                        kernel_size=self._kernel_size,
                        stride=self._stride,
                        padding=self._left_padding,
                    )
                )
                layers.append(self.activation)
                in_channels = conv_channels

            self.conv2d_subsampling = False

        self.conv = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, lengths: tp.Optional[torch.Tensor] = None):
        if lengths is None:
            lengths = torch.ones(x.size(0), device=x.device).long() * x.size(1)

        if self.conv2d_subsampling:
            lengths = self.calc_length(
                lengths,
                all_paddings=self._left_padding + self._right_padding,
                kernel_size=self._kernel_size,
                stride=self._stride,
                ceil_mode=self._ceil_mode,
                repeat_num=self._sampling_num,
            )
            x = x.unsqueeze(1)
            x = self.conv(x)
            b, c, t, f = x.size()
            x = self.out(x.transpose(1, 2).reshape(b, t, -1))
        else:
            x = x.transpose(1, 2).contiguous()

            for stage in range(self._sampling_num):
                x = self.conv[2 * stage](x)
                x = self.conv[2 * stage + 1](x)
                lengths = self.calc_length(
                    lengths,
                    all_paddings=self._left_padding + self._right_padding,
                    kernel_size=self._kernel_size,
                    stride=self._stride,
                    ceil_mode=self._ceil_mode,
                    repeat_num=1,
                )
                pad_mask = torch.arange(x.size(2), device=x.device).unsqueeze(0) >= lengths.unsqueeze(-1)
                # Zero pads after each biased conv: the first conv turns zero pads into
                # ReLU(bias) != 0, so the last valid frames of a row would depend on how
                # far the batch is padded. Out-of-place on purpose: masked_fill_ would
                # bump the version counter of the tensor the inplace ReLU saved.
                x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)

            x = x.transpose(1, 2)

        return x, lengths

    def calc_length(self, lengths, all_paddings, kernel_size, stride, ceil_mode, repeat_num=1):  # type: ignore
        """Calculates the output length of a Tensor passed through a convolution or max pooling layer"""
        add_pad: float = all_paddings - kernel_size
        one: float = 1.0
        for i in range(repeat_num):
            lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + one
            if ceil_mode:
                lengths = torch.ceil(lengths)
            else:
                lengths = torch.floor(lengths)
        return lengths.to(dtype=torch.int)


class ConformerFeedForward(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_ff: int,
            dropout: float,
            activation: str = 'silu',
        ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.linear1 = nn.Linear(d_model, d_ff)

        if activation == 'silu':
            activation_md = nn.SiLU(inplace=True)
        else:
            raise ValueError('Only silu activation supported!')

        self.activation = activation_md
        # self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        # x = self.dropout(x)
        x = self.linear2(x)
        return x


class NemoConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: tp.Union[str, int] = 0,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ):
        if isinstance(padding, int):
            self._left_padding = padding
            self._right_padding = padding
        elif isinstance(padding, list) and len(padding) == 2 and padding[0] + padding[1] == kernel_size - 1:
            self._left_padding = padding[0]
            self._right_padding = padding[1]
        else:
            raise ValueError(f"Invalid padding param: {padding}!")

        self._left_cache_len = padding
        self._right_cache_len = padding

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, x: torch.Tensor):
        base_dtype = x.dtype
        x = nn.functional.pad(x, pad=(self._left_padding, self._right_padding))
        x = x.to(base_dtype)
        return super().forward(x)


class ConformerConvolution(nn.Module):

    def __init__(
        self,
        d_model: int,
        kernel_size: int,
        conv_context_size: tp.Union[int, tp.List[int]],
        norm_type: str = "batch_norm",
        causal_conv_chunk: tp.Optional[int] = None,
        use_causal_conv: bool = False,
    ):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.norm_type = norm_type

        assert not (use_causal_conv and causal_conv_chunk), \
            "Cannot use causal conv with causal conv chunking"

        if isinstance(conv_context_size, int):
            conv_context_size = [conv_context_size, conv_context_size]

        if causal_conv_chunk is not None:
            super_conv_context = 0
        elif use_causal_conv:
            super_conv_context = [kernel_size - 1, 0]
        else:
            super_conv_context = conv_context_size

        assert conv_context_size is not None, 'conv_context_size is None'
        self.pointwise_conv1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.depthwise_conv = NemoConv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=1,
            padding=super_conv_context,
            groups=d_model,
            bias=True,
        )

        assert norm_type in ["batch_norm", "layer_norm"]
        if norm_type == "batch_norm":
            self.batch_norm = nn.BatchNorm1d(d_model)
        elif norm_type == "layer_norm":
            self.batch_norm = nn.LayerNorm(d_model)

        self.activation = nn.SiLU(inplace=True)
        self.pointwise_conv2 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.causal_conv_chunk = causal_conv_chunk
        self.conv_ctx_size = conv_context_size[0]
        if self.causal_conv_chunk is not None:
            assert conv_context_size[0] == conv_context_size[1], \
                f"Expected sym values in paddings {conv_context_size} for chunking"

    def forward(self, x: torch.Tensor, pad_mask: tp.Optional[torch.Tensor] = None):
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = nn.functional.glu(x, dim=1)
        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)

        if self.causal_conv_chunk is not None:
            b, c, t = x.shape
            chunk_size = self.causal_conv_chunk
            assert chunk_size >= self.conv_ctx_size, \
                f'Expected chunk {chunk_size} >= conv context {self.conv_ctx_size}'
            x = F.pad(x, pad=(0, (chunk_size - t % chunk_size) % chunk_size))  # t_new % chunk_size == 0
            assert x.shape[2] % chunk_size == 0
            n_chunks = x.shape[2] // chunk_size
            x = x.reshape(b, c, n_chunks, chunk_size)
            cached = x[:, :, :, -self.conv_ctx_size:].roll(1, dims=-2) # [b, c, N, ctx_size]
            cached[:, :, 0, :] = 0
            x = F.pad(torch.cat([cached, x], dim=-1), (0, self.conv_ctx_size)) # [b, c, N, ctx_size + chunk_size + ctx_size]
            x = x.transpose(1, 2).reshape(b * n_chunks, c, -1)
            x = self.depthwise_conv(x)
            x = x.reshape(b, n_chunks, c, -1).transpose(1, 2).reshape(b, c, n_chunks * chunk_size)
            x = x[:, :, :t]
            assert x.shape[-1] == t, f'{x.shape}[-1] != {b, c, t}[-1]'
        else:
            x = self.depthwise_conv(x)

        if self.norm_type == "layer_norm":
            x = x.transpose(1, 2)
            x = self.batch_norm(x)
            x = x.transpose(1, 2)
        else:
            x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        return x.transpose(1, 2)


class ChunkRotaryPositionalEmbedding(nn.Module):

    def __init__(self, dim: int, base: int, attn_chunk_size: int, max_position_seq_len: int) -> None:
        """Chunk-wise rotary positional embedding
        Args:
            dim: Dimension of embedding
            base: Base value for exponential
            attn_chunk_size: Size of the chunk
        """
        super().__init__()
        self.dim = dim
        self.base = base
        self.attn_chunk_size = attn_chunk_size
        self.max_position_seq_len = max_position_seq_len

        self.coss, self.sins = self.create_pe(self.max_position_seq_len, "cpu")

    def create_pe(self, length: int, device: tp.Union[str, torch.device]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """Create sin, cos chunks pe tables of size chunk_size x dim//2.
        """
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        t = torch.arange(self.attn_chunk_size, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        blocks = math.ceil(length / self.attn_chunk_size)
        coss = torch.cat([cos for _ in range(blocks)], dim=0).to(device)
        sins = torch.cat([sin for _ in range(blocks)], dim=0).to(device)
        return coss, sins

    def extend_pe(self, length: int, device: tp.Union[str, torch.device]) -> None:
        if hasattr(self, "coss") and self.coss.shape[0] >= length:
            self.coss, self.sins = self.coss.to(device), self.sins.to(device)
            return None
        self.coss, self.sins = self.create_pe(length, device)

    def forward(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.Tuple[torch.Tensor, ...]]:
        seq_len = x.shape[1]
        coss = self.coss[:seq_len, :]
        sins = self.sins[:seq_len, :]
        return x, (coss, sins)


class ChunkRotaryPositionMultiHeadedAttention(nn.Module):

    def __init__(
        self,
        n_head: int,
        n_feat: int,
        dropout: tp.Optional[float] = None,
        attn_chunk_size: tp.Optional[int] = None,
        window_size: tp.Tuple[int, int] = (-1, -1),
        attention_backend: str = "auto",
    ):
        """Construct a ChunkRotaryPositionMultiHeadedAttention object."""
        super().__init__()

        assert n_feat % n_head == 0
        if attention_backend not in ("auto", "flash_attn", "sdpa"):
            raise ValueError(
                "attention_backend must be one of 'auto', 'flash_attn', or "
                f"'sdpa', got {attention_backend!r}"
            )
        if attention_backend == "flash_attn" and not FLASH_ATTN_AVAILABLE:
            logging.warning(
                "flash_attn is not installed; falling back to torch SDPA."
            )
        self.attention_backend = (
            "flash_attn"
            if FLASH_ATTN_AVAILABLE and attention_backend != "sdpa"
            else "sdpa"
        )

        self.d_k = n_feat // n_head
        self.s_d_k = math.sqrt(self.d_k)
        self.h = n_head

        self.attn_chunk_size = attn_chunk_size
        self.window_size = window_size
        if self.window_size != (-1, -1):
            logging.warning(f"Using non-default window_size = {self.window_size}")
        self.dropout_value = dropout if dropout else 0.0

        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)

    @staticmethod
    def _apply_rotary_emb(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the non-interleaved rotary embedding used by flash-attn."""
        rotary_dim = cos.shape[-1] * 2
        if rotary_dim > x.shape[-1]:
            raise ValueError(
                f"Rotary dimension {rotary_dim} exceeds head dimension {x.shape[-1]}"
            )

        cos = cos[:x.shape[1]].unsqueeze(0).unsqueeze(2)
        sin = sin[:x.shape[1]].unsqueeze(0).unsqueeze(2)
        x_rot = x[..., :rotary_dim]
        x1, x2 = x_rot.chunk(2, dim=-1)
        rotated = torch.cat(
            (x1 * cos - x2 * sin, x1 * sin + x2 * cos),
            dim=-1,
        )
        return torch.cat((rotated, x[..., rotary_dim:]), dim=-1)

    def _flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pad_mask: torch.Tensor,
        dropout: float,
    ) -> torch.Tensor:
        assert flash_attn_varlen_func is not None
        assert pad_input is not None and unpad_input is not None

        n_batch, seqlen, _ = q.shape
        valid_mask = ~pad_mask
        q_unpad, indices_q, _, max_seqlen_q, _ = unpad_input(q, valid_mask)
        q_unpad = rearrange(q_unpad, "nnz (h d) -> nnz h d", h=self.h)
        k_unpad, _, _, _, _ = unpad_input(k, valid_mask)
        k_unpad = rearrange(k_unpad, "nnz (h d) -> nnz h d", h=self.h)
        v_unpad, _, _, _, _ = unpad_input(v, valid_mask)
        v_unpad = rearrange(v_unpad, "nnz (h d) -> nnz h d", h=self.h)

        lengths = valid_mask.sum(1)
        chunk_lengths = [
            [self.attn_chunk_size] * (length // self.attn_chunk_size)
            + ([length % self.attn_chunk_size] if length % self.attn_chunk_size else [])
            for length in lengths.tolist()
        ]
        chunk_lengths = torch.tensor(
            list(chain.from_iterable(chunk_lengths)),
            dtype=torch.int32,
            device=q.device,
        )
        cu_seqlens = F.pad(chunk_lengths.cumsum(0), (1, 0), value=0).to(torch.int32)

        assert chunk_lengths.sum().item() == q_unpad.size(0), (
            "sum of chunks lengths differs from total frames length "
            f"{chunk_lengths.sum().item()} != {q_unpad.size(0)}"
        )

        output_unpad = flash_attn_varlen_func(
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens,
            cu_seqlens,
            max_seqlen_q,
            max_seqlen_q,
            dropout_p=dropout,
            window_size=self.window_size,
        )
        return pad_input(
            rearrange(output_unpad, "nnz h d -> nnz (h d)"),
            indices_q,
            n_batch,
            seqlen,
        )

    def _sdpa_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pad_mask: torch.Tensor,
        dropout: float,
    ) -> torch.Tensor:
        """Chunked attention equivalent to the flash-attn varlen call."""
        batch_size, seqlen, _ = q.shape
        chunk_size = self.attn_chunk_size
        padded_length = math.ceil(seqlen / chunk_size) * chunk_size
        padding = padded_length - seqlen

        def into_chunks(x: torch.Tensor) -> torch.Tensor:
            x = F.pad(x, (0, 0, 0, padding))
            x = x.view(batch_size, -1, chunk_size, self.h, self.d_k)
            return x.permute(0, 1, 3, 2, 4).reshape(
                -1, self.h, chunk_size, self.d_k
            )

        q_chunks = into_chunks(q)
        k_chunks = into_chunks(k)
        v_chunks = into_chunks(v)

        valid_mask = F.pad(~pad_mask, (0, padding), value=False)
        valid_mask = valid_mask.view(-1, chunk_size)
        attn_mask = valid_mask[:, None, None, :]

        assert self.window_size == (-1, -1), "Window size is not supported for SDPA!"

        output = F.scaled_dot_product_attention(
            q_chunks,
            k_chunks,
            v_chunks,
            attn_mask=attn_mask,
            dropout_p=dropout,
        )

        output = output.masked_fill(
            ~valid_mask[:, None, :, None],
            0.0,
        )
        output = output.view(
            batch_size, -1, self.h, chunk_size, self.d_k
        ).permute(0, 1, 3, 2, 4)
        return output.reshape(batch_size, padded_length, -1)[:, :seqlen]

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        pos_emb: torch.Tensor,
        pad_mask: torch.Tensor,
        **kwargs: tp.Dict,
    ) -> torch.Tensor:
        """Compute chunk-wise rotary position attention.
        Args:
            query: Query tensor T X B X C
            key: Key tensor T X B X C
            value: Value tensor T X B X C
            key_padding_mask: Mask tensor T X B
        Returns:
            torch.Tensor: Output tensor T X B X D.
        """
        B, T, _ = value.size()
        query = query.view(B, T, self.h, self.d_k)
        key = key.view(B, T, self.h, self.d_k)
        value = value.view(B, T, self.h, self.d_k)

        # query, key: seq_len x bs x nh x hd
        # cos, sin: seq_len x hd/2
        cos, sin = pos_emb
        cos, sin = cos.to(query.dtype), sin.to(query.dtype)
        query = self._apply_rotary_emb(query, cos, sin)
        key = self._apply_rotary_emb(key, cos, sin)

        query = query.view(B, T, self.h * self.d_k)
        key = key.view(B, T, self.h * self.d_k)
        value = value.view(B, T, self.h * self.d_k)

        q = self.linear_q(query)
        k = self.linear_k(key)
        v = self.linear_v(value)

        dropout = self.dropout_value if self.training else 0.0
        if self.attention_backend == "flash_attn":
            out = self._flash_attention(q, k, v, pad_mask, dropout)
        else:
            out = self._sdpa_attention(q, k, v, pad_mask, dropout)

        out = self.linear_out(out)
        return out


class ConformerLayer(nn.Module):
    """A single block of the Conformer encoder.

    Args:
        d_model (int): input dimension of MultiheadAttentionMechanism and PositionwiseFeedForward
        d_ff (int): hidden dimension of PositionwiseFeedForward
        n_heads (int): number of heads for multi-head attention
        conv_kernel_size (int): kernel size for depthwise convolution in convolution module
        dropout (float): dropout probabilities for linear layers
        dropout_att (float): dropout probabilities for attention distributions
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        self_attention_model: str = "rotary",
        n_heads: int = 4,
        conv_kernel_size: int = 31,
        conv_norm_type: str = "batch_norm",
        dropout: float = 0.1,
        dropout_att: float = 0.1,
        attn_chunk_size: tp.Optional[tp.Any] = None,
        use_causal_conv_chunking: bool = False,
        use_causal_conv: bool = False,
        window_size: tp.Tuple[int, int] = (-1, -1),
        attention_backend: str = "auto",
        *args: tp.List,
        **kwargs: tp.Dict,
    ):
        super().__init__()

        self.self_attention_model = self_attention_model
        self.n_heads = n_heads
        self.fc_factor = 0.5

        # first feed forward module
        self.norm_feed_forward1 = LayerNorm(d_model)
        self.feed_forward1 = ConformerFeedForward(
            d_model=d_model, d_ff=d_ff, dropout=dropout
        )

        # convolution module
        self.norm_conv = LayerNorm(d_model)
        self.conv = ConformerConvolution(
            d_model=d_model,
            kernel_size=conv_kernel_size,
            norm_type=conv_norm_type,
            conv_context_size=(conv_kernel_size - 1) // 2,
            causal_conv_chunk=attn_chunk_size if use_causal_conv_chunking else None,
            use_causal_conv=use_causal_conv,
        )

        # multi-headed self-attention module
        self.norm_self_att = LayerNorm(d_model)

        assert self_attention_model == "chunk_rotary"
        self.self_attn = ChunkRotaryPositionMultiHeadedAttention(
            n_head=n_heads,
            n_feat=d_model,
            dropout=dropout_att,
            attn_chunk_size=attn_chunk_size,
            window_size=window_size,
            attention_backend=attention_backend,
        )

        # second feed forward module
        self.norm_feed_forward2 = LayerNorm(d_model)
        self.feed_forward2 = ConformerFeedForward(
            d_model=d_model, d_ff=d_ff, dropout=dropout
        )

        # self.dropout = nn.Dropout(dropout)
        self.norm_out = LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        att_mask: tp.Optional[torch.Tensor] = None,
        pos_emb: tp.Optional[torch.Tensor] = None,
        pad_mask: tp.Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x (torch.Tensor): input signals (B, T, d_model)
            att_mask (torch.Tensor): attention masks(B, T, T)
            pos_emb (torch.Tensor): (L, 1, d_model)
            pad_mask (torch.Tensor): padding mask
        Returns:
            x (torch.Tensor): (B, T, d_model)
        """
        residual = x
        x = self.norm_feed_forward1(x)
        x = self.feed_forward1(x)
        # residual = residual + self.dropout(x) * self.fc_factor
        residual = residual + x * self.fc_factor

        x = self.norm_self_att(residual)

        assert self.self_attention_model == "chunk_rotary"
        assert pos_emb is not None and att_mask is None and pad_mask is not None
        x = self.self_attn(query=x, key=x, value=x, pos_emb=pos_emb, pad_mask=pad_mask)

        # residual = residual + self.dropout(x)
        residual = residual + x

        x = self.norm_conv(residual)
        x = self.conv(x, pad_mask=pad_mask)

        # residual = residual + self.dropout(x)
        residual = residual + x

        x = self.norm_feed_forward2(residual)
        x = self.feed_forward2(x)
        # residual = residual + self.dropout(x) * self.fc_factor
        residual = residual + x * self.fc_factor

        x = self.norm_out(residual)

        # Explicitly cast to float16 to avoid fp32 activation checkpointing
        x = x.to(dtype=torch.bfloat16)
        return x


def _length_bucketed_groups(
    spec_lengths: torch.Tensor, pad_waste_cap: float = 0.5
) -> tp.List[tp.List[int]]:
    """Greedy length buckets (descending): close a group once padding it to the group
    max would exceed ``pad_waste_cap`` relative padded overhead. One homogeneous batch
    returns a single group (zero-overhead fast path in the caller)."""
    lengths = [max(int(length), 1) for length in spec_lengths.tolist()]
    order = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    groups: tp.List[tp.List[int]] = []
    cur: tp.List[int] = []
    cur_max = 0
    cur_sum = 0

    for i in order:
        if not cur:
            cur, cur_max, cur_sum = [i], lengths[i], lengths[i]
            continue

        if cur_max * (len(cur) + 1) > (1.0 + pad_waste_cap) * (cur_sum + lengths[i]):
            groups.append(cur)
            cur, cur_max, cur_sum = [i], lengths[i], lengths[i]
        else:
            cur.append(i)
            cur_sum += lengths[i]

    if cur:
        groups.append(cur)

    return groups


def _sp_padded_len(seq_len: int, sp_size: int, attn_chunk_size: int) -> int:
    """Pad the stack so every rank's core is a whole number of attention chunks."""
    div = sp_size * attn_chunk_size

    return -(-seq_len // div) * div


def _sp_slice_bounds(
    padded_len: int, sp_rank: int, sp_size: int, overlap: int
) -> tp.Tuple[int, int]:
    """[start, finish) of this rank's slice: its core plus one active halo per
    interior side (edge ranks one halo, middle ranks two). Requires core >= overlap."""
    core = padded_len // sp_size
    start = sp_rank * core - (overlap if sp_rank > 0 else 0)
    finish = (sp_rank + 1) * core + (overlap if sp_rank < sp_size - 1 else 0)

    return start, finish


def _is_rank_zero() -> bool:
    # No composer here: the modalities package is also imported at inference.
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0

    return True


class ConformerEncoder(PreTrainedModel):
    config_class = ConformerConfig

    def __init__(self, config: ConformerConfig):
        super().__init__(config)

        d_ff = config.d_model * config.ff_expansion_factor
        self.d_model = config.d_model
        self.n_layers = config.n_layers
        self._feat_in = config.feat_in
        self.scale = math.sqrt(self.d_model)
        self.subsampling_factor = config.subsampling_factor

        self.attn_chunk_size = config.attn_chunk_size
        self.conv_norm_type = config.conv_norm_type
        # Stamped by the top-level model before every component forward; the
        # encoder's own .training may be flipped by a frozen-encoder eval wrap.
        self._top_level_training = False
        # debug purpose only
        self._sp_infeasible_warned = False
        self._sp_engaged_logged = False

        self.self_attention_model = config.self_attention_model

        assert self.self_attention_model == "chunk_rotary"
        self.pos_enc = ChunkRotaryPositionalEmbedding(
            dim=config.d_model // config.n_heads,
            base=config.pos_emb_max_len,
            attn_chunk_size=config.attn_chunk_size,
            max_position_seq_len=config.max_position_seq_len,
        )

        subsampling_conv_channels = config.subsampling_conv_channels

        if config.subsampling_conv_channels == -1:
            subsampling_conv_channels = config.d_model

        if config.subsampling and config.subsampling_factor > 1:
            if config.subsampling == "stacking":
                self.pre_encode = StackingSubsampler(
                    subsampling_factor=config.subsampling_factor,
                    feat_in=config.feat_in,
                    feat_out=config.d_model,
                )
            else:
                self.pre_encode = ConvSubsampling(
                    subsampling_factor=config.subsampling_factor,
                    feat_in=config.feat_in,
                    feat_out=config.d_model,
                    conv_channels=subsampling_conv_channels,
                    subsampling=config.subsampling,
                )

        self._feat_out = config.d_model

        self.pos_emb_max_len = config.pos_emb_max_len

        self.layers = nn.ModuleList()
        for i in range(config.n_layers):
            layer = ConformerLayer(
                d_model=config.d_model,
                d_ff=d_ff,
                self_attention_model=config.self_attention_model,
                n_heads=config.n_heads,
                conv_kernel_size=config.conv_kernel_size,
                conv_norm_type=config.conv_norm_type,
                dropout=config.dropout,
                dropout_att=config.dropout_att,
                attn_chunk_size=config.attn_chunk_size,
                layer_index=i,
                use_causal_conv_chunking=config.use_causal_conv_chunking,
                use_causal_conv=config.use_causal_conv,
                window_size=config.window_size,
                attention_backend=config.attention_backend,
            )
            self.layers.append(layer)

        if config.feat_out > 0 and config.feat_out != self._feat_out:
            self.out_proj = nn.Linear(self._feat_out, config.feat_out)
            self._feat_out = config.feat_out
        else:
            self.out_proj = None
            self._feat_out = config.d_model

        self.set_max_audio_length(self.pos_emb_max_len, 'cpu')

        self.length_bucketing = bool(getattr(config, "length_bucketing", True))
        # debug purpose only
        self._bucketing_log_budget = 5

        if not self.length_bucketing:
            return

        if config.use_per_layer_wrapping:
            raise ValueError(
                "length_bucketing runs the encoder several times per microbatch, "
                "which needs the encoder to be a single gigafsdp wrap unit; set "
                "length_bucketing=False or use_per_layer_wrapping=False"
            )

        if config.subsampling != "striding_conv1d":
            raise ValueError(
                "length_bucketing is only exact with the striding_conv1d stem "
                "(its pads are zeroed between conv stages); set "
                f"length_bucketing=False for subsampling={config.subsampling!r}"
            )

        if config.conv_norm_type != "layer_norm":
            raise ValueError(
                "length_bucketing requires conv_norm_type='layer_norm': batch_norm "
                "statistics would differ per bucket group"
            )

    def set_max_audio_length(self, seq_length: int, device: tp.Optional[tp.Union[str, torch.device]]) -> None:
        """Sets maximum input length. Pre-calculates internal seq_range mask.
        """
        device = "cpu" if device is None else device
        self.pos_enc.extend_pe(seq_length, device)

    def forward(self, audio_signal: torch.Tensor, lengths: torch.Tensor) -> tp.Tuple[torch.Tensor, ...]:
        """Conformer encoder forward.

        Args:
            audio_signal (torch.Tensor): bs x seq_len x n_mel, padded to the batch max
            lengths (torch.Tensor): bs

        Returns:
            tp.Tuple[torch.Tensor, ...]: processed audio_signal and lengths.
        """
        is_sp_active = self._encoder_sp_active()

        if self.length_bucketing and audio_signal.size(0) > 1:
            groups = _length_bucketed_groups(lengths)

            if len(groups) > 1:
                return self._forward_groups(audio_signal, lengths, groups, is_sp_active)

        return self._forward_stack(audio_signal, lengths, is_sp_active)

    def _encoder_sp_active(self) -> bool:
        """Shard the post-stem stacks over the SP group when one of size > 1
        exists (composer ``fsdp_config.sp_size``) and the TOP-LEVEL model is
        training. Training microbatches are gathered over the TP-SP group before
        the forward, so every rank of the group holds the same audio; eval and
        predict batches are gathered over TP only and differ per SP rank, hence
        the training condition. Whether a given stack is long enough is decided
        per stack by ``_sp_layout``.
        """
        if composer_dist is None or not self._top_level_training:
            return False

        return (composer_dist.get_sp_group_size() or 1) > 1

    def _forward_groups(
        self,
        audio_signal: torch.Tensor,
        lengths: torch.Tensor,
        groups: tp.List[tp.List[int]],
        is_sp_active: bool = False,
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """Run the stack once per length-homogeneous group of clips.

        The sub-forwards sit BELOW the gigafsdp wrap boundary: with the whole
        encoder as one wrap unit the STM hooks fire once per microbatch (one
        weight gather, one buffer claim, one gated backward region), and the
        per-group calls issue no collectives, so rank-divergent group counts
        are legal (per-layer wrap units would make the STM call count
        data-dependent and desync NCCL; rejected in ``__init__``). Under SP the
        lengths are identical across the SP group, groups are formed from them
        BEFORE any slicing, and the only collective inside a per-group call
        (the SP gather) is a pure function of those values. Pads are
        zeroed between the stem conv stages, so trimming each group to its
        own max is mathematically identical to one stack padded to the batch
        max (bit-identical in exact arithmetic; bf16 GEMM blocking still
        depends on the batch shape).

        Args:
            audio_signal: ``[bs, T_max, n_mel]`` padded to the batch max.
            lengths: per-clip frame counts, ``[bs]``.
            groups: clip indices per group, from ``_length_bucketed_groups``
                (descending by length, so the first group holds the longest clip).
            is_sp_active: forwarded to every per-group ``_forward_stack``.

        Returns:
            ``[bs, T_out_max, D]`` outputs in the original clip order and the
            ``[bs]`` output lengths.
        """
        n_clips = audio_signal.size(0)
        lengths_host = lengths.tolist()
        out: tp.Optional[torch.Tensor] = None
        out_lens: tp.Optional[torch.Tensor] = None

        for group in groups:
            idx = torch.tensor(group, device=audio_signal.device)
            trim = max(1, max(lengths_host[i] for i in group))
            group_out, group_lens = self._forward_stack(
                audio_signal[idx, :trim], lengths[idx], is_sp_active
            )

            if out is None:
                out = group_out.new_zeros(n_clips, group_out.size(1), group_out.size(2))
                out_lens = group_lens.new_zeros(n_clips)

            assert group_out.size(1) <= out.size(1), "groups are not sorted by length"
            out[idx, : group_out.size(1)] = group_out
            out_lens[idx] = group_lens

        if self._bucketing_log_budget > 0 and _is_rank_zero():
            self._bucketing_log_budget -= 1
            single_stack = n_clips * max(lengths_host)
            bucketed = sum(len(g) * max(lengths_host[i] for i in g) for g in groups)
            # WARNING level: training jobs filter INFO at the root logger.
            logging.warning(
                f"conformer length bucketing: {n_clips} clips -> {len(groups)} groups, "
                f"{bucketed} padded frames vs {single_stack} as one stack "
                f"({bucketed / single_stack:.2f}x)"
            )

        return out, out_lens

    def _forward_stack(
        self,
        audio_signal: torch.Tensor,
        lengths: torch.Tensor,
        is_sp_active: bool = False,
    ) -> tp.Tuple[torch.Tensor, ...]:
        """Single padded stack through the stem and all layers.

        With ``is_sp_active`` the post-stem stack is sharded over the SP group:
        each rank runs its core plus one halo of ``attn_chunk_size * n_layers``
        frames per interior side, the cores are all-gathered along time
        (reduce-scatter backward). Exact vs sp1 in eval mode or with dropout 0;
        with dropout > 0 every slice draws its own dropout masks, so the result
        is a different valid dropout sample, not bit-equal.
        """
        max_audio_length: int = audio_signal.size(1)
        self.set_max_audio_length(seq_length=max_audio_length, device=audio_signal.device)

        if isinstance(self.pre_encode, nn.Linear):
            audio_signal = self.pre_encode(audio_signal)
        else:
            audio_signal, lengths = self.pre_encode(x=audio_signal, lengths=lengths)

        if self.self_attention_model != "chunk_rotary":
            raise ValueError("Wrong attention type.")

        seq_len = audio_signal.size(1)
        sp_layout = self._sp_layout(seq_len) if is_sp_active else None

        if sp_layout is not None:
            sp_rank, sp_size, padded_len, overlap = sp_layout
            start, finish = _sp_slice_bounds(padded_len, sp_rank, sp_size, overlap)
            # The rotary tables are periodic in attn_chunk_size, so a misaligned
            # slice would silently rotate every frame by the wrong phase.
            chunk = self.attn_chunk_size
            assert start % chunk == 0 and finish % chunk == 0, (
                f"SP slice [{start}, {finish}) is not attention-chunk aligned"
            )
            # Pad only this rank's slice to [start, finish): the full padded
            # stack is never materialized, and F.pad copies, so the slice is
            # contiguous and the layers retain nothing wider than the slice.
            valid_end = min(finish, seq_len)
            audio_signal = F.pad(
                audio_signal[:, start:valid_end], (0, 0, 0, finish - valid_end)
            )
            self.set_max_audio_length(seq_length=padded_len, device=audio_signal.device)
            pos_emb = (self.pos_enc.coss[start:finish], self.pos_enc.sins[start:finish])
            padding_length = (lengths - start).clamp(min=0, max=finish - start)
        else:
            audio_signal, pos_emb = self.pos_enc(x=audio_signal)
            padding_length = lengths

        max_audio_length = audio_signal.size(1)

        # NOTE (Sbr, fedorovgv) : double negatiation (here and later) is necessary, it is the torch bug!
        pad_mask = torch.arange(0, max_audio_length, device=audio_signal.device).expand(
            padding_length.size(0), -1
        ) < padding_length.unsqueeze(-1)
        pad_mask = ~pad_mask

        for _, layer in enumerate(self.layers):
            audio_signal = layer(audio_signal, att_mask=None, pos_emb=pos_emb, pad_mask=pad_mask)

        if sp_layout is not None:
            core_start = overlap if sp_rank > 0 else 0
            core_end = audio_signal.size(1) - (overlap if sp_rank < sp_size - 1 else 0)
            assert core_end - core_start == padded_len // sp_size, (
                f"core [{core_start}, {core_end}) does not tile "
                f"{padded_len} / {sp_size}"
            )
            # Reduce-scatter backward (never the raw gather_along_dim: peers would
            # enter torch.cat as constants and each rank would keep only its own
            # diagonal ~1/sp of the gradient). The contiguous clone is mandatory:
            # gather_along_dim assigns its input as this rank's all_gather
            # destination, i.e. an in-place NCCL write into a tensor the last
            # layer saved for backward (and a narrow along dim 1 is not
            # contiguous).
            audio_signal = gather_from_sequence_parallel_region(
                audio_signal[:, core_start:core_end].clone(
                    memory_format=torch.contiguous_format
                )
            )
            audio_signal = audio_signal[:, :seq_len]

        if self.out_proj is not None:
            audio_signal = self.out_proj(audio_signal)

        return audio_signal, lengths

    def _sp_layout(self, seq_len: int) -> tp.Optional[tp.Tuple[int, int, int, int]]:
        """(sp_rank, sp_size, padded_len, overlap) for sharding a post-stem stack of
        `seq_len` frames, or None when the stack runs replicated: the core per
        rank would be shorter than the halo, or sp_size > 2 * n_layers (a rank's
        slice could then start past the end of the stack). Pure function of
        sp-group-identical values, so every rank of the group takes the same
        branch.

        The halo is `attn_chunk_size * n_layers` frames per interior side: chunk
        attention never crosses an aligned cut, the depthwise conv leaks at most
        one conv reach per layer and the next layer's attention smears that over
        one whole chunk, so contamination advances <= 1 chunk per layer.
        """
        assert self.conv_norm_type == "layer_norm", (
            "encoder SP requires conv_norm_type=layer_norm: batch_norm statistics "
            "would differ per SP slice"
        )
        sp_size = composer_dist.get_sp_group_size()
        chunk = self.attn_chunk_size
        overlap = chunk * len(self.layers)
        padded_len = _sp_padded_len(seq_len, sp_size, chunk)

        if sp_size > 2 * len(self.layers) or padded_len // sp_size < overlap:
            if not self._sp_infeasible_warned and _is_rank_zero():
                logging.warning(
                    f"encoder SP: {seq_len}-frame stack runs replicated at "
                    f"sp={sp_size} (core {padded_len // sp_size}, halo {overlap}, "
                    f"{len(self.layers)} layers; logged once)"
                )

            self._sp_infeasible_warned = True

            return None

        if not self._sp_engaged_logged and _is_rank_zero():
            logging.warning(
                f"encoder SP: {seq_len}-frame stack sharded over sp={sp_size} "
                f"(halo {overlap}; logged once)"
            )

        self._sp_engaged_logged = True

        return composer_dist.get_sp_group_rank(), sp_size, padded_len, overlap


class ConformerForAudioLanguageModeling(ConformerEncoder, BaseOmniComponent):
    def forward(
        self, inputs: List[Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        spectrograms, spectrogram_lengths, sizes_list = self._concat_inputs(inputs)
        spectrograms, spectrogram_lengths = super().forward(spectrograms, spectrogram_lengths)
        return self._split_outputs(spectrograms, spectrogram_lengths, sizes_list)

    def _concat_inputs(
        self, inputs: List[Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for input_item in inputs:
            assert "spectrograms" in input_item and "spectrogram_lengths" in input_item, (
                f"`spectrograms` and `spectrogram_lengths` is required, but not found in input_item: {input_item.keys()}"
            )

        sizes_list = [len(input_item["spectrogram_lengths"]) for input_item in inputs]
        spectrograms = torch.cat([input_item["spectrograms"] for input_item in inputs], dim=0)
        spectrogram_lengths = torch.cat([input_item["spectrogram_lengths"] for input_item in inputs], dim=0)
        spectrograms = self._reshape_spectrograms_to_batched(spectrograms, spectrogram_lengths)
        return spectrograms, spectrogram_lengths, sizes_list

    def _split_outputs(
        self, spectrograms: torch.Tensor, spectrogram_lengths: torch.Tensor, sizes_list: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        spectrogram_lengths_splitted = torch.split(spectrogram_lengths, sizes_list)
        spectrograms_splitted = torch.split(spectrograms, sizes_list)

        outputs = []
        for spec, lengths in zip(spectrograms_splitted, spectrogram_lengths_splitted):
            outputs.extend([spec, lengths])
        return tuple(outputs)

    def _reshape_spectrograms_to_batched(
        self,
        spectrograms: torch.Tensor,
        spectrogram_lengths: torch.Tensor,
    ) -> torch.Tensor:
        max_length = int(spectrogram_lengths.max())
        spectrograms_splitted = torch.split(spectrograms, spectrogram_lengths.tolist())
        return torch.cat([
            F.pad(sample.unsqueeze(0), (0, 0, 0, max_length - sample.shape[0]))
            for sample in spectrograms_splitted
        ], dim=0)

    def unwrap_tuple_output(
        self,
        inputs: List[Dict[str, torch.Tensor]],
        outputs: Tuple[torch.Tensor, torch.Tensor],
    ) -> List[Dict[str, torch.Tensor]]:
        return [{"spectrograms": outputs[idx], "spectrogram_lengths": outputs[idx + 1]} for idx in range(0, len(outputs), 2)]

    def get_giga_fsdp_modules_to_wrap_with_names(
        self, prefix: str = ""
    ) -> List[Tuple[torch.nn.Module, str]]:
        if not self.config.use_per_layer_wrapping:
            return [(self, prefix)]

        modules_to_wrap_with_names = [(self.pre_encode, "pre_encode")] + [
            (layer, f"layers.{idx}")
            for idx, layer in enumerate(self.layers)
        ]

        if prefix and not prefix.endswith("."):
            prefix += "."

        modules_to_wrap_with_names = [
            (item[0], prefix + item[1]) for item in modules_to_wrap_with_names
        ]
        return modules_to_wrap_with_names

    def get_giga_fsdp_modules_for_activation_checkpointing(
        self,
    ) -> List[torch.nn.Module]:
        return [self.pre_encode] + list(self.layers)
