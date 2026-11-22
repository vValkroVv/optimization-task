# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exact CUDA-13 consuming cross-entropy for two folded BF16 heads on H100."""

import ctypes
import hashlib
import math
from collections.abc import Callable, Sequence
from functools import cache
from pathlib import Path

import torch

from torchtitan.tools.logging import logger

LIBRARY_ENV = "GIFT_EXACT_FOLDED_CE_LIBRARY"
LIBRARY_SHA256 = "41357457bb454c721ee72b5b9e4151cff7056f5e3b473a1241b39c4161925a11"
_METHOD = 13
_ROWS = 8192
_VOCAB = 128256


@cache
def _load_launcher(library_path: str) -> Callable[..., int]:
    path = Path(library_path)
    if not path.is_absolute():
        raise ValueError(f"{LIBRARY_ENV} must be an absolute path")
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise RuntimeError(f"Cannot read {LIBRARY_ENV} at {path}") from error
    if digest != LIBRARY_SHA256:
        raise ValueError(f"{LIBRARY_ENV} SHA256 is {digest}, expected {LIBRARY_SHA256}")

    launch = ctypes.CDLL(str(path)).centered_ce_launch
    pointer = ctypes.c_void_p
    launch.argtypes = [
        pointer,
        pointer,
        pointer,
        pointer,
        ctypes.c_int,
        pointer,
        ctypes.c_int,
    ]
    launch.restype = ctypes.c_int
    return launch


@torch.library.custom_op("torchtitan::exact_folded_ce_consume", mutates_args={"logits"})
def _consume(
    logits: torch.Tensor,
    labels: torch.Tensor,
    scales: torch.Tensor,
    library_path: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    # The artifact is validated at loss construction. The tensor arguments
    # come from ExactFoldedCrossEntropy's guarded, first-order training path.
    losses = logits.new_empty((_ROWS,), dtype=torch.float32)
    status = _load_launcher(library_path)(
        logits.data_ptr(),
        labels.data_ptr(),
        losses.data_ptr(),
        scales.data_ptr(),
        _ROWS,
        torch.cuda.current_stream(logits.device).cuda_stream,
        _METHOD,
    )
    if status:
        raise RuntimeError(f"Exact folded CE CUDA launch failed with status {status}")
    main, mtp = losses.chunk(2)
    return main.sum(), mtp.sum()


@_consume.register_fake
def _consume_fake(logits, labels, scales, library_path):
    return logits.new_empty((), dtype=torch.float32), logits.new_empty(
        (), dtype=torch.float32
    )


class _PrecomputedGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, scales, loss):
        result = loss.compiled(logits, labels, scales, loss.library_path)
        torch.autograd.graph.increment_version(logits)
        ctx.save_for_backward(logits, scales)
        ctx.loss = loss
        return result

    @staticmethod
    def backward(ctx, *incoming):
        if torch.is_grad_enabled():
            raise RuntimeError("Exact consuming CE supports first-order backward only")
        logits, scales = ctx.saved_tensors
        loss = ctx.loss
        if not loss.scales_checked:
            # Compare the prepared scales against the actual VJP once, during
            # warmup. Later chunks use this same loss arithmetic and immediate backward.
            actual = torch.stack(incoming).float()
            if not torch.equal(actual, scales):
                raise RuntimeError(
                    "The prepared exact CE scales differ from the loss VJP"
                )
            loss.scales_checked = True
            logger.info(
                "LM_HEAD_EXACT_FOLDED_CE_ACTIVE method=cuda-13 backend=inductor "
                "rows=%d vocab=%d sha256=%s",
                _ROWS,
                _VOCAB,
                LIBRARY_SHA256,
            )
        return logits, None, None, None


class ExactFoldedCrossEntropy:
    def __init__(self, library_path: str) -> None:
        _load_launcher(library_path)
        self.library_path = library_path
        self.compiled = torch.compile(_consume, backend="inductor", fullgraph=True)
        self.scales_checked = False
        self._weight_key: tuple[torch.device, int, tuple[float, ...]] | None = None
        self._weights: torch.Tensor | None = None
        self._denominator: float | None = None
        self._scales: torch.Tensor | None = None

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        weights: Sequence[float],
        global_valid_tokens: torch.Tensor | float | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            not logits.requires_grad
            or not logits.is_cuda
            or logits.dtype != torch.bfloat16
            or logits.shape != (4, 2048, _VOCAB)
            or not logits.is_contiguous()
            or logits.storage_offset() % 8
            or labels.shape != logits.shape[:-1]
            or labels.device != logits.device
            or labels.dtype != torch.int64
            or not labels.is_contiguous()
            or len(weights) != 2
        ):
            raise ValueError(
                "Exact folded CE requires the C2/B2/S4096 MTP1 BF16 logits "
                "and matching contiguous int64 labels"
            )
        if (
            not isinstance(global_valid_tokens, float)
            or not math.isfinite(global_valid_tokens)
            or global_valid_tokens <= 0
        ):
            raise ValueError(
                "Exact folded CE requires a positive DP-reduced float token count"
            )

        # The trainer passes the same denominator to every chunk and microbatch
        # of a step. Include all inputs and the CUDA stream in the cache key.
        stream = torch.cuda.current_stream(logits.device)
        weight_key = (
            logits.device,
            stream.cuda_stream,
            tuple(weights),
        )
        if self._weight_key != weight_key:
            self._weights = logits.new_tensor(weights, dtype=torch.float32)
            self._weight_key = weight_key
            self._denominator = None
        if self._denominator != global_valid_tokens:
            assert self._weights is not None
            self._scales = self._weights / global_valid_tokens
            self._denominator = global_valid_tokens
        return _PrecomputedGradient.apply(
            logits.view(_ROWS, _VOCAB), labels.view(_ROWS), self._scales, self
        )
