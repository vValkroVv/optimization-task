from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch

from metric_sampler.capture.kernels import activation_ext
from metric_sampler.capture.kernels.graphed import GraphedComputation
from metric_sampler.capture.recorder import CaptureCapacityError, RankRecorder, ResolvedSlot
from metric_sampler.capture.stats.helpers.norms import MATRIX_DIMENSIONS
from metric_sampler.capture.stats.helpers.reductions import (
    add_segment_sums_of_squares,
    column_sums,
    entropy_sum,
    float64_scalar,
)
from metric_sampler.contract.metrics import INT32_MAX, WORDS_PER_FLOAT64

_ROUTER_SCORES_MEAN = "router/scores_mean"
_ROUTER_BALANCE = "monitor/router_balance_local"
_ROUTER_BALANCE_COUNTS = "monitor/router_balance_counts_local"
_ROUTER_BIAS = "monitor/router_bias_local"
_EXPERT_ACTIVATION_SUMSQ = "monitor/expert_activation_sumsq_local"
_EXPERT_ACTIVATION_COUNT = "monitor/expert_activation_count_local"
# What a router call sums, as router_sums lays it out: the normalized, raw and biased top-k score sums, the
# biased-score entropy sum, the z-loss, the sequence auxiliary loss and alpha.
_ROUTER_SUMS_WIDTH = 7
_VALIDITY_MASK_DTYPES = (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def router_sums(
    scores: torch.Tensor, biased_scores: torch.Tensor, normalized_topk: torch.Tensor, raw_topk: torch.Tensor,
    biased_topk: torch.Tensor, mask: torch.Tensor, z_loss: torch.Tensor, seq_aux_loss: torch.Tensor,
    alpha: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One router call's float64 [normalized, raw and biased top-k score sums over valid tokens,
    biased-score entropy sum, z-loss, sequence auxiliary loss, alpha], its valid token count and
    its float64 per-expert score sums."""
    tokens = scores.shape[0]
    topk_sums = [torch.where(mask[:, None], topk.reshape(tokens, topk.shape[-1]).double(), 0.).sum()
                 for topk in (normalized_topk, raw_topk, biased_topk)]
    sums = torch.stack((*topk_sums, entropy_sum(biased_scores), z_loss, seq_aux_loss, alpha))
    return sums, mask.sum(dtype=torch.int64), column_sums(scores)


class _HostFlag:
    """A boolean on its way to the host: a device flag is copied without waiting for it."""

    def __init__(self, flag: torch.Tensor) -> None:
        self._event = None
        self._host = flag
        if flag.is_cuda:  # pragma: no cover
            self._host = torch.empty((), dtype=torch.bool, pin_memory=True)
            self._host.copy_(flag, non_blocking=True)
            self._event = torch.cuda.Event()
            self._event.record()

    def value(self) -> bool:
        if self._event is not None:  # pragma: no cover
            self._event.synchronize()
        return bool(self._host.item())


@dataclass
class _RouterPopulation:
    values: torch.Tensor
    valid_tokens: torch.Tensor
    score_sums: torch.Tensor
    bias: torch.Tensor
    tokens: int
    calls: int
    # The extension wrote the population's recorder slots; the flush records nothing for it.
    recorded: bool = False


@dataclass(frozen=True)
class _RouterSlots:
    mean: ResolvedSlot
    balance: ResolvedSlot
    counts: ResolvedSlot
    bias: ResolvedSlot


@dataclass
class _ExpertPopulation:
    counts: list[int]
    tokens: int


class NativeRouterBalanceCalls(Protocol):
    """The extension's RouterBalanceCalls: the routed layers' balance observers, made from C++."""

    def flush(self) -> None: ...

    def has_population(self, layer: int, microbatch: int) -> bool: ...

    def router(self, layer: int) -> Callable[..., None]: ...


class NativeExpertActivationCalls(Protocol):
    """The extension's ExpertActivationCalls: the MoE layers' expert activation observers, made from C++."""

    def flush(self) -> list[tuple[int, int, list[int]]]: ...

    def has_population(self, layer: int, microbatch: int) -> bool: ...

    def experts(self, layer: int) -> Callable[..., None]: ...


class ExpertBalanceStats:
    """Rank/microbatch router operands and local expert moments before monitor reductions."""

    def __init__(self, recorder: RankRecorder, experts: int, local_experts: int, expert_offset: int) -> None:
        if not 0 <= expert_offset < experts or not 0 < local_experts <= experts - expert_offset:
            raise ValueError("expert ownership must be a nonempty slice of global expert IDs")
        self.recorder = recorder
        self.experts = experts
        self.local_experts = local_experts
        self.expert_offset = expert_offset
        self._routers: dict[tuple[int, int], _RouterPopulation] = {}
        self._router_graphs: dict[
            tuple[object, ...], GraphedComputation[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        ] = {}
        # Per (layer, microbatch): the extension's population sums, reused every step, and the recorder slots.
        self._router_buffers: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._router_slots: dict[tuple[int, int], _RouterSlots] = {}
        # The microbatch's validity masks by tensor: every layer is given the same padding mask.
        self._masks: dict[int, tuple[torch.Tensor, int, torch.Tensor]] = {}
        self._activations: dict[tuple[int, int], _ExpertPopulation] = {}
        self._activation_slots: dict[tuple[int, int], ResolvedSlot] = {}
        # Integer masks are checked on the device and reported once per step, never with a sync per call.
        self._nonbinary_mask: torch.Tensor | None = None
        # The extension's flag of an integer mask value other than zero or one, which it reads in place.
        self._nonbinary_flag: torch.Tensor | None = None
        self._pending_check: _HostFlag | None = None
        # The extension's observers, once native_calls and native_expert_calls made them; their populations end
        # with this object's.
        self._native: NativeRouterBalanceCalls | None = None
        self._native_experts: NativeExpertActivationCalls | None = None

    def native_calls(self, sampler: object) -> NativeRouterBalanceCalls | None:  # pragma: no cover
        """The extension's balance observers, which take a call as observe_router does, without a Python call:
        only in a training forward with grad, as `sampler`'s _capture_forward_active and _grad_forward tell, in
        `sampler`'s microbatch. A call the extension does not take goes to observe_router. None without the
        extension or over a host recorder: the extension writes the recorder's CUDA words."""
        extension = activation_ext.prebuilt()
        if not extension or not self.recorder.buffer.is_cuda:
            return None
        self._native = extension.RouterBalanceCalls(
            sampler, self.recorder, self._native_slots, self.observe_router, self._routers, self.experts,
            self.recorder.recorded_flags(), self._nonbinary_flag_on, CaptureCapacityError,
            torch._C._current_graph_task_id,
        )
        return self._native

    def native_expert_calls(self, sampler: object) -> NativeExpertActivationCalls | None:  # pragma: no cover
        """The extension's expert activation observers, which take a call as observe_experts does, without a
        Python call: only in a training forward with grad, as `sampler`'s _capture_forward_active and _grad_forward
        tell, in `sampler`'s microbatch. A call the extension does not take goes to observe_experts. None without
        the extension or over a host recorder: the extension writes the recorder's CUDA words."""
        extension = activation_ext.prebuilt()
        if not extension or not self.recorder.buffer.is_cuda:
            return None
        self._native_experts = extension.ExpertActivationCalls(
            sampler, self.recorder, self._native_expert_slot, self.observe_experts, self._activations,
            self.local_experts, self.expert_offset, self.recorder.recorded_flags(), CaptureCapacityError,
            torch._C._current_graph_task_id,
        )
        return self._native_experts

    def _native_expert_slot(self, layer: int, microbatch: int) -> tuple[int, int]:  # pragma: no cover
        """The (offset, slot index) of the activation slot of an expert module call ExpertActivationCalls takes."""
        slot = self._activation_slot(layer, microbatch)
        return slot.offset, slot.slot_index

    def _native_slots(self, layer: int, microbatch: int) -> tuple[list[int], list[int], list[int]]:  # pragma: no cover
        """The (offsets, widths, slot indices) of the slots of a router RouterBalanceCalls takes."""
        slots = self._slots(layer, microbatch)
        resolved = (slots.mean, slots.balance, slots.counts, slots.bias)
        return ([slot.offset for slot in resolved], [slot.width for slot in resolved],
                [slot.slot_index for slot in resolved])

    def observe_router(
        self, layer: int, microbatch: int, scores: torch.Tensor, biased_scores: torch.Tensor,
        normalized_topk: torch.Tensor, raw_topk: torch.Tensor, biased_topk: torch.Tensor,
        valid: torch.Tensor | None, bias: torch.Tensor | None,
        z_loss: torch.Tensor, seq_aux_loss: torch.Tensor | None, alpha: float,
    ) -> None:
        if scores.ndim < MATRIX_DIMENSIONS or scores.shape[-1] != self.experts or not scores.is_floating_point():
            raise ValueError("router scores must contain every expert for every source token")
        if (biased_scores.shape != scores.shape or biased_scores.device != scores.device
                or not biased_scores.is_floating_point()):
            raise ValueError("biased router scores must match the complete source scores")
        tokens = scores.numel() // self.experts
        key = (layer, microbatch)
        previous = self._routers.get(key)
        total_tokens = tokens + (0 if previous is None else previous.tokens)
        total_calls = 1 if previous is None else previous.calls + 1
        if total_tokens > INT32_MAX or total_calls > INT32_MAX:
            raise CaptureCapacityError("router population exceeds Int32 token or call counts")
        device = scores.device
        self._check_validity(valid, tokens, device)
        self._check_topk_and_bias(normalized_topk, raw_topk, biased_topk, bias, tokens, device, previous)
        if scores.is_cuda:  # pragma: no cover
            extension = activation_ext.prebuilt()
            if (extension and scores.dtype == torch.float32 and biased_scores.dtype == torch.float32
                    and scores.stride(-1) == 1 and scores.numel()):
                # One extension call: the sums go straight into the population in two launches, which
                # also write the population's recorder slots, so the flush has nothing to record for it.
                population = previous
                slots = self._slots(layer, microbatch)
                if population is None:
                    buffers = self._router_buffers.get(key)
                    if buffers is None:
                        buffers = self._router_buffers[key] = (
                            torch.empty(_ROUTER_SUMS_WIDTH, dtype=torch.float64, device=device),
                            torch.empty((), dtype=torch.int64, device=device),
                            torch.empty(self.experts, dtype=torch.float64, device=device),
                        )
                    population = self._routers[key] = _RouterPopulation(
                        *buffers, self.recorder.view(slots.bias), 0, 0, recorded=True,
                    )
                self.recorder.mark_recorded(
                    [slots.mean.slot_index, slots.balance.slot_index, slots.counts.slot_index, slots.bias.slot_index])
                # The mask is read as it is, each layer's a new tensor: an integer one's values other than zero
                # and one go to the flag require_binary_masks reads, without a conversion or check per call.
                extension.router_accumulate(
                    scores, biased_scores, normalized_topk, raw_topk, biased_topk, valid, z_loss, seq_aux_loss, alpha,
                    population.values, population.valid_tokens, population.score_sums, previous is None,
                    self.recorder.buffer, slots.mean.offset, slots.balance.offset, slots.counts.offset,
                    slots.bias.offset, total_tokens, total_calls, bias,
                    None if valid is None or valid.dtype == torch.bool else self._nonbinary_flag_on(device),
                )
                population.tokens = total_tokens
                population.calls = total_calls
                return
        mask = self._validity_mask(valid, device)
        # One no_grad scope instead of a detach per operand: each detach is an op on every call.
        with torch.no_grad():
            scores = scores.reshape(-1, self.experts)
            biased_scores = biased_scores.reshape_as(scores)
            current_bias = torch.zeros(self.experts, device=device) if bias is None else bias.float().clone()
            operands = (
                scores, biased_scores, normalized_topk, raw_topk, biased_topk,
                torch.ones(tokens, dtype=torch.bool, device=device) if mask is None else mask,
                float64_scalar(z_loss, device),
                torch.full((), float("nan"), device=device, dtype=torch.float64) if seq_aux_loss is None
                else float64_scalar(seq_aux_loss, device),
                torch.full((), alpha, device=device, dtype=torch.float64),
            )
            values, valid_tokens, score_sums = self._router_sums(operands)
            if previous is None:
                # Copies: a replayed graph overwrites its outputs on the next call.
                self._routers[key] = _RouterPopulation(
                    values.clone(), valid_tokens.clone(), score_sums.clone(), current_bias, tokens, 1,
                )
            else:
                previous.values += values
                previous.valid_tokens += valid_tokens
                previous.score_sums += score_sums
                previous.tokens = total_tokens
                previous.calls = total_calls

    def _check_topk_and_bias(
        self, normalized_topk: torch.Tensor, raw_topk: torch.Tensor, biased_topk: torch.Tensor,
        bias: torch.Tensor | None, tokens: int, device: torch.device, previous: _RouterPopulation | None,
    ) -> None:
        for topk in (normalized_topk, raw_topk, biased_topk):
            if (topk.ndim < MATRIX_DIMENSIONS or topk.shape[-1] == 0 or topk.numel() != tokens * topk.shape[-1]
                    or topk.device != device or not topk.is_floating_point()):
                raise ValueError("router top-k values must cover every source token on its device")
        if bias is not None and (
                bias.shape != (self.experts,) or bias.device != device or not bias.is_floating_point()):
            raise ValueError("router bias must contain every expert on the source device")
        if previous is not None and not torch.equal(
                previous.bias, torch.zeros(self.experts, device=device) if bias is None else bias.detach().float()):
            raise ValueError("router bias changed within one microbatch; a single snapshot would lose information")

    @staticmethod
    def _check_validity(valid: torch.Tensor | None, tokens: int, device: torch.device) -> None:
        if valid is None:
            return
        if valid.dtype not in _VALIDITY_MASK_DTYPES or valid.device != device:
            raise ValueError("router validity mask must be Boolean or binary integer on the source device")
        if valid.numel() != tokens:
            raise ValueError("router validity mask must cover every source token")

    def _validity_mask(self, valid: torch.Tensor | None, device: torch.device) -> torch.Tensor | None:
        """The router's validity mask as one Boolean per token, converted and checked once per tensor in a
        microbatch: a repeat of an integer mask's binary check, an accumulation, does not change it."""
        if valid is None:
            return None
        cached = self._masks.get(id(valid))
        if cached is not None and cached[0] is valid and cached[1] == valid._version:
            mask = cached[2]
        else:
            with torch.no_grad():
                if valid.dtype != torch.bool:
                    nonbinary = ((valid != 0) & (valid != 1)).any()
                    self._nonbinary_mask = (
                        nonbinary if self._nonbinary_mask is None else self._nonbinary_mask | nonbinary)
                mask = valid.reshape(-1).bool()
            self._masks[id(valid)] = (valid, valid._version, mask)
        return mask

    def _nonbinary_flag_on(self, device: torch.device) -> torch.Tensor:  # pragma: no cover
        if self._nonbinary_flag is None:
            self._nonbinary_flag = torch.zeros(1, dtype=torch.int32, device=device)
        return self._nonbinary_flag

    def _slots(self, layer: int, microbatch: int) -> _RouterSlots:  # pragma: no cover
        slots = self._router_slots.get((layer, microbatch))
        if slots is None:
            resolve = self.recorder.resolve
            slots = self._router_slots[layer, microbatch] = _RouterSlots(
                resolve(_ROUTER_SCORES_MEAN, layer, microbatch),
                resolve(_ROUTER_BALANCE, layer, microbatch),
                resolve(_ROUTER_BALANCE_COUNTS, layer, microbatch),
                resolve(_ROUTER_BIAS, layer, microbatch),
            )
        return slots

    def _router_sums(self, operands: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if operands[0].is_cuda:  # pragma: no cover
            # Dozens of small launches per routed layer: one graph replay instead.
            key = tuple((operand.shape, operand.stride(), operand.dtype, operand.device) for operand in operands)
            graphed = self._router_graphs.get(key)
            if graphed is None:
                graphed = self._router_graphs[key] = GraphedComputation(router_sums, *operands)
            return graphed(*operands)
        return router_sums(*operands)

    def observe_experts(self, layer: int, microbatch: int, intermediate: torch.Tensor, counts: list[int]) -> None:
        if len(counts) != self.local_experts or any(type(count) is not int or count < 0 for count in counts):
            raise ValueError("expert token counts must be exact nonnegative integers for every local expert")
        tokens = sum(counts)
        key = (layer, microbatch)
        previous = self._activations.get(key)
        total_tokens = tokens + (0 if previous is None else previous.tokens)
        if total_tokens > INT32_MAX:
            raise CaptureCapacityError("expert population exceeds Int32 token counts")
        if (not isinstance(intermediate, torch.Tensor) or intermediate.ndim != MATRIX_DIMENSIONS
                or intermediate.shape[0] != tokens):
            raise ValueError("expert capture requires the complete unpadded two-dimensional intermediate")
        slot = self._activation_slot(layer, microbatch)
        # The step's buffer starts zeroed: only the owned experts' float64 words are written.
        add_segment_sums_of_squares(
            intermediate, counts, self.recorder.buffer, slot.offset + WORDS_PER_FLOAT64 * self.expert_offset,
            previous is None,
        )
        if previous is None:
            self.recorder.mark_recorded([slot.slot_index])
            self._activations[key] = _ExpertPopulation(list(counts), tokens)
        else:
            previous.counts = [total + count for total, count in zip(previous.counts, counts)]
            previous.tokens = total_tokens

    def _activation_slot(self, layer: int, microbatch: int) -> ResolvedSlot:
        slot = self._activation_slots.get((layer, microbatch))
        if slot is None:
            slot = self._activation_slots[layer, microbatch] = self.recorder.resolve(
                _EXPERT_ACTIVATION_SUMSQ, layer, microbatch)
        return slot

    def flush(self) -> None:
        if self._native is not None:  # pragma: no cover
            self._native.flush()
        for (layer, microbatch), population in self._routers.items():
            if population.recorded:  # pragma: no cover
                continue
            device = population.values.device
            counts = torch.stack((torch.full((), population.tokens, device=device, dtype=torch.int64),
                                  population.valid_tokens,
                                  torch.full((), population.calls, device=device, dtype=torch.int64))).to(torch.int32)
            mean = (population.score_sums / population.tokens if population.tokens
                    else torch.full_like(population.score_sums, float("nan")))
            self.recorder.record(_ROUTER_SCORES_MEAN, layer, microbatch, mean.float())
            self.recorder.record(_ROUTER_BALANCE, layer, microbatch, population.values.contiguous().view(torch.int32))
            self.recorder.record(_ROUTER_BALANCE_COUNTS, layer, microbatch, counts)
            self.recorder.record(_ROUTER_BIAS, layer, microbatch, population.bias)
        expert_counts = [(layer, microbatch, activations.counts)
                         for (layer, microbatch), activations in self._activations.items()]
        if self._native_experts is not None:  # pragma: no cover
            expert_counts += self._native_experts.flush()
        for layer, microbatch, local_counts in expert_counts:
            words = [0] * self.experts
            words[self.expert_offset:self.expert_offset + self.local_experts] = local_counts
            self.recorder.record_host_words(_EXPERT_ACTIVATION_COUNT, layer, microbatch, words)
        self._routers.clear()
        self._activations.clear()
        self._masks.clear()

    def require_binary_masks(self) -> None:
        """Raises if an integer router mask held other than 0 or 1. The masks observed since the last
        call are checked at the next call (or by finish): their flag reaches the host without the
        step waiting for it, and by then it has long landed."""
        pending, self._pending_check = self._pending_check, None
        nonbinary, self._nonbinary_mask = self._nonbinary_mask, None
        if self._nonbinary_flag is not None:  # pragma: no cover
            # What the extension flagged since the last call, and the flag cleared for the masks after it.
            flagged = self._nonbinary_flag[0] != 0
            self._nonbinary_flag.zero_()
            nonbinary = flagged if nonbinary is None else nonbinary | flagged
        if nonbinary is not None:
            self._pending_check = _HostFlag(nonbinary)
        self._raise_if_nonbinary(pending)

    def finish(self) -> None:
        """Checks the masks of the last step that called require_binary_masks."""
        pending, self._pending_check = self._pending_check, None
        self._raise_if_nonbinary(pending)

    @staticmethod
    def _raise_if_nonbinary(flag: _HostFlag | None) -> None:
        if flag is not None and flag.value():
            raise ValueError("integer router validity mask must contain only zero or one")
