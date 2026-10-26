from __future__ import annotations

import math
from dataclasses import dataclass
from types import ModuleType

import torch

from metric_sampler.capture.recorder import RankRecorder, ResolvedSlot
from metric_sampler.capture.stats.helpers.histograms import bin_indices
from metric_sampler.contract.metrics import (
    EXPERT_LOSS_BINS,
    LOSS_POSITION_BUCKETS,
    NO_LAYER,
    NO_MICROBATCH,
    ROUTER_ENTROPY_BINS,
    SAMPLE_LOSS_BINS,
    TOKEN_LOSS_BINS,
    WORST_SAMPLE_COUNT,
    WORST_SAMPLE_FIELDS,
)

# The layer of a metric recorded once per microbatch, not per layer.
# A sample's id is [stream-local shard id, index in shard].
_SAMPLE_ID_COLUMNS = 2
# A worst sample's statistics are [mean loss, max loss].
_WORST_SAMPLE_STAT_FIELDS = 2


def _token_loss_kernel() -> ModuleType:  # pragma: no cover
    from metric_sampler.capture.kernels import token_loss_kernel

    return token_loss_kernel


@dataclass(frozen=True)
class _MainLossSlots:
    token_histogram: ResolvedSlot
    stream_sum: ResolvedSlot
    stream_sumsq: ResolvedSlot
    stream_count: ResolvedSlot
    stream_samples: ResolvedSlot
    position_sum: ResolvedSlot
    position_count: ResolvedSlot


@dataclass(frozen=True)
class RoutingTarget:
    """Where one routing call of a layer is recorded: the layer's position in the top-k buffer, the buffer, and the
    microbatch's expert counts, expert-by-stream counts and entropy histogram slots."""

    position: int
    topk_buffer: torch.Tensor
    counts: ResolvedSlot
    stream_counts: ResolvedSlot
    entropy_histogram: ResolvedSlot
    stream_ids: torch.Tensor | None
    tokens_per_sample: int
    streams: int


@dataclass(frozen=True)
class _SegmentExpertTable:
    layers: list[int]
    positions: torch.Tensor
    offsets: torch.Tensor
    slot_indices: list[int]
    detailed_slot_indices: list[int]


class TokenLossStats:
    """Per-microbatch token-loss and expert-loss statistics, accumulated entirely on the device.

    Every public method is step-hot: no boolean-mask indexing, `nonzero`, `bincount`, `.item()`
    or similar host syncs; token validity is folded into 0/1 weights fed to `index_add_`.

    Without `token_loss` only the expert-loss statistics of the routed layers are recorded: the
    token histograms, stream, position, sample and MTP statistics are left out."""

    def __init__(
        self,
        recorder: RankRecorder,
        routed_layers: dict[int, int],
        mtp_layers: list[int],
        experts: int,
        top_k: int,
        streams: int,
        token_loss: bool = True,
    ) -> None:
        self._recorder = recorder
        self._token_loss = token_loss
        self._routed_layers = routed_layers
        self._mtp_layers = mtp_layers
        self._layer_positions = {layer: position for position, layer in enumerate(routed_layers)}
        self._experts = experts
        self._top_k = top_k
        self._streams = streams
        self._detailed = False
        self._microbatch = NO_MICROBATCH
        self._stream_ids: torch.Tensor | None = None
        self._sample_ids: torch.Tensor | None = None
        self._observed_layers: set[int] = set()
        self._topk_buffer: torch.Tensor | None = None
        self._topk_tokens: int | None = None
        self._routing_slots: dict[tuple[int, int], tuple[ResolvedSlot, ResolvedSlot, ResolvedSlot]] = {}
        self._main_loss_slots: dict[int, _MainLossSlots] = {}
        self._microbatch_loss_slots: dict[int, tuple[ResolvedSlot, ResolvedSlot]] = {}
        self._mtp_histogram_slots: dict[int, list[ResolvedSlot]] = {}
        self._mtp_histogram_offsets: dict[int, torch.Tensor] = {}
        self._segment_expert_tables: dict[tuple[int, int], _SegmentExpertTable] = {}

    def begin_step(self, detailed: bool) -> None:
        self._detailed = detailed

    def begin_microbatch(
        self, microbatch: int, stream_ids: torch.Tensor | None, sample_ids: torch.Tensor | None = None
    ) -> None:
        if stream_ids is None and self._streams != 1:
            raise ValueError("stream_ids is None but streams != 1")
        if stream_ids is not None:
            if stream_ids.ndim != 1:
                raise ValueError("stream_ids must be a one-dimensional sample vector")
            if stream_ids.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
                raise ValueError("stream_ids must have an integer dtype")
            torch._assert_async(
                ((stream_ids >= 0) & (stream_ids < self._streams)).all(),
                "stream_ids contain an id outside the configured stream range",
            )
            stream_ids = stream_ids.to(dtype=torch.int32).contiguous()
        self._microbatch = microbatch
        self._stream_ids = stream_ids
        self._sample_ids = sample_ids
        self._observed_layers = set()
        self._topk_tokens = None

    def observe_routing(self, layer: int, topk_indices: torch.Tensor, scores: torch.Tensor) -> None:
        tokens = topk_indices.shape[0]
        target = self.begin_routing(layer, tokens, topk_indices.device)

        # Detached so the stats stay out of the autograd graph: joining it would hold memory and,
        # under activation checkpointing, change the tensors saved by the original forward.
        scores = scores.detach()
        if topk_indices.device.type == "cuda":  # pragma: no cover
            self._observe_routing_cuda(target, topk_indices, scores)
            return
        expert_indices = topk_indices.flatten().to(torch.long)
        counts = self._recorder.accumulator("router/tokens_per_expert", layer, self._microbatch)
        counts.index_add_(0, expert_indices, torch.ones_like(expert_indices, dtype=counts.dtype))
        if self._streams == 1:
            combined = expert_indices
        else:
            assert self._stream_ids is not None
            stream_index = self._stream_ids.to(device=topk_indices.device, dtype=torch.long).repeat_interleave(
                target.tokens_per_sample
            )
            combined = expert_indices * self._streams + stream_index.unsqueeze(-1).expand(-1, self._top_k).flatten()
        stream_counts = self._recorder.accumulator(
            "router/stream_tokens_per_expert", layer, self._microbatch
        )
        stream_counts.index_add_(0, combined, torch.ones_like(combined, dtype=stream_counts.dtype))
        target.topk_buffer[target.position, :tokens].copy_(topk_indices)
        self._record_entropy_histogram(layer, scores)

    @staticmethod
    def _tokens_per_sample(tokens: int, stream_ids: torch.Tensor) -> int:
        samples = stream_ids.numel()
        if samples == 0 or tokens % samples:
            raise ValueError(f"routing token count {tokens} must be divisible by stream sample count {samples}")
        return tokens // samples

    def begin_routing(self, layer: int, tokens: int, device: torch.device) -> RoutingTarget:
        """Registers a routing call of `layer` over `tokens` tokens in the microbatch and returns where it is
        recorded, the top-k buffer large enough for it."""
        position = self._layer_positions[layer]
        if self._stream_ids is None:
            if self._streams != 1:
                raise ValueError("stream_ids is None but streams != 1")
            tokens_per_sample = 1
        else:
            if self._stream_ids.device != device:
                raise ValueError("stream_ids and routed tokens must be on the same device")
            tokens_per_sample = self._tokens_per_sample(tokens, self._stream_ids)
        if self._topk_tokens is not None and tokens != self._topk_tokens:
            raise ValueError(f"observe_routing for layer {layer} got {tokens} tokens, expected {self._topk_tokens}")
        if self._topk_buffer is None or tokens > self._topk_buffer.shape[1]:
            self._topk_buffer = torch.zeros(
                (len(self._routed_layers), tokens, self._top_k), dtype=torch.int32, device=device
            )
        self._topk_tokens = tokens
        self._observed_layers.add(layer)
        key = (layer, self._microbatch)
        slots = self._routing_slots.get(key)
        if slots is None:
            resolve = self._recorder.resolve
            slots = self._routing_slots[key] = (
                resolve("router/tokens_per_expert", layer, self._microbatch),
                resolve("router/stream_tokens_per_expert", layer, self._microbatch),
                resolve("router/entropy_hist", layer, self._microbatch),
            )
        return RoutingTarget(
            position, self._topk_buffer, *slots, self._stream_ids, tokens_per_sample, self._streams
        )

    def _record_entropy_histogram(self, layer: int, scores: torch.Tensor) -> None:
        # Normalize in FP64: subtracting two raw-score logarithms loses bins near an edge.
        scores = scores.to(torch.float64)
        probabilities = scores / scores.sum(dim=-1, keepdim=True)
        entropy = -torch.special.xlogy(probabilities, probabilities).sum(dim=-1) / math.log(self._experts)
        bin_index = bin_indices(entropy, ROUTER_ENTROPY_BINS)
        histogram = self._recorder.accumulator("router/entropy_hist", layer, self._microbatch)
        histogram.index_add_(0, bin_index, torch.ones_like(bin_index, dtype=histogram.dtype))

    def _stream_index(self, batch_size: int, sequence_length: int, device: torch.device) -> torch.Tensor:
        if self._stream_ids is None:
            return torch.zeros((batch_size, sequence_length), dtype=torch.long, device=device)
        return self._stream_ids.to(torch.long).unsqueeze(-1).expand(batch_size, sequence_length)

    def observe_loss(self, loss: torch.Tensor, labels_valid: torch.Tensor) -> None:
        loss = loss.detach()
        if loss.device.type == "cuda":  # pragma: no cover
            self._observe_loss_cuda(loss, labels_valid)
            return

        finite = torch.isfinite(loss)
        valid = labels_valid & finite
        # Zeroed (not masked-out) so every downstream sum/histogram already excludes invalid
        # and non-finite tokens without needing a second weight multiply.
        zeroed_loss = torch.where(valid, loss, torch.zeros_like(loss))

        batch_size, sequence_length = loss.shape[1], loss.shape[2]
        stream_index = self._stream_index(batch_size, sequence_length, loss.device)

        if self._token_loss:
            nonfinite = (labels_valid & ~finite).sum()
            self._recorder.record("loss/nonfinite_tokens", NO_LAYER, self._microbatch, nonfinite.to(torch.int32))
            self._record_token_max(zeroed_loss[0])
            self._record_main_stream_stats(zeroed_loss[0], valid[0], stream_index)
            self._record_position_stats(zeroed_loss[0], valid[0], sequence_length)
            if self._detailed:
                self._record_sample_histogram_by_stream(
                    zeroed_loss[0], valid[0], stream_index, labels_valid[0] & ~finite[0]
                )
            self._record_mtp_histograms(zeroed_loss, valid)
        self._record_expert_loss(zeroed_loss, valid, stream_index)

    def _record_token_max(self, main_zeroed_loss: torch.Tensor) -> None:
        # main_zeroed_loss is already zeroed at invalid/non-finite positions, and cross-entropy is
        # never negative, so a plain max over it already reads 0 when nothing is valid.
        self._recorder.record("loss/token_max", NO_LAYER, self._microbatch, main_zeroed_loss.max())

    def _record_main_stream_stats(
        self, main_zeroed_loss: torch.Tensor, main_valid: torch.Tensor, stream_index: torch.Tensor
    ) -> None:
        bin_index = bin_indices(main_zeroed_loss, TOKEN_LOSS_BINS)
        combined_index = stream_index * TOKEN_LOSS_BINS.count + bin_index
        token_histogram = self._recorder.accumulator("loss/token_hist_by_stream", NO_LAYER, self._microbatch)
        token_histogram.index_add_(0, combined_index.flatten(), main_valid.to(token_histogram.dtype).flatten())

        flat_stream_index = stream_index.flatten()
        stream_sum = self._recorder.accumulator("loss/stream_sum", NO_LAYER, self._microbatch)
        stream_sum.index_add_(0, flat_stream_index, main_zeroed_loss.flatten())
        stream_sum_of_squares = self._recorder.accumulator("loss/stream_sumsq", NO_LAYER, self._microbatch)
        stream_sum_of_squares.index_add_(0, flat_stream_index, (main_zeroed_loss * main_zeroed_loss).flatten())
        stream_count = self._recorder.accumulator("loss/stream_count", NO_LAYER, self._microbatch)
        stream_count.index_add_(0, flat_stream_index, main_valid.to(stream_count.dtype).flatten())

        sample_stream_index = stream_index[:, 0]
        stream_samples = self._recorder.accumulator("loss/stream_samples", NO_LAYER, self._microbatch)
        stream_samples.index_add_(
            0, sample_stream_index, torch.ones_like(sample_stream_index, dtype=stream_samples.dtype)
        )

    def _record_position_stats(
        self, main_zeroed_loss: torch.Tensor, main_valid: torch.Tensor, sequence_length: int
    ) -> None:
        # Position within the packed sequence, not within a document (see loss/position_sum).
        position = torch.arange(sequence_length, device=main_zeroed_loss.device)
        bucket = (position * LOSS_POSITION_BUCKETS // sequence_length).expand_as(main_zeroed_loss).flatten()

        position_sum = self._recorder.accumulator("loss/position_sum", NO_LAYER, self._microbatch)
        position_sum.index_add_(0, bucket, main_zeroed_loss.flatten())
        position_count = self._recorder.accumulator("loss/position_count", NO_LAYER, self._microbatch)
        position_count.index_add_(0, bucket, main_valid.to(position_count.dtype).flatten())

    def _record_sample_histogram_by_stream(
        self, main_zeroed_loss: torch.Tensor, main_valid: torch.Tensor, stream_index: torch.Tensor,
        main_nonfinite: torch.Tensor,
    ) -> None:
        sample_count = main_valid.to(main_zeroed_loss.dtype).sum(dim=-1)
        sample_mean = main_zeroed_loss.sum(dim=-1) / sample_count.clamp_min(1)
        has_valid = sample_count > 0
        bin_index = bin_indices(sample_mean, SAMPLE_LOSS_BINS)
        combined_index = stream_index[:, 0] * SAMPLE_LOSS_BINS.count + bin_index
        histogram = self._recorder.accumulator("loss/sample_hist_by_stream", NO_LAYER, self._microbatch)
        histogram.index_add_(0, combined_index, has_valid.to(histogram.dtype))

        if self._sample_ids is not None:
            self._record_worst_samples(
                self._sample_ids, main_zeroed_loss, main_valid, main_nonfinite, stream_index, sample_mean,
                sample_count,
            )

    def _record_worst_samples(
        self, sample_ids: torch.Tensor, loss: torch.Tensor, valid: torch.Tensor, nonfinite: torch.Tensor,
        stream_index: torch.Tensor, sample_mean: torch.Tensor, sample_count: torch.Tensor,
    ) -> None:
        if sample_ids.shape != (loss.shape[0], _SAMPLE_ID_COLUMNS):
            raise ValueError("sample_ids must contain [stream-local shard_id, index_in_shard] for each sample")
        if sample_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("sample_ids must be integer tensors")
        nonfinite_count = nonfinite.sum(dim=-1)
        scores = torch.where(sample_count > 0, sample_mean, -torch.inf)
        scores = torch.where(nonfinite_count > 0, torch.inf, scores)
        selected = scores.topk(min(WORST_SAMPLE_COUNT, loss.shape[0])).indices
        max_loss, max_position = torch.where(valid, loss, -torch.inf).max(dim=-1)
        position = torch.where(nonfinite_count > 0, nonfinite.to(torch.int32).argmax(dim=-1), max_position)

        info = self._recorder.accumulator("loss/worst_sample_info", NO_LAYER, self._microbatch)
        info = info.view(WORST_SAMPLE_COUNT, WORST_SAMPLE_FIELDS)
        stats = self._recorder.accumulator("loss/worst_sample_stats", NO_LAYER, self._microbatch)
        stats = stats.view(WORST_SAMPLE_COUNT, _WORST_SAMPLE_STAT_FIELDS)
        info.zero_()
        stats.zero_()
        selected_count = selected.numel()
        info[:selected_count].copy_(torch.stack((
            stream_index[selected, 0], sample_ids[selected, 0], sample_ids[selected, 1],
            position[selected], sample_count[selected].to(torch.int64), nonfinite_count[selected],
        ), dim=-1))
        stats[:selected_count].copy_(torch.stack((sample_mean[selected], max_loss[selected]), dim=-1))

    def _record_mtp_histograms(self, zeroed_loss: torch.Tensor, valid: torch.Tensor) -> None:
        for head, mtp_layer in enumerate(self._mtp_layers, start=1):
            bin_index = bin_indices(zeroed_loss[head], TOKEN_LOSS_BINS)
            histogram = self._recorder.accumulator("loss/mtp_token_hist", mtp_layer, self._microbatch)
            histogram.index_add_(0, bin_index.flatten(), valid[head].to(histogram.dtype).flatten())

    def _record_expert_loss(
        self, zeroed_loss: torch.Tensor, valid: torch.Tensor, stream_index: torch.Tensor
    ) -> None:
        flat_stream_index = stream_index.flatten()
        assignments = {
            segment: self._assignments(zeroed_loss[segment], valid[segment])
            for segment in set(self._routed_layers.values())
        }
        for layer, segment in self._routed_layers.items():
            if layer not in self._observed_layers:
                raise ValueError(f"routing for layer {layer} was not observed this microbatch")
            assert self._topk_buffer is not None

            assignment_loss, assignment_loss_squared, assignment_weight = assignments[segment]
            expert_index = self._topk_buffer[self._layer_positions[layer], :self._topk_tokens].flatten()

            expert_sum = self._recorder.accumulator("expert_loss/sum", layer, self._microbatch)
            expert_sum.index_add_(0, expert_index, assignment_loss)
            expert_sum_of_squares = self._recorder.accumulator("expert_loss/sumsq", layer, self._microbatch)
            expert_sum_of_squares.index_add_(0, expert_index, assignment_loss_squared)
            expert_count = self._recorder.accumulator("expert_loss/count", layer, self._microbatch)
            expert_count.index_add_(0, expert_index, assignment_weight)

            if self._detailed:
                self._record_detailed_expert_loss(
                    layer, expert_index, assignment_loss, assignment_weight, flat_stream_index
                )

    def _assignments(
        self, segment_zeroed_loss: torch.Tensor, segment_valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Per-token loss/sq-loss/weight repeated per top-k assignment, in the same [token, k]
        # order as the flattened routing indices; shared across a segment's layers.
        assignment_loss = segment_zeroed_loss.flatten().unsqueeze(-1).expand(-1, self._top_k).flatten()
        assignment_weight = segment_valid.flatten().unsqueeze(-1).expand(-1, self._top_k).flatten().to(torch.int32)
        return assignment_loss, assignment_loss * assignment_loss, assignment_weight

    def _record_detailed_expert_loss(
        self,
        layer: int,
        expert_index: torch.Tensor,
        assignment_loss: torch.Tensor,
        assignment_weight: torch.Tensor,
        flat_stream_index: torch.Tensor,
    ) -> None:
        bin_index = bin_indices(assignment_loss, EXPERT_LOSS_BINS)
        histogram = self._recorder.accumulator("expert_loss/hist", layer, self._microbatch)
        histogram_index = expert_index * EXPERT_LOSS_BINS.count + bin_index
        histogram.index_add_(0, histogram_index, assignment_weight)

        assignment_stream_index = flat_stream_index.unsqueeze(-1).expand(-1, self._top_k).flatten()
        combined_stream_index = expert_index * self._streams + assignment_stream_index

        expert_stream_count = self._recorder.accumulator("expert_loss/stream_count", layer, self._microbatch)
        expert_stream_count.index_add_(0, combined_stream_index, assignment_weight)
        expert_stream_sum = self._recorder.accumulator("expert_loss/stream_sum", layer, self._microbatch)
        expert_stream_sum.index_add_(0, combined_stream_index, assignment_loss)

    def _resolve_main_loss_slots(self) -> _MainLossSlots:  # pragma: no cover
        slots = self._main_loss_slots.get(self._microbatch)
        if slots is not None:
            return slots
        resolve = self._recorder.resolve
        slots = _MainLossSlots(
            token_histogram=resolve("loss/token_hist_by_stream", NO_LAYER, self._microbatch),
            stream_sum=resolve("loss/stream_sum", NO_LAYER, self._microbatch),
            stream_sumsq=resolve("loss/stream_sumsq", NO_LAYER, self._microbatch),
            stream_count=resolve("loss/stream_count", NO_LAYER, self._microbatch),
            stream_samples=resolve("loss/stream_samples", NO_LAYER, self._microbatch),
            position_sum=resolve("loss/position_sum", NO_LAYER, self._microbatch),
            position_count=resolve("loss/position_count", NO_LAYER, self._microbatch),
        )
        self._main_loss_slots[self._microbatch] = slots
        return slots

    def _resolve_microbatch_loss_slots(self) -> tuple[ResolvedSlot, ResolvedSlot]:  # pragma: no cover
        slots = self._microbatch_loss_slots.get(self._microbatch)
        if slots is None:
            resolve = self._recorder.resolve
            slots = (
                resolve("loss/nonfinite_tokens", NO_LAYER, self._microbatch),
                resolve("loss/token_max", NO_LAYER, self._microbatch),
            )
            self._microbatch_loss_slots[self._microbatch] = slots
        return slots

    def _resolve_mtp_histogram_offsets(self, device: torch.device) -> torch.Tensor:  # pragma: no cover
        if self._microbatch not in self._mtp_histogram_offsets:
            slots = [
                self._recorder.resolve("loss/mtp_token_hist", layer, self._microbatch) for layer in self._mtp_layers
            ]
            self._mtp_histogram_slots[self._microbatch] = slots
            # The kernel takes one offset per MTP head, a dummy one when there is none.
            offsets = [slot.offset for slot in slots] or [0]
            self._mtp_histogram_offsets[self._microbatch] = torch.tensor(offsets, dtype=torch.int64, device=device)
        return self._mtp_histogram_offsets[self._microbatch]

    def _resolve_segment_expert_table(
        self, segment: int, device: torch.device
    ) -> _SegmentExpertTable:  # pragma: no cover
        key = (segment, self._microbatch)
        table = self._segment_expert_tables.get(key)
        if table is not None:
            return table
        layers = [layer for layer, layer_segment in self._routed_layers.items() if layer_segment == segment]
        resolve = self._recorder.resolve
        every_step = ["expert_loss/sum", "expert_loss/sumsq", "expert_loss/count"]
        detailed_only = ["expert_loss/hist", "expert_loss/stream_count", "expert_loss/stream_sum"]
        slots = [[resolve(name, layer, self._microbatch) for name in every_step + detailed_only] for layer in layers]
        table = _SegmentExpertTable(
            layers=layers,
            positions=torch.tensor(
                [self._layer_positions[layer] for layer in layers], dtype=torch.int32, device=device
            ),
            offsets=torch.tensor([[slot.offset for slot in row] for row in slots], dtype=torch.int64, device=device),
            slot_indices=[slot.slot_index for row in slots for slot in row[: len(every_step)]],
            detailed_slot_indices=[slot.slot_index for row in slots for slot in row],
        )
        self._segment_expert_tables[key] = table
        return table

    def _observe_routing_cuda(
        self, target: RoutingTarget, topk_indices: torch.Tensor, scores: torch.Tensor
    ) -> None:  # pragma: no cover
        self._recorder.mark_recorded(
            [target.counts.slot_index, target.stream_counts.slot_index, target.entropy_histogram.slot_index]
        )
        _token_loss_kernel().launch_routing_kernel(
            topk_indices, scores, target.topk_buffer, target.position, self._recorder.buffer,
            target.entropy_histogram.offset, target.counts.offset, target.stream_counts.offset,
            target.stream_ids if target.stream_ids is not None else topk_indices,
            target.tokens_per_sample,
            self._streams, self._experts,
        )

    def _observe_loss_cuda(self, loss: torch.Tensor, labels_valid: torch.Tensor) -> None:  # pragma: no cover
        if self._stream_ids is None:
            stream_ids = torch.zeros(loss.shape[1], dtype=torch.int32, device=loss.device)
        else:
            stream_ids = self._stream_ids.to(torch.int32)
        labels_valid = labels_valid.to(torch.int32)
        if self._token_loss:
            self._launch_token_loss_kernel(loss, labels_valid, stream_ids)
        for segment in set(self._routed_layers.values()):
            table = self._resolve_segment_expert_table(segment, loss.device)
            for layer in table.layers:
                if layer not in self._observed_layers:
                    raise ValueError(f"routing for layer {layer} was not observed this microbatch")
            assert self._topk_buffer is not None
            self._recorder.mark_recorded(table.detailed_slot_indices if self._detailed else table.slot_indices)
            _token_loss_kernel().launch_expert_loss_kernel(
                loss[segment],
                labels_valid[segment],
                stream_ids,
                self._topk_buffer,
                table.positions,
                table.offsets,
                self._recorder.buffer,
                self._recorder.float_buffer(),
                self._top_k,
                self._streams,
                self._detailed,
            )

    def _launch_token_loss_kernel(
        self, loss: torch.Tensor, labels_valid: torch.Tensor, stream_ids: torch.Tensor
    ) -> None:  # pragma: no cover
        main = self._resolve_main_loss_slots()
        nonfinite_slot, token_max_slot = self._resolve_microbatch_loss_slots()
        mtp_histogram_offsets = self._resolve_mtp_histogram_offsets(loss.device)
        slot_indices = [
            main.token_histogram.slot_index,
            main.stream_sum.slot_index,
            main.stream_sumsq.slot_index,
            main.stream_count.slot_index,
            main.stream_samples.slot_index,
            main.position_sum.slot_index,
            main.position_count.slot_index,
            nonfinite_slot.slot_index,
            token_max_slot.slot_index,
        ]
        slot_indices += [slot.slot_index for slot in self._mtp_histogram_slots[self._microbatch]]
        self._recorder.mark_recorded(slot_indices)

        int32_buffer = self._recorder.buffer
        float32_buffer = self._recorder.float_buffer()
        _token_loss_kernel().launch_token_loss_kernel(
            loss,
            labels_valid,
            stream_ids,
            int32_buffer,
            float32_buffer,
            main.token_histogram.offset,
            main.stream_sum.offset,
            main.stream_sumsq.offset,
            main.stream_count.offset,
            main.stream_samples.offset,
            main.position_sum.offset,
            main.position_count.offset,
            token_max_slot.offset,
            nonfinite_slot.offset,
            mtp_histogram_offsets,
        )
        if self._detailed:
            self._record_sample_histogram_by_stream_cuda(loss, labels_valid)

    def _record_sample_histogram_by_stream_cuda(
        self, loss: torch.Tensor, labels_valid: torch.Tensor
    ) -> None:  # pragma: no cover
        finite = torch.isfinite(loss[0])
        valid = (labels_valid[0] != 0) & finite
        zeroed_loss = torch.where(valid, loss[0], torch.zeros_like(loss[0]))
        self._record_sample_histogram_by_stream(
            zeroed_loss, valid, self._stream_index(loss.shape[1], loss.shape[2], loss.device),
            (labels_valid[0] != 0) & ~finite,
        )
