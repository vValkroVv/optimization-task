from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from metric_sampler.capture.kernels import activation_ext
from metric_sampler.capture.layout import Layout
from metric_sampler.capture.recorder import CaptureCapacityError, RankRecorder
from metric_sampler.capture.ring import ShmRing
from metric_sampler.capture.stats import expert_balance
from metric_sampler.capture.stats.expert_balance import (
    ExpertBalanceStats,
    NativeExpertActivationCalls,
    NativeRouterBalanceCalls,
)
from metric_sampler.capture.stats.helpers.quantiles import exact_quantiles, row_entropy
from metric_sampler.capture.stats.helpers.reductions import float64_scalar
from metric_sampler.contract.catalog import metric_by_name
from metric_sampler.contract.metrics import QUANTILE_COUNTS_WIDTH, QUANTILE_LEVELS

# Words a router population records besides the score mean and the bias: the float64 balance sums (as int32 words)
# and the three counts.
_BALANCE_SUMS = 7
_BALANCE_WORDS = 2 * _BALANCE_SUMS
_BALANCE_COUNT_WORDS = 3
_BALANCE_RECORD_WORDS = _BALANCE_WORDS + _BALANCE_COUNT_WORDS
_ENTROPY_QUANTILE_WORDS = 2 * len(QUANTILE_LEVELS)
# The most rows the single-block entropy quantile kernel takes; a larger population is left to exact_quantiles.
_ENTROPY_QUANTILE_ROW_LIMIT = 1 << 17
# A mask marks a token valid when its uniform draw is below the share.
_VALID_TOKEN_SHARE = 0.9
_INTEGER_MASK_VALID_SHARE = 0.8
# The layers of the native router check whose calls differ from the others': the integer host z-loss, bfloat16 scores.
_HOST_Z_LOSS_LAYER = 2
_BFLOAT16_LAYER = 3


def _same_bits(got: torch.Tensor, expected: torch.Tensor) -> bool:
    # A NaN sum stays NaN; device arithmetic gives the canonical NaN, a copy keeps the NaN it was given.
    both_nan = torch.isnan(got) & torch.isnan(expected)
    return bool((both_nan | (got.view(torch.int64) == expected.view(torch.int64))).all())


def _entropy_quantiles_parity() -> list[str]:
    """entropy_quantiles_into against exact_quantiles(row_entropy(scores)): the quantile words and the counts bit for
    bit, at an even and an odd word offset, with a zero row and a non-finite one among the scores; a population too
    large for one block is left to exact_quantiles, nothing written."""
    extension = activation_ext.prebuilt()
    generator = torch.Generator(device='cuda').manual_seed(7)
    checked = []
    for rows, experts in ((8192, 192), (100, 64), (1, 8)):
        scores = torch.rand(rows, experts, device='cuda', generator=generator)
        if rows > 1:
            scores[1] = 0.
            scores[-1, 0] = float('nan')
        selected, population = exact_quantiles(row_entropy(scores))
        for offset in (0, 3):
            words = torch.full((64,), -1, dtype=torch.int32, device='cuda')
            assert extension.entropy_quantiles_into(scores, words, offset, offset + 30)
            assert torch.equal(
                words[offset:offset + _ENTROPY_QUANTILE_WORDS].cpu(), selected.view(torch.int32).cpu()
            ), (rows, offset)
            assert torch.equal(
                words[offset + 30:offset + 30 + QUANTILE_COUNTS_WIDTH].cpu(), population.cpu()
            ), (rows, offset)
        checked.append(f'entropy_quantiles_into_{rows}x{experts}')
    words = torch.full((64,), -1, dtype=torch.int32, device='cuda')
    assert not extension.entropy_quantiles_into(
        torch.rand(_ENTROPY_QUANTILE_ROW_LIMIT + 1, 2, device='cuda'), words, 0, 30
    )
    assert bool((words == -1).all())
    checked.append('entropy_quantiles_into_too_large')
    return checked


def _integer_mask_parity() -> list[str]:
    """router_accumulate with the mask as integers of every width the router may give, read in place, against the
    same mask as booleans: the population and its recorder words bit for bit, the nonbinary flag left at zero; a
    value of two sets the flag."""
    extension = activation_ext.prebuilt() or activation_ext.build()
    device = torch.device('cuda')
    generator = torch.Generator(device='cuda').manual_seed(20261014)
    tokens, experts, top_k = 4096, 64, 6
    scores = torch.rand(tokens, experts, device=device, generator=generator).softmax(-1)
    raw_topk = scores.topk(top_k, -1).values
    normalized_topk = raw_topk / raw_topk.sum(-1, keepdim=True)
    valid = torch.rand(tokens, device=device, generator=generator) < _INTEGER_MASK_VALID_SHARE
    z_loss = torch.rand((), device=device, generator=generator)

    def accumulate(mask, nonbinary):
        population = (torch.zeros(_BALANCE_SUMS, dtype=torch.float64, device=device),
                      torch.zeros((), dtype=torch.int64, device=device),
                      torch.zeros(experts, dtype=torch.float64, device=device))
        words = torch.full((2 * experts + _BALANCE_RECORD_WORDS,), -1, dtype=torch.int32, device=device)
        extension.router_accumulate(scores, scores, normalized_topk, raw_topk, raw_topk, mask, z_loss, None, 0.01,
                                    *population, True, words, 0, experts, experts + _BALANCE_WORDS,
                                    experts + _BALANCE_RECORD_WORDS, tokens, 1, None, nonbinary)
        return population, words

    expected, expected_words = accumulate(valid, None)
    checked = []
    for dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        flag = torch.zeros(1, dtype=torch.int32, device=device)
        population, words = accumulate(valid.to(dtype), flag)
        assert all(_same_bits(got, want) if got.is_floating_point() else torch.equal(got, want)
                   for got, want in zip(population, expected)), dtype
        assert torch.equal(words, expected_words) and flag.item() == 0, dtype
        checked.append(f'router_mask_{str(dtype).split(".")[-1]}')
    nonbinary = valid.to(torch.int32)
    nonbinary[7] = 2
    flag = torch.zeros(1, dtype=torch.int32, device=device)
    accumulate(nonbinary, flag)
    assert flag.item() == 1
    checked.append('router_mask_nonbinary_flag')
    return checked


def _router_accumulate_parity() -> list[str]:
    """router_accumulate against the population the reference router_sums leaves (a copy of the first call's
    sums, then +=), three calls a population: the entropy, column sums, counts and losses bit for bit, the
    top-k sums, reduced in another order, to 1e-12. A second population started by first=True instead of
    zeros, which also writes the recorder slots, must equal the first bit for bit, and its slots the words
    the flush recorded from such a population: the score mean, the balance values, the counts and the bias."""
    extension = activation_ext.prebuilt() or activation_ext.build()
    device = torch.device('cuda')
    generator = torch.Generator(device='cuda').manual_seed(20261013)
    checked = []
    for tokens, experts, top_k, padded, masked, topk_dtype, loss in (
            (4096, 192, 8, False, False, torch.float32, 'float32'),
            (8192, 192, 8, True, True, torch.float32, 'float32'),
            (4096, 192, 8, False, True, torch.bfloat16, 'host'), (2, 4, 2, False, False, torch.float32, 'bfloat16'),
            (1000, 64, 6, True, True, torch.float16, 'float64'), (300, 16, 4, False, True, torch.float32, 'int64')):
        reference = None
        values = torch.zeros(_BALANCE_SUMS, dtype=torch.float64, device=device)
        valid_tokens = torch.zeros((), dtype=torch.int64, device=device)
        score_sums = torch.zeros(experts, dtype=torch.float64, device=device)
        # Uninitialized: first=True must not read them.
        in_place = (torch.full((_BALANCE_SUMS,), float('nan'), dtype=torch.float64, device=device),
                    torch.full((), -7, dtype=torch.int64, device=device),
                    torch.full((experts,), float('nan'), dtype=torch.float64, device=device))
        words = torch.full((2 * experts + _BALANCE_RECORD_WORDS,), -1, dtype=torch.int32, device=device)
        expert_bias = None if masked else torch.rand(experts, device=device, generator=generator) - 0.5
        for call in range(3):
            logits = torch.randn(tokens, experts + (5 if padded else 0), device=device, generator=generator)
            scores = logits.softmax(-1)[:, :experts]
            biased_scores = scores + torch.rand(experts, device=device, generator=generator) * 0.01
            raw_topk = scores.topk(top_k, -1).values.to(topk_dtype)
            biased_topk = biased_scores.topk(top_k, -1).values.to(topk_dtype)
            normalized_topk = (raw_topk.float() / raw_topk.float().sum(-1, keepdim=True)).to(topk_dtype)
            mask = torch.rand(tokens, device=device, generator=generator) < _VALID_TOKEN_SHARE if masked else None
            z_loss = torch.rand((), device=device, generator=generator)
            # A disabled z-loss is the integer host zero.
            z_loss = (torch.tensor(0) if loss == 'int64' else z_loss.cpu() if loss == 'host'
                      else z_loss.to(getattr(torch, loss)))
            seq_aux_loss = None if call == 1 else torch.rand((), device=device, generator=generator)
            alpha = 0.001 * (call + 1)
            sums = expert_balance.router_sums(
                scores, biased_scores, normalized_topk, raw_topk, biased_topk,
                torch.ones(tokens, dtype=torch.bool, device=device) if mask is None else mask,
                float64_scalar(z_loss, device),
                torch.full((), float('nan'), device=device, dtype=torch.float64) if seq_aux_loss is None
                else float64_scalar(seq_aux_loss, device),
                torch.full((), alpha, device=device, dtype=torch.float64))
            if reference is None:
                reference = [part.clone() for part in sums]
            else:
                for total, part in zip(reference, sums):
                    total += part
            extension.router_accumulate(scores, biased_scores, normalized_topk, raw_topk, biased_topk, mask, z_loss,
                                        seq_aux_loss, alpha, values, valid_tokens, score_sums)
            extension.router_accumulate(scores, biased_scores, normalized_topk, raw_topk, biased_topk, mask, z_loss,
                                        seq_aux_loss, alpha, *in_place, call == 0, words, 0, experts,
                                        experts + _BALANCE_WORDS, experts + _BALANCE_RECORD_WORDS,
                                        tokens * (call + 1), call + 1, expert_bias)
        torch.cuda.synchronize()
        assert _same_bits(in_place[0], values) and _same_bits(in_place[2], score_sums), tokens
        assert torch.equal(in_place[1], valid_tokens), tokens
        # The flush's expressions of the population, word for word.
        flushed_mean = (score_sums / (tokens * 3)).float()
        flushed_counts = torch.stack((torch.full((), tokens * 3, device=device, dtype=torch.int64), valid_tokens,
                                      torch.full((), 3, device=device, dtype=torch.int64))).to(torch.int32)
        flushed_bias = torch.zeros(experts, device=device) if expert_bias is None else expert_bias.float().clone()
        assert torch.equal(words[:experts], flushed_mean.view(torch.int32)), tokens
        assert torch.equal(words[experts:experts + _BALANCE_WORDS], values.contiguous().view(torch.int32)), tokens
        assert torch.equal(
            words[experts + _BALANCE_WORDS:experts + _BALANCE_RECORD_WORDS], flushed_counts
        ), tokens
        assert torch.equal(words[experts + _BALANCE_RECORD_WORDS:], flushed_bias.view(torch.int32)), tokens
        expected, expected_valid, expected_columns = (total.cpu() for total in reference)
        got = values.cpu()
        assert _same_bits(got[3:], expected[3:]), (tokens, got, expected)
        assert _same_bits(score_sums.cpu(), expected_columns), tokens
        assert valid_tokens.item() == expected_valid.item(), (tokens, valid_tokens, expected_valid)
        torch.testing.assert_close(got[:3], expected[:3], rtol=1e-12, atol=0)
        checked.append(f'router_accumulate_{tokens}x{experts}_k{top_k}_{"masked" if masked else "all"}_{loss}')
    return checked


class _Sampler:
    """The sampler attributes RouterBalanceCalls reads."""

    def __init__(self) -> None:
        self._capture_forward_active = True
        self._grad_forward = True
        self.microbatch = 0


@dataclass(frozen=True)
class _Sides:
    """The two recorders of a native parity check: the reference path's and the extension's, with the extension's
    observers by layer."""

    recorders: list[RankRecorder]
    reference: ExpertBalanceStats
    stats: ExpertBalanceStats
    sampler: _Sampler
    native: NativeRouterBalanceCalls | NativeExpertActivationCalls
    observers: list[Callable[..., None]]


def _native_router_parity() -> list[str]:
    """RouterBalanceCalls against observe_router, the reference path: two steps of two microbatches
    over four routed layers -- a Boolean mask, a bias and padded scores; an integer mask, no bias and no sequence
    auxiliary loss; no mask and the integer host z-loss; bfloat16 scores, which go to observe_router -- with repeated
    calls: the recorder words and recorded flags bit for bit after each step, and the nonbinary flag of an integer
    mask raised by both. A call outside a capture forward takes nothing; a call observe_router refuses raises the
    same error, and a call for a population of the other side raises."""
    device = torch.device('cuda')
    experts, top_k, tokens = 64, 6, 512
    names = [('router/scores_mean', experts), ('monitor/router_balance_local', _BALANCE_WORDS),
             ('monitor/router_balance_counts_local', _BALANCE_COUNT_WORDS), ('monitor/router_bias_local', experts)]
    layout = Layout([(metric_by_name(name), layer, microbatch, width)
                     for layer in range(4) for microbatch in (0, 1) for name, width in names])
    rings = [ShmRing.create('expert-native-' + uuid.uuid4().hex, layout, 1) for _ in range(2)]
    recorders = [RankRecorder(layout, ring, device) for ring in rings]
    try:
        reference, stats = (ExpertBalanceStats(recorder, experts, experts, 0) for recorder in recorders)
        sampler = _Sampler()
        native = stats.native_calls(sampler)
        assert native is not None
        sides = _Sides(recorders, reference, stats, sampler, native, [native.router(layer) for layer in range(4)])
        generator = torch.Generator(device='cuda').manual_seed(20261016)
        bias = torch.rand(experts, device=device, generator=generator) - 0.5

        def operands(layer, nonbinary=False):
            logits = torch.randn(tokens, experts + (5 if layer == 0 else 0), device=device, generator=generator)
            scores = logits.softmax(-1)[:, :experts]
            if layer == _BFLOAT16_LAYER:
                scores = scores.bfloat16()
            layer_bias = bias if layer in (0, 2) else None
            biased_scores = scores if layer_bias is None else scores + layer_bias.to(scores.dtype)
            raw_topk = scores.topk(top_k, -1).values
            mask = None
            if layer in (0, 3):
                mask = torch.rand(tokens, device=device, generator=generator) < _VALID_TOKEN_SHARE
            elif layer == 1:
                mask = (torch.rand(tokens, device=device, generator=generator) < _VALID_TOKEN_SHARE).to(torch.int32)
                if nonbinary:
                    mask[5] = 2
            # A disabled z-loss is the integer host zero.
            z_loss = (torch.tensor(0) if layer == _HOST_Z_LOSS_LAYER
                      else torch.rand((), device=device, generator=generator))
            seq_aux_loss = None if layer == 1 else torch.rand((), device=device, generator=generator)
            return (scores, biased_scores, raw_topk / raw_topk.sum(-1, keepdim=True), raw_topk,
                    biased_scores.topk(top_k, -1).values, mask, layer_bias, z_loss, seq_aux_loss,
                    0.001 * (layer + 1))

        checked = _router_steps(sides, operands)
        checked += _router_nonbinary_mask(sides)
        checked += _router_outside_capture(sides, operands)
        checked += _router_refusals(sides, operands, bias)
        checked += _router_one_side_a_population(sides, operands, bias)
        reference.flush()
        stats.flush()
        return checked
    finally:
        torch.cuda.synchronize()
        for recorder, ring in zip(recorders, rings):
            recorder.close()
            ring.unlink()


def _router_steps(sides: _Sides, operands: Callable[..., tuple]) -> list[str]:
    checked = []
    for step in range(2):
        for recorder in sides.recorders:
            recorder.begin_step(False)
        for microbatch in (0, 1):
            sides.sampler.microbatch = microbatch
            for layer in range(4):
                for _ in range(1 if layer == _HOST_Z_LOSS_LAYER else 2 + layer % 2):
                    call = operands(layer, nonbinary=step == 1 and microbatch == 1)
                    sides.reference.observe_router(layer, microbatch, *call)
                    sides.observers[layer](*call)
            populated = [sides.native.has_population(layer, microbatch) for layer in range(4)]
            assert populated == [True, True, True, False]
            sides.reference.flush()
            sides.stats.flush()
            assert not any(sides.native.has_population(layer, microbatch) for layer in range(4))
        torch.cuda.synchronize()
        assert torch.equal(sides.recorders[0].buffer, sides.recorders[1].buffer), step
        assert torch.equal(sides.recorders[0].recorded_flags(), sides.recorders[1].recorded_flags()), step
        assert bool(sides.recorders[1].recorded_flags().all()), step
        sides.reference.require_binary_masks()
        sides.stats.require_binary_masks()
        checked.append(f'native_router_step{step}')
    return checked


def _router_nonbinary_mask(sides: _Sides) -> list[str]:
    for side in (sides.reference, sides.stats):
        try:
            side.finish()
        except ValueError as error:
            assert 'zero or one' in str(error)
        else:
            raise AssertionError('nonbinary integer mask was accepted')
    return ['native_router_nonbinary_mask']


def _router_outside_capture(sides: _Sides, operands: Callable[..., tuple]) -> list[str]:
    sides.sampler._capture_forward_active = False
    before = sides.recorders[1].buffer.clone()
    sides.observers[0](*operands(0))
    torch.cuda.synchronize()
    assert not sides.native.has_population(0, 0) and torch.equal(sides.recorders[1].buffer, before)
    sides.sampler._capture_forward_active = True
    return ['native_router_outside_capture']


def _router_refusal(observe: Callable[..., None], *call: object) -> str:
    try:
        observe(*call)
    except ValueError as error:
        return str(error)
    raise AssertionError('accepted')


def _router_refusals(sides: _Sides, operands: Callable[..., tuple], bias: torch.Tensor) -> list[str]:
    checked = []
    sides.sampler.microbatch = 0
    good = operands(0)
    refused = {
        'experts': (good[0][:, 1:], good[1][:, 1:], *good[2:]),
        'biased_shape': (good[0], good[1][1:], *good[2:]),
        'mask_device': (*good[:5], good[5].cpu(), *good[6:]),
        'mask_tokens': (*good[:5], good[5][1:], *good[6:]),
        'topk_device': (*good[:2], good[2].cpu(), *good[3:]),
        'bias_experts': (*good[:6], bias[1:], *good[7:]),
    }
    for name, call in refused.items():
        message = _router_refusal(sides.reference.observe_router, 0, 0, *call)
        assert _router_refusal(sides.observers[0], *call) == message, name
        assert not sides.native.has_population(0, 0), name
        checked.append(f'native_router_refuses_{name}')
    sides.reference.observe_router(0, 0, *good)
    sides.observers[0](*good)
    changed = (*good[:6], bias * 2, *good[7:])
    message = _router_refusal(sides.reference.observe_router, 0, 0, *changed)
    assert 'bias changed' in message and _router_refusal(sides.observers[0], *changed) == message
    checked.append('native_router_refuses_bias_change')
    return checked


def _router_one_side_a_population(sides: _Sides, operands: Callable[..., tuple], bias: torch.Tensor) -> list[str]:
    bfloat16 = operands(_BFLOAT16_LAYER)
    float32 = (bfloat16[0].float(), bfloat16[1].float(), *bfloat16[2:])
    sides.observers[_BFLOAT16_LAYER](*bfloat16)
    assert 'did not take' in _router_refusal(sides.observers[_BFLOAT16_LAYER], *float32)
    assert 'extension took' in _router_refusal(sides.observers[0], *bfloat16[:6], bias, *bfloat16[7:])
    return ['native_router_one_side_a_population']


def _native_expert_parity() -> list[str]:
    """ExpertActivationCalls against observe_experts, the reference path: two steps of two
    microbatches over three MoE layers of a rank owning experts 4-7 of 8 -- bfloat16 intermediates called twice, a
    float32 one with an empty expert, and float64 ones, which go to observe_experts -- the recorder words, the host
    words of the counts and the recorded flags bit for bit after each step. A call outside a capture forward takes
    nothing; a call observe_experts refuses raises the same error, and a call for a population of the other side
    raises."""
    device = torch.device('cuda')
    experts, local_experts, offset = 8, 4, 4
    names = [('monitor/expert_activation_sumsq_local', 2 * experts), ('monitor/expert_activation_count_local', experts)]
    layout = Layout([(metric_by_name(name), layer, microbatch, width)
                     for layer in range(3) for microbatch in (0, 1) for name, width in names])
    rings = [ShmRing.create('expert-act-' + uuid.uuid4().hex, layout, 1) for _ in range(2)]
    recorders = [RankRecorder(layout, ring, device) for ring in rings]
    try:
        reference, stats = (ExpertBalanceStats(recorder, experts, local_experts, offset) for recorder in recorders)
        sampler = _Sampler()
        native = stats.native_expert_calls(sampler)
        assert native is not None
        sides = _Sides(recorders, reference, stats, sampler, native, [native.experts(layer) for layer in range(3)])
        generator = torch.Generator(device='cuda').manual_seed(20261016)
        dtypes = (torch.bfloat16, torch.float32, torch.float64)

        def call(layer):
            counts = [int(count) for count in torch.randint(0, 40, (local_experts,), generator=generator,
                                                             device=device).tolist()]
            if layer == 1:
                counts[2] = 0
            intermediate = torch.randn(sum(counts), 24, device=device, generator=generator).to(dtypes[layer])
            return intermediate, counts

        checked = _expert_steps(sides, call)
        checked += _expert_outside_capture(sides, call)
        sampler.microbatch = 0
        intermediate, counts = call(0)
        checked += _expert_refusals(sides, intermediate, counts)
        checked += _expert_one_side_a_population(sides, call, intermediate, counts)
        reference.flush()
        stats.flush()
        return checked
    finally:
        torch.cuda.synchronize()
        for recorder, ring in zip(recorders, rings):
            recorder.close()
            ring.unlink()


def _expert_steps(sides: _Sides, call: Callable[..., tuple]) -> list[str]:
    checked = []
    for step in range(2):
        for recorder in sides.recorders:
            recorder.begin_step(False)
        for microbatch in (0, 1):
            sides.sampler.microbatch = microbatch
            for layer in range(3):
                for _ in range(1 if layer == 1 else 2):
                    intermediate, counts = call(layer)
                    sides.reference.observe_experts(layer, microbatch, intermediate, list(counts))
                    sides.observers[layer](intermediate, list(counts))
            populated = [sides.native.has_population(layer, microbatch) for layer in range(3)]
            assert populated == [True, True, False]
            sides.reference.flush()
            sides.stats.flush()
            assert not any(sides.native.has_population(layer, microbatch) for layer in range(3))
        torch.cuda.synchronize()
        assert torch.equal(sides.recorders[0].buffer, sides.recorders[1].buffer), step
        assert (sides.recorders[0]._host_staging == sides.recorders[1]._host_staging).all(), step
        assert torch.equal(sides.recorders[0].recorded_flags(), sides.recorders[1].recorded_flags()), step
        checked.append(f'native_experts_step{step}')
    return checked


def _expert_outside_capture(sides: _Sides, call: Callable[..., tuple]) -> list[str]:
    sides.sampler._capture_forward_active = False
    before = sides.recorders[1].buffer.clone()
    sides.observers[0](*call(0))
    torch.cuda.synchronize()
    assert not sides.native.has_population(0, 0) and torch.equal(sides.recorders[1].buffer, before)
    sides.sampler._capture_forward_active = True
    return ['native_experts_outside_capture']


def _expert_refusal(observe: Callable[..., None], *arguments: object) -> str:
    try:
        observe(*arguments)
    except (ValueError, CaptureCapacityError) as error:
        return f'{type(error).__name__}: {error}'
    raise AssertionError('accepted')


def _expert_refusals(sides: _Sides, intermediate: torch.Tensor, counts: list[int]) -> list[str]:
    checked = []
    refused = {
        'counts_length': (intermediate, counts[:-1]),
        'counts_bool': (intermediate, [True, *counts[1:]]),
        'counts_negative': (intermediate, [-1, *counts[1:]]),
        'counts_capacity': (intermediate, [1 << 40, *counts[1:]]),
        'counts_beyond_int64': (intermediate, [1 << 70, *counts[1:]]),
        'rows': (intermediate[1:], counts),
        'not_a_tensor': (intermediate.tolist(), counts),
    }
    for name, arguments in refused.items():
        message = _expert_refusal(sides.reference.observe_experts, 0, 0, *arguments)
        assert _expert_refusal(sides.observers[0], *arguments) == message, (name, message)
        assert not sides.native.has_population(0, 0), name
        checked.append(f'native_experts_refuses_{name}')
    return checked


def _expert_one_side_a_population(
    sides: _Sides, call: Callable[..., tuple], intermediate: torch.Tensor, counts: list[int]
) -> list[str]:
    float64 = call(2)
    sides.observers[2](*float64)
    assert 'did not take' in _expert_refusal(sides.observers[2], intermediate, counts)
    sides.observers[0](intermediate, counts)
    assert 'extension took' in _expert_refusal(sides.observers[0], float64[0], float64[1])
    return ['native_experts_one_side_a_population']


def _record_reference_cases(recorder: RankRecorder) -> tuple[torch.Tensor, torch.Tensor]:
    stats = ExpertBalanceStats(recorder, 4, 2, 2)
    scores = torch.tensor([[.125, .25, .5, .125], [.5, .25, .125, .125]], device='cuda')
    bias = torch.tensor([.125, 0., .125, 0.], device='cuda')
    raw = scores[:, :2]
    normalized = raw / raw.sum(-1, keepdim=True)
    valid = torch.tensor([[1, 0]], dtype=torch.int32, device='cuda')
    # Integer masks are checked on the device and raise at the next require_binary_masks or finish.
    mask_check = ExpertBalanceStats(recorder, 4, 2, 2)
    mask_check.observe_router(0, 0, scores, scores + bias, normalized, raw, (scores + bias)[:, :2],
                              torch.tensor([[1, 2]], dtype=torch.int32, device='cuda'), bias,
                              torch.tensor(.25, device='cuda'), torch.tensor(.125, device='cuda'), .01)
    mask_check.require_binary_masks()
    try:
        mask_check.finish()
    except ValueError as error:
        assert 'zero or one' in str(error)
    else:
        raise AssertionError('nonbinary integer mask was accepted')
    for _ in range(2):
        stats.observe_router(0, 0, scores, scores + bias, normalized, raw, (scores + bias)[:, :2],
                             valid, bias, torch.tensor(.25, device='cuda'), torch.tensor(.125, device='cuda'), .01)
    intermediate = torch.ones(260, 2, dtype=torch.bfloat16, device='cuda')
    intermediate[-3:] *= 2
    stats.observe_experts(0, 0, intermediate, [257, 3])
    stats.observe_experts(0, 0, torch.tensor([[3., 4.], [1., 2.]], device='cuda'), [1, 1])
    stats.observe_experts(1, 0, torch.empty(0, 2, device='cuda'), [0, 0])
    stats.observe_experts(2, 0, torch.tensor([[float('nan')], [float('inf')]], device='cuda'), [1, 1])
    nonfinite = torch.tensor([[1., 1., float('nan'), float('inf')]], device='cuda')
    stats.observe_router(1, 0, nonfinite, nonfinite, nonfinite[:, :2], nonfinite[:, :2], nonfinite[:, :2],
                         None, None, torch.tensor(0., device='cuda'), None, 0.)
    stats.flush()
    return scores, bias


def _check_recorded_words(layout: Layout, recorder: RankRecorder, scores: torch.Tensor, bias: torch.Tensor) -> None:
    def read(name, layer=0):
        slot = layout.slot(name, layer, 0)
        if name == 'monitor/expert_activation_count_local':
            # Staged on the host and overlaid onto the slot by step_end, not written to the device buffer.
            return torch.from_numpy(recorder._host_staging[slot.offset:slot.offset + slot.width].copy())
        value = recorder.buffer[slot.offset:slot.offset + slot.width].cpu()
        if name in ('router/scores_mean', 'monitor/router_bias_local'):
            return value.view(torch.float32)
        if name in ('monitor/router_balance_local', 'monitor/expert_activation_sumsq_local'):
            return value.view(torch.float64)
        return value

    assert read('monitor/router_balance_counts_local').tolist() == [4, 2, 2]
    assert read('monitor/expert_activation_count_local').tolist() == [0, 0, 258, 4]
    assert read('monitor/expert_activation_sumsq_local').tolist() == [0., 0., 539., 29.]
    assert torch.equal(read('router/scores_mean'), scores.cpu().mean(0))
    assert torch.equal(read('monitor/router_bias_local'), bias.cpu())
    expected = torch.tensor([2., .75, 1., 0., .5, .25, .02], dtype=torch.float64)
    biased = (scores + bias).cpu().double()
    expected[3] = -2 * (biased * (biased + 1e-12).log()).sum()
    torch.testing.assert_close(read('monitor/router_balance_local'), expected, rtol=1e-7, atol=1e-13)
    assert read('monitor/expert_activation_count_local', 1).tolist() == [0, 0, 0, 0]
    assert read('monitor/expert_activation_sumsq_local', 1).tolist() == [0., 0., 0., 0.]
    nonfinite_sums = read('monitor/expert_activation_sumsq_local', 2)
    assert torch.isnan(nonfinite_sums[2]) and torch.isposinf(nonfinite_sums[3])
    assert read('monitor/expert_activation_count_local', 2).tolist() == [0, 0, 1, 1]
    assert torch.isnan(read('monitor/router_balance_local', 1)[3:6:2]).all()
    assert read('monitor/router_balance_counts_local', 1).tolist() == [1, 1, 1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict CUDA recorder checks for rank-local expert balancing operands."
    )
    parser.add_argument('--output', required=True, type=Path)
    arguments = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; CPU cannot pass the GPU check')
    names = [('router/scores_mean', 4), ('monitor/router_balance_local', _BALANCE_WORDS),
             ('monitor/router_balance_counts_local', _BALANCE_COUNT_WORDS), ('monitor/router_bias_local', 4),
             ('monitor/expert_activation_sumsq_local', 8), ('monitor/expert_activation_count_local', 4)]
    layout = Layout([(metric_by_name(name), layer, 0, width) for layer in (0, 1, 2) for name, width in names])
    ring = ShmRing.create('expert-gpu-' + uuid.uuid4().hex, layout, 1)
    recorder = RankRecorder(layout, ring, torch.device('cuda'))
    started = time.perf_counter()
    try:
        recorder.begin_step(True)
        scores, bias = _record_reference_cases(recorder)
        torch.cuda.synchronize()
        _check_recorded_words(layout, recorder, scores, bias)
        parity = (_router_accumulate_parity() + _integer_mask_parity() + _entropy_quantiles_parity()
                  + _native_router_parity() + _native_expert_parity())
        result = {'passed': True, 'device': torch.cuda.get_device_name(),
                  'checks': ['all_batch_rows', 'valid_denominator', 'repeated_union', 'expert_count257',
                             'ownership_structural_zeros', 'float64_sumsq_words', 'local_bias', 'complete_score_mean',
                             'empty_expert_segments', 'nonfinite_population_preserved', 'disabled_auxiliary_nan',
                             *parity],
                  'mask_contract': 'Int32 zero/one masks accepted; '
                                   'nonbinary raised by require_binary_masks or finish',
                  'source_sha256': hashlib.sha256(Path(expert_balance.__file__).read_bytes()).hexdigest(),
                  'elapsed_seconds': time.perf_counter() - started,
                  'training_overhead_measured': False}
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({'passed': True, 'output': str(arguments.output)}))
    finally:
        torch.cuda.synchronize()
        recorder.close()
        ring.unlink()


if __name__ == '__main__':
    main()
