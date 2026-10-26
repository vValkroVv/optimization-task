from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.utils.checkpoint as checkpoint

from metric_sampler.capture.dump import ActivationDumpConfigError, DumpCapture, DumpConfig, DumpSettings
from metric_sampler.capture.dump_buffer import DTYPE_NOT_CAPTURED, DumpBuffer, DumpEntry, decode_dump_entry
from metric_sampler.capture.ring import STATE_FREE, STATE_READY, STATE_WRITING

pytestmark = pytest.mark.unit

_TOKENS = 4
_HIDDEN = 8
_ENTRIES = (DumpEntry("residual", 0), DumpEntry("residual", 1), DumpEntry("final_norm", -1))


@pytest.fixture
def make_capture(dump_buffers):
    """Builds a (DumpCapture, DumpBuffer) pair and closes the capture (which closes the buffer's
    torch/numpy views) before the underlying `dump_buffers` fixture closes and unlinks it."""

    created = []

    def factory(
        entries: tuple[DumpEntry, ...] = _ENTRIES,
        tokens: int = _TOKENS,
        hidden: int = _HIDDEN,
        interval: int = 1,
        max_bytes: int = 10**9,
        global_rank: int = 0,
    ) -> tuple[DumpCapture, DumpBuffer]:
        buffer = dump_buffers(entries, tokens, hidden)
        config = DumpConfig(interval=interval, entries=entries, tokens=tokens, max_bytes=max_bytes)
        capture = DumpCapture(config, buffer, torch.device("cpu"), global_rank)
        created.append(capture)
        return capture, buffer

    yield factory

    for capture in created:
        capture.close()


def test_begin_step_is_never_active_when_interval_is_zero(make_capture) -> None:
    capture, buffer = make_capture(interval=0)

    capture.begin_step(0)

    assert capture._active is False
    assert capture.dumps_skipped == 0
    assert buffer.state == STATE_FREE


def test_begin_step_skips_a_step_matching_neither_interval_nor_request(make_capture) -> None:
    capture, buffer = make_capture(interval=2)

    capture.begin_step(3)

    assert capture._active is False
    assert capture.dumps_skipped == 0
    assert buffer.state == STATE_FREE


def test_begin_step_captures_on_the_interval_for_global_rank_zero(make_capture) -> None:
    capture, buffer = make_capture(interval=2, global_rank=0)

    capture.begin_step(4)

    assert capture._active is True
    assert buffer.state == STATE_WRITING


def test_begin_step_does_not_trigger_on_interval_for_a_nonzero_global_rank(make_capture) -> None:
    capture, buffer = make_capture(interval=2, global_rank=1)

    capture.begin_step(4)

    assert capture._active is False
    assert buffer.state == STATE_FREE



def test_begin_step_skips_and_counts_when_the_buffer_is_busy(make_capture) -> None:
    capture, buffer = make_capture(interval=1)
    buffer.claim_for_writing()

    capture.begin_step(1)

    assert capture._active is False
    assert capture.dumps_skipped == 1



def test_begin_step_skips_and_sets_budget_exhausted_once_max_bytes_is_reached(make_capture) -> None:
    bytes_per_dump = len(_ENTRIES) * _TOKENS * _HIDDEN * 4
    capture, buffer = make_capture(interval=1, max_bytes=bytes_per_dump)

    capture.begin_step(1)
    capture.step_end()
    buffer.release()

    capture.begin_step(2)

    assert capture._active is False
    assert capture.budget_exhausted is True
    assert capture.dumps_skipped == 1


def test_dumps_skipped_is_cumulative_across_steps(make_capture) -> None:
    capture, buffer = make_capture(interval=1)
    buffer.claim_for_writing()

    capture.begin_step(1)
    capture.begin_step(2)

    assert capture.dumps_skipped == 2


def test_observe_input_ids_is_a_noop_when_not_capturing(make_capture) -> None:
    capture, buffer = make_capture(interval=0)

    capture.begin_step(1)
    capture.observe_input_ids(torch.tensor([[1, 2, 3, 4]]))

    assert list(buffer.token_ids()) == [0] * _TOKENS


def test_observe_input_ids_is_a_noop_outside_microbatch_zero(make_capture) -> None:
    capture, buffer = make_capture(interval=1)
    capture.begin_step(1)
    capture.begin_microbatch(1, None)

    capture.observe_input_ids(torch.tensor([[10, 11, 12, 13]]))

    assert list(buffer.token_ids()) == [0] * _TOKENS


def test_observe_input_ids_writes_the_first_tokens(make_capture) -> None:
    capture, buffer = make_capture(interval=1)
    capture.begin_step(1)
    capture.begin_microbatch(0, None)

    capture.observe_input_ids(torch.tensor([[10, 11, 12, 13, 14, 15]]))

    assert list(buffer.token_ids()) == [10, 11, 12, 13]


def test_observe_input_ids_raises_when_shorter_than_tokens(make_capture) -> None:
    capture, _ = make_capture(interval=1)
    capture.begin_step(1)
    capture.begin_microbatch(0, None)

    with pytest.raises(ValueError):
        capture.observe_input_ids(torch.tensor([[1, 2]]))


def _capturing(capture, step: int = 1) -> None:
    capture.begin_step(step)
    capture.begin_microbatch(0, None)


def test_observe_is_a_noop_when_not_capturing(make_capture) -> None:
    capture, buffer = make_capture(interval=0)
    capture.begin_step(1)

    capture.observe("residual", 0, torch.zeros(1, _TOKENS, _HIDDEN, dtype=torch.bfloat16))

    assert list(buffer.entry_dtype_codes()) == [DTYPE_NOT_CAPTURED] * len(_ENTRIES)


def test_observe_is_a_noop_for_a_point_layer_outside_the_configured_entries(make_capture) -> None:
    capture, buffer = make_capture(interval=1)
    _capturing(capture)

    capture.observe("residual", 99, torch.zeros(1, _TOKENS, _HIDDEN, dtype=torch.bfloat16))

    assert list(buffer.entry_dtype_codes()) == [DTYPE_NOT_CAPTURED] * len(_ENTRIES)


def test_observe_captures_bf16_fp16_and_fp32_entries(make_capture) -> None:
    capture, buffer = make_capture(interval=1)
    _capturing(capture)

    residual0 = torch.arange(_TOKENS * _HIDDEN, dtype=torch.bfloat16).reshape(1, _TOKENS, _HIDDEN)
    residual1 = torch.arange(_TOKENS * _HIDDEN, dtype=torch.float16).reshape(1, _TOKENS, _HIDDEN) * 2
    final_norm = torch.arange(_TOKENS * _HIDDEN, dtype=torch.float32).reshape(1, _TOKENS, _HIDDEN) * 3

    capture.observe("residual", 0, residual0)
    capture.observe("residual", 1, residual1)
    capture.observe("final_norm", -1, final_norm)

    codes = list(buffer.entry_dtype_codes())
    assert codes == [0, 1, 2]
    for index, expected in enumerate((residual0, residual1, final_norm)):
        itemsize = 2 if index < 2 else 4
        raw = buffer.entry_bytes(index)[: _TOKENS * _HIDDEN * itemsize].copy()
        decoded = decode_dump_entry(raw, codes[index], _TOKENS, _HIDDEN)
        np.testing.assert_allclose(decoded, expected.float().reshape(_TOKENS, _HIDDEN).numpy())


def test_observe_takes_only_the_first_tokens_rows(make_capture) -> None:
    capture, buffer = make_capture(interval=1)
    _capturing(capture)
    tensor = torch.arange(8 * _HIDDEN, dtype=torch.bfloat16).reshape(1, 8, _HIDDEN)

    capture.observe("residual", 0, tensor)

    raw = buffer.entry_bytes(0)[: _TOKENS * _HIDDEN * 2].copy()
    decoded = decode_dump_entry(raw, 0, _TOKENS, _HIDDEN)
    np.testing.assert_allclose(decoded, tensor[0, :_TOKENS].float().numpy())


def test_observe_raises_when_fewer_rows_than_tokens(make_capture) -> None:
    capture, _ = make_capture(interval=1)
    _capturing(capture)

    with pytest.raises(ValueError):
        capture.observe("residual", 0, torch.zeros(1, _TOKENS - 1, _HIDDEN, dtype=torch.bfloat16))


def test_observe_raises_for_an_unsupported_dtype(make_capture) -> None:
    capture, _ = make_capture(interval=1)
    _capturing(capture)

    with pytest.raises(ValueError):
        capture.observe("residual", 0, torch.zeros(1, _TOKENS, _HIDDEN, dtype=torch.float64))


def test_observe_under_no_grad_does_nothing(make_capture) -> None:
    capture, buffer = make_capture(interval=1)
    _capturing(capture)

    with torch.no_grad():
        capture.observe("residual", 0, torch.ones(1, _TOKENS, _HIDDEN, dtype=torch.bfloat16))

    assert list(buffer.entry_dtype_codes()) == [DTYPE_NOT_CAPTURED] * len(_ENTRIES)


def test_observe_is_not_double_counted_under_checkpointing_recompute(make_capture) -> None:
    capture, buffer = make_capture(interval=1)
    _capturing(capture)
    calls: list[int] = []

    def block(inputs: torch.Tensor) -> torch.Tensor:
        doubled = inputs * 2.0
        calls.append(1)
        capture.observe("residual", 0, doubled.to(torch.bfloat16).unsqueeze(0))
        return (doubled * doubled).sum()

    inputs = torch.arange(_TOKENS * _HIDDEN, dtype=torch.float32).reshape(_TOKENS, _HIDDEN).requires_grad_(True)
    checkpoint.checkpoint(block, inputs, use_reentrant=False).backward()

    assert len(calls) == 2  # forward, plus a recompute in backward
    raw = buffer.entry_bytes(0)[: _TOKENS * _HIDDEN * 2].copy()
    decoded = decode_dump_entry(raw, 0, _TOKENS, _HIDDEN)
    np.testing.assert_allclose(decoded, (inputs.detach() * 2.0).numpy(), atol=2e-2)


def test_observe_loss_is_a_noop_when_not_capturing(make_capture) -> None:
    capture, buffer = make_capture(interval=0)
    capture.begin_step(1)

    capture.observe_loss(torch.zeros(1, 1, _TOKENS), torch.ones(1, 1, _TOKENS, dtype=torch.bool))

    assert list(buffer.token_loss()) == [0.0] * _TOKENS


def test_observe_loss_writes_nan_for_invalid_or_nonfinite_tokens(make_capture) -> None:
    capture, buffer = make_capture(interval=1)
    _capturing(capture)
    loss = torch.tensor([[[0.1, 0.2, 0.3, float("nan")]]])
    valid = torch.tensor([[[True, False, True, True]]])

    capture.observe_loss(loss, valid)

    values = list(buffer.token_loss())
    assert values[0] == pytest.approx(0.1)
    assert np.isnan(values[1])
    assert values[2] == pytest.approx(0.3)
    assert np.isnan(values[3])


def test_observe_loss_maps_stream_id_by_sample(make_capture) -> None:
    entries = (DumpEntry("residual", 0),)
    capture, buffer = make_capture(entries=entries, tokens=4, interval=1)
    capture.begin_step(1)
    capture.begin_microbatch(0, torch.tensor([5, 9]))
    # seq_len=2, 4 tokens => samples [0, 0, 1, 1] -> stream ids [5, 5, 9, 9]
    loss = torch.zeros(1, 2, 2)
    valid = torch.ones(1, 2, 2, dtype=torch.bool)

    capture.observe_loss(loss, valid)

    assert list(buffer.stream_ids()) == [5, 5, 9, 9]


def test_observe_loss_defaults_stream_id_to_zero_without_stream_ids(make_capture) -> None:
    capture, buffer = make_capture(interval=1)
    _capturing(capture)
    loss = torch.zeros(1, 1, _TOKENS)
    valid = torch.ones(1, 1, _TOKENS, dtype=torch.bool)

    capture.observe_loss(loss, valid)

    assert list(buffer.stream_ids()) == [0] * _TOKENS


def test_observe_loss_raises_when_shorter_than_tokens(make_capture) -> None:
    capture, _ = make_capture(interval=1)
    _capturing(capture)

    with pytest.raises(ValueError):
        capture.observe_loss(torch.zeros(1, 1, _TOKENS - 1), torch.ones(1, 1, _TOKENS - 1, dtype=torch.bool))


def test_step_end_is_a_noop_when_not_active(make_capture) -> None:
    capture, buffer = make_capture(interval=0)

    capture.begin_step(1)
    capture.step_end()

    assert buffer.state == STATE_FREE


def test_step_end_marks_the_buffer_ready_with_the_captured_step(make_capture) -> None:
    capture, buffer = make_capture(interval=1)
    capture.begin_step(5)

    capture.step_end()

    assert buffer.state == STATE_READY
    assert buffer.step == 5
    assert capture._active is False


def test_close_releases_the_buffer_so_it_can_be_closed_again(dump_buffers) -> None:
    buffer = dump_buffers(_ENTRIES, _TOKENS, _HIDDEN)
    config = DumpConfig(interval=1, entries=_ENTRIES, tokens=_TOKENS, max_bytes=10**9)
    capture = DumpCapture(config, buffer, torch.device("cpu"), global_rank=0)

    capture.close()
    buffer.close()  # idempotent: DumpCapture.close() already closed it


@pytest.mark.parametrize("config", [None, {}, {"interval": 0, "points": ["residual"], "layers": [0]}])
def test_dump_settings_without_a_positive_interval_disable_dumps(config) -> None:
    assert DumpSettings.parse(config) == DumpSettings(interval=0, points=(), layers=(), tokens=0, max_gb=0.0)


def test_dump_settings_take_the_configured_points_and_layers_with_default_tokens_and_budget() -> None:
    settings = DumpSettings.parse({"interval": 50, "points": ["residual", "embed"], "layers": [3, 1]})

    assert settings == DumpSettings(interval=50, points=("residual", "embed"), layers=(3, 1), tokens=256, max_gb=10.0)


@pytest.mark.parametrize(("config", "message"), [
    ({"interval": 5, "points": ["residual"]}, "requires points and layers"),
    ({"interval": 5, "layers": [0]}, "requires points and layers"),
    ({"interval": 5, "points": ["residual", "swiglu_in"], "layers": [0]}, r"unsupported points \['swiglu_in'\]"),
    ({"interval": 5, "points": ["residual"], "layers": [0], "tokens": 0}, "tokens must be positive, got 0"),
    ({"interval": 5, "points": ["residual"], "layers": [0], "max_gb": -1}, "max_gb must be positive, got -1.0"),
])
def test_dump_settings_reject_an_unusable_config(config, message) -> None:
    with pytest.raises(ActivationDumpConfigError, match=message):
        DumpSettings.parse(config)


def test_dump_entries_take_layerless_points_once_and_decoder_points_in_the_layers_that_observe_them() -> None:
    settings = DumpSettings(interval=10, points=("residual", "final_norm", "shared_expert_out"), layers=(2, 0, 5),
                            tokens=4, max_gb=1.0)

    entries = settings.entries({"residual": [0, 1, 2], "shared_expert_out": [5]})

    assert entries == (DumpEntry("residual", 2), DumpEntry("residual", 0), DumpEntry("final_norm", -1),
                       DumpEntry("shared_expert_out", 5))


def test_dump_entries_reject_settings_the_model_has_no_entry_for() -> None:
    settings = DumpSettings(interval=10, points=("residual",), layers=(7,), tokens=4, max_gb=1.0)

    with pytest.raises(ActivationDumpConfigError, match="produced no valid"):
        settings.entries({"residual": [0, 1]})


def test_dump_settings_config_carries_the_budget_in_bytes() -> None:
    settings = DumpSettings(interval=10, points=("embed",), layers=(0,), tokens=4, max_gb=1.5)

    assert settings.config(_ENTRIES) == DumpConfig(interval=10, entries=_ENTRIES, tokens=4, max_bytes=1_500_000_000)
