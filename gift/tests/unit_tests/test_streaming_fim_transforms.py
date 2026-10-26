"""Unit tests for FIMTransform: per-sample determinism and fail-fast guards"""

import pytest

pytest.importorskip("streaming")
pytest.importorskip("tree_sitter")
pytest.importorskip("tree_sitter_language_pack")

import torch

from tests.unit_tests.streaming_helpers import CharTokenizer
from torchtitan.datasets.streaming.text import transforms as transforms_module
from torchtitan.datasets.streaming.text.transforms import FIMTransform
from torchtitan.config.schema.datasets.streaming.text.transforms import (
    DEFAULT_FIM_RATES,
)

FIM_VARIANTS = ("vanilla_fim", "inline_fim", "multiline_fim", "syntax_aware_fim")

_CODE_TEXT = "def add(a, b):\n    total = a + b\n    return total\n"


def _code_tokens() -> torch.Tensor:
    return torch.tensor(CharTokenizer().encode(_CODE_TEXT), dtype=torch.int64)


def _make_transform(fim_rates, *, seed=1234, tokenizer=None) -> FIMTransform:
    return FIMTransform(
        tokenizer=tokenizer if tokenizer is not None else CharTokenizer(),
        max_seq_len=64,
        seed=seed,
        default_fim_rates=fim_rates,
    )


@pytest.mark.parametrize("variant", FIM_VARIANTS)
def test_same_seed_epoch_idx_is_bit_identical(variant):
    """
    Randomness must come only from (seed, epoch, idx), so independent
    transform instances produce the same sample bit for bit on every rank
    """
    tokens = _code_tokens()
    first = _make_transform({variant: 1.0})
    second = _make_transform({variant: 1.0})

    out_first = first(tokens, stream_id=0, idx=7, epoch=1)
    out_second = second(tokens, stream_id=0, idx=7, epoch=1)

    assert torch.equal(out_first, out_second)
    assert torch.equal(out_first, first(tokens, stream_id=0, idx=7, epoch=1))

    tokenizer = CharTokenizer()
    assert int(out_first[0]) in {
        tokenizer.token_to_id("<|fim_prefix|>"),
        tokenizer.token_to_id("<|fim_suffix|>"),
    }
    assert len(out_first) <= 64
    assert int(out_first[-1]) == tokenizer.eos_id
    assert int(out_first[-2]) != tokenizer.eos_id


def test_output_varies_across_idx_epoch_and_seed():
    tokens = _code_tokens()
    transform = _make_transform({"vanilla_fim": 1.0})

    def fingerprint(out: torch.Tensor) -> bytes:
        return out.numpy().tobytes()

    by_idx = {
        fingerprint(transform(tokens, stream_id=0, idx=idx, epoch=0))
        for idx in range(8)
    }
    by_epoch = {
        fingerprint(transform(tokens, stream_id=0, idx=0, epoch=epoch))
        for epoch in range(8)
    }
    by_seed = {
        fingerprint(
            _make_transform({"vanilla_fim": 1.0}, seed=seed)(
                tokens, stream_id=0, idx=0, epoch=0
            )
        )
        for seed in range(8)
    }

    assert len(by_idx) > 1
    assert len(by_epoch) > 1
    assert len(by_seed) > 1


def test_zero_rates_leave_tokens_untouched():
    tokens = _code_tokens()
    transform = _make_transform(dict(DEFAULT_FIM_RATES))
    assert transform(tokens, stream_id=0) is tokens


def test_stream_rates_select_by_stream_id():
    tokens = _code_tokens()
    transform = FIMTransform(
        tokenizer=CharTokenizer(),
        max_seq_len=64,
        seed=1,
        stream_fim_rates=[{"vanilla_fim": 0.0}, {"vanilla_fim": 1.0}],
        default_fim_rates={"vanilla_fim": 0.0},
    )

    assert transform(tokens, stream_id=0, idx=5) is tokens
    transformed = transform(tokens, stream_id=1, idx=5)
    assert len(transformed) <= 64
    assert not torch.equal(transformed[: len(tokens)], tokens)


def test_enabled_fim_requires_sample_idx():
    transform = _make_transform({"vanilla_fim": 1.0})
    with pytest.raises(ValueError, match="idx=None"):
        transform(_code_tokens(), stream_id=0)


def test_syntax_aware_fails_fast_when_grammars_unavailable(monkeypatch):
    """Broken tree-sitter must abort construction, not silently degrade FIM"""

    def broken_get_parser(lang):
        raise ImportError(f"no grammar for {lang}")

    monkeypatch.setattr(transforms_module, "get_parser", broken_get_parser)
    with pytest.raises(RuntimeError, match="Tree-sitter grammars failed to load"):
        _make_transform({"syntax_aware_fim": 0.5})


def test_missing_fim_tokens_in_vocab_raise():
    transform = _make_transform(
        {"vanilla_fim": 1.0}, tokenizer=CharTokenizer(with_fim_tokens=False)
    )
    with pytest.raises(RuntimeError, match="do not exist in vocab"):
        transform(_code_tokens(), stream_id=0, idx=0)
