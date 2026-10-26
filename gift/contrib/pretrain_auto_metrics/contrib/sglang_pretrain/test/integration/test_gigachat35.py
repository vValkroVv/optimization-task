"""End-to-end checks that the gigachat3.5 model is supported by the sglang runtime.

These exercise the parts unit tests cannot: real checkpoint loading,
`Gigachat35ForCausalLM` construction, a forward pass producing logits, and the
MTP / EAGLE speculative-decoding path (`Gigachat35ForCausalLMNextN`).

Requires a GPU and the checkpoint from the `gigachat35_checkpoint` fixture, so
they are skipped automatically off-cluster.

The baseline test compares greedy completions against a committed golden file.
Regenerate it (after an intentional model/runtime change) with:

    GIGACHAT35_UPDATE_GOLDENS=1 python -m pytest test/integration/test_gigachat35.py
"""

import json
import os
from pathlib import Path

import pytest

# Prompts whose greedy continuation keeps a top-1/top-2 logprob margin of at least
# 0.125 over all 32 tokens on every test checkpoint, with and without the fp32
# gated-norm gate. Many natural prompts hit exact bf16 ties on these small
# checkpoints, so any last-bit change (e.g. the verify batch shape under MTP)
# flips their golden. max_new_tokens is kept small so a full run is quick.
_PROMPTS = [
    "The Pythagorean theorem states that",
    "def is_prime(n):\n",
    "Q: What is the capital of Japan?\nA:",
]
_GREEDY = {"temperature": 0.0, "max_new_tokens": 32}
_TP_SIZE = int(os.environ.get("GIGACHAT35_TP_SIZE", "1"))

# Engine settings shared by every launch — mirrors bench_mtp.sh.
_ENGINE_KWARGS = dict(
    trust_remote_code=True,
    tp_size=_TP_SIZE,
    mem_fraction_static=0.75,
    disable_radix_cache=True,
)

_GOLDENS_DIR = Path(__file__).parent / "goldens"
_UPDATE_GOLDENS = os.environ.get("GIGACHAT35_UPDATE_GOLDENS") == "1"


def _golden_path(checkpoint) -> Path:
    """Per-checkpoint golden file (the checkpoint subdir name keeps them distinct)."""
    return _GOLDENS_DIR / f"gigachat35_baseline_{Path(checkpoint).name}.json"


def _generate(model_path, prompts, sampling_params, **extra_engine_kwargs):
    """Launch an sglang Engine, greedily generate, and tear it down.

    Returns the list of completion strings. Only one Engine is alive at a time so
    the baseline and MTP runs do not compete for GPU memory.
    """
    import sglang as sgl

    engine = sgl.Engine(model_path=str(model_path), **_ENGINE_KWARGS, **extra_engine_kwargs)
    try:
        outputs = engine.generate(prompts, sampling_params)
    finally:
        engine.shutdown()
    return [o["text"] for o in outputs]


@pytest.fixture(scope="module")
def baseline_outputs(gigachat35_checkpoint):
    """Greedy completions with plain (non-speculative) decoding."""
    return _generate(gigachat35_checkpoint, _PROMPTS, _GREEDY)


@pytest.mark.integration
@pytest.mark.gigachat35
@pytest.mark.slow
class TestGigachat35Generation:
    def test_baseline_matches_golden(self, gigachat35_checkpoint, baseline_outputs):
        """Checkpoint loads, model builds, and greedy output matches the golden file.

        Run with GIGACHAT35_UPDATE_GOLDENS=1 to (re)create the golden file after an
        intentional change.
        """
        golden_path = _golden_path(gigachat35_checkpoint)
        completions = dict(zip(_PROMPTS, baseline_outputs))

        if _UPDATE_GOLDENS:
            golden_path.parent.mkdir(parents=True, exist_ok=True)
            golden_path.write_text(
                json.dumps(completions, indent=2, ensure_ascii=False) + "\n"
            )
            pytest.skip(f"updated golden file at {golden_path}")

        if not golden_path.exists():
            pytest.skip(
                f"no golden file at {golden_path}; regenerate with "
                "GIGACHAT35_UPDATE_GOLDENS=1"
            )

        expected = json.loads(golden_path.read_text())
        assert completions == expected, (
            "greedy output drifted from the golden file — if this change is "
            "intentional, regenerate with GIGACHAT35_UPDATE_GOLDENS=1"
        )

    def test_mtp_speculative_decoding_is_lossless(
        self, gigachat35_checkpoint, gigachat35_num_mtp_layers, baseline_outputs
    ):
        """EAGLE speculative decoding must reproduce greedy output exactly.

        Speculative decoding is lossless by construction: the target model
        verifies every drafted token, so greedy output is identical to the
        non-speculative run. A mismatch means the MTP heads
        (`Gigachat35ForCausalLMNextN`) are wired or loaded incorrectly.
        """
        if gigachat35_num_mtp_layers < 1:
            pytest.skip("checkpoint has no MTP heads (num_nextn_predict_layers == 0)")

        steps = gigachat35_num_mtp_layers
        mtp_outputs = _generate(
            gigachat35_checkpoint,
            _PROMPTS,
            _GREEDY,
            speculative_algorithm="EAGLE",
            speculative_num_steps=steps,
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=steps + 1,
        )

        assert len(mtp_outputs) == len(baseline_outputs)
        for prompt, base, mtp in zip(_PROMPTS, baseline_outputs, mtp_outputs):
            assert mtp == base, (
                f"MTP output diverged from baseline for prompt {prompt!r}\n"
                f"  baseline: {base!r}\n"
                f"  mtp:      {mtp!r}"
            )
