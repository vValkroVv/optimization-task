"""Tests for the per-building-block analytical FLOP calculator.

``get_nparams_and_flops`` sums each building block's forward matmul FLOPs from
its real ``Linear`` shapes (see ``get_blockwise_nparams_and_flops`` in
``torchtitan/models/utils.py``). These tests pin that sum against an
independent oracle computed by a *different* method — introspecting the built
``nn.Linear`` modules — so they catch drift in the per-block formulas.
"""

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode
from tests.unit_tests.config_helpers import load_model_spec, PRESET_ROOT
from torchtitan.config import load_raw_document
from torchtitan.models.common.attention.softmax.gqa import BaseAttention
from torchtitan.config.schema.models.common.attention.softmax.gqa import (
    causal_softmax_core_flops,
)


_SEQ_LEN = 512


def _overrides(preset: str, *, sdpa: bool = True, all_to_all: bool = False) -> dict:
    """Per named block: sdpa inner attention and/or an all_to_all MoE dispatcher."""
    blocks = load_raw_document(str(PRESET_ROOT / preset)).data["model"]["config"][
        "blocks"
    ]
    out = {}
    for name, block in blocks.items():
        entry = {}
        if sdpa and "attention" in block:
            entry["attention"] = {"inner_attention": {"type": "sdpa"}}
        if all_to_all and "moe" in block:
            entry["moe"] = {"experts": {"token_dispatcher": {"type": "all_to_all"}}}
        if entry:
            out[name] = entry
    return {"config": {"blocks": out}}


def _build(name: str):
    """Build a (config, model) pair for a debugmodel of the named model."""
    if name == "llama3":
        from torchtitan.models.llama3.model import Llama3Model

        preset = "llama3/debugmodel.yaml"
        cfg = load_model_spec(preset, _overrides(preset)).model
        return cfg, Llama3Model(cfg)
    if name == "llama4":
        from torchtitan.models.llama4.model import Llama4Model

        preset = "llama4/debugmodel.yaml"
        # llama4_chunked blocks need flex inner attention.
        cfg = load_model_spec(
            preset, _overrides(preset, sdpa=False, all_to_all=True)
        ).model
        return cfg, Llama4Model(cfg)
    if name == "qwen3":
        from torchtitan.models.qwen3.model import Qwen3Model

        preset = "qwen3/debugmodel.yaml"
        cfg = load_model_spec(preset, _overrides(preset)).model
        return cfg, Qwen3Model(cfg)
    if name == "deepseek_v3":
        from torchtitan.models.deepseek_v3.model import DeepSeekV3Model

        preset = "deepseek_v3/debugmodel.yaml"
        cfg = load_model_spec(preset, _overrides(preset, all_to_all=True)).model
        return cfg, DeepSeekV3Model(cfg)
    if name == "gpt_oss":
        from torchtitan.models.gpt_oss.model import GptOssModel

        preset = "gpt_oss/debugmodel.yaml"
        cfg = load_model_spec(
            preset, _overrides(preset, sdpa=False, all_to_all=True)
        ).model
        return cfg, GptOssModel(cfg)
    if name == "gift_moe":
        from torchtitan.models.gift_moe._base.model import GiftMoEModel

        preset = "gift_moe/debugmodel.yaml"
        cfg = load_model_spec(preset, _overrides(preset, all_to_all=True)).model
        return cfg, GiftMoEModel(cfg)
    raise ValueError(f"unknown model {name}")


# Subset used for the exact oracle check: dense-GQA, GQA+MoE, MLA+MoE.
_MODELS = ["llama3", "qwen3", "gift_moe"]
# Full set used for the registry sweep (one debugmodel per native model family).
_ALL_MODELS = ["llama3", "llama4", "qwen3", "deepseek_v3", "gpt_oss", "gift_moe"]


def _oracle_fwd_flops_per_token(cfg, model, seq_len: int) -> int:
    """Independent forward-FLOPs-per-token count.

    Uses a different method than the implementation: it introspects the *built*
    ``nn.Linear`` modules for every projection (attention QKV/output, router
    gate, dense/shared FFN, lm_head), then adds the two terms that are not
    plain linears — the routed-expert SwiGLUs (raw grouped parameters; only
    ``top_k`` run per token) and the causal softmax attention core.
    """
    # All projection matmuls are nn.Linear (embeddings are nn.Embedding and so
    # excluded, which is correct — the lookup is not a matmul).
    proj = sum(
        2 * m.in_features * m.out_features
        for m in model.modules()
        if isinstance(m, nn.Linear)
    )

    routed = 0
    core = 0
    for layer in cfg.layers:
        att = layer.attention
        if hasattr(att, "qk_nope_head_dim"):  # MLA
            qk = att.qk_nope_head_dim + att.qk_rope_head_dim
            v = att.v_head_dim
        else:  # GQA-style
            hd = att.head_dim if att.head_dim is not None else att.dim // att.n_heads
            qk = v = hd
        core += causal_softmax_core_flops(
            n_heads=att.n_heads, seq_len=seq_len, qk_head_dim=qk, v_head_dim=v
        )
        moe = getattr(layer, "moe", None)
        if moe is not None:
            # Routed experts use raw grouped parameters (not nn.Linear); each
            # active expert is a SwiGLU = 6*dim*hidden, top_k active per token.
            routed += moe.router.top_k * 6 * moe.experts.dim * moe.experts.hidden_dim

    return proj + routed + core


class TestBlockwiseFlops(unittest.TestCase):
    def test_matches_independent_oracle(self):
        """Reported FLOPs/token == 3 (fwd + 2x bwd) * the independent forward count."""
        for name in _MODELS:
            with self.subTest(model=name):
                cfg, model = _build(name)
                _, flops = cfg.owner().get_nparams_and_flops(cfg, model, _SEQ_LEN)
                expected = 3 * _oracle_fwd_flops_per_token(cfg, model, _SEQ_LEN)
                self.assertEqual(flops, expected)

    def test_nparams_counts_unique_parameters(self):
        """Reported nparams de-duplicates shared (tied) parameters."""
        for name in _MODELS:
            with self.subTest(model=name):
                cfg, model = _build(name)
                nparams, _ = cfg.owner().get_nparams_and_flops(cfg, model, _SEQ_LEN)
                unique = sum(
                    p.numel() for p in {id(p): p for p in model.parameters()}.values()
                )
                self.assertEqual(nparams, unique)

    def test_flops_scale_with_seq_len_via_attention_core(self):
        """Only the attention core scales with seq_len; projections are flat.

        Doubling seq_len adds exactly one extra core term (the projection and
        FFN/MoE terms are sequence-independent).
        """
        cfg, model = _build("gift_moe")
        _, flops_s = cfg.owner().get_nparams_and_flops(cfg, model, _SEQ_LEN)
        _, flops_2s = cfg.owner().get_nparams_and_flops(cfg, model, 2 * _SEQ_LEN)
        core_s = 3 * sum(
            causal_softmax_core_flops(
                n_heads=layer.attention.n_heads,
                seq_len=_SEQ_LEN,
                qk_head_dim=layer.attention.qk_nope_head_dim
                + layer.attention.qk_rope_head_dim,
                v_head_dim=layer.attention.v_head_dim,
            )
            for layer in cfg.layers
        )
        self.assertEqual(flops_2s - flops_s, core_s)


class TestModelRegistrySweep(unittest.TestCase):
    def test_every_model_reports_positive_flops(self):
        """Every native model's debugmodel produces a positive int count.

        This is the enforcement net: a new attention/FFN/MoE/block variant that
        forgets ``fwd_flops_per_token`` raises (``NotImplementedError`` from the
        base, or ``AttributeError``) here rather than silently reporting 0.
        """
        for name in _ALL_MODELS:
            with self.subTest(model=name):
                cfg, model = _build(name)
                nparams, flops = cfg.owner().get_nparams_and_flops(cfg, model, _SEQ_LEN)
                self.assertIsInstance(nparams, int)
                self.assertIsInstance(flops, int)
                self.assertGreater(nparams, 0)
                self.assertGreater(flops, 0)


class TestTransformerBlockSumsSublayers(unittest.TestCase):
    def test_block_equals_attention_plus_channel_mixer(self):
        """A block's FLOPs == attention + (FFN | MoE); norms contribute 0.

        Pins the introspection in ``TransformerBlock.Config.fwd_flops_per_token``
        (it sums over the block's own ``Module.Config`` sublayer fields).
        """
        for name in _MODELS:
            cfg, _ = _build(name)
            for layer in cfg.layers:
                block = layer.fwd_flops_per_token(_SEQ_LEN)
                expected = layer.attention.fwd_flops_per_token(_SEQ_LEN)
                if getattr(layer, "moe", None) is not None:
                    expected += layer.moe.fwd_flops_per_token(_SEQ_LEN)
                elif getattr(layer, "feed_forward", None) is not None:
                    expected += layer.feed_forward.fwd_flops_per_token(_SEQ_LEN)
                # attention_norm + ffn_norm contribute 0.
                self.assertEqual(block, expected)
                self.assertEqual(layer.attention_norm.fwd_flops_per_token(_SEQ_LEN), 0)


class TestFlopCounterCrossCheck(unittest.TestCase):
    def test_dense_matmuls_match_measured(self):
        """Analytical linear-matmul FLOPs == matmuls actually executed.

        Runs a forward pass on a dense model under ``FlopCounterMode`` and
        compares the measured matmul (``aten.mm``/``aten.addmm``) FLOPs to the
        analytical forward count *minus the attention core*. The core is
        excluded because the SDPA kernel's attention matmuls are not counted on
        this path (and MoE expert matmuls reflect actually-dispatched tokens,
        not the analytical active fraction — hence a dense model here). This is
        an independent, graph-level check that the per-block ``Linear`` sums
        match the real executed matmuls.
        """
        B, S = 2, 256
        cfg, model = _build("llama3")
        model.init_states()
        model.eval()

        _, flops_per_token = cfg.owner().get_nparams_and_flops(cfg, model, S)
        core_per_token = sum(
            causal_softmax_core_flops(
                n_heads=layer.attention.n_heads,
                seq_len=S,
                qk_head_dim=(
                    layer.attention.head_dim or cfg.dim // layer.attention.n_heads
                ),
                v_head_dim=(
                    layer.attention.head_dim or cfg.dim // layer.attention.n_heads
                ),
            )
            for layer in cfg.layers
        )
        # forward linear FLOPs/token = (fwd + 2*bwd)/3 - attention core.
        expected_linear = B * S * (flops_per_token // 3 - core_per_token)

        tokens = torch.randint(0, cfg.vocab_size, (B, S))
        with FlopCounterMode(display=False) as fc:
            model(tokens)
        global_counts = fc.flop_counts["Global"]
        measured_matmul = sum(
            flops
            for op, flops in global_counts.items()
            if str(op) in ("aten.mm", "aten.addmm")
        )
        self.assertEqual(measured_matmul, expected_linear)


class TestCausalSoftmaxCoreFlops(unittest.TestCase):
    def test_halves_the_quadratic_term(self):
        """The core applies the causal 1/2 to QK^T + scores*V."""
        n_heads, seq_len, qk, v = 8, 1024, 64, 64
        # Unhalved forward cost of the two matmuls.
        unhalved = 2 * n_heads * seq_len * qk + 2 * n_heads * seq_len * v
        self.assertEqual(
            causal_softmax_core_flops(
                n_heads=n_heads, seq_len=seq_len, qk_head_dim=qk, v_head_dim=v
            ),
            unhalved // 2,
        )


class TestBaseAttentionRaises(unittest.TestCase):
    def test_unimplemented_attention_raises(self):
        """The base fwd_flops_per_token raises rather than silently returning 0."""
        with self.assertRaises(NotImplementedError):
            # The base method only reads type(self).__name__, so any object works.
            BaseAttention.Config.fwd_flops_per_token(SimpleNamespace(), _SEQ_LEN)


if __name__ == "__main__":
    unittest.main()
