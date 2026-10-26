"""Unit tests for the Multi-Token Prediction heads (:mod:`mtp`).

Covers the standalone :class:`MTPBlock` (build, forward shapes, FLOP accounting,
``input_proj`` dim validation) and the :class:`MTP` container (stacked output and
the sequential embedding-roll semantics, where head ``k`` sees the input
embeddings rolled left by ``k+1`` with the tail zeroed).
"""

import copy
import dataclasses
import unittest

import torch
from torchtitan.components.loss import ChunkedCELoss, IGNORE_INDEX, MTPLoss

from torchtitan.models.common.attention import (
    GQAttention,
    MaskType,
    MLA,
    ScaledDotProductAttention,
)
from torchtitan.models.common.attention.qkv import QKVLinear
from torchtitan.models.common.decoder_blocks import Decoder, MTP, MTPBlock
from tests.unit_tests.config_helpers import (
    init_policy,
    load_experiment_config,
    load_model_spec,
)
from torchtitan.models.common.decoder_blocks.transformer_block import (
    ResidualTransformerBlock,
)
from torchtitan.config.schema.models.common.decoder_blocks.layer_group import (
    LayerGroupConfig,
)
from torchtitan.models.common.embeddings.rope import ComplexRoPE
from torchtitan.models.common.ffn import FeedForward
from torchtitan.models.common.nn_modules import Linear
from torchtitan.models.common.norms import GatedRMSNorm, RMSNorm
from torchtitan.models.gift_moe._base.state_dict_adapter import GiftMoEStateDictAdapter


_VOCAB = 16


# Tiny dims keep the test CPU-fast.
_DIM = 64
_N_HEADS = 4
_HEAD_DIM = _DIM // _N_HEADS
_HIDDEN = 128
_SEQ_LEN = 8
_BSZ = 2


def _block_config() -> "ResidualTransformerBlock.Config":
    """A tiny GQA + dense-FFN pre-norm block (no RoPE)."""
    return ResidualTransformerBlock.Config(
        attention=GQAttention.Config(
            n_heads=_N_HEADS,
            dim=_DIM,
            n_kv_heads=_N_HEADS,
            head_dim=_HEAD_DIM,
            qkv_linear=QKVLinear.Config(
                head_dim=_HEAD_DIM,
                wq=Linear.Config(in_features=_DIM, out_features=_N_HEADS * _HEAD_DIM),
                wkv=Linear.Config(in_features=_DIM, out_features=_N_HEADS * _HEAD_DIM),
            ),
            wo=Linear.Config(in_features=_N_HEADS * _HEAD_DIM, out_features=_DIM),
            inner_attention=ScaledDotProductAttention.Config(),
            use_rope=False,
            mask_type=MaskType.CAUSAL,
        ),
        feed_forward=FeedForward.Config(
            dim=_DIM,
            hidden_dim=_HIDDEN,
            w1=Linear.Config(in_features=_DIM, out_features=_HIDDEN),
            w2=Linear.Config(in_features=_HIDDEN, out_features=_DIM),
            w3=Linear.Config(in_features=_DIM, out_features=_HIDDEN),
        ),
        attention_norm=RMSNorm.Config(normalized_shape=_DIM),
        ffn_norm=RMSNorm.Config(normalized_shape=_DIM),
    )


def _module_config(*, in_features: int = 2 * _DIM) -> "MTPBlock.Config":
    return MTPBlock.Config(
        dim=_DIM,
        block=_block_config(),
        input_proj=Linear.Config(in_features=in_features, out_features=_DIM),
        token_norm=RMSNorm.Config(normalized_shape=_DIM),
        hidden_norm=RMSNorm.Config(normalized_shape=_DIM),
        final_norm=RMSNorm.Config(normalized_shape=_DIM),
    )


class TestMTPBlock(unittest.TestCase):
    def test_input_proj_dim_validation(self):
        # in_features must be 2*dim (concat of hidden + embedding).
        with self.assertRaises(ValueError):
            _module_config(in_features=_DIM).build()

    def test_flops_is_block_plus_proj(self):
        cfg = _module_config()
        expected = cfg.block.fwd_flops_per_token(
            _SEQ_LEN
        ) + cfg.input_proj.fwd_flops_per_token(_SEQ_LEN)
        self.assertEqual(cfg.fwd_flops_per_token(_SEQ_LEN), expected)
        self.assertGreater(cfg.fwd_flops_per_token(_SEQ_LEN), 0)


class TestMTPContainer(unittest.TestCase):
    def test_flops_sum_over_heads(self):
        cfg = MTP.Config(layers=[_module_config() for _ in range(2)])
        self.assertEqual(
            cfg.fwd_flops_per_token(_SEQ_LEN),
            sum(m.fwd_flops_per_token(_SEQ_LEN) for m in cfg.layers),
        )

    def test_embedding_roll_semantics(self):
        """Head k must receive the embeddings rolled left by (k+1), tail zeroed."""
        num_heads = 3
        mtp = MTP.Config(layers=[_module_config() for _ in range(num_heads)]).build()

        captured: list[torch.Tensor] = []

        def make_stub():
            def stub(prev_hidden, token_emb, *args, **kwargs):
                captured.append(token_emb.clone())
                return prev_hidden  # pass hidden through unchanged

            return stub

        for layer in mtp.layers.values():
            layer.forward = make_stub()

        emb = torch.randn(_BSZ, _SEQ_LEN, _DIM)
        mtp(torch.randn(_BSZ, _SEQ_LEN, _DIM), emb, rope_cache=None)

        self.assertEqual(len(captured), num_heads)
        for k, seen in enumerate(captured):
            shift = k + 1
            expected = torch.roll(emb, shifts=-shift, dims=1)
            expected[:, -shift:, :] = 0.0
            torch.testing.assert_close(seen, expected)


def _make_tiny_decoder(num_mtp: int, *, n_layers: int = 2) -> "Decoder":
    """A minimal Decoder with MTP enabled (GQA + dense FFN, no RoPE)."""
    # The trunk and the heads place the same named block; materialize() derives
    # the embedding, head, norms and MTP projections around it.
    cfg = Decoder.Config(
        dim=_DIM,
        vocab_size=_VOCAB,
        blocks={"block": _block_config()},
        layers=[LayerGroupConfig(block="block", repeat=n_layers)],
        mtp=(
            MTP.Config(layers=[LayerGroupConfig(block="block", repeat=num_mtp)])
            if num_mtp
            else None
        ),
        norm=RMSNorm.Config(normalized_shape=_DIM),
        rope=ComplexRoPE.Config(dim=16, max_seq_len=_SEQ_LEN),
        init=init_policy("depth_scaled"),
    )
    return cfg.materialize().build()


class TestDecoderMTPIntegration(unittest.TestCase):
    def test_eval_returns_main_logits(self):
        model = _make_tiny_decoder(2)
        model.eval()
        tokens = torch.randint(0, _VOCAB, (_BSZ, _SEQ_LEN))
        out = model(tokens)
        # Eval skips MTP -> standard [B, S, vocab] logits.
        self.assertEqual(out.shape, (_BSZ, _SEQ_LEN, _VOCAB))

    def test_end_to_end_with_loss(self):
        num_mtp = 2
        model = _make_tiny_decoder(num_mtp)
        model.train()
        model._skip_lm_head = True
        loss_fn = MTPLoss.Config(num_chunks=2, loss_weights=[0.3, 0.3]).build()
        loss_fn.set_lm_head(model.lm_head)

        tokens = torch.randint(0, _VOCAB, (_BSZ, _SEQ_LEN))
        labels = torch.randint(0, _VOCAB, (_BSZ, _SEQ_LEN))
        gvt = (labels != IGNORE_INDEX).sum()

        pred = model(tokens)
        loss = loss_fn(pred, labels, gvt)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        # Gradient must reach the MTP modules and the trunk.
        embed_grad = model.tok_embeddings.weight.grad
        self.assertIsNotNone(embed_grad)
        # At least one MTP parameter receives gradient.
        self.assertTrue(
            any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in model.mtp.parameters()
            )
        )


class TestMTPLoss(unittest.TestCase):
    def _make_loss(self, weights):
        loss = MTPLoss.Config(num_chunks=2, loss_weights=weights).build()
        loss.set_lm_head(Linear.Config(in_features=_DIM, out_features=_VOCAB).build())
        return loss

    def _stacked_pred(self, num_mtp, *, requires_grad=True):
        return torch.randn(
            num_mtp + 1, _BSZ, _SEQ_LEN, _DIM, requires_grad=requires_grad
        )

    def test_scalar_loss_and_grad_routing(self):
        num_mtp = 2
        loss = self._make_loss([0.3, 0.3])
        pred = self._stacked_pred(num_mtp)
        labels = torch.randint(0, _VOCAB, (_BSZ, _SEQ_LEN))
        gvt = (labels != IGNORE_INDEX).sum()
        out = loss(pred, labels, gvt)
        self.assertEqual(out.shape, ())
        self.assertTrue(torch.isfinite(out))
        out.backward()
        self.assertIsNotNone(pred.grad)
        self.assertEqual(pred.grad.shape, (num_mtp + 1, _BSZ, _SEQ_LEN, _DIM))
        # Every head should receive gradient.
        for head in range(num_mtp + 1):
            self.assertGreater(pred.grad[head].abs().sum().item(), 0.0)

    def test_zero_weights_equals_main_only(self):
        """With all MTP weights 0, the total equals the main head's CE alone."""
        num_mtp = 2
        torch.manual_seed(0)
        pred = self._stacked_pred(num_mtp, requires_grad=False)
        labels = torch.randint(0, _VOCAB, (_BSZ, _SEQ_LEN))
        gvt = (labels != IGNORE_INDEX).sum()

        full = self._make_loss([0.0, 0.0])
        # Reuse the same lm_head weights for a fair comparison.
        main_only = self._make_loss([0.0, 0.0])
        main_only.lm_head.load_state_dict(full.lm_head.state_dict())

        total = full(pred, labels, gvt)
        main = main_only._chunked_loss_and_grad(pred[0], labels, gvt)[0]
        torch.testing.assert_close(total, main)

    def test_loss_weights_length_validation(self):
        loss = self._make_loss([0.3])  # 1 weight
        pred = self._stacked_pred(2)  # but 2 MTP heads
        labels = torch.randint(0, _VOCAB, (_BSZ, _SEQ_LEN))
        with self.assertRaises(ValueError):
            loss(pred, labels, (labels != IGNORE_INDEX).sum())

    def test_eval_3d_fallback(self):
        """In eval the model returns [B, S, D] (no MTP); MTPLoss falls back to
        the main-only chunked CE rather than erroring on the missing head dim."""
        loss = self._make_loss([0.3, 0.3])
        hidden = torch.randn(_BSZ, _SEQ_LEN, _DIM)
        labels = torch.randint(0, _VOCAB, (_BSZ, _SEQ_LEN))
        out = loss(hidden, labels, (labels != IGNORE_INDEX).sum())
        self.assertEqual(out.shape, ())
        self.assertTrue(torch.isfinite(out))

    def test_shift_labels(self):
        labels = torch.arange(_SEQ_LEN).unsqueeze(0).expand(_BSZ, _SEQ_LEN).contiguous()
        shifted = MTPLoss._shift_labels(labels, 2)
        # Position i now holds the label originally at i+2; last 2 are ignored.
        torch.testing.assert_close(shifted[:, :-2], labels[:, 2:])
        self.assertTrue((shifted[:, -2:] == IGNORE_INDEX).all())

    def test_shift_labels_drops_targets_across_a_document_boundary(self):
        # Two documents of 4 in an 8-long sequence, so positions restart at 4.
        labels = torch.arange(8).unsqueeze(0)
        positions = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
        shifted = MTPLoss._shift_labels(labels, 2, positions)
        # Positions 2 and 3 would predict into the next document; 6 and 7 wrap.
        self.assertEqual(
            shifted[0].tolist(),
            [2, 3, IGNORE_INDEX, IGNORE_INDEX, 6, 7, IGNORE_INDEX, IGNORE_INDEX],
        )

    def test_shift_labels_without_boundaries_matches_the_plain_roll(self):
        labels = torch.arange(_SEQ_LEN).unsqueeze(0).expand(_BSZ, _SEQ_LEN).contiguous()
        positions = torch.arange(_SEQ_LEN).unsqueeze(0).expand(_BSZ, _SEQ_LEN)
        for k in (1, 2, 3):
            torch.testing.assert_close(
                MTPLoss._shift_labels(labels, k, positions.contiguous()),
                MTPLoss._shift_labels(labels, k),
            )

    def test_pack_targets_round_trips_positions(self):
        loss = self._make_loss([0.3])
        labels = torch.randint(0, _VOCAB, (_BSZ, _SEQ_LEN))
        positions = torch.arange(_SEQ_LEN).unsqueeze(0).expand(_BSZ, _SEQ_LEN)
        packed = loss.pack_targets(labels, {"positions": positions})
        self.assertEqual(packed.shape, (_BSZ, 2, _SEQ_LEN))
        torch.testing.assert_close(packed[:, 0], labels)
        torch.testing.assert_close(packed[:, 1], positions.to(labels.dtype))
        # Nothing to fold without positions, and a plain CE never folds.
        torch.testing.assert_close(loss.pack_targets(labels, {}), labels)
        torch.testing.assert_close(
            ChunkedCELoss(ChunkedCELoss.Config(num_chunks=2)).pack_targets(
                labels, {"positions": positions}
            ),
            labels,
        )


class TestMTPPipelineParallel(unittest.TestCase):
    """Pipeline-parallel support: FQN split + the input-embedding carry.

    These exercise the PP code paths on CPU without a process group: the stage
    split is pure logic, and the carry is validated by running the split stage
    modules in sequence and comparing to the unsplit forward.
    """

    def test_fqn_split_places_mtp_in_last_stage(self):
        from torchtitan.distributed.pipeline_parallel import (
            _generate_fqn_per_model_part,
            _partition_segments,
        )

        # The MTP heads fall in the output tail (they follow the last non-block
        # run) and are reported as blocks the last stage carries — derived from
        # the module tree, not declared by the model.
        stem, body, output, output_blocks = _partition_segments(_make_tiny_decoder(2))
        self.assertEqual(stem, ["tok_embeddings"])
        self.assertEqual(output, ["norm", "lm_head", "mtp.layers.0", "mtp.layers.1"])
        self.assertEqual(output_blocks, 2)
        self.assertEqual(_partition_segments(_make_tiny_decoder(0))[3], 0)
        self.assertEqual(
            _partition_segments(_make_tiny_decoder(0))[2], ["norm", "lm_head"]
        )

        kwargs = dict(input_module_fqns=stem, output_module_fqns=output)
        # Single stage gets everything including the heads.
        single = _generate_fqn_per_model_part(1, body, **kwargs)
        self.assertIn("mtp.layers.1", single[0])
        # Multi-stage: heads only in the last stage, alongside norm + lm_head.
        stages = _generate_fqn_per_model_part(2, body, **kwargs)
        self.assertIn("mtp.layers.0", stages[-1])
        self.assertIn("lm_head", stages[-1])
        self.assertTrue(all(not f.startswith("mtp") for s in stages[:-1] for f in s))
        self.assertIn("tok_embeddings", stages[0])
        # A model without heads never emits an mtp FQN.
        stem0, body0, output0, _ = _partition_segments(_make_tiny_decoder(0))
        no_mtp = _generate_fqn_per_model_part(
            2, body0, input_module_fqns=stem0, output_module_fqns=output0
        )
        self.assertTrue(all(not f.startswith("mtp") for s in no_mtp for f in s))

    def test_carry_through_split_stages_matches_single(self):
        """Splitting into PP stages and threading the embedding carry reproduces
        the unsplit MTP forward exactly."""
        from torchtitan.distributed.pipeline_parallel import (
            _generate_fqn_per_model_part,
            _partition_segments,
            _split_module,
        )

        n_layers, num_mtp = 4, 2
        model = _make_tiny_decoder(num_mtp, n_layers=n_layers)
        model.train()
        model._skip_lm_head = True
        tokens = torch.randint(0, _VOCAB, (_BSZ, _SEQ_LEN))

        reference = model(tokens)  # unsplit stacked output
        self.assertEqual(reference.shape, (num_mtp + 1, _BSZ, _SEQ_LEN, _DIM))

        stem, body, output, _ = _partition_segments(model)
        stage_fqns = _generate_fqn_per_model_part(
            2, body, input_module_fqns=stem, output_module_fqns=output
        )
        stage0 = _split_module(model, stage_fqns[0])
        stage1 = _split_module(model, stage_fqns[1])
        for stage in (stage0, stage1):
            stage.train()
            stage._skip_lm_head = True

        # First stage emits the (hidden, input_embeds) carry tuple ...
        out0 = stage0(tokens)
        self.assertIsInstance(out0, tuple)
        self.assertEqual(len(out0), 2)
        # ... which the last stage consumes to produce the stacked output.
        out1 = stage1(*out0)
        torch.testing.assert_close(out1, reference)


class TestMTPStateDictAdapter(unittest.TestCase):
    """HF DeepSeek-V3 <-> torchtitan key mapping for the MTP heads."""

    def _adapter_and_mtp_sd(self, *, dense_head: bool):
        # The heads are placed from a named trunk block: the dense or the MoE one.
        head = "mla_dense" if dense_head else "mla_moe"
        spec = load_model_spec(
            "gift_moe/debugmodel.yaml",
            {"config": {"mtp": {"layers": [{"block": head, "repeat": 2}]}}},
        )
        cfg = spec.model
        adapter = GiftMoEStateDictAdapter(cfg, None)
        model = cfg.build()
        model.init_states()
        sd = model.state_dict()
        mtp_sd = {k: v for k, v in sd.items() if k.startswith("mtp.")}
        return cfg, adapter, mtp_sd

    def test_dense_round_trip(self):
        """to_hf -> from_hf recovers every MTP weight exactly (dense block)."""
        cfg, adapter, mtp_sd = self._adapter_and_mtp_sd(dense_head=True)
        n_main = len(cfg.layers)

        hf = adapter.to_hf(mtp_sd)
        # MTP heads land at HF layer indices [n_main, n_main + num_mtp).
        hf_layers = sorted(
            {int(k.split(".")[2]) for k in hf if k.startswith("model.layers.")}
        )
        self.assertEqual(hf_layers, [n_main, n_main + 1])
        # MTP-specific HF tensors are emitted.
        for name in ("enorm", "hnorm", "eh_proj", "shared_head.norm"):
            self.assertTrue(any(name in k for k in hf), f"missing {name} in HF keys")

        back = adapter.from_hf(hf)
        self.assertEqual(set(back), set(mtp_sd))
        for k, v in mtp_sd.items():
            torch.testing.assert_close(back[k], v)

    def test_main_expert_round_trip(self):
        """to_hf -> from_hf round-trips the main (non-MTP) MoE experts exactly.

        Regression guard for the adapter map matching the real ``GroupedExperts``
        param names (``w13_E2FD``/``w2_EDF``); a bare ``w1``/``w2``/``w3``
        map (the state introduced by #3425) KeyErrors here."""
        spec = load_model_spec("gift_moe/debugmodel.yaml")
        cfg = spec.model
        adapter = GiftMoEStateDictAdapter(cfg, None)
        model = cfg.build()
        model.init_states()
        sd = model.state_dict()
        main_experts = {k: v for k, v in sd.items() if "moe.experts" in k}
        self.assertTrue(main_experts, "debug model should have MoE expert weights")

        back = adapter.from_hf(adapter.to_hf(main_experts))
        self.assertEqual(set(back), set(main_experts))
        for k, v in main_experts.items():
            torch.testing.assert_close(back[k], v)

    def test_grouped_expert_helpers_round_trip(self):
        """The shared ``_grouped_experts_{to,from}_hf`` helpers (used by both the
        main and MTP conversion paths) round-trip a grouped-expert weight on the
        offline path: split into per-expert HF tensors, then concat back."""
        _, adapter, _ = self._adapter_and_mtp_sd(dense_head=True)
        num_experts = 4
        grouped = torch.randn(num_experts, 8, 6)
        titan_key = "layers.{}.moe.experts.w1"
        hf_key = "model.layers.{}.mlp.experts.{}.gate_proj.weight"

        hf_out = adapter._grouped_experts_to_hf(
            grouped, titan_key, {titan_key: hf_key}, "0", num_experts
        )
        self.assertEqual(len(hf_out), num_experts)

        expert_weights_by_layer: dict = {}
        converted = None
        for expert in range(num_experts):
            converted = adapter._grouped_experts_from_hf(
                expert_weights_by_layer,
                titan_key,
                "0",
                expert,
                hf_out[hf_key.format(0, expert)],
                num_experts,
            )
        assert converted is not None
        key, stacked = converted
        self.assertEqual(key, titan_key.format(0))
        torch.testing.assert_close(stacked, grouped)
        self.assertEqual(expert_weights_by_layer, {})

    def test_moe_head_round_trip(self):
        """The other head kind, placed from the MoE trunk block."""
        _, adapter, mtp_sd = self._adapter_and_mtp_sd(dense_head=False)
        back = adapter.from_hf(adapter.to_hf(mtp_sd))
        self.assertEqual(set(back), set(mtp_sd))
        for k, v in mtp_sd.items():
            torch.testing.assert_close(back[k], v)

    def test_a_checkpoint_of_the_other_head_kind_is_rejected(self):
        """The maps hold both kinds, so a mismatch would otherwise leave the head
        at its init (dense model, MoE checkpoint reads ``num_experts`` off None)."""
        for from_dense, into_dense in ((True, False), (False, True)):
            with self.subTest(checkpoint="dense" if from_dense else "moe"):
                _, exporter, mtp_sd = self._adapter_and_mtp_sd(dense_head=from_dense)
                hf = exporter.to_hf(mtp_sd)
                _, loader, _ = self._adapter_and_mtp_sd(dense_head=into_dense)
                with self.assertRaises(ValueError):
                    loader.from_hf(hf)

    def test_eh_proj_half_swap(self):
        """input_proj <-> eh_proj swaps the two input-feature halves (involution)."""
        _, adapter, mtp_sd = self._adapter_and_mtp_sd(dense_head=True)
        w = mtp_sd["mtp.layers.0.input_proj.weight"]
        swapped = adapter._swap_eh_proj_halves(w)
        self.assertFalse(torch.equal(w, swapped))
        torch.testing.assert_close(adapter._swap_eh_proj_halves(swapped), w)

    def test_shared_clones_skipped_on_load(self):
        """HF embed_tokens / shared_head.head clones per MTP layer are dropped."""
        cfg, adapter, _ = self._adapter_and_mtp_sd(dense_head=True)
        n_main = len(cfg.layers)
        fake_hf = {
            f"model.layers.{n_main}.embed_tokens.weight": torch.randn(4, 4),
            f"model.layers.{n_main}.shared_head.head.weight": torch.randn(4, 4),
        }
        self.assertEqual(adapter.from_hf(fake_hf), {})

    def test_shared_clones_emitted_on_export(self):
        """to_hf emits per-MTP-layer embed_tokens / shared_head.head clones from
        the shared main tensors, so the exported HF checkpoint is loadable, and
        from_hf drops them again (full round trip)."""
        cfg, adapter, mtp_sd = self._adapter_and_mtp_sd(dense_head=True)
        n_main = len(cfg.layers)
        embed = torch.randn(cfg.vocab_size, cfg.dim)
        head = torch.randn(cfg.vocab_size, cfg.dim)
        # A full-ish export: the MTP weights plus the shared main tensors.
        full = {**mtp_sd, "tok_embeddings.weight": embed, "lm_head.weight": head}

        hf = adapter.to_hf(full)
        for i in range(len(cfg.mtp.layers)):
            hf_layer = n_main + i
            ek = f"model.layers.{hf_layer}.embed_tokens.weight"
            hk = f"model.layers.{hf_layer}.shared_head.head.weight"
            torch.testing.assert_close(hf[ek], embed)
            torch.testing.assert_close(hf[hk], head)

        # Loading drops the clones; the MTP weights round-trip exactly.
        back_mtp = {
            k: v for k, v in adapter.from_hf(hf).items() if k.startswith("mtp.")
        }
        self.assertEqual(set(back_mtp), set(mtp_sd))
        for k, v in mtp_sd.items():
            torch.testing.assert_close(back_mtp[k], v)

    def test_moe_block_expert_key_mapping(self):
        """MoE MTP block reuses the main expert convention under mtp.*.block.*."""
        _, adapter, _ = self._adapter_and_mtp_sd(dense_head=False)
        self.assertEqual(
            adapter.mtp_from_hf_map["model.layers.{}.mlp.experts.{}.gate_proj.weight"],
            "mtp.layers.{}.block.moe.experts.w1_EFD",
        )
        self.assertEqual(
            adapter.mtp_from_hf_map["model.layers.{}.eh_proj.weight"],
            "mtp.layers.{}.input_proj.weight",
        )


class TestGift35MTP(unittest.TestCase):
    """MTP wired into the gift_3_5 hybrid (GDN + MLA) model."""

    def _mtp_config(self):
        spec = load_model_spec(
            "gift_moe/gift_3_5/debugmodel.yaml",
            {"config": {"mtp": {"layers": [{"block": "mtp", "repeat": 2}]}}},
        )
        return spec.model

    def test_heads_are_mla_dense_blocks_with_gated_norms(self):
        cfg = self._mtp_config()
        self.assertIsNotNone(cfg.mtp)
        self.assertEqual(len(cfg.mtp.layers), 2)
        for module_cfg in cfg.mtp.layers:
            # Built from the trunk's dims: MLA attention, and a dense FFN -- one
            # extra predicted position does not need the trunk's expert capacity.
            self.assertIsInstance(module_cfg.block.attention, MLA.Config)
            self.assertIsNone(module_cfg.block.moe)
            self.assertIsNotNone(module_cfg.block.feed_forward)
            # The head inherits gift_3_5's gated norm family everywhere.
            for norm in (
                module_cfg.token_norm,
                module_cfg.hidden_norm,
                module_cfg.final_norm,
                module_cfg.block.attention_norm,
                module_cfg.block.ffn_norm,
                module_cfg.block.post_attn_norm,
                module_cfg.block.post_ffn_norm,
            ):
                self.assertIsInstance(norm, GatedRMSNorm.Config)

    def test_update_from_config_reaches_mtp_heads(self):
        """The rope contract and the sharding plan both reach the MTP heads.

        A head is an ordinary block behind ``mtp.layers``, so anything the trunk
        gets by walking the config tree has to arrive here too -- the earlier
        per-family sync loop only walked ``self.layers`` and needed a second pass
        for the heads.
        """
        trainer_cfg = load_experiment_config("gift_3_5_debug.yaml")
        model_cfg = trainer_cfg.model_spec.model
        model_cfg.owner().prepare_config(model_cfg, config=trainer_cfg)

        # The cache is sized to the run; the declared context is not touched.
        self.assertEqual(model_cfg.rope.cache_len, trainer_cfg.training.seq_len)
        self.assertEqual(model_cfg.rope.max_seq_len, 4096)
        self.assertLess(model_cfg.rope.cache_len, model_cfg.rope.max_seq_len)
        for module_cfg in model_cfg.mtp.layers:
            attention = module_cfg.block.attention
            # No YaRN in this preset, so no amplitude compensation is claimed.
            self.assertIsNone(attention.yarn_factor)
            self.assertEqual(attention.qk_rope_head_dim, model_cfg.rope.dim)
            # Sharding config populated (head norms + input_proj).
            self.assertIsNotNone(module_cfg.token_norm.sharding_config)
            self.assertIsNotNone(module_cfg.input_proj.sharding_config)

    def test_the_rope_check_reaches_the_mtp_heads(self):
        """A stale ``yarn_factor`` on the MTP heads fails; unset, it is filled."""
        from torchtitan.models.common.embeddings.rope import YarnScaling

        trainer_cfg = load_experiment_config("gift_3_5_debug.yaml")
        model_cfg = trainer_cfg.model_spec.model
        model_cfg.rope = dataclasses.replace(
            model_cfg.rope,
            scaling=YarnScaling(rope_factor=40.0, original_seq_len=4096),
        )
        for layer in model_cfg.layers:
            if isinstance(layer.attention, MLA.Config):
                layer.attention.yarn_factor = 40.0
        # An unset factor would be filled, so only an explicit one can disagree.
        for module_cfg in model_cfg.mtp.layers:
            module_cfg.block.attention.yarn_factor = 20.0
        with self.assertRaisesRegex(ValueError, "set yarn_factor=40.0"):
            model_cfg.owner().prepare_config(model_cfg, config=trainer_cfg)

        for module_cfg in model_cfg.mtp.layers:
            module_cfg.block.attention.yarn_factor = None
        model_cfg.owner().prepare_config(model_cfg, config=trainer_cfg)
        for module_cfg in model_cfg.mtp.layers:
            self.assertEqual(module_cfg.block.attention.yarn_factor, 40.0)

    def test_gdn_mtp_head_builds_and_shards(self):
        """A head may use GDN (linear) attention instead of MLA: the head is a
        config the caller writes, so swapping its attention is enough. The
        non-RoPE GDN attention gets no TP plan (like a trunk GDN layer) while the
        head norms, input_proj and FFN are still sharded."""
        trainer_cfg = load_experiment_config("gift_3_5_debug.yaml")
        model_cfg = trainer_cfg.model_spec.model
        gdn = next(
            layer.attention
            for layer in model_cfg.layers
            if not isinstance(layer.attention, MLA.Config)
        )
        for module_cfg in model_cfg.mtp.layers:
            module_cfg.block.attention = copy.deepcopy(gdn)

        model_cfg.owner().prepare_config(
            model_cfg, config=trainer_cfg
        )  # must NOT raise
        for module_cfg in model_cfg.mtp.layers:
            self.assertNotIsInstance(module_cfg.block.attention, MLA.Config)
            self.assertIsNotNone(module_cfg.input_proj.sharding_config)
            self.assertIsNotNone(module_cfg.final_norm.sharding_config)
            self.assertIsNotNone(module_cfg.block.feed_forward.w1.sharding_config)

    def test_ep_mtp_experts_match_main_sharding(self):
        """A head given a MoE explicitly must have its experts sharded exactly
        like the main layers' (same mesh axes + placements), so the FSDP expert
        placement fn agrees with the params' EP-DTensor sharding (no mismatch).

        Heads are dense by default, so the MoE is put back here -- this is the
        path a caller takes by placing a MoE block in ``mtp.layers``.
        """
        cfg = load_experiment_config(
            "gift_3_5_debug.yaml", "--parallelism.expert_parallel_degree", "2"
        )
        model_cfg = cfg.model_spec.model
        trunk_moe = next(
            lc.moe for lc in model_cfg.layers if getattr(lc, "moe", None) is not None
        )
        for module_cfg in model_cfg.mtp.layers:
            module_cfg.block.feed_forward = None
            module_cfg.block.moe = copy.deepcopy(trunk_moe)
        model_cfg.owner().prepare_config(model_cfg, config=cfg)

        main_moe = next(
            lc.moe for lc in model_cfg.layers if getattr(lc, "moe", None) is not None
        )
        main_shardings = main_moe.experts.sharding_config.state_shardings
        for module_cfg in model_cfg.mtp.layers:
            self.assertEqual(
                module_cfg.block.moe.experts.sharding_config.state_shardings,
                main_shardings,
            )


if __name__ == "__main__":
    unittest.main()
