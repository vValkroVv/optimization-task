"""HF checkpoint conversion for the GIFT 3.5 hybrid.

Covers :class:`GiftHybridStateDictAdapter`: bit-exact ``from_hf(to_hf(sd))``
round-trips on built debug models (GDN + MLA + MoE + dense + gated norms + MTP,
incl. a GDN MTP head); the GDN fused ``in_proj_qkvz`` + ``conv1d``
grouped<->contiguous surgery; and that FP8 import routes to the quantized reader.
"""

import copy
import json
import tempfile
import types
import unittest

import torch

from tests.unit_tests.config_helpers import load_model_spec
from torchtitan.config.schema.models.common.attention.qkv import QKVLinearConfig
from torchtitan.config.schema.models.common.attention.softmax.gate import (
    SigmoidGateConfig,
)
from torchtitan.config.schema.models.common.nn_modules import LinearConfig
from torchtitan.config.schema.protocols.module import Ones
from torchtitan.models.common.attention import (
    MLA,
    FusedQKVLinear,
    GQAttention,
)
from torchtitan.models.common.nn_modules import Linear
from torchtitan.models.gift_moe.gift_3_5 import GiftHybridStateDictAdapter


def _mtp(n: int) -> dict:
    """*n* MTP heads of the preset's ``mtp`` block (the presets have none)."""
    return {"mtp": {"layers": [{"block": "mtp", "repeat": n}]}} if n else {}


def _randomize(state_dict: dict) -> dict:
    """Fresh distinct values so round-trip mismatches can't hide behind ties."""
    return {
        k: (torch.randn_like(v) if v.dtype.is_floating_point else v.clone())
        for k, v in state_dict.items()
    }


def _give_heads_gdn_attention(model_cfg) -> None:
    """Rewrite the MTP heads to use the trunk's GDN attention instead of MLA."""
    gdn = next(
        layer.attention
        for layer in model_cfg.layers
        if not isinstance(layer.attention, MLA.Config)
    )
    for module_cfg in model_cfg.mtp.layers:
        module_cfg.block.attention = copy.deepcopy(gdn)


_INIT = {"weight": Ones()}


def _gqa_config(mla, *, fused: bool, n_heads: int = 16, n_kv_heads: int = 4):
    """A GQA attention sized like the MLA module it replaces, in either Q/K/V form."""
    dim = mla.dim
    # The rotated width, not dim // n_heads: one RoPE serves the whole stack, so a
    # GQA layer sharing it with MLA layers must rotate the same number of dims.
    head_dim = mla.qk_rope_head_dim
    qkv = QKVLinearConfig(
        wq=LinearConfig(param_init=_INIT), wkv=LinearConfig(param_init=_INIT)
    )
    if fused:
        qkv = FusedQKVLinear.Config(
            head_dim=head_dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            wqkv=Linear.Config(
                in_features=dim,
                out_features=(n_heads + 2 * n_kv_heads) * head_dim,
                param_init=_INIT,
            ),
        )
    return GQAttention.Config(
        dim=dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        qkv_linear=qkv,
        wo=LinearConfig(param_init=_INIT),
        gate=SigmoidGateConfig(gate_proj=LinearConfig(param_init=_INIT)),
        # Same softmax kernel the preset picked for its MLA layers.
        inner_attention=copy.deepcopy(mla.inner_attention),
    )


def _use_gqa_attention(model_cfg, *, fused: bool, heads_too: bool = True) -> None:
    """Swap every softmax-attention (MLA) module for GQA, trunk and MTP heads."""
    for layer in model_cfg.layers:
        if isinstance(layer.attention, MLA.Config):
            layer.attention = _gqa_config(layer.attention, fused=fused)
    if heads_too and model_cfg.mtp is not None:
        for module_cfg in model_cfg.mtp.layers:
            attn = module_cfg.block.attention
            if isinstance(attn, MLA.Config):
                module_cfg.block.attention = _gqa_config(attn, fused=fused)


class TestGiftHybridRoundTrip(unittest.TestCase):
    """Full-model to_hf -> from_hf recovers every weight bit-for-bit."""

    def _round_trip(self, spec):
        cfg = spec.model
        with torch.device("cpu"):
            sd = _randomize(cfg.build().state_dict())
        adapter = GiftHybridStateDictAdapter(cfg, None)

        hf = adapter.to_hf(sd)
        for k in hf:  # nothing left in torchtitan naming
            self.assertTrue(
                k.startswith("model.") or k == "lm_head.weight",
                f"non-HF key leaked into to_hf output: {k}",
            )
        back = adapter.from_hf(hf)
        self.assertEqual(set(back), set(sd), "round-trip changed the key set")
        for k, v in sd.items():
            torch.testing.assert_close(back[k], v, rtol=0, atol=0)
        return hf

    def test_debug_mla_mtp_round_trip(self):
        spec = load_model_spec("gift_moe/gift_3_5/debugmodel.yaml", {"config": _mtp(2)})
        hf = self._round_trip(spec)
        for probe in (
            "model.embed_tokens.weight",
            "model.norm.gate_up_projection.weight",
            # GDN (fused in_proj_qkvz, under self_attn, no _attn.)
            "model.layers.0.self_attn.in_proj_qkvz.weight",
            "model.layers.0.self_attn.in_proj_ba.weight",
            "model.layers.0.self_attn.conv1d.weight",
            "model.layers.0.self_attn.norm.weight",
            "model.layers.0.self_attn.out_proj.weight",
            "model.layers.0.self_attn.A_log",
            "model.layers.0.self_attn.dt_bias",
            # gated sandwich norms, at layer root
            "model.layers.0.input_layernorm.gate_up_projection.weight",
            "model.layers.0.post_self_attn_layernorm.gate_down_projection.weight",
            "model.layers.0.post_feedforward_layernorm.weight",
            # MLA (DeepSeek-V3 fused names) at a full-attn layer
            "model.layers.2.self_attn.kv_a_proj_with_mqa.weight",
            "model.layers.2.self_attn.kv_b_proj.weight",
            "model.layers.2.self_attn.kv_a_layernorm.gate_up_projection.weight",
            "model.layers.2.self_attn.gate_proj.weight",
            # MoE per-expert + router + shared
            "model.layers.1.mlp.experts.0.gate_proj.weight",
            "model.layers.1.mlp.gate.weight",
            "model.layers.1.mlp.gate.e_score_correction_bias",
            "model.layers.1.mlp.shared_experts.up_proj.weight",
            # MTP nextn head (layer n_main + k)
            "model.layers.6.enorm.gate_up_projection.weight",
            "model.layers.6.hnorm.weight",
            "model.layers.6.eh_proj.weight",
            "model.layers.6.shared_head.norm.gate_up_projection.weight",
            "model.layers.6.embed_tokens.weight",
        ):
            self.assertIn(probe, hf)

    def test_debug_gdn_mtp_round_trip(self):
        """MTP head with GDN attention exercises the GDN path inside the head."""
        spec = load_model_spec("gift_moe/gift_3_5/debugmodel.yaml", {"config": _mtp(2)})
        _give_heads_gdn_attention(spec.model)
        hf = self._round_trip(spec)
        self.assertIn("model.layers.6.self_attn.in_proj_qkvz.weight", hf)

    def test_debug_no_mtp_round_trip(self):
        spec = load_model_spec("gift_moe/gift_3_5/debugmodel.yaml", {"config": _mtp(0)})
        self._round_trip(spec)

    def test_gqa_plus_gdn_round_trip(self):
        """GQA layers instead of MLA: separate q/k/v, the layout HF stores."""
        spec = load_model_spec("gift_moe/gift_3_5/debugmodel.yaml", {"config": _mtp(2)})
        _use_gqa_attention(spec.model, fused=False)
        hf = self._round_trip(spec)
        for probe in (
            "model.layers.2.self_attn.q_proj.weight",
            "model.layers.2.self_attn.k_proj.weight",
            "model.layers.2.self_attn.v_proj.weight",
            "model.layers.2.self_attn.o_proj.weight",
            "model.layers.2.self_attn.gate_proj.weight",
            # the GDN layers still take the fused linear-attention path
            "model.layers.0.self_attn.in_proj_qkvz.weight",
        ):
            self.assertIn(probe, hf)
        # MLA-only keys must be gone from the exported checkpoint.
        self.assertNotIn("model.layers.2.self_attn.kv_a_proj_with_mqa.weight", hf)

    def test_gqa_fused_qkv_round_trip(self):
        """One fused titan tensor <-> three HF ones, per KV group."""
        spec = load_model_spec("gift_moe/gift_3_5/debugmodel.yaml", {"config": _mtp(2)})
        _use_gqa_attention(spec.model, fused=True)
        hf = self._round_trip(spec)
        for name in ("q", "k", "v"):
            self.assertIn(f"model.layers.2.self_attn.{name}_proj.weight", hf)
        # Nothing fused leaks into the HF layout.
        self.assertNotIn("model.layers.2.self_attn.qkv_proj.weight", hf)

    def test_gqa_mtp_head_round_trip(self):
        """A GQA head on a GDN trunk: heads and trunk disagree on attention type."""
        spec = load_model_spec("gift_moe/gift_3_5/debugmodel.yaml", {"config": _mtp(2)})
        _give_heads_gdn_attention(spec.model)
        _use_gqa_attention(spec.model, fused=False, heads_too=False)
        hf = self._round_trip(spec)
        self.assertIn("model.layers.6.self_attn.in_proj_qkvz.weight", hf)
        self.assertIn("model.layers.2.self_attn.q_proj.weight", hf)

    def test_mixing_mla_and_gqa_is_rejected(self):
        """Both want HF ``q_proj`` for a different tensor; one flat map cannot
        say which, so the adapter refuses instead of mapping every layer alike."""
        spec = load_model_spec("gift_moe/gift_3_5/debugmodel.yaml", {"config": _mtp(0)})
        layers = spec.model.layers
        mla_idx = [
            i
            for i, layer in enumerate(layers)
            if isinstance(layer.attention, MLA.Config)
        ]
        self.assertGreaterEqual(len(mla_idx), 2, "need two MLA layers to mix")
        layers[mla_idx[0]].attention = _gqa_config(
            layers[mla_idx[0]].attention, fused=False
        )
        with self.assertRaises(ValueError) as cm:
            GiftHybridStateDictAdapter(spec.model, None)
        self.assertIn("q_proj", str(cm.exception))


class TestGiftHybridGDNFusion(unittest.TestCase):
    """The GDN in_proj_qkvz fuse + conv1d reorder match the expected layout."""

    def setUp(self):
        cfg = load_model_spec(
            "gift_moe/gift_3_5/debugmodel.yaml", {"config": _mtp(0)}
        ).model
        self.adapter = GiftHybridStateDictAdapter(cfg, None)

    def test_in_proj_qkvz_grouping_and_conv_reorder(self):
        # GVA dims: num_v = 2 * num_k, so ratio = 2.
        nk, nv, hk, hv = 2, 4, 3, 2
        ratio = nv // nk
        cfg = types.SimpleNamespace(
            num_k_heads=nk, num_v_heads=nv, head_k_dim=hk, head_v_dim=hv
        )
        key_dim, value_dim = nk * hk, nv * hv  # 6, 8
        conv_dim = 2 * key_dim + value_dim  # 20
        gw = 2 * hk + ratio * hv  # 10 (titan per-group qkv width)
        d, ksz = 1, 4
        tp, hp = "layers.0.attention", "model.layers.0.self_attn."

        qkv = torch.randn(conv_dim, d)
        z = torch.randn(value_dim, d)
        conv = torch.randn(conv_dim, 1, ksz)
        sd = {
            f"{tp}.qkv.qkv_proj.weight": qkv,
            f"{tp}.z_proj.weight": z,
            f"{tp}.qkv.qkv_conv1d.weight": conv,
            f"{tp}.ba_proj.weight": torch.randn(2 * nv, d),
            f"{tp}.o_norm.weight": torch.randn(hv),
            f"{tp}.o_proj.weight": torch.randn(d, value_dim),
            f"{tp}.A_log": torch.randn(nv),
            f"{tp}.dt_bias": torch.randn(nv),
        }
        out: dict = {}
        self.adapter._gdn_to_hf(cfg, tp, hp, sd, out)

        # in_proj_qkvz: per key-head group the layout is [q, k, v, z].
        qkvz = out[f"{hp}in_proj_qkvz.weight"]
        self.assertEqual(qkvz.shape, (2 * key_dim + 2 * value_dim, d))
        grp = qkvz.reshape(nk, 2 * hk + 2 * ratio * hv, d)
        q_g, k_g, v_g, z_g = torch.split(grp, [hk, hk, ratio * hv, ratio * hv], dim=1)
        qkv_g = qkv.reshape(nk, gw, d)
        torch.testing.assert_close(q_g, qkv_g[:, :hk, :])
        torch.testing.assert_close(k_g, qkv_g[:, hk : 2 * hk, :])
        torch.testing.assert_close(v_g, qkv_g[:, 2 * hk :, :])
        torch.testing.assert_close(z_g, z.reshape(nk, ratio * hv, d))

        # conv1d: de-grouped to contiguous [all_q | all_k | all_v].
        conv_hf = out[f"{hp}conv1d.weight"]
        self.assertEqual(conv_hf.shape, (conv_dim, 1, ksz))
        cq, ck, cv = torch.split(conv_hf, [key_dim, key_dim, value_dim], dim=0)
        conv_g = conv.reshape(nk, gw, 1, ksz)
        torch.testing.assert_close(cq.reshape(nk, hk, 1, ksz), conv_g[:, :hk])
        torch.testing.assert_close(
            cv.reshape(nk, ratio * hv, 1, ksz), conv_g[:, 2 * hk :]
        )

        # inverse recovers the titan tensors exactly.
        back: dict = {}
        self.adapter._gdn_from_hf(cfg, hp, tp, out, back)
        for k, v in sd.items():
            torch.testing.assert_close(back[k], v, rtol=0, atol=0)


class TestGiftHybridMaps(unittest.TestCase):
    """Map wiring that the debug model (q_lora=0) does not exercise."""

    def test_low_rank_q_norm_gated_mapping(self):
        # The 3b config uses q_lora_rank>0 -> gated q_a_layernorm; build only the
        # config (not the model) to check the map entry without allocating 3B params.
        spec = load_model_spec("gift_moe/gift_3_5/3b.yaml", {"config": _mtp(1)})
        adapter = GiftHybridStateDictAdapter(spec.model, None)
        self.assertEqual(
            adapter.from_hf_map.get(
                "model.layers.{}.self_attn.q_a_layernorm.gate_up_projection.weight"
            ),
            "layers.{}.attention.q_norm.gate_up_weight",
        )
        # MLA latents stay fused (DeepSeek-V3), no dkv/kr/uk/uv splits.
        self.assertIn(
            "model.layers.{}.self_attn.kv_a_proj_with_mqa.weight", adapter.from_hf_map
        )

    def test_fp8_import_uses_quantized_reader(self):
        cfg = load_model_spec(
            "gift_moe/gift_3_5/debugmodel.yaml", {"config": _mtp(0)}
        ).model
        adapter = GiftHybridStateDictAdapter(cfg, None)
        with tempfile.TemporaryDirectory() as d:
            reader = adapter.get_hf_storage_reader(d, from_quantized=True)
        self.assertEqual(type(reader).__name__, "QuantizedHuggingFaceStorageReader")

    def test_hf_import_rejects_another_norm_parametrization(self):
        cfg = load_model_spec(
            "gift_moe/gift_4/debugmodel.yaml", {"config": _mtp(0)}
        ).model
        adapter = GiftHybridStateDictAdapter(cfg, None)
        config = adapter.hf_config(cfg)
        with tempfile.TemporaryDirectory() as d:
            with open(f"{d}/config.json", "w") as f:
                json.dump(config, f)
            adapter.get_hf_storage_reader(d)
            with open(f"{d}/config.json", "w") as f:
                json.dump({**config, "norm_type": "ZeroCenteredGatedNorm"}, f)
            with self.assertRaisesRegex(
                ValueError, "norm_type='ZeroCenteredGatedNorm'"
            ):
                adapter.get_hf_storage_reader(d)


# NOTE: the online (sharded-DTensor) save/load path is validated separately with a
# distributed run — the GDN fusion gathers its reshape tensors to Replicate and
# reshards like the input, so to_hf/from_hf round-trip bit-exactly under FSDP at
# shard degrees that both do and don't divide num_k_heads.

if __name__ == "__main__":
    unittest.main()
