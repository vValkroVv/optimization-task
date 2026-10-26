"""State-dict adapter for the GIFT 3.5 hybrid model (GDN linear-attn + MLA MoE).

Converts to/from this model's HuggingFace safetensors layout. That layout is
DeepSeek-V3 for the MLA / MoE / dense / multi-token-prediction blocks, with two
GIFT-3.5 specifics:

  * every sandwich/final norm is a *gated* RMSNorm — ``weight`` +
    ``gate_up_projection.weight`` + ``gate_down_projection.weight`` — and there
    are two extra sandwich norms (``post_self_attn_layernorm``,
    ``post_feedforward_layernorm``) directly under ``model.layers.{L}.``;
  * the GDN (linear-attention) layers use a *fused* input projection
    ``self_attn.in_proj_qkvz`` (q/k/v/z interleaved per key-head group) +
    ``in_proj_ba`` / ``conv1d`` / ``norm`` / ``out_proj`` / ``A_log`` / ``dt_bias``.

The DeepSeek-V3 base adapter (:class:`GiftMoEStateDictAdapter`) handles the MLA /
MoE / dense / MTP keys and FP8 dequant. Works both offline (plain tensors, the
``convert_{to,from}_hf.py`` CLI) and online (sharded ``DTensor``s): the GDN fusion
reshapes a few projection weights, which DTensor cannot do across an unevenly-
sharded dim, so those tensors are gathered to ``Replicate`` for the surgery and
resharded like their input (see :meth:`_gather` / :meth:`_reshard_like`).
"""

from typing import Any

import torch

from torchtitan.models.common.attention import GatedDeltaAttention
from torchtitan.models.gift_moe._base.model import GiftMoEModel
from torchtitan.models.gift_moe._base.state_dict_adapter import GiftMoEStateDictAdapter
from torchtitan.tools.logging import logger

__all__ = ["GiftHybridStateDictAdapter"]

# GDN params that map 1:1 (HF suffix under self_attn. -> titan suffix under attention.).
_GDN_DIRECT = {
    "in_proj_ba.weight": "ba_proj.weight",
    "norm.weight": "o_norm.weight",
    "out_proj.weight": "o_proj.weight",
    "A_log": "A_log",
    "dt_bias": "dt_bias",
}

# Keys the fusion reads on top of the 1:1 ``_GDN_DIRECT`` pairs, per GDN module.
_GDN_FUSED_TITAN = ("qkv.qkv_proj.weight", "z_proj.weight", "qkv.qkv_conv1d.weight")
_GDN_FUSED_HF = ("in_proj_qkvz.weight", "conv1d.weight")


class GiftHybridStateDictAdapter(GiftMoEStateDictAdapter):
    """GIFT 3.5 hybrid HF state-dict converter (offline + online): adds gated norms
    and the fused-``in_proj`` GDN layers on top of the DeepSeek-V3 base."""

    def __init__(
        self,
        model_config: GiftMoEModel.Config,
        hf_assets_path: str | None,
    ):
        super().__init__(model_config, hf_assets_path)
        self._extend_gated_norm_maps()
        if model_config.mtp is not None:
            # Re-derive the MTP block maps so its (now gated) sandwich norms are
            # picked up from the extended from_hf_map, then add the head norms.
            self._build_mtp_maps()
            self._extend_mtp_gated_norm_maps()

    # ---- gated-norm map extensions ------------------------------------------

    @staticmethod
    def _add_gate_projections(m: dict, hf_prefix: str, titan_prefix: str) -> None:
        m[f"{hf_prefix}.gate_up_projection.weight"] = f"{titan_prefix}.gate_up_weight"
        m[f"{hf_prefix}.gate_down_projection.weight"] = (
            f"{titan_prefix}.gate_down_weight"
        )

    def _extend_gated_norm_maps(self) -> None:
        m = self.from_hf_map
        # Norms the base already maps as a single ``.weight`` -> add the gate MLP.
        gated = {
            "model.layers.{}.input_layernorm": "layers.{}.attention_norm",
            "model.layers.{}.post_attention_layernorm": "layers.{}.ffn_norm",
            "model.layers.{}.self_attn.kv_a_layernorm": "layers.{}.attention.kv_norm",
            "model.norm": "norm",
        }
        if "model.layers.{}.self_attn.q_a_layernorm.weight" in m:
            gated["model.layers.{}.self_attn.q_a_layernorm"] = (
                "layers.{}.attention.q_norm"
            )
        for hf_p, tt_p in gated.items():
            self._add_gate_projections(m, hf_p, tt_p)
        # Sandwich norms GIFT 3.5 adds that DeepSeek-V3 has no slot for.
        for hf_p, tt_p in {
            "model.layers.{}.post_self_attn_layernorm": "layers.{}.post_attn_norm",
            "model.layers.{}.post_feedforward_layernorm": "layers.{}.post_ffn_norm",
        }.items():
            m[f"{hf_p}.weight"] = f"{tt_p}.weight"
            self._add_gate_projections(m, hf_p, tt_p)

    def _extend_mtp_gated_norm_maps(self) -> None:
        # The base maps the nextn head norms (enorm/hnorm/shared_head.norm) as a
        # single ``.weight``; add their gate MLP (block sandwich norms are already
        # covered by the re-derivation from the extended from_hf_map).
        for hf_p, tt_p in {
            "model.layers.{}.enorm": "mtp.layers.{}.token_norm",
            "model.layers.{}.hnorm": "mtp.layers.{}.hidden_norm",
            "model.layers.{}.shared_head.norm": "mtp.layers.{}.final_norm",
        }.items():
            self._add_gate_projections(self.mtp_from_hf_map, hf_p, tt_p)
        self.mtp_to_hf_map = {v: k for k, v in self.mtp_from_hf_map.items()}

    # ---- GDN layers (fused in_proj_qkvz; conv1d grouped<->contiguous) --------

    # Named by type, not by "everything that is not MLA": a GQA layer in a hybrid
    # has none of these tensors and must fall through to the base key maps.

    def _gdn_specs(self) -> list[tuple[str, str, Any]]:
        """The linear-attention modules, main layers and MTP heads alike."""
        return [
            (tp, hp, cfg)
            for tp, hp, cfg in self.attention_modules()
            if isinstance(cfg, GatedDeltaAttention.Config)
        ]

    @staticmethod
    def _gdn_dims(cfg) -> tuple[int, int, int, int, int]:
        nk, nv = cfg.num_k_heads, cfg.num_v_heads
        hk, hv = cfg.head_k_dim, cfg.head_v_dim
        return nk, nv, hk, hv, nv // nk

    def _gdn_to_hf(self, cfg, tp: str, hp: str, sd: dict, out: dict) -> None:
        nk, nv, hk, hv, ratio = self._gdn_dims(cfg)
        key_dim, value_dim = nk * hk, nv * hv
        gw = 2 * hk + ratio * hv  # titan per-k-head-group width (q,k,v)

        # in_proj_qkvz: titan's grouped [q,k,v] proj + grouped z, spliced per group
        # into HF's [q, k, v, z]. fla lays qkv_proj out grouped by key-head and the
        # HF layout reshapes in_proj_qkvz to (num_k_heads, 2*hk + 2*ratio*hv).
        qkv_ref = sd[f"{tp}.qkv.qkv_proj.weight"]
        qkv = self._gather(qkv_ref)
        d = qkv.shape[1]
        q, k, v = torch.split(qkv.reshape(nk, gw, d), [hk, hk, ratio * hv], dim=1)
        z = self._gather(sd[f"{tp}.z_proj.weight"]).reshape(nk, ratio * hv, d)
        qkvz = torch.cat([q, k, v, z], dim=1).reshape(nk * (gw + ratio * hv), d)
        out[f"{hp}in_proj_qkvz.weight"] = self._reshard_like(qkvz, qkv_ref).contiguous()

        # conv1d: titan applies the conv to the grouped qkv stream; the HF layout
        # applies it to the de-grouped contiguous [all_q | all_k | all_v]. Depthwise
        # conv => reorder the channel (dim-0) axis grouped -> contiguous.
        conv_ref = sd[f"{tp}.qkv.qkv_conv1d.weight"]
        conv = self._gather(conv_ref)
        ksz = conv.shape[-1]
        cq, ck, cv = torch.split(
            conv.reshape(nk, gw, 1, ksz), [hk, hk, ratio * hv], dim=1
        )
        conv_hf = torch.cat(
            [
                cq.reshape(key_dim, 1, ksz),
                ck.reshape(key_dim, 1, ksz),
                cv.reshape(value_dim, 1, ksz),
            ],
            dim=0,
        )
        out[f"{hp}conv1d.weight"] = self._reshard_like(conv_hf, conv_ref).contiguous()

        for hf_s, tt_s in _GDN_DIRECT.items():
            out[f"{hp}{hf_s}"] = sd[f"{tp}.{tt_s}"]

    def _gdn_from_hf(self, cfg, hp: str, tp: str, hf: dict, out: dict) -> None:
        nk, nv, hk, hv, ratio = self._gdn_dims(cfg)
        key_dim, value_dim = nk * hk, nv * hv
        conv_dim = 2 * key_dim + value_dim
        gwz = 2 * hk + 2 * ratio * hv  # HF per-group width (q,k,v,z)

        qkvz_ref = hf[f"{hp}in_proj_qkvz.weight"]
        qkvz = self._gather(qkvz_ref)
        d = qkvz.shape[1]
        q, k, v, z = torch.split(
            qkvz.reshape(nk, gwz, d), [hk, hk, ratio * hv, ratio * hv], dim=1
        )
        qkv = torch.cat([q, k, v], dim=1).reshape(conv_dim, d)
        out[f"{tp}.qkv.qkv_proj.weight"] = self._reshard_like(
            qkv, qkvz_ref
        ).contiguous()
        z = z.reshape(value_dim, d)
        out[f"{tp}.z_proj.weight"] = self._reshard_like(z, qkvz_ref).contiguous()

        conv_ref = hf[f"{hp}conv1d.weight"]
        conv = self._gather(conv_ref)
        ksz = conv.shape[-1]
        cq, ck, cv = torch.split(conv, [key_dim, key_dim, value_dim], dim=0)
        conv_g = torch.cat(
            [
                cq.reshape(nk, hk, 1, ksz),
                ck.reshape(nk, hk, 1, ksz),
                cv.reshape(nk, ratio * hv, 1, ksz),
            ],
            dim=1,
        ).reshape(conv_dim, 1, ksz)
        out[f"{tp}.qkv.qkv_conv1d.weight"] = self._reshard_like(
            conv_g, conv_ref
        ).contiguous()

        for hf_s, tt_s in _GDN_DIRECT.items():
            out[f"{tp}.{tt_s}"] = hf[f"{hp}{hf_s}"]

    # ---- partial state dicts ------------------------------------------------

    @staticmethod
    def _gdn_titan_keys(tp: str) -> list[str]:
        return [f"{tp}.{s}" for s in (*_GDN_FUSED_TITAN, *_GDN_DIRECT.values())]

    @staticmethod
    def _gdn_hf_keys(hp: str) -> list[str]:
        return [f"{hp}{s}" for s in (*_GDN_FUSED_HF, *_GDN_DIRECT)]

    @staticmethod
    def _warn_unconsumed(gdn_sd: dict[str, Any], consumed: set[str]) -> None:
        leftover = sorted(set(gdn_sd) - consumed)
        if leftover:
            logger.warning(
                f"{len(leftover)} GDN key(s) matched a linear-attention prefix but "
                f"were not converted (unknown suffix?): {leftover[:5]}"
            )

    # ---- public API ---------------------------------------------------------

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        specs = self._gdn_specs()
        gdn_prefixes = tuple(f"{tp}." for tp, _, _ in specs)
        gdn_sd = {k: v for k, v in state_dict.items() if k.startswith(gdn_prefixes)}
        rest = {k: v for k, v in state_dict.items() if k not in gdn_sd}

        hf_state_dict = super().to_hf(rest)
        consumed: set[str] = set()
        for tp, hp, cfg in specs:
            keys = self._gdn_titan_keys(tp)
            if not self._module_present(keys, gdn_sd, tp):
                continue
            self._gdn_to_hf(cfg, tp, hp, gdn_sd, hf_state_dict)
            consumed.update(keys)
        self._warn_unconsumed(gdn_sd, consumed)
        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        specs = self._gdn_specs()
        gdn_prefixes = tuple(hp for _, hp, _ in specs)  # model.layers.{i}.self_attn.
        gdn_hf = {k: v for k, v in hf_state_dict.items() if k.startswith(gdn_prefixes)}
        rest = {k: v for k, v in hf_state_dict.items() if k not in gdn_hf}

        state_dict = super().from_hf(rest)
        consumed: set[str] = set()
        for tp, hp, cfg in specs:
            keys = self._gdn_hf_keys(hp)
            if not self._module_present(keys, gdn_hf, hp):
                continue
            self._gdn_from_hf(cfg, hp, tp, gdn_hf, state_dict)
            consumed.update(keys)
        self._warn_unconsumed(gdn_hf, consumed)
        return state_dict
