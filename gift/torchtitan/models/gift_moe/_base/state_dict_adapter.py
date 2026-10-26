import json
import os
import re
from typing import Any

import torch
from torch.distributed.checkpoint import HuggingFaceStorageReader

from torch.distributed.tensor import DTensor, Replicate

from torchtitan.models.common.attention import FusedQKVLinear, GQAttention, MLA
from torchtitan.models.utils import MoEStateDictAdapter
from torchtitan.tools.logging import logger

from .hf_config import build_hf_config, default_gigachat_remote_code_dir
from .model import GiftMoEModel


class GiftMoEStateDictAdapter(MoEStateDictAdapter[GiftMoEModel.Config]):
    """
    StateDictAdapter for GiftMoE model.
    """

    @staticmethod
    def hf_config(
        model_config: GiftMoEModel.Config,
    ) -> dict[str, Any]:
        """The ``config.json`` for what ``to_hf`` writes.

        ``deepseek_v3`` for a plain MLA/MoE/RMSNorm stack, ``gigachat3_5`` for
        anything DeepSeek-V3 cannot express (GDN layers, gated attention,
        gated or sandwich norms, MLA latent scaling, logit soft-cap).
        """
        return build_hf_config(model_config)

    @staticmethod
    def default_hf_remote_code_dir() -> str | None:
        return default_gigachat_remote_code_dir()

    def __init__(
        self,
        model_config: GiftMoEModel.Config,
        hf_assets_path: str | None,
    ):
        super().__init__(model_config, hf_assets_path)
        # MTP heads are stored in HF DeepSeek-V3 format as extra layers at
        # indices [n_main_layers, n_main_layers + num_mtp); see _build_mtp_maps.
        self.layer_configs = model_config.placed_layers()
        self.mtp_layer_configs = (
            model_config.mtp.placed_layers() if model_config.mtp is not None else ()
        )
        self.n_main_layers = len(self.layer_configs)
        self.from_hf_map = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            # MLP Module
            "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
            # Transformer Layer
            "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            # MoE Module
            "model.layers.{}.mlp.experts.{}.gate_proj.weight": "layers.{}.moe.experts.w1_EFD",
            "model.layers.{}.mlp.experts.{}.up_proj.weight": "layers.{}.moe.experts.w3_EFD",
            "model.layers.{}.mlp.experts.{}.down_proj.weight": "layers.{}.moe.experts.w2_EDF",
            "model.layers.{}.mlp.gate.weight": "layers.{}.moe.router.gate.weight",
            "model.layers.{}.mlp.shared_experts.gate_proj.weight": "layers.{}.moe.shared_experts.w1.weight",
            "model.layers.{}.mlp.shared_experts.up_proj.weight": "layers.{}.moe.shared_experts.w3.weight",
            "model.layers.{}.mlp.shared_experts.down_proj.weight": "layers.{}.moe.shared_experts.w2.weight",
            "model.layers.{}.mlp.gate.e_score_correction_bias": "layers.{}.moe.expert_bias_E",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "lm_head.weight",
        }

        # Attention keys come from the attention configs the model actually uses,
        # not from layers[0]: a hybrid mixes types layer by layer.
        self.from_hf_map.update(self._attention_key_map())

        self.mtp_from_hf_map: dict[str, str] = {}
        self.mtp_to_hf_map: dict[str, str] = {}
        if model_config.mtp is not None:
            self._build_mtp_maps()

    # HF DeepSeek-V3 per-MTP-layer keys that are clones of shared tensors
    # (the main embedding / lm_head); we don't store separate copies.
    _MTP_SKIP_SUFFIXES = ("embed_tokens.weight", "shared_head.head.weight")

    # ---- attention modules, whatever type each layer uses -------------------

    def attention_modules(self) -> list[tuple[str, str, Any]]:
        """``(titan attention prefix, HF self_attn prefix, config)`` per module.

        Main layers and MTP blocks alike; the head's HF layer index is offset by
        ``n_main_layers``, matching the nextn indexing. Subclasses filter this by
        attention type instead of walking the config themselves.
        """
        mods = [
            (f"layers.{i}.attention", f"model.layers.{i}.self_attn.", layer.attention)
            for i, layer in enumerate(self.layer_configs)
        ]
        if self.model_config.mtp is not None:
            mods += [
                (
                    f"mtp.layers.{k}.block.attention",
                    f"model.layers.{self.n_main_layers + k}.self_attn.",
                    module.block.attention,
                )
                for k, module in enumerate(self.mtp_layer_configs)
            ]
        return mods

    @staticmethod
    def attention_key_pairs(cfg: Any) -> dict[str, str]:
        """HF -> titan templates for one attention config's own tensors.

        Fused Q/K/V is deliberately absent: HF always stores q/k/v apart, so those
        need a split rather than a rename (see ``_fused_qkv_specs``). Linear
        attention (GDN / KDA) is fused too and handled by the hybrid subclass.
        """
        hf, tt = "model.layers.{}.self_attn.", "layers.{}.attention."
        if isinstance(cfg, MLA.Config):
            pairs = {
                f"{hf}kv_a_proj_with_mqa.weight": f"{tt}wkv_a.weight",
                f"{hf}kv_a_layernorm.weight": f"{tt}kv_norm.weight",
                f"{hf}kv_b_proj.weight": f"{tt}wkv_b.weight",
                f"{hf}o_proj.weight": f"{tt}wo.weight",
            }
            if cfg.q_lora_rank != 0:
                pairs.update(
                    {
                        f"{hf}q_a_proj.weight": f"{tt}wq_a.weight",
                        f"{hf}q_a_layernorm.weight": f"{tt}q_norm.weight",
                        f"{hf}q_b_proj.weight": f"{tt}wq_b.weight",
                    }
                )
            else:
                pairs[f"{hf}q_proj.weight"] = f"{tt}wq.weight"
        elif isinstance(cfg, GQAttention.Config):
            pairs = {f"{hf}o_proj.weight": f"{tt}wo.weight"}
            if not isinstance(cfg.qkv_linear, FusedQKVLinear.Config):
                pairs.update(
                    {
                        f"{hf}q_proj.weight": f"{tt}qkv_linear.wq.weight",
                        f"{hf}k_proj.weight": f"{tt}qkv_linear.wk.weight",
                        f"{hf}v_proj.weight": f"{tt}qkv_linear.wv.weight",
                    }
                )
            if cfg.qk_norm is not None:
                # One config, two modules: q and k normalize independently.
                pairs[f"{hf}q_norm.weight"] = f"{tt}q_norm.weight"
                pairs[f"{hf}k_norm.weight"] = f"{tt}k_norm.weight"
        else:
            return {}
        if getattr(cfg, "gate", None) is not None:
            pairs[f"{hf}gate_proj.weight"] = f"{tt}gate.gate_proj.weight"
        return pairs

    def _attention_key_map(self) -> dict[str, str]:
        """Union of ``attention_key_pairs`` over the types this model uses.

        The maps are ``{}``-templated over the layer index, so one entry serves
        every layer of a type and a hybrid needs the union. Two types wanting the
        same key for different tensors cannot be expressed that way -- rejected
        here rather than mapping every layer the last-seen way, since ``to_hf``
        inverts this map and a dict inversion would drop the clash silently.
        """
        merged: dict[str, str] = {}
        titan_owner: dict[str, str] = {}
        for _, _, cfg in self.attention_modules():
            for hf_key, titan_key in self.attention_key_pairs(cfg).items():
                clash = (
                    ("HF", hf_key, merged[hf_key], titan_key)
                    if merged.get(hf_key, titan_key) != titan_key
                    else ("titan", titan_key, titan_owner[titan_key], hf_key)
                    if titan_owner.get(titan_key, hf_key) != hf_key
                    else None
                )
                if clash is not None:
                    side, key, first, second = clash
                    raise ValueError(
                        f"{type(self).__qualname__} cannot map this model: the "
                        f"{side} key {key} is needed as both {first} and {second}. "
                        "Mixing these attention types in one checkpoint layout is "
                        "not supported."
                    )
                merged[hf_key] = titan_key
                titan_owner[titan_key] = hf_key
        return merged

    # ---- fused Q/K/V: one titan tensor, three HF ones -----------------------

    def _fused_qkv_specs(self) -> list[tuple[str, str, Any]]:
        """Attention modules whose Q/K/V is a single fused titan tensor."""
        return [
            (tp, hp, cfg)
            for tp, hp, cfg in self.attention_modules()
            if isinstance(cfg, GQAttention.Config)
            and isinstance(cfg.qkv_linear, FusedQKVLinear.Config)
        ]

    @staticmethod
    def _fused_qkv_heads(cfg: Any) -> tuple[int, int, int]:
        qkv = cfg.qkv_linear
        return qkv.n_heads, qkv.n_kv_heads, qkv.head_dim

    def _fused_qkv_to_hf(self, cfg: Any, hp: str, fused: Any) -> dict[str, Any]:
        n_heads, n_kv_heads, head_dim = self._fused_qkv_heads(cfg)
        wq, wk, wv = self.fused_to_separate_qkv(
            self._gather(fused), n_heads, n_kv_heads, head_dim
        )
        return {
            f"{hp}{name}_proj.weight": self._reshard_like(w, fused).contiguous()
            for name, w in (("q", wq), ("k", wk), ("v", wv))
        }

    def _fused_qkv_from_hf(self, cfg: Any, hp: str, hf: dict[str, Any]) -> Any:
        ref = hf[f"{hp}q_proj.weight"]
        wq, wk, wv = (self._gather(hf[f"{hp}{n}_proj.weight"]) for n in "qkv")
        n_heads, n_kv_heads, head_dim = self._fused_qkv_heads(cfg)
        fused = self.separate_to_fused_qkv(wq, wk, wv, n_heads, n_kv_heads, head_dim)
        return self._reshard_like(fused, ref).contiguous()

    # A fusion reshapes weights along their sharded (dim-0) axis, which DTensor
    # rejects unless the shard divides the target dim. Gather to a full
    # (Replicate) tensor for the surgery, then reshard the result like its input.
    # Offline (plain-tensor) inputs pass through untouched.
    @staticmethod
    def _gather(t):
        if isinstance(t, DTensor):
            return t.redistribute(t.device_mesh, [Replicate()] * t.device_mesh.ndim)
        return t

    @staticmethod
    def _reshard_like(out, ref):
        if isinstance(ref, DTensor):
            return out.redistribute(ref.device_mesh, ref.placements)
        return out

    @staticmethod
    def _module_present(keys: list[str], sd: dict[str, Any], name: str) -> bool:
        """Whether *sd* carries this module, tolerating a partial state dict.

        The specs come from the full model config, so a pipeline stage legitimately
        holds none of a module's keys; holding only some of them raises.
        """
        missing = [k for k in keys if k not in sd]
        if not missing:
            return True
        if len(missing) == len(keys):
            return False
        raise ValueError(
            f"Incomplete module '{name}' in the state dict: missing {missing}. "
            "A module must be either fully present or fully absent."
        )

    def _build_mtp_maps(self) -> None:
        """Build HF<->titan key maps for the MTP heads.

        HF stores each MTP head as a normal decoder layer at index
        ``n_main_layers + i`` plus a few MTP-specific tensors. The decoder-layer
        sublayers reuse the main map's titan suffix under ``mtp.layers.{}.block.``;
        the MTP-specific tensors map to the head's own norms / projection. The
        ``{}`` in both templates is the *MTP-local* index ``i`` (the HF<->local
        layer-index offset of ``n_main_layers`` is applied separately).
        """
        for hf_tmpl, titan_tmpl in self.from_hf_map.items():
            if not hf_tmpl.startswith("model.layers.{}."):
                continue  # embeddings / final norm / lm_head: shared, skip
            hf_suffix = hf_tmpl[len("model.layers.{}.") :]
            titan_suffix = titan_tmpl[len("layers.{}.") :]
            self.mtp_from_hf_map["model.layers.{}." + hf_suffix] = (
                "mtp.layers.{}.block." + titan_suffix
            )
        self.mtp_from_hf_map.update(
            {
                "model.layers.{}.enorm.weight": "mtp.layers.{}.token_norm.weight",
                "model.layers.{}.hnorm.weight": "mtp.layers.{}.hidden_norm.weight",
                "model.layers.{}.eh_proj.weight": "mtp.layers.{}.input_proj.weight",
                "model.layers.{}.shared_head.norm.weight": "mtp.layers.{}.final_norm.weight",
            }
        )
        self.mtp_to_hf_map = {v: k for k, v in self.mtp_from_hf_map.items()}

    def _check_mtp_ffn_kind(self, local_layer: int, titan_key: str) -> None:
        """Reject a key whose FFN kind this head does not have.

        Heads are dense or MoE (the ``mtp_block`` template) and the key maps hold both,
        so a checkpoint of the other kind would otherwise leave the head at its
        init, or read ``num_experts`` off ``None``.
        """
        mtp = self.model_config.mtp
        assert mtp is not None  # only reached for keys _is_mtp_hf_key accepted
        block = self.mtp_layer_configs[local_layer].block
        if ".block.moe." in titan_key and block.moe is None:
            carries = "a MoE"
        elif ".block.feed_forward." in titan_key and block.feed_forward is None:
            carries = "a dense FFN"
        else:
            return
        has = "a MoE" if block.moe is not None else "a dense FFN"
        raise ValueError(
            f"the checkpoint carries {carries} for MTP head {local_layer}, which "
            f"has {has}. Build the model with a matching ``mtp_block``."
        )

    @staticmethod
    def _swap_eh_proj_halves(weight: torch.Tensor) -> torch.Tensor:
        """Swap the two input-feature halves of the MTP input projection.

        torchtitan concatenates ``[hidden, embedding]`` while HF
        DeepSeek-V3 ``eh_proj`` uses ``[embedding, hidden]``. The weight is
        ``[dim, 2*dim]``; swapping the two column halves converts either way.
        """
        half = weight.shape[1] // 2
        return torch.cat([weight[:, half:], weight[:, :half]], dim=1)

    def _hf_layer_index(self, key: str) -> int | None:
        m = re.match(r"model\.layers\.(\d+)\.", key)
        return int(m.group(1)) if m is not None else None

    def _is_mtp_hf_key(self, key: str) -> bool:
        if self.model_config.mtp is None:
            return False
        idx = self._hf_layer_index(key)
        return idx is not None and idx >= self.n_main_layers

    def get_hf_storage_reader(
        self, path: str, from_quantized: bool = False
    ) -> HuggingFaceStorageReader:
        """
        Override default get_hf_storage_reader function to return QuantizedHFStorageReader.
        """
        self._check_norm_parametrization(path)
        if from_quantized:
            from torch.distributed.checkpoint.quantized_hf_storage import (
                QuantizedHuggingFaceStorageReader,
            )

            # NOTE: Now we use Quantized HF storage reader to read GiftMoE 671B model.
            # If loading checkpoints without quantization, use HuggingFaceStorageReader instead
            BLOCK_SIZE = 128
            return QuantizedHuggingFaceStorageReader(
                path=path,
                target_dtype=torch.float32,
                block_size=BLOCK_SIZE,
                thread_count=4,
            )
        else:
            return HuggingFaceStorageReader(path)

    def _check_norm_parametrization(self, path: str) -> None:
        config_path = os.path.join(path, "config.json")
        if not os.path.isfile(config_path):
            logger.warning(
                f"{config_path} not found: the norm parametrization of the "
                "checkpoint is not verified against the model."
            )
            return
        with open(config_path) as f:
            stored = json.load(f)
        expected = build_hf_config(self.model_config)
        for key in ("norm_type", "linear_gating_type"):
            if key in stored and key in expected and stored[key] != expected[key]:
                raise ValueError(
                    f"{config_path} has {key}={stored[key]!r} but the model expects "
                    f"{expected[key]!r}: its norm weights would load silently with "
                    "a different meaning."
                )

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """
        1. Convert between the HF shape and the torchtitan shape.
        2. Split the GroupedExperts' weight into separate expert's weight.
        """
        to_hf_map = {v: k for k, v in self.from_hf_map.items()}

        hf_state_dict = {}
        mtp_items = {}

        # Split fused Q/K/V into the three tensors HF stores; the rest is renames.
        fused_specs = self._fused_qkv_specs()
        fused_keys = {f"{tp}.qkv_linear.wqkv.weight" for tp, _, _ in fused_specs}
        for tp, hp, cfg in fused_specs:
            key = f"{tp}.qkv_linear.wqkv.weight"
            if key in state_dict:
                hf_state_dict.update(self._fused_qkv_to_hf(cfg, hp, state_dict[key]))

        for key, value in state_dict.items():
            if key in fused_keys:
                continue
            if key.startswith("mtp."):
                # MTP heads are converted separately (distinct layer indexing,
                # eh_proj half-swap); see _to_hf_mtp.
                mtp_items[key] = value
                continue
            if "moe.experts" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                # pyrefly: ignore [missing-attribute]
                layer_num = re.search(r"\d+", key).group(0)
                hf_state_dict.update(
                    self._grouped_experts_to_hf(
                        value,
                        abstract_key,
                        to_hf_map,
                        layer_num,
                        self._num_routed_experts(),
                    )
                )

            elif "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                # pyrefly: ignore [missing-attribute]
                layer_num = re.search(r"\d+", key).group(0)
                new_key = to_hf_map.get(abstract_key)
                if new_key is not None:
                    hf_state_dict[new_key.format(layer_num)] = value

            elif key in to_hf_map:
                hf_state_dict[to_hf_map[key]] = value

        if mtp_items:
            hf_state_dict.update(self._to_hf_mtp(mtp_items))
        if self.model_config.mtp is not None:
            hf_state_dict.update(self._to_hf_mtp_shared_clones(state_dict))

        return hf_state_dict

    def _to_hf_mtp_shared_clones(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Per-MTP-layer clones of the shared embedding / output head.

        HF DeepSeek-V3 stores an ``embed_tokens.weight`` and a
        ``shared_head.head.weight`` inside every MTP decoder layer (clones of the
        main ``model.embed_tokens`` / ``lm_head``). ``from_hf`` drops them on load
        (see ``_MTP_SKIP_SUFFIXES``), but an exported checkpoint must include them
        to be loadable by transformers.

        Emits only the clones whose source tensor this state dict holds. Under PP
        that is a partial contribution — the embedding and the output head sit on
        different stages — and the ranks together cover both.
        """
        clones: dict[str, Any] = {}
        embed = state_dict.get("tok_embeddings.weight")
        head = state_dict.get("lm_head.weight")
        # Tied models keep a single tensor; fall back to it for the head clone.
        # Only when actually tied — otherwise a stage that owns the embedding but
        # not lm_head would write the embedding under the head's key.
        if head is None and getattr(self.model_config, "enable_weight_tying", False):
            head = embed
        for local_layer in range(len(self.mtp_layer_configs)):
            hf_layer = self.n_main_layers + local_layer
            if embed is not None:
                clones[f"model.layers.{hf_layer}.embed_tokens.weight"] = embed
            if head is not None:
                clones[f"model.layers.{hf_layer}.shared_head.head.weight"] = head
        return clones

    def _to_hf_mtp(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert MTP-head torchtitan weights to HF DeepSeek-V3 keys.

        Experts are split, the HF layer index is offset by ``n_main_layers``,
        and ``input_proj`` halves are swapped for ``eh_proj``. The shared
        embedding / lm_head clones HF expects per MTP layer are not emitted.
        """
        hf_state_dict = {}
        for key, value in state_dict.items():
            local_layer = int(re.search(r"\d+", key).group(0))
            hf_layer = local_layer + self.n_main_layers
            if "block.moe.experts" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                moe = self.mtp_layer_configs[local_layer].block.moe
                if moe is None:
                    raise ValueError(f"MTP head {local_layer} has no MoE experts")
                num_experts = moe.num_experts
                hf_state_dict.update(
                    self._grouped_experts_to_hf(
                        value,
                        abstract_key,
                        self.mtp_to_hf_map,
                        str(hf_layer),
                        num_experts,
                    )
                )
            else:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                new_key = self.mtp_to_hf_map[abstract_key].format(hf_layer)
                if key.endswith("input_proj.weight"):
                    value = self._swap_eh_proj_halves(value)
                hf_state_dict[new_key] = value
        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """
        1. When loading from HF checkpoint, dequantize the weights from float8 to float32.
        2. Convert between the HF shape and the torchtitan shape.
        3. Concat separate expert's weight into GroupedExperts' weight.
        """

        state_dict = {}
        expert_weights_by_layer = {}  # {layer: {abstract_key: {expert_id: tensor}}}
        mtp_items = {}

        # Fuse the three HF tensors back into one where the model wants them fused.
        fused_keys: set[str] = set()
        for tp, hp, cfg in self._fused_qkv_specs():
            trio = [f"{hp}{n}_proj.weight" for n in "qkv"]
            if not self._module_present(trio, hf_state_dict, hp):
                continue
            state_dict[f"{tp}.qkv_linear.wqkv.weight"] = self._fused_qkv_from_hf(
                cfg, hp, hf_state_dict
            )
            fused_keys.update(trio)

        for key, value in hf_state_dict.items():
            if key in fused_keys:
                continue
            if self._is_mtp_hf_key(key):
                mtp_items[key] = value
                continue
            if "mlp.experts" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=2)
                layer_num, expert_num = re.findall(r"\d+", key)
                titan_abstract_key = self.from_hf_map[abstract_key]
                converted = self._grouped_experts_from_hf(
                    expert_weights_by_layer,
                    titan_abstract_key,
                    layer_num,
                    expert_num,
                    value,
                    self._num_routed_experts(),
                )
                if converted is not None:
                    new_key, new_value = converted
                    state_dict[new_key] = new_value

            elif "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                # pyrefly: ignore [missing-attribute]
                layer_num = re.search(r"\d+", key).group(0)
                new_key = self.from_hf_map[abstract_key]
                new_key = new_key.format(layer_num)
                state_dict[new_key] = value

            else:
                new_key = self.from_hf_map[key]
                state_dict[new_key] = value

        if mtp_items:
            state_dict.update(self._from_hf_mtp(mtp_items))

        self._validate_grouped_experts_from_hf(expert_weights_by_layer)
        return state_dict

    def _from_hf_mtp(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert HF DeepSeek-V3 MTP-layer weights to torchtitan MTP keys.

        The HF layer index is shifted back by ``n_main_layers``, experts are
        concatenated, ``eh_proj`` halves are swapped, and the cloned
        ``embed_tokens`` / ``shared_head.head`` are dropped.
        """
        state_dict = {}
        expert_weights_by_layer = {}
        for key, value in hf_state_dict.items():
            hf_layer = self._hf_layer_index(key)
            local_layer = hf_layer - self.n_main_layers
            suffix = key.split(f"model.layers.{hf_layer}.", 1)[1]
            if suffix in self._MTP_SKIP_SUFFIXES:
                continue

            if "mlp.experts" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=2)
                expert_num = re.findall(r"\d+", key)[1]
                titan_abstract_key = self.mtp_from_hf_map[abstract_key]
                self._check_mtp_ffn_kind(local_layer, titan_abstract_key)
                moe = self.mtp_layer_configs[local_layer].block.moe
                if moe is None:
                    raise ValueError(f"MTP head {local_layer} has no MoE experts")
                num_experts = moe.num_experts
                converted = self._grouped_experts_from_hf(
                    expert_weights_by_layer,
                    titan_abstract_key,
                    str(local_layer),
                    expert_num,
                    value,
                    num_experts,
                )
                if converted is not None:
                    new_key, new_value = converted
                    state_dict[new_key] = new_value
            else:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                titan_abstract_key = self.mtp_from_hf_map[abstract_key]
                self._check_mtp_ffn_kind(local_layer, titan_abstract_key)
                new_key = titan_abstract_key.format(local_layer)
                if suffix == "eh_proj.weight":
                    value = self._swap_eh_proj_halves(value)
                state_dict[new_key] = value

        self._validate_grouped_experts_from_hf(expert_weights_by_layer, context="MTP")
        return state_dict
