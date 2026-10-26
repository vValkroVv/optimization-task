"""The decoder Config: the module tree the YAML describes, field for field.

A model preset writes the decoder's own fields. Repeated layers are named once
under ``blocks`` and placed by ``layers``::

    blocks:
      dense: {attention: {...}, feed_forward: {...}, ...}
      moe:   {attention: {...}, moe: {...}, ...}
    layers:
      - {block: dense}
      - {block: moe, repeat: 2}
      - {cycle: [dense, moe], repeat: 3}

``materialize()`` places an independent copy of each named block at every
position, hands the decoder's ``dim`` and the position to it, applies the
init policy and checks the result. Nothing here is model-family specific.
"""

import copy
from dataclasses import field, fields
from typing import ClassVar, TypeVar, overload

from torchtitan.config.schema.models.common.regularization import validate_z_loss_coef
from torchtitan.config.schema.distributed.parallelism import sequence_length_divisor
from torchtitan.config.configurable import (
    config,
    Derived,
    independent_config_positions,
    type_name_of,
)
from torchtitan.config.provenance import SourceMap
from torchtitan.config.schema.components.lora import FrozenConfig, unwrap_frozen
from torchtitan.config.schema.models.common.attention.base import BaseAttentionConfig
from torchtitan.config.schema.models.common.attention.const import (
    AttentionMaskKind,
    MaskType,
)
from torchtitan.config.schema.models.common.decoder_blocks.layer_group import (
    LayerGroupConfig,
    check_block_names,
    count_layers,
    place_blocks,
    placed_configs,
    reject_positions,
)
from torchtitan.config.schema.models.common.decoder_blocks.mtp import (
    MTPBlockConfig,
    MTPConfig,
)
from torchtitan.config.schema.models.common.decoder_blocks.transformer_block import (
    ResidualTransformerBlockConfig,
    TransformerBlockConfig,
)
from torchtitan.config.schema.models.common.embeddings.rope.rope import RoPEConfig
from torchtitan.config.schema.models.common.embeddings.token_embeddings import (
    EmbeddingConfig,
)
from torchtitan.config.schema.models.common.init_policy import (
    InitPolicy,
    tied_embedding_init,
)
from torchtitan.config.schema.models.common.moe.dispatch.hybridep import (
    HybridEPTokenDispatcherConfig,
)
from torchtitan.config.schema.models.common.nn_modules import LinearConfig
from torchtitan.config.schema.models.common.norms.types import NormConfig
from torchtitan.config.schema.models.common.regularization import HiddenZLoss
from torchtitan.config.schema.models.common.validation import (
    resolve_child,
    validate_linear_shapes,
    validate_values,
)
from torchtitan.config.schema.protocols.model import (
    BaseModelConfig,
    ModelConfigConverterConfig,
)
from torchtitan.config.schema.protocols.module import iter_module_configs, ModuleConfig


_Block = TypeVar("_Block", bound=TransformerBlockConfig)


@config
class DecoderConfig(BaseModelConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.models.common.decoder_blocks.decoder:Decoder"
    )
    runtime_manages_sharding: ClassVar[bool] = True
    block_types: ClassVar[tuple[type, ...]] = (ResidualTransformerBlockConfig,)
    """Block classes this model's runtime builds (exact types)."""
    attention_types: ClassVar[tuple[type, ...] | None] = None
    """Attention schemas implemented by this runtime's sharding preparation."""
    supports_mtp_model_parallelism: ClassVar[bool] = False
    """The runtime supplies MTP plans for tensor, pipeline and expert parallelism."""

    dim: int
    vocab_size: int
    # One RoPE for the whole stack; every attention fills or checks what it
    # depends on against it.
    rope: RoPEConfig
    norm: NormConfig
    """The final norm; also the MTP heads' norms."""
    init: InitPolicy
    blocks: dict[str, TransformerBlockConfig] = field(default_factory=dict)
    """Named block configs that ``layers`` and ``mtp`` place."""
    # Converters may wrap placed layers and the head in ``FrozenConfig``; YAML
    # writes neither.
    layers: list[LayerGroupConfig | TransformerBlockConfig] = field(
        default_factory=list
    )
    tok_embeddings: Derived[EmbeddingConfig] = None
    lm_head: Derived[LinearConfig] = None
    mtp: MTPConfig | None = None
    hidden_z_loss_patterns: dict[str, float | HiddenZLoss] = field(default_factory=dict)
    """FQN glob -> coefficient or input/output loss spec, including MTP heads."""
    converters: list[ModelConfigConverterConfig] = field(default_factory=list)

    # -- the written config ------------------------------------------------

    def tree_fields(self) -> tuple:
        # Validate all written modules, including inline layers. After placement
        # the named blocks are bookkeeping rather than runtime modules.
        # Converters are never modules; their templates are not live configs.
        skip = {"blocks", "converters"} if self.materialized else {"converters"}
        return tuple(f for f in fields(self) if f.name not in skip)

    def __post_init__(self) -> None:
        for pattern, spec in self.hidden_z_loss_patterns.items():
            coef = spec.coef if isinstance(spec, HiddenZLoss) else spec
            suffix = ".coef" if isinstance(spec, HiddenZLoss) else ""
            validate_z_loss_coef(coef, f"hidden_z_loss_patterns.{pattern}{suffix}")

        if self.dim <= 0:
            raise ValueError(f"dim must be positive, got {self.dim}")
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if not self.layers:
            raise ValueError("a decoder needs at least one entry in `layers`")
        if not self.materialized:
            reject_positions(self.blocks)
        check_block_names(self.layers, self.blocks)
        if self.mtp is not None:
            check_block_names(self.mtp.layers, self.blocks, path="mtp.layers")
        if self.materialized:
            self._validate_tree()

    @property
    def n_layers(self) -> int:
        return count_layers(self.layers)

    @property
    def num_mtp(self) -> int:
        if self.mtp is None:
            return 0
        return count_layers(self.mtp.layers)

    @overload
    def placed_layers(self) -> tuple[ResidualTransformerBlockConfig, ...]: ...

    @overload
    def placed_layers(self, kind: type[_Block]) -> tuple[_Block, ...]: ...

    def placed_layers(
        self, kind: type[TransformerBlockConfig] = ResidualTransformerBlockConfig
    ) -> tuple[TransformerBlockConfig, ...]:
        return placed_configs(self.layers, kind, path="layers")

    # -- materialize ---------------------------------------------------------

    def _place(
        self, block: TransformerBlockConfig, layer_id: int, *, where: str | None = None
    ) -> TransformerBlockConfig:
        if isinstance(block, FrozenConfig):
            return block
        return resolve_child(
            block,
            type(block),
            name=where or f"layers.{layer_id}",
            dims=dict(dim=self.dim, layer_id=layer_id),
        )

    def materialize(self) -> "DecoderConfig":
        """The module tree to build: every position its own complete config."""
        from torchtitan.config.schema.models.common.init_policy import apply_init
        from torchtitan.config.schema.models.common.regularization import (
            set_hidden_z_loss,
        )

        if self.materialized:
            self.validate_materialized()
            return independent_config_positions(self)
        dim, policy = self.dim, self.init
        layers = [
            self._place(block, i)
            for i, block in enumerate(place_blocks(self.layers, self.blocks))
        ]
        for i, block in enumerate(layers):
            apply_init(block, policy, layer_id=i, dim=dim)

        norm = resolve_child(
            copy.deepcopy(self.norm),
            type(self.norm),
            name="norm",
            dims=dict(normalized_shape=dim),
        )
        mtp = None
        if self.mtp is not None:
            heads = []
            for head, entry in enumerate(place_blocks(self.mtp.layers, self.blocks)):
                layer_id = len(layers) + head
                if isinstance(entry, MTPBlockConfig):
                    head_cfg = entry
                    block = entry.block
                else:
                    head_cfg = MTPBlockConfig(block=entry)
                    block = entry
                where = f"mtp.layers.{head}"
                head_cfg = resolve_child(
                    head_cfg, MTPBlockConfig, name=where, dims=dict(dim=dim)
                )

                def head_norm(given, name):
                    if given is None:
                        return copy.deepcopy(norm)
                    return resolve_child(
                        given,
                        type(given),
                        name=f"{where}.{name}",
                        dims=dict(normalized_shape=dim),
                    )

                head_cfg = head_cfg._copy(
                    block=self._place(
                        block, layer_id, where=f"mtp.layers.{head}.block"
                    ),
                    input_proj=resolve_child(
                        head_cfg.input_proj,
                        LinearConfig,
                        name=f"{where}.input_proj",
                        dims=dict(in_features=2 * dim, out_features=dim),
                    ),
                    token_norm=head_norm(head_cfg.token_norm, "token_norm"),
                    hidden_norm=head_norm(head_cfg.hidden_norm, "hidden_norm"),
                    final_norm=head_norm(head_cfg.final_norm, "final_norm"),
                )
                apply_init(head_cfg, policy, layer_id=layer_id, dim=dim)
                heads.append(head_cfg)
            mtp = self.mtp._copy(layers=heads)

        tree = independent_config_positions(self)
        tree._set_metadata("materialized", True)
        tree.layers = [*layers]
        tree.mtp = mtp
        tree.norm = norm
        tree.tok_embeddings = self._embedding()
        tree.lm_head = self._head()
        tree._complete(policy)
        tree._fill_shared_mask_type()
        # The stack-level modules; the layers are already done (idempotent).
        apply_init(tree, policy, layer_id=0, dim=dim)
        tree._fill_rope_contract()
        if self.hidden_z_loss_patterns:
            set_hidden_z_loss(tree, self.hidden_z_loss_patterns)
        tree.validate_materialized()
        return tree

    def validate_materialized(self) -> None:
        BaseModelConfig.validate_materialized(self)
        self._validate_tree()

    def _complete(self, policy: InitPolicy) -> None:
        """Decoder-variant completion of the placed tree (none by default)."""

    def _derived_child(self, child, factory, *, name, sources, extra=None):
        """Complete *child* from the decoder's fields, recording which YAML
        field each filled width came from."""
        inherited = {
            target: self._source_fields.get(
                source, f"{self._source_path}.{source}".lstrip(".")
            )
            for target, source in sources.items()
            if child is None or getattr(child, target) is None
        }
        result = resolve_child(
            copy.deepcopy(child),
            factory,
            name=name,
            dims={target: getattr(self, source) for target, source in sources.items()},
            extra=extra,
        )
        result._source_fields = SourceMap({**result._source_fields, **inherited})
        result._source_origins = self._source_origins
        return result

    def _embedding(self) -> EmbeddingConfig:
        return self._derived_child(
            self.tok_embeddings,
            EmbeddingConfig,
            name="tok_embeddings",
            sources={"num_embeddings": "vocab_size", "embedding_dim": "dim"},
        )

    def _head(self) -> LinearConfig:
        if isinstance(self.lm_head, FrozenConfig):
            return placed_configs(
                [copy.deepcopy(self.lm_head)], LinearConfig, path="lm_head"
            )[0]
        return self._derived_child(
            self.lm_head,
            LinearConfig,
            name="lm_head",
            sources={"in_features": "dim", "out_features": "vocab_size"},
            extra={"init_role": "lm_head"},
        )

    def _fill_shared_mask_type(self) -> None:
        """Attention without a mask type of its own (linear attention) takes the
        one the stack's other attention layers share: one masks object serves
        every layer."""
        attentions = [
            child
            for _, child in iter_module_configs(self)
            if isinstance(child, BaseAttentionConfig)
        ]
        known = {a.mask_type for a in attentions if a.mask_type is not None}
        if len(known) > 1:
            _ = self.attention  # raises: the layers mix masking
        shared = known.pop() if known else MaskType.CAUSAL
        for attention in attentions:
            if attention.mask_type is None:
                attention.mask_type = shared

    def _fill_rope_contract(self) -> None:
        for _, child in iter_module_configs(self):
            if isinstance(child, BaseAttentionConfig):
                child.accept_rope(self.rope)

    # -- the materialized tree ----------------------------------------------

    def layer_summary(self) -> list[dict[str, str | int]]:
        """Report the selected components on each materialized decoder layer."""
        tree = self if self.materialized else self.materialize()
        result: list[dict[str, str | int]] = []
        for index, wrapped in enumerate(tree.layers):
            layer = wrapped.inner if isinstance(wrapped, FrozenConfig) else wrapped
            row: dict[str, str | int] = {
                "index": index,
                "type": type_name_of(type(layer)) or type(layer).__name__,
            }
            for f in layer.tree_fields():
                child = getattr(layer, f.name)
                if isinstance(child, FrozenConfig):
                    child = child.inner
                if isinstance(child, ModuleConfig):
                    row[f.name] = type_name_of(type(child)) or type(child).__name__
            result.append(row)
        return result

    @property
    def document_packing(self) -> bool:
        # Linear attention takes its mask type when placed.
        tree = self if self.materialized else self.materialize()
        return tree.attention.attention_mask_kind == AttentionMaskKind.VARLEN

    @property
    def attention(self) -> BaseAttentionConfig:
        """The attention configuration shared by the decoder's masking object."""
        configs = [
            child
            for _, child in iter_module_configs(self)
            if isinstance(child, BaseAttentionConfig)
        ]
        if not configs:
            raise ValueError(f"{type(self).__qualname__} has no attention config")
        kinds = {(child.attention_mask_kind, child.mask_type) for child in configs}
        if len(kinds) > 1:
            raise ValueError(
                f"{type(self).__qualname__} mixes attention masking across "
                f"layers ({sorted(map(str, kinds))}); the decoder builds one "
                "attention_masks object for all of them."
            )
        return configs[0]

    def sequence_length_divisor(self, parallelism) -> int:
        return sequence_length_divisor(
            tp=parallelism.tensor_parallel_degree,
            cp=parallelism.context_parallel_degree,
        )

    def validate_parallelism(self, parallelism):
        """Restrictions declared by the modules in the tree, for every model."""
        from torchtitan.config.schema.models.common.attention.softmax.gqa import (
            SoftmaxAttentionConfig,
        )
        from torchtitan.config.schema.models.common.attention.softmax.kernels.base import (
            InnerAttentionConfig,
        )
        from torchtitan.config.validation import Issue

        issues = BaseModelConfig.validate_parallelism(self, parallelism)
        modules = [module for _, module in iter_module_configs(self)]

        def name(module) -> str:
            return type_name_of(type(module)) or type(module).__name__

        def report(path: str, what: str, offenders: set[str]) -> None:
            if offenders:
                issues.append(Issue(path=path, message=f"{what} {sorted(offenders)}."))

        if parallelism.tensor_parallel_degree > 1:
            report(
                "parallelism.tensor_parallel_degree",
                "Tensor Parallel does not support",
                {
                    name(m)
                    for m in modules
                    if not getattr(type(m), "supports_tensor_parallel", True)
                },
            )
        if parallelism.context_parallel_degree > 1:
            unsupported = {
                name(m)
                for m in modules
                if not isinstance(m, InnerAttentionConfig)
                and not getattr(type(m), "supports_context_parallel", True)
            }
            for m in modules:
                if isinstance(m, InnerAttentionConfig):
                    if not m.supports_context_parallel:
                        unsupported.add(name(m))
                elif isinstance(m, SoftmaxAttentionConfig):
                    allowed = type(m).context_parallel_kernels
                    if allowed is not None and not isinstance(
                        unwrap_frozen(m.inner_attention), allowed
                    ):
                        unsupported.add(f"{name(m)} with {name(m.inner_attention)}")
            report(
                "parallelism.context_parallel_degree",
                "Context Parallel does not support",
                unsupported,
            )
        if self.num_mtp:
            if not self.supports_mtp_model_parallelism:
                for degree_field in (
                    "tensor_parallel_degree",
                    "pipeline_parallel_degree",
                    "expert_parallel_degree",
                ):
                    if getattr(parallelism, degree_field) > 1:
                        issues.append(
                            Issue(
                                path=f"parallelism.{degree_field}",
                                message=(
                                    f"{type_name_of(type(self))} has no MTP model-parallel "
                                    "sharding contract; use degree 1 for this runtime."
                                ),
                            )
                        )
            # The MTP embedding shift runs along the (sharded) sequence dim.
            if (
                parallelism.enable_sequence_parallel
                and parallelism.tensor_parallel_degree > 1
            ):
                issues.append(
                    Issue(
                        path="parallelism.enable_sequence_parallel",
                        message=(
                            "MTP is incompatible with sequence parallelism: the MTP "
                            "embedding shift runs along the (sharded) sequence dim. Use "
                            "tensor parallelism with "
                            "parallelism.enable_sequence_parallel=false."
                        ),
                    )
                )
            if parallelism.context_parallel_degree > 1:
                issues.append(
                    Issue(
                        path="parallelism.context_parallel_degree",
                        message=(
                            "MTP is incompatible with context parallelism: the MTP "
                            "embedding shift runs along the (sharded) sequence dim."
                        ),
                    )
                )
        return issues

    def _validate_token_dispatch(self) -> None:
        fqns = [fqn for fqn, *_ in self.traverse(HybridEPTokenDispatcherConfig)]
        if not fqns or self.attention.attention_mask_kind != AttentionMaskKind.VARLEN:
            return
        raise ValueError(
            f"{type(self).__qualname__} pairs HybridEP token dispatch with "
            "varlen attention, which this tree has never been run: the "
            "combination is unvalidated, not known-broken. Use the deepep or "
            "all-to-all dispatcher for varlen, or a non-varlen attention "
            f"kernel with HybridEP. HybridEP dispatchers at: {', '.join(fqns)}."
        )

    def validate_hf_export(self, *, step: float | None = None):
        """What the runtime key/shape round-trip (``check_hf_export``) cannot see.

        HF builds an affine RMSNorm per norm slot; requiring one here also gives
        every norm a weight, so an extra norm fails that round-trip. HF keeps one
        attention, dense-FFN and MoE layout for all layers, and has no residual
        scale. ``step``: the 0-based schedule step of the exported weights,
        ``None`` for any step.
        """
        from torchtitan.config.schema.models.common.norms.base import BaseNormConfig
        from torchtitan.config.schema.models.common.norms.rmsnorm import (
            RMSNormConfig,
        )
        from torchtitan.config.validation import Issue

        issues = [
            Issue(
                path=f"model.{path}",
                message="HF export needs an affine RMSNorm here, got "
                f"{type(module).__name__}"
                f"{'' if getattr(module, 'elementwise_affine', True) else ' without affine'}",
            )
            for path, module in iter_module_configs(self)
            if isinstance(module, BaseNormConfig)
            and path.startswith(("norm", "layers.", "mtp."))
            and not (type(module) is RMSNormConfig and module.elementwise_affine)
        ]
        issues += self._residual_scale_hf_issues(step)
        # Config widths, not module types: LoRA/FP8/Frozen wrappers keep them.
        widths: dict[str, list[tuple[str, dict]]] = {
            "attention": [],
            "dense FFN": [],
            "MoE": [],
        }
        for i, layer in enumerate(self.placed_layers(TransformerBlockConfig)):
            layer = unwrap_frozen(layer)
            attn = unwrap_frozen(layer.attention)
            heads = getattr(attn, "n_heads", None)
            fields_ = {
                "n_heads": heads,
                "n_kv_heads": getattr(attn, "n_kv_heads", None) or heads,
                "head_dim": getattr(attn, "head_dim", None)
                or (self.dim // heads if heads else None),
            }
            for name in (
                "q_lora_rank",
                "kv_lora_rank",
                "qk_nope_head_dim",
                "qk_rope_head_dim",
                "v_head_dim",
            ):
                fields_[name] = getattr(attn, name, None)
            widths["attention"].append((f"layers.{i}.attention", fields_))
            if layer.feed_forward is not None:
                ffn = unwrap_frozen(layer.feed_forward)
                widths["dense FFN"].append(
                    (f"layers.{i}.feed_forward", {"hidden_dim": ffn.hidden_dim})
                )
            if layer.moe is not None:
                moe = unwrap_frozen(layer.moe)
                shared = unwrap_frozen(moe.shared_experts)
                widths["MoE"].append(
                    (
                        f"layers.{i}.moe",
                        {
                            "num_experts": moe.experts.num_experts,
                            "hidden_dim": moe.experts.hidden_dim,
                            "shared_hidden_dim": getattr(shared, "hidden_dim", None),
                            **{
                                name: getattr(moe.router, name)
                                for name in (
                                    "top_k",
                                    "score_func",
                                    "route_norm",
                                    "route_scale",
                                    "num_expert_groups",
                                    "num_limited_groups",
                                )
                            },
                        },
                    )
                )
        for label, entries in widths.items():
            for path, value in entries[1:]:
                first_path, first = entries[0]
                if diff := [k for k in value if value[k] != first[k]]:
                    issues.append(
                        Issue(
                            path=f"model.{path}",
                            message=f"HF export needs one {label} layout for all "
                            "layers; differs from "
                            f"{first_path} in "
                            + ", ".join(f"{k}={value[k]} vs {first[k]}" for k in diff),
                        )
                    )
        return issues

    def _residual_scale_hf_issues(self, step: float | None):
        """HF has no residual scale: it computes as ``alpha == 1``."""
        from torchtitan.config.validation import Issue

        unsaturated = [
            (f"{where}.{name}", scale)
            for where, block in self._placed_blocks()
            for name in ("attn_residual_scale", "ffn_residual_scale")
            if (scale := getattr(unwrap_frozen(block), name, None)) is not None
            and scale.start_value != 1.0
            and (step is None or step < scale.warmup_steps)
        ]
        if not unsaturated:
            return []
        path, scale = unsaturated[0]
        at = "any HF save" if step is None else f"the HF save at schedule step {step:g}"
        return [
            Issue(
                path=f"model.{path}",
                message=f"{len(unsaturated)} residual scale(s) warm up from "
                f"{scale.start_value} for {scale.warmup_steps} steps (here), but HF "
                f"has none and computes as alpha == 1: {at} may be mid-warmup.",
            )
        ]

    def _placed_blocks(self) -> list[tuple[str, TransformerBlockConfig]]:
        placed: list[tuple[str, TransformerBlockConfig]] = [
            (f"layers.{i}", b)
            for i, b in enumerate(self.placed_layers(TransformerBlockConfig))
        ]
        if self.mtp is not None:
            placed += [
                (f"mtp.layers.{i}.block", head.block)
                for i, head in enumerate(self.mtp.placed_layers())
            ]
        return placed

    def _validate_block_contract(self) -> None:
        """The blocks this model's runtime builds."""
        model = type(self).__qualname__
        placed = self._placed_blocks()
        allowed = type(self).block_types
        for where, block in placed:
            inner = unwrap_frozen(block)
            if not isinstance(inner, TransformerBlockConfig):
                raise ValueError(f"{where}: expected a TransformerBlockConfig")
            attention_types = type(self).attention_types
            if attention_types is not None and not isinstance(
                unwrap_frozen(inner.attention), attention_types
            ):
                raise ValueError(
                    f"{where}.attention: {model} runtime supports "
                    f"{', '.join(t.__name__ for t in attention_types)}, got "
                    f"{type(unwrap_frozen(inner.attention)).__name__}"
                )
            if type(inner) not in allowed:
                names = ", ".join(type_name_of(t) or t.__name__ for t in allowed)
                raise ValueError(
                    f"{where}: {model} builds {names} blocks, not "
                    f"{type_name_of(type(inner)) or type(inner).__name__}"
                )

    def _validate_tree(self) -> None:
        owner = type(self).__qualname__
        if not self.layers:
            raise ValueError("a materialized decoder needs at least one layer")
        # Every materialized tree has both (materialize sets them, the shared
        # Derived check enforces it); the assert narrows the types.
        assert self.tok_embeddings is not None and self.lm_head is not None
        validate_values(
            owner,
            (
                "tok_embeddings.embedding_dim",
                self.tok_embeddings.embedding_dim,
                self.dim,
            ),
            (
                "tok_embeddings.num_embeddings",
                self.tok_embeddings.num_embeddings,
                self.vocab_size,
            ),
        )
        for i, layer in enumerate(self.layers):
            if isinstance(layer, LayerGroupConfig):
                raise ValueError(f"layers.{i}: a materialized tree has no layer groups")
        self._validate_block_contract()
        validate_linear_shapes(
            owner, (self.lm_head, "lm_head", self.dim, self.vocab_size)
        )
        validate_values(owner, ("norm", self.norm.normalized_shape, self.dim))
        # Checks only: materialize() and the runtime constructor fill the rope
        # contract.
        _ = self.attention
        self._validate_token_dispatch()


@config
class TiedDecoderConfig(DecoderConfig):
    """Decoder whose builder can tie the embedding to the LM head."""

    enable_weight_tying: bool = False

    def _complete(self, policy: InitPolicy) -> None:
        if self.enable_weight_tying:
            if self.tok_embeddings is None:
                raise ValueError("weight tying needs completed token embeddings")
            self.tok_embeddings.param_init = tied_embedding_init(policy)

    def validate_parallelism(self, parallelism):
        from torchtitan.config.validation import Issue

        issues = DecoderConfig.validate_parallelism(self, parallelism)
        if self.enable_weight_tying and parallelism.pipeline_parallel_degree > 1:
            issues.append(
                Issue(
                    path="parallelism.pipeline_parallel_degree",
                    message="Weight tying is not supported with Pipeline Parallel.",
                )
            )
        return issues
