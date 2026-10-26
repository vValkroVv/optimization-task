# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import fields

import torch
import torch.nn as nn

from torchtitan.config.schema.protocols.module import KaimingUniform, Zeros
from torchtitan.config.schema.protocols.sharding import Replicate
from torchtitan.config.configurable import config
from typing import ClassVar, TypeAlias

from torchtitan.config.schema.components.lora import FrozenConfig, LoRAConverterConfig
from torchtitan.models.common.decoder_blocks.sharding import dense_param_placement
from torchtitan.models.common.nn_modules import Linear
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.sharding import ShardingConfig
from torchtitan.tools.logging import logger


def build_frozen(*, config: FrozenConfig, **kwargs) -> nn.Module:
    """Build and freeze the inner module without a wrapper or new parameter FQNs."""
    instance = config.inner.build(**kwargs)
    instance.requires_grad_(False)
    return instance


def _lora_adapter_sharding(
    base_sharding: ShardingConfig | None,
) -> tuple[ShardingConfig | None, ShardingConfig | None]:
    """Derive LoRA adapter sharding from the base linear's TP sharding.

    ``lora_b`` mirrors the base weight's TP shard so its output matches
    ``base_out``'s placement (Shard(-1) for colwise, Partial for rowwise).

    ``lora_a`` weight is Replicate (small rank dim, no benefit from TP
    sharding).
    """
    base_weight_sharding = (
        base_sharding.state_shardings.get("weight") if base_sharding else None
    )
    if base_weight_sharding is None:
        return None, None

    replicate_weight = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=Replicate())},
    )
    # lora_b: same TP shard as base weight → output matches base_out
    lora_b_sharding = ShardingConfig(
        state_shardings={"weight": base_weight_sharding},
    )
    return replicate_weight, lora_b_sharding


_lora_class_cache: dict[type, type] = {}


def _get_lora_cls(parent_cls: type) -> type:
    """Get or create a LoRA subclass for *parent_cls* (e.g. Linear, FP8Linear).

    The returned class has a proper ``Config`` that extends the parent's Config
    with ``rank`` and ``alpha``.  Adapters are built in ``__init__`` from the
    base config's dimensions and sharding.
    """
    if parent_cls in _lora_class_cache:
        return _lora_class_cache[parent_cls]

    parent_config_cls = parent_cls.Config  # pyrefly: ignore [missing-attribute]

    class LoRALinear(parent_cls):  # type: ignore[valid-type, misc]
        @config
        class Config(parent_config_cls):  # type: ignore[misc]
            _lora_applied: ClassVar[bool] = True
            rank: int
            alpha: float

        def __init__(self, config: Config) -> None:
            super().__init__(config)
            for param in nn.Module.parameters(self):
                param.requires_grad_(False)
            self._lora_scaling = config.alpha / config.rank
            lora_a_sharding, lora_b_sharding = _lora_adapter_sharding(
                config.sharding_config
            )
            self.lora_a = Linear.Config(
                in_features=config.in_features,
                out_features=config.rank,
                bias=False,
                sharding_config=lora_a_sharding,
                param_init={"weight": KaimingUniform(a=math.sqrt(5))},
            ).build()
            self.lora_b = Linear.Config(
                in_features=config.rank,
                out_features=config.out_features,
                bias=False,
                sharding_config=lora_b_sharding,
                param_init={"weight": Zeros()},
            ).build()

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            base_out = super().forward(input)
            lora_out = self.lora_b(self.lora_a(input))
            return base_out + self._lora_scaling * lora_out

    LoRALinear.__name__ = f"LoRA{parent_cls.__name__}"
    LoRALinear.__qualname__ = f"LoRA{parent_cls.__name__}"
    _lora_class_cache[parent_cls] = LoRALinear
    return LoRALinear


class LoRAConverter(ModelConfigConverter):
    """Apply LoRA adapters to Linear layers in a model.

    Operates on the model config tree: target Linear configs are replaced
    with ``LoRALinear.Config`` (which builds a LoRA subclass with frozen base
    and trainable adapters). Non-target Linear modules are wrapped with
    ``FrozenConfig``.

    When ``target_modules`` is None (default), every ``Linear.Config`` is
    converted.  When specified, only configs whose FQN's last segment matches
    one of the entries are converted (e.g. ``["wq", "wv"]``).
    """

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = LoRAConverterConfig

    def __init__(self, config: Config, **kwargs):
        if config.rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {config.rank}")
        self.config = config
        self.rank = config.rank
        self.alpha = config.alpha
        self.target_modules = (
            set(config.target_modules) if config.target_modules is not None else None
        )
        if self.target_modules is None:
            logger.info(
                f"LoRA training active with rank={self.rank}, alpha={self.alpha} "
                f"(all Linear layers)"
            )
        else:
            logger.info(
                f"LoRA training active with rank={self.rank}, alpha={self.alpha}, "
                f"target_modules={sorted(self.target_modules)}"
            )

    def _make_lora_config(self, cfg: Linear.Config):
        """Create a LoRALinear.Config from a base Linear.Config."""
        lora_cls = _get_lora_cls(cfg.owner())
        converted = lora_cls.Config(  # pyrefly: ignore [missing-attribute]
            **{f.name: getattr(cfg, f.name) for f in fields(cfg) if f.init},
            rank=self.rank,
            alpha=self.alpha,
        )
        for config_field in fields(cfg):
            if not config_field.init:
                setattr(converted, config_field.name, getattr(cfg, config_field.name))
        converted.copy_metadata_from(cfg)
        return converted

    def convert(self, model_config) -> None:
        """Walk the model config tree for Linear modules.

        Target Linear modules get their config replaced with
        ``LoRALinear.Config``.  Non-target Linear modules are wrapped
        with ``FrozenConfig``.
        """
        locations = list(model_config.traverse(Linear.Config))
        for fqn, cfg, _, _ in locations:
            if getattr(type(cfg), "_lora_applied", False):
                raise ValueError(
                    f"LoRA was already applied to {fqn!r}; repeated conversion is unsupported"
                )
        matched = set()

        # First pass: replace target Linears with LoRA, freeze non-targets
        for fqn, cfg, parent, attr in locations:
            last_segment = fqn.rsplit(".", 1)[-1]
            is_target = (
                self.target_modules is None or last_segment in self.target_modules
            )
            if is_target:
                new_cfg = self._make_lora_config(cfg)
                matched.add(last_segment)
            else:
                new_cfg = FrozenConfig(inner=cfg)

            if isinstance(parent, (list, dict)):
                parent[attr] = new_cfg
            else:
                setattr(parent, attr, new_cfg)

        unmatched = (self.target_modules or set()) - matched
        if unmatched:
            logger.warning(
                f"LoRA target_modules {sorted(unmatched)} did not match any "
                f"Linear.Config in the model config tree."
            )
