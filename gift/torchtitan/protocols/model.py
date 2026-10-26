# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from collections.abc import Sequence
from typing import cast, Protocol, TypeVar

import torch.nn as nn

from torchtitan.config import Configurable
from torchtitan.config.configs import ParallelismConfig
from torchtitan.config.schema.protocols.model import BaseModelConfig
from typing import TypeAlias

from torchtitan.config.schema.protocols.model import ModelConfigConverterConfig

from .module import Module

_ModelConfig = TypeVar("_ModelConfig", bound=BaseModelConfig)


class ModelRuntimeConfig(Protocol):
    """The configuration needed for runtime model preparation."""

    parallelism: ParallelismConfig


def materialize_runtime_model_config(
    model_config: _ModelConfig, *, apply_converters: bool = True
) -> BaseModelConfig:
    """Materialize a model config and convert its independent module tree.

    Materialized inputs have already selected their converter policy, including
    checkpoint export's opt-out; constructors must not apply converters again.
    """
    if model_config.materialized:
        # The tree may have been edited in place since it was built
        # (``prepare_config``, converters); check it where the runtime takes it.
        model_config.validate_materialized()
        return model_config
    from torchtitan.config.build import materialize_model_config

    tree = materialize_model_config(model_config)
    if apply_converters:
        converters = cast(
            Sequence[ModelConfigConverterConfig],
            getattr(model_config, "converters", ()),
        )
        apply_model_converters(tree, converters)
        tree.validate_materialized()
    return tree


def apply_model_converters(
    tree: BaseModelConfig, converters: Sequence[ModelConfigConverterConfig]
) -> None:
    """Apply the validated public converter pipeline to its independent tree."""
    from torchtitan.config.errors import ConfigSchemaError

    for index, converter in enumerate(converters):
        try:
            converter.build().convert(tree)
        except ValueError as exc:
            raise ConfigSchemaError(f"model.config.converters[{index}]: {exc}") from exc


class ModelConfigConverter(Configurable):
    """Base class for converters that transform the model config tree.

    Subclasses implement ``convert()`` to modify configs before model build
    (e.g. quantization, LoRA).
    """

    Config: TypeAlias = ModelConfigConverterConfig

    @abstractmethod
    def convert(self, model_config) -> None:
        raise NotImplementedError


class BaseModel(Module):
    """Base class for all model classes.

    Models inherit from BaseModel (which is Module = nn.Module + Configurable).
    Each model exposes its module-tree schema through Config.
    The model is constructed via ``config.build()``.

    ``init_states`` (from Module) auto-recurses; override only for custom
    ordering (e.g., weight tying before init).
    """

    @classmethod
    def prepare_config(
        cls, model_config: BaseModelConfig, *, config: ModelRuntimeConfig, **kwargs
    ) -> None:
        """Prepare a materialized config for this run before building modules."""
        raise NotImplementedError(f"{cls.__qualname__}.prepare_config")

    @classmethod
    def get_nparams_and_flops(
        cls, model_config: BaseModelConfig, model: nn.Module, seq_len: int
    ) -> tuple[int, int]:
        """Return parameter count and training FLOPs per token."""
        raise NotImplementedError(f"{cls.__qualname__}.get_nparams_and_flops")

    def init_weights(self, **kwargs) -> None:
        """Backward-compatible alias for ``init_states``.

        External tools (e.g., AutoParallel) wrap ``init_weights`` with
        DTensor-aware interception. This alias ensures they can find it.
        """
        # TODO: remove this once autoparallel has wrap_init_states
        buffer_device = kwargs.get("buffer_device")
        self.init_states(buffer_device=buffer_device)

    def verify_module_protocol(self) -> None:
        """Verify all submodules satisfy the ``Module`` protocol.

        Catches non-``Module`` submodules early with a clear error message,
        preventing obscure failures when the ``Module`` protocol is being
        used later.

        Override in models where some internal ``nn.Module`` submodules
        cannot conform to the ``Module`` protocol.
        """
        failures: list[tuple[str, str]] = []
        for fqn, mod in self.named_modules():
            if not isinstance(mod, Module):
                failures.append((fqn, type(mod).__name__))
        if failures:
            details = ", ".join(f"'{fqn}' ({cls})" for fqn, cls in failures)
            raise RuntimeError(
                f"The following modules do not satisfy the Module protocol: {details}"
            )

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = BaseModelConfig
