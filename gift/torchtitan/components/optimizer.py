# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import re
from collections import defaultdict
from collections.abc import Callable, Iterator
from typing import Any, Generic, overload, TypeAlias, TypeVar

import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import Optimizer
from torchtitan.config import Configurable
from torchtitan.config.schema.components.optimizer import (
    OptimizerName,
    ParamGroupConfig,
    OptimizersContainerConfig,
    OptimizersInBackwardContainerConfig,
)
from torchtitan.tools.logging import logger

__all__ = [
    "OptimizersContainer",
    "OptimizersInBackwardContainer",
    "ParamGroupConfig",
]


T = TypeVar("T", bound=Optimizer)


# TODO: Right now this class is biased towards AdamW. We should refactor to
# support mixed optimizers, including Muon.
class OptimizersContainer(Optimizer, Stateful, Configurable, Generic[T]):
    """A container for multiple optimizers.

    This class is used to wrap multiple optimizers into a single object that can be
    used to reduce the complexity of the training loop. This mimics the behavior of
    ``torch.optim.Optimizer``. This class currently only supports ``Adam`` and ``AdamW``.

    **Note**
    Users who want to customize the optimizer behavior can inherit from this class and
    extend the functionality as needed. The following methods must follow the same signature
    as ``torch.optim.Optimizer`` class: ``step()``, ``zero_grad()``, ``state_dict()``,
    ``load_state_dict()``.

    **Limitations**
    This class assumes that all the optimizers are the same type and have the same
    configurations. With this assumption, TorchTitan can support lr scheduler resharding
    (e.g., loading a checkpoint with a different number of GPUs and/or different
    parallelization strategy). Note that ``get_optimizer_state_dict`` already enables the
    resharding for the optimizer state but not for the lr scheduler state, hence the limitation.

    Args:
        model_parts (List[nn.Module]): List of model parts to be optimized.
        optimizer_kwargs (Dict[str, Any]): Keyword arguments for the optimizers.
        name (str): Name of the optimizers.
    """

    Config: TypeAlias = OptimizersContainerConfig

    optimizers: list[T]
    model_parts: list[nn.Module]

    @staticmethod
    def _resolve_optimizer_cls(name: str) -> type:
        return getattr(torch.optim, OptimizerName(name).value)

    @staticmethod
    def _build_optimizer_kwargs(config: Config) -> dict[str, Any]:
        assert config.implementation in [
            "fused",
            "foreach",
            "for-loop",
            "fused_opt_states_bf16",
        ]
        fused = config.implementation in ("fused", "fused_opt_states_bf16")
        return {
            "lr": config.lr,
            "betas": (config.beta1, config.beta2),
            "eps": config.eps,
            "weight_decay": config.weight_decay,
            "fused": fused,
            "foreach": config.implementation == "foreach",
        }

    @staticmethod
    def _build_param_groups(
        model: nn.Module,
        config: Config,
        default_kwargs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Build PyTorch param groups from model parameters and config.

        Each parameter is assigned to the first matching ParamGroupConfig pattern,
        or to the default group if no pattern matches. Returns a list of dicts
        with "params" key and optimizer kwargs, suitable for passing to an optimizer.
        """
        if not config.param_groups:
            params = [p for p in model.parameters() if p.requires_grad]
            return [{"params": params, **default_kwargs}]

        compiled_patterns = [re.compile(pg.pattern) for pg in config.param_groups]

        # group_index -> list of params; None means default group
        grouped_params: dict[int | None, list[nn.Parameter]] = defaultdict(list)

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            matched_index = None
            for i, pat in enumerate(compiled_patterns):
                if pat.search(name):
                    matched_index = i
                    break
            grouped_params[matched_index].append(param)

        # Warn for patterns that matched nothing
        for i, pg in enumerate(config.param_groups):
            if i not in grouped_params:
                logger.warning(
                    f"Optimizer param_groups pattern '{pg.pattern}' "
                    f"matched no parameters"
                )

        result = []
        # Default group first (unmatched params)
        if None in grouped_params:
            result.append({"params": grouped_params[None], **default_kwargs})

        # Then each matched group in pattern order
        for i, pg in enumerate(config.param_groups):
            if i not in grouped_params:
                continue
            group_kwargs = {**default_kwargs}
            group_kwargs["lr"] = default_kwargs["lr"] * pg.lr_multiplier
            group_kwargs["weight_decay"] = (
                default_kwargs["weight_decay"] * pg.weight_decay_multiplier
            )
            if pg.beta1 is not None or pg.beta2 is not None:
                default_beta1, default_beta2 = default_kwargs["betas"]
                group_kwargs["betas"] = (
                    pg.beta1 if pg.beta1 is not None else default_beta1,
                    pg.beta2 if pg.beta2 is not None else default_beta2,
                )
            result.append({"params": grouped_params[i], **group_kwargs})

        return result

    def __init__(self, config: Config, *, model_parts: list[nn.Module]) -> None:
        optimizer_cls = self._resolve_optimizer_cls(config.name)
        optimizer_kwargs = self._build_optimizer_kwargs(config)
        all_params = []
        self.optimizers = []
        self.model_parts = model_parts
        for model in self.model_parts:
            param_groups = self._build_param_groups(model, config, optimizer_kwargs)
            self.optimizers.append(optimizer_cls(param_groups))
            for group in param_groups:
                all_params.extend(group["params"])
        if config.implementation == "fused_opt_states_bf16":
            self._register_bf16_optimizer_state_hook()
        self._validate_length(len(self.model_parts))
        self._post_init(all_params, optimizer_kwargs)

    def __iter__(self) -> Iterator[T]:
        return iter(self.optimizers)

    def __len__(self) -> int:
        return len(self.optimizers)

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        assert closure is None, "OptimizersContainer does not support closures"
        for optimizer in self.optimizers:
            optimizer.step()
        return None

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        func = functools.partial(
            get_optimizer_state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        return {
            k: v
            for sd in map(func, self.model_parts, self.optimizers)
            for k, v in sd.items()
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        func = functools.partial(
            set_optimizer_state_dict,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        list(map(func, self.model_parts, self.optimizers))

    def _validate_length(self, expected_length: int) -> None:
        assert expected_length == len(self.optimizers), (
            "Must pass one optimizer per model part or per param if "
            "using OptimizersInBackwardContainer."
        )

    def _post_init(
        self, all_params: list[nn.Parameter], optimizer_kwargs: dict[str, Any]
    ) -> None:
        # We need to call Optimizer.__init__() to initialize some necessary optimizer
        # functionality such as hooks.
        Optimizer.__init__(self, all_params, optimizer_kwargs)

    def _register_bf16_optimizer_state_hook(self) -> None:
        """Register a step pre-hook to create Adam optimizer states in bfloat16.

        The hook pre-populates optimizer state before Adam's lazy initialization
        runs, so that ``_init_group`` finds non-empty state and skips its own
        fp32 allocation. The fused CUDA kernel then sees the dtype mismatch
        between fp32 params and bf16 states, dispatching to the mixed-precision
        kernel (``FusedAdamMathFunctorMP``).
        """

        def _bf16_state_init_hook(
            optimizer: Optimizer, args: tuple, kwargs: dict
        ) -> None:
            for group in optimizer.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = optimizer.state[p]
                    if len(state) == 0:
                        state["step"] = (
                            torch.zeros((), dtype=torch.float32, device=p.device)
                            if group.get("capturable") or group.get("fused")
                            else torch.tensor(0.0, dtype=torch.float32)
                        )
                        state["exp_avg"] = torch.zeros_like(
                            p, dtype=torch.bfloat16, memory_format=torch.preserve_format
                        )
                        state["exp_avg_sq"] = torch.zeros_like(
                            p, dtype=torch.bfloat16, memory_format=torch.preserve_format
                        )
                        if group.get("amsgrad"):
                            state["max_exp_avg_sq"] = torch.zeros_like(
                                p,
                                dtype=torch.bfloat16,
                                memory_format=torch.preserve_format,
                            )

        for optim in self.optimizers:
            optim.register_step_pre_hook(_bf16_state_init_hook)

    def init_cache_state_dict(self) -> None:
        """Initialize cached state dict for TorchFT. No-op for base class."""
        pass


class OptimizersInBackwardContainer(OptimizersContainer):
    """OptimizersContainer for executing ``optim.step()`` in backward pass.

    This class extend ``OptimizersContainer`` to support optimizer step in
    backward pass. ``step()`` and ``zero_grad()`` are no-op in this class.
    Instead, ``register_post_accumulate_grad_hook`` is used to register a hook to
    execute these methods when the gradient is accumulated.
    """

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = OptimizersInBackwardContainerConfig

    def __init__(self, config: Config, *, model_parts: list[nn.Module]) -> None:
        optimizer_cls = self._resolve_optimizer_cls(config.name)
        optimizer_kwargs = self._build_optimizer_kwargs(config)
        all_params = []
        self.model_parts = model_parts

        # Build a mapping from param -> effective kwargs using param group config
        param_to_kwargs: dict[nn.Parameter, dict[str, Any]] = {}
        for model in self.model_parts:
            param_groups = self._build_param_groups(model, config, optimizer_kwargs)
            for group in param_groups:
                group_kwargs = {k: v for k, v in group.items() if k != "params"}
                for p in group["params"]:
                    param_to_kwargs[p] = group_kwargs

        optim_dict = {}
        for model in self.model_parts:
            for p in model.parameters():
                if p.requires_grad:
                    optim_dict[p] = optimizer_cls([p], **param_to_kwargs[p])
                all_params.append(p)

        def optim_hook(param) -> None:
            optim_dict[param].step()
            optim_dict[param].zero_grad()

        for model in self.model_parts:
            for param in model.parameters():
                if param.requires_grad:
                    param.register_post_accumulate_grad_hook(optim_hook)

        self.optimizers = list(optim_dict.values())

        self._validate_length(
            sum(len(list(model.parameters())) for model in self.model_parts)
        )
        self._post_init(all_params, optimizer_kwargs)

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        return None

    def zero_grad(self, set_to_none: bool = True) -> None:
        pass
