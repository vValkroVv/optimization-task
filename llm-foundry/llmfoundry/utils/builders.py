# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os
import logging
import torch
import re
import functools
from collections import OrderedDict
from typing import Any, Dict, Iterable, Optional, Union
from composer import algorithms
from composer.callbacks import (
    EarlyStopper,
    LRMonitor,
    MemoryMonitor,
    OptimizerMonitor,
    RuntimeEstimator,
    SpeedMonitor,
    ActivationMonitor,
    TrainingMetricsMonitor,
    EMA,
    OnlineProfiler,
    SystemMetricsMonitor,
    WeightMonitor,
)
from composer.core import Evaluator
from composer.datasets.in_context_learning_evaluation import get_icl_task_dataloader
from composer.loggers import MLFlowLogger, TensorboardLogger, WandBLogger
from composer.optim import DecoupledAdamW, DecoupledFusedAdamW, Adan, Muon
from composer.optim.scheduler import (
    ConstantWithWarmupScheduler,
    CosineAnnealingWithWarmupScheduler,
    LinearWithWarmupScheduler,
    InvSquareRootWithWarmupAndCooldownScheduler,
    ConstantMultiStepWithGammaWithWarmupScheduler,
    LinearScheduler,
    ConstantScheduler,
)
from composer.utils import dist, global_actlog_state
from torch.optim.optimizer import Optimizer
from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as om
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from llmfoundry.callbacks import (
    FDiffMetrics,
    Generate,
    GlobalLRScaling,
    LayerFreezing,
    MonolithicCheckpointSaver,
    ScheduledGarbageCollector,
    ExpertBalancingMonitor,
    DSAMonitor,
    DSAKLMonitor,
    TokenDispatcherMonitor,
    DynamicBlockLRRescaler,
    DynamicBlockBiasRateRescaler,
    JobCallbackDispatcher,
    JobCallbackDispatcherWEB,
    StageTimeProfiler,
    Autometrics,
    DivergenceDetector,
    ProcessorFallbackCounter,
    DataProfilerCallback,
    OptimizerHealthMonitor,
    UpdateWeightRatioMonitor,
    RouterMarginMonitor,
    TopkAccuracyMonitor,
    MtpAccLenMonitor,
    MtpLossStage,
    NsysCaptureCallback,
    MoeRanksTiming,
    ProResWarmup,
)
from llmfoundry.callbacks.memory_stats_callback import MemoryStatsSaver
from llmfoundry.callbacks import EvalGauntlet
from llmfoundry.callbacks import ParallelismConsistencyCheck
from llmfoundry.utils.config_utils import to_dict_container
from llmfoundry.data.datasets.mmlu import DEFAULT_CHOICE_NUM2SYMBOL, get_mmlu_dataloader
from llmfoundry.data.datasets.omni_mcqa_dataset import get_omni_mcqa_dataloader
from llmfoundry.optim import DecoupledAdaLRLion, DecoupledClipLion, DecoupledLionW
from llmfoundry.scheduler import PerGroupComposerScheduler, StackedScheduler


log = logging.getLogger(__name__)

# Registry for :func:`build_scheduler` and per-param-group scheduler overrides.
SCHEDULER_CLASS_REGISTRY = {
    "constant_with_warmup": ConstantWithWarmupScheduler,
    "cosine_with_warmup": CosineAnnealingWithWarmupScheduler,
    "linear_decay_with_warmup": LinearWithWarmupScheduler,
    "linear": LinearScheduler,
    "inv_square_root_with_warmup_and_cooldown": InvSquareRootWithWarmupAndCooldownScheduler,
    "constant_multi_step_with_gamma_with_warmup": ConstantMultiStepWithGammaWithWarmupScheduler,
    "constant": ConstantScheduler,
    "stacked": StackedScheduler,
}


def _resolve_scheduler_param_group_index(
    entry: dict[str, Any],
    optimizer_param_str_matches: Optional[list[str]],
) -> int:
    """Resolve optimizer ``param_groups`` index (0 = default remainder group).

    Mutates ``entry`` by removing ``param_str_match`` or ``param_group_index``.
    """
    if "param_group_index" in entry:
        return int(entry.pop("param_group_index"))
    if "param_str_match" in entry:
        match = str(entry.pop("param_str_match"))
        if not optimizer_param_str_matches:
            raise ValueError(
                "Scheduler `param_groups` uses `param_str_match`, but the optimizer "
                "has no `param_groups`. Add optimizer `param_groups` or use "
                "`param_group_index` instead.",
            )
        for i, p in enumerate(optimizer_param_str_matches):
            if p == match:
                return i + 1
        raise ValueError(
            f"No optimizer param group with param_str_match={match!r}. "
            f"Known optimizer `param_groups` patterns: {optimizer_param_str_matches}",
        )
    raise ValueError(
        "Each scheduler `param_groups` entry must contain `param_str_match` or "
        "`param_group_index`.",
    )


def build_callback(name: str, kwargs: Dict[str, Any]):
    if name == "lr_monitor":
        return LRMonitor()
    elif name == "memory_monitor":
        return MemoryMonitor()
    elif name == "memory_stats_saver":
        return MemoryStatsSaver(**kwargs)
    elif name == "speed_monitor":
        return SpeedMonitor(
            window_size=kwargs.get("window_size", 1),
            gpu_flops_available=kwargs.get("gpu_flops_available", None),
        )
    elif name == "fdiff":
        return FDiffMetrics(**kwargs)
    elif name == "runtime_estimator":
        return RuntimeEstimator()
    elif name == "moe_ranks_timing":
        return MoeRanksTiming(
            prefix=kwargs.get("prefix", "moe_rank_timers"),
            timers_level=kwargs.get("timers_level", 1),
            moe_layer_name=kwargs.get("moe_layer_name", "DeepseekGMMMoeBlock"),
        )
    elif name == "parallelism_consistency_callback":
        return ParallelismConsistencyCheck(
            batch_interval=kwargs.get("batch_interval", 50)
        )
    elif name == "optimizer_monitor":
        # MODIFIED: add `batch_log_interval` parameter config read
        return OptimizerMonitor(
            log_optimizer_metrics=kwargs.get("log_optimizer_metrics", True),
            batch_log_interval=kwargs.get("batch_log_interval", 10),
            log_pre_clip_gradients=kwargs.get("log_pre_clip_gradients", False),
        )
    elif name == "activation_monitor":
        ignore_module_types = kwargs.get("ignore_module_types", None)
        if ignore_module_types:
            ignore_module_types = om.to_object(ignore_module_types)

        keep_only_module_types = kwargs.get("keep_only_module_types", None)
        if keep_only_module_types:
            keep_only_module_types = om.to_object(keep_only_module_types)

        stat_list = kwargs.get("stat_list", None)
        if stat_list:
            stat_list = om.to_object(stat_list)

        monitor = ActivationMonitor(
            interval=kwargs.get("interval", "25ba"),
            ignore_module_types=ignore_module_types,
            keep_only_module_types=keep_only_module_types,
            only_log_wandb=kwargs.get("only_log_wandb", False),
            log_module_output=kwargs.get("log_module_output", True),
            log_gradients=kwargs.get("log_gradients", False),
            log_grad_input=kwargs.get("log_grad_input", False),
            stat_list=stat_list,
        )
        global_actlog_state.set_monitor_variable(monitor)
        return monitor
    elif name == "weight_monitor":
        ignore_module_types = kwargs.get("ignore_module_types", None)
        if ignore_module_types:
            ignore_module_types = om.to_object(ignore_module_types)

        keep_only_module_types = kwargs.get("keep_only_module_types", None)
        if keep_only_module_types:
            keep_only_module_types = om.to_object(keep_only_module_types)

        expert_module_types = kwargs.get("expert_module_types", None)
        if expert_module_types:
            expert_module_types = om.to_object(expert_module_types)

        stat_list = kwargs.get("stat_list", None)
        if stat_list:
            stat_list = om.to_object(stat_list)

        tp_size = kwargs.get("tp_size", 1)

        return WeightMonitor(
            interval=kwargs.get("interval", "25ba"),
            ignore_module_types=ignore_module_types,
            keep_only_module_types=keep_only_module_types,
            expert_module_types=expert_module_types,
            only_log_wandb=kwargs.get("only_log_wandb", False),
            stat_list=stat_list,
            tp_size=tp_size,
        )
    elif name == "generate_callback":
        prompts = kwargs.pop("prompts")
        return Generate(prompts=list(prompts), **kwargs)
    elif name == "global_lr_scaling":
        return GlobalLRScaling(**kwargs)
    elif name == "layer_freezing":
        return LayerFreezing(**kwargs)
    elif name == "mono_ckpt_saver":
        return MonolithicCheckpointSaver(**kwargs)
    elif name == "scheduled_gc":
        return ScheduledGarbageCollector(**kwargs)
    elif name == "early_stopper":
        return EarlyStopper(**kwargs)
    elif name == "training_metrics_monitor":
        return TrainingMetricsMonitor(
            batch_log_interval=kwargs.get("batch_log_interval", 10)
        )
    elif name == "metric_sampler":
        from llmfoundry.callbacks.metric_sampler import MetricSampler

        return MetricSampler(**kwargs)
    elif name == "expert_balancing_monitor":
        return ExpertBalancingMonitor(**kwargs)
    elif name == "dsa_monitor":
        return DSAMonitor(**kwargs)
    elif name == "dsa_kl_monitor":
        return DSAKLMonitor(**kwargs)
    elif name == "token_dispatcher_monitor":
        return TokenDispatcherMonitor(**kwargs)
    elif name == "dynamic_block_lr_rescaler":
        return DynamicBlockLRRescaler(**kwargs)
    elif name == "dynamic_block_bias_rate_rescaler":
        return DynamicBlockBiasRateRescaler(**kwargs)
    elif name == "stage_time_profiler":
        return StageTimeProfiler()
    elif name == "job_dispatcher_callback":
        return JobCallbackDispatcher(all_kwargs=kwargs)
    elif name == "ema_weights":
        return EMA(
            alpha=kwargs["alpha"],
            update_period=kwargs.get("update_period", 1),
            start_batch=kwargs.get("start_batch", 1),
            start_batch_force_update=kwargs.get("start_batch_force_update", True),
        )
    elif name == "job_dispatcher_callback_web":
        return JobCallbackDispatcherWEB(all_kwargs=kwargs)
    elif name == "autometrics":
        return Autometrics(callback_config=kwargs)
    elif name == "divergence_detector":
        return DivergenceDetector(**kwargs)
    elif name == "processor_fallback_counter":
        return ProcessorFallbackCounter(**kwargs)
    elif name == "data_profiler":
        return DataProfilerCallback(**kwargs)
    elif name == "online_profiler":
        return OnlineProfiler(**kwargs)
    elif name == "optimizer_health_monitor":
        return OptimizerHealthMonitor(**kwargs)
    elif name == "update_weight_ratio_monitor":
        return UpdateWeightRatioMonitor(**kwargs)
    elif name == "router_margin_monitor":
        return RouterMarginMonitor(**kwargs)
    elif name == "topk_accuracy_monitor":
        return TopkAccuracyMonitor(**kwargs)
    elif name == "mtp_acc_len_monitor":
        return MtpAccLenMonitor(**kwargs)
    elif name == "mtp_loss_stage":
        return MtpLossStage(**kwargs)
    elif name == "nsys_capture":
        return NsysCaptureCallback(**kwargs)
    elif name == "system_metrics_monitor":
        return SystemMetricsMonitor(**kwargs)
    elif name == "prores_warmup":
        return ProResWarmup(**kwargs)
    else:
        raise ValueError(f"Not sure how to build callback: {name}")


def build_logger(name: str, kwargs: Dict[str, Any]):
    if name == "wandb":
        return WandBLogger(**kwargs)
    elif name == "tensorboard":
        return TensorboardLogger(**kwargs)
    elif name == "mlflow":
        return MLFlowLogger(**kwargs)
    elif name == "metric_sampler":
        from llmfoundry.callbacks.metric_sampler_logger import MetricSamplerLogger

        return MetricSamplerLogger(**kwargs)
    else:
        raise ValueError(f"Not sure how to build logger: {name}")


def build_algorithm(name: str, kwargs: Dict[str, Any]):
    if name == "gradient_clipping":
        return algorithms.GradientClipping(**kwargs)
    elif name == "alibi":
        return algorithms.Alibi(**kwargs)
    elif name == "fused_layernorm":
        return algorithms.FusedLayerNorm(**kwargs)
    elif name == "gated_linear_units":
        return algorithms.GatedLinearUnits(**kwargs)
    elif name == "low_precision_layernorm":
        return algorithms.LowPrecisionLayerNorm(**kwargs)
    elif name == "update_bias":
        return algorithms.UpdateBias(**kwargs)
    elif name == "sequence_warmup":
        return algorithms.SeqLengthWarmup(**kwargs)
    elif name == "skip_moe_disbalance_step":
        return algorithms.SkipDisbalanceStep(**kwargs)
    elif name == "update_dynamic_router":
        return algorithms.UpdateDynamicGate()
    else:
        raise ValueError(f"Not sure how to build algorithm: {name}")


def _apply_disable_grad(model: torch.nn.Module, patterns: Union[str, list]) -> None:
    if isinstance(patterns, str):
        patterns = [patterns]
    for pat in patterns:
        for n, p in model.named_parameters():
            if re.search(pat, n):
                p.requires_grad_(False)
                log.debug(f"Setting `{n}.requires_grad = False`.")


def _extract_param_groups(
    model: torch.nn.Module,
    optimizer_config: Optional[dict[str, Any]] = None,
) -> Union[Iterable[torch.Tensor], Iterable[dict[str, Any]]]:
    """Extracts parameter groups defined in the optimizer config.

    The optimizer_config defines the optimizer args. It can additionally have key
    `disable_grad` which is a string or list of strings. If a string matches a
    parameter name, then that parameter will have `requires_grad=False`. This is
    useful for freezing parameters. It can additionally have a key
    `param_groups` which is a list of dicts. In this dict, key `param_str_match`
    defines a string; if a parameter name contains this string, then it will be
    in this parameter group. This is useful for grouping parameters together.
    The dict can also contain any other key that is a valid optimizer arg.
    Note: to handle name overlap conflicts, params are assigned to parameter
    groups and added to `param_groups` in the order that `param_str_match` appear
    in `param_groups`.

    Usage
    To disable gradient for all parameters that contain the string "norm" or "bias":
    ```
    optimizer_config: {
        "name": "decoupled_lionw",
        "lr": 1e-3,
        "weight_decay": 1e-2,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "disable_grad": ["norm", "bias"]
    }
    ```

    To create and modify the optimizer parameters for all parameters that contain
    the string "norm" and "bias" separately:
    ```
    optimizer_config: {
        "name": "decoupled_lionw",
        "lr": 1e-3,
        "weight_decay": 1e-2,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "param_groups": [
            {
                "param_str_match": "norm",
                "lr": 1e-4,
                "weight_decay": 0.0,
            },
            {
                "param_str_match": "bias",
                "lr": 5e-4,
                "weight_decay": 0.0,
            },
        ],
    }
    ```

    Args:
        model (torch.nn.Module): model to extract parameters from
        optimizer_config (Dict[str, Any]): optimizer config

    Returns:
        Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]: an iterable of
            torch.Tensor's or dict's. Specifies what Tensors should be optimized
            and their param groupings.
    """
    if optimizer_config is None:
        return model.parameters()

    if "disable_grad" in optimizer_config.keys():
        # We only get here for non-gigafsdp models.
        str_matches = optimizer_config.pop("disable_grad")
        _apply_disable_grad(model, str_matches)

    param_groups_config = optimizer_config.pop("param_groups", None)
    if param_groups_config is not None:
        params = []
        param_dict = OrderedDict((n, p) for n, p in model.named_parameters())

        log.debug(f"Default optimizer settings: {optimizer_config}.")
        for param_group_config in param_groups_config:
            str_match = param_group_config.pop("param_str_match")
            filter_fn = functools.partial(re.search, str_match)
            param_names = [n for n in param_dict.keys() if filter_fn(n)]
            group_params = {"params": [param_dict.pop(n) for n in param_names]}
            group_params.update(param_group_config)

            log.debug(
                f"Creating optimizer param_group with parameters: {param_names} "
                + f"(extracted using {str_match=}). The param_group optimizer "
                + f"setting overrides are: {param_group_config}."
            )

            params.append(group_params)

        params.insert(0, {"params": param_dict.values()})
        return params

    return model.parameters()


def _apply_lr_scales(
    model: torch.nn.Module,
    optimizer: Optimizer,
    lr_scales_config: Any,
) -> None:
    """Register config-driven static per-parameter LR multipliers on the optimizer.

    Lets a config statically knock down (or up) the effective LR of specific
    parameters without creating extra optimizer param groups. Each entry is a
    mapping with a ``param_str_match`` regex (matched against parameter names,
    same convention as ``param_groups``/``disable_grad``) and a ``scale`` float.
    Entries are applied in order, so a later entry overrides an earlier one for
    any param both match.

    The scale is held on the optimizer as a *static base* multiplier
    (``base_lr_scale``), separate from the *dynamic* per-param scale
    (``lr_scale``) that :class:`DynamicBlockLRRescaler` writes. The optimizer
    applies the product ``base * dynamic`` each step, so a config base (e.g. 0.5)
    and a dynamic reduction compose without clobbering each other -- the base is
    the floor-relative-to-original starting point, the dynamic part reduces from
    1.0 on top of it.

    IMPORTANT: parameter identity does not survive FSDP param-flattening, so the
    base cannot simply be keyed by tensor here (the optimizer's param groups are
    rebuilt with fresh tensors during FSDP sharding, orphaning the keys). Instead
    the regex spec is stashed on the optimizer and re-resolved *by name* after
    sharding (``dist_strategy`` calls ``optimizer.apply_base_lr_scales`` post-FSDP,
    on every run including resumes). The application below covers the non-FSDP
    case; under FSDP it is harmlessly overwritten by the post-shard re-resolution.

    Usage::

        optimizer:
          name: decoupled_adamw
          lr: 3e-4
          lr_scales:
            - param_str_match: '\\.layers\\.(3|4|5)\\.'
              scale: 0.5
    """
    if not lr_scales_config:
        return
    if not hasattr(optimizer, "set_lr_scales_spec"):
        log.warning(
            "`lr_scales` was provided in the optimizer config but optimizer "
            f"{type(optimizer).__name__} has no `set_lr_scales_spec`; ignoring. "
            "Use an optimizer that supports per-parameter LR scaling (e.g. "
            "DecoupledAdamW)."
        )
        return

    optimizer.set_lr_scales_spec(list(lr_scales_config))
    # Resolve now so the base is live for non-FSDP runs; FSDP re-resolves
    # post-shard (param identities change there), so this is overwritten then.
    optimizer.apply_base_lr_scales(model.named_parameters())


def build_optimizer(
    model: torch.nn.Module,
    name: str,
    optimizer_config: dict[str, Any],
) -> Optimizer:
    # Config-driven static LR multipliers; applied after construction via
    # `set_lr_scale` (not an optimizer constructor kwarg), so pop before kwargs.
    lr_scales_config = optimizer_config.pop("lr_scales", None)

    params = _extract_param_groups(model, optimizer_config)
    kwargs = {**optimizer_config}

    if "params" in kwargs:
        raise ValueError(
            "The `params` will be automatically extracted from the model and "
            + "optimizer config. Please remove it from the optimizer config kwargs.",
        )

    kwargs["params"] = params

    OPTIMIZER_REGESTRY = {
        "decoupled_adamw": DecoupledAdamW,
        "decoupled_fused_adamw": DecoupledFusedAdamW,
        "decoupled_lionw": DecoupledLionW,
        "clip_lion": DecoupledClipLion,
        "adalr_lion": DecoupledAdaLRLion,
        "adan": Adan,
        "muon": Muon,
    }

    if name not in OPTIMIZER_REGESTRY:
        raise ValueError(f"Not sure how to build optimizer: {name}")

    optimizer = OPTIMIZER_REGESTRY[name.lower()](**kwargs)
    _apply_lr_scales(model, optimizer, lr_scales_config)
    return optimizer


def build_scheduler(
    cfg: DictConfig,
    optimizer_param_str_matches: Optional[list[str]] = None,
) -> Any:
    """Build a Composer scheduler, optionally with per-optimizer-param-group overrides.

    When ``cfg`` contains a non-empty ``param_groups`` list, returns
    :class:`~llmfoundry.scheduler.per_group_scheduler.PerGroupComposerScheduler`.
    Each entry must specify ``param_str_match`` (same string as in the optimizer
    ``param_groups`` config) or ``param_group_index`` (``0`` = default remainder group,
    ``1`` = first matched optimizer group, etc.). Remaining keys are merged with the
    top-level scheduler kwargs; ``name`` may override the scheduler type for that group.

    Args:
        cfg: Scheduler OmegaConf section.
        optimizer_param_str_matches: ``param_str_match`` strings from the optimizer config,
            in order (excluding the default param group). Required when using
            ``param_str_match`` in scheduler ``param_groups``.
    """
    cfg = cfg.copy()

    param_groups_cfg = cfg.pop("param_groups", None)
    if not param_groups_cfg:
        scheduler_name = cfg.pop("name")
        scheduler_cls = SCHEDULER_CLASS_REGISTRY.get(scheduler_name)
        if scheduler_cls is None:
            raise ValueError(f"Not sure how to build scheduler: {scheduler_name}")
        return scheduler_cls(**cfg)

    scheduler_name = cfg.pop("name")
    scheduler_cls = SCHEDULER_CLASS_REGISTRY.get(scheduler_name)
    if scheduler_cls is None:
        raise ValueError(f"Not sure how to build scheduler: {scheduler_name}")

    default_kwargs = om.to_container(cfg, resolve=True)
    default_scheduler = scheduler_cls(**default_kwargs)

    per_group_schedulers: dict[int, Any] = {}
    for raw_entry in param_groups_cfg:
        entry = om.to_container(raw_entry, resolve=True)
        if not isinstance(entry, dict):
            raise TypeError(
                "Each scheduler `param_groups` entry must be a mapping, "
                f"got {type(entry)}",
            )
        idx = _resolve_scheduler_param_group_index(
            entry,
            optimizer_param_str_matches,
        )
        if idx in per_group_schedulers:
            raise ValueError(
                f"Duplicate scheduler settings for optimizer param group index {idx}",
            )
        merged = {**default_kwargs, **entry}
        group_name = merged.pop("name", scheduler_name)
        group_cls = SCHEDULER_CLASS_REGISTRY.get(group_name)
        if group_cls is None:
            raise ValueError(f"Not sure how to build scheduler: {group_name}")
        per_group_schedulers[idx] = group_cls(**merged)

    return PerGroupComposerScheduler(default_scheduler, per_group_schedulers)


def build_tokenizer(om_tokenizer_config: DictConfig) -> PreTrainedTokenizerBase:
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    resolved_om_tokenizer_config = om.to_container(om_tokenizer_config, resolve=True)
    tokenizer_kwargs = resolved_om_tokenizer_config.get(  # type: ignore
        "kwargs", {}
    )
    tokenizer_name = resolved_om_tokenizer_config["name"]  # type: ignore
    # A local path that does not exist falls through to the Hub, where it fails as
    # "Repo id must be in the form 'repo_name' or 'namespace/repo_name'" - which says
    # nothing about the directory being missing.
    if os.path.isabs(str(tokenizer_name)) and not os.path.exists(tokenizer_name):
        raise FileNotFoundError(
            f"tokenizer.name points at a local path that does not exist: {tokenizer_name}"
        )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)

    # HuggingFace does not respect the model_max_length kwarg, and overrides it with
    # min(kwargs['model_max_length'], original_config['model_max_length']), so we
    # explicitly set it here
    tokenizer.model_max_length = tokenizer_kwargs.get(
        "model_max_length",
        int(1e30),
    )

    return tokenizer


def extract_categories(icl_tasks: Union[str, list[dict[str, Any]]]):
    categoriy_labels = set()
    task_with_aggregation = []
    for task in icl_tasks:
        if "gauntlet_tags" in task:
            task_with_aggregation.append(task)
            for label in task["gauntlet_tags"]:
                categoriy_labels.add(label)

    categories = []
    for label in categoriy_labels:
        category = {}
        category["name"] = label
        category["benchmarks"] = []
        for task in task_with_aggregation:
            if label in task["gauntlet_tags"]:
                for num in task["num_fewshot"]:
                    bench = {}
                    bench["name"] = task["label"]
                    bench["num_fewshot"] = num
                    category["benchmarks"].append(bench)
        categories.append(category)

    return categories


def build_icl_data_and_gauntlet(
    icl_tasks_config: Union[str, list[dict[str, Any]]],
    eval_gauntlet_config: Optional[Union[str, dict[str, Any]]],
    tokenizer: PreTrainedTokenizerBase,
    icl_seq_len: int,
    device_eval_batch_size: int,
    cached_datasets_directory_template: Optional[str] = None,
) -> tuple[list[Evaluator], list[str], Optional[EvalGauntlet]]:
    icl_evaluators, logger_keys = build_icl_evaluators(
        icl_tasks_config,
        tokenizer,
        icl_seq_len,
        device_eval_batch_size,
        cached_datasets_directory_template=cached_datasets_directory_template,
    )
    eval_gauntlet_cb = None
    if eval_gauntlet_config is not None:
        assert isinstance(eval_gauntlet_config, DictConfig)
        eval_gauntlet = to_dict_container(eval_gauntlet_config)

        eval_gauntlet["categories"] = extract_categories(icl_tasks_config)
        eval_gauntlet["logger_keys"] = logger_keys
        eval_gauntlet["benchmark_sizes"] = {
            e.label: e.dataloader.num_samples for e in icl_evaluators
        }
        eval_gauntlet_cb = EvalGauntlet(**eval_gauntlet)
    return icl_evaluators, logger_keys, eval_gauntlet_cb


def build_icl_evaluators(
    icl_tasks: Union[str, ListConfig],
    tokenizer: PreTrainedTokenizerBase,
    default_max_seq_len: int,
    default_batch_size: int,
    destination_dir: Optional[str] = None,
    cached_datasets_directory_template: Optional[str] = None,
):
    if destination_dir is None:
        destination_dir = os.getcwd()
    evaluators = []
    logger_keys = []

    icl_tasks_list = None
    if isinstance(icl_tasks, str):
        print(f"Extracting ICL task config from path: {icl_tasks}")
        with open(icl_tasks, "r") as icl_f:
            icl_task_cfg = om.load(icl_f)
        icl_tasks_list = icl_task_cfg.icl_tasks
    else:
        icl_tasks_list = icl_tasks

    mmlu_tasks_names = [
        "mmlu",
        "fakemmlu",
        "babymmlu",
        "teenmmlu",
        "rummlu",
        "medmmlu",
        "biommlu",
        "chemmmlu",
        "mathmmlu",
    ]

    vision_mcqa_tasks_names = [
        "mmmu",
        "mmbench",
        "rusdocvqa",
        "chart_summ",
        "vim_mmmu",
        "mme-realworld-ocr",
        "blink",
        "erqa",
        "2d_spatial_ability",
        "3d_depth_ability",
        "counting_ability",
        "multi_view_ability",
    ]

    def _validate_cfg(icl_cfg: DictConfig):
        assert "label" in icl_cfg
        assert "dataset_uri" in icl_cfg and icl_cfg.dataset_uri is not None
        assert "icl_task_type" in icl_cfg
        assert "num_fewshot" in icl_cfg

        if "metric_names" not in icl_cfg:
            if icl_cfg.icl_task_type == "language_modeling":
                icl_cfg.metric_names = ["InContextLearningLMAccuracy"]
            elif (
                icl_cfg.icl_task_type == "multiple_choice"
                or icl_cfg.icl_task_type in (mmlu_tasks_names + vision_mcqa_tasks_names)
            ):
                icl_cfg.metric_names = ["InContextLearningMultipleChoiceAccuracy"]
            elif icl_cfg.icl_task_type == "schema":
                icl_cfg.metric_names = ["InContextLearningMultipleChoiceAccuracy"]
            elif icl_cfg.icl_task_type == "question_answering":
                icl_cfg.metric_names = ["InContextLearningQAAccuracy"]
            else:
                raise ValueError(
                    f"No metric_names defined, unable to build default metrics for icl_task_type={icl_cfg.icl_task_type}."
                )

        if icl_cfg.icl_task_type not in (mmlu_tasks_names + vision_mcqa_tasks_names):
            if "prompt_string" not in icl_cfg:
                icl_cfg.prompt_string = ""
            if "example_delimiter" not in icl_cfg:
                icl_cfg.example_delimiter = "\n"
            if "continuation_delimiter" not in icl_cfg:
                icl_cfg.continuation_delimiter = " "
        elif "mmlu" in icl_cfg.icl_task_type:  # one of MMLU-like benchmarks
            # first, set special overrides if needed
            if icl_cfg.icl_task_type in ["babymmlu", "teenmmlu"]:
                if "prompt_string" not in icl_cfg:
                    if icl_cfg.icl_task_type == "babymmlu":
                        icl_cfg.prompt_string = ""
                    else:  # teenmmlu
                        icl_cfg.prompt_string = "The following are multiple choice questions (with answers) about {subject}."
                if "prompt_choice_template" not in icl_cfg:
                    icl_cfg.prompt_choice_template = ""
                if "prompt_answer_template" not in icl_cfg:
                    icl_cfg.prompt_answer_template = ""
            elif icl_cfg.icl_task_type in [
                "rummlu",
                "chemmmlu",
                "mathmmlu",
                "biommlu",
                "medmmlu",
            ]:
                if "prompt_string" not in icl_cfg:
                    if icl_cfg.icl_task_type == "rummlu":
                        icl_cfg.prompt_string = "Ниже приведены вопросы с множественным выбором (с ответами) по {subject}"
                    else:
                        icl_cfg.prompt_string = "Ниже приведены вопросы с множественным выбором (с ответами) по теме {subject}. Напиши только букву ответа."
                if "prompt_choice_template" not in icl_cfg:
                    icl_cfg.prompt_choice_template = "{choice_symbol}. {choice}"
                if "prompt_answer_template" not in icl_cfg:
                    icl_cfg.prompt_answer_template = "Ответ:"
            elif icl_cfg.icl_task_type == "medmmlu":
                if "prompt_string" not in icl_cfg:
                    icl_cfg.prompt_string = "Ты - опытный врач-клинический психолог. Ты проводишь прием пациента.\nТебе будут задаваться вопросы с вариантами ответов по твоей специальности.\nТвоя специальность - {subject}!\nТвоя задача - выбрать один фактологически верный вариант ответа!\nВажно: в своем ответе верни только одну букву A, B, C или D!"
                if "prompt_choice_template" not in icl_cfg:
                    icl_cfg.prompt_choice_template = "{choice_symbol}. {choice}"
                if "prompt_answer_template" not in icl_cfg:
                    icl_cfg.prompt_answer_template = "Ответ:"

            # now, fill in the default values that remain empty
            if "prompt_string" not in icl_cfg:
                icl_cfg.prompt_string = "The following are multiple choice questions (with answers) about {subject}."
            if "prompt_query_template" not in icl_cfg:
                icl_cfg.prompt_query_template = "{query}"
            if "prompt_choice_template" not in icl_cfg:
                icl_cfg.prompt_choice_template = "{choice_symbol}. {choice}"
            if "prompt_answer_template" not in icl_cfg:
                icl_cfg.prompt_answer_template = "Answer:"
            if "example_delimiter" not in icl_cfg:
                icl_cfg.example_delimiter = "\n\n"
            if "continuation_delimiter" not in icl_cfg:
                icl_cfg.continuation_delimiter = "\n"
            if "dataset_json_filename" not in icl_cfg:
                icl_cfg.dataset_json_filename = "test.jsonl"
            if "few_shot_dataset_json_filename" not in icl_cfg:
                icl_cfg.few_shot_dataset_json_filename = "few_shot.jsonl"
            if "choice_num2symbol" not in icl_cfg:
                icl_cfg.choice_num2symbol = DEFAULT_CHOICE_NUM2SYMBOL
        elif icl_cfg.icl_task_type in vision_mcqa_tasks_names:
            if icl_cfg.icl_task_type == "mmmu":
                if "prompt_string" not in icl_cfg:
                    icl_cfg.prompt_string = "The following are multiple choice questions (with answers) about {subject}. Answer the question with only one letter."
                if "dataset_json_filename" not in icl_cfg:
                    icl_cfg.dataset_json_filename = (
                        "data/mmmu_testqa_4_choices_intrain.jsonl"
                    )
            elif icl_cfg.icl_task_type == "mmbench":
                if "prompt_string" not in icl_cfg:
                    icl_cfg.prompt_string = "The following are multiple choice questions (with answers). Answer the question with only one letter."
                if "dataset_json_filename" not in icl_cfg:
                    icl_cfg.dataset_json_filename = "mmbench_4_choices_intrain.jsonl"
            elif icl_cfg.icl_task_type == "mme-realworld-ocr":
                if "prompt_string" not in icl_cfg:
                    icl_cfg.prompt_string = "The following are multiple choice questions (with answers). Answer the question with only one letter."
                if "dataset_json_filename" not in icl_cfg:
                    icl_cfg.dataset_json_filename = (
                        "intrain_final_ocr_MME_RealWorld.json"
                    )
                icl_cfg.choice_num2symbol = {
                    0: "A",
                    1: "B",
                    2: "C",
                    3: "D",
                    4: "E",
                }
            elif icl_cfg.icl_task_type == "rusdocvqa":
                if "prompt_string" not in icl_cfg:
                    icl_cfg.prompt_string = "Ниже приведен вопрос с несколькими вариантами ответов. В качестве ответа предоставь только одну букву."
                if "prompt_answer_template" not in icl_cfg:
                    icl_cfg.prompt_answer_template = "Ответ:"
                if "dataset_json_filename" not in icl_cfg:
                    icl_cfg.dataset_json_filename = "rusdocvqa_intrain.jsonl"
            elif icl_cfg.icl_task_type == "chart_summ":
                if "prompt_string" not in icl_cfg:
                    icl_cfg.prompt_string = "Ниже приведен вопрос с несколькими вариантами ответов. В качестве ответа предоставь только одну букву."
                if "prompt_answer_template" not in icl_cfg:
                    icl_cfg.prompt_answer_template = "Ответ:"
                if "dataset_json_filename" not in icl_cfg:
                    icl_cfg.dataset_json_filename = "chart_summ_intrain.jsonl"
            elif icl_cfg.icl_task_type == "vim_mmmu":
                if "prompt_string" not in icl_cfg:
                    icl_cfg.prompt_string = "Follow the instructions on the image."
                if "dataset_json_filename" not in icl_cfg:
                    icl_cfg.dataset_json_filename = "vim_mmmu__intrain.jsonl"
            # Robotics intrain benchmarks
            elif icl_cfg.icl_task_type == "blink":
                if "prompt_string" not in icl_cfg:
                    icl_cfg.prompt_string = "The following are multiple choice questions (with answers). Answer the question with only one letter."
                if "dataset_json_filename" not in icl_cfg:
                    icl_cfg.dataset_json_filename = "blink_intrain.jsonl"
            elif icl_cfg.icl_task_type == "erqa":
                if "prompt_string" not in icl_cfg:
                    icl_cfg.prompt_string = "The following are multiple choice questions (with answers). Answer the question with only one letter."
                if "dataset_json_filename" not in icl_cfg:
                    icl_cfg.dataset_json_filename = "erqa_intrain.jsonl"

            if "prompt_string" not in icl_cfg:
                icl_cfg.prompt_string = (
                    "The following are multiple choice questions (with answers)"
                )
            if "prompt_query_template" not in icl_cfg:
                icl_cfg.prompt_query_template = "{query}"
            if "prompt_choice_template" not in icl_cfg:
                icl_cfg.prompt_choice_template = "{choice_symbol}. {choice}"
            if "prompt_answer_template" not in icl_cfg:
                icl_cfg.prompt_answer_template = "Answer:"
            if "example_delimiter" not in icl_cfg:
                icl_cfg.example_delimiter = "\n\n"
            if "continuation_delimiter" not in icl_cfg:
                icl_cfg.continuation_delimiter = "\n"
            if "few_shot_dataset_json_filename" not in icl_cfg:
                icl_cfg.few_shot_dataset_json_filename = None
            if "choice_num2symbol" not in icl_cfg:
                icl_cfg.choice_num2symbol = {0: "A", 1: "B", 2: "C", 3: "D"}
            if "num_fewshot" not in icl_cfg:
                icl_cfg.num_fewshot = [0]
            if "dataset_images_dirname" not in icl_cfg:
                icl_cfg.dataset_images_dirname = "images"
            if "max_seq_len" not in icl_cfg:
                icl_cfg.max_seq_len = 4096
        else:
            raise ValueError(
                f"No default settings for icl_task_type={icl_cfg.icl_task_type}."
            )

        if "max_seq_len" not in icl_cfg:
            icl_cfg.max_seq_len = default_max_seq_len
        if "batch_size" not in icl_cfg:
            icl_cfg.batch_size = default_batch_size
        if "pass_at_k" not in icl_cfg:
            icl_cfg.pass_at_k = 1
        if "num_beams" not in icl_cfg:
            icl_cfg.num_beams = 20

    for icl_cfg in icl_tasks_list:
        _validate_cfg(icl_cfg)
        for num_fewshot in list(icl_cfg.num_fewshot):
            if tokenizer.pad_token_id is None:
                # Current workaround to support GPT2 tokenizer with `pad_token_id = None`
                pad_tok_id = tokenizer.eos_token_id
            else:
                pad_tok_id = tokenizer.pad_token_id
            label = f"{icl_cfg.label}/{num_fewshot}-shot"
            metric_names = list(icl_cfg.metric_names)
            # TODO: fix Composer bug when copying local paths and destination exists
            destination_path = f"{destination_dir}/{icl_cfg.label}-{num_fewshot}.jsonl"
            if dist.get_local_rank() == 0 and os.path.exists(destination_path):
                os.remove(destination_path)
            dist.barrier()
            early_stopping_criteria = icl_cfg.get("early_stopping_criteria", None)
            if isinstance(early_stopping_criteria, ListConfig):
                early_stopping_criteria = om.to_container(early_stopping_criteria)
            assert early_stopping_criteria is None or isinstance(
                early_stopping_criteria, list
            )

            if icl_cfg.icl_task_type not in (
                mmlu_tasks_names + vision_mcqa_tasks_names
            ):
                dataloaders = get_icl_task_dataloader(
                    icl_cfg.icl_task_type,
                    icl_cfg.dataset_uri,
                    tokenizer,
                    batch_size=icl_cfg.batch_size,
                    max_seq_len=icl_cfg.max_seq_len,
                    pad_tok_id=pad_tok_id,
                    num_fewshot=num_fewshot,
                    prompt_string=icl_cfg.prompt_string,
                    example_delimiter=icl_cfg.example_delimiter,
                    continuation_delimiter=icl_cfg.continuation_delimiter,
                    question_prelimiter=icl_cfg.get("question_prelimiter", ""),
                    destination_path=destination_path,
                    pass_at_k=icl_cfg.pass_at_k,
                    generations_per_sample=icl_cfg.num_beams,
                    has_categories=icl_cfg.get("has_categories", False),
                    cot_delimiter=icl_cfg.get("cot_delimiter", ""),
                    early_stopping_criteria=early_stopping_criteria,
                    do_normalization=icl_cfg.get("do_normalization", True),
                    cached_datasets_directory_template=cached_datasets_directory_template,
                )
            elif (
                icl_cfg.icl_task_type in mmlu_tasks_names
                and icl_cfg.icl_task_type not in ["babymmlu", "teenmmlu"]
            ):
                dataloaders = get_mmlu_dataloader(
                    cfg=icl_cfg,
                    batch_size=icl_cfg.batch_size,
                    dataset_uri=icl_cfg.dataset_uri,
                    tokenizer=tokenizer,
                    max_seq_len=icl_cfg.max_seq_len,
                    pad_tok_id=pad_tok_id,
                    num_fewshot=num_fewshot,
                    prompt_string=icl_cfg.prompt_string,
                    prompt_query_template=icl_cfg.prompt_query_template,
                    prompt_choice_template=icl_cfg.prompt_choice_template,
                    prompt_answer_template=icl_cfg.prompt_answer_template,
                    example_delimiter=icl_cfg.example_delimiter,
                    continuation_delimiter=icl_cfg.continuation_delimiter,
                    dataset_json_filename=icl_cfg.dataset_json_filename,
                    few_shot_dataset_json_filename=icl_cfg.few_shot_dataset_json_filename,
                    choice_num2symbol=icl_cfg.choice_num2symbol,
                    has_categories=icl_cfg.get("has_categories", False),
                    cached_datasets_directory_template=cached_datasets_directory_template,
                )
            elif icl_cfg.icl_task_type in ["babymmlu", "teenmmlu"]:
                dataloaders = get_mmlu_dataloader(
                    cfg=icl_cfg,
                    batch_size=icl_cfg.batch_size,
                    dataset_uri=icl_cfg.dataset_uri,
                    tokenizer=tokenizer,
                    max_seq_len=icl_cfg.max_seq_len,
                    pad_tok_id=pad_tok_id,
                    num_fewshot=num_fewshot,
                    prompt_string=icl_cfg.prompt_string,
                    prompt_query_template=icl_cfg.prompt_query_template,
                    prompt_choice_template=icl_cfg.prompt_choice_template,
                    prompt_answer_template=icl_cfg.prompt_answer_template,
                    example_delimiter=icl_cfg.example_delimiter,
                    continuation_delimiter=icl_cfg.continuation_delimiter,
                    dataset_json_filename=icl_cfg.dataset_json_filename,
                    few_shot_dataset_json_filename=icl_cfg.few_shot_dataset_json_filename,
                    choice_num2symbol=icl_cfg.choice_num2symbol,
                    has_categories=icl_cfg.get("has_categories", False),
                    cached_datasets_directory_template=cached_datasets_directory_template,
                    babymmlu=True,
                )
            elif icl_cfg.icl_task_type in vision_mcqa_tasks_names:
                dataloaders = get_omni_mcqa_dataloader(
                    cfg=icl_cfg,
                    batch_size=icl_cfg.batch_size,
                    dataset_uri=icl_cfg.dataset_uri,
                    tokenizer=tokenizer,
                    max_seq_len=icl_cfg.max_seq_len,
                    pad_tok_id=pad_tok_id,
                    num_fewshot=num_fewshot,
                    prompt_string=icl_cfg.prompt_string,
                    prompt_query_template=icl_cfg.prompt_query_template,
                    prompt_choice_template=icl_cfg.prompt_choice_template,
                    prompt_answer_template=icl_cfg.prompt_answer_template,
                    example_delimiter=icl_cfg.example_delimiter,
                    continuation_delimiter=icl_cfg.continuation_delimiter,
                    dataset_json_filename=icl_cfg.dataset_json_filename,
                    dataset_files_dirname=icl_cfg.dataset_files_dirname,
                    few_shot_dataset_json_filename=icl_cfg.few_shot_dataset_json_filename,
                    choice_num2symbol=icl_cfg.choice_num2symbol,
                    cached_datasets_directory_template=cached_datasets_directory_template,
                )
            else:
                raise ValueError(
                    f"No default settings for icl_task_type={icl_cfg.icl_task_type}."
                )
            if (
                hasattr(icl_cfg, "has_categories")
                and icl_cfg.has_categories
                and isinstance(dataloaders, dict)
            ):
                for category in dataloaders.keys():
                    logger_keys.extend(
                        [f"metrics/{label}/{category}/{m}" for m in metric_names]
                    )
                    evaluators.append(
                        Evaluator(
                            label=f"{label}/{category}",
                            dataloader=dataloaders[category],
                            metric_names=metric_names,
                        ),
                    )
            else:
                logger_keys.extend([f"metrics/{label}/{m}" for m in metric_names])
                evaluators.append(
                    Evaluator(
                        label=label, dataloader=dataloaders, metric_names=metric_names
                    ),
                )

    return evaluators, logger_keys
