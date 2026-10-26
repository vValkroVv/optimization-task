"""Checks of the materialized model config against its sibling sections
(parallelism, training, loss, dataloader).

Torch-free, so both validation stages run them, and
runtime model preparation raises on their errors: one implementation,
so pre-submit cannot pass what the runtime rejects. Issue paths are relative to
the document root.
"""

import re
from dataclasses import fields

from torchtitan.config.schema.models.common.embeddings.rope.rope import RoPEConfig
from torchtitan.config.schema.models.common.moe.dispatch.deepep import (
    DeepEPTokenDispatcherConfig,
)
from torchtitan.config.schema.models.common.moe.dispatch.hybridep import (
    HybridEPTokenDispatcherConfig,
)
from torchtitan.config.schema.models.common.moe.moe import MoEConfig
from torchtitan.config.schema.protocols.module import iter_module_configs
from torchtitan.config.validation import Issue, issue_for_field


def model_fits_parallelism(model_config, parallelism) -> list[Issue]:
    """Check heads, experts, the token dispatcher and vocab against the TP/EP
    degrees."""
    issues: list[Issue] = model_config.validate_parallelism(parallelism)
    tp = parallelism.tensor_parallel_degree
    ep = parallelism.expert_parallel_degree
    if tp > 1:
        # One issue per distinct head count: the projections under an
        # attention repeat its counts.
        seen: set[int] = set()
        # Every head-splitting module, not only the decoder's attention: the
        # vision encoder shards by head too. TP shards K/V by head as well.
        for fqn, module_config in iter_module_configs(model_config):
            for key in ("n_heads", "n_kv_heads", "num_k_heads", "num_v_heads"):
                heads = getattr(module_config, key, None)
                if isinstance(heads, int) and heads % tp and heads not in seen:
                    seen.add(heads)
                    issues.append(
                        issue_for_field(
                            module_config,
                            key=key,
                            path=f"model.{fqn}.{key}",
                            message=f"{heads} is not divisible by "
                            f"parallelism.tensor_parallel_degree {tp}",
                        )
                    )
        vocab = getattr(
            getattr(model_config, "tok_embeddings", None), "num_embeddings", None
        )
        if (
            isinstance(vocab, int)
            and not parallelism.disable_loss_parallel
            and vocab % tp
        ):
            issues.append(
                issue_for_field(
                    model_config.tok_embeddings,
                    key="num_embeddings",
                    path="model.tok_embeddings.num_embeddings",
                    message=f"{vocab} is not divisible by tensor_parallel_degree "
                    f"{tp}; loss parallel shards the vocab axis",
                    severity="warning",
                )
            )
    seen_experts: set[int] = set()
    seen_dispatchers: set[type] = set()
    for fqn, moe, _, _ in model_config.traverse(MoEConfig):
        if ep > 1 and moe.num_experts % ep and moe.num_experts not in seen_experts:
            seen_experts.add(moe.num_experts)
            issues.append(
                issue_for_field(
                    moe.experts,
                    key="num_experts",
                    path=f"model.{fqn}.experts.num_experts",
                    message=f"{moe.num_experts} is not divisible by "
                    f"parallelism.expert_parallel_degree {ep}",
                )
            )
        dispatcher_type = type(moe.experts.token_dispatcher)
        if (
            ep == 1
            and issubclass(
                dispatcher_type,
                (DeepEPTokenDispatcherConfig, HybridEPTokenDispatcherConfig),
            )
            and dispatcher_type not in seen_dispatchers
        ):
            seen_dispatchers.add(dispatcher_type)
            issues.append(
                issue_for_field(
                    moe.experts,
                    key="token_dispatcher",
                    path=f"model.{fqn}.experts.token_dispatcher",
                    message=f"{dispatcher_type.__qualname__} requires "
                    "parallelism.expert_parallel_degree > 1",
                )
            )
    return issues


def seq_len_fits_model(
    model_config, seq_len: int, parallelism, *, path: str = "training.seq_len"
) -> list[Issue]:
    issues = [
        Issue(
            path=path,
            message=f"{seq_len} exceeds model.{fqn}.max_seq_len {rope.max_seq_len}",
        )
        for fqn, rope, _, _ in model_config.traverse(RoPEConfig)
        if seq_len > rope.max_seq_len
    ]
    divisor = model_config.sequence_length_divisor(parallelism)
    if seq_len % divisor:
        issues.append(
            Issue(
                path=path,
                message=f"must be divisible by sequence partition size {divisor}",
            )
        )
    return issues


def mtp_matches_loss(model_config, loss) -> list[Issue]:
    mtp = getattr(model_config, "mtp", None)
    num_mtp = 0 if mtp is None else len(mtp.layers)
    weights = getattr(loss, "loss_weights", None)
    if weights is not None:
        if len(weights) != num_mtp:
            return [
                Issue(
                    path="loss.loss_weights",
                    message=f"{len(weights)} weight(s) for {num_mtp} MTP head(s); "
                    "one weight per MTP head is required",
                )
            ]
    elif num_mtp > 0:
        return [
            Issue(
                path="loss",
                message=f"the model has {num_mtp} MTP head(s) but the loss is not "
                "`type: mtp` with one loss weight per head",
            )
        ]
    return []


_VISION_PATCH_KEYS = ("patch_size", "temporal_patch_size", "spatial_merge_size")


def vision_dataloader_agrees(model_config, dataloader) -> list[Issue]:
    """Check the image pipeline produces the patches the vision encoder expects."""
    vision = getattr(model_config, "vision_encoder", None)
    multimodal = all(hasattr(dataloader, key) for key in _VISION_PATCH_KEYS)
    if vision is not None and not multimodal:
        return [
            Issue(
                path="dataloader",
                message=f"the model has a vision encoder but the dataloader is "
                f"{type(dataloader).__qualname__}, not hf_multimodal; the vision "
                "encoder would never see an image",
                severity="warning",
            )
        ]
    if multimodal and vision is None:
        return [
            Issue(
                path="dataloader",
                message="hf_multimodal dataloader but the model has no vision "
                "encoder; the images it produces have nowhere to go",
                severity="warning",
            )
        ]
    if vision is None:
        return []
    model_values = {
        "patch_size": vision.patch_embed.patch_size,
        "temporal_patch_size": vision.patch_embed.temporal_patch_size,
        "spatial_merge_size": vision.spatial_merge_size,
    }
    return [
        Issue(
            path=f"dataloader.{key}",
            message=f"{getattr(dataloader, key)} does not match the vision encoder's "
            f"{key} {value}; the image pipeline must produce the patches the "
            "vision encoder was built for",
        )
        for key, value in model_values.items()
        if getattr(dataloader, key) != value
    ]


def fp8_grouped_experts_issues(experts, *, path: str = "model.experts") -> list[Issue]:
    """DeepGEMM/dispatcher layout restrictions, without enabling FP8."""
    issues: list[Issue] = []
    dispatcher = experts.token_dispatcher
    if not isinstance(
        dispatcher, (DeepEPTokenDispatcherConfig, HybridEPTokenDispatcherConfig)
    ):
        issues.append(
            Issue(
                path=path + ".token_dispatcher",
                message=(
                    "sparse_use_grouped_gemm experts require comm_backend "
                    "'deepep' or 'hybridep'"
                ),
            )
        )
    if experts.score_before_experts:
        issues.append(
            Issue(
                path=path + ".score_before_experts",
                message=(
                    "sparse_use_grouped_gemm experts require score_before_experts=False"
                ),
            )
        )
    if isinstance(dispatcher, HybridEPTokenDispatcherConfig):
        if experts.dim is None or experts.dim % 512:
            issues.append(
                Issue(
                    path=path + ".dim",
                    message=f"fp8 HybridEP dispatch requires dim % 512 == 0, got dim={experts.dim}",
                )
            )
        if dispatcher.pad_multiple is None or dispatcher.pad_multiple % 128:
            issues.append(
                Issue(
                    path=path + ".token_dispatcher.pad_multiple",
                    message=(
                        "fp8 HybridEP dispatch requires pad_multiple to be a "
                        "multiple of 128"
                    ),
                )
            )
        if dispatcher.non_blocking_capacity_factor is not None:
            issues.append(
                Issue(
                    path=path + ".token_dispatcher.non_blocking_capacity_factor",
                    message=(
                        "fp8 HybridEP dispatch does not support "
                        "non_blocking_capacity_factor yet: the fixed-capacity tail "
                        "reaches gradients as NaN"
                    ),
                )
            )
        if not dispatcher.fuse_permute:
            issues.append(
                Issue(
                    path=path + ".token_dispatcher.fuse_permute",
                    message="fp8 HybridEP dispatch requires fuse_permute=True",
                )
            )
    if experts.param_init is None or "w13_E2FD" not in experts.param_init:
        issues.append(
            Issue(
                path=path + ".param_init",
                message=(
                    "sparse_use_grouped_gemm experts require a param_init entry "
                    "for fused w13_E2FD"
                ),
            )
        )
    return issues


def dense_fp8_replacement_issues(model_config, pattern: str) -> list[Issue]:
    """Reject specialized projection state that a dense FP8 replacement loses."""
    from torchtitan.config.schema.models.common.nn_modules import LinearConfig
    from torchtitan.config.schema.ops.float8.cuda_gemm import FP8LinearConfig

    if not pattern:
        return []
    selected = re.compile(pattern)
    target_fields = {f.name for f in fields(FP8LinearConfig) if f.init}
    issues = []
    for fqn, linear, _, _ in model_config.traverse(LinearConfig):
        if not selected.search(fqn):
            continue
        extra = {f.name for f in fields(linear) if f.init} - target_fields
        if extra:
            issues.append(
                Issue(
                    path="fp8.dense_include_linear_layers",
                    message=f"dense FP8 cannot replace specialized linear {fqn!r}: "
                    f"unsupported fields {sorted(extra)}; exclude this projection",
                )
            )
    return issues


def model_fits_fp8(model_config, fp8, parallelism) -> list[Issue]:
    from torchtitan.config.schema.models.common.moe.experts.gmm import (
        GroupedExpertsConfig,
    )

    issues = dense_fp8_replacement_issues(model_config, fp8.dense_include_linear_layers)
    if fp8.dense_include_linear_layers and parallelism.tensor_parallel_degree > 1:
        issues.append(
            Issue(
                path="fp8.dense_include_linear_layers",
                message="Dense fp8 does not support tensor parallelism",
            )
        )
    for fqn, experts, _, _ in model_config.traverse(GroupedExpertsConfig):
        if fp8.sparse_use_grouped_gemm or experts.use_float8_grouped_gemm:
            prefix = f"model.{fqn}."
            for issue in fp8_grouped_experts_issues(experts, path=f"model.{fqn}"):
                key = issue.path.removeprefix(prefix)
                owner = experts
                if key.startswith("token_dispatcher."):
                    owner = experts.token_dispatcher
                    key = key.removeprefix("token_dispatcher.")
                mapped = issue_for_field(
                    owner, key=key, path=issue.path, message=issue.message
                )
                if key == "dim" and mapped.source is None:
                    source_path = "model.config.dim"
                    source = model_config._source_origins.get(source_path)
                    if source:
                        mapped = Issue(
                            path=source_path,
                            message=issue.message,
                            source=source,
                            runtime_path=issue.path,
                        )
                issues.append(mapped)
    unique = {}
    for issue in issues:
        unique.setdefault(
            (issue.path, issue.message, issue.severity, issue.source), issue
        )
    return list(unique.values())


def model_module_field_issues(model_config) -> list[Issue]:
    """Reject compact module state that its runtime builder cannot honor."""
    issues: list[Issue] = []
    configs = [("", model_config), *iter_module_configs(model_config)]
    for fqn, module in configs:
        path = "model.config" + ("." + fqn if fqn else "")
        if not type(module).default_init_role:
            # Materialization records role labels even on containers. They
            # are descriptive metadata once initializer tables are explicit.
            fields = (
                ("param_init",)
                if getattr(model_config, "materialized", False)
                else ("param_init", "init_role")
            )
            for field in fields:
                if getattr(module, field) is not None:
                    issues.append(
                        Issue(
                            path=f"{path}.{field}",
                            message=(
                                "this module owns no parameters initialized by an init role; "
                                "configure its parameter-owning children instead"
                            ),
                        )
                    )
        elif module.param_init == {}:
            issues.append(
                Issue(
                    path=f"{path}.param_init",
                    message=(
                        "empty initializer table; use an explicit {type: skip} "
                        "for each parameter whose initialization should be skipped"
                    ),
                )
            )
        if isinstance(module, RoPEConfig) and module.hidden_z_loss is not None:
            issues.append(
                Issue(
                    path=f"{path}.hidden_z_loss",
                    message=(
                        "RoPE is a cache/position transform, not a hidden-activation "
                        "loss target"
                    ),
                )
            )
        if module.sharding_config is not None and (
            isinstance(module, RoPEConfig) or model_config.runtime_manages_sharding
        ):
            issues.append(
                Issue(
                    path=f"{path}.sharding_config",
                    message=(
                        "sharding is prepared by the model runtime from parallelism; "
                        "a YAML sharding_config would be overwritten"
                    ),
                )
            )
    return issues


def require_model_module_fields(model_config) -> None:
    issues = model_module_field_issues(model_config)
    if issues:
        raise ValueError("; ".join(str(issue) for issue in issues))
