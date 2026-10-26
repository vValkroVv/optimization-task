import math
from abc import ABC, abstractmethod
import typing as tp

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from composer.utils import global_actlog_state
from composer.utils.dist import get_ep_group

from llmfoundry.models.layers.actlog_names import attach_actlog_name
from llmfoundry.models.layers.ffn import LlamaMLP, ParallelMLP
from llmfoundry.models.ops import grouped_gemm as gg
from llmfoundry.models.ops.hidden_z_loss_hook import apply_hidden_z_loss_hook
from llmfoundry.utils.misc import divide
from llmfoundry.models.parallel.tensor.utils import _initialize_ep_weight
from llmfoundry.models.layers.moe.sonic_cuda import require_sonic_moe_cuda_device
from llmfoundry.models.ops.float8.grouped_gemm import GroupedGemmFp8Wrapper
from llmfoundry.models.ops.float8.triton_kernels.fused_swiglu_quantization import fused_swiglu_quantized_fn
from llmfoundry.models.ops.float8.triton_kernels.utils import build_m_indices
from llmfoundry.models.ops.float8.triton_kernels.dequantization_deprecated import dequantize_kernel as fp8_dequantize
from composer.utils.dist import get_ep_group_size
from llmfoundry.models.ops.float8.triton_kernels.fused_te_ops import (
    unpad_and_unpermute_fn,
    pad_permuted_for_grouped_gemm,
    unpad_grouped_gemm_output,
)
from llmfoundry.models.ops.float8.triton_kernels.fixed_transformer_engine_permutation import (
    moe_unpermute as fused_unpermute,
)
from llmfoundry.models.ops.float8.fp8_tensor import Float8BlockwiseQTensor, InputLayoutType


# from sonicmoe.moe import MoE
try:
    from llmfoundry.models.layers.moe.sonicmoe.enums import ActivationType
    from llmfoundry.models.layers.moe.sonicmoe.functional import moe_general_routing_inputs
except (ImportError, AssertionError):
    ActivationType = None
    moe_general_routing_inputs = None

# TODO: could this be made into something like / replaced by llmfoundry.models.ops.mlp.FusedGatedMLPFunc?
@torch.compile
def fused_swiglu(
    x: torch.Tensor, 
    probs: torch.Tensor,
    swiglu_limit: float = 0.0,
    clip_counts: tp.Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Fused SwiGLU with optional activation clipping (DeepSeek-style).

    When ``swiglu_limit > 0``:
        gate = clamp(gate, -inf, swiglu_limit)
        up   = clamp(up,   -swiglu_limit, swiglu_limit)

    ``clip_counts`` (int32[2]) is mutated in-place when provided and
    ``swiglu_limit > 0``, accumulating [gate_clamped, up_clamped] counts.
    """
    gate, up = torch.chunk(x, 2, dim=-1)
    if swiglu_limit > 0:
        if clip_counts is not None:
            clip_counts[0].add_(gate.ge(swiglu_limit).sum().to(torch.int32))
            clip_counts[1].add_(up.le(-swiglu_limit).logical_or(up.ge(swiglu_limit)).sum().to(torch.int32))
        gate = gate.clamp(max=swiglu_limit)
        up = up.clamp(-swiglu_limit, swiglu_limit)
    x = F.silu(gate) * up
    res = x * probs
    return res.to(x.dtype)


def _clip_monitor_enabled(swiglu_limit: float) -> bool:
    # Must give the same answer in forward and act-ckpt recompute: fused_swiglu is
    # torch.compile'd and the graph with clip_counts saves more tensors for backward.
    return bool(global_actlog_state.enable_monitor and swiglu_limit > 0)


def _clip_monitor_reporting(swiglu_limit: float) -> bool:
    # EP all_reduce + logging only in the real forward; from the recompute it
    # deadlocks when some EP ranks have no tokens.
    if not _clip_monitor_enabled(swiglu_limit):
        return False
    monitor = global_actlog_state.monitor
    return monitor is None or getattr(monitor, "is_forward", True)


# TODO: move this to utils
def _get_init_gain(packed_param: nn.Parameter,
                   logical_shape: tp.Iterable[int],
                   ep_size: int,
                   split_dim: int) -> float:
    logical_shape = torch.Size(logical_shape)
    packed_shape = packed_param.size()
    assert len(packed_shape) == 2
    assert len(logical_shape) == 2

    # row parallel
    if split_dim == 0:
        full_packed_shape = packed_shape[0] * ep_size, packed_shape[1]
    elif split_dim == 1:
        full_packed_shape = packed_shape[0], packed_shape[1] * ep_size
    else:
        raise ValueError(f"Invalid split_dim: {split_dim}")

    return math.sqrt(sum(full_packed_shape) / sum(logical_shape))


class GroupedAbstractMLP(nn.Module, ABC):
    """Abstract base for grouped MLP experts. Subclasses set fused_permute and implement forward."""

    def __init__(self, config, ep_size: int) -> None:
        super().__init__()
        self.config = config
        self.num_local_experts = divide(self.config.num_routed_experts, ep_size)

        # up_moe / gate_moe hidden z-loss coefs (overwritten by the owning MoE
        # block via `_attach_included_proj_coefs`); default off so experts built
        # outside a block still have the attributes.
        self._up_moe_z_loss_coef = 0.0
        self._gate_moe_z_loss_coef = 0.0
        self._included_proj_z_loss = {}

        self.tp_size = 1
        assert self.tp_size == 1, "Tensor parallel is not supported for MoE"

        self.ep_size = ep_size

        self.hidden_size = self.config.hidden_size
        self.intermediate_size = self.config.intermediate_size
        intermediate_size = self.intermediate_size

        fc1_output_size = intermediate_size * 2 * self.num_local_experts
        fc1_output_size_per_partition = divide(fc1_output_size, self.tp_size)
        fc2_input_size = intermediate_size * self.num_local_experts
        fc2_input_size_per_partition = divide(fc2_input_size, self.tp_size)

        self.fc1_output_size_per_partition = fc1_output_size_per_partition
        self.fc2_input_size_per_partition = fc2_input_size_per_partition

        self._use_new_expert_weight_layout = config.use_new_expert_weight_layout
        if self._use_new_expert_weight_layout:
            weight1_shape = (self.num_local_experts, intermediate_size * 2, self.config.hidden_size,)
            weight2_shape = (self.num_local_experts, self.config.hidden_size, intermediate_size,)
        else:
            weight1_shape = (self.hidden_size, self.fc1_output_size_per_partition,)
            weight2_shape = (self.fc2_input_size_per_partition, self.hidden_size,)

        self.weight1 = nn.Parameter(torch.empty(weight1_shape))
        self.weight2 = nn.Parameter(torch.empty(weight2_shape))

        # after gigafsdp wrap, it deletes the weight attributes from the module
        # so we need to store them here for log in repr
        self._weight1_shape = self.weight1.shape
        self._weight2_shape = self.weight2.shape

        self._use_float8_grouped_gemm = self.config.use_float8_grouped_gemm
        self._float8_sparse_fused_swiglu_quant = self.config.float8_sparse_fused_swiglu_quant
        self._float8_triton_row2col = self.config.float8_triton_row2col
        self._swiglu_limit = getattr(self.config, "swiglu_limit", 0.0)

        # int32[2] buffers: [gate_clamped_count, up_clamped_count].
        # Filled by the forward kernel when ACTIVATION_MONITOR_ON=True.
        # persistent=False: excluded from state_dict (transient monitoring state).
        # int32 is safe because the counts are rank-local and bounded by the local
        # element count (~1e8); do not sum them across ranks in this dtype.
        self.register_buffer('clip_counts', torch.zeros(2, dtype=torch.int32), persistent=False)

        # Default activation-log name; the owning MoE block overrides it via
        # `attach_actlog_name` once it knows its layer index (this module is built
        # without one).  See `llmfoundry.models.layers.actlog_names`.
        attach_actlog_name(self, "block_sparse_moe", None)

        _fp8_num_sms = getattr(config, 'float8_deep_gemm_num_sms', None)
        self.handle1 = GroupedGemmFp8Wrapper(self.num_local_experts, num_sms=_fp8_num_sms) if self._use_float8_grouped_gemm else None
        self.handle2 = GroupedGemmFp8Wrapper(self.num_local_experts, num_sms=_fp8_num_sms) if self._use_float8_grouped_gemm else None

        self._deep_ep_enabled = self.config.moe_enable_deepep

        self._init_params = dict(init_method=nn.init.xavier_normal_, use_master_weight=config.use_master_weight, init_type=config.init_type)
        self._init_rules()


    def _init_rules(self):
        assert self.tp_size == 1, "Tensor parallel is not supported for MoE"
        if self.ep_size > 1:
            def raise_init_error(*args, **kwargs):
                raise RuntimeError("This module must be inited by it's parent reset_parameters")

            self.weight1.skip_init = True
            self.weight1.reset_parameters = raise_init_error

            self.weight2.skip_init = True
            self.weight2.reset_parameters = raise_init_error


    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        ...

    def _apply_moe_proj_z_loss(self, fc1_output: torch.Tensor) -> torch.Tensor:
        """Splice a hidden z-loss gradient into the up / gate halves of the
        fused gate+up grouped-GEMM output and stash the logging scalars.

        `fused_swiglu` splits the fc1 output as ``gate, up = chunk(x, 2, -1)``,
        so the first half is `gate_moe` and the second is `up_moe`. Pass-through
        (values unchanged); re-concatenated so downstream ops are unaffected.
        No-op when both coefs are 0 / in eval / on an empty (no-token) batch.
        """
        gate_coef = self._gate_moe_z_loss_coef
        up_coef = self._up_moe_z_loss_coef
        if (gate_coef == 0.0 and up_coef == 0.0) or fc1_output.nelement() == 0:
            return fc1_output
        gate, up = torch.chunk(fc1_output, 2, dim=-1)
        gate, gate_loss = apply_hidden_z_loss_hook(
            gate, gate_coef, training=self.training
        )
        up, up_loss = apply_hidden_z_loss_hook(
            up, up_coef, training=self.training
        )
        self._included_proj_z_loss = {"gate_moe": gate_loss, "up_moe": up_loss}
        return torch.cat([gate, up], dim=-1)

    def __repr__(self):
        return f"{self.__class__.__name__}(weight1_shape={self._weight1_shape}, weight2_shape={self._weight2_shape})"

    def _report_swiglu_clip_counts(self, fc1_output: torch.Tensor) -> None:
        """Report SwiGLU activation-clip counts / rates to the activation log.

        Shared between the FP8-quantised and plain (non-FP8) code paths.
        ``self.clip_counts`` must have been filled by the preceding
        ``fused_swiglu`` / ``fused_swiglu_quantized_fn`` call.

        Counts are summed across the EP group, so ``*_clip_count`` stays the
        layer-wide figure it has always been (comparable with earlier traces of
        a run).  The element total MUST be summed in the same collective over
        the same group: per-rank token counts are very uneven under DeepEP, so
        an EP-summed numerator over a ``local * ep_size`` denominator inflates
        the share by the load-imbalance factor (~2x on Ultra-700B).
        """
        if not _clip_monitor_reporting(self._swiglu_limit):
            return
        # fc1_output holds gate and up side by side, so half its elements are the
        # gate half the clamp scanned -- padding rows included, matching the
        # kernel's scan region.
        local_total = fc1_output.numel() // 2
        # int64: a per-rank count fits in the int32 buffer (bounded by the local
        # element count), the EP sum of ep_size of them need not.
        stats = torch.stack([
            self.clip_counts[0].to(torch.int64),
            self.clip_counts[1].to(torch.int64),
            torch.as_tensor(
                local_total, dtype=torch.int64, device=self.clip_counts.device
            ),
        ])
        if self.ep_size > 1:
            dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=get_ep_group())
        # clamp to 1 to avoid div-by-0 on a degenerate empty input; stay on
        # device (no .item()) so this adds no host sync per layer.
        total = stats[2].to(torch.float32).clamp_min(1.0)
        for idx, half in ((0, "gate"), (1, "up")):
            count = stats[idx].to(torch.float32)
            global_actlog_state.use_monitor_variable(
                count.unsqueeze(0),
                self._actlog_name,
                f"_{half}_clip_count",
                layer_depends=self._actlog_layer_depends)
            global_actlog_state.use_monitor_variable(
                (count / total).unsqueeze(0),
                self._actlog_name,
                f"_{half}_clip_share",
                layer_depends=self._actlog_layer_depends)

    def reset_parameters_giga(self):
        assert self.tp_size == 1, "Tensor parallel is not supported for MoE"
        hidden_size = self.config.hidden_size
        intermediate_size = self.config.intermediate_size
        if self._use_new_expert_weight_layout:
            packed_w1 = self.weight1.new_empty(hidden_size, self.fc1_output_size_per_partition)
            packed_w2 = self.weight2.new_empty(self.fc2_input_size_per_partition, hidden_size)
        else:
            packed_w1 = self.weight1
            packed_w2 = self.weight2
        weight1_gain = _get_init_gain(
            packed_w1,
            (hidden_size, intermediate_size),
            ep_size=self.ep_size,
            split_dim=1
        )
        weight2_gain = _get_init_gain(
            packed_w2,
            (intermediate_size, hidden_size),
            ep_size=self.ep_size,
            split_dim=0
        )

        if self.ep_size > 1:
            self._init_params["device"] = self.weight1.device
            assert torch.device('meta') != self._init_params["device"], (
                f"Module {self.__class__.__name__} must be moved to GPU from meta device before initialization"
            )

            if self._use_new_expert_weight_layout:
                w1 = torch.empty(
                    self.config.hidden_size, self.fc1_output_size_per_partition,
                    device=self._init_params["device"], dtype=self.weight1.dtype,
                )
            else:
                w1 = self.weight1

            _initialize_ep_weight(
                w1,
                input_size=self.fc1_output_size_per_partition * self.ep_size,
                output_size=self.hidden_size,
                partition_dim=1,
                per_partition_size=self.fc1_output_size_per_partition,
                gain=weight1_gain,
                return_master_weight=False,
                **self._init_params
            )
            if self._use_new_expert_weight_layout:
                with torch.no_grad():
                    self.weight1.copy_(
                        w1.view(self.config.hidden_size, self.num_local_experts, -1).permute(1, 2, 0)
                    )

            if self._use_new_expert_weight_layout:
                w2 = torch.empty(
                    self.fc2_input_size_per_partition, self.config.hidden_size,
                    device=self._init_params["device"], dtype=self.weight2.dtype,
                )
            else:
                w2 = self.weight2

            _initialize_ep_weight(
                w2,
                input_size=self.hidden_size,
                output_size=self.fc2_input_size_per_partition * self.ep_size,
                partition_dim=0,
                per_partition_size=self.fc2_input_size_per_partition,
                gain=weight2_gain,
                return_master_weight=False,
                **self._init_params
            )
            if self._use_new_expert_weight_layout:
                with torch.no_grad():
                    self.weight2.copy_(
                        w2.view(self.num_local_experts, -1, self.config.hidden_size).permute(0, 2, 1)
                    )

        else:
            self._init_params["init_method"](self.weight1, gain=weight1_gain)
            self._init_params["init_method"](self.weight2, gain=weight2_gain)

    def reset_parameters_deepseek(self):
        # Same as Giga init, but without `gain` parameter for normal init func.
        assert self.tp_size == 1, "Tensor parallel is not supported for MoE"
        if self.ep_size > 1:
            self._init_params["device"] = self.weight1.device
            assert torch.device('meta') != self._init_params["device"], (
                f"Module {self.__class__.__name__} must be moved to GPU from meta device before initialization"
            )

            if self._use_new_expert_weight_layout:
                w1 = torch.empty(
                    self.config.hidden_size, self.fc1_output_size_per_partition,
                    device=self._init_params["device"], dtype=self.weight1.dtype,
                )
            else:
                w1 = self.weight1

            _initialize_ep_weight(
                w1,
                input_size=self.fc1_output_size_per_partition * self.ep_size,
                output_size=self.hidden_size,
                partition_dim=1,
                per_partition_size=self.fc1_output_size_per_partition,
                return_master_weight=False,
                **self._init_params
            )

            if self._use_new_expert_weight_layout:
                with torch.no_grad():
                    self.weight1.copy_(
                        w1.view(self.config.hidden_size, self.num_local_experts, -1).permute(1, 2, 0)
                    )

            if self._use_new_expert_weight_layout:
                w2 = torch.empty(
                    self.fc2_input_size_per_partition, self.config.hidden_size,
                    device=self._init_params["device"], dtype=self.weight2.dtype,
                )
            else:
                w2 = self.weight2

            _initialize_ep_weight(
                w2,
                input_size=self.hidden_size,
                output_size=self.fc2_input_size_per_partition * self.ep_size,
                partition_dim=0,
                per_partition_size=self.fc2_input_size_per_partition,
                return_master_weight=False,
                **self._init_params
            )

            if self._use_new_expert_weight_layout:
                with torch.no_grad():
                    self.weight2.copy_(
                        w2.view(self.num_local_experts, -1, self.config.hidden_size).permute(0, 2, 1)
                    )

        else:
            self._init_params["init_method"](self.weight1)
            self._init_params["init_method"](self.weight2)

    def reset_parameters(self):
        # TODO(m1kol): Combine these init functions, make them more universal.
        # Now some params assume xavier init (gain).
        if self.config.init_type == "giga":
            self.reset_parameters_giga()
        elif self.config.init_type == "deepseek":
            self.reset_parameters_deepseek()
        else:
            raise ValueError(f'Expected `init_type` to be one of "giga" or "deepseek", got {self.config.init_type}.')


class GroupedLlamaMLP(GroupedAbstractMLP):
    def __init__(self, config, ep_size: int) -> None:
        self.fused_permute = False
        super().__init__(config, ep_size)

        self.rank_local_activation_observer = None
        self._monitor_activations = False
        self._per_expert_activation_norms = None
        self._monitor_weight_norms = False
        self._per_expert_weight_norms = None


    def compute_per_expert_activation_norms(self, intermediate, tokens_per_expert,
                                             total_tokens: int):
        """Compute per-expert accumulated squared-norm sums and token counts.
        """
        if total_tokens == 0:
            return (
                torch.zeros(self.num_local_experts, device=intermediate.device, dtype=intermediate.dtype),
                torch.zeros(self.num_local_experts, device=intermediate.device, dtype=intermediate.dtype),
            )
        device = intermediate.device
        # Use .pow(2).sum(-1) instead of .norm(-1).pow(2) to skip a redundant sqrt.
        token_sq_sums = intermediate[:total_tokens].pow(2).sum(dim=-1)
        expert_ids = torch.repeat_interleave(
            torch.arange(self.num_local_experts, device=device),
            tokens_per_expert.to(device).long(),
        )
        expert_sq_sums = torch.zeros(
            self.num_local_experts, device=device, dtype=token_sq_sums.dtype
        )
        expert_sq_sums.scatter_add_(0, expert_ids, token_sq_sums)

        tok_counts = tokens_per_expert.to(device=device, dtype=expert_sq_sums.dtype)
        return expert_sq_sums, tok_counts

    def activate_activation_monitor(self):
        self._monitor_activations = True
        self._acc_activation_sq_sums = None
        self._acc_activation_counts = None

    def get_activation_norms(self):
        """Return (sq_sums, counts) GPU tensors, or None if no data collected."""
        if self._acc_activation_sq_sums is None or self._acc_activation_counts is None:
            return None
        return self._acc_activation_sq_sums, self._acc_activation_counts

    def reset_activation_monitor(self):
        self._per_expert_activation_norms = None
        self._acc_activation_sq_sums = None
        self._acc_activation_counts = None
        self._monitor_activations = False

    def _compute_weight_norms(self, w1, w2):
        w1_norms = w1.flatten(1).norm(dim=1)
        w2_norms = w2.flatten(1).norm(dim=1)
        return w1_norms.tolist(), w2_norms.tolist()

    def activate_weight_norm_monitor(self):
        self._monitor_weight_norms = True

    def get_weight_norms(self):
        return self._per_expert_weight_norms

    def reset_weight_norm_monitor(self):
        self._per_expert_weight_norms = None


    def forward(self,
                permuted_local_hidden_states: torch.Tensor,
                tokens_per_expert: torch.Tensor,
                tokens_per_expert_cpu: torch.Tensor,
                tokens_per_expert_list: list[int],
                permuted_probs: torch.Tensor,
                weights_quantized: tp.Optional[tp.Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:

        # FP8 blockwise AG enabled
        giga_fp8_managed = bool(getattr(self, "_giga_fp8_managed_params", None))
        gathered_w1 = getattr(self, "_weight1_quantized", None)
        gathered_w2 = getattr(self, "_weight2_quantized", None)
        if giga_fp8_managed:
            if weights_quantized is not None:
                raise RuntimeError("")
            if gathered_w1 is None or gathered_w2 is None:
                raise RuntimeError("")

        # Reset the per-forward up/gate hidden z-loss stash (cleared here so an
        # empty-batch / disabled forward does not log a stale value).
        self._included_proj_z_loss = {}

        monitor_clips = _clip_monitor_enabled(self._swiglu_limit)
        monitored_intermediate = None
        if monitor_clips:
            self.clip_counts.zero_()

        if giga_fp8_managed and not (self._use_float8_grouped_gemm and (self.ep_size == 1 or self._deep_ep_enabled)):
            raise RuntimeError(
                "giga_fsdp fp8 all-gather experts require the float8 grouped-gemm path " +
                "(use_float8_grouped_gemm with ep_size == 1 or DeepEP); the bf16 fallback " +
                "would compute on the NaN carrier."
            )

        if self._use_float8_grouped_gemm and (self.ep_size == 1 or self._deep_ep_enabled) and (
                self.training or weights_quantized is not None or giga_fp8_managed):
            assert self._use_new_expert_weight_layout, "Old expert weight layout is not supported for float8 training."
            w1, w2 = self.weight1, self.weight2

            # TODO (fedorovgv): remove duplicate here! we can hold it in memory
            groups_padded_sizes_list = [(m + 127) // 128 * 128 for m in tokens_per_expert_list]
            groups_padded_sizes_tensor = torch.tensor(
                groups_padded_sizes_list,
                dtype=torch.int32,
                device=permuted_local_hidden_states.device)
            m_indices = build_m_indices(groups_padded_sizes_list, groups_padded_sizes_tensor)

            # correct padding in fp8 blockwise ag scenarion for eval
            needs_local_pad = (
                (weights_quantized is not None or giga_fp8_managed)
                and permuted_local_hidden_states.shape[0] != sum(groups_padded_sizes_list)
            )
            if needs_local_pad:
                assert not self.training, (
                    "Local expert-block padding is an eval-only path; during training "
                    "the dispatcher already pads tokens to a multiple of 128."
                )
                orig_num_tokens = permuted_local_hidden_states.shape[0]
                permuted_local_hidden_states, permuted_probs, u2p_index = (
                    pad_permuted_for_grouped_gemm(
                        permuted_local_hidden_states,
                        permuted_probs,
                        tokens_per_expert_list,
                        groups_padded_sizes_list,
                    )
                )

            global_actlog_state.use_monitor_variable(
                permuted_local_hidden_states,
                self._actlog_name + ".gmm_upgate",
                "_input.0", layer_depends=self._actlog_layer_depends)
            assert self.handle1 is not None, "GroupedGemmFp8Wrapper is not available!"
            fc1_output = self.handle1.group_gemm(
                tensor_groups=permuted_local_hidden_states,
                weights=w1,
                weights_quantized=weights_quantized[0] if weights_quantized is not None else None,
                weights_quantized_gathered=gathered_w1,
                tensor_groups_sizes=groups_padded_sizes_list,
                tensor_groups_sizes_padded=groups_padded_sizes_tensor,
                m_indices=m_indices,
                triton_row2col=self._float8_triton_row2col,
            )
            global_actlog_state.use_monitor_variable(
                fc1_output,
                self._actlog_name + ".fused_swiglu",
                "_input.0", layer_depends=self._actlog_layer_depends)

            fc1_output = self._apply_moe_proj_z_loss(fc1_output)

            if self._float8_sparse_fused_swiglu_quant:
                monitor_clips = _clip_monitor_enabled(self._swiglu_limit)
                if monitor_clips:
                    self.clip_counts.zero_()

                intermediate = fused_swiglu_quantized_fn(
                    fc1_output, permuted_probs,
                    swiglu_limit=self._swiglu_limit,
                    clip_counts=self.clip_counts if monitor_clips else None,
                )

                if _clip_monitor_reporting(self._swiglu_limit):
                    # Sum across EP group; every EP rank now holds the layer-wide
                    # total. Monitor's amax over the subsequent all_gather returns
                    # this value directly (all EP ranks agree on the same number).
                    dist.all_reduce(self.clip_counts, op=dist.ReduceOp.SUM, group=get_ep_group())
                    # Approximate global per-half element total = (local elements) * ep_size.
                    # Exact when token routing is balanced across EP ranks; close enough for
                    # monitoring purposes (we just want a fraction, not a fully exact ratio).
                    total = float(max((fc1_output.numel() // 2) * self.ep_size, 1))
                    gate_share = self.clip_counts[0].float() / total
                    up_share = self.clip_counts[1].float() / total
                    global_actlog_state.use_monitor_variable(
                        self.clip_counts[0].float().unsqueeze(0),
                        "model.model.layers.{}.block_sparse_moe",
                        "_gate_clip_count")
                    global_actlog_state.use_monitor_variable(
                        gate_share.unsqueeze(0),
                        "model.model.layers.{}.block_sparse_moe",
                        "_gate_clip_share")
                    global_actlog_state.use_monitor_variable(
                        self.clip_counts[1].float().unsqueeze(0),
                        "model.model.layers.{}.block_sparse_moe",
                        "_up_clip_count")
                    global_actlog_state.use_monitor_variable(
                        up_share.unsqueeze(0),
                        "model.model.layers.{}.block_sparse_moe",
                        "_up_clip_share")
            else:
                permuted_probs = permuted_probs.unsqueeze(-1)
                intermediate = fused_swiglu(
                    fc1_output, permuted_probs,
                    swiglu_limit=self._swiglu_limit,
                    clip_counts=self.clip_counts if monitor_clips else None,
                )

            self._report_swiglu_clip_counts(fc1_output)

            if global_actlog_state.enable_monitor or global_actlog_state.activation_observer is not None:
                monitored_intermediate = intermediate
                if self._float8_sparse_fused_swiglu_quant:
                    assert isinstance(intermediate, Float8BlockwiseQTensor) \
                            and intermediate._input_layout_type == InputLayoutType.ROWWISE, "Rowwise input layout is required for monitoring."
                    monitored_intermediate = fp8_dequantize(intermediate._data, intermediate._scale)
                global_actlog_state.use_monitor_variable(
                    monitored_intermediate,
                    self._actlog_name + ".gmm_down",
                    "_input.0", layer_depends=self._actlog_layer_depends)   

            assert self.handle2 is not None, "GroupedGemmFp8Wrapper is not available!"
            fc2_output = self.handle2.group_gemm(
                tensor_groups=intermediate,
                weights=w2,
                weights_quantized=weights_quantized[1] if weights_quantized is not None else None,
                weights_quantized_gathered=gathered_w2,
                tensor_groups_sizes=groups_padded_sizes_list,
                tensor_groups_sizes_padded=groups_padded_sizes_tensor,
                m_indices=m_indices,
                triton_row2col=self._float8_triton_row2col,
            )

            # Undo the eval-only local padding applied before fc1: gather the
            # real (non-padding) rows back into the compact per-expert layout.
            if needs_local_pad:
                assert not self.training, (
                    "Local expert-block padding is an eval-only path; during training "
                    "the dispatcher already pads tokens to a multiple of 128."
                )
                fc2_output = unpad_grouped_gemm_output(
                    fc2_output, u2p_index, orig_num_tokens)
        else:
            if self._use_new_expert_weight_layout:
                w1 = self.weight1.view(self.num_local_experts, -1, self.config.hidden_size)
                w2 = self.weight2.view(self.num_local_experts, self.config.hidden_size, -1)
                trans_b = True
            else:
                w1 = self.weight1.view(self.num_local_experts, self.config.hidden_size, -1)
                w2 = self.weight2.view(self.num_local_experts, -1, self.config.hidden_size)
                trans_b = False

            global_actlog_state.use_monitor_variable(
                permuted_local_hidden_states,
                self._actlog_name + ".gmm_upgate",
                "_input.0", layer_depends=self._actlog_layer_depends)
            fc1_output = gg.ops.gmm(permuted_local_hidden_states, w1, tokens_per_expert_cpu, trans_b=trans_b)
            global_actlog_state.use_monitor_variable(
                fc1_output,
                self._actlog_name + ".fused_swiglu",
                "_input.0", layer_depends=self._actlog_layer_depends)

            fc1_output = self._apply_moe_proj_z_loss(fc1_output)

            intermediate = fused_swiglu(
                fc1_output, permuted_probs.unsqueeze(-1),
                swiglu_limit=self._swiglu_limit,
                clip_counts=self.clip_counts if monitor_clips else None,
            )

            self._report_swiglu_clip_counts(fc1_output)

            global_actlog_state.use_monitor_variable(
                intermediate,
                self._actlog_name + ".gmm_down",
                "_input.0", layer_depends=self._actlog_layer_depends)
            fc2_output = gg.ops.gmm(intermediate, w2, tokens_per_expert_cpu, trans_b=trans_b)

        global_actlog_state.use_monitor_variable(
            fc2_output,
            self._actlog_name + ".gmm_down",
            "_output.0", layer_depends=self._actlog_layer_depends)
        
        if self.rank_local_activation_observer is not None and self.training:
            self.rank_local_activation_observer(intermediate, tokens_per_expert_list)

        if self._monitor_activations:
            with torch.no_grad():
                total_tokens = int(tokens_per_expert_cpu.sum().item())
                if total_tokens > 0:
                    # tokens_per_expert may be None when ep_size=1 (non-DeepEP
                    # dispatch); fall back to tokens_per_expert_cpu.
                    tok_counts = tokens_per_expert if tokens_per_expert is not None else tokens_per_expert_cpu
                    # Float8BlockwiseQTensor.dequantize() has a reshape bug; use
                    # the Triton fp8_dequantize kernel instead (same as
                    # global_actlog_state).
                    if isinstance(intermediate, Float8BlockwiseQTensor):
                        inter_for_norms = monitored_intermediate
                        if inter_for_norms is None:
                            inter_for_norms = fp8_dequantize(
                                intermediate._data,
                                intermediate._scale,
                            )
                    else:
                        inter_for_norms = intermediate
                    sq_sums, counts = self.compute_per_expert_activation_norms(
                        inter_for_norms, tok_counts, total_tokens,
                    )
                else:
                    sq_sums = torch.zeros(
                        self.num_local_experts, device=intermediate.device,
                        dtype=intermediate.dtype,
                    )
                    counts = torch.zeros(
                        self.num_local_experts, device=intermediate.device,
                        dtype=intermediate.dtype,
                    )
                if self._acc_activation_sq_sums is None:
                    self._acc_activation_sq_sums = sq_sums
                    self._acc_activation_counts = counts
                else:
                    self._acc_activation_sq_sums += sq_sums
                    self._acc_activation_counts += counts

        if self._monitor_weight_norms:
            if giga_fp8_managed:
                self._per_expert_weight_norms = None
                self._monitor_weight_norms = False
            else:
                with torch.no_grad():
                    if self.training and self._use_float8_grouped_gemm and self._deep_ep_enabled:
                        w1 = self.weight1.view(self.num_local_experts, -1, self.config.hidden_size)
                        w2 = self.weight2.view(self.num_local_experts, self.config.hidden_size, -1)
                    self._per_expert_weight_norms = self._compute_weight_norms(w1, w2)
                    self._monitor_weight_norms = False

        return fc2_output


if moe_general_routing_inputs is None or ActivationType is None:
    class GroupedSonicMLP(GroupedAbstractMLP):
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "GroupedSonicMLP requires the sonicmoe package and a CUDA GPU "
                "with compute capability >= 9.0 (Hopper sm90)."
            )

        def forward(self, *args, **kwargs):
            raise RuntimeError(
                "GroupedSonicMLP requires the sonicmoe package and a CUDA GPU "
                "with compute capability >= 9.0 (Hopper sm90)."
            )
else:
    class GroupedSonicMLP(GroupedAbstractMLP):
        def __init__(self, config, ep_size: int) -> None:
            require_sonic_moe_cuda_device()
            assert not config.use_float8_grouped_gemm, (
                "GroupedSonicMLP does not support float8 grouped GEMM. "
                "Float8 is only supported for dispatch via DeepEP, not for SonicMoE experts."
            )
            assert not config.use_new_expert_weight_layout, (
                "GroupedSonicMLP does not support use_new_expert_weight_layout."
            )
            self.fused_permute = True
            self.intermediate_size = config.intermediate_size
            super().__init__(config, ep_size)
            self.activation_function = ActivationType.SWIGLU
            self.stream_id = torch.cuda.current_stream().cuda_stream

        def forward(
            self,
            local_hidden_states: torch.Tensor,
            token_idxs: torch.Tensor,
            expert_idxs: torch.Tensor,
            expert_scores: torch.Tensor,
        ) -> torch.Tensor:

            # weight1: (H, 2*I*E) -> view(E, 2*I, H) -> permute(1,2,0) -> (2*I, H, E)
            #   strides: (2*I*H, H, 1)  -> (H, 1, 2*I*H) -- strides[1]=1 ✓
            # weight2: (I*E, H)  -> view(E, H, I)   -> permute(1,2,0) -> (H, I, E)
            #   strides: (H*I, I, 1)    -> (I, 1, H*I)   -- strides[1]=1 ✓
            w1 = self.weight1.view(
                self.num_local_experts, 2 * self.intermediate_size, self.hidden_size
            ).permute(1, 2, 0)
            w2 = self.weight2.view(
                self.num_local_experts, self.hidden_size, self.intermediate_size
            ).permute(1, 2, 0)

            output_hidden_states, _ = moe_general_routing_inputs(
                x=local_hidden_states,
                router_scores=expert_scores.view(-1),
                token_indices=token_idxs.view(-1),
                expert_indices=expert_idxs.view(-1),
                w1=w1,
                b1=None,
                w2=w2,
                b2=None,
                E=self.num_local_experts,
                stream_id=self.stream_id,
                activation_type=self.activation_function,
                is_inference_mode_enabled=False
            )
            return output_hidden_states



# shared expert computation can be overlapped with MoE dispatch
# a separate class facilitates this
# inspired by https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/shared_experts.py
class SharedExpertLlamaMLP(LlamaMLP):
    # The shared expert is a child of `self_attn` in the module tree (so the MoE
    # dispatch can overlap with it); match the name the forward hooks and
    # `blocks.py` already use for its output instead of reporting as `.mlp`,
    # which is the *dense* MLP and only exists on the first_k_dense layers.
    _actlog_base = "self_attn.shared_experts"

    def __init__(self, config, layer_idx: int | None = None) -> None:
        ffn_dim = config.intermediate_size * config.n_shared_experts
        super().__init__(
            config,
            intermediate_size=ffn_dim,
            fused_mlp_checkpoint_lvl=config.fused_mlp_checkpoint_lvl,
            layer_idx=layer_idx,
        )


class SharedExpertParallelLlamaMLP(ParallelMLP):
    # See `SharedExpertLlamaMLP._actlog_base`.
    _actlog_base = "self_attn.shared_experts"

    def __init__(self, config, layer_idx: int | None = None) -> None:
        ffn_dim = config.intermediate_size * config.n_shared_experts
        super().__init__(
            config,
            intermediate_size=ffn_dim,
            fused_mlp_checkpoint_lvl=config.fused_mlp_checkpoint_lvl,
            layer_idx=layer_idx,
        )

class ScMoEBlockGeMM(nn.Module):
    def __init__(self, config, token_dispatcher) -> None:
        super().__init__()

        self.config = config

        ep_size = get_ep_group_size()
        if ep_size is None:
            ep_size = 1

        self.experts = GroupedLlamaMLP(config, ep_size)
        self.token_dispatcher = token_dispatcher

    def forward(
        self,
        recv_x,
        dispatched_indices,
        dispatched_probs,
        tokens_per_expert,
        ctx,
        is_float8_training: bool = False,
        weights_quantized: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        (
            permuted_hidden_states,
            tokens_per_expert_tensor_gpu,
            tokens_per_expert_tensor_cpu,
            tokens_per_expert_list,
            permuted_probs,
        ) = self.token_dispatcher.post_dispatch_scmoe(
            recv_x,
            dispatched_indices,
            dispatched_probs,
            tokens_per_expert,
            ctx,
            is_float8_training=is_float8_training,
        )
        expert_output = self.experts(
            permuted_hidden_states,
            tokens_per_expert_tensor_gpu,
            tokens_per_expert_tensor_cpu,
            tokens_per_expert_list,
            permuted_probs,
            weights_quantized,
        )
        if is_float8_training:
            recv_order_hidden = unpad_and_unpermute_fn(
                expert_output,
                ctx.deepep_reversed_mapping_for_combine,
                ctx.deepep_reversed_mapping_for_combine_padded,
                ctx.deepep_recv_hidden_shape,
                ctx.deepep_pad_starts,
                ctx.deepep_pad_counts,
            )
        else:
            recv_order_hidden = fused_unpermute(
                expert_output,
                ctx.deepep_reversed_mapping_for_combine,
                merging_probs=None,
                restore_shape=ctx.deepep_recv_hidden_shape,
            )
        return recv_order_hidden
