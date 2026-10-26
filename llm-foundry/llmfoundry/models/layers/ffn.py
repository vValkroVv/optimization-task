# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""GPT Blocks used for the GPT Model."""

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from composer.utils import dist, global_actlog_state
from llmfoundry.models.layers.actlog_names import attach_actlog_name
from llmfoundry.models.ops.float8.cuda_gemm import FP8Linear
from llmfoundry.models.ops.hidden_z_loss_hook import apply_hidden_z_loss_hook
from llmfoundry.models.ops.included_proj_z_loss import resolve_included_proj_coef
from llmfoundry.models.layers.activations import ACT2FN
from llmfoundry.models.layers.fc import FC_CLASS_REGISTRY
from llmfoundry.models.parallel.tensor import (
    copy_to_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region)


log = logging.getLogger(__name__)


def _init_dense_mlp_proj_z_loss(module, config, layer_idx):
    """Resolve the dense-MLP `gate_mlp`/`up_mlp` hidden z-loss coefs for this
    layer and initialise the logging stash. Call from an MLP `__init__`."""
    module._gate_mlp_z_loss_coef = resolve_included_proj_coef(
        config, "gate_mlp", layer_idx
    )
    module._up_mlp_z_loss_coef = resolve_included_proj_coef(
        config, "up_mlp", layer_idx
    )
    module._included_proj_z_loss = {}


def _apply_dense_mlp_proj_z_loss(module, gate_out, up_out):
    """Splice the hidden z-loss gradient into the `gate_proj` / `up_proj`
    outputs (pass-through) and stash the logging scalars on the module. No-op
    when the coefs are 0 / in eval. Returns the (possibly wrapped) tensors."""
    gate_out, gate_loss = apply_hidden_z_loss_hook(
        gate_out, module._gate_mlp_z_loss_coef, training=module.training
    )
    up_out, up_loss = apply_hidden_z_loss_hook(
        up_out, module._up_mlp_z_loss_coef, training=module.training
    )
    module._included_proj_z_loss = {"gate_mlp": gate_loss, "up_mlp": up_loss}
    return gate_out, up_out


class ParallelMLP(torch.nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    # See `LlamaMLP._actlog_base`.
    _actlog_base = "mlp"

    def __init__(
        self,
        config,
        intermediate_size=None,
        fused_mlp_checkpoint_lvl=None,
        layer_idx: int | None = None,
    ):
        super(ParallelMLP, self).__init__()

        self.layer_idx = layer_idx
        attach_actlog_name(self, type(self)._actlog_base, layer_idx)
        _init_dense_mlp_proj_z_loss(self, config, layer_idx)
        self.hidden_size = config.hidden_size

        self.add_bias = False

        self.intermediate_size = intermediate_size or config.intermediate_size

        self.gate_proj = FC_CLASS_REGISTRY["ColumnParallelLinear"](
            config.hidden_size,
            self.intermediate_size,
            config=config,
            bias=self.add_bias,
            gather_output=False,
        )
        self.up_proj = FC_CLASS_REGISTRY["ColumnParallelLinear"](
            config.hidden_size,
            self.intermediate_size,
            config=config,
            bias=self.add_bias,
            gather_output=False,
        )

        self._is_fused = config.fused_mlp
        if self._is_fused:
            assert not self.add_bias, "bias are not supported"
            from llmfoundry.models.ops.mlp import fused_mlp_func

            self._fused_func = fused_mlp_func

        self._checkpoint_lvl = (
            fused_mlp_checkpoint_lvl or config.fused_mlp_checkpoint_lvl
        )

        def swiglu(gate_proj, up_proj):
            return torch.nn.functional.silu(gate_proj) * up_proj

        self.activation_func = swiglu

        self.down_proj = FC_CLASS_REGISTRY["RowParallelLinear"](
            self.intermediate_size,
            config.hidden_size,
            config=config,
            bias=self.add_bias,
            input_is_parallel=True,
        )

        self.down_proj._is_residual = True

        if config.init_type in {"dclm", "olmo3"}:
            if config.init_type == "dclm" and self.layer_idx is None:
                raise RuntimeError(
                    "DCLM init depends on layer's depth, but got `layer_idx=None`."
                )

            init_std = 1 / math.sqrt(self.hidden_size)
            self.gate_proj._init_std = init_std
            self.up_proj._init_std = init_std

            if config.init_type == "dclm":
                # dclm init style
                if self.layer_idx != -1:
                    init_std = (
                        1
                        / math.sqrt(self.intermediate_size)
                        / math.sqrt(2 * (self.layer_idx + 1))
                    )
                else:
                    log.info(
                        "`ParallelMLP` got `layer_idx=-1` with `init_type=dclm`. Skipping std depth scaling for `down_proj`."
                    )
            else:
                # olmo3 init style
                init_std = init_std / math.sqrt(2 * config.num_hidden_layers)

            self.down_proj._init_std = init_std

    def forward(
        self,
        hidden_states: torch.Tensor,
        activation_checkpointing_on_layer: bool = False,
        logical_batch_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self._is_fused:
            input_parallel = copy_to_tensor_model_parallel_region(hidden_states)
            output_parallel = self._fused_func(
                x=input_parallel,
                weight1=self.gate_proj._tp_linear_submodule.weight,
                weight2=self.up_proj._tp_linear_submodule.weight,
                weight3=self.down_proj._tp_linear_submodule.weight,
                # we fully recompute the activations, so no need to checkpoint them
                checkpoint_lvl=0
                if activation_checkpointing_on_layer
                else self._checkpoint_lvl,
                actlog_name=self._actlog_name,
                actlog_layer_depends=self._actlog_layer_depends,
            )
            output = reduce_from_tensor_model_parallel_region(output_parallel)
            return output

        tp_kwargs = (
            dict(logical_batch_size=logical_batch_size)
            if logical_batch_size is not None
            else dict()
        )

        global_actlog_state.use_monitor_variable(
            hidden_states,
            self._actlog_name + ".up_proj",
            "_input.0",
            layer_depends=self._actlog_layer_depends,
        )
        if self._checkpoint_lvl and not activation_checkpointing_on_layer:
            output = checkpoint.checkpoint(
                self.swiglu_fn, hidden_states, tp_kwargs, use_reentrant=False
            )
        else:
            output = self.swiglu_fn(hidden_states, tp_kwargs)

        return output

    def swiglu_fn(self, x, tp_kwargs):
        x1 = self.gate_proj(x, **tp_kwargs)
        x2 = self.up_proj(x, **tp_kwargs)
        x1, x2 = _apply_dense_mlp_proj_z_loss(self, x1, x2)
        return self.down_proj(self.activation_func(x1, x2))


class LlamaMLP(nn.Module):
    # Name this module reports itself under in the activation log; subclasses
    # override it so they stop masquerading as the dense MLP.
    _actlog_base = "mlp"

    def __init__(
        self,
        config,
        intermediate_size=None,
        fused_mlp_checkpoint_lvl=None,
        layer_idx: int | None = None,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        attach_actlog_name(self, type(self)._actlog_base, layer_idx)
        _init_dense_mlp_proj_z_loss(self, config, layer_idx)

        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            device=config.init_device,
        )
        self.up_proj = nn.Linear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            device=config.init_device,
        )
        self.down_proj = nn.Linear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            device=config.init_device,
        )
        self.act_fn = ACT2FN[config.hidden_act]

        self._is_fused = config.fused_mlp
        if self._is_fused:
            from llmfoundry.models.ops.mlp import fused_mlp_func

            self._fused_func = fused_mlp_func

        self._sep_product_silu_quant = None
        self._swiglu_limit = getattr(config, "swiglu_limit", 0.0)
        if getattr(config, "float8_dense_fused_swiglu_quant", False):
            from llmfoundry.models.ops.float8._cudacherry import swiglu_quantize
            self._sep_product_silu_quant = swiglu_quantize
        if self._swiglu_limit > 0:
            self.register_buffer(
                'clip_counts', torch.zeros(2, dtype=torch.int32), persistent=False
            )

        self._checkpoint_lvl = fused_mlp_checkpoint_lvl or config.fused_mlp_checkpoint_lvl
        self.down_proj._is_residual = True

        self.down_proj._is_residual = True

        if config.init_type in {"dclm", "olmo3"}:
            if config.init_type == "dclm" and self.layer_idx is None:
                raise RuntimeError(
                    "DCLM init depends on layer's depth, but got `layer_idx=None`."
                )

            init_std = 1 / math.sqrt(self.hidden_size)
            self.gate_proj._init_std = init_std
            self.up_proj._init_std = init_std

            if config.init_type == "dclm":
                # dclm init style
                if self.layer_idx != -1:
                    init_std = (
                        1
                        / math.sqrt(self.intermediate_size)
                        / math.sqrt(2 * (self.layer_idx + 1))
                    )
                else:
                    log.info(
                        "`LlamaMLP` got `layer_idx=-1` with `init_type=dclm`. Skipping std depth scaling for `down_proj`."
                    )
            else:
                # olmo3 init style
                init_std = init_std / math.sqrt(2 * config.num_hidden_layers)

            self.down_proj._init_std = init_std

    def forward(self, x, residual=None, activation_checkpointing_on_layer=False):
        if getattr(self.config, "float8_dense_fused_swiglu_quant", False):
            # assert not self._checkpoint_lvl > 0, "checkpoint_lvl is not implemented for FP8"
            assert self._is_fused is False, "fused_mlp=True and float8_dense_fused_swiglu_quant=True not compatible"
            assert all(isinstance(m, FP8Linear) for m in \
                    (self.gate_proj, self.up_proj, self.down_proj)), "gate_proj, up_proj, down_proj must be FP8Linear"
            if self._checkpoint_lvl and not activation_checkpointing_on_layer:
                down_proj = checkpoint.checkpoint(
                        self.float8_dense_fused_swiglu_quant_fn, x, residual, use_reentrant=False
                )
            else:
                down_proj = self.float8_dense_fused_swiglu_quant_fn(x, residual)
            return down_proj

        if self._is_fused:
            assert self._swiglu_limit <= 0, (
                "fused_mlp=True and swiglu_limit>0 are not compatible: "
                "the fused MLP kernel does not support SwiGLU activation clipping"
            )
            return self._fused_func(
                x=x,
                weight1=self.gate_proj.weight,
                weight2=self.up_proj.weight,
                weight3=self.down_proj.weight,
                # we fully recompute the activations, so no need to checkpoint them
                checkpoint_lvl=0
                if activation_checkpointing_on_layer
                else self._checkpoint_lvl,
                actlog_name=self._actlog_name,
                actlog_layer_depends=self._actlog_layer_depends,
            )

        monitor_clips = global_actlog_state.enable_monitor and self._swiglu_limit > 0 and hasattr(self, 'clip_counts')
        if monitor_clips:
            self.clip_counts.zero_()

        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [
                    F.linear(x, gate_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ],
                dim=-1,
            )
            up_proj = torch.cat(
                [
                    F.linear(x, up_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ],
                dim=-1,
            )

            gate_proj, up_proj = _apply_dense_mlp_proj_z_loss(self, gate_proj, up_proj)

            if self._swiglu_limit > 0:
                if hasattr(self, 'clip_counts'):
                    self.clip_counts[0].add_(gate_proj.ge(self._swiglu_limit).sum().to(torch.int32))
                    self.clip_counts[1].add_(
                        up_proj.le(-self._swiglu_limit).logical_or(up_proj.ge(self._swiglu_limit)).sum().to(torch.int32))
                gate_proj = gate_proj.clamp(max=self._swiglu_limit)
                up_proj = up_proj.clamp(-self._swiglu_limit, self._swiglu_limit)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
            clip_total = gate_proj.numel()
        else:
            global_actlog_state.use_monitor_variable(
                x,
                self._actlog_name + ".up_proj",
                "_input.0",
                layer_depends=self._actlog_layer_depends,
            )
            if self._checkpoint_lvl and not activation_checkpointing_on_layer:
                down_proj = checkpoint.checkpoint(
                    self.swiglu_fn, x, use_reentrant=False
                )
            else:
                down_proj = self.swiglu_fn(x)
            # One gate (and one up) element per (token, intermediate) pair.  `x`
            # is (batch, seq, hidden) in the dense/shared-expert path and 2-D
            # when packed, so derive the token count from every dim but the last
            # instead of assuming dim 0 is it.
            clip_total = (x.numel() // x.shape[-1]) * self.intermediate_size

        if monitor_clips:
            self._log_swiglu_clip_metrics(clip_total)

        return down_proj

    def _log_swiglu_clip_metrics(self, clip_total: int) -> None:
        """Report SwiGLU activation-clip counts / rates to the activation log.

        ``clip_total`` is the number of gate (== number of up) elements the
        clamp scanned on this rank, so the share is this rank's local clip rate.
        The monitor reduces across ranks with the configured stat, i.e. with the
        default ``stat_list: [amax]`` the logged value is the worst rank's rate.
        Keep numerator and denominator in the same (rank-local) scope.
        """
        total = float(max(clip_total, 1))
        for idx, half in ((0, "gate"), (1, "up")):
            count = self.clip_counts[idx].float()
            global_actlog_state.use_monitor_variable(
                count.unsqueeze(0),
                self._actlog_name,
                f"_{half}_clip_count",
                layer_depends=self._actlog_layer_depends,
            )
            global_actlog_state.use_monitor_variable(
                (count / total).unsqueeze(0),
                self._actlog_name,
                f"_{half}_clip_share",
                layer_depends=self._actlog_layer_depends,
            )

    def swiglu_fn(self, x):
        gate_out = self.gate_proj(x)
        up_out = self.up_proj(x)
        gate_out, up_out = _apply_dense_mlp_proj_z_loss(self, gate_out, up_out)
        if self._swiglu_limit > 0:
            if hasattr(self, 'clip_counts'):
                self.clip_counts[0].add_(gate_out.ge(self._swiglu_limit).sum().to(torch.int32))
                self.clip_counts[1].add_(
                    up_out.le(-self._swiglu_limit).logical_or(up_out.ge(self._swiglu_limit)).sum().to(torch.int32))
            gate_out = gate_out.clamp(max=self._swiglu_limit)
            up_out = up_out.clamp(-self._swiglu_limit, self._swiglu_limit)
        return self.down_proj(self.act_fn(gate_out) * up_out)

    def float8_dense_fused_swiglu_quant_fn(self, x, residual = None):
        gate_out = self.gate_proj(x, do_unpad_out=False)
        up_out = self.up_proj(x, do_unpad_out=False)
        gate_out, up_out = _apply_dense_mlp_proj_z_loss(self, gate_out, up_out)

        monitor_clips = global_actlog_state.enable_monitor and self._swiglu_limit > 0
        if monitor_clips:
            self.clip_counts.zero_()

        prod_silu_out = self._sep_product_silu_quant(
            gate_out, up_out,
            torch.float8_e4m3fn,
            True,   # fp8_input_store
            True,   # power_two_max_round
            self._swiglu_limit,
            self.clip_counts if monitor_clips else None,
        )
        base_shape = getattr(x, "orig_shape", x.shape)
        prod_silu_out.orig_shape = (*base_shape[:-1], prod_silu_out.shape[-1])

        if monitor_clips:
            # `gate_out` is the projection output before unpad (do_unpad_out=False),
            # which is exactly the region the clamp kernel scans -- so numerator and
            # denominator both include the padding rows and stay consistent.
            self._log_swiglu_clip_metrics(gate_out.numel())

        return self.down_proj(prod_silu_out, residual=residual)


FFN_CLASS_REGISTRY = {"ParallelMLP": ParallelMLP, "LlamaMLP": LlamaMLP}


def build_ffn(
    tp_size: int,
    config: Any,
    layer_idx: int | None = None,
) -> LlamaMLP | ParallelMLP:
    ffn_type = (
        FFN_CLASS_REGISTRY["LlamaMLP"]
        if tp_size == 1
        else FFN_CLASS_REGISTRY["ParallelMLP"]
    )
    return ffn_type(
        config,
        intermediate_size=config.dense_intermediate_size,
        fused_mlp_checkpoint_lvl=config.dense_fused_mlp_checkpoint_lvl,
        layer_idx=layer_idx,
    )
