# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Optimizers with weight decay decoupled from the learning rate.

These optimizers are based off of `Decoupled Weight Decay Regularization <https://arxiv.org/abs/1711.05101>`_, which
proposes this decoupling. In general, it is recommended to use these optimizers over their native PyTorch equivalents.
"""

from __future__ import annotations

from collections.abc import Mapping
import logging
import math
import re
from typing import Iterable, List, Optional, Tuple, Union

import torch
from torch.optim import SGD, AdamW
from torch.optim.optimizer import required  # type: ignore

from composer.functional.triton_kernels import (
    adam_metric_sums_batch, adamw_cautious_parameter_update, float32_reciprocal,
)
from composer.optim.utils import (
    IProcessGroupEnabled,
    convert_dtype,
    create_denoms,
    optimized_mul,
    optimized_mul_per_param,
    update_moments,
    update_params,
)
from composer.utils import dist


log = logging.getLogger(__name__)

__all__ = ['DecoupledSGDW', 'DecoupledAdamW', 'DecoupledFusedAdamW']


def _report_grad_nan_origin() -> None:
    """Ask GigaFSDP which rank's gradients were already non-finite before reduce_scatter.

    The count logged above is taken from gradient SHARDS, which are sums over the whole
    group, so it cannot name the node -- the shard lands on its owner, not on the producer.
    The first skip only arms the pre-reduce check; from the next one it names the rank.
    No-op when GigaFSDP is not in use.
    """
    try:
        from giga_fsdp.giga_fsdp import report_grad_nan_origin_and_arm
    except ImportError:
        return
    report_grad_nan_origin_and_arm()


class DecoupledSGDW(SGD):
    """SGD optimizer with the weight decay term decoupled from the learning rate.

    NOTE: Since `weight_decay` is no longer scaled by `lr`, you will likely want to use much smaller values
    for `weight_decay` than you would if using `torch.optim.SGD`. In this optimizer, the value `weight_decay` translates exactly to:
    'On every optimizer update, every weight element will be multiplied by `(1.0 - weight_decay_t)`'.
    The term `weight_decay_t` will follow the same schedule as `lr_t` but crucially will not be scaled by `lr`.

    Argument defaults are copied from :class:`torch.optim.SGD`.

    Why use this optimizer? The standard `SGD <https://pytorch.org/docs/stable/generated/torch.optim.SGD.html?highlight=sgd#torch.optim.SGD>`_
    optimizer couples the weight decay term with the gradient calculation. This ties the optimal value
    of :attr:`weight_decay` to :attr:`lr` and can also hurt generalization in practice. For more details
    on why decoupling might be desirable, see `Decoupled Weight Decay Regularization <https://arxiv.org/abs/1711.05101>`_.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate.
        momentum (int, optional): Momentum factor. Default: ``0``.
        dampening (int, optional): Dampening factor applied to the momentum. Default: ``0``.
        weight_decay (int, optional): Decoupled weight decay factor. Default: ``0``.
        nesterov (bool, optional): Enables Nesterov momentum updates. Default: ``False``.
    """

    def __init__(
            self,
            params: Union[Iterable[torch.Tensor], Iterable[dict]],
            lr: float = required,  # type: ignore
            momentum: float = 0,
            dampening: float = 0,
            weight_decay: float = 0,
            nesterov: bool = False):
        if weight_decay >= 1e-3:
            log.warning(
                f'You are using a high value of `weight_decay={weight_decay}` for the `DecoupledSGDW` optimizer. Are you sure you want to do this? '
                f'Your model\'s weights will be multiplied by {1.0 - weight_decay} on every step!')
        super().__init__(params=params,
                         lr=lr,
                         momentum=momentum,
                         dampening=dampening,
                         weight_decay=weight_decay,
                         nesterov=nesterov)
        for group in self.param_groups:
            group['initial_lr'] = group['lr']

    @staticmethod
    def sgdw(params: List[torch.Tensor], d_p_list: List[torch.Tensor], momentum_buffer_list: List[torch.Tensor], *,
             weight_decay: float, momentum: float, lr: float, initial_lr: float, dampening: float, nesterov: bool):
        r"""Functional API that performs SGDW algorithm computation.

        Args:
            params (list): List of parameters to update
            d_p_list (list): List of parameter gradients
            momentum_buffer_list (list): List of momentum buffers
            weight_decay (float): Decoupled weight decay factor
            momentum (float): Momentum factor
            lr (float): Learning rate
            initial_lr (float): Initial learning rate
            dampening (float): Dampening factor for momentum update
            nesterov (bool): Enables Nesterov momentum updates
        """
        for i, param in enumerate(params):

            d_p = d_p_list[i]

            if momentum != 0:
                buf = momentum_buffer_list[i]

                if buf is None:
                    buf = torch.clone(d_p).detach()
                    momentum_buffer_list[i] = buf
                else:
                    buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                if nesterov:
                    d_p = d_p.add(buf, alpha=momentum)
                else:
                    d_p = buf

            if weight_decay != 0:
                decay_factor = (lr / initial_lr) if initial_lr else 1.0
                param.mul_(1 - decay_factor * weight_decay)

            param.add_(d_p, alpha=-lr)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']
            initial_lr = group['initial_lr']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            self.sgdw(params_with_grad,
                      d_p_list,
                      momentum_buffer_list,
                      weight_decay=weight_decay,
                      momentum=momentum,
                      lr=lr,
                      initial_lr=initial_lr,
                      dampening=dampening,
                      nesterov=nesterov)

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss

class DecoupledAdamW(AdamW, IProcessGroupEnabled):
    """Adam optimizer with the weight decay term decoupled from the learning rate.

    NOTE: Since `weight_decay` is no longer scaled by `lr`, you will likely want to use much smaller values
    for `weight_decay` than you would if using `torch.optim.Adam` or `torch.optim.AdamW`. In this optimizer, the value `weight_decay` translates exactly to:
    'On every optimizer update, every weight element will be multiplied by `(1.0 - weight_decay_t)`'.
    The term `weight_decay_t` will follow the same schedule as `lr_t` but crucially will not be scaled by `lr`.

    Argument defaults are similar to :class:`torch.optim.AdamW` but we make two changes:
    * The default for ``weight_decay`` is changed from ``1e-2`` -> ``1e-5`` because in `DecoupledAdamW`, the weight decay is decoupled and no longer scaled by the `lr=1e-3`.
    * The default for ``betas`` is changed from ``(0.9, 0.999)`` to ``(0.9, 0.95)`` to reflect community best-practices for the beta2 hyperparameter.

    Why use this optimizer? The standard `AdamW <https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html#torch.optim.AdamW>`_
    optimizer explicitly couples the weight decay term with the learning rate. This ties the
    optimal value of :attr:`weight_decay` to :attr:`lr` and can also hurt generalization in practice. For more details on
    why decoupling might be desirable, see `Decoupled Weight Decay Regularization <https://arxiv.org/abs/1711.05101>`_.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): Learning rate. Default: ``1e-3``.
        betas (tuple, optional): Coefficients used for computing running averages of gradient and its square
                                 Default: ``(0.9, 0.95)``.
        eps (float, optional): Term added to the denominator to improve numerical stability. Default: ``1e-8``.
        weight_decay (float, optional): Decoupled weight decay factor. Default: ``1e-5``.
        amsgrad (bool, optional): Enables the amsgrad variant of Adam. Default: ``False``.
        cautious_weight_decay (bool, optional): Apply weight decay only where the Adam update and parameter have
            matching signs. The selective decay and parameter update are fused in a Triton kernel. Default: ``False``.
    """

    metric_functions = {
        'l2_norm/moment': lambda param, optim_state, step_tensor: torch.linalg.vector_norm(optim_state['exp_avg']),
        'l2_norm/moment_sq': lambda param, optim_state, step_tensor: torch.linalg.vector_norm(optim_state['exp_avg_sq']),
        'l2_norm/param': lambda param, optim_state, step_tensor: torch.linalg.vector_norm(param.data),
        'l2_norm/update': lambda param, optim_state, step_tensor: torch.linalg.vector_norm(step_tensor),
        'l2_norm/grad': lambda param, optim_state, step_tensor: torch.linalg.vector_norm(param.grad),
    }

    def __init__(self,
                 params: Union[Iterable[torch.Tensor], Iterable[dict]],
                 lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.95),
                 eps: float = 1e-8,
                 weight_decay: float = 1e-5,
                 amsgrad: bool = False,
                 cautious_weight_decay: bool = False,
                 per_fqn_overrides: Optional[List[dict]] = None,
                 ):
        if weight_decay >= 1e-3:
            log.warning(
                f'You are using a high value of `weight_decay={weight_decay}` for the `DecoupledAdamW` optimizer. Are you sure you want to do this? '
                f'Your model\'s weights will be multiplied by {1.0 - weight_decay} on every step!')
        super().__init__(params=params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        for group in self.param_groups:
            group['initial_lr'] = group['lr']
        self.amsgrad = amsgrad
        self.defaults['cautious_weight_decay'] = cautious_weight_decay
        for group in self.param_groups:
            group.setdefault('cautious_weight_decay', cautious_weight_decay)

        self.skip_updates = 0
        self.total_updates = 0

        # Optional per-parameter learning-rate multiplier, keyed by the parameter
        # tensor (identity); absent params default to 1.0 (no scaling). Lets a
        # callback scale the effective LR of individual parameters/blocks without
        # creating extra optimizer param groups. The multiplier scales BOTH the
        # Adam update and the decoupled weight-decay term, so the weight-decay to
        # effective-LR ratio stays constant (a true per-parameter LR knob, like
        # the LR scheduler which also preserves that ratio).
        self.lr_scale = {}

        # Static per-parameter LR multiplier, set from the config (``lr_scales``)
        # and held SEPARATELY from the dynamic ``lr_scale`` above. The effective
        # multiplier applied each step is the product ``base_lr_scale * lr_scale``
        # so a config-driven base (e.g. 0.5 for some blocks) and a dynamic
        # reduction (e.g. DynamicBlockLRRescaler) compose without clobbering each
        # other. The base is keyed by parameter identity, but identity does not
        # survive FSDP param-flattening, so the resolving regex spec is stashed in
        # ``_lr_scales_spec`` and re-applied by name post-FSDP (see
        # ``apply_base_lr_scales``).
        self.base_lr_scale = {}
        self._lr_scales_spec: Optional[list] = None

        # Per-FQN (sub-supertensor) LR / weight-decay multipliers. Under GigaFSDP
        # the optimizer sees one flattened shard per SuperTensorModule that packs
        # several parameters (fqns) end-to-end, so a plain param_group can only set
        # one lr/weight_decay for the WHOLE module. ``per_fqn_overrides`` lets a
        # config knock the lr and/or weight_decay of INDIVIDUAL fqns inside such a
        # module up or down, by walking the packed fqn chunks of each shard (the
        # same layout Muon uses) and painting a per-element multiplier vector onto
        # the local shard slice. Each entry is
        # ``{"param_str_match": <regex>, "lr_mult": <float>, "wd_mult": <float>}``;
        # a fqn takes each multiplier from the LAST matching entry that specifies
        # it (unmatched fqns keep 1.0). The multipliers are independent -- lr_mult
        # scales only the Adam step, wd_mult only the decoupled weight decay -- and
        # compose multiplicatively with the per-param ``lr_scale``/``base_lr_scale``
        # above. Vectors are built lazily on the first step (fqn chunk metadata and
        # the fsdp process group are only wired onto the param_groups after FSDP
        # sharding) and rebuilt after ``load_state_dict``.
        self._per_fqn_overrides = per_fqn_overrides
        self._per_fqn_inited = False
        self._has_per_fqn = False
        self._per_fqn_lr_vec: dict = {}
        self._per_fqn_wd_vec: dict = {}
        self._fqn_mult_cache: dict = {}
        self._validate_per_fqn_overrides()
        # Host copies of the device step counters for metric reports, advanced with them in step().
        self._host_steps: dict = {}

    def _validate_per_fqn_overrides(self) -> None:
        """Validate per-FQN override entries before the first optimizer step."""
        allowed = {'param_str_match', 'lr_mult', 'wd_mult'}
        for index, raw_entry in enumerate(self._per_fqn_overrides or []):
            if not isinstance(raw_entry, Mapping):
                raise TypeError(
                    f'per_fqn_overrides[{index}] must be a mapping, got {type(raw_entry).__name__}.')
            unknown = sorted(set(raw_entry) - allowed)
            if unknown:
                raise ValueError(
                    f'per_fqn_overrides[{index}] has unsupported keys {unknown}; '
                    f'allowed keys are {sorted(allowed)}.')
            if 'param_str_match' not in raw_entry:
                raise ValueError(f'per_fqn_overrides[{index}] requires `param_str_match`.')
            try:
                re.compile(raw_entry['param_str_match'])
            except (TypeError, re.error) as e:
                raise ValueError(
                    f'per_fqn_overrides[{index}] has invalid `param_str_match`: '
                    f'{raw_entry["param_str_match"]!r}.') from e
            for key in ('lr_mult', 'wd_mult'):
                if key not in raw_entry:
                    continue
                value = float(raw_entry[key])
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(
                        f'per_fqn_overrides[{index}].{key} must be finite and >= 0, got {value}.')

    def set_lr_scale(self, param: torch.Tensor, scale: float) -> None:
        """Set the dynamic per-parameter LR multiplier for ``param`` (1.0 = unscaled)."""
        self.lr_scale[param] = scale

    def get_lr_scale(self, param: torch.Tensor) -> float:
        """Return the dynamic per-parameter LR multiplier for ``param`` (default 1.0)."""
        return self.lr_scale.get(param, 1.0)

    def set_base_lr_scale(self, param: torch.Tensor, scale: float) -> None:
        """Set the static (config-driven) per-parameter LR multiplier for ``param``."""
        self.base_lr_scale[param] = scale

    def get_base_lr_scale(self, param: torch.Tensor) -> float:
        """Return the static per-parameter LR multiplier for ``param`` (default 1.0)."""
        return self.base_lr_scale.get(param, 1.0)

    def get_effective_lr_scale(self, param: torch.Tensor) -> float:
        """Return the full per-parameter LR multiplier: ``base * dynamic``."""
        return self.base_lr_scale.get(param, 1.0) * self.lr_scale.get(param, 1.0)

    def set_lr_scales_spec(self, spec: Optional[list]) -> None:
        """Stash the config ``lr_scales`` spec so it can be re-applied post-FSDP.

        ``spec`` is a list of ``{"param_str_match": <regex>, "scale": <float>}``
        mappings (the same format accepted in the optimizer config). It is kept so
        that, once FSDP has rebuilt the param groups with fresh (flattened) tensors
        whose identity no longer matches the pre-FSDP keys, the base scales can be
        re-resolved by parameter *name* via :meth:`apply_base_lr_scales`.
        """
        self._lr_scales_spec = spec

    def apply_base_lr_scales(self, named_parameters) -> None:
        """(Re)resolve ``_lr_scales_spec`` against ``named_parameters`` into ``base_lr_scale``.

        Clears any previously-resolved base scales first (so stale pre-FSDP keys
        do not linger), then matches each spec entry's regex against the given
        parameter names and records the scale per matched parameter. Later entries
        override earlier ones for a param both match. No-op when no spec is set.
        """
        spec = self._lr_scales_spec
        if not spec:
            return
        import re
        self.base_lr_scale = {}
        named = list(named_parameters)
        for entry in spec:
            entry = dict(entry)
            str_match = entry.pop('param_str_match')
            scale = float(entry.pop('scale'))
            if entry:
                raise ValueError(
                    f'Unexpected keys in `lr_scales` entry: {list(entry.keys())}. '
                    'Only `param_str_match` and `scale` are supported.')
            matched = 0
            for name, p in named:
                if re.search(str_match, name):
                    self.base_lr_scale[p] = scale
                    matched += 1
            if matched == 0:
                log.warning(
                    f'`lr_scales` entry param_str_match={str_match!r} matched no '
                    'parameters; nothing scaled for this entry.')
            else:
                log.info(
                    f'Applied base lr_scale={scale} to {matched} parameters '
                    f'matching {str_match!r}.')

    def _resolve_fqn_mult(self, fqn: str) -> Tuple[float, float]:
        """Resolve ``(lr_mult, wd_mult)`` for ``fqn`` from ``per_fqn_overrides``.

        Later matching entries override the multiplier keys they explicitly
        specify; unmatched fqns get ``(1.0, 1.0)``.
        """
        cached = self._fqn_mult_cache.get(fqn)
        if cached is not None:
            return cached
        lr_mult, wd_mult = 1.0, 1.0
        for entry in (self._per_fqn_overrides or []):
            if re.search(entry['param_str_match'], fqn):
                if 'lr_mult' in entry:
                    lr_mult = float(entry['lr_mult'])
                if 'wd_mult' in entry:
                    wd_mult = float(entry['wd_mult'])
        self._fqn_mult_cache[fqn] = (lr_mult, wd_mult)
        return lr_mult, wd_mult

    def get_per_fqn_lr_multiplier(self, param: torch.Tensor) -> Optional[torch.Tensor]:
        """Return the per-element LR multiplier for a packed parameter, if active."""
        return self._per_fqn_lr_vec.get(param)

    def get_per_fqn_wd_multiplier(self, param: torch.Tensor) -> Optional[torch.Tensor]:
        """Return the per-element weight-decay multiplier for a packed parameter, if active."""
        return self._per_fqn_wd_vec.get(param)

    def _build_per_fqn_vec_for_param(self, p: torch.Tensor, fqns: List[str], numels: List[int]) -> None:
        """Paint per-element lr/wd multiplier vectors onto one flattened shard.

        ``fqns``/``numels`` describe the parameters packed (in order) into the
        UNSHARDED flat tensor that ``p`` is the local shard of. Only the fqn
        chunks overlapping this rank's shard slice are written; a vector is stored
        only if some element actually differs from 1.0.
        """
        if len(fqns) != len(numels):
            log.warning(
                f'per_fqn_overrides: fqn/numel length mismatch for packed parameter '
                f'(fqns={len(fqns)}, numels={len(numels)}); skipping per-fqn lr/wd.')
            return
        if any(not isinstance(numel, int) or numel < 0 for numel in numels):
            log.warning(
                f'per_fqn_overrides: invalid packed parameter numels {numels}; '
                'skipping per-fqn lr/wd.')
            return

        mults = [self._resolve_fqn_mult(fqn) for fqn in fqns]
        if all(lr == 1.0 and wd == 1.0 for lr, wd in mults):
            return

        total_numel = sum(numels)
        shard_size = p.numel()
        rank = self.get_fsdp_pg_rank(fqns[0])
        world_size = self.get_fsdp_pg_world_size(fqns[0])
        local_start = shard_size * rank
        local_end = local_start + shard_size

        # The sharded flat param must tile the (padded) unsharded total; if it does
        # not, the offset math would be wrong -- skip rather than corrupt weights.
        if shard_size * world_size < total_numel:
            log.warning(
                f'per_fqn_overrides: shard_size*world_size ({shard_size * world_size}) < '
                f'total_numel ({total_numel}) for module starting at {fqns[0]!r}; '
                'skipping per-fqn lr/wd for this module.')
            return

        lr_vec = torch.ones(shard_size, device=p.device, dtype=torch.float32)
        wd_vec = torch.ones(shard_size, device=p.device, dtype=torch.float32)
        any_lr = any_wd = False
        offset = 0
        for (lr_mult, wd_mult), numel in zip(mults, numels):
            chunk_start, chunk_end = offset, offset + numel
            offset = chunk_end
            if lr_mult == 1.0 and wd_mult == 1.0:
                continue
            lo = max(chunk_start, local_start)
            hi = min(chunk_end, local_end)
            if hi <= lo:
                continue  # this fqn chunk is not on the local shard
            loc_lo, loc_hi = lo - local_start, hi - local_start
            if lr_mult != 1.0:
                lr_vec[loc_lo:loc_hi] = lr_mult
                any_lr = True
            if wd_mult != 1.0:
                wd_vec[loc_lo:loc_hi] = wd_mult
                any_wd = True

        if any_lr:
            self._per_fqn_lr_vec[p] = lr_vec.to(p.dtype).view_as(p)
        if any_wd:
            self._per_fqn_wd_vec[p] = wd_vec.to(p.dtype).view_as(p)

    def _maybe_build_per_fqn(self) -> None:
        """Build per-fqn multiplier vectors once, from the live param_groups.

        Reads the fqn-chunk layout Muon relies on (``main_shards_fqns`` /
        ``main_shards_numels`` plus the ``layernorm_*`` variants, ordered
        layernorm-then-main to match ``group['params']``). No-op when no overrides
        are configured; warns once if overrides are set but no metadata is present.
        """
        if self._per_fqn_inited:
            return
        self._per_fqn_inited = True
        self._per_fqn_lr_vec = {}
        self._per_fqn_wd_vec = {}
        self._has_per_fqn = False
        if not self._per_fqn_overrides:
            return

        found_meta = False
        for group in self.param_groups:
            fqns_per_param = (list(group.get('layernorm_shards_fqns', []))
                              + list(group.get('main_shards_fqns', [])))
            numels_per_param = (list(group.get('layernorm_shards_numels', []))
                                + list(group.get('main_shards_numels', [])))
            if not fqns_per_param:
                continue
            found_meta = True
            params = group['params']
            if len(params) != len(fqns_per_param) or len(numels_per_param) != len(fqns_per_param):
                log.warning(
                    'per_fqn_overrides: param/metadata length mismatch in group '
                    f'(params={len(params)}, fqns={len(fqns_per_param)}, '
                    f'numels={len(numels_per_param)}); skipping per-fqn for this group.')
                continue
            for p, fqns, numels in zip(params, fqns_per_param, numels_per_param):
                self._build_per_fqn_vec_for_param(p, fqns, numels)

        if not found_meta:
            log.warning(
                'per_fqn_overrides was provided but no shard-fqn metadata '
                '(main_shards_fqns / layernorm_shards_fqns) was found on any optimizer '
                'param group; per-fqn lr/wd is inactive.')
        self._has_per_fqn = bool(self._per_fqn_lr_vec or self._per_fqn_wd_vec)

    @staticmethod
    def adamw(params: List[torch.Tensor], grads: List[torch.Tensor], exp_avgs: List[torch.Tensor],
              exp_avg_sqs: List[torch.Tensor], max_exp_avg_sqs: List[torch.Tensor], state_steps: List[torch.Tensor], *,
              amsgrad: bool, beta1: float, beta2: float, lr: float, initial_lr: float, weight_decay: float,
              eps: float, lr_scales: Optional[List[float]] = None,
              cautious_weight_decay: bool = False,
              lr_mult_vecs: Optional[List[Optional[torch.Tensor]]] = None,
              wd_mult_vecs: Optional[List[Optional[torch.Tensor]]] = None) -> None:
        r"""Functional API that performs AdamW algorithm computation with decoupled weight decay.

        Args:
            params (list): List of parameters to update.
            grads (list): List of parameter gradients.
            exp_avgs (list): List of average gradients.
            exp_avg_sqs (list): List of average squared gradients.
            max_exp_avg_sqs (list): List of max average squared gradients for amsgrad updates.
            state_steps (list): List of steps taken for all parameters.
            amsgrad (bool): Enables amsgrad variant of Adam.
            beta1 (float): Coefficient for computing the moving average of gradient values.
            beta2 (float): Coefficient for computing the moving average of squared gradient values.
            lr (float): Learning rate.
            initial_lr (float): Initial learning rate.
            weight_decay (float): Factor for decoupled weight decay
            eps (float): Term added to the denominator to improve numerical stability.
            lr_scales (list, optional): Per-parameter LR multipliers (aligned with
                ``params``). ``None`` means no scaling (all 1.0).
            lr_mult_vecs (list, optional): Per-parameter, per-ELEMENT LR multiplier
                tensors (aligned with ``params``, each shaped like its param or
                ``None``). Applied on top of ``lr_scales`` for per-fqn LR control.
            wd_mult_vecs (list, optional): Per-parameter, per-ELEMENT weight-decay
                multiplier tensors (aligned with ``params``, ``None`` = 1.0).
                Independent of ``lr_mult_vecs``.

        """
        for i, param in enumerate(params):
            grad = grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step = state_steps[i].item()
            step_scale = lr_scales[i] if lr_scales is not None else 1.0
            lr_vec = lr_mult_vecs[i] if lr_mult_vecs is not None else None
            wd_vec = wd_mult_vecs[i] if wd_mult_vecs is not None else None

            # Perform step weight decay. Scale by step_scale too, so the per-param
            # LR multiplier keeps the weight-decay / effective-LR ratio constant;
            # wd_vec adds an independent per-fqn weight-decay multiplier on top.
            # Standard decay is applied before the Adam update. Cautious decay
            # is fused into the parameter update below, after moments are updated.
            if weight_decay != 0 and not cautious_weight_decay:
                decay_factor = (lr / initial_lr) if initial_lr else 1.0
                wd = step_scale * decay_factor * weight_decay
                if wd_vec is not None:
                    param.mul_(1 - wd * wd_vec)
                else:
                    param.mul_(1 - wd)

            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])

            step_size = step_scale * lr / bias_correction1
            if cautious_weight_decay:
                decay_factor = (lr / initial_lr) if initial_lr else 1.0
                source = max_exp_avg_sqs[i] if amsgrad else exp_avg_sq
                cautious_wd = step_scale * decay_factor * weight_decay
                if lr_vec is None and wd_vec is None:
                    adamw_cautious_parameter_update(
                        param,
                        exp_avg,
                        source,
                        step_size,
                        math.sqrt(bias_correction2),
                        eps,
                        cautious_wd,
                    )
                else:
                    denom = (source.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                    update = exp_avg / denom
                    if lr_vec is not None:
                        update = update * lr_vec
                    decay_param = torch.where(param * exp_avg >= 0, param, 0)
                    if wd_vec is not None:
                        param.sub_(decay_param * wd_vec, alpha=cautious_wd)
                    else:
                        param.sub_(decay_param, alpha=cautious_wd)
                    param.add_(update, alpha=-step_size)
            else:
                source = max_exp_avg_sqs[i] if amsgrad else exp_avg_sq
                denom = (source.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                if lr_vec is not None:
                    param.addcdiv_(exp_avg * lr_vec, denom, value=-step_size)
                else:
                    param.addcdiv_(exp_avg, denom, value=-step_size)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Resolve per-fqn lr/wd multiplier vectors once (metadata is only wired
        # onto the param_groups after FSDP sharding, so this must be lazy).
        self._maybe_build_per_fqn()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']
            eps = group['eps']
            lr = group['lr']
            if 'initial_lr' not in group:
                group['initial_lr'] = lr
            initial_lr = group['initial_lr']
            weight_decay = group['weight_decay']
            cautious_weight_decay = group.get('cautious_weight_decay', False)

            for p in group['params']:
                if p.grad is None or not p.requires_grad:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                grads.append(p.grad)

            is_good = []
            num_devices_with_inf_grads = torch.tensor([0], device=torch.cuda.current_device())
            for i in range(len(params_with_grad)):
                is_good.append(torch.isfinite(grads[i]).all().item())
            num_good = sum(is_good)
            if num_good < len(grads):
                num_devices_with_inf_grads = torch.tensor([1], device=torch.cuda.current_device())
            torch.distributed.all_reduce(num_devices_with_inf_grads)
            if num_devices_with_inf_grads > 0:
                log.critical(f"Skip step because grads had nan. The number of devices with infinite gradients: {num_devices_with_inf_grads.cpu().item()}")
                _report_grad_nan_origin()
                self.total_updates += 1
                self.skip_updates += 1
                continue

            for p in params_with_grad:
                state = self.state[p]

                # State initialization
                if 'step' not in state:
                    self.init_moments(state, p, amsgrad)
                    self._host_steps[p] = 0.0

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                if amsgrad:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                # Update the steps for each param group update
                state['step'] += 1
                if p in self._host_steps:
                    self._host_steps[p] += 1
                # Record the step after step update
                state_steps.append(state['step'])

            # Effective per-param multiplier = static base * dynamic scale.
            lr_scales = (
                [self.get_effective_lr_scale(p) for p in params_with_grad]
                if (self.lr_scale or self.base_lr_scale) else None
            )

            # Per-fqn (per-element) multipliers, only when active. Kept out of the
            # call otherwise so subclasses whose `adamw` lacks these kwargs (e.g.
            # DecoupledFusedAdamW) are unaffected.
            per_fqn_kwargs = {}
            if self._has_per_fqn:
                per_fqn_kwargs['lr_mult_vecs'] = [self._per_fqn_lr_vec.get(p) for p in params_with_grad]
                per_fqn_kwargs['wd_mult_vecs'] = [self._per_fqn_wd_vec.get(p) for p in params_with_grad]

            self.adamw_update(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=amsgrad,
                beta1=beta1,
                beta2=beta2,
                lr=lr,
                initial_lr=initial_lr,
                weight_decay=weight_decay,
                eps=eps,
                lr_scales=lr_scales,
                cautious_weight_decay=cautious_weight_decay,
                **per_fqn_kwargs,
            )
            self.total_updates += 1
        return loss

    def init_moments(self, state, p, amsgrad):
        state['step'] = torch.zeros((), dtype=torch.float, device=p.device)
        # Exponential moving average of gradient values
        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
        # Exponential moving average of squared gradient values
        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
        if amsgrad:
            # Maintains max of all exp. moving avg. of sq. grad. values
            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

    def adamw_update(self, *args, **kwargs) -> None:
        self.adamw(*args, **kwargs)

    def load_state_dict(self, state_dict) -> None:
        super().load_state_dict(state_dict)
        # Force a lazy rebuild of the per-fqn vectors: param identities and the
        # shard-fqn metadata on the groups may have changed across resume.
        self._per_fqn_inited = False
        self._fqn_mult_cache = {}
        self._host_steps = {}

    def host_step(self, param: torch.Tensor) -> float:
        """The parameter's optimizer step count without a device synchronization once known: read
        from its counter the first time after a state load, then advanced with it by step()."""
        step = self._host_steps.get(param)
        if step is None:
            step = self._host_steps[param] = self.state[param]['step'].item()
        return step

    def dist_reduce_metrics(self, optimizer_metrics):
        local_keys = list(optimizer_metrics.keys())
        all_gathered_keys = dist.all_gather_object(local_keys)
        all_keys = set()
        for keys in all_gathered_keys:
            all_keys.update(keys)

        # Sort keys to ensure every rank has the same keys order
        # Only L2 norm metric keys are present, can apply regular sort
        all_keys = sorted(all_keys)
        for metric in all_keys:
            if metric.startswith('l2_norm'):
                reduced = optimizer_metrics.get(metric, torch.tensor(0.0, device=torch.cuda.current_device()))
                if dist.get_world_size() > 1:
                    dist.all_reduce(reduced, reduce_operation='SUM')

                optimizer_metrics[metric] = math.sqrt(reduced)
            elif metric.startswith('amax/'):
                reduced = optimizer_metrics.get(metric, torch.tensor(0.0, device=torch.cuda.current_device()))
                if dist.get_world_size() > 1:
                    dist.all_reduce(reduced, reduce_operation='MAX')
                optimizer_metrics[metric] = reduced
            else:
                reduced = optimizer_metrics.get(metric, torch.tensor(0.0, device=torch.cuda.current_device()))
                if dist.get_world_size() > 1:
                    dist.all_reduce(reduced, reduce_operation='SUM')
                optimizer_metrics[metric] = reduced / dist.get_world_size()
       
        #compute eff_lr, upd_norm/weight_norm, upd_rate
        l2_keys = [k for k in all_keys if k.startswith('l2_norm/grad/')]
        for grad_key in l2_keys:
            name = grad_key.removeprefix('l2_norm/grad/')
            update_key = f'l2_norm/update/{name}'
            param_key = f'l2_norm/param/{name}'
            if update_key in optimizer_metrics:
                optimizer_metrics[f'effective_lr/{name}'] = (
                                    optimizer_metrics[update_key] / (optimizer_metrics[grad_key] + 1e-8)
                )

            if update_key in optimizer_metrics and param_key in optimizer_metrics:
                optimizer_metrics[f'update_weight_ratio/{name}'] = (
                    optimizer_metrics[update_key] / (optimizer_metrics[param_key] + 1e-8)
                )

            if param_key in optimizer_metrics:
                optimizer_metrics[f'update_rate/{name}'] = (
                    optimizer_metrics[grad_key] / (optimizer_metrics[param_key] + 1e-8)
                )
        return optimizer_metrics

    def pre_reduce_metrics(self, optimizer_metrics):
        """Preprocess metrics to reduce across ranks correctly."""
        # Only L2 norm metric keys are present, can skip sorting at this stage
        for metric in optimizer_metrics:
            if metric.startswith('l2_norm'):
                # L2 norms need to be squared, before they are reduced via summation
                optimizer_metrics[metric] = optimizer_metrics[metric]**2

        return optimizer_metrics

    def report_metrics_for_parameters(self, named_parameters: list, optimizer_metrics: dict) -> dict:
        """``report_per_parameter_metrics`` of each (name, parameter), the fused metrics of every
        CUDA parameter with a gradient computed together."""
        fused = []
        for name, param in named_parameters:
            if param in self.state and param.is_cuda and param.grad is not None:
                fused.append((name, param))
            else:
                self.report_per_parameter_metrics(param, name, optimizer_metrics)
        if fused:
            self._report_fused_metrics(fused, optimizer_metrics)
        return optimizer_metrics

    def _report_fused_metrics(self, fused: list, optimizer_metrics: dict) -> None:
        """The metrics of (name, parameter) pairs of CUDA parameters with optimizer state and a
        gradient, every parameter reduced in one launch per dtype combination; a metric of float32
        parameters is cast or bias-corrected for all of them in one operation, to the same values."""
        lr = self.param_groups[0]['lr']
        eps = self.param_groups[0]['eps']
        weight_decay = self.param_groups[0]['weight_decay']
        cautious_weight_decay = self.param_groups[0].get('cautious_weight_decay', False)
        initial_lr = self.param_groups[0]['initial_lr']
        beta1, beta2 = self.param_groups[0]['betas']
        decay_factor = (lr / initial_lr) if initial_lr else 1.0
        entries, corrections, dtypes = [], [], []
        for _, param in fused:
            state = self.state[param]
            step = self.host_step(param)
            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step
            lr_scale = self.get_effective_lr_scale(param)
            entries.append((
                param.data, param.grad, state['exp_avg'], state['exp_avg_sq'], lr_scale * lr / bias_correction1,
                math.sqrt(bias_correction2), eps, -lr_scale * weight_decay * decay_factor,
            ))
            corrections.append((bias_correction1, bias_correction2))
            # The dtypes of [moment, moment_sq, param, update, grad].
            dtypes.append((state['exp_avg'].dtype, state['exp_avg_sq'].dtype, param.dtype, state['exp_avg'].dtype,
                           param.grad.dtype))
        sums, maxima = adam_metric_sums_batch(entries, cautious_weight_decay)
        norms = sums.sqrt()
        if all(dtype == torch.float32 for row in dtypes for dtype in row):
            values = norms.float()
            # A float32 tensor divided by a Python scalar is a product with the scalar's float32
            # reciprocal.
            reciprocals = torch.tensor(
                [[float32_reciprocal(c) if c > 0 else 0.0 for c in pair] for pair in corrections], dtype=torch.float32,
                pin_memory=True,
            ).to(values.device, non_blocking=True)
            rows = list(zip(values, values[:, :2] * reciprocals))
        else:
            rows = [([norms[index, column].to(dtype) for column, dtype in enumerate(row_dtypes)], None)
                    for index, row_dtypes in enumerate(dtypes)]
        for (name, _), (value, corrected), (bias_correction1, bias_correction2), row_dtypes, row_maxima in zip(
                fused, rows, corrections, dtypes, maxima):
            for column, metric in enumerate(('moment', 'moment_sq', 'param', 'update', 'grad')):
                optimizer_metrics[f'l2_norm/{metric}/{name}'] = value[column]
            # bias-corrected moments
            if bias_correction1 > 0:
                optimizer_metrics[f'l2_norm/moment_bc/{name}'] = (
                    value[0] / bias_correction1 if corrected is None else corrected[0])
            if bias_correction2 > 0:
                optimizer_metrics[f'l2_norm/moment_sq_bc/{name}'] = (
                    value[1] / bias_correction2 if corrected is None else corrected[1])
            # max of raw moment tensors -- for spotting outliers in optimizer state
            optimizer_metrics[f'amax/moment/{name}'] = row_maxima[0].to(row_dtypes[0])
            optimizer_metrics[f'amax/moment_sq/{name}'] = row_maxima[1].to(row_dtypes[1])

    def report_per_parameter_metrics(self, param: torch.Tensor, name: str, optimizer_metrics: dict):
        if param in self.state and param.is_cuda and param.grad is not None:
            self._report_fused_metrics([(name, param)], optimizer_metrics)
            return optimizer_metrics
        lr = self.param_groups[0]['lr']
        eps = self.param_groups[0]['eps']
        weight_decay = self.param_groups[0]['weight_decay']
        cautious_weight_decay = self.param_groups[0].get('cautious_weight_decay', False)
        initial_lr = self.param_groups[0]['initial_lr']

        beta1, beta2 = self.param_groups[0]['betas']
        if param in self.state:
            param_optim_state = self.state[param]
            step = self.host_step(param)
            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step
            lr_scale = self.get_effective_lr_scale(param)
            step_size = lr_scale * lr / bias_correction1
            decay_factor = (lr / initial_lr) if initial_lr else 1.0
            exp_avg = param_optim_state['exp_avg']
            exp_avg_sq = param_optim_state['exp_avg_sq']
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            step_tensor = step_size * exp_avg.div(denom)
            if cautious_weight_decay:
                decay_param = torch.where(step_tensor * param >= 0, param, 0)
                step_tensor.add_(decay_param, alpha=-lr_scale * weight_decay * decay_factor)
            else:
                step_tensor.add_(param, alpha=-lr_scale * weight_decay * decay_factor)
            for metric in self.metric_functions:
                optimizer_metrics[f'{metric}/{name}'] = self.metric_functions[metric](param, param_optim_state,
                                                                                      step_tensor)
            # bias-corrected moments 
            if bias_correction1 > 0:
                optimizer_metrics[f'l2_norm/moment_bc/{name}'] = (
                    optimizer_metrics[f'l2_norm/moment/{name}'] / bias_correction1
                )
            if bias_correction2 > 0:
                optimizer_metrics[f'l2_norm/moment_sq_bc/{name}'] = (
                    optimizer_metrics[f'l2_norm/moment_sq/{name}'] / bias_correction2
                )

            # max/min of raw moment tensors — for spotting outliers in optimizer state
            optimizer_metrics[f'amax/moment/{name}'] = exp_avg.abs().amax()
            optimizer_metrics[f'amax/moment_sq/{name}'] = exp_avg_sq.amax()
        return optimizer_metrics

def step_counts(optimizer: torch.optim.Optimizer) -> dict:
    """Each parameter's optimizer step count for metric reports: the host copies DecoupledAdamW
    keeps, otherwise the device counters read with one host synchronization."""
    if isinstance(optimizer, DecoupledAdamW):
        return {p: optimizer.host_step(p) for p, state in optimizer.state.items() if 'step' in state}
    counters = {p: state['step'] for p, state in optimizer.state.items()
                if torch.is_tensor(state.get('step')) and state['step'].is_cuda}
    if not counters:
        return {}
    return dict(zip(counters, torch.stack([step.reshape(()) for step in counters.values()]).tolist()))


class DecoupledFusedAdamW(DecoupledAdamW):
    """Adam optimizer with the weight decay term decoupled from the learning rate.

    NOTE: Since `weight_decay` is no longer scaled by `lr`, you will likely want to use much smaller values
    for `weight_decay` than you would if using `torch.optim.Adam` or `torch.optim.AdamW`. In this optimizer, the value `weight_decay` translates exactly to:
    'On every optimizer update, every weight element will be multiplied by `(1.0 - weight_decay_t)`'.
    The term `weight_decay_t` will follow the same schedule as `lr_t` but crucially will not be scaled by `lr`.

    Argument defaults are similar to :class:`torch.optim.AdamW` but we make two changes:
    * The default for ``weight_decay`` is changed from ``1e-2`` -> ``1e-5`` because in `DecoupledAdamW`, the weight decay is decoupled and no longer scaled by the `lr=1e-3`.
    * The default for ``betas`` is changed from ``(0.9, 0.999)`` to ``(0.9, 0.95)`` to reflect community best-practices for the beta2 hyperparameter.

    Why use this optimizer? The standard `AdamW <https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html#torch.optim.AdamW>`_
    optimizer explicitly couples the weight decay term with the learning rate. This ties the
    optimal value of :attr:`weight_decay` to :attr:`lr` and can also hurt generalization in practice. For more details on
    why decoupling might be desirable, see `Decoupled Weight Decay Regularization <https://arxiv.org/abs/1711.05101>`_.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): Learning rate. Default: ``1e-3``.
        betas (tuple, optional): Coefficients used for computing running averages of gradient and its square
                                 Default: ``(0.9, 0.95)``.
        eps (float, optional): Term added to the denominator to improve numerical stability. Default: ``1e-8``.
        weight_decay (float, optional): Decoupled weight decay factor. Default: ``1e-5``.
        amsgrad (bool, optional): Enables the amsgrad variant of Adam. Default: ``False``.
        cautious_weight_decay (bool, optional): Apply weight decay only where the Adam update and parameter have
            matching signs. The selective decay and parameter update are fused in a Triton kernel. Default: ``False``.
    """

    def __init__(self,
                 params: Union[Iterable[torch.Tensor], Iterable[dict]],
                 state_dtype: str = 'float',
                 num_layers_compile: int = 2,
                 **kwargs):
        super().__init__(params=params, **kwargs)

        # Per-fqn lr/wd is only implemented for the (non-fused) DecoupledAdamw
        # update; the fused/compiled kernel here does not apply the per-element
        # vectors, so disable it rather than silently ignoring the config.
        if self._per_fqn_overrides:
            log.warning(
                'per_fqn_overrides is not supported for DecoupledFusedAdamW; '
                'ignoring. Use `decoupled_adamw` for per-fqn lr/weight_decay.')
            self._per_fqn_overrides = None

        if state_dtype == "bfloat16":
            self.state_dtype = torch.bfloat16
        else:
            self.state_dtype = torch.float
        self.num_layers_compile = num_layers_compile

    @staticmethod
    def adamw(params: List[torch.Tensor], grads: List[torch.Tensor], exp_avgs: List[torch.Tensor],
              exp_avg_sqs: List[torch.Tensor], max_exp_avg_sqs: List[torch.Tensor], state_steps: List[torch.Tensor], *,
              amsgrad: bool, beta1: float, beta2: float, lr: float, initial_lr: float, weight_decay: float,
              eps: float, num_layers_compile: int, lr_scales: Optional[List[float]] = None,
              cautious_weight_decay: bool = False) -> None:
        r"""Functional API that performs AdamW algorithm computation with decoupled weight decay.

        Args:
            params (list): List of parameters to update.
            grads (list): List of parameter gradients.
            exp_avgs (list): List of average gradients.
            exp_avg_sqs (list): List of average squared gradients.
            max_exp_avg_sqs (list): List of max average squared gradients for amsgrad updates.
            state_steps (list): List of steps taken for all parameters.
            amsgrad (bool): Enables amsgrad variant of Adam.
            beta1 (float): Coefficient for computing the moving average of gradient values.
            beta2 (float): Coefficient for computing the moving average of squared gradient values.
            lr (float): Learning rate.
            initial_lr (float): Initial learning rate.
            weight_decay (float): Factor for decoupled weight decay
            eps (float): Term added to the denominator to improve numerical stability.
            num_layers_compile (int): The number of layers being updated at the same time
            lr_scales (list, optional): Per-parameter LR multipliers (aligned with
                ``params``). ``None`` means no scaling (all 1.0).
        """
        steps = [s.item() for s in state_steps]
        bias_correction1_list = [1 - beta1 ** step for step in steps]
        bias_correction2_list = [1 - beta2 ** step for step in steps]
        sqrt_bc2_list = [torch.tensor(math.sqrt(bc2)) for bc2 in bias_correction2_list]
        step_scales = lr_scales if lr_scales is not None else [1.0] * len(params)
        values_list = [
            torch.tensor(step_scales[i] * lr / bc1) if bc1 != 0 else 0.0
            for i, bc1 in enumerate(bias_correction1_list)
        ]

        if weight_decay != 0 and not cautious_weight_decay:
            decay_factor = (lr / initial_lr) if initial_lr else 1.0
            if lr_scales is None:
                decay = torch.tensor(1 - decay_factor * weight_decay)
                optimized_mul(params, decay)
            else:
                # Scale weight decay per-param too, keeping the weight-decay /
                # effective-LR ratio constant under per-param LR scaling.
                decays = [
                    torch.tensor(1 - lr_scales[i] * decay_factor * weight_decay)
                    for i in range(len(params))
                ]
                optimized_mul_per_param(params, decays)
        
        for start in range(0, len(params), num_layers_compile):
            end = start + num_layers_compile
            params_slice = params[start:end]
            if not params_slice:
                continue

            grads_slice = grads[start:end]
            
            #for fp32, operations don't do anything
            exp_avgs_slice = convert_dtype(exp_avgs[start:end], torch.float)
            exp_avg_sqs_slice = convert_dtype(exp_avg_sqs[start:end], torch.float)

            sqrt_bc2_slice = sqrt_bc2_list[start:end]
            values_slice = values_list[start:end]
            if amsgrad:
                max_exp_avg_sqs_slice = convert_dtype(max_exp_avg_sqs[start:end], torch.float)

            update_moments(exp_avgs_slice, exp_avg_sqs_slice, beta1, beta2, grads_slice)

            if amsgrad:
                torch._foreach_maximum_(max_exp_avg_sqs_slice, exp_avg_sqs_slice)
                source_slice = max_exp_avg_sqs_slice
            else:
                source_slice = exp_avg_sqs_slice

            if cautious_weight_decay:
                decay_factor = (lr / initial_lr) if initial_lr else 1.0
                step_scales_slice = step_scales[start:end]
                for param, exp_avg, source, value, sqrt_bc2, step_scale in zip(
                    params_slice, exp_avgs_slice, source_slice, values_slice, sqrt_bc2_slice, step_scales_slice
                ):
                    adamw_cautious_parameter_update(
                        param,
                        exp_avg,
                        source,
                        value.item(),
                        sqrt_bc2.item(),
                        eps,
                        step_scale * decay_factor * weight_decay,
                    )
            else:
                denoms = create_denoms(source_slice, sqrt_bc2_slice, eps)
                update_params(params_slice, exp_avgs_slice, denoms, values_slice)

            #for fp32, operations don't do anything
            for i in range(len(params_slice)):
                exp_avgs[start+i].copy_(exp_avgs_slice[i])
                exp_avg_sqs[start+i].copy_(exp_avg_sqs_slice[i])
                if amsgrad:
                    max_exp_avg_sqs[start+i].copy_(max_exp_avg_sqs_slice[i])


    def init_moments(self, state, p, amsgrad):
        state['step'] = torch.zeros((), dtype=torch.float, device=p.device)
        # Exponential moving average of gradient values
        state['exp_avg'] = torch.zeros_like(p, dtype=self.state_dtype, memory_format=torch.preserve_format)
        # Exponential moving average of squared gradient values
        state['exp_avg_sq'] = torch.zeros_like(p, dtype=self.state_dtype, memory_format=torch.preserve_format)
        if amsgrad:
            # Maintains max of all exp. moving avg. of sq. grad. values
            state['max_exp_avg_sq'] = torch.zeros_like(p, dtype=self.state_dtype, memory_format=torch.preserve_format)

    def adamw_update(self, *args, **kwargs) -> None:
        self.adamw(*args, **kwargs, num_layers_compile = self.num_layers_compile)
