import logging
import math
from collections import deque
from typing import Any, Callable, Dict, Iterable, Tuple, Union, Optional, List

import torch
import torch.distributed as dist
from torch.optim import AdamW

from composer.utils import dist as composer_dist
from composer.functional import XXT, ba_plus_cAA
from composer.optim.utils import IMixedPrecisionEnabled, IProcessGroupEnabled


log = logging.getLogger(__name__)

stable_express_coeffs = [
    (4.3633196929180516, -9.811256518747468, 5.675226467931637),
    (2.866621594839609, -5.795447810619724, 3.0818716353307565),
    (3.5403137630471084, -7.930198507816198, 5.287368036851983),
    (2.723506415440377, -3.8908326029158373, 2.2649294843181877),
    (1.8506628876692959, -1.1970626367762347, 0.3463997491069388),
]
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def orthogonalize(
    G: torch.Tensor,
    *,
    split_baddbmm: bool = False,
    use_polar: bool = False,
    math_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    X = G.to(dtype=math_dtype)

    transposed = False
    if X.size(-2) > X.size(-1):
        X = X.mT
        transposed = True

    norm = X.norm(dim=(-2, -1), keepdim=True)
    if use_polar:
        X = X / (1.02 * norm + 1e-6)
    else:
        X = X / (norm + 1e-6)

    X = X.contiguous()
    A = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
    B = torch.empty_like(A)
    C = torch.empty_like(X)

    if split_baddbmm:
        BX_matmul = torch.bmm if X.ndim > 2 else torch.mm
    else:
        aX_plus_BX = torch.baddbmm if X.ndim > 2 else torch.addmm

    coeffs_list = polar_express_coeffs if use_polar else stable_express_coeffs
    for a, b, c in coeffs_list:
        XXT(X, out=A)
        ba_plus_cAA(A, alpha=c, beta=b, out=B)

        if split_baddbmm:
            BX_matmul(B, X, out=C)  # type: ignore
            C.add_(X, alpha=a)
        else:
            aX_plus_BX(X, B, X, beta=a, out=C)  # type: ignore

        X, C = C, X

    if transposed:
        X = X.mT
    return X


class BufferRing:
    """Ring of GPU buffers with CUDA-event-based lifetime tracking.

    ``acquire(comm_stream)`` returns the next buffer index, blocking
    comm_stream if compute hasn't finished reading it yet.
    ``release(b, compute_stream)`` records when compute is done with
    buffer *b* so it can be safely reused.
    """

    def __init__(self, num_bufs: int, size: int, device: torch.device, dtype: torch.dtype) -> None:
        self.bufs = [torch.empty(size, device=device, dtype=dtype) for _ in range(num_bufs)]
        self._compute_done: List[Optional[torch.cuda.Event]] = [None] * num_bufs
        self._next = 0
        self.num_bufs = num_bufs

    def acquire(self, comm_stream: torch.cuda.Stream) -> int:
        b = self._next
        self._next = (self._next + 1) % self.num_bufs
        if self._compute_done[b] is not None:
            comm_stream.wait_event(self._compute_done[b])
        return b

    def release(self, b: int, compute_stream: torch.cuda.Stream) -> None:
        evt = torch.cuda.Event()
        compute_stream.record_event(evt)
        self._compute_done[b] = evt


class Muon(AdamW, IProcessGroupEnabled, IMixedPrecisionEnabled):
    def __init__(
        self,
        params: Union[Iterable[torch.Tensor], Iterable[dict]],
        lr: float = 1e-3,
        lr_muon: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        betas_muon: Tuple[float, float] = (0.95, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 1e-5,
        weight_decay_muon: float = 1e-5,
        use_normuon: bool = True,
        normuon_row_only: bool = False,
        use_polar_express: bool = False,
        inclusive_substrings: Iterable[str] = ("att", "mlp", "experts.weight"),
        exclusive_substrings: Iterable[str] = ("norm", "bias"),
        num_buffers: int = 3,
        num_moe_experts: Optional[int] = None,
        expert_inclusive_substrings: Iterable[str] = ("experts.weight",),
        model_hidden_size: Optional[int] = None,
        expert_hidden_size: Optional[int] = None,
        split_expert_gate_up: bool = False,
        moe_impl: Optional[str] = None,
        use_new_expert_weight_layout: bool = False,
        communicate_in_bf16: bool = False,
    ):
        assert getattr(composer_dist, "get_pp_group_size", lambda: None)() is None
        assert moe_impl in ('DeepseekGMMMoeBlock', 'SonicGMMMoeBlock', None)
        assert composer_dist.get_tp_group_size() in (1, None)
        assert num_buffers < 5, "Damn bro, that's a lot of buffers, R U ok?"
        assert num_buffers >= 2, "Single buffer would feel lonely, please use at least 2"

        super().__init__(
            params=params, lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, amsgrad=False,
        )

        for group in self.param_groups:
            group["initial_lr"] = group["lr"]
            group["betas_muon"] = betas_muon

        self.amsgrad = False
        self.skip_updates = 0
        self.total_updates = 0
        self.lr_muon = lr_muon
        self.muon_weight_decay = weight_decay_muon
        self.use_normuon = use_normuon
        self.normuon_row_only = normuon_row_only
        self.use_polar_express = use_polar_express
        self.inclusive_substrings = tuple(inclusive_substrings)
        self.exclusive_substrings = tuple(exclusive_substrings)

        self.num_moe_experts = num_moe_experts
        self.expert_inclusive_substrings = tuple(expert_inclusive_substrings)
        self.model_hidden_size = model_hidden_size
        self.expert_hidden_size = expert_hidden_size
        self.split_expert_gate_up = split_expert_gate_up
        self.moe_impl = moe_impl
        self.use_new_expert_weight_layout = use_new_expert_weight_layout
        self.num_buffers = num_buffers
        self.communicate_in_bf16 = communicate_in_bf16

        self.params_offsets: List[List[int]] = []
        self.params_numels: List[List[int]] = []
        self.params_shapes: List[List[Tuple]] = []
        self.params_fqns: List[List[str]] = []

        self._param_has_muon: List[bool] = []
        self._param_has_adam: List[bool] = []
        self._muon_eas_packed_offsets: List[List[int]] = []
        self._muon_eas_total_numels: List[int] = []
        self._max_muon_numel: int = 0

        self._should_apply_muon_cache: Dict[str, bool] = {}
        self._logical_shapes_cache: Dict[str, Tuple[int, ...]] = {}
        self._optim_state_dtype: Optional[torch.dtype] = None
        self._needs_reinit = True

    @property
    def _state_dtype(self) -> torch.dtype:
        if self._optim_state_dtype is None:
            return torch.float32
        return self._optim_state_dtype

    def adam_update_step(
        self, unshard_param, unshard_grad, local_ea, local_eas,
        step, beta1, beta2, lr, initial_lr, weight_decay, eps,
    ) -> None:
        compute_dtype = unshard_param.dtype

        if weight_decay != 0:
            decay_factor = (lr / initial_lr) if initial_lr else 1.0
            unshard_param.mul_(1 - decay_factor * weight_decay)

        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step

        grad = unshard_grad.to(compute_dtype) if unshard_grad.dtype != compute_dtype else unshard_grad

        local_ea.mul_(beta1).add_(grad, alpha=1 - beta1)
        local_eas.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        denom = (local_eas.sqrt() / math.sqrt(bias_correction2)).add_(eps)
        step_size = lr / bias_correction1
        unshard_param.addcdiv_(local_ea, denom, value=-step_size)

    def _get_logical_shape(self, shape, fqn):
        if fqn in self._logical_shapes_cache:
            return self._logical_shapes_cache[fqn]
        if self.num_moe_experts is None or self.model_hidden_size is None:
            return shape
        if self.use_new_expert_weight_layout and len(shape) == 3:
            for sub in self.expert_inclusive_substrings:
                if sub in fqn:
                    E = self.num_moe_experts
                    I = self.expert_hidden_size
                    if self.split_expert_gate_up and I is not None and shape[1] == 2 * I:
                        logical = (2 * E, I, shape[2])
                        self._logical_shapes_cache[fqn] = logical
                        return logical
                    return shape
            return shape
        if len(shape) != 2:
            return shape
        for sub in self.expert_inclusive_substrings:
            if sub in fqn:
                E = self.num_moe_experts
                H = self.model_hidden_size
                I = self.expert_hidden_size
                total = shape[0] * shape[1]
                assert total % (E * H) == 0, (
                    f"Cannot unfuse {fqn} shape {shape} with {E} experts and hidden {H}"
                )
                other = total // (E * H)

                # w1 has gate+up fused, so other == 2*I; optionally split them
                # for independent orthogonalization.
                if self.split_expert_gate_up and I is not None and other == 2 * I:
                    if self.moe_impl == 'SonicGMMMoeBlock':
                        logical = (2 * E, I, H)
                    else:  # g_gemm
                        logical = (2 * E, H, I)
                else:
                    # w2 or w1 without expert_hidden_size
                    if self.moe_impl == 'SonicGMMMoeBlock':
                        logical = (E, other, H) if shape[0] <= shape[1] else (E, H, other)
                    else:
                        logical = (E, H, other) if shape[0] <= shape[1] else (E, other, H)

                self._logical_shapes_cache[fqn] = logical
                return logical
        return shape

    def _maybe_unfuse_experts(self, param, fqn):
        if fqn not in self._logical_shapes_cache:
            return param
        if param.ndim != 2 and not (self.use_new_expert_weight_layout and param.ndim == 3):
            return param
        logical = self._logical_shapes_cache[fqn]

        E = self.num_moe_experts
        if self.split_expert_gate_up and self.expert_hidden_size is not None and logical[0] == 2 * E:  # type: ignore
            H = self.model_hidden_size
            I = self.expert_hidden_size
            if self.use_new_expert_weight_layout:
                # (E, 2*I, H) -> (2*E, I, H)
                return param.reshape(2 * E, I, H)  # type: ignore
            elif self.moe_impl == 'SonicGMMMoeBlock':
                # sonic interleaves [g0,u0,g1,u1,...] per expert, so memory order
                # is already (E, 2, I, H) — just reshape.
                return param.reshape(2 * E, I, H)  # type: ignore
            else:  # g_gemm
                # g_gemm chunks [gate_0..gate_{I-1}, up_0..up_{I-1}] per expert,
                # so we need a permute to group by gate/up.
                return param.view(E, H, 2, I).permute(0, 2, 1, 3).reshape(2 * E, H, I)  # type: ignore

        return param.view(*logical)

    def _maybe_refuse_experts(self, param, fqn):
        """Inverse of _maybe_unfuse_experts: returns a contiguous flat (1-D) tensor
        in the original physical memory order."""
        if fqn not in self._logical_shapes_cache:
            return param.contiguous().view(-1)

        logical = self._logical_shapes_cache[fqn]

        E = self.num_moe_experts
        if self.split_expert_gate_up and self.expert_hidden_size is not None and logical[0] == 2 * E:  # type: ignore
            H = self.model_hidden_size
            I = self.expert_hidden_size
            param = param.contiguous()
            if self.use_new_expert_weight_layout:
                # (2*E, I, H) contiguous -> (E, 2*I, H) memory order.
                return param.view(-1)
            elif self.moe_impl == 'SonicGMMMoeBlock':
                # (2*E, I, H) contiguous is already sonic memory order (E, 2, I, H).
                return param.view(-1)
            else:  # g_gemm
                # (2*E, H, I) -> (E, 2, H, I) -> permute -> (E, H, 2, I) -> flatten
                return param.view(E, 2, H, I).permute(0, 2, 1, 3).contiguous().view(-1)  # type: ignore

        return param.contiguous().view(-1)

    def _apply_variance_reduction(self, v_chunk, second_momentum_buffer, beta2):
        if self.normuon_row_only:
            red_dim = -1
        else:
            red_dim = -1 if v_chunk.shape[-2] >= v_chunk.shape[-1] else -2
        compute_dtype = self.mixed_precision.reduce_dtype or torch.float32

        v_mean = v_chunk.to(compute_dtype).square().mean(dim=red_dim, keepdim=True)
        red_dim_size = v_chunk.size(red_dim)

        v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True).mul_(red_dim_size)
        v_norm = v_norm_sq.sqrt_()

        second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
        step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt_()

        scaled_sq_sum = (v_mean * red_dim_size) * step_size.to(compute_dtype).square()
        v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt_()

        final_scale = step_size * (v_norm / v_norm_new.clamp_min_(1e-10))
        return v_chunk.mul_(final_scale.type_as(v_chunk))

    def muon_update_step(
        self, local_param, full_update, full_update_fqn, local_muon_eas,
        beta2, lr, initial_lr, muon_weight_decay, local_offset, local_numel,
    ) -> None:
        decay_factor = (lr / initial_lr) if initial_lr else 1.0
        lr_muon = self.lr_muon * decay_factor

        is_large = max(full_update.shape[-2:]) > 1024
        O = orthogonalize(
            full_update, split_baddbmm=is_large,
            use_polar=self.use_polar_express,
            math_dtype=self.mixed_precision.reduce_dtype,
        )

        if self.use_normuon:
            O = self._apply_variance_reduction(O, local_muon_eas, beta2)

        scale = 0.2 * (max(full_update.shape[-2], full_update.shape[-1]) ** 0.5)

        if muon_weight_decay != 0:
            local_param.mul_(1 - muon_weight_decay * decay_factor)

        O_flat = self._maybe_refuse_experts(O, full_update_fqn)
        local_update = O_flat[local_offset : local_offset + local_numel]
        local_param.add_(local_update.to(local_param.dtype), alpha=-lr_muon * scale)

    def should_apply_muon(self, param_fqn: str) -> bool:
        cached = self._should_apply_muon_cache.get(param_fqn)
        if cached is not None:
            return cached
        val = (
            any(sub in param_fqn for sub in self.inclusive_substrings)
            and not any(sub in param_fqn for sub in self.exclusive_substrings)
        )
        self._should_apply_muon_cache[param_fqn] = val
        return val

    def _get_second_momentum_shape(self, param_shape):
        if len(param_shape) == 2:
            h, w = param_shape
            if self.normuon_row_only:
                return (h, 1)
            return (h, 1) if h >= w else (1, w)
        if len(param_shape) == 3:
            batch, h, w = param_shape
            if self.normuon_row_only:
                return (batch, h, 1)
            return (batch, h, 1) if h >= w else (batch, 1, w)
        raise ValueError(f"Unsupported param shape: {param_shape}")

    def _get_second_momentum_numel(self, param_shape):
        return math.prod(self._get_second_momentum_shape(param_shape))

    def _ensure_state_for_param(self, param_idx: int, param: torch.Tensor) -> None:
        state = self.state
        state_dtype = self._state_dtype
        has_adam = self._param_has_adam[param_idx]
        has_muon = self._param_has_muon[param_idx]

        if has_adam:
            if f"exp_avg_{param_idx}" not in state:
                state[f"exp_avg_{param_idx}"] = torch.zeros_like(param, dtype=state_dtype)
            if f"exp_avg_sq_{param_idx}" not in state:
                state[f"exp_avg_sq_{param_idx}"] = torch.zeros_like(param, dtype=state_dtype)

        if has_muon:
            if f"muon_eas_{param_idx}" not in state:
                muon_eas_size = self._muon_eas_total_numels[param_idx]
                state[f"muon_eas_{param_idx}"] = (
                    torch.zeros(muon_eas_size, device=param.device, dtype=state_dtype)
                    if muon_eas_size > 0
                    else torch.zeros(0, device=param.device, dtype=state_dtype)
                )
            if f"muon_ea_{param_idx}" not in state:
                state[f"muon_ea_{param_idx}"] = torch.zeros_like(param, dtype=state_dtype)

    def _launch_muon_gather(self, param_idx, param_fqns, param, U_buf, muon_beta1):
        buf_dtype = U_buf.dtype

        total_numel = sum(self.params_numels[param_idx])
        local_world_size = self.get_fsdp_pg_world_size(param_fqns[0])
        padded_total = ((total_numel + local_world_size - 1) // local_world_size) * local_world_size
        assert param.numel() * local_world_size == padded_total, (
            f"{total_numel = } {padded_total = } {param.numel() = } "
            f"{local_world_size = }, {param_fqns = }"
        )

        muon_ea_local = self.state[f"muon_ea_{param_idx}"]
        ea_dtype = muon_ea_local.dtype
        grad_cast = param.grad.to(ea_dtype) if param.grad.dtype != ea_dtype else param.grad

        muon_ea_local.lerp_(grad_cast, 1 - muon_beta1)

        # lerp(grad, ea, beta1) = (1 - beta1)*grad + beta1*ea  (Nesterov combine).
        # Element-wise, so local combine + gather = gather + combine.
        combined = torch.lerp(grad_cast, muon_ea_local, muon_beta1)
        return combined.to(buf_dtype) if combined.dtype != buf_dtype else combined
        # ea_cast = muon_ea_local.to(buf_dtype) if muon_ea_local.dtype != buf_dtype else muon_ea_local
        # return torch.lerp(grad_cast, ea_cast, muon_beta1)

    def _process_muon_param(
        self, param_idx, param, param_offsets, param_numels, param_shapes,
        param_fqns, U_buf, adam_beta1, adam_beta2, muon_beta2, eps,
        lr, initial_lr, weight_decay, muon_weight_decay, state_step,
    ) -> None:
        total_numel = sum(param_numels)
        params_per_rank = param.numel()
        local_rank = self.get_fsdp_pg_rank(param_fqns[0])
        local_start = params_per_rank * local_rank
        local_end = local_start + params_per_rank

        exp_avg = self.state.get(f"exp_avg_{param_idx}")
        exp_avg_sq = self.state.get(f"exp_avg_sq_{param_idx}")
        muon_eas_packed = self.state.get(f"muon_eas_{param_idx}")

        U = U_buf[:total_numel]
        packed_offsets = self._muon_eas_packed_offsets[param_idx]
        state_dtype = self._state_dtype

        for chunk_idx, (offset, numel, shape, fqn) in enumerate(
            zip(param_offsets, param_numels, param_shapes, param_fqns)
        ):
            chunk_start = offset
            chunk_end = offset + numel
            if chunk_end <= local_start or chunk_start >= local_end:
                continue

            abs_lo = max(chunk_start, local_start)
            abs_hi = min(chunk_end, local_end)
            loc_lo = abs_lo - local_start
            loc_hi = abs_hi - local_start

            local_param_slice = param[loc_lo:loc_hi]
            needs_cast = local_param_slice.dtype != state_dtype
            local_param_buf = local_param_slice.to(state_dtype) if needs_cast else local_param_slice

            if self.should_apply_muon(fqn):
                chunk_U = self._maybe_unfuse_experts(
                    U[offset : offset + numel].view(*shape), fqn
                )
                logical_shape = chunk_U.shape
                second_mom_shape = self._get_second_momentum_shape(logical_shape)
                second_mom_numel = math.prod(second_mom_shape)
                packed_off = packed_offsets[chunk_idx]

                self.muon_update_step(
                    local_param=local_param_buf,
                    full_update=chunk_U,
                    full_update_fqn=fqn,
                    local_muon_eas=muon_eas_packed[
                        packed_off : packed_off + second_mom_numel
                    ].view(*second_mom_shape),
                    beta2=muon_beta2, lr=lr, initial_lr=initial_lr,
                    muon_weight_decay=muon_weight_decay,
                    local_offset=abs_lo - chunk_start,
                    local_numel=abs_hi - abs_lo,
                )
            else:
                self.adam_update_step(
                    unshard_param=local_param_buf,
                    unshard_grad=param.grad[loc_lo:loc_hi],
                    local_ea=exp_avg[loc_lo:loc_hi],
                    local_eas=exp_avg_sq[loc_lo:loc_hi],
                    step=state_step, beta1=adam_beta1, beta2=adam_beta2,
                    lr=lr, initial_lr=initial_lr,
                    weight_decay=weight_decay, eps=eps,
                )

            if needs_cast:
                local_param_slice.copy_(local_param_buf.to(local_param_slice.dtype))

    def _init_optimizer_state_once(self, group, device, dtype):
        if not self._needs_reinit:
            return

        self._optim_state_dtype = dtype

        assert "layernorm_parameter_orig_shapes" in group
        assert "main_parameter_orig_shapes" in group
        assert "layernorm_shards_fqns" in group
        assert "main_shards_fqns" in group
        assert "layernorm_shards_numels" in group
        assert "main_shards_numels" in group

        self.params_offsets = []
        self.params_numels = []
        self.params_shapes = group["layernorm_parameter_orig_shapes"] + group["main_parameter_orig_shapes"]
        self.params_fqns = group["layernorm_shards_fqns"] + group["main_shards_fqns"]

        for param_numels in (group["layernorm_shards_numels"] + group["main_shards_numels"]):
            self.params_offsets.append([0])
            self.params_numels.append([])
            for param_numel in param_numels[:-1]:
                self.params_offsets[-1].append(self.params_offsets[-1][-1] + param_numel)
                self.params_numels[-1].append(param_numel)
            self.params_numels[-1].append(param_numels[-1])

        self._param_has_muon = []
        self._param_has_adam = []
        for param_fqns in self.params_fqns:
            muon_flags = [self.should_apply_muon(fqn) for fqn in param_fqns]
            self._param_has_muon.append(any(muon_flags))
            self._param_has_adam.append(any(not f for f in muon_flags))

        # _get_logical_shape must run here so the cache is populated
        # before _maybe_unfuse_experts is called during step().
        self._muon_eas_packed_offsets = []
        self._muon_eas_total_numels = []
        for param_shapes, param_fqns, has_muon in zip(
            self.params_shapes, self.params_fqns, self._param_has_muon
        ):
            chunk_offsets: List[int] = []
            packed_offset = 0
            if has_muon:
                for shape, fqn in zip(param_shapes, param_fqns):
                    chunk_offsets.append(packed_offset)
                    if self.should_apply_muon(fqn):
                        logical_shape = self._get_logical_shape(shape, fqn)
                        packed_offset += self._get_second_momentum_numel(logical_shape)
            self._muon_eas_packed_offsets.append(chunk_offsets)
            self._muon_eas_total_numels.append(packed_offset)

        max_muon_numel = 0
        for has_muon, numels in zip(self._param_has_muon, self.params_numels):
            if has_muon:
                max_muon_numel = max(max_muon_numel, sum(numels))
        self._max_muon_numel = max_muon_numel

        if hasattr(self, "_pending_state") and self._pending_state:
            state_dtype = self._state_dtype

            for _idx, pstate in self._pending_state.items():
                if "step" in pstate:
                    self.state["step"] = int(pstate["step"])
                    break

            for _idx, pstate in self._pending_state.items():
                param_idx_int = int(_idx)
                for src_key, dst_prefix in [
                    ("exp_avg", "exp_avg_"),
                    ("exp_avg_sq", "exp_avg_sq_"),
                    ("muon_ea", "muon_ea_"),
                    ("muon_eas", "muon_eas_"),
                ]:
                    if src_key in pstate:
                        dst_key = f"{dst_prefix}{param_idx_int}"
                        self.state[dst_key] = pstate[src_key].to(device=device, dtype=state_dtype)
            del self._pending_state

        self._needs_reinit = False

    @torch.no_grad()
    def step(self, closure: Union[Callable, None] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        assert len(self.param_groups) == 1
        group = self.param_groups[0]

        first_param = group["params"][0]
        self._init_optimizer_state_once(group, first_param.device, first_param.dtype)

        adam_beta1, adam_beta2 = group["betas"]
        muon_beta1, muon_beta2 = group["betas_muon"]
        eps = group["eps"]
        lr = group["lr"]
        if "initial_lr" not in group:
            group["initial_lr"] = lr
        initial_lr = group["initial_lr"]
        weight_decay = group["weight_decay"]

        state = self.state
        state["step"] = int(state.get("step", 0)) + 1
        state_step = state["step"]
        state_dtype = self._state_dtype

        adam_immediate = []
        muon_queue = []
        for param_idx, (param, param_offsets, param_numels, param_shapes, param_fqns, has_muon) in enumerate(
            zip(
                group["params"], self.params_offsets, self.params_numels,
                self.params_shapes, self.params_fqns, self._param_has_muon,
            )
        ):
            self._ensure_state_for_param(param_idx, param)
            if has_muon:
                muon_queue.append((param_idx, param, param_offsets, param_numels, param_shapes, param_fqns))
            else:
                adam_immediate.append((param_idx, param))

        num_bufs = self.num_buffers
        comm_stream = self.communication_stream
        compute_stream = self.compute_stream

        if self._max_muon_numel > 0 and muon_queue:
            comm_dtype = torch.bfloat16 if self.communicate_in_bf16 else state_dtype
            max_gather_numel = max(
                qparam.numel() * self.get_fsdp_pg_world_size(qfqns[0])
                for _, qparam, _, _, _, qfqns in muon_queue
            )
            ring = BufferRing(num_bufs, max_gather_numel, first_param.device, comm_dtype)
            pending: deque = deque()

            def launch_gather(j: int) -> None:
                qj_param_idx, qj_param, _, _, _, qj_param_fqns = muon_queue[j]

                b = ring.acquire(comm_stream)
                local_combined = self._launch_muon_gather(
                    qj_param_idx, qj_param_fqns, qj_param, ring.bufs[b], muon_beta1,
                )

                compute_done = torch.cuda.Event()
                compute_stream.record_event(compute_done)
                comm_stream.wait_event(compute_done)

                pg = self.get_fsdp_process_group(qj_param_fqns[0])
                gather_numel = qj_param.numel() * dist.get_world_size(pg)
                U = ring.bufs[b][:gather_numel]

                with torch.cuda.stream(comm_stream):
                    dist.all_gather_into_tensor(U, local_combined, group=pg)
                    gather_done = torch.cuda.Event()
                    comm_stream.record_event(gather_done)

                pending.append((j, b, gather_done, local_combined))

            # Preflight: fill pipeline so first gather(s) overlap with Adam.
            frontier = 0
            while frontier < min(num_bufs - 1, len(muon_queue)):
                launch_gather(frontier)
                frontier += 1
        else:
            ring = None

        for param_idx, param in adam_immediate:
            exp_avg = state[f"exp_avg_{param_idx}"]
            exp_avg_sq = state[f"exp_avg_sq_{param_idx}"]

            needs_cast = param.dtype != state_dtype
            param_buf = param.to(state_dtype) if needs_cast else param

            with torch.cuda.stream(compute_stream):
                self.adam_update_step(
                    unshard_param=param_buf, unshard_grad=param.grad,
                    local_ea=exp_avg, local_eas=exp_avg_sq,
                    step=state_step, beta1=adam_beta1, beta2=adam_beta2,
                    lr=lr, initial_lr=initial_lr, weight_decay=weight_decay, eps=eps,
                )

            if needs_cast:
                param.copy_(param_buf.to(param.dtype))

        if ring is not None:
            for i in range(len(muon_queue)):
                qi, b, gather_done, _temps = pending.popleft()
                assert qi == i

                # Launch next gather before blocking so it overlaps with compute.
                if frontier < len(muon_queue):
                    launch_gather(frontier)
                    frontier += 1

                compute_stream.wait_event(gather_done)

                param_idx, param, param_offsets, param_numels, param_shapes, param_fqns = muon_queue[i]
                self._process_muon_param(
                    param_idx=param_idx, param=param,
                    param_offsets=param_offsets, param_numels=param_numels,
                    param_shapes=param_shapes, param_fqns=param_fqns,
                    U_buf=ring.bufs[b],
                    adam_beta1=adam_beta1, adam_beta2=adam_beta2,
                    muon_beta2=muon_beta2, eps=eps,
                    lr=lr, initial_lr=initial_lr,
                    weight_decay=weight_decay, muon_weight_decay=self.muon_weight_decay,
                    state_step=state_step,
                )

                ring.release(b, compute_stream)

        return loss

    def state_dict(self) -> Dict[str, Any]:
        muon_meta = {
            "lr_muon": self.lr_muon,
            "muon_weight_decay": self.muon_weight_decay,
            "use_normuon": self.use_normuon,
            "normuon_row_only": self.normuon_row_only,
            "use_polar_express": self.use_polar_express,
            "skip_updates": self.skip_updates,
            "total_updates": self.total_updates,
            "params_offsets": self.params_offsets,
            "params_numels": self.params_numels,
            "params_shapes": self.params_shapes,
            "params_fqns": self.params_fqns,
        }

        if hasattr(self, "_pending_state") and self._pending_state:
            return {
                "state": self._pending_state,
                "param_groups": [
                    {k: v for k, v in group.items() if k != "params"}
                    for group in self.param_groups
                ],
                "muon_meta": muon_meta,
            }

        params_state: Dict[int, Dict[str, Any]] = {}
        for param_idx, _param in enumerate(self.param_groups[0]["params"]):
            pstate: Dict[str, Any] = {"step": int(self.state.get("step", 0))}
            for key_prefix in ("exp_avg", "exp_avg_sq", "muon_ea", "muon_eas"):
                full_key = f"{key_prefix}_{param_idx}"
                if full_key in self.state:
                    pstate[key_prefix] = self.state[full_key].detach()
            params_state[param_idx] = pstate

        return {
            "state": params_state,
            "param_groups": [
                {k: v for k, v in group.items() if k != "params"}
                for group in self.param_groups
            ],
            "muon_meta": muon_meta,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        meta = state_dict.get("muon_meta", None)
        self._pending_state = state_dict.get("state", {})

        for saved_group, group in zip(state_dict.get("param_groups", []), self.param_groups):
            for k, v in saved_group.items():
                if k != "params":
                    group[k] = v

        if meta is not None:
            self.lr_muon = meta["lr_muon"]
            self.muon_weight_decay = meta["muon_weight_decay"]
            self.use_normuon = meta.get("use_normuon", True)
            self.normuon_row_only = meta.get("normuon_row_only", False)
            self.use_polar_express = meta.get("use_polar_express", False)
            self.skip_updates = meta["skip_updates"]
            self.total_updates = meta["total_updates"]
            self.params_offsets = meta["params_offsets"]
            self.params_numels = meta["params_numels"]
            self.params_shapes = meta["params_shapes"]
            self.params_fqns = meta["params_fqns"]

        self._needs_reinit = True
