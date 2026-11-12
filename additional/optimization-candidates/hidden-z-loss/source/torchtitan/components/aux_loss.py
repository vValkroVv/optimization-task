"""Per-step lifecycle of every term the step's loss decomposes into."""

from typing import Any

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
from torch import nn
from torch.distributed.tensor import DTensor

from torchtitan.distributed import ParallelDims
from torchtitan.models.common.regularization import wire_aux_loss

__all__ = ["AuxLossTracker"]


class AuxLossTracker:
    """Reset, weigh and report every term the step's loss decomposes into.

    Two producers, one owner: the grad-only aux/z splices of every model part,
    and the loss object's own components (CE, and the MTP terms when there are
    heads). They are kept apart inside because they are weighed differently --
    a splice is scaled by the microbatch's share of the token count, while a
    loss component arrives already divided by ``global_valid_tokens`` -- but
    they open, reduce and get named in one place.

    ``microbatch_splits`` is the division ``build_pipeline_schedule`` made (torch
    exposes it only as ``_n_microbatches``): under PP the schedule cuts ``labels``
    again and each cut fires the splices once.
    """

    def __init__(
        self,
        model_parts: list[nn.Module],
        *,
        parallel_dims: ParallelDims,
        device: torch.device,
        microbatch_splits: int = 1,
        loss_fn: Any = None,
    ) -> None:
        if microbatch_splits < 1:
            raise ValueError(
                f"microbatch_splits must be >= 1, got {microbatch_splits}."
            )
        self._parallel_dims = parallel_dims
        self._splits = microbatch_splits
        self._loss_fn = loss_fn
        # Declared, not discovered: see BaseLoss.declared_components.
        self._loss_names: tuple[str, ...] = (
            tuple(loss_fn.declared_components()) if loss_fn is not None else ()
        )
        pp_mesh = (
            parallel_dims.get_optional_mesh("pp") if parallel_dims.pp_enabled else None
        )
        self._states = [
            wire_aux_loss(
                part,
                device=device,
                pp_group=None if pp_mesh is None else pp_mesh.get_group(),
            )
            for part in model_parts
        ]

    def begin_step(self) -> None:
        """Open a step: zero the splices and the loss's own decomposition.

        One "start of step" for every term of the loss, which is the point of
        this object owning both: the totals carry between logs otherwise, and
        the trainer used to reset the two halves in two places.
        """
        for state in self._states:
            state.reset()
        if self._loss_fn is not None:
            self._loss_fn.reset_components()

    def keep_microbatches(self, losses: list[torch.Tensor]) -> None:
        """Drop the loss calls that were not microbatches of this step.

        On its first ``schedule.step`` the pipeline runs the loss once more on
        the last stage to learn the gradient shapes; that call never enters the
        schedule's ``losses``, and where it lands among the real microbatches
        is a torch implementation detail — so the calls are matched against
        ``losses`` themselves, not counted.
        """
        if self._loss_fn is not None:
            self._loss_fn.keep_microbatches(losses)

    def weigh(
        self, *, valid_tokens: torch.Tensor, global_valid_tokens: torch.Tensor
    ) -> None:
        """Weigh by this microbatch's share of the global token count -- the
        denominator the main loss uses."""
        share = valid_tokens / global_valid_tokens / self._splits
        for state in self._states:
            state.scale.copy_(share)

    def _loss_metrics(self) -> dict[str, float]:
        """Reduce the loss components together over the loss mesh.

        Only the final pipeline stage produces these components. Read the
        reduced vector once instead of synchronizing the host per component.
        """
        if self._loss_fn is None:
            return {}
        produced = self._loss_fn.components()
        names = [name for name in self._loss_names if produced.get(name) is not None]
        if not names:
            return {}
        mesh = (
            self._parallel_dims.get_optional_mesh("loss")
            if self._parallel_dims.dp_cp_enabled
            else None
        )
        if mesh is None:
            return {f"loss/train/{name}": float(produced[name]) for name in names}

        local_values = []
        for name in names:
            value = produced[name].detach()
            if isinstance(value, DTensor):
                assert all(
                    p.is_replicate() or p.is_partial() for p in value.placements
                ), (
                    "Loss metrics require Replicate/Partial DTensors, "
                    f"got {value.placements}"
                )
                value = value.to_local()
            local_values.append(value.reshape(()))
        values = funcol.all_reduce(
            torch.stack(local_values), reduceOp=c10d.ReduceOp.SUM.name, group=mesh
        )
        return {
            f"loss/train/{name}": value for name, value in zip(names, values.tolist())
        }

    def metrics(self) -> dict[str, float]:
        """Reduce every term of the step's loss and name it.

        Both halves of the decomposition leave through here: the grad-only
        splices (``aux/*``, never added to ``loss`` -- the sum over microbatches
        and ranks is what the gradient saw) and the loss's own components
        (``loss/train/*``, already normalized by ``global_valid_tokens``).

        Deliberately published through the trainer's ``metrics_processor``,
        not the callback emitter: these are core telemetry of the loss, they
        exist with callbacks disabled (the emitter then does not run at all),
        and they live on the trainer's ``log_freq`` cadence, not on the
        emitter's every-step one.
        """
        parallel_dims = self._parallel_dims
        reported: dict[str, float] = {}
        for state in self._states:
            if not state.names:
                continue
            # Two groups: no mesh spans dp and pp but not tp, and the scalar is
            # already replicated across tp. Over pp the sum gathers -- each name
            # is non-zero on one stage. (dist_sum takes a lone scalar, not a vector.)
            groups = (
                [parallel_dims.get_optional_mesh("loss")]
                if parallel_dims.dp_cp_enabled
                else []
            )
            if parallel_dims.pp_enabled:
                groups.append(parallel_dims.get_optional_mesh("pp"))
            values = state.totals
            for group in filter(None, groups):
                values = funcol.all_reduce(
                    values, reduceOp=c10d.ReduceOp.SUM.name, group=group
                )
            for name, value in zip(state.names, values.tolist()):
                reported[f"aux/{name}"] = value
                # Per-layer says where a term grows; the `kind` total, summed from
                # the reduced numbers, is what compares against the loss.
                key = f"aux/{name.split('/', 1)[0]}"
                reported[key] = reported.get(key, 0.0) + value
        return reported | self._loss_metrics()
