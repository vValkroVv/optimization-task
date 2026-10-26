# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Stateless learning rate schedulers.

Stateless schedulers solve some of the problems associated with PyTorch's built-in schedulers provided in
:mod:`torch.optim.lr_scheduler`. The primary design goal of the schedulers provided in this module is to allow
schedulers to interface directly with Composer's :mod:`~composer.core.time` abstraction. This means that schedulers can
be configured using arbitrary but explicit time units.

See :class:`~.ComposerScheduler` for more information on stateless schedulers.
"""

import inspect
import logging
import math
import textwrap
import warnings
from typing import TYPE_CHECKING, Callable, List, Union

from torch.optim.lr_scheduler import LambdaLR

from composer.core import PyTorchScheduler, State, Time, TimeUnit

if TYPE_CHECKING:
    from typing import Protocol
else:
    # subclasses of Protocol cannot be instantiated in Python 3.8
    Protocol = object

log = logging.getLogger(__name__)

__all__ = [
    'ComposerScheduler', 'compile_composer_scheduler', 'StepScheduler', 'MultiStepScheduler', 'ConstantScheduler',
    'LinearScheduler', 'ExponentialScheduler', 'CosineAnnealingScheduler', 'CosineAnnealingWarmRestartsScheduler',
    'PolynomialScheduler', 'MultiStepWithWarmupScheduler', 'ConstantWithWarmupScheduler', 'LinearWithWarmupScheduler',
    'CosineAnnealingWithWarmupScheduler', 'PolynomialWithWarmupScheduler', 'LinearWithStartTimeScheduler',
    'InvSquareRootScheduler', 'InvSquareRootWithWarmupAndCooldownScheduler', 'ConstantMultiStepWithGammaWithWarmupScheduler'
]


class ComposerScheduler(Protocol):
    r"""Specification for a stateless scheduler function.

    While this specification is provided as a Python class, an ordinary function can implement this interface as long
    as it matches the signature of this interface's :meth:`~.ComposerScheduler.__call__` method.

    For example, a scheduler that halves the learning rate after 10 epochs could be written as:

    .. code:: python

        def ten_epoch_decay_scheduler(state: State) -> float:
            if state.timestamp.epoch < 10:
                return 1.0
            return 0.5

        # ten_epoch_decay_scheduler is a valid ComposerScheduler
        trainer = Trainer(
            schedulers=[ten_epoch_decay_scheduler],
            ...
        )

    In order to allow schedulers to be configured, schedulers may also written as callable classes:

    .. code:: python

        class VariableEpochDecayScheduler(ComposerScheduler):

            def __init__(num_epochs: int):
                self.num_epochs = num_epochs

            def __call__(state: State) -> float:
                if state.time.epoch < self.num_epochs:
                    return 1.0
                return 0.5

        ten_epoch_decay_scheduler = VariableEpochDecayScheduler(num_epochs=10)
        # ten_epoch_decay_scheduler is also a valid ComposerScheduler
        trainer = Trainer(
            schedulers=[ten_epoch_decay_scheduler],
            ...
        )

    The constructions of ``ten_epoch_decay_scheduler`` in each of the examples above are equivalent. Note that neither
    scheduler uses the ``scale_schedule_ratio`` parameter. As long as this parameter is not used when initializing
    :class:`.Trainer`, it is not required that any schedulers implement that parameter.

    .. automethod:: __call__
    """

    def __call__(self, state: State, ssr: float = 1.0) -> float:
        r"""Calculate the current learning rate multiplier :math:`\alpha`.

        A scheduler function should be a pure function that returns a multiplier to apply to the optimizer's provided
        learning rate, given the current trainer state, and optionally a "scale schedule ratio" (SSR). A typical
        implementation will read ``state.timestamp``, and possibly other fields like ``state.max_duration``, to determine
        the trainer's latest temporal progress.

        .. note::
            All instances of :class:`~.ComposerScheduler` output a `multiplier` for the learning rate, rather than the
            learning rate directly. By convention, we use the symbol :math:`\alpha` to refer to this multiplier. This
            means that the learning rate :math:`\eta` at time :math:`t` can be represented as
            :math:`\eta(t) = \eta_i \times \alpha(t)`, where :math:`\eta_i` represents the learning rate used to
            initialize the optimizer.

        .. note::
            It is possible to use multiple schedulers, in which case their effects will stack multiplicatively.

        The ``ssr`` param indicates that the schedule should be "stretched" accordingly. In symbolic terms, where
        :math:`\alpha_\sigma(t)` represents the scheduler output at time :math:`t` using scale schedule ratio
        :math:`\sigma`:

        .. math::
            \alpha_{\sigma}(t) = \alpha(t / \sigma)

        Args:
            state (State): The current Composer Trainer state.
            ssr (float): The scale schedule ratio. In general, the learning rate computed by this
                scheduler at time :math:`t` with an SSR of 1.0 should be the same as that computed by
                this scheduler at time :math:`t \times s` with an SSR of :math:`s`. Default = ``1.0``.

        Returns:
            alpha (float): A multiplier to apply to the optimizer's provided learning rate.
        """
        raise NotImplementedError


def _convert_time(time: Union[str, Time[int], Time[float]], state: State, ssr: float = 1.0) -> Time[int]:
    if isinstance(time, str):
        time = Time.from_timestring(time)

    assert state.max_duration is not None, 'max_duration should be set whenever schedulers are invoked'

    if time.unit == TimeUnit.DURATION:
        if state.max_duration.unit == TimeUnit.EPOCH:
            if state.dataloader_len is None:
                raise RuntimeError('Cannot convert time, as state.dataloader_len is None.')
            return Time(int(time.value * int(state.dataloader_len) * state.max_duration.value), TimeUnit.BATCH)
        return Time(int(time.value * state.max_duration.value), state.max_duration.unit)
    elif time.unit == TimeUnit.EPOCH:
        # Epochs do not provide sufficient granularity for SSR scaling
        # e.g. if max_duration = 1ep, then any SSR would result in a new duration of 0.
        # so, convert the time into batches
        if state.dataloader_len is None:
            raise RuntimeError('Cannot convert time, as state.dataloader_len is None.')
        time = Time(value=time.value * int(state.dataloader_len), unit=TimeUnit.BATCH)

    return Time(value=int(time.value * ssr), unit=time.unit)


def _compose_lambda_for_scheduler(
    scheduler: ComposerScheduler,
    state: State,
    ssr: float,
) -> Callable[[int], float]:
    """Build the epoch -> LR multiplier closure used by :class:`torch.optim.lr_scheduler.LambdaLR`."""
    scheduler_sig = inspect.signature(scheduler)

    def scheduler_fn(epoch: int) -> float:
        del epoch  # unused. Provided by the pytorch LambdaLR

        # if the ssr is 1.0, don't pass it to the scheduler. This allows users to pass in lambdas that only take
        # one parameter -- the state
        if len(scheduler_sig.parameters) == 1:
            if ssr == 1.0:
                return scheduler(state)
            else:
                raise ValueError(
                    textwrap.dedent(f"""\
                    Scheduler {scheduler} does not support `scale_schedule_ratio`.
                    To use `scale_schedule_ratio`, the scheduler must take two arguments (state, ssr)"""))
        return scheduler(state, ssr)

    return scheduler_fn


def compile_composer_scheduler(scheduler: ComposerScheduler, state: State, ssr: float = 1.0) -> PyTorchScheduler:
    """Converts a stateless scheduler into a PyTorch scheduler object.

    While the resulting scheduler provides a ``.step()`` interface similar to other PyTorch schedulers, the scheduler is
    also given a bound reference to the current :class:`~composer.core.State`. This means that any internal state updated
    by ``.step()`` can be ignored, and the scheduler can instead simply use the bound state to recalculate the current
    learning rate.

    Args:
        scheduler (ComposerScheduler): A stateless scheduler, provided as a :class:`~.ComposerScheduler` object.
        state (State): The Composer Trainer's state.

    Returns:
        compiled_scheduler (PyTorchScheduler): The scheduler, in a form compatible with PyTorch scheduler interfaces.
    """
    optimizers = state.optimizers
    if len(optimizers) != 1:
        raise NotImplementedError('Providing functional schedulers is unsupported with multiple optimizers.')
    optimizer = optimizers[0]

    # Per-optimizer-param-group schedulers (e.g. llmfoundry PerGroupComposerScheduler): one LambdaLR multiplier per group.
    if hasattr(scheduler, 'per_group_schedulers') and hasattr(scheduler, 'get_scheduler_for_group'):
        lr_lambdas = [
            _compose_lambda_for_scheduler(
                scheduler.get_scheduler_for_group(group_idx),
                state,
                ssr,
            ) for group_idx in range(len(optimizer.param_groups))
        ]
        return LambdaLR(optimizer, lr_lambdas)

    return LambdaLR(
        optimizer,
        _compose_lambda_for_scheduler(scheduler, state, ssr),
    )


class StepScheduler(ComposerScheduler):
    r"""Decays the learning rate discretely at fixed intervals.

    .. seealso::
        This scheduler is based on :class:`~torch.optim.lr_scheduler.StepLR` from PyTorch.

    Decays the learning rate by a factor of ``gamma`` periodically, with a frequency determined by ``step_size``.

    Specifically, the learning rate multiplier :math:`\alpha` can be expressed as:

    .. math::
        \alpha(t) = \gamma ^ {\text{floor}(t / \rho)}

    Where :math:`\rho` represents the time between changes to the learning rate (the step size), and
    :math:`\gamma` represents the multiplicative decay factor.

    Args:
        step_size (str | Time): Time between changes to the learning rate.
        gamma (float): Multiplicative decay factor. Default = ``0.1``.
    """

    def __init__(self, step_size: Union[str, Time], gamma: float = 0.1):
        self.step_size = step_size
        self.gamma = gamma

    def __call__(self, state: State, ssr: float = 1.0):
        step_size = _convert_time(self.step_size, state, ssr=ssr)
        current_time = state.timestamp.get(step_size.unit)
        steps = int(current_time / step_size)

        return self.gamma**steps


class MultiStepScheduler(ComposerScheduler):
    r"""Decays the learning rate discretely at fixed milestones.

    .. seealso::
        This scheduler is based on :class:`~torch.optim.lr_scheduler.MultiStepLR` from PyTorch.

    Decays the learning rate by a factor of ``gamma`` whenever a time milestone in ``milestones`` is reached.

    Specifically, the learning rate multiplier :math:`\alpha` can be expressed as:

    .. math::
        \alpha(t) = \gamma ^ x

    Where :math:`x` represents the amount of milestones that have been reached, and :math:`\gamma` represents the
    multiplicative decay factor.

    Args:
        milestones (List[str | Time]): Times at which the learning rate should change.
        gamma (float): Multiplicative decay factor. Default = ``0.1``.
    """

    def __init__(self, milestones: List[Union[str, Time]], gamma: float = 0.1):
        self.milestones = milestones
        self.gamma = gamma

    def __call__(self, state: State, ssr: float = 1.0):
        milestones = [_convert_time(milestone, state, ssr=ssr) for milestone in self.milestones]

        factor = 1.0
        for milestone in milestones:
            if state.timestamp >= milestone:
                factor *= self.gamma

        return factor


class ConstantScheduler(ComposerScheduler):
    r"""Maintains a fixed learning rate.

    This scheduler is based on  :class:`~torch.optim.lr_scheduler.ConstantLR` from PyTorch.

    The default settings for this scheduler simply maintain a learning rate factor of 1 for the entire training
    duration. However, both the factor and the duration of this scheduler can be configured.

    Specifically, the learning rate multiplier :math:`\alpha` can be expressed as:

    .. math::
        \alpha(t) = \begin{cases} \alpha, & \text{if } t < t_{max} \\ 1.0 & \text{otherwise} \end{cases}

    Where :math:`\alpha` represents the learning rate multiplier to maintain while this scheduler is active, and
    :math:`t_{max}` represents the duration of this scheduler.

    Args:
        alpha (float): Learning rate multiplier to maintain while this scheduler is active. Default = ``1.0``.
        t_max (str | Time): Duration of this scheduler. Default = ``"1dur"``.
    """

    def __init__(self, alpha: float = 1.0, t_max: Union[str, Time] = '1dur') -> None:
        self.alpha = alpha
        self.t_max = t_max

    def __call__(self, state: State, ssr: float = 1.0) -> float:
        t_max = _convert_time(self.t_max, state, ssr=ssr)

        if state.timestamp < t_max:
            return self.alpha

        return 1.0


class LinearScheduler(ComposerScheduler):
    r"""Adjusts the learning rate linearly.

    .. seealso::
        This scheduler is based on :class:`~torch.optim.lr_scheduler.LinearLR` from PyTorch.

    .. warning::
        Note that the defaults for this scheduler differ from the defaults for
        :class:`~torch.optim.lr_scheduler.LinearLR`. The PyTorch scheduler, by default, linearly increases the learning
        rate multiplier from 1.0 / 3 to 1.0, whereas this implementation, by default, linearly decreases the multiplier
        rom 1.0 to 0.0.

    Linearly adjusts the learning rate multiplier from ``alpha_i`` to ``alpha_f`` over ``t_{max}`` time.

    Specifically, the learning rate multiplier :math:`\alpha` can be expressed as:

    .. math::
        \alpha(t) = \alpha_i + (alpha_f - \alpha_i) \times \tau

    Given :math:`\tau`, the fraction of time elapsed (clipped to the interval :math:`[0, 1]`), as:

    .. math::
        \tau = t / t_{max}

    Where :math:`\alpha_i` represents the initial learning rate multiplier, :math:`\alpha_f` represents
    the learning rate multiplier to decay to, and :math:`t_{max}` represents the duration of this scheduler.

    Args:
        alpha_i (float): Initial learning rate multiplier. Default = ``1.0``.
        alpha_f (float): Final learning rate multiplier. Default = ``0.0``.
        t_max (str | Time): The duration of this scheduler. Default = ``"1dur"``.
    """

    def __init__(self, alpha_i: float = 1.0, alpha_f: float = 0.0, t_max: Union[str, Time] = '1dur'):
        self.alpha_i = alpha_i
        self.alpha_f = alpha_f
        self.t_max = Time.from_timestring(t_max) if isinstance(t_max, str) else t_max

    def __call__(self, state: State, ssr: float = 1.0):
        t_max = _convert_time(self.t_max, state, ssr=ssr)
        current_time = state.timestamp.get(t_max.unit)
        frac_of_total = min(1.0, (current_time / t_max).value)

        current_factor = self.alpha_i + frac_of_total * (self.alpha_f - self.alpha_i)

        return current_factor


class ExponentialScheduler(ComposerScheduler):
    r"""Decays the learning rate exponentially.

    .. seealso::
        This scheduler is based on :class:`~torch.optim.lr_scheduler.ExponentialLR` from PyTorch.

    Exponentially decays the learning rate such that it decays by a factor of ``gamma`` every ``decay_period`` time.

    Specifically, the learning rate multiplier :math:`\alpha` can be expressed as:

    .. math::
        \alpha(t) = \gamma ^ {t / \rho}

    Where :math:`\rho` represents the decay period, and :math:`\gamma` represents the multiplicative decay factor.

    Args:
        decay_period (str | Time): Decay period. Default = ``"1ep"``.
        gamma (float): Multiplicative decay factor.
    """

    def __init__(self, gamma: float, decay_period: Union[str, Time] = '1ep'):
        self.gamma = gamma
        self.decay_period = decay_period

    def __call__(self, state: State, ssr: float = 1.0):
        decay_period = _convert_time(self.decay_period, state, ssr)
        current_time_in_decay_units = state.timestamp.get(decay_period.unit)

        return self.gamma**float(current_time_in_decay_units / decay_period)


def _cosine_anneal(x: float, min_y: float = 0.0, max_y: float = 1.0) -> float:
    """Implements a cosine decay curve.

    Curve is cos(x) on domain [0, pi], stretched to the domain [0, 1] and range [min_y, max_y]. Additionally, param x is
    clipped to the interval [0, 1]
    """
    x = min(max(x, 0.0), 1.0)
    return min_y + (max_y - min_y) * (1 + math.cos(x * math.pi)) / 2


class CosineAnnealingScheduler(ComposerScheduler):
    r"""Decays the learning rate according to the decreasing part of a cosine curve.

    .. seealso::
        This scheduler is based on :class:`~torch.optim.lr_scheduler.CosineAnnealingLR` from PyTorch.

    Specifically, the learning rate multiplier :math:`\alpha` can be expressed as:

    .. math::
        \alpha(t) = \alpha_f + (1 - \alpha_f) \times \frac{1}{2} (1 + \cos(\pi \times \tau))

    Given :math:`\tau`, the fraction of time elapsed (clipped to the interval :math:`[0, 1]`), as:

    .. math::
        \tau = t / t_{max}

    Where :math:`t_{max}`
    represents the duration of this scheduler, and :math:`\alpha_f` represents the learning rate multiplier to decay to.

    Args:
        t_max (str | Time): The duration of this scheduler. Default = ``"1dur"``.
        alpha_f (float): Learning rate multiplier to decay to. Default = ``0.0``.
    """

    def __init__(self, t_max: Union[str, Time] = '1dur', alpha_f: float = 0.0):
        self.t_max = t_max
        self.alpha_f = alpha_f

    def __call__(self, state: State, ssr: float = 1.0):
        t_max = _convert_time(self.t_max, state, ssr=ssr)
        current_time = state.timestamp.get(t_max.unit)
        frac_of_total = (current_time / t_max).value

        return _cosine_anneal(x=frac_of_total, min_y=self.alpha_f)


class CosineAnnealingWarmRestartsScheduler(ComposerScheduler):
    r"""Cyclically decays the learning rate according to the decreasing part of a cosine curve.

    .. seealso::
        This scheduler is based on :class:`~torch.optim.lr_scheduler.CosineAnnealingWarmRestarts` from PyTorch.

    This scheduler resembles a regular cosine annealing curve, as seen in :class:`~.CosineAnnealingScheduler`, except
    that after the curve first completes ``t_0`` time, the curve resets to the start. The durations of subsequent cycles
    are each multiplied by ``t_mult``.

    Specifically, the learning rate multiplier :math:`\alpha` can be expressed as:

    .. math::
        \alpha(t) = \alpha_f + (1 - \alpha_f) \times \frac{1}{2}(1 + \cos(\pi \times \tau_i))

    Given :math:`\tau_i`, the fraction of time elapsed through the :math:`i^\text{th}` cycle, as:

    .. math::
        \tau_i = (t - \sum_{j=0}^{i-1} t_0 t_{mult}^j) / (t_0 t_{mult}^i)

    Where :math:`t_0`
    represents the period of the first cycle, :math:`t_{mult}` represents the multiplier for the duration of successive
    cycles, and :math:`\alpha_f` represents the learning rate multiplier to decay to.

    Args:
        t_0 (str | Time): The period of the first cycle.
        t_mult (float): The multiplier for the duration of successive cycles. Default = ``1.0``.
        alpha_f (float): Learning rate multiplier to decay to. Default = ``0.0``.
    """

    def __init__(self, t_0: Union[str, Time], t_mult: float = 1.0, alpha_f: float = 0.0):
        self.t_0 = t_0
        self.t_mult = t_mult
        self.alpha_f = alpha_f

    def __call__(self, state: State, ssr: float = 1.0):
        t_0 = _convert_time(self.t_0, state, ssr=ssr)
        current_interval_len = t_0
        current_interval_end = t_0
        while current_interval_end <= state.timestamp.get(current_interval_end.unit):
            if current_interval_len.value == 0:
                raise ValueError(
                    'Interval between restarts for cosine annealing/warm restarts scheduler has decayed to 0.')

            current_interval_len = Time(value=int(self.t_mult * current_interval_len.value),
                                        unit=current_interval_len.unit)
            current_interval_end += current_interval_len

        current_interval_start = current_interval_end - current_interval_len
        frac_of_current_interval = ((state.timestamp.get(t_0.unit) - current_interval_start) /
                                    current_interval_len).value

        return _cosine_anneal(x=frac_of_current_interval, min_y=self.alpha_f)


class PolynomialScheduler(ComposerScheduler):
    r"""Sets the learning rate to be proportional to a power of the fraction of training time left.

    Specifically, the learning rate multiplier :math:`\alpha` can be expressed as:

    .. math::
        \alpha(t) = \alpha_f + (1 - \alpha_f) \times (1 - \tau) ^ {\kappa}

    Given :math:`\tau`, the fraction of time elapsed (clipped to the interval :math:`[0, 1]`), as:

    .. math::
        \tau = t / t_{max}

    Where :math:`\kappa`
    represents the exponent to be used for the proportionality relationship, :math:`t_{max}` represents the duration of
    this scheduler, and :math:`\alpha_f` represents the learning rate multiplier to decay to.

    Args:
        power (float): The exponent to be used for the proportionality relationship.
        t_max (str | Time): The duration of this scheduler. Default = ``"1dur"``.
        alpha_f (float): Learning rate multiplier to decay to. Default = ``0.0``.
    """

    def __init__(self, power: float, t_max: Union[str, Time] = '1dur', alpha_f: float = 0.0):
        self.t_max = t_max
        self.power = power
        self.alpha_f = alpha_f

    def __call__(self, state: State, ssr: float = 1.0):
        t_max = _convert_time(self.t_max, state, ssr=ssr)
        current_time = state.timestamp.get(t_max.unit)
        frac_of_total = (current_time / t_max).value

        coeff = (1 - frac_of_total)**self.power
        current_factor = self.alpha_f + coeff * (1.0 - self.alpha_f)
        return current_factor


def _raise_if_warmup_and_max_duration_incompatible(t_warmup: Union[str, Time], t_max: Union[str, Time]):
    if isinstance(t_warmup, str):
        t_warmup = Time.from_timestring(t_warmup)
    if isinstance(t_max, str):
        t_max = Time.from_timestring(t_max)
    units_same = t_warmup.unit == t_max.unit
    warmup_is_dur = t_warmup.unit == TimeUnit('dur')
    batches_vs_epochs = (t_warmup.unit == TimeUnit('ba') and
                         t_max.unit == TimeUnit('ep')) or (t_warmup.unit == TimeUnit('ep') and
                                                           t_max.unit == TimeUnit('ba'))
    if not units_same and not warmup_is_dur and not batches_vs_epochs:
        raise ValueError(f'Cannot use warmup scheduler with max_duration {t_max} and warmup {t_warmup}. '
                         't_warmup units must be the same as max_duration units, warmup must be in units "dur", '
                         'max_duration must be "ba" and t_warmup "ep", or max_duration must be "ep" and t_warmup "ba".')


class MultiStepWithWarmupScheduler(ComposerScheduler):
    r"""Decays the learning rate discretely at fixed milestones, with an initial warmup.

    .. seealso::
        This scheduler is based on :class:`~.MultiStepScheduler`, with an added warmup.

    Starts with a linear warmup over ``t_warmup`` time, then decays the learning rate by a factor of ``gamma``
    whenever a time milestone in ``milestones`` is reached.

    Specifically, the learning rate multiplier :math:`\alpha` can be expressed as:

    .. math::
        \alpha(t) = \begin{cases}
            t / t_{warmup}, & \text{if } t < t_{warmup} \\
            \gamma ^ x & \text{otherwise}
        \end{cases}

    Where :math:`t_{warmup}` represents the warmup time, :math:`x` represents the amount of milestones that have been
    reached, and :math:`\gamma` represents the multiplicative decay factor.

    .. warning::
        All milestones should be greater than ``t_warmup``; otherwise, they will have no effect on the computed learning
        rate multiplier until the warmup has completed.

    .. warning::
            By default, initial warmup time is **not** scaled according to any provided scale schedule ratio.
            To change this behavior, set ``scale_warmup=True``.

    Args:
        t_warmup (str | Time): Warmup time.
        milestones (List[str | Time]): Times at which the learning rate should change.
        gamma (float): Multiplicative decay factor. Default = ``0.1``.
        scale_warmup (float): SSR also scales the warmup period. Default = ``False``.
    """

    def __init__(self,
                 t_warmup: Union[str, Time],
                 milestones: List[Union[str, Time]],
                 gamma: float = 0.1,
                 scale_warmup: bool = False):
        self.t_warmup = t_warmup
        self.milestones = milestones
        self.gamma = gamma
        self.scale_warmup = scale_warmup
        self.warmup_scheduler = LinearScheduler(alpha_i=0.0, alpha_f=1.0, t_max=t_warmup)
        self.step_scheduler = MultiStepScheduler(milestones=milestones, gamma=gamma)

    def __call__(self, state: State, ssr: float = 1.0):
        assert state.max_duration is not None, 'max_duration should be set whenever schedulers are invoked'
        _raise_if_warmup_and_max_duration_incompatible(self.t_warmup, state.max_duration)
        t_warmup = _convert_time(self.t_warmup, state)
        if t_warmup.value == 0:
            warnings.warn(
                textwrap.dedent("""\
                The warmup duration is 0. If you specified warmup as a fraction of total
                training duration, take note that the warmup duration is calculated in the
                same unit as the trainer's max_duration parameter."""))

        if state.timestamp < t_warmup:
            if self.scale_warmup:
                return self.warmup_scheduler(state, ssr)
            return self.warmup_scheduler(state)

        return self.step_scheduler(state, ssr)


class ConstantWithWarmupScheduler(ComposerScheduler):
    r"""Maintains a fixed learning rate, with an initial warmup.

    This scheduler is based on  :class:`~torch.optim.lr_scheduler.ConstantLR` from PyTorch, with an added warmup.

    Starts with a linear warmup over ``t_warmup`` time, then simply maintains a learning rate factor of 1 for the entire training
    duration. However, both the factor and the duration of this scheduler can be configured.

    Specifically, the learning rate multiplier :math:`\alpha` can be expressed as:

    .. math::
        \alpha(t) = \begin{cases}
        t / t_{warmup}, & \text{if } t < t_{warmup} \\
        \alpha, & \text{if } t < t_{max} \\
        1.0 & \text{otherwise} \end{cases}

    Where :math:`\alpha` represents the learning rate multiplier to maintain while this scheduler is active, and
    :math:`t_{max}` represents the duration of this scheduler.

    .. warning::
            By default, initial warmup time is **not** scaled according to any provided scale schedule ratio.
            To change this behavior, set ``scale_warmup=True``.

    Args:
        t_warmup (str | Time): Warmup time.
        alpha (float): Learning rate multiplier to maintain while this scheduler is active. Default = ``1.0``.
        t_max (str | Time): Duration of this scheduler. Default = ``"1dur"``.
        scale_warmup (float): SSR also scales the warmup period. Default = ``False``.
    """

    def __init__(self,
                 t_warmup: Union[str, Time],
                 alpha: float = 1.0,
                 t_max: Union[str, Time] = '1dur',
                 scale_warmup: bool = False) -> None:
        self.t_warmup = t_warmup
        self.alpha = alpha
        self.t_max = t_max
        self.scale_warmup = scale_warmup
        self.scheduler = LinearWithWarmupScheduler(t_warmup=t_warmup,
                                                   alpha_i=alpha,
                                                   alpha_f=alpha,
                                                   t_max=t_max,
                                                   scale_warmup=scale_warmup)

    def __call__(self, state: State, ssr: float = 1.0) -> float:
        return self.scheduler(state, ssr)


class LinearWithWarmupScheduler(ComposerScheduler):
    r"""Adjusts the learning rate linearly, with an initial warmup.

    .. seealso::
        This scheduler is based on :class:`~.LinearScheduler`, with an added warmup.

    Linearly adjusts the learning rate multiplier from ``alpha_i`` to ``alpha_f`` over ``t_{max}`` time.

    Specifically, the learning rate multiplier :math:`\alpha` can be expressed as:

    .. math::
        \alpha(t) = \begin{cases}
            t / t_{warmup}, & \text{if } t < t_{warmup} \\
            \alpha_i + (alpha_f - \alpha_i) \times \tau_w & \text{otherwise}
        \end{cases}

    Given :math:`\tau_w`, the fraction of post-warmup time elapsed (clipped to the interval :math:`[0, 1]`), as:

    .. math::
        \tau_w = (t - t_{warmup}) / t_{max}

    Where :math:`t_{warmup}` represents the warmup time, :math:`\alpha_i` represents the initial learning rate multiplier,
    and :math:`\alpha_f` represents the learning rate multiplier to decay to, and :math:`t_{max}` represents the duration
    of this scheduler.


    .. warning::
        By default, the initial warmup time is **not** scaled according to any provided scale schedule ratio! However, the duration of
        the scheduler is still scaled accordingly. To achieve this, after warmup, the scheduler's "slope" will be
        slightly distorted from what would otherwise be expected. To scale the entire schedule, set ``scale_warmup=True``.

    Args:
        t_warmup (str | Time): Warmup time.
        alpha_i (float): Initial learning rate multiplier. Default = ``1.0``.
        alpha_f (float): Final learning rate multiplier. Default = ``0.0``.
        t_max (str | Time): The duration of this scheduler. Default = ``"1dur"``.
        scale_warmup (float): SSR also scales the warmup period. Default = ``False``.
    """

    def __init__(self,
                 t_warmup: Union[str, Time],
                 alpha_i: float = 1.0,
                 alpha_f: float = 0.0,
                 t_max: Union[str, Time] = '1dur',
                 scale_warmup: bool = False):
        self.t_warmup = t_warmup
        self.alpha_i = alpha_i
        self.alpha_f = alpha_f
        self.t_max = t_max
        self.scale_warmup = scale_warmup
        self.warmup_scheduler = LinearScheduler(alpha_i=0.0, alpha_f=alpha_i, t_max=t_warmup)

    def __call__(self, state: State, ssr: float = 1.0):
        assert state.max_duration is not None, 'max_duration should be set whenever schedulers are invoked'
        _raise_if_warmup_and_max_duration_incompatible(self.t_warmup, state.max_duration)
        t_warmup = _convert_time(self.t_warmup, state)
        if t_warmup.value == 0:
            warnings.warn(
                textwrap.dedent("""\
                The warmup duration is 0. If you specified warmup as a fraction of total
                training duration, take note that the warmup duration is calculated in the
                same unit as the trainer's max_duration parameter."""))

        if state.timestamp < t_warmup:
            if self.scale_warmup:
                return self.warmup_scheduler(state, ssr)
            return self.warmup_scheduler(state)

        t_max = _convert_time(self.t_max, state, ssr=ssr)
        current_time = state.timestamp.get(t_warmup.unit)
        frac_of_total = ((current_time - t_warmup) / (t_max - t_warmup)).value if (t_max > t_warmup) else 0.0
        frac_of_total = min(1.0, frac_of_total)

        current_factor = self.alpha_i + frac_of_total * (self.alpha_f - self.alpha_i)

        return current_factor


class CosineAnnealingWithWarmupScheduler(ComposerScheduler):
    r"""Decays the learning rate according to the decreasing part of a cosine curve, with an initial warmup.

    .. seealso::
        This scheduler is based on :class:`~.CosineAnnealingScheduler`, with an added warmup.

    Specifically, the learning rate multiplier :math:`\alpha` can be expressed as:

    .. math::
        \alpha(t) = \begin{cases}
            t / t_{warmup}, & \text{if } t < t_{warmup} \\
            \alpha_f + (1 - \alpha_f) \times \frac{1}{2} (1 + \cos(\pi \times \tau_w)) & \text{otherwise}
        \end{cases}

    Given :math:`\tau_w`, the fraction of post-warmup time elapsed (clipped to the interval :math:`[0, 1]`), as:

    .. math::
       \tau_w = (t - t_{warmup}) / t_{max}

    Where :math:`t_{warmup}` represents the warmup time, :math:`t_{max}` represents the duration of this scheduler, and
    :math:`\alpha_f` represents the learning rate multiplier to decay to.

    .. warning::
            By default, initial warmup time is **not** scaled according to any provided scale schedule ratio.
            To change this behavior, set ``scale_warmup=True``.

    Args:
        t_warmup (str | Time): Warmup time.
        t_max (str | Time): The duration of this scheduler. Default = ``"1dur"``.
        alpha_f (float): Learning rate multiplier to decay to. Default = ``0.0``.
        scale_warmup (float): SSR also scales the warmup period. Default = ``False``.
    """

    def __init__(self,
                 t_warmup: Union[str, Time],
                 t_max: Union[str, Time] = '1dur',
                 alpha_f: float = 0.0,
                 scale_warmup: bool = False):
        self.t_warmup = t_warmup
        self.t_max = t_max
        self.alpha_f = alpha_f
        self.scale_warmup = scale_warmup
        self.warmup_scheduler = LinearScheduler(alpha_i=0.0, alpha_f=1.0, t_max=t_warmup)

    def __call__(self, state: State, ssr: float = 1.0):
        assert state.max_duration is not None, 'max_duration should be set whenever schedulers are invoked'
        _raise_if_warmup_and_max_duration_incompatible(self.t_warmup, state.max_duration)
        t_warmup = _convert_time(self.t_warmup, state)
        if t_warmup.value == 0:
            warnings.warn(
                textwrap.dedent("""\
                The warmup duration is 0. If you specified warmup as a fraction of total
                training duration, take note that the warmup duration is calculated in the
                same unit as the trainer's max_duration parameter."""))

        if state.timestamp < t_warmup:
            if self.scale_warmup:
                return self.warmup_scheduler(state, ssr)
            return self.warmup_scheduler(state)

        t_max = _convert_time(self.t_max, state, ssr=ssr)
        current_time = state.timestamp.get(t_warmup.unit)
        frac_of_total = ((current_time - t_warmup) / (t_max - t_warmup)).value if (t_max > t_warmup) else 0.0
        frac_of_total = min(1.0, frac_of_total)

        return _cosine_anneal(x=frac_of_total, min_y=self.alpha_f)


class PolynomialWithWarmupScheduler(ComposerScheduler):
    r"""Decays the learning rate according to a power of the fraction of training time left, with an initial warmup.

    .. seealso::
        This scheduler is based on :class:`~.PolynomialScheduler`, with an added warmup.

    Specifically, the learning rate multiplier :math:`\alpha` can be expressed as:

    .. math::
        \alpha(t) = \begin{cases}
            t / t_{warmup}, & \text{if } t < t_{warmup} \\
            \alpha_f + (1 - \alpha_f) \times (1 - \tau_w) ^ {\kappa} & \text{otherwise}
        \end{cases}

    Given :math:`\tau_w`, the fraction of post-warmup time elapsed (clipped to the interval :math:`[0, 1]`), as:

    .. math::
       \tau_w = (t - t_{warmup}) / t_{max}

    Where :math:`\kappa` represents the exponent to be used for the proportionality relationship,
    :math:`t_{warmup}` represents the warmup time, :math:`t_{max}` represents the duration of this scheduler, and
    :math:`\alpha_f` represents the learning rate multiplier to decay to.

    .. warning::
            By default, initial warmup time is **not** scaled according to any provided scale schedule ratio.
            To change this behavior, set ``scale_warmup=True``.

    Args:
        t_warmup (str | Time): Warmup time.
        power (float): The exponent to be used for the proportionality relationship. Default = ``2.0``.
        t_max (str | Time): The duration of this scheduler. Default = ``"1dur"``.
        alpha_f (float): Learning rate multiplier to decay to. Default = ``0.0``.
        scale_warmup (float): SSR also scales the warmup period. Default = ``False``.
    """

    def __init__(self,
                 t_warmup: Union[str, Time],
                 power: float = 2.0,
                 t_max: Union[str, Time] = '1dur',
                 alpha_f: float = 0.0,
                 scale_warmup: bool = False):
        self.t_warmup = t_warmup
        self.power = power
        self.t_max = t_max
        self.alpha_f = alpha_f
        self.scale_warmup = scale_warmup
        self.warmup_scheduler = LinearScheduler(alpha_i=0.0, alpha_f=1.0, t_max=t_warmup)

    def __call__(self, state: State, ssr: float = 1.0):
        assert state.max_duration is not None, 'max_duration should be set whenever schedulers are invoked'
        _raise_if_warmup_and_max_duration_incompatible(self.t_warmup, state.max_duration)
        t_warmup = _convert_time(self.t_warmup, state)
        if t_warmup.value == 0:
            warnings.warn(
                textwrap.dedent("""\
                The warmup duration is 0. If you specified warmup as a fraction of total
                training duration, take note that the warmup duration is calculated in the
                same unit as the trainer's max_duration parameter."""))

        if state.timestamp < t_warmup:
            if self.scale_warmup:
                return self.warmup_scheduler(state, ssr)
            return self.warmup_scheduler(state)

        t_max = _convert_time(self.t_max, state, ssr=ssr)
        current_time = state.timestamp.get(t_warmup.unit)
        frac_of_total = ((current_time - t_warmup) / (t_max - t_warmup)).value if (t_max > t_warmup) else 0.0
        frac_of_total = min(1.0, frac_of_total)

        coeff = (1 - frac_of_total)**self.power
        current_factor = self.alpha_f + coeff * (1.0 - self.alpha_f)
        return current_factor

class LinearWithStartTimeScheduler(ComposerScheduler):
    r"""Adjusts the learning rate linearly from the specified start time to max time.

    .. warning::
        Note that the defaults for this scheduler differ from the defaults for
        :class:`~torch.optim.lr_scheduler.LinearLR`. The PyTorch scheduler, by default, linearly increases the learning
        rate multiplier from 1.0 / 3 to 1.0, whereas this implementation, by default, linearly decreases the multiplier
        rom 1.0 to 0.0.

    Linearly adjusts the learning rate multiplier from ``alpha_i`` to ``alpha_f`` over ``[t_{start}, t_{max}]`` period.
    Args:
        alpha_i (float): Initial learning rate multiplier. Default = ``1.0``.
        alpha_f (float): Final learning rate multiplier. Default = ``0.0``.
        t_start (str | Time): Start time for the scheduler. Default = ``"0dur"``.
        t_max (str | Time): End time of this scheduler. Default = ``"1dur"``.
    """

    def __init__(
        self,
        alpha_i: float = 1.0,
        alpha_f: float = 0.0,
        t_start: Union[str, Time] = "0dur",
        t_max: Union[str, Time] = "1dur"
    ):
        self.alpha_i = alpha_i
        self.alpha_f = alpha_f
        self.t_start = Time.from_timestring(t_start) if isinstance(t_start, str) else t_start
        self.t_max = Time.from_timestring(t_max) if isinstance(t_max, str) else t_max

    def __call__(self, state: State, ssr: float = 1.0):
        t_start = _convert_time(self.t_start, state, ssr=ssr)
        t_max = _convert_time(self.t_max, state, ssr=ssr)

        # Avoid error by converting duration unit to batch unit for `t_start` or
        # `t_max` if their units are not the same.
        # Usually this would be the case when `t_start` is batch unit and `t_max` is dur unit.
        if t_max.unit == TimeUnit("dur"):
            t_max = Time(int(t_max.value * state.max_duration.value), TimeUnit("ba"))
        if t_start.unit == TimeUnit("dur"):
            t_start = Time(int(t_start.value * state.max_duration.value), TimeUnit("ba"))

        assert t_start <= t_max, f"Start time is greater than max time, {t_start} and {t_max} were provided!"
        if t_start == t_max:
            warnings.warn("Start time is equal to max time, scheduling will not have affect.")

        current_time = state.timestamp.get(t_max.unit)
        frac_of_total = (current_time - t_start) / (t_max - t_start)
        frac_of_total = min(1.0, frac_of_total.value)
        current_factor = self.alpha_i + frac_of_total * (self.alpha_f - self.alpha_i)

        return current_factor

class InvSquareRootScheduler(ComposerScheduler):
    r"""Inverse square root step-based scheduler. Decays the learning rate proportional
    to inverse square root of a step number.

    Accepts an initial decay factor ``alpha_i`` that can be used to compensate for some
    skipped steps (during warmup, for example).

    If ``t_max`` is provided and is less than maximum traning duration, acts as a constant
    scheduler with a multiplier equal to the last decay step multiplier.

    Specifically, the learning rate multiplier :math:`\alpha` can be expressed as:

    .. math::
        \alpha(t) = \begin{cases}
            \alpha_i / \sqrt(step\_num), & \text{if } t < t_{max} \\
            \alpha_i / \sqrt(t_{max}) & \text{otherwise}
        \end{cases}

    Args:
        alpha_i (float): Initial decay factor. Default = ``1.0``.
        timescale (int | float): Timescale value to adjust the speed of multiplier descrease.
        t_max (str | Time): Max duration of the scheduler. Default = ``"1dur"``.
    """
    def __init__(
        self,
        alpha_i: float = 1.0,
        timescale: Union[int, float] = 10000,
        t_max: Union[str, Time] = "1dur"
    ) -> None:
        self.alpha_i = alpha_i
        self.timescale = timescale
        self.t_max = t_max

        self.shift = self.timescale

    def __call__(self, state: State, ssr: float = 1) -> float:
        t_max = _convert_time(self.t_max, state, ssr=ssr)
        current_time = state.timestamp.get(t_max.unit)

        if current_time < t_max:
            step = float(current_time.value)
            alpha = self.alpha_i / math.sqrt((step + self.shift) / self.timescale)  # timescaled
        else:
            alpha = self.alpha_i * float(t_max.value)**-0.5

        return alpha


class InvSquareRootWithWarmupAndCooldownScheduler(ComposerScheduler):
    r"""Inverse square root step-based scheduler with warmup and cooldown phases.

    Applies linear warmup first ``t_warmup`` steps to 1.0 multiplier, applies cooldown
    after the ``t_cooldown`` steps until the end to a ``alpha_f`` multiplier. Decays
    the learning rate proportional to inverse square root of a step number in between.

    Args:
        alpha_f (float): Final multiplier for cooldown stage.
        t_warmup (str | Time): Warmup time.
        t_cooldown (str | Time): Cooldown start time.
        timescale (int | float): Timescale value to adjust the speed of multiplier descrease.
        scale_warmup (float): SSR also scales the warmup period. Default = ``False``.
    """
    def __init__(
        self,
        alpha_f: float,
        t_warmup: Union[str, Time],
        t_cooldown: Union[str, Time],
        timescale: Union[int, float] = 10000,
        scale_warmup: bool = False
    ):
        self.alpha_f = alpha_f
        self.t_warmup = Time.from_timestring(t_warmup) if isinstance(t_warmup, str) else t_warmup
        self.t_cooldown = Time.from_timestring(t_cooldown) if isinstance(t_cooldown, str) else t_cooldown
        self.timescale = timescale
        self.scale_warmup = scale_warmup

        self.shift = float(self.timescale - self.t_warmup.value)
        self.cooldown_init_alpha = 1 / math.sqrt((float(self.t_cooldown.value) + self.shift) / self.timescale)
        self.warmup_scheduler = LinearScheduler(alpha_i=0.0, alpha_f=1.0, t_max=t_warmup)
        self.cooldown_scheduler = LinearWithStartTimeScheduler(
            alpha_i=self.cooldown_init_alpha,
            alpha_f=self.alpha_f,
            t_start=self.t_cooldown, 
            t_max="1dur"
        )

    def __call__(self, state: State, ssr: float = 1.0) -> float:
        assert state.max_duration is not None, 'max_duration should be set whenever schedulers are invoked'
        _raise_if_warmup_and_max_duration_incompatible(self.t_warmup, state.max_duration)

        t_warmup = _convert_time(self.t_warmup, state)
        if t_warmup.value == 0:
            warnings.warn(
                textwrap.dedent("""\
                The warmup duration is 0. If you specified warmup as a fraction of total
                training duration, take note that the warmup duration is calculated in the
                same unit as the trainer's max_duration parameter."""))

        t_cooldown = _convert_time(self.t_cooldown, state, ssr)
        if t_cooldown > state.max_duration:
            warnings.warn("Cooldown start value is higher than max duration, cooldown won't be applied!")

        if state.timestamp < t_warmup:
            if self.scale_warmup:
                return self.warmup_scheduler(state, ssr)
            return self.warmup_scheduler(state)
        elif state.timestamp < t_cooldown:
            current_time = state.timestamp.get(t_cooldown.unit)
            step = float(current_time.value)
            alpha = 1 / math.sqrt((step + self.shift) / self.timescale)  # timescaled

            return alpha
        else:
            return self.cooldown_scheduler(state, ssr)


class ConstantMultiStepWithGammaWithWarmupScheduler(ComposerScheduler):
    r"""Multi step scheduler with the specified gamma for each step. Gamma is
    applied independently at each step to the initial learning rate.

    Args:
        milestones (List[str | Time]): Times at which the learning rate decay should be applied.
        milestones_gamma (List[float]): Gamma value to be applied at each milestone step.
        t_warmup (str | Time): Warmup time.
        scale_warmup (float): SSR also scales the warmup period. Default = ``False``.
    """
    def __init__(
        self,
        milestones: List[Union[str, Time]],
        milestones_gamma: List[float],
        t_warmup: Union[str, Time],
        scale_warmup: bool = False
    ):
        assert len(milestones) == len(milestones_gamma), f"Length of milestones and milestones gamma must be the same, {len(milestones)} and {len(milestones_gamma)} were provided!"

        self.milestones = milestones
        self.milestones_gamma = milestones_gamma
        self.t_warmup = t_warmup
        self.scale_warmup = scale_warmup

        self.warmup_scheduler = LinearScheduler(alpha_i=0.0, alpha_f=1.0, t_max=t_warmup)
        self.milestones_schedulers = [
            ConstantScheduler(alpha=gamma)
            for gamma in self.milestones_gamma
        ]
        self.milestones_schedulers.insert(0, ConstantScheduler(alpha=1.0))

    def __call__(self, state: State, ssr: float = 1.0) -> float:
        assert state.max_duration is not None, 'max_duration should be set whenever schedulers are invoked'
        _raise_if_warmup_and_max_duration_incompatible(self.t_warmup, state.max_duration)

        t_warmup = _convert_time(self.t_warmup, state)
        if t_warmup.value == 0:
            warnings.warn(
                textwrap.dedent("""\
                The warmup duration is 0. If you specified warmup as a fraction of total
                training duration, take note that the warmup duration is calculated in the
                same unit as the trainer's max_duration parameter."""))

        if state.timestamp < t_warmup:
            if self.scale_warmup:
                return self.warmup_scheduler(state, ssr)
            return self.warmup_scheduler(state)

        milestones = [_convert_time(milestone, state, ssr=ssr) for milestone in self.milestones]
        current_time = state.timestamp.get(milestones[0].unit)
        scheduler_idx = 0
        for milestone in milestones:
            if current_time < milestone:
                break
            else:
                scheduler_idx += 1

        return self.milestones_schedulers[scheduler_idx](state, ssr)
