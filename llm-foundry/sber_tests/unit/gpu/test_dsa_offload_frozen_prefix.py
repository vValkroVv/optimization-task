"""Composer CPU-offload state machine under a frozen trunk prefix (DSA stage-1 pattern).

Stage 1 freezes the trunk and trains only the indexers, so some layer boundaries have no
autograd graph: their sync is skipped, backward never visits their offload slot, and the
slot stays dirty into the next batch. Without the armed-slots sweep, batch 2's push_tensor
asserts `Invalid state: offload_finished` -- the exact crash observed on a 512-GPU stage-1
run. These are the distilled state-machine cases (a 1-batch test CANNOT catch this class:
the failure only fires at the batch-2 cycle boundary).

Also locks is_recomputing() (the DSA selection-replay-cache hook) as thread-local.
"""
import threading

import pytest

import torch
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)
from composer.trainer import cpu_offload


def make_ctx(num_layers=2, model_layers=3):
    ctx, sync = cpu_offload.get_cpu_offload_context(
        enabled=True, num_layers=num_layers, model_layers=model_layers)
    push, _pop = ctx.get_hooks()
    return ctx, sync, push


def offloadable(shape=(64, 64)):
    t = torch.randn(*shape, device="cuda")
    t.activation_offloading = True
    return t


def boundary_tensor():
    x = torch.randn(8, device="cuda", requires_grad=True)
    return x * 2.0                                   # non-leaf with grad_fn


def run_cycle(ctx, sync, push, arm=(True, True, True), backward=True):
    """One forward 'cycle' over 3 slots; arm[i]=False models a frozen (graph-less) boundary."""
    boundaries = []
    for slot in range(3):
        with ctx:
            push(offloadable())
        if arm[slot]:
            y = boundary_tensor()
            sync(y)
            boundaries.append(y)
    if backward:
        for y in boundaries:
            y.sum().backward()
    return boundaries


def _idle_slots(sync):
    """Slots holding nothing: either never offloaded or released by the sweep."""
    return {i for i, st in sync.layer_states.items() if st.state == "not_offloaded"}


class TestFrozenPrefixWraparound:
    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_frozen_unarmed_slot_released_at_wraparound(self):
        """A slot pushed but never armed (frozen boundary) must be silently released at
        the next cycle -- the 512-GPU batch-2 crash regression."""
        ctx, sync, push = make_ctx()
        run_cycle(ctx, sync, push, arm=(False, True, True), backward=True)
        with ctx:                                     # batch 2: must NOT assert offload_finished
            push(offloadable())
        state = ctx.offload_synchronizer
        assert state.armed_slots == set(), "a slot is still waiting for a backward that ran"
        assert _idle_slots(state) >= set(state.layer_states) - {state.current_layer_id}, (
            "the sweep left a slot dirty, so batch 3 would hit the same assertion"
        )

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_inflight_armed_slots_fail_fast(self):
        """Armed slots whose backward never ran are a LIVE graph: wraparound must raise,
        never silently release them."""
        ctx, sync, push = make_ctx()
        run_cycle(ctx, sync, push, arm=(True, True, True), backward=False)
        with pytest.raises(AssertionError, match="await"):
            with ctx:
                pass

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_normal_cycles_sweep_inert(self):
        ctx, sync, push = make_ctx()
        for _ in range(3):
            run_cycle(ctx, sync, push, arm=(True, True, True), backward=True)
            assert ctx.offload_synchronizer.armed_slots == set(), (
                "every armed slot must be disarmed by its bwd_step"
            )

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_armed_clean_slot_fails_fast(self):
        """An armed but never-offloaded slot with backward pending still holds live hooks
        that would mutate the synchronizer mid-cycle -- must also fail fast."""
        ctx, sync, push = make_ctx()
        boundaries = run_cycle(ctx, sync, push, arm=(True, True, True), backward=False)
        boundaries[0].sum().backward()
        boundaries[1].sum().backward()               # slot 2 (clean, non-offloaded) stays armed
        with pytest.raises(AssertionError, match="await"):
            with ctx:
                pass

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_frozen_slot_prefetch_skipped(self):
        """With a prefetch offset, a trainable tick's start_reload_map may point at a
        frozen slot: its reload must be skipped (buffers would sit unconsumed)."""
        ctx, sync, push = make_ctx(num_layers=1, model_layers=3)   # start_reload_map[1] == [0]
        run_cycle(ctx, sync, push, arm=(False, True, True), backward=True)
        assert ctx.offload_synchronizer.layer_states[0].state == "offload_finished"
        with ctx:                                     # released as frozen leftover, no raise
            push(offloadable())

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_offload_started_slot_finished_before_release(self):
        """A slot swept while its D2H copy is in flight must be finished (event waited)
        before freeing, not have the copy's GPU source dropped mid-flight."""
        ctx, sync, push = make_ctx()
        slot = ctx.offload_synchronizer.layer_states[0]
        slot.fwd_gpu_tensor_group.tensor_list.append(offloadable())
        slot.fwd_gpu_tensor_group.events.append(torch.cuda.Event())
        slot.fwd_gpu_tensor_group.events[-1].record(torch.cuda.current_stream())
        slot.start_offload()
        assert slot.state == "offload_started"
        ctx.offload_synchronizer.num_of_fwds = ctx.offload_synchronizer.num_layers - 1
        with ctx:                                     # wraparound sweep: finish-then-free
            push(offloadable())
        assert slot.state == "not_offloaded"
        torch.cuda.synchronize()


class TestIsRecomputingThreadLocal:
    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_recompute_flag_is_per_thread(self):
        """A recompute on one autograd thread must not make another thread replay caches."""
        from composer.trainer import torch_utils_checkpoint as tuc
        seen = {}

        def worker():
            tuc._recompute_state.active = True        # simulate an in-progress recompute
            seen["worker"] = tuc.is_recomputing()

        assert tuc.is_recomputing() is False
        t = threading.Thread(target=worker)
        t.start()
        t.join()
        assert seen["worker"] is True
        assert tuc.is_recomputing() is False, "recompute flag leaked across threads"
