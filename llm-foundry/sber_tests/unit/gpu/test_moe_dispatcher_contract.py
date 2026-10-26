from types import SimpleNamespace

import pytest
import torch

from llmfoundry.models.layers.moe import dispatchers

pytestmark = pytest.mark.unit


def _config(*, drop_by_group_capacity: bool = True) -> SimpleNamespace:
    """Return the minimum production dispatcher configuration."""
    return SimpleNamespace(
        num_routed_experts=4,
        dispatcher_balancing_strategy="fixed_capacity",
        expert_capacity_factor=None,
        expert_capacity_tokens=2,
        drop_by_group_capacity=drop_by_group_capacity,
        num_experts_per_tok=1,
        moe_enable_deepep=False,
        moe_deepep_num_sms=None,
    )


def _dispatcher(
    monkeypatch: pytest.MonkeyPatch,
    *,
    ep_size: int,
    drop_by_group_capacity: bool = True,
) -> dispatchers.TokenDispatcher:
    """Build a real CUDA TokenDispatcher with deterministic group sizing."""
    monkeypatch.setattr(dispatchers, "get_ep_group_size", lambda: ep_size)
    return dispatchers.TokenDispatcher(
        _config(drop_by_group_capacity=drop_by_group_capacity),
    )


@pytest.mark.unit
@pytest.mark.gpu
def test_monitor_updates_preserve_allocated_cuda_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated monitoring writes must not replace CUDA buffers."""
    dispatcher = _dispatcher(monkeypatch, ep_size=1)
    dispatcher._init_dispatcher_monitor()
    original_pointers = [
        tensor.data_ptr() for tensor in dispatcher.get_dispatcher_monitor()
    ]

    dispatcher._update_dispatcher_monitor(
        3,
        torch.tensor([1, 2, 3, 4], device="cuda"),
        torch.tensor([1, 1, 2, 3], device="cuda"),
        torch.tensor([0, 1, 2, 2], device="cuda"),
    )
    dispatcher._update_dispatcher_monitor(
        5,
        torch.tensor([4, 3, 2, 1], device="cuda"),
        torch.tensor([3, 2, 1, 0], device="cuda"),
        torch.tensor([2, 1, 0, 0], device="cuda"),
    )

    buffers = dispatcher.get_dispatcher_monitor()
    assert [tensor.data_ptr() for tensor in buffers] == original_pointers
    assert buffers[0].tolist() == [5, 5, 5, 5]
    assert buffers[1].tolist() == [4, 3, 2, 1]
    assert buffers[2].tolist() == [3, 2, 1, 0]
    assert buffers[3].tolist() == [2, 1, 0, 0]


@pytest.mark.unit
@pytest.mark.gpu
def test_ep_one_never_resolves_or_reduces_an_ep_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """EP=1 must make its group-capacity decision using local counts only."""
    dispatcher = _dispatcher(monkeypatch, ep_size=1)

    def fail_group_lookup() -> None:
        """Fail if the EP=1 fast path tries to resolve a process group."""
        raise AssertionError("EP=1 must not resolve an EP group")

    def fail_reduce(*args, **kwargs) -> None:
        """Fail if the EP=1 fast path performs a collective."""
        raise AssertionError("EP=1 must not all-reduce")

    monkeypatch.setattr(dispatchers, "get_ep_group", fail_group_lookup)
    monkeypatch.setattr(dispatchers, "all_reduce", fail_reduce)
    routing_map = torch.tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0]],
        dtype=torch.bool,
        device="cuda",
    )
    probabilities = torch.tensor(
        [[0.9, 0.0, 0.0, 0.0], [0.0, 0.8, 0.0, 0.0]],
        device="cuda",
    )

    actual_map, actual_probabilities = dispatcher._drop_token_by_capacity_factor(
        probabilities,
        routing_map,
    )

    assert torch.equal(actual_map, routing_map)
    assert torch.equal(actual_probabilities, probabilities)


@pytest.mark.gpu
@pytest.mark.parametrize("prefix_argument", ["prefix_mask", "observed_prefix_mask"])
@pytest.mark.parametrize("drop_by_group_capacity", [False, True])
def test_assignment_observer_preserves_local_counts_and_masks_across_microbatches(
    monkeypatch: pytest.MonkeyPatch, drop_by_group_capacity: bool, prefix_argument: str,
) -> None:
    dispatcher = _dispatcher(
        monkeypatch, ep_size=1, drop_by_group_capacity=drop_by_group_capacity,
    )
    recorded = []

    def observe(before, valid_before, after, valid_after):
        recorded.append(tuple(value.cpu().tolist() for value in (before, valid_before, after, valid_after)))

    dispatcher.assignment_observer = observe
    assignments = torch.tensor([0] * 9 + [1], device="cuda")
    routing_map = torch.nn.functional.one_hot(assignments, num_classes=4).bool()
    probabilities = routing_map * torch.arange(10, 0, -1, device="cuda").unsqueeze(1)
    padding_mask = torch.tensor([[1, 0, 1, 1, 1, 1, 1, 1, 1, 1]], device="cuda")
    prefix_mask = torch.tensor([[1, 1, 0, 1, 1, 1, 1, 1, 1, 1]], device="cuda")

    actual_map, actual_probs = dispatcher._drop_token_by_capacity_factor(
        probabilities, routing_map, padding_mask, **{prefix_argument: prefix_mask},
    )
    expected_map = routing_map.clone()
    expected_map[2:9] = False
    torch.testing.assert_close(actual_map, expected_map)
    torch.testing.assert_close(actual_probs, probabilities * expected_map)

    second_map = torch.tensor([[0, 0, 1, 0], [0, 0, 0, 1]], device="cuda", dtype=torch.bool)
    second_probs = second_map.float()
    actual_map, actual_probs = dispatcher._drop_token_by_capacity_factor(second_probs, second_map)
    torch.testing.assert_close(actual_map, second_map)
    torch.testing.assert_close(actual_probs, second_probs)
    assert recorded == [
        ([9, 1, 0, 0], [7, 1, 0, 0], [2, 1, 0, 0], [1, 1, 0, 0]),
        ([0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]),
    ]


@pytest.mark.gpu
def test_dispatcher_monitor_counts_prefixes_that_only_the_assignment_observer_drops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The GMM and ScMoE blocks give the prefix mask to the assignment observer only: the dispatcher
    monitor's counts without paddings and prefixes stay the counts without paddings, as before."""
    dispatcher = _dispatcher(monkeypatch, ep_size=1)
    dispatcher._init_dispatcher_monitor()
    dispatcher._active_dispatcher_monitor = True
    recorded = []

    def observe(before, valid_before, after, valid_after):
        recorded.append(tuple(value.cpu().tolist() for value in (before, valid_before, after, valid_after)))

    dispatcher.assignment_observer = observe
    routing_map = torch.nn.functional.one_hot(torch.tensor([0, 0, 1, 1], device="cuda"), num_classes=4).bool()
    probabilities = routing_map.float()
    padding_mask = torch.tensor([[1, 1, 1, 0]], device="cuda")
    prefix_mask = torch.tensor([[0, 1, 1, 1]], device="cuda")

    dispatcher._drop_token_by_capacity_factor(
        probabilities, routing_map, padding_mask, observed_prefix_mask=prefix_mask,
    )

    buffers = dispatcher.get_dispatcher_monitor()
    assert buffers[1].tolist() == [2, 2, 0, 0]
    assert buffers[2].tolist() == [2, 1, 0, 0]
    assert buffers[3].tolist() == [2, 1, 0, 0]
    assert recorded == [([2, 2, 0, 0], [1, 1, 0, 0], [2, 2, 0, 0], [1, 1, 0, 0])]


@pytest.mark.unit
@pytest.mark.gpu
@pytest.mark.parametrize(
    ("remote_counts", "expected_kept_rows"),
    [
        pytest.param([0, 0, 0, 0], [True, True, True, True], id="group-not-overloaded"),
        pytest.param([6, 0, 0, 0], [True, False, False, True], id="group-overloaded"),
    ],
)
def test_ep_two_uses_global_group_capacity_decision(
    monkeypatch: pytest.MonkeyPatch,
    remote_counts: list[int],
    expected_kept_rows: list[bool],
) -> None:
    """EP>1 must use one global SUM to decide whether local dropping is needed."""
    dispatcher = _dispatcher(monkeypatch, ep_size=2)
    ep_group = object()
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(dispatchers, "get_ep_group", lambda: ep_group)

    def record_reduce(
        *, tensor: torch.Tensor, reduce_operation: str, group: object
    ) -> None:
        """Record the collective and add deterministic counts from the remote rank."""
        calls.append((reduce_operation, group))
        tensor.add_(torch.tensor(remote_counts, device=tensor.device))

    monkeypatch.setattr(dispatchers, "all_reduce", record_reduce)
    routing_map = torch.tensor(
        [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]],
        dtype=torch.bool,
        device="cuda",
    )
    probabilities = torch.tensor(
        [[0.9, 0, 0, 0], [0.8, 0, 0, 0], [0.7, 0, 0, 0], [0, 0, 0.6, 0]],
        device="cuda",
    )

    actual_map, actual_probabilities = dispatcher._drop_token_by_capacity_factor(
        probabilities,
        routing_map,
    )

    expected_map = routing_map.clone()
    expected_map[:, 0] &= torch.tensor(expected_kept_rows, device="cuda")
    torch.testing.assert_close(actual_map, expected_map)
    torch.testing.assert_close(
        actual_probabilities,
        probabilities * expected_map,
    )
    assert calls == [("SUM", ep_group)]
