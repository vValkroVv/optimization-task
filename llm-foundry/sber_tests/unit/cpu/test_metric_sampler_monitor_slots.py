"""The slots of the rank-local monitors: their widths are those of the words and values the monitors' observers
record, and the files the sampler writes next to its windows name the monitors' catalogs and orders."""

import json
import struct
import types
import uuid

import pytest

pytestmark = pytest.mark.unit

pytest.importorskip("composer", reason="Requires the CPU training runtime")

import torch  # noqa: E402

from llmfoundry.callbacks.metric_sampler import MetricSampler  # noqa: E402
from llmfoundry.callbacks.metric_sampler_monitors import RankLocalMonitorCapture  # noqa: E402
from metric_sampler.capture.layout import Layout  # noqa: E402
from metric_sampler.capture.recorder import RankRecorder  # noqa: E402
from metric_sampler.capture.ring import ShmRing  # noqa: E402
from metric_sampler.contract.catalog import metric_by_name  # noqa: E402

_EXPERTS = 4
_PARAMETERS = ["layers.0.k_proj", "layers.0.q_proj"]
_LR_GROUPS = ["optimizer_0/AdamW/group_0", "optimizer_0/AdamW/group_1"]
_WIDTHS = {
    "monitor/optimizer_health_local": 132,
    "monitor/training_local_counts": 12,
    "monitor/memory_local": 33,
    "monitor/training_local": 12,
    "monitor/stage_time_local": 18,
    "monitor/lr_group_local": 2,
    "monitor/materialized_weight_local": 3,
    "monitor/optimizer_local": 18,
    "monitor/update_weight_local": 8,
    "monitor/weight_local": 2,
    "monitor/topk_local_counts": 4,
    "monitor/activation_local": 2,
    "monitor/activation_local_observations": 2,
    "monitor/router_margin_local": 8,
    "router/scores_mean": _EXPERTS,
    "monitor/router_balance_local": 14,
    "monitor/router_balance_counts_local": 3,
    "monitor/router_bias_local": _EXPERTS,
    "monitor/expert_activation_sumsq_local": 2 * _EXPERTS,
    "monitor/expert_activation_count_local": _EXPERTS,
}


def _monitored_sampler(tmp_path) -> MetricSampler:
    sampler = MetricSampler(str(tmp_path), rank_local_monitors=True)
    sampler.microbatches = 2
    monitors = sampler.monitor_capture
    monitors._lr_group_names = _LR_GROUPS
    monitors._lr_group_indices = {name: index for index, name in enumerate(_LR_GROUPS)}
    monitors._optimizer_parameter_names = _PARAMETERS
    monitors._update_parameter_names = _PARAMETERS
    monitors._materialized_parameter_names = _PARAMETERS
    monitors._parameter_indices = {name: index for index, name in enumerate(_PARAMETERS)}
    monitors._activation_catalog = ["m:input.0", "m:output.0"]
    monitors._topk_monitor = types.SimpleNamespace(_enabled=True)
    monitors._expert_modules = [(0, object())]
    monitors.materialized_weights = types.SimpleNamespace(flush=lambda: None)
    return sampler


def _requests(sampler: MetricSampler) -> list[tuple[str, int, int, int]]:
    return sampler.monitor_capture.slot_requests(_EXPERTS, {0: 0})


def test_the_monitors_slots_have_the_widths_of_the_contract(tmp_path):
    widths = {}
    for name, _, _, width in _requests(_monitored_sampler(tmp_path)):
        widths.setdefault(name, set()).add(width)

    assert widths == {name: {width} for name, width in _WIDTHS.items()}


@pytest.fixture
def recording(tmp_path):
    sampler = _monitored_sampler(tmp_path)
    layout = Layout([(metric_by_name(name), layer, microbatch, width) for name, layer, microbatch, width in
                     _requests(sampler)])
    ring = ShmRing.create(f"ms_test_{uuid.uuid4().hex}", layout, slot_count=1)
    sampler.recorder = RankRecorder(layout, ring, torch.device("cpu"))
    sampler.recorder.begin_step(False)
    yield sampler, ring
    sampler.recorder.close()
    ring.unlink()


def _words(value, fmt: str) -> list[int]:
    return list(struct.unpack("<ii", struct.pack(fmt, value)))


def _recorded(sampler: MetricSampler, ring: ShmRing, name: str, layer: int, microbatch: int) -> list[int]:
    slot = sampler.recorder.resolve(name, layer, microbatch)
    return ring.payload(0)[slot.offset : slot.offset + slot.width].tolist()


def test_the_observers_record_as_many_words_as_their_slots_hold(recording):
    sampler, ring = recording
    monitors = sampler.monitor_capture
    monitors._observe_lr_group(0, "AdamW", 1, 0.5)
    monitors._observe_memory_local({"allocated_bytes.all.current": 5, "num_alloc_retries": 2**63}, 7, 11)
    monitors._observe_training_local(8, 4096, 1.25)
    monitors._observe_local_flops(3.5e12)
    monitors._flush_monitor_additions()
    monitors._observe_stage_time({"forward": 0.5, "dataloader_fetch": 2.0})
    monitors._observe_training_counts((torch.tensor(3), 5, None, 2**40))
    monitors._observe_optimizer_health(torch.arange(66, dtype=torch.float64))
    update_values = torch.arange(8, dtype=torch.float64).reshape(2, 4)
    monitors._observe_update_weight(_PARAMETERS, update_values, torch.arange(4, dtype=torch.float32).reshape(2, 2))
    monitors._observe_topk(torch.arange(4, dtype=torch.int32))
    sampler.recorder.step_end(0)

    assert _recorded(sampler, ring, "monitor/lr_group_local", 1, -1) == _words(0.5, "<d")
    assert _recorded(sampler, ring, "monitor/optimizer_health_local", -1, -1) == list(
        struct.unpack("<132i", struct.pack("<66d", *range(66)))
    )
    assert _recorded(sampler, ring, "monitor/topk_local_counts", -1, 0) == [0, 1, 2, 3]
    assert _recorded(sampler, ring, "monitor/update_weight_local", 1, -1) == list(
        struct.unpack("<8i", struct.pack("<4d", 4.0, 5.0, 6.0, 7.0))
    )


def test_a_value_the_monitor_did_not_report_is_recorded_as_absent_words(recording):
    sampler, ring = recording
    monitors = sampler.monitor_capture
    monitors._observe_memory_local({"allocated_bytes.all.current": 5, "num_alloc_retries": 2**63}, 7, 11)
    monitors._observe_stage_time({"forward": 0.5})
    monitors._observe_training_counts((torch.tensor(3), 5, None, 2**40))
    monitors._observe_training_local(8, 4096, 1.25)
    monitors._flush_monitor_additions()
    sampler.recorder.step_end(0)

    memory = [0] * 22 + [0] * 11
    for position, value in ((0, 5), (8, 2**63), (9, 7), (10, 11)):
        memory[2 * position : 2 * position + 2] = list(struct.unpack("<ii", struct.pack("<Q", value)))
        memory[22 + position] = 1
    assert _recorded(sampler, ring, "monitor/memory_local", -1, -1) == memory
    assert _recorded(sampler, ring, "monitor/stage_time_local", -1, -1) == (
        [0, 0] + _words(0.5, "<d") + [0] * 8 + [0, 1, 0, 0, 0, 0]
    )
    assert _recorded(sampler, ring, "monitor/training_local_counts", -1, -1) == [3, 0, 5, 0, 0, 0, 0, 256, 1, 1, 0, 1]
    assert _recorded(sampler, ring, "monitor/training_local", -1, -1) == (
        _words(8, "<Q") + _words(4096, "<Q") + _words(1.25, "<d") + [0, 0] + [1, 1, 1, 0]
    )


def test_the_catalog_the_ownership_and_the_contract_are_written_next_to_the_windows(tmp_path):
    monitors = _monitored_sampler(tmp_path).monitor_capture
    monitors._catalog = [{"rank": -1, "kind": "parameter", "index": 0, "name": "layers.0.k_proj"}]
    monitors._routed_gates = [(0, types.SimpleNamespace(n_routed_experts=_EXPERTS))]
    ownership = [
        {"rank": 0, "layer": 0, "global_expert_ids": [0, 1]}, {"rank": 1, "layer": 0, "global_expert_ids": [2, 3]},
    ]

    monitors._write_monitor_files("job-0", [ownership[:1], ownership[1:]])

    directory = tmp_path / "attempt=job-0"
    assert sorted(path.name for path in directory.iterdir()) == [
        "monitor_capture_contract.json", "monitor_expert_ownership.json", "monitor_parameters.json",
    ]
    assert json.loads((directory / "monitor_parameters.json").read_text()) == monitors._catalog
    assert json.loads((directory / "monitor_expert_ownership.json").read_text()) == ownership
    contract = json.loads((directory / "monitor_capture_contract.json").read_text())
    assert list(contract) == [
        "update_weight_parameter_names", "optimizer_parameter_names", "activation_names", "optimizer_group_names",
        "optimizer_health_field_order", "optimizer_local_field_order", "routed_layers", "global_expert_ids",
        "expert_ownership_file", "expert_moments", "router_balance_order", "router_balance_counts_order",
        "router_balance_scope", "activation_statistics", "activation_presence", "weight_statistics",
        "optimizer_statistics", "training_counter_order", "learning_rate_group_names", "lr_source_instant",
        "memory_stat_order", "memory_source_instant", "training_local_order", "training_local_source",
        "stage_time_order", "stage_time_source", "materialized_parameter_names", "materialized_weight_source",
        "materialized_weight_presence", "float64_and_uint64_words",
    ]
    assert contract["optimizer_health_field_order"] == [
        "numel", "frac_eps_count", "frac_10eps_count", "update_abs_sum", "grad_contrib_sq_sum", "wd_contrib_sq_sum",
    ]
    assert contract["router_balance_counts_order"] == ["all_tokens", "valid_tokens", "forward_calls"]
    assert contract["training_counter_order"] == [
        "skip_updates", "total_updates", "zeroed_module_grads", "processed_module_grads",
    ]
    assert contract["training_local_order"] == ["rank_samples", "rank_tokens", "rank_batch_seconds", "device_flops"]
    assert contract["memory_stat_order"] == list(RankLocalMonitorCapture._memory_stat_names())
    assert contract["global_expert_ids"] == [0, 1, 2, 3] and contract["learning_rate_group_names"] == _LR_GROUPS
