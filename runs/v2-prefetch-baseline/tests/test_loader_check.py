from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


HELPER_PATH = Path(__file__).resolve().parents[1] / "loader_check.py"
SPEC = importlib.util.spec_from_file_location("task_v2_loader_check", HELPER_PATH)
assert SPEC is not None and SPEC.loader is not None
loader_check = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = loader_check
SPEC.loader.exec_module(loader_check)


def test_off_on_content_identity_ignores_worker_process_pid() -> None:
    original = {
        "worker_id": 1,
        "worker_pid": 123,
        "sample_ids": [10, 18],
        "input_row_hashes": ["input-a", "input-b"],
        "labels_sha256": "labels",
        "positions_sha256": "positions",
    }
    restarted_worker = {**original, "worker_pid": 456}

    assert loader_check.batch_content(original) == loader_check.batch_content(restarted_worker)
    assert loader_check.sequence_summary([original]) == loader_check.sequence_summary(
        [restarted_worker]
    )
    wrong_assignment = {**restarted_worker, "worker_id": 0}
    assert loader_check.sequence_summary([original]) != loader_check.sequence_summary(
        [wrong_assignment]
    )


def test_queue_ahead_cursor_is_compared_to_the_unwrapped_delivered_cursor() -> None:
    delivered = {"dp_rank_0": {"epoch": 0, "sample_in_epoch": 24}}
    producer_cursor = {"epoch": 0, "sample_in_epoch": 48}

    loader_check.assert_prefetch_cursor_ahead(delivered, producer_cursor)
    with pytest.raises(AssertionError, match="mistaken for the delivered cursor"):
        loader_check.assert_prefetch_cursor_ahead(delivered, delivered["dp_rank_0"])


def test_row_boundary_counts_are_summed_over_all_owners() -> None:
    owner_counts = [
        {
            "eos_to_bos_pairs": rank + 1,
            "eos_to_eos_pairs": 2,
            "rows_starting_with_eos": rank % 2,
            "rows_starting_mid_document": 3,
        }
        for rank in range(8)
    ]

    assert loader_check.sum_owner_edge_counts(owner_counts) == {
        "eos_to_bos_pairs": 36,
        "eos_to_eos_pairs": 16,
        "rows_starting_with_eos": 4,
        "rows_starting_mid_document": 24,
    }


def test_first_epoch_ids_require_exact_global_coverage() -> None:
    assert loader_check.validate_first_epoch_ids([[0, 2], [1, 3]], 4) == [0, 1, 2, 3]
    with pytest.raises(AssertionError, match="row coverage differs"):
        loader_check.validate_first_epoch_ids([[0, 2], [1, 1]], 4)


def test_decoded_dcp_payload_matches_all_saved_scalar_keys_and_values() -> None:
    saved_cursor = {
        "dp_rank_0": {"epoch": 1, "sample_in_epoch": 24, "shuffle_seed": 2048}
    }
    decoded = {"dataloader": saved_cursor}

    validated = loader_check.validate_decoded_dcp_state(decoded, saved_cursor)
    assert validated == decoded
    assert loader_check.scalar_leaf_paths(validated) == [
        "dataloader.dp_rank_0.epoch",
        "dataloader.dp_rank_0.sample_in_epoch",
        "dataloader.dp_rank_0.shuffle_seed",
    ]
    with pytest.raises(AssertionError, match="values/keys differ"):
        loader_check.validate_decoded_dcp_state(
            {"dataloader": {"dp_rank_0": {"epoch": 1}}}, saved_cursor
        )


def test_skew_proof_requires_peer_epoch_one_next_to_start_inside_rank0_pause() -> None:
    start, end = 1_000, 2_000
    owners = []
    for rank in range(8):
        epoch1_started = 2_100 if rank == 0 else start + 10 * rank
        details = {
            "rank0_pause_started_monotonic_ns": start,
            "rank0_pause_ended_monotonic_ns": end,
            "pause_active_during_skew": True,
            "pause_released": True,
            "committed_cursor_matches_off55": True,
            "last_ready_cursor_matches_off56": True,
            "ready_items_held": 1,
            "rollover_advance": {
                "new_epoch_first_batch": {
                    "batch_index": 56,
                    "started_monotonic_ns": epoch1_started,
                    # Peer next() may complete after rank 0 releases its pause.
                    "ended_monotonic_ns": 2_500 if rank else 2_200,
                }
            },
        }
        owners.append({"rank": rank, "rank0_boundary_pause": {"details": details}})

    proof = loader_check.validate_boundary_skew(owners)
    assert proof["status"] == "passed"
    assert len(proof["peer_epoch1_next_calls_started_during_rank0_pause"]) == 7
    assert all(
        item["epoch1_next_ended_monotonic_ns"] > end
        for item in proof["peer_epoch1_next_calls_started_during_rank0_pause"]
    )

    owners[3]["rank0_boundary_pause"]["details"]["rollover_advance"][
        "new_epoch_first_batch"
    ]["started_monotonic_ns"] = end + 1
    with pytest.raises(AssertionError, match="did not start during rank-0 pause"):
        loader_check.validate_boundary_skew(owners)
