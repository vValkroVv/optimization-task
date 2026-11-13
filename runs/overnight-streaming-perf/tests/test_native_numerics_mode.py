from __future__ import annotations

import builtins
import hashlib
import json
import struct
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import measure
import parse_results
import run


MANIFEST = {
    "schema": "native-numerical-receipt-v1",
    "capture_updates": [1, 2, 50],
    "expected_microbatches": 32,
    "scope": "diagnostic only; no TPS",
}


def _tensor_observation(update, phase, name, *, rank=0, microbatch=None, dtype="torch.float32"):
    return {
        "update": update,
        "phase": phase,
        "name": name,
        "microbatch": microbatch,
        "present": True,
        "rank": rank,
        "dtype": dtype,
        "source_device_type": "cuda",
        "global_shape": [],
        "local_shape": [],
        "local_stride": [],
        "local_numel": 1,
        "mesh": None,
        "placements": [],
        "sha256": "a" * 64,
        "is_finite": True if dtype.startswith("torch.float") else None,
    }


def _counter_observation(rank, update, value, phase):
    digest = hashlib.sha256(struct.pack("=q", value)).hexdigest()
    item = _tensor_observation(update, phase, "trainer.ntokens_seen", rank=rank, dtype="torch.int64")
    item["sha256"] = digest
    item["is_finite"] = None
    return item


def _receipt(rank=0):
    observations = [
        {"update": 0, "phase": "after_load", "name": "trainer.step", "value": 0},
        _counter_observation(rank, 0, 0, "after_load"),
        _tensor_observation(0, "initial_parameter", "model.0.weight", rank=rank),
    ]
    for update, chunks, tokens in ((1, 2, 2_096_640), (2, 2, 4_193_280), (50, 32, 104_832_000)):
        for microbatch in range(32):
            observations.extend(
                [
                    _tensor_observation(update, "dataloader_input", "input.tokens", rank=rank, microbatch=microbatch, dtype="torch.int64"),
                    _tensor_observation(update, "dataloader_input", "labels", rank=rank, microbatch=microbatch, dtype="torch.int64"),
                    {
                        "update": update,
                        "phase": "microbatch_metadata",
                        "name": "trainer.forward_backward_step",
                        "microbatch": microbatch,
                        "active_num_chunks": chunks,
                        "global_valid_tokens": {
                            "shape": [], "dtype": "python.float", "device_type": None,
                            "python_type": "float", "value": 2_096_640.0,
                        },
                    },
                    _tensor_observation(update, "microbatch_output", "loss.local_normalized", rank=rank, microbatch=microbatch),
                ]
            )
        observations.extend(
            [
                {
                    "update": update, "phase": "update_summary", "name": "global_valid_tokens",
                    "microbatch": None, "present": True, "python_type": "float", "dtype": "python.float",
                    "shape": [], "global_shape": [], "local_shape": [], "local_stride": [],
                    "local_numel": 1, "source_device_type": None, "value": 2_096_640.0,
                    "sha256": None, "is_finite": True,
                },
                _tensor_observation(update, "grad_preclip", "model.0.weight.grad", rank=rank),
                _tensor_observation(update, "clip_result", "clip.total_norm", rank=rank),
                _tensor_observation(update, "grad_postclip", "model.0.weight.grad", rank=rank),
                {"update": update, "phase": "optimizer_group", "name": "optimizer.0.group.0", "optimizer_class": "torch.optim.AdamW", "hyperparameters": {"lr": 0.001}},
                _tensor_observation(update, "post_optimizer_parameter", "model.0.weight", rank=rank),
                {"update": update, "phase": "callback_state", "name": "callbacks.batch_end", "microbatch": None, "event": "batch_end", "step": update, "record_count": 0},
                {"update": update, "phase": "batch_end", "name": "trainer.step", "value": update},
                _counter_observation(rank, update, tokens, "batch_end"),
            ]
        )
    return {
        "schema": "native-numerical-receipt-v1",
        "complete": True,
        "rank": rank,
        "capture_updates": [1, 2, 50],
        "expected_microbatches": 32,
        "hash_semantics": "exact bytes",
        "clip_norm_semantics": "native total norm",
        "ema_state": "not captured",
        "callback_state_scope": "batch end only",
        "training_metadata": {"initial_step": 0, "mtp_loss_weights": [0.3], "initial_num_chunks": 2},
        "positions_semantics": "source positions",
        "microbatch_capture": "CPU inputs and deferred scalars",
        "scope": "Parameter, gradient, optimizer-state and update-summary copies to CPU synchronize and are diagnostic instrumentation, not TPS.",
        "coverage": {
            str(update): {
                "input_batches": 32, "microbatches": 32, "summary_calls": 1,
                "clip_calls": 1, "optimizer_calls": 1, "batch_end_calls": 1,
            }
            for update in (1, 2, 50)
        },
        "observations": observations,
    }


def _write_run(tmp_path: Path, *, ranks=range(8), mutate=None):
    measurement = tmp_path / "measurement"
    measurement.mkdir(exist_ok=True)
    metadata = {
        "run_id": "native-numerics-test",
        "steps": 50,
        "warmup_updates": 0,
        "window_updates": 0,
        "monitor_interval": 50,
        "native_numerics": MANIFEST,
        "config_hashes_before": {"native_numerics_sha256": "b" * 64},
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    (tmp_path / "config-hashes-final.json").write_text(json.dumps({"native_numerics_sha256": "b" * 64}))
    for rank in ranks:
        receipt = _receipt(rank)
        if mutate is not None and rank == 0:
            mutate(receipt)
        (measurement / f"native-numerics-rank{rank}.json").write_text(json.dumps(receipt))
    return metadata


def _reference_tensor_observation(
    rank: int,
    update: int,
    phase: str,
    name: str,
    *,
    microbatch=None,
    dtype="torch.float32",
) -> dict:
    return {
        "update": update,
        "phase": phase,
        "name": name,
        "microbatch": microbatch,
        "rank": rank,
        "present": True,
        "dtype": dtype,
        "source_device_type": "cuda",
        "global_shape": [1],
        "local_shape": [1],
        "local_stride": [1],
        "local_numel": 1,
        "mesh": None,
        "placements": [],
        "owner": None,
        "aliases": [],
        "sha256": hashlib.sha256(f"{rank}:{update}:{phase}:{name}:{microbatch}".encode()).hexdigest(),
        "is_finite": True if dtype.startswith(("torch.float", "torch.bfloat", "torch.complex")) else None,
    }


def _write_reference_run(
    tmp_path: Path, capture_update: int, *, world_size: int = 8
) -> Path:
    prefix_updates = list(range(1, capture_update + 1))
    namespace = tmp_path / "reference-namespace"
    namespace.mkdir()
    measurement = tmp_path / "measurement"
    measurement.mkdir()
    hashes = {
        "native_numerics_sha256": "a" * 64,
        "native_numerics_payload_sha256": "b" * 64,
    }
    training_metadata = {
        "initial_step": 0,
        "mtp_loss_weights": [0.3],
        "initial_num_chunks": 2,
        "resolved_config_sha256": "c" * 64,
        "ownership": [],
        "deterministic_backend_flags": {},
    }
    expected_coverage = {
        str(capture_update): {
            "input_batches": 32,
            "microbatches": 32,
            "summary_calls": 1,
            "clip_calls": 1,
            "optimizer_calls": 1,
            "batch_end_calls": 1,
        }
    }
    metadata = {
        "run_id": f"reference-{capture_update}",
        "steps": capture_update,
        "warmup_updates": 0,
        "window_updates": 0,
        "monitor_interval": 50,
        "config_hashes_before": hashes,
        "native_reference": {
            "schema": "native-numerical-calibration-v1",
            "mode": "export",
            "capture_update": capture_update,
            "prefix_updates": prefix_updates,
            "expected_microbatches": 32,
            "monitor_interval": 50,
            "reference_namespace": str(namespace),
        },
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    (tmp_path / "config-hashes-final.json").write_text(json.dumps(hashes))

    for rank in range(world_size):
        observations = [
            _reference_tensor_observation(rank, 0, "initial_parameter", "model.0.weight")
        ]
        initial_router_observation = _reference_tensor_observation(
            rank,
            0,
            "router_after_load",
            "model.0.router.tokens_per_expert_E",
        )
        initial_router_observation["owner"] = {
            "part_index": 0,
            "module_fqn": "router",
        }
        observations.append(initial_router_observation)
        for name in ("python.random", "numpy.random", "torch.cpu", "torch.cuda.current_rank"):
            observations.append(
                {
                    "update": 0,
                    "phase": "fit_start_rng",
                    "name": name,
                    "microbatch": None,
                    "rank": rank,
                    "present": True,
                    "dtype": "sha256",
                    "owner": None,
                    "aliases": [],
                    "sha256": hashlib.sha256(name.encode()).hexdigest(),
                }
            )
        for update in prefix_updates:
            for microbatch in range(32):
                for name in ("input.tokens", "labels"):
                    observations.append(
                        _reference_tensor_observation(
                            rank,
                            update,
                            "dataloader_input",
                            name,
                            microbatch=microbatch,
                            dtype="torch.int64",
                        )
                    )
        chunk_count = 32 if capture_update == 50 else 2
        for microbatch in range(32):
            observations.append(
                {
                    "update": capture_update,
                    "phase": "microbatch_metadata",
                    "name": "trainer.forward_backward_step",
                    "microbatch": microbatch,
                    "rank": rank,
                    "present": True,
                    "active_num_chunks": chunk_count,
                }
            )
            observations.append(
                _reference_tensor_observation(
                    rank,
                    capture_update,
                    "microbatch_output",
                    "loss.local_normalized",
                    microbatch=microbatch,
                )
            )
        for phase, name in (
            ("grad_preclip", "model.0.weight.grad"),
            ("grad_postclip", "model.0.weight.grad"),
            ("post_optimizer_parameter", "model.0.weight"),
        ):
            observations.append(
                _reference_tensor_observation(rank, capture_update, phase, name)
            )
        router_observation = _reference_tensor_observation(
            rank,
            capture_update,
            "router_post_optimizer",
            "model.0.router.tokens_per_expert_E",
        )
        router_observation["owner"] = {"part_index": 0, "module_fqn": "router"}
        observations.append(router_observation)

        for phase, name, value, present in (
            ("loss_components", "loss.components.scalar.float", 0.0, True),
            ("update_summary", "scalar.int", 7, True),
            ("update_summary", "scalar.none", None, False),
        ):
            observations.append(
                {
                    "update": capture_update,
                    "phase": phase,
                    "name": name,
                    "microbatch": None,
                    "rank": rank,
                    "present": present,
                    "value": value,
                    "owner": None,
                    "aliases": [],
                }
            )

        observations.extend(
            [
                {
                    "update": capture_update,
                    "phase": "optimizer_group",
                    "name": "optimizer.0.group.0",
                    "microbatch": None,
                    "rank": rank,
                    "present": True,
                    "optimizer_class": "torch.optim.AdamW",
                    "hyperparameters": {"lr": 0.001, "betas": [0.9, 0.999]},
                    "owner": None,
                    "aliases": [],
                },
                {
                    "update": capture_update,
                    "phase": "scheduler_state",
                    "name": "trainer.lr_schedulers",
                    "microbatch": None,
                    "rank": rank,
                    "present": True,
                    "value": {"_last_lr": [0.001]},
                    "owner": None,
                    "aliases": [],
                },
                _reference_tensor_observation(
                    rank,
                    capture_update,
                    "optimizer_state",
                    "model.0.weight.optimizer.step",
                ),
                _reference_tensor_observation(
                    rank,
                    capture_update,
                    "update_summary",
                    "global_valid_tokens",
                ),
            ]
        )

        monitor = {"status": "not_applicable"}
        if capture_update == 50:
            record_metadata = [
                {
                    "index": 0,
                    "name": "train.loss",
                    "step": 50,
                    "reduce_mode": "online",
                    "dtype": "torch.float32",
                    "shape": [],
                    "scalar": True,
                    "multiplicity": 1,
                },
                {
                    "index": 1,
                    "name": "memory.max_reserved_bytes",
                    "step": 50,
                    "reduce_mode": "online",
                    "dtype": "torch.float32",
                    "shape": [],
                    "scalar": True,
                    "multiplicity": 1,
                },
            ]
            monitor = {
                "status": "captured",
                "event": "batch_end",
                "update": 50,
                "emitter_calls": 1,
                "record_count": len(record_metadata),
                "records": record_metadata,
                "publish_calls": [
                    {
                        "index": 0,
                        "step": 50,
                        "metric_names": [item["name"] for item in record_metadata],
                    }
                ],
            }
            for item in record_metadata:
                observations.append(
                    {
                        **item,
                        "update": 50,
                        "phase": "monitor_record",
                        "name": f"record[{item['index']}].{item['name']}",
                        "microbatch": None,
                        "rank": rank,
                        "present": True,
                        "owner": None,
                        "aliases": [],
                        "record_name": item["name"],
                    }
                )
            for metric_index, (metric_name, value, is_resource) in enumerate(
                (
                    ("train.loss", 1.25, False),
                    ("memory.max_reserved_bytes", 2048.0, True),
                )
            ):
                observations.append(
                    {
                        "update": 50,
                        "phase": "monitor_publish",
                        "name": f"publish[0].{metric_name}",
                        "microbatch": None,
                        "rank": rank,
                        "present": True,
                        "owner": None,
                        "aliases": [],
                        "publish_call_index": 0,
                        "publish_metric_index": metric_index,
                        "publish_step": 50,
                        "metric_name": metric_name,
                        "value": value,
                        "resource_observation": is_resource,
                    }
                )

        rank_dir = namespace / f"rank-{rank:02d}"
        tensors_dir = rank_dir / "tensors"
        tensors_dir.mkdir(parents=True)
        manifest_items = []
        payload_index = 0
        for observation in observations:
            key = {
                field: observation[field]
                for field in ("rank", "update", "phase", "name", "microbatch")
            }
            entry = {"key": key, "metadata": observation}
            if (
                observation["update"] == capture_update
                and observation["phase"] not in {"dataloader_input", "monitor_record"}
                and observation.get("present") is True
                and str(observation.get("dtype", "")).startswith("torch.")
            ):
                payload_name = f"tensors/{payload_index}.bin"
                (rank_dir / payload_name).write_bytes(b"\0\0\0\0")
                entry.update(payload_file=payload_name, nbytes=4)
                payload_index += 1
            manifest_items.append(entry)
        manifest = {
            "schema": "native-numerical-reference-v1",
            "complete": True,
            "run_id": f"reference-{capture_update}",
            "rank": rank,
            "capture_update": capture_update,
            "prefix_updates": prefix_updates,
            "coverage": expected_coverage,
            "training_metadata": training_metadata,
            "monitor_attestation": monitor,
            "observations": manifest_items,
        }
        manifest_path = rank_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        receipt = {
            "schema": "native-numerical-calibration-v1",
            "complete": True,
            "rank": rank,
            "capture_updates": [capture_update],
            "prefix_updates": prefix_updates,
            "expected_microbatches": 32,
            "coverage": expected_coverage,
            "training_metadata": training_metadata,
            "monitor_attestation": monitor,
            "observations": observations,
            "reference": {
                "mode": "export",
                "capture_update": capture_update,
                "manifest": str(manifest_path),
            },
        }
        (measurement / f"native-numerics-rank{rank}.json").write_text(json.dumps(receipt))
    return namespace


def test_native_numerics_runner_and_helper_flags_are_forwarded_and_bounded(tmp_path, monkeypatch):
    args = run._parse_args(["--run-id", "n1", "--steps", "50", "--native-numerics"])
    assert args.native_numerics
    paths = {key: tmp_path / key for key in ("measurement", "config", "native", "logs")}
    command = run._command(args, paths)
    assert "--native-numerics" in command
    ordinary = run._parse_args(["--run-id", "n2", "--steps", "50"])
    assert "--native-numerics" not in run._command(ordinary, paths)

    for extra in (
        ["--steps", "49"],
        ["--warmup", "1"],
        ["--window", "50", "--warmup", "1", "--steps", "51"],
        ["--monitor-interval", "5"],
        ["--profile-update", "2", "--warmup", "1", "--steps", "2"],
    ):
        with pytest.raises(SystemExit):
            run._parse_args(["--run-id", "n3", "--steps", "50", "--native-numerics", *extra])

    helper = [
        "--measurement-dir", str(tmp_path), "--run-id", "n1", "--steps", "50",
        "--warmup", "0", "--window", "0", "--monitor-interval", "50",
        "--native-numerics", "--", "experiment.yaml",
    ]
    assert measure._parse_args(helper)[0].native_numerics
    with pytest.raises(SystemExit):
        measure._parse_args(helper[:helper.index("--steps") + 1] + ["49"] + helper[helper.index("--steps") + 2:])


@pytest.mark.parametrize("capture_update", [1, 2, 50])
def test_reference_capture_update_is_bounded_and_forwarded(tmp_path, capture_update):
    namespace = tmp_path / f"reference-{capture_update}"
    args = run._parse_args(
        [
            "--run-id",
            f"reference-{capture_update}",
            "--steps",
            str(capture_update),
            "--native-reference-output",
            str(namespace),
            "--native-capture-update",
            str(capture_update),
        ]
    )
    assert args.native_capture_update == capture_update
    command = run._command(args, {key: tmp_path / key for key in ("measurement", "config", "native", "logs")})
    assert "--native-reference-output" in command
    assert str(namespace) in command
    assert command[command.index("--native-capture-update") + 1] == str(capture_update)

    helper = [
        "--measurement-dir",
        str(tmp_path),
        "--run-id",
        f"reference-{capture_update}",
        "--steps",
        str(capture_update),
        "--warmup",
        "0",
        "--window",
        "0",
        "--monitor-interval",
        "50",
        "--native-reference-input",
        str(namespace),
        "--native-capture-update",
        str(capture_update),
        "--",
        "experiment.yaml",
    ]
    parsed, _ = measure._parse_args(helper)
    assert parsed.native_capture_update == capture_update
    with pytest.raises(SystemExit):
        measure._parse_args([*helper[: helper.index("--native-capture-update") + 2], "--capture-microbatches", *helper[helper.index("--") :]])


def test_reference_cli_rejects_mixed_modes_and_wrong_prefix_length():
    with pytest.raises(SystemExit):
        run._parse_args(
            [
                "--run-id", "reference-mixed", "--steps", "1",
                "--native-numerics", "--native-reference-output", "reference",
                "--native-capture-update", "1",
            ]
        )
    with pytest.raises(SystemExit):
        run._parse_args(
            [
                "--run-id", "reference-long", "--steps", "2",
                "--native-reference-output", "reference",
                "--native-capture-update", "1",
            ]
        )


@pytest.mark.parametrize("capture_update", [1, 2, 50])
def test_reference_parser_checks_observation_counts_and_selected_chunk_count(
    tmp_path, monkeypatch, capture_update
):
    monkeypatch.setattr(parse_results, "WORLD_SIZE", 1)
    _write_reference_run(tmp_path, capture_update, world_size=1)
    result = parse_results.summarize_run(tmp_path)
    assert result["status"] == "complete"
    assert result["mode"] == "native_reference_export"
    assert result["capture_update"] == capture_update
    assert result["prefix_updates"] == list(range(1, capture_update + 1))
    assert result["payload_count"] == 38


def _observation_identity(item):
    return tuple(item[field] for field in ("rank", "update", "phase", "name", "microbatch"))


def _write_consistent_reference_compare_run(
    tmp_path: Path,
    *,
    capture_update: int = 50,
    allow_drift: bool = True,
    scenario: str | None = None,
) -> tuple[Path, Path]:
    import torch

    import numerics_payload

    export_run = tmp_path / "export-run"
    export_run.mkdir()
    namespace = _write_reference_run(export_run, capture_update, world_size=1)
    export_metadata = json.loads((export_run / "metadata.json").read_text())
    export_receipt = json.loads(
        (export_run / "measurement" / "native-numerics-rank0.json").read_text()
    )
    reference_manifest_path = namespace / "rank-00" / "manifest.json"
    reference_manifest = json.loads(reference_manifest_path.read_text())
    rank_dir = namespace / "rank-00"
    reference_observations = {
        _observation_identity(item): item for item in export_receipt["observations"]
    }
    changed_tensor_key = (
        0,
        capture_update,
        "grad_preclip",
        "model.0.weight.grad",
        None,
    )
    step_tensor_key = (
        0,
        capture_update,
        "optimizer_state",
        "model.0.weight.optimizer.step",
        None,
    )
    denominator_tensor_key = (
        0,
        capture_update,
        "update_summary",
        "global_valid_tokens",
        None,
    )
    router_count_key = (
        0,
        capture_update,
        "router_post_optimizer",
        "model.0.router.tokens_per_expert_E",
        None,
    )
    initial_router_key = (
        0,
        0,
        "router_after_load",
        "model.0.router.tokens_per_expert_E",
        None,
    )
    reference_payloads = {}

    for entry in reference_manifest["observations"]:
        item = entry["metadata"]
        key = _observation_identity(item)
        payload_file = entry.get("payload_file")
        if scenario == "integer_router_count" and key == router_count_key:
            item["dtype"] = "torch.int64"
            item["is_finite"] = None
            entry["nbytes"] = 8
            reference_observations[key]["dtype"] = "torch.int64"
            reference_observations[key]["is_finite"] = None
        if payload_file is not None:
            payload_path = rank_dir / payload_file
            if key == step_tensor_key:
                payload = torch.tensor([1.0], dtype=torch.float32).numpy().tobytes()
            elif key == denominator_tensor_key:
                payload = torch.tensor([1024.0], dtype=torch.float32).numpy().tobytes()
            elif key == router_count_key:
                route_dtype = torch.int64 if scenario == "integer_router_count" else torch.float32
                route_value = 128 if scenario == "integer_router_count" else 128.0
                payload = torch.tensor([route_value], dtype=route_dtype).numpy().tobytes()
            elif scenario in {"matching_nan", "one_sided_nan"} and key == changed_tensor_key:
                payload = torch.tensor([float("nan")], dtype=torch.float32).numpy().tobytes()
            else:
                payload = payload_path.read_bytes()
            payload_path.write_bytes(payload)
            reference_payloads[key] = payload
            reference_tensor = torch.frombuffer(
                bytearray(payload),
                dtype=torch.int64
                if item.get("dtype") == "torch.int64"
                else torch.float32,
            ).clone()
            digest = hashlib.sha256(payload).hexdigest()
            item["sha256"] = digest
            item["is_finite"] = (
                bool(torch.isfinite(reference_tensor).all().item())
                if reference_tensor.is_floating_point()
                else None
            )
            reference_observations[key]["sha256"] = digest
            reference_observations[key]["is_finite"] = item["is_finite"]

    reference_manifest_path.write_text(json.dumps(reference_manifest))
    (export_run / "measurement" / "native-numerics-rank0.json").write_text(
        json.dumps(export_receipt)
    )

    current_receipt = json.loads(json.dumps(export_receipt))
    current_observations = {
        _observation_identity(item): item for item in current_receipt["observations"]
    }
    current_tensor_overrides = {}
    if scenario is None and allow_drift:
        current_tensor_overrides[changed_tensor_key] = torch.tensor(
            [0.25], dtype=torch.float32
        )
        current_observations[
            (0, capture_update, "loss_components", "loss.components.scalar.float", None)
        ]["value"] = 0.25
        if capture_update == 50:
            current_observations[
                (0, 50, "monitor_publish", "publish[0].train.loss", None)
            ]["value"] = 1.5
            current_observations[
                (0, 50, "monitor_publish", "publish[0].memory.max_reserved_bytes", None)
            ]["value"] = 3072.0
    elif scenario == "optimizer_lr":
        current_observations[
            (0, capture_update, "optimizer_group", "optimizer.0.group.0", None)
        ]["hyperparameters"]["lr"] = 0.002
    elif scenario == "scheduler_lr":
        current_observations[
            (0, capture_update, "scheduler_state", "trainer.lr_schedulers", None)
        ]["value"]["_last_lr"][0] = 0.002
    elif scenario == "adamw_step":
        current_tensor_overrides[step_tensor_key] = torch.tensor(
            [2.0], dtype=torch.float32
        )
    elif scenario == "global_valid_tokens":
        current_tensor_overrides[denominator_tensor_key] = torch.tensor(
            [2048.0], dtype=torch.float32
        )
    elif scenario == "router_count":
        current_tensor_overrides[router_count_key] = torch.tensor(
            [127.0], dtype=torch.float32
        )
    elif scenario == "integer_router_count":
        current_tensor_overrides[router_count_key] = torch.tensor(
            [127], dtype=torch.int64
        )
    elif scenario == "initial_router":
        current_observations[initial_router_key]["sha256"] = "e" * 64
    elif scenario == "one_sided_nan":
        current_tensor_overrides[changed_tensor_key] = torch.tensor(
            [0.0], dtype=torch.float32
        )
    elif scenario not in {None, "matching_nan"}:
        raise AssertionError(f"unknown comparison fixture scenario: {scenario}")

    for key, tensor in current_tensor_overrides.items():
        payload = tensor.numpy().tobytes()
        current_observations[key]["sha256"] = hashlib.sha256(payload).hexdigest()
        current_observations[key]["is_finite"] = (
            bool(torch.isfinite(tensor).all().item())
            if tensor.is_floating_point()
            else None
        )

    compare_run = tmp_path / "compare-run"
    compare_measurement = compare_run / "measurement"
    compare_measurement.mkdir(parents=True)
    (compare_run / "metadata.json").write_text(
        json.dumps(
            {
                **export_metadata,
                "run_id": "reference-compare",
                "native_reference": {
                    **export_metadata["native_reference"],
                    "mode": "compare",
                },
            }
        )
    )
    (compare_run / "config-hashes-final.json").write_text(
        json.dumps(export_metadata["config_hashes_before"])
    )

    comparison_dir = compare_measurement / "native-numerical-comparison" / "rank-00"
    observation_dir = comparison_dir / "observations"
    observation_dir.mkdir(parents=True)
    identities = {
        _observation_identity(item): item for item in current_receipt["observations"]
    }
    gated_ids = {
        key
        for key in identities
        if key[1] == 0
        or (key[2] == "dataloader_input" and key[1] <= capture_update)
    }
    expected_result_ids = set(identities) - gated_ids
    payload_ids = set()
    result_rows = []
    for entry in reference_manifest["observations"]:
        reference_item = entry["metadata"]
        key = _observation_identity(reference_item)
        payload_file = entry.get("payload_file")
        if key not in expected_result_ids:
            continue
        key_object = {
            field: reference_item[field]
            for field in ("rank", "update", "phase", "name", "microbatch")
        }
        key_digest = hashlib.sha256(
            json.dumps(key_object, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        relative_result = f"observations/{key_digest}.json"
        current_item = identities[key]
        result = {
            "schema": "native-numerical-comparison-v1",
            "key": key_object,
            "metadata_equal": True,
        }
        if payload_file is not None:
            payload_ids.add(key)
            current_tensor = current_tensor_overrides.get(key)
            if current_tensor is None:
                tensor_dtype = (
                    torch.int64
                    if identities[key].get("dtype") == "torch.int64"
                    else torch.float32
                )
                current_tensor = torch.frombuffer(
                    bytearray(reference_payloads[key]), dtype=tensor_dtype
                ).clone()
            result.update(
                comparison_eligible=True,
                metrics=numerics_payload._compare_tensor_file(
                    current_tensor,
                    rank_dir / payload_file,
                    reference_item,
                    expected_current_sha256=current_item["sha256"],
                    expected_current_finite=current_item["is_finite"],
                ),
            )
        elif (
            reference_item.get("phase") == "monitor_publish"
            and reference_item.get("resource_observation") is True
        ):
            result.update(
                comparison_eligible=False,
                reference_value=reference_item["value"],
                current_value=current_item["value"],
                reason="timing, throughput, remaining-time, or memory observation",
            )
        else:
            result.update(
                comparison_eligible=True,
                numeric_comparison=numerics_payload._numeric_tree(
                    reference_item, current_item
                ),
            )
        result_path = comparison_dir / relative_result
        result_bytes = (
            json.dumps(
                result,
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
        result_path.write_bytes(result_bytes)
        result_rows.append(
            {
                "key": key_object,
                "result": relative_result,
                "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
            }
        )
        if payload_file is not None:
            (rank_dir / payload_file).unlink()

    expected_inputs = [
        item
        for key, item in identities.items()
        if key[2] == "dataloader_input" and key[1] <= capture_update
    ]
    prefix_rows = [
        {
            "key": {
                field: item[field]
                for field in ("rank", "update", "phase", "name", "microbatch")
            },
            "sha256": item["sha256"],
            "equal": True,
        }
        for item in expected_inputs
    ]
    (comparison_dir / "initial-gates.json").write_text(
        json.dumps(
            {
                "schema": "native-numerical-comparison-v1",
                "complete": True,
                "gate": "initial_state",
                "rank": 0,
                "observation_count": sum(key[1] == 0 for key in identities),
                "training_metadata_equal": True,
            }
        )
    )
    (comparison_dir / "prefix-inputs.json").write_text(
        json.dumps(
            {
                "schema": "native-numerical-comparison-v1",
                "complete": True,
                "gate": "all_prefix_cpu_inputs",
                "rank": 0,
                "updates": list(range(1, capture_update + 1)),
                "observation_count": len(prefix_rows),
                "observations": prefix_rows,
            }
        )
    )
    comparison_path = comparison_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "schema": "native-numerical-comparison-v1",
                "complete": True,
                "run_id": "reference-compare",
                "rank": 0,
                "capture_update": capture_update,
                "reference_manifest_sha256": hashlib.sha256(
                    reference_manifest_path.read_bytes()
                ).hexdigest(),
                "reference_run_id": f"reference-{capture_update}",
                "observation_count": len(identities),
                "payload_comparison_count": len(payload_ids),
                "scalar_comparison_count": len(expected_result_ids - payload_ids),
                "initial_gate": "exact_match",
                "prefix_input_gate": "all_hashes_equal",
                "observations": result_rows,
            }
        )
    )
    current_receipt["reference"] = {
        "mode": "compare",
        "capture_update": capture_update,
        "manifest": str(comparison_path),
    }
    (compare_measurement / "native-numerics-rank0.json").write_text(
        json.dumps(current_receipt)
    )
    return compare_run, comparison_dir


@pytest.mark.parametrize("capture_update", [1, 2, 50])
def test_reference_parser_accepts_bound_nonzero_comparison_and_resource_publication(
    tmp_path, monkeypatch, capture_update
):
    monkeypatch.setattr(parse_results, "WORLD_SIZE", 1)
    compare_run, comparison_dir = (
        _write_consistent_reference_compare_run(
            tmp_path, capture_update=capture_update
        )
    )
    result = parse_results.summarize_run(compare_run)
    assert result["status"] == "complete"
    assert result["mode"] == "native_reference_compare"
    assert result["payload_count"] == 38

    observations = {
        _observation_identity(json.loads(path.read_text())["key"]): json.loads(
            path.read_text()
        )
        for path in (comparison_dir / "observations").glob("*.json")
    }
    grad = observations[
        (0, capture_update, "grad_preclip", "model.0.weight.grad", None)
    ]
    assert grad["metrics"]["exact_hash_match"] is False
    assert grad["metrics"]["max_abs_error"] == 0.25
    scalar = observations[
        (0, capture_update, "loss_components", "loss.components.scalar.float", None)
    ]
    assert scalar["numeric_comparison"]["children"][0]["current"] == 0.25
    if capture_update == 50:
        resource = observations[
            (0, 50, "monitor_publish", "publish[0].memory.max_reserved_bytes", None)
        ]
        assert resource["comparison_eligible"] is False
        assert resource["reference_value"] == 2048.0
        assert resource["current_value"] == 3072.0
    assert observations[(0, capture_update, "update_summary", "scalar.int", None)][
        "numeric_comparison"
    ] == {}
    assert observations[(0, capture_update, "update_summary", "scalar.none", None)][
        "numeric_comparison"
    ] == {}


@pytest.mark.parametrize(
    "tamper",
    ["missing", "forged", "wrongcount", "wrongreference", "currentdigest", "invalidkind"],
)
def test_reference_parser_rejects_incomplete_or_unbound_payload_results(
    tmp_path, monkeypatch, tamper
):
    monkeypatch.setattr(parse_results, "WORLD_SIZE", 1)
    compare_run, comparison_dir = _write_consistent_reference_compare_run(
        tmp_path, allow_drift=False
    )
    comparison_path = comparison_dir / "comparison.json"
    comparison_manifest = json.loads(comparison_path.read_text())
    row = next(
        item
        for item in comparison_manifest["observations"]
        if item["key"]["phase"] == "grad_preclip"
    )
    result_path = comparison_dir / row["result"]
    comparison = json.loads(result_path.read_text())
    if tamper == "missing":
        del comparison["metrics"]
    elif tamper == "forged":
        comparison["metrics"]["exact_hash_match"] = False
    elif tamper == "wrongcount":
        comparison["metrics"]["count"] += 1
    elif tamper == "wrongreference":
        comparison["metrics"]["reference_sha256"] = "0" * 64
    elif tamper == "currentdigest":
        comparison["metrics"]["current_sha256"] = "0" * 64
    else:
        comparison["comparison_eligible"] = False
    result_bytes = (
        json.dumps(
            comparison,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    result_path.write_bytes(result_bytes)
    row["result_sha256"] = hashlib.sha256(result_bytes).hexdigest()
    comparison_path.write_text(json.dumps(comparison_manifest))

    result = parse_results.summarize_run(compare_run)
    assert result["status"] == "incomplete"


@pytest.mark.parametrize(
    "scenario",
    [
        "optimizer_lr",
        "scheduler_lr",
        "adamw_step",
        "global_valid_tokens",
        "integer_router_count",
        "matching_nan",
        "one_sided_nan",
        "initial_router",
    ],
)
def test_reference_parser_keeps_config_counters_and_nonfinite_tensors_exact(
    tmp_path, monkeypatch, scenario
):
    monkeypatch.setattr(parse_results, "WORLD_SIZE", 1)
    compare_run, _ = _write_consistent_reference_compare_run(
        tmp_path, allow_drift=False, scenario=scenario
    )

    result = parse_results.summarize_run(compare_run)
    assert result["status"] == "incomplete"


def test_reference_parser_accepts_fp32_router_count_delta(tmp_path, monkeypatch):
    monkeypatch.setattr(parse_results, "WORLD_SIZE", 1)
    compare_run, comparison_dir = _write_consistent_reference_compare_run(
        tmp_path, allow_drift=False, scenario="router_count"
    )

    result = parse_results.summarize_run(compare_run)
    assert result["status"] == "complete"
    router_row = next(
        row
        for row in json.loads((comparison_dir / "comparison.json").read_text())[
            "observations"
        ]
        if row["key"]["name"] == "model.0.router.tokens_per_expert_E"
    )
    router_result = json.loads((comparison_dir / router_row["result"]).read_text())
    assert router_result["metrics"]["reference_sha256"] != router_result["metrics"]["current_sha256"]
    assert router_result["metrics"]["max_abs_error"] == 1.0


def test_config_hashes_include_numerics_module_only_when_enabled():
    default_hashes = run._config_hashes()
    enabled_hashes = run._config_hashes(native_numerics=True)
    assert "native_numerics_sha256" not in default_hashes
    assert enabled_hashes["native_numerics_sha256"] == run.sha256_file(
        run.HELPER_DIR / "numerics.py"
    )


def test_runner_failure_finally_retains_enabled_numerics_hash(tmp_path, monkeypatch):
    attempt = tmp_path / "attempt"
    paths = {"attempt": attempt}
    attempt.mkdir()
    for name in ("logs", "preflight", "config", "native", "measurement"):
        paths[name] = attempt / name
        paths[name].mkdir()
    owned_tmp = tmp_path / "owned-tmp"
    owned_tmp.mkdir()
    monkeypatch.setattr(run, "PYTHON", Path(sys.executable))
    monkeypatch.setattr(run, "verify_task_root", lambda root: None)
    monkeypatch.setattr(run, "_make_run_dirs", lambda run_id: paths)
    monkeypatch.setattr(run, "_data_root", lambda root: tmp_path / "mds")
    monkeypatch.setattr(run, "_filesystem_status", lambda run_paths, data: {})
    monkeypatch.setattr(run, "_make_local_tmpdir", lambda run_id: (owned_tmp, {}))
    monkeypatch.setattr(run, "_run_environment", lambda run_paths: {})
    monkeypatch.setattr(
        run,
        "_write_experiment_config",
        lambda path: path.write_text("test config\n", encoding="utf-8"),
    )
    source = {"head": "fixed", "tracked_and_untracked_sha256": "a" * 64}
    monkeypatch.setattr(run, "source_snapshot", lambda root: source)
    monkeypatch.setattr(
        run,
        "_runtime_preflight",
        lambda env, run_paths: (_ for _ in ()).throw(RuntimeError("sentinel preflight failure")),
    )
    monkeypatch.setattr(run, "_cleanup_local_tmpdir", lambda path: path.rmdir())
    monkeypatch.setattr(run, "_launch", lambda *args, **kwargs: pytest.fail("launch must not run"))

    original_hashes = run._config_hashes
    native_flags = []
    observed_native_hashes = []

    def record_hash_call(*, native_numerics=False, native_reference=False):
        native_flags.append(native_numerics)
        hashes = original_hashes(
            native_numerics=native_numerics,
            native_reference=native_reference,
        )
        observed_native_hashes.append(hashes.get("native_numerics_sha256"))
        return hashes

    monkeypatch.setattr(run, "_config_hashes", record_hash_call)
    assert run.main(["--run-id", "n-failure", "--steps", "50", "--native-numerics"]) == 1
    assert native_flags == [True, True]
    assert observed_native_hashes == [run.sha256_file(run.HELPER_DIR / "numerics.py")] * 2
    final_hashes = json.loads((attempt / "config-hashes-final.json").read_text())
    assert final_hashes["native_numerics_sha256"] == run.sha256_file(
        run.HELPER_DIR / "numerics.py"
    )
    assert json.loads((attempt / "failure.json").read_text())["status"] == "FAIL"


@pytest.mark.parametrize("change", ["input", "gradient", "optimizer_dtype", "denominator"])
def test_exact_comparator_detects_changed_receipt_fields(tmp_path, change):
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    a = _receipt(0)
    b = _receipt(0)
    if change == "input":
        next(item for item in b["observations"] if item["phase"] == "dataloader_input")["sha256"] = "c" * 64
    elif change == "gradient":
        next(item for item in b["observations"] if item["phase"] == "grad_postclip")["sha256"] = "d" * 64
    elif change == "optimizer_dtype":
        next(item for item in b["observations"] if item["phase"] == "post_optimizer_parameter")["dtype"] = "torch.bfloat16"
    else:
        next(item for item in b["observations"] if item["phase"] == "update_summary")["value"] += 1.0
    left.write_text(json.dumps(a))
    right.write_text(json.dumps(b))
    assert parse_results.compare_native_numerical_receipts(left, left) == {"equal": True, "mismatch_paths": []}
    comparison = parse_results.compare_native_numerical_receipts(left, right)
    assert not comparison["equal"]
    assert comparison["mismatch_paths"]


def test_native_numerics_parser_accepts_valid_eight_rank_capture(tmp_path):
    _write_run(tmp_path)
    result = parse_results.summarize_run(tmp_path)
    assert result["status"] == "complete"
    assert result["mode"] == "native_numerics"
    assert result["scope"] == "diagnostic only; no TPS"
    assert "tps" not in result and "throughput" not in result


@pytest.mark.parametrize(
    "phase",
    [
        "dataloader_input",
        "microbatch_output",
        "grad_preclip",
        "grad_postclip",
        "post_optimizer_parameter",
        "clip_result",
    ],
)
def test_parser_rejects_identity_only_tensor_observations(tmp_path, phase):
    def strip_payload(receipt):
        row = next(
            item for item in receipt["observations"]
            if item["update"] == 1 and item["phase"] == phase
        )
        for key in (
            "present", "rank", "dtype", "source_device_type", "global_shape",
            "local_shape", "local_stride", "local_numel", "mesh", "placements",
            "sha256", "is_finite",
        ):
            row.pop(key, None)

    _write_run(tmp_path, mutate=strip_payload)
    result = parse_results.summarize_run(tmp_path)
    assert result["status"] == "incomplete"
    assert any("malformed or non-finite records" in error for error in result["errors"])


def test_parser_requires_floating_tensor_finite_flag(tmp_path):
    def omit_finite(receipt):
        row = next(
            item for item in receipt["observations"]
            if item["update"] == 1 and item["phase"] == "microbatch_output"
        )
        row["is_finite"] = None

    _write_run(tmp_path, mutate=omit_finite)
    result = parse_results.summarize_run(tmp_path)
    assert result["status"] == "incomplete"
    assert any("malformed or non-finite records" in error for error in result["errors"])


def test_parser_accepts_none_gradients(tmp_path):
    def none_grad(receipt):
        grad = next(
            item for item in receipt["observations"]
            if item["update"] == 1 and item["phase"] == "grad_preclip"
        )
        grad.clear()
        grad.update(
            update=1,
            phase="grad_preclip",
            name="model.0.optional.grad",
            microbatch=None,
            present=False,
            dtype=None,
            shape=None,
            sha256=None,
            is_finite=None,
        )

    _write_run(tmp_path, mutate=none_grad)
    result = parse_results.summarize_run(tmp_path)
    assert result["status"] == "complete"


def test_parser_requires_captured_trainer_steps(tmp_path):
    def remove_steps(receipt):
        receipt["observations"] = [
            item for item in receipt["observations"]
            if not (item.get("phase") == "batch_end" and item.get("name") == "trainer.step")
        ]

    _write_run(tmp_path, mutate=remove_steps)
    result = parse_results.summarize_run(tmp_path)
    assert result["status"] == "incomplete"
    assert any("captured trainer.step does not match update" in error for error in result["errors"])


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("missing_rank", "exactly eight"),
        ("malformed", "schema or completion"),
        ("count", "call counts"),
        ("chunks", "active chunk count"),
        ("duplicate", "duplicate observation"),
        ("nonfinite", "malformed or non-finite"),
        ("manifest", "manifest is missing or differs"),
    ],
)
def test_native_numerics_parser_rejects_invalid_receipts(tmp_path, case, expected):
    if case == "missing_rank":
        _write_run(tmp_path, ranks=range(7))
    elif case == "malformed":
        _write_run(tmp_path, mutate=lambda receipt: receipt.update(schema="wrong"))
    elif case == "count":
        _write_run(tmp_path, mutate=lambda receipt: receipt["coverage"]["1"].update(clip_calls=2))
    elif case == "chunks":
        def change_chunks(receipt):
            next(item for item in receipt["observations"] if item["phase"] == "microbatch_metadata")["active_num_chunks"] = 8
        _write_run(tmp_path, mutate=change_chunks)
    elif case == "duplicate":
        def duplicate(receipt):
            row = next(item for item in receipt["observations"] if item["phase"] == "after_load")
            receipt["observations"].append(row.copy())
        _write_run(tmp_path, mutate=duplicate)
    elif case == "nonfinite":
        def nonfinite(receipt):
            next(item for item in receipt["observations"] if item["phase"] == "microbatch_output")["is_finite"] = False
        _write_run(tmp_path, mutate=nonfinite)
    else:
        metadata = _write_run(tmp_path)
        metadata["native_numerics"] = {**MANIFEST, "scope": "throughput"}
        (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    result = parse_results.summarize_run(tmp_path)
    assert result["status"] == "incomplete"
    assert any(expected in error for error in result["errors"])


def test_native_observer_installs_before_native_main_and_restores_on_error(tmp_path, monkeypatch):
    class FakeTrainer:
        def forward_backward_step(self):
            pass

    class FakeOptimizer:
        def step(self):
            pass

    class FakeCallbacks:
        def emit(self):
            pass

    def clip(*args, **kwargs):
        pass

    dist_utils = SimpleNamespace(clip_grad_norm_=clip)
    trainer_module = SimpleNamespace(
        Trainer=FakeTrainer, OptimizersContainer=FakeOptimizer,
        CallbackManager=FakeCallbacks, dist_utils=dist_utils,
    )
    originals = (FakeTrainer.forward_backward_step, FakeOptimizer.step, FakeCallbacks.emit, clip)
    lifecycle = []

    class Adapter:
        def __enter__(self):
            lifecycle.append("adapter_enter")
            return self
        def __exit__(self, exc_type, exc, tb):
            lifecycle.append(("adapter_exit", exc_type))

    def native_main(argv):
        lifecycle.append("native_main")
        assert FakeTrainer.forward_backward_step is not originals[0]
        assert FakeOptimizer.step is not originals[1]
        assert FakeCallbacks.emit is not originals[2]
        assert dist_utils.clip_grad_norm_ is not originals[3]
        raise RuntimeError("sentinel config.build failure")

    package = ModuleType("torchtitan")
    package.__path__ = []
    train_module = ModuleType("torchtitan.train")
    train_module.main = native_main
    monkeypatch.setitem(sys.modules, "torchtitan", package)
    monkeypatch.setitem(sys.modules, "torchtitan.train", train_module)
    monkeypatch.setenv("RANK", "0")
    args = SimpleNamespace(native_numerics=True, measurement_dir=tmp_path)
    with pytest.raises(RuntimeError, match="sentinel config.build"):
        measure._run_native_main(args, [], Adapter(), trainer_module)
    assert lifecycle[0] == "adapter_enter"
    assert lifecycle[1] == "native_main"
    assert lifecycle[2] == ("adapter_exit", RuntimeError)
    assert (FakeTrainer.forward_backward_step, FakeOptimizer.step, FakeCallbacks.emit, dist_utils.clip_grad_norm_) == originals
    assert list(tmp_path.iterdir()) == []


def test_default_native_path_does_not_import_numerics_or_write_receipts(tmp_path, monkeypatch):
    calls = []
    original_import = builtins.__import__

    def track_import(name, *args, **kwargs):
        if name == "numerics":
            calls.append(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", track_import)
    package = ModuleType("torchtitan")
    package.__path__ = []
    train_module = ModuleType("torchtitan.train")
    train_module.main = lambda argv: None
    monkeypatch.setitem(sys.modules, "torchtitan", package)
    monkeypatch.setitem(sys.modules, "torchtitan.train", train_module)

    class Adapter:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False

    args = SimpleNamespace(native_numerics=False, measurement_dir=tmp_path)
    measure._run_native_main(args, [], Adapter(), SimpleNamespace())
    assert calls == []
    assert list(tmp_path.iterdir()) == []
