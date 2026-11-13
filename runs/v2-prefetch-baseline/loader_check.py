#!/usr/bin/env python3
"""Bounded CPU/Gloo acceptance check for the V2 streaming loader variant."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


TASK_DIR = Path(__file__).resolve().parent
TASK_ROOT = TASK_DIR.parents[1]
CAMPAIGN_DIR = TASK_ROOT / "runs" / "overnight-streaming-perf"
GIFT_ROOT = TASK_ROOT / "gift"
RAW_ROOT = TASK_DIR / "artifacts" / "raw"
EVIDENCE_PARENT = RAW_ROOT
PYTHON = Path("/opt/torchtitan-venv/bin/python3.12")
PREPARE_MDS = CAMPAIGN_DIR / "prepare_mds.py"
RUNNER = CAMPAIGN_DIR / "run.py"
EXPECTED_TORCH = "2.15.0.dev20260822+cu130"
EXPECTED_STREAMING = "0.13.0"
MAX_PROBE_SECONDS = 120
MAX_STUDY_SECONDS = 900
DIST_TIMEOUT_SECONDS = 120
SKEW_SLEEP_SECONDS = 1.0
DATA_BATCHES = 68
QUEUE_DEPTH = 32


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True
    ).strip()


def source_identity() -> dict[str, Any]:
    """Identify current task source and its in-progress GIFT patch without pinning an old HEAD."""
    diff = subprocess.check_output(
        ["git", "-C", str(TASK_ROOT), "diff", "--binary", "HEAD", "--", "gift/"]
    )
    status = git_output(
        TASK_ROOT, "status", "--porcelain=v1", "--untracked-files=all", "--", "gift/"
    )
    untracked_raw = subprocess.check_output(
        [
            "git",
            "-C",
            str(TASK_ROOT),
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            "gift/",
        ]
    )
    untracked = {
        raw.decode("utf-8"): sha256_file(TASK_ROOT / raw.decode("utf-8"))
        for raw in untracked_raw.split(b"\0")
        if raw
    }
    return {
        "task_head": git_output(TASK_ROOT, "rev-parse", "HEAD"),
        "gift_patch_sha256": hashlib.sha256(diff).hexdigest(),
        "gift_status": status.splitlines(),
        "untracked_gift_files_sha256": untracked,
    }


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_g1_helper() -> Any:
    return load_module(PREPARE_MDS, "v2_g1_prepare_mds")


def tensor_hash(tensor: Any) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    header = f"{contiguous.dtype}:{tuple(contiguous.shape)}:".encode()
    return hashlib.sha256(header + contiguous.numpy().tobytes()).hexdigest()


def scalar_tree(value: Any, path: str = "state") -> Any:
    """Copy checkpoint state while rejecting tensors and non-scalar leaves."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): scalar_tree(child, f"{path}.{key}") for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [scalar_tree(child, f"{path}[{index}]") for index, child in enumerate(value)]
    raise TypeError(f"{path} contains non-scalar checkpoint value {type(value).__name__}")


def batch_content(record: dict[str, Any]) -> dict[str, Any]:
    """Keep assignment and tensor content, excluding nondeterministic process identity."""
    return {key: value for key, value in record.items() if key != "worker_pid"}


def assert_prefetch_cursor_ahead(
    delivered_state: dict[str, Any], last_ready_cursor: dict[str, Any] | None
) -> None:
    """Compare the queue's inner Mosaic cursor with the wrapped checkpoint state."""
    if set(delivered_state) != {"dp_rank_0"}:
        raise AssertionError(f"unexpected delivered loader state keys: {sorted(delivered_state)}")
    if last_ready_cursor is None:
        raise AssertionError("prefetch queue has no last-ready cursor")
    delivered_cursor = delivered_state["dp_rank_0"]
    if scalar_tree(last_ready_cursor) == scalar_tree(delivered_cursor):
        raise AssertionError("producer-ahead state was mistaken for the delivered cursor")


def sum_owner_edge_counts(owner_counts: list[dict[str, int]]) -> dict[str, int]:
    """Sum per-owner first-epoch row-boundary counts using Python integers."""
    if not owner_counts:
        raise AssertionError("no owner boundary counts were gathered")
    keys = set(owner_counts[0])
    if not keys or any(set(counts) != keys for counts in owner_counts):
        raise AssertionError("owners reported different boundary-count fields")
    totals: dict[str, int] = {}
    for key in sorted(keys):
        values = [counts[key] for counts in owner_counts]
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values):
            raise AssertionError(f"invalid nonnegative integer boundary count for {key}")
        totals[key] = sum(values)
    return totals


def validate_first_epoch_ids(owner_ids: list[list[int]], row_count: int) -> list[int]:
    """Require all owners together to cover each real-MDS row exactly once."""
    ids = [sample_id for rank_ids in owner_ids for sample_id in rank_ids]
    expected = list(range(row_count))
    if sorted(ids) != expected:
        raise AssertionError(
            f"eight-owner row coverage differs: total={len(ids)}, unique={len(set(ids))}, "
            f"expected={row_count} unique rows"
        )
    return sorted(ids)


def validate_boundary_skew(owner_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Prove peer epoch-1 next() calls started while rank 0 held its producer pause."""
    if [item.get("rank") for item in owner_results] != list(range(8)):
        raise AssertionError("skew proof requires all eight ordered owner summaries")
    rank0 = owner_results[0]["rank0_boundary_pause"]["details"]
    started = rank0.get("rank0_pause_started_monotonic_ns")
    ended = rank0.get("rank0_pause_ended_monotonic_ns")
    if not isinstance(started, int) or not isinstance(ended, int) or ended <= started:
        raise AssertionError(f"rank 0 pause timestamps are invalid: {started}, {ended}")
    if not rank0.get("pause_active_during_skew") or not rank0.get("pause_released"):
        raise AssertionError("rank 0 prefetch pause was not held and released")
    if not rank0.get("committed_cursor_matches_off55"):
        raise AssertionError("rank 0 pause did not hold the committed 55-delivery cursor")
    if not rank0.get("last_ready_cursor_matches_off56") or not rank0.get("ready_items_held", 0):
        raise AssertionError("rank 0 pause did not retain the last old-epoch ready item")
    rank0_rollover = rank0.get("rollover_advance", {})
    rank0_first_epoch = rank0_rollover.get("new_epoch_first_batch", {})
    if rank0_first_epoch.get("batch_index") != 56:
        raise AssertionError("rank 0 did not enter epoch 1 with the expected batch")
    if rank0_first_epoch.get("started_monotonic_ns", 0) < ended:
        raise AssertionError("rank 0 entered epoch 1 before releasing its producer pause")

    peers = []
    for item in owner_results[1:]:
        details = item["rank0_boundary_pause"]["details"]
        if details.get("rank0_pause_started_monotonic_ns") != started:
            raise AssertionError(f"rank {item['rank']} received a different rank-0 pause start")
        first_epoch = details.get("rollover_advance", {}).get("new_epoch_first_batch", {})
        call_started = first_epoch.get("started_monotonic_ns")
        call_ended = first_epoch.get("ended_monotonic_ns")
        if first_epoch.get("batch_index") != 56 or not isinstance(call_started, int):
            raise AssertionError(f"rank {item['rank']} did not start the epoch-1 next() call")
        if not started <= call_started <= ended:
            raise AssertionError(f"rank {item['rank']} epoch-1 next() did not start during rank-0 pause")
        if not isinstance(call_ended, int) or call_ended <= call_started:
            raise AssertionError(f"rank {item['rank']} epoch-1 next() did not complete")
        peers.append(
            {
                "rank": item["rank"],
                "epoch1_next_started_monotonic_ns": call_started,
                "epoch1_next_ended_monotonic_ns": call_ended,
                "started_during_rank0_pause": True,
            }
        )
    return {
        "status": "passed",
        "rank0_pause_started_monotonic_ns": started,
        "rank0_pause_ended_monotonic_ns": ended,
        "rank0_pause_seconds": (ended - started) / 1_000_000_000,
        "rank0_committed_cursor_matches_off55": True,
        "rank0_retained_old_epoch_last_ready_item": True,
        "peer_epoch1_next_calls_started_during_rank0_pause": peers,
        "rank0_epoch1_next_started_after_pause_release": rank0_first_epoch[
            "started_monotonic_ns"
        ],
        "completion_deadlines": {
            "process_group_seconds": DIST_TIMEOUT_SECONDS,
            "launcher_seconds": MAX_STUDY_SECONDS,
            "all_eight_owners_completed_boundary_and_later_checkpoint_gates": True,
        },
    }


def validate_reused_probe(run_root: Path) -> dict[str, Any]:
    """Reuse only the previous passed probe when its fixture and active source still match."""
    receipt_path = run_root / "receipt.json"
    probe_path = run_root / "probe" / "rank0.json"
    receipt = json.loads(receipt_path.read_text())
    probe = json.loads(probe_path.read_text())
    check = next(
        (item for item in receipt.get("checks", []) if item.get("name") == "world1_real_mds_record_probe"),
        None,
    )
    if receipt.get("status") != "passed" or check is None or not check.get("passed"):
        raise AssertionError("reused world-one probe does not have a passed gate receipt")
    if Path(check.get("result_path", "")).resolve() != probe_path.resolve():
        raise AssertionError("reused probe path differs from its recorded gate result")
    if (
        probe.get("status") != "passed"
        or probe.get("backend") != "gloo"
        or probe.get("world_size") != 1
        or not probe.get("default_zero_has_no_prefetch_iterator")
    ):
        raise AssertionError("reused probe lacks the required world-one default-zero Gloo proof")
    mds = probe.get("mds", {})
    if mds.get("manifest_sha256") != sha256_file(CAMPAIGN_DIR / "mds-data.json"):
        raise AssertionError("reused world-one probe did not use the current fixed MDS manifest")
    runtime = probe.get("runtime", {})
    if (
        runtime.get("cuda_visible_devices") != ""
        or runtime.get("torch_num_threads") != 1
        or runtime.get("thread_environment")
        != {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    ):
        raise AssertionError("reused world-one probe did not use the reviewed CPU thread limits")
    active_files = runtime.get("active_task_gift_files", {})
    if not active_files:
        raise AssertionError("reused world-one probe did not record active task source files")
    for item in active_files.values():
        path = Path(item["path"]).resolve()
        if not path.is_relative_to(GIFT_ROOT.resolve()) or sha256_file(path) != item["sha256"]:
            raise AssertionError(f"reused world-one probe source identity changed: {path}")
    return {
        "status": "passed",
        "receipt_path": str(receipt_path.resolve()),
        "receipt_sha256": sha256_file(receipt_path),
        "probe_result_path": str(probe_path.resolve()),
        "probe_result_sha256": sha256_file(probe_path),
        "mds_manifest_sha256": mds["manifest_sha256"],
        "active_task_gift_file_count": len(active_files),
        "active_task_gift_file_hashes_still_match": True,
        "runtime": {
            "torch": runtime.get("torch"),
            "mosaicml_streaming": runtime.get("mosaicml_streaming"),
            "OMP_NUM_THREADS": runtime["thread_environment"]["OMP_NUM_THREADS"],
            "MKL_NUM_THREADS": runtime["thread_environment"]["MKL_NUM_THREADS"],
            "torch_num_threads": runtime["torch_num_threads"],
            "cuda_visible_devices": runtime["cuda_visible_devices"],
        },
    }


def scalar_leaf_paths(value: Any, prefix: str = "") -> list[str]:
    """List nested state paths so persisted DCP keys are visible in the receipt."""
    if isinstance(value, dict):
        return [
            path
            for key in sorted(value)
            for path in scalar_leaf_paths(value[key], f"{prefix}.{key}" if prefix else str(key))
        ]
    if isinstance(value, list):
        return [
            path
            for index, child in enumerate(value)
            for path in scalar_leaf_paths(child, f"{prefix}[{index}]")
        ]
    return [prefix]


def validate_decoded_dcp_state(
    decoded: Any, expected_loader_state: dict[str, Any]
) -> dict[str, Any]:
    """Compare the complete CPU-decoded checkpoint tree with the saved cursor."""
    actual = scalar_tree(decoded, "decoded")
    expected = scalar_tree({"dataloader": expected_loader_state}, "expected")
    if actual != expected:
        raise AssertionError(
            f"decoded persisted DCP values/keys differ from the saved cursor: {actual!r} != {expected!r}"
        )
    return actual


def state_payload(loader: Any) -> dict[str, Any]:
    state = loader.state_dict()
    if set(state) != {"dp_rank_0"}:
        raise AssertionError(f"unexpected loader state keys: {sorted(state)}")
    scalar_tree(state)
    return copy.deepcopy(state)


def verify_active_imports() -> dict[str, dict[str, str]]:
    import torchtitan.components.tokenizer as tokenizer
    import torchtitan.config.schema.datasets.streaming.core.dataloader as core_config
    import torchtitan.config.schema.datasets.streaming.text.configs as text_configs
    import torchtitan.config.schema.datasets.streaming.text.dataloader as text_config
    import torchtitan.distributed.data_world as data_world
    import torchtitan.datasets.streaming.core.dataloader as core_loader
    import torchtitan.datasets.streaming.core.prefetch as prefetch
    import torchtitan.datasets.streaming.core.streams as streams
    import torchtitan.datasets.streaming.text.collators as collators
    import torchtitan.datasets.streaming.text.datasets as text_datasets
    import torchtitan.datasets.streaming.text.dataloader as text_loader
    import torchtitan.datasets.streaming.text.transforms as transforms

    modules = {
        "tokenizer": tokenizer,
        "core_config": core_config,
        "text_config": text_config,
        "text_configs": text_configs,
        "data_world": data_world,
        "core_loader": core_loader,
        "prefetch": prefetch,
        "streams": streams,
        "collators": collators,
        "text_datasets": text_datasets,
        "text_loader": text_loader,
        "transforms": transforms,
    }
    recorded: dict[str, dict[str, str]] = {}
    root = GIFT_ROOT.resolve()
    for name, module in modules.items():
        source = Path(module.__file__).resolve()
        if not source.is_relative_to(root):
            raise RuntimeError(f"{name} imported outside current task GIFT source: {source}")
        recorded[name] = {"path": str(source), "sha256": sha256_file(source)}
    return recorded


def runtime_identity() -> dict[str, Any]:
    import torch
    import streaming
    from streaming.base.dataset import StreamingDataset
    from streaming.base.dataloader import StreamingDataLoader
    from torchtitan.datasets.streaming.core.dataloader import BaseStreamingDataLoader

    if torch.__version__ != EXPECTED_TORCH:
        raise RuntimeError(f"Torch pin mismatch: expected {EXPECTED_TORCH}, got {torch.__version__}")
    streaming_version = importlib.metadata.version("mosaicml-streaming")
    if streaming_version != EXPECTED_STREAMING:
        raise RuntimeError(
            f"Mosaic Streaming pin mismatch: expected {EXPECTED_STREAMING}, got {streaming_version}"
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU/Gloo child must hide CUDA devices explicitly")
    if os.environ.get("OMP_NUM_THREADS") != "1" or os.environ.get("MKL_NUM_THREADS") != "1":
        raise RuntimeError("CPU/Gloo child must use the reviewed OMP_NUM_THREADS=1, MKL_NUM_THREADS=1 limits")
    if torch.get_num_threads() != 1:
        raise RuntimeError(f"CPU/Gloo child has {torch.get_num_threads()} Torch intra-op threads, expected 1")
    if torch.cuda.is_available():
        raise RuntimeError("CPU/Gloo child unexpectedly has an available CUDA device")

    modules = verify_active_imports()
    package_files = {
        "streaming_dataset": Path(inspect.getsourcefile(StreamingDataset)).resolve(),
        "streaming_dataloader": Path(inspect.getsourcefile(StreamingDataLoader)).resolve(),
        "streaming_package": Path(streaming.__file__).resolve(),
        "loader_adapter": Path(inspect.getsourcefile(BaseStreamingDataLoader)).resolve(),
    }
    return {
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_prefix": str(Path(sys.prefix).resolve()),
        "torch": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "thread_environment": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "mosaicml_streaming": streaming_version,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "active_task_gift_files": modules,
        "installed_reader_files": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in package_files.items()
        },
    }


def validate_mds_inputs(mds: Any) -> dict[str, Any]:
    manifest = json.loads(mds.MANIFEST_PATH.read_text())
    actual_inputs = {
        "corpus": mds.sha256_file(mds.SOURCE_PATH),
        "tokenizer_json": mds.sha256_file(mds.TOKENIZER_DIR / "tokenizer.json"),
        "tokenizer_config": mds.sha256_file(mds.TOKENIZER_DIR / "tokenizer_config.json"),
    }
    if actual_inputs != manifest["input_hashes"] or actual_inputs != mds.EXPECTED_INPUT_HASHES:
        raise AssertionError("real-MDS corpus/tokenizer inputs do not match the accepted receipt")
    actual_mds = mds.describe_mds_output()
    if actual_mds != manifest["mds_output"]:
        raise AssertionError("canonical real-MDS shard/index identity differs from its receipt")
    split = (mds.MDS_ROOT / "train").resolve()
    if not split.is_dir():
        raise FileNotFoundError(f"canonical MDS split is missing: {split}")
    return {
        "manifest_sha256": sha256_file(mds.MANIFEST_PATH),
        "mds_root": str(mds.MDS_ROOT.resolve()),
        "mds_split": str(split),
        "input_hashes": actual_inputs,
        "index_sha256": actual_mds["index_sha256"],
        "row_count": actual_mds["row_count"],
        "shard_count": actual_mds["shard_count"],
    }


def make_loader(
    mds: Any,
    *,
    prefetch_batches: int,
    persistent_workers: bool,
    namespace: str,
) -> tuple[Any, Any, Any]:
    """Use the validated G1 builder while overriding only the scoped loader knobs."""
    original_config = mds.StreamingTextDataLoaderConfig
    original_namespace_factory = mds.get_dataloader_shared_memory_namespace
    original_world_size = mds.WORLD_SIZE

    def selected_config(**kwargs: Any) -> Any:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["num_prefetch_batches"] = prefetch_batches
        return original_config(**kwargs)

    mds.StreamingTextDataLoaderConfig = selected_config
    mds.get_dataloader_shared_memory_namespace = lambda _role: namespace
    mds.WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    try:
        import torch

        torch.manual_seed(2048)
        loader, tokenizer, data_world = mds.build_loader()
    finally:
        mds.StreamingTextDataLoaderConfig = original_config
        mds.get_dataloader_shared_memory_namespace = original_namespace_factory
        mds.WORLD_SIZE = original_world_size

    if loader.num_prefetch_batches != prefetch_batches:
        raise AssertionError("requested prefetch depth did not reach the active loader")
    if loader._loader.persistent_workers != persistent_workers:
        raise AssertionError("requested worker persistence did not reach Torch DataLoader")
    if loader._loader.num_workers != 2 or loader._loader.pin_memory is not False:
        raise AssertionError("loader worker count or canonical pin_memory setting changed")
    if data_world.process_group is None or data_world.process_group is not mds.dist.group.WORLD:
        raise AssertionError("loader does not hold the explicit Gloo data-world group")
    if data_world.world_size != int(os.environ["WORLD_SIZE"]):
        raise AssertionError("loader data-world size differs from the active owner world")
    if data_world.shared_memory_namespace != namespace:
        raise AssertionError("task-specific Mosaic shared-memory namespace was not retained")
    return loader, tokenizer, data_world


def validate_batch(
    mds: Any,
    batch: Any,
    *,
    rank: int,
    batch_index: int,
    tokenizer: Any,
) -> tuple[dict[str, Any], dict[str, int]]:
    import numpy as np
    import torch

    inputs, labels, worker_id, worker_pid = batch
    input_ids = inputs["input"]
    positions = inputs["positions"]
    expected_shape = (mds.LOCAL_BATCH_SIZE, mds.SEQ_LEN)
    if tuple(input_ids.shape) != expected_shape or tuple(labels.shape) != expected_shape:
        raise AssertionError(f"rank={rank} batch={batch_index} input/label shape mismatch")
    if tuple(positions.shape) != expected_shape:
        raise AssertionError(f"rank={rank} batch={batch_index} positions shape mismatch")
    if input_ids.dtype is not torch.int64 or labels.dtype is not torch.int64:
        raise AssertionError(f"rank={rank} input/label tensors must remain int64")
    if positions.dtype is not torch.int32:
        raise AssertionError(f"rank={rank} positions tensor must remain int32")
    if worker_id != batch_index % mds.NUM_WORKERS:
        raise AssertionError(
            f"rank={rank} batch={batch_index} worker={worker_id}, expected {batch_index % mds.NUM_WORKERS}"
        )
    if not torch.equal(labels[:, :-1], input_ids[:, 1:]):
        raise AssertionError(f"rank={rank} batch={batch_index} causal label shift changed")
    if not bool(torch.all(labels[:, -1] == mds.IGNORE_INDEX)):
        raise AssertionError(f"rank={rank} batch={batch_index} last labels are not -100")
    if int((labels != mds.IGNORE_INDEX).sum().item()) != mds.LOCAL_BATCH_SIZE * (mds.SEQ_LEN - 1):
        raise AssertionError(f"rank={rank} batch={batch_index} valid-label count changed")

    row_hashes: list[str] = []
    observed_edges = {
        "eos_to_bos_pairs": 0,
        "eos_to_eos_pairs": 0,
        "rows_starting_with_eos": 0,
        "rows_starting_mid_document": 0,
    }
    expected_manifest = json.loads(mds.MANIFEST_PATH.read_text())
    expected_rows = expected_manifest["data_contract"]["row_sha256_by_sample_id"]
    sample_ids = []
    for row_index in range(mds.LOCAL_BATCH_SIZE):
        tokens = input_ids[row_index]
        digest = mds.row_hash(tokens)
        sample_id = 16 * (batch_index % mds.EPOCH_BATCHES) + rank + 8 * row_index
        if digest != expected_rows[sample_id]:
            raise AssertionError(
                f"rank={rank} batch={batch_index} row={row_index} differs from real-MDS sample {sample_id}"
            )
        row_hashes.append(digest)
        sample_ids.append(sample_id)
        token_array = tokens.numpy()
        if int(token_array.min()) < 0 or int(token_array.max()) >= tokenizer.get_vocab_size():
            raise AssertionError(f"rank={rank} batch={batch_index} token outside tokenizer vocabulary")
        if batch_index < mds.EPOCH_BATCHES:
            observed_edges = {
                key: observed_edges[key] + value
                for key, value in mds.row_edge_counts(
                    [token_array.astype("<u4", copy=False)],
                    eos_id=tokenizer.eos_id,
                    bos_id=tokenizer.bos_id,
                ).items()
            }
        if not np.array_equal(
            positions[row_index].numpy(), mds.independent_positions(token_array, tokenizer.eos_id)
        ):
            raise AssertionError(f"rank={rank} batch={batch_index} position reset changed")

    record = {
        "worker_id": int(worker_id),
        "worker_pid": int(worker_pid),
        "sample_ids": sample_ids,
        "input_row_hashes": row_hashes,
        "labels_sha256": tensor_hash(labels),
        "positions_sha256": tensor_hash(positions),
    }
    return record, observed_edges


def sequence_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    contents = [batch_content(item) for item in records]
    return {
        "batch_count": len(records),
        "input_rows_sha256": sha256_json([item["input_row_hashes"] for item in contents]),
        "labels_sha256": sha256_json([item["labels_sha256"] for item in contents]),
        "positions_sha256": sha256_json([item["positions_sha256"] for item in contents]),
        "sequence_sha256": sha256_json(contents),
    }


def wait_for_queue_ahead(loader: Any, timeout_seconds: float = 10.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    observed: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        prefetcher = loader._prefetch_iterator
        if prefetcher is None:
            time.sleep(0.01)
            continue
        with prefetcher._condition:
            ready = list(prefetcher._ready)
            thread_alive = prefetcher._thread.is_alive()
        committed = scalar_tree(loader._committed_state)
        ahead_cursor = scalar_tree(ready[-1][1]) if ready else None
        observed = {
            "ready_batches": len(ready),
            "thread_alive": thread_alive,
            "committed_cursor": committed,
            "last_ready_cursor": ahead_cursor,
            "queue_ahead_of_delivered_cursor": bool(ready) and ahead_cursor != committed,
        }
        if observed["queue_ahead_of_delivered_cursor"]:
            return observed
        time.sleep(0.01)
    raise TimeoutError(f"prefetch producer did not get ahead of delivery: {observed}")


def inspect_checkpoint(path: Path, expected_state: dict[str, Any]) -> dict[str, Any]:
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
    import torch

    reader = FileSystemReader(str(path))
    metadata = reader.read_metadata()
    keys = sorted(metadata.state_dict_metadata)
    required_prefix = "dataloader.dp_rank_0"
    if not any(key == required_prefix or key.startswith(required_prefix + ".") for key in keys):
        raise AssertionError(f"DCP metadata lacks persisted {required_prefix}: {keys}")
    if not any(key.startswith("dataloader.") for key in keys):
        raise AssertionError(f"DCP metadata contains no dataloader target: {keys}")
    if any(key.startswith(("model.", "optimizer.", "ema.", "gpu_buffer.")) for key in keys):
        raise AssertionError(f"loader-only checkpoint contains an out-of-scope key: {keys}")
    decoded_path = path.parent / f"{path.name}.decoded.pt"
    dcp_to_torch_save(path, decoded_path)
    decoded_file_sha256 = sha256_file(decoded_path)
    decoded = torch.load(decoded_path, map_location="cpu", weights_only=True)
    decoded_state = validate_decoded_dcp_state(decoded, expected_state)
    return {
        "state_dict_metadata_keys": keys,
        "metadata_types": {
            key: type(value).__name__ for key, value in sorted(metadata.state_dict_metadata.items())
        },
        "metadata_file_sha256": sha256_file(path / ".metadata"),
        "expected_scalar_state": scalar_tree(expected_state),
        "persisted_dcp_values": decoded_state,
        "persisted_dcp_scalar_paths": scalar_leaf_paths(decoded_state),
        "decoded_torch_save_path": str(decoded_path.resolve()),
        "decoded_torch_save_sha256": decoded_file_sha256,
        "actual_file_names": sorted(child.name for child in path.iterdir()),
    }


def dcp_save(loader: Any, data_world: Any, checkpoint_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch.distributed.checkpoint as dcp
    import torch.distributed as dist

    expected = state_payload(loader)
    dcp.save(
        {"dataloader": loader},
        checkpoint_id=str(checkpoint_path),
        process_group=data_world.process_group,
    )
    after = state_payload(loader)
    if after != expected:
        raise AssertionError("DCP save changed the delivered loader cursor")
    proof: list[Any] = [None]
    if data_world.rank == 0:
        try:
            proof[0] = {
                "status": "passed",
                "metadata": inspect_checkpoint(checkpoint_path, expected),
            }
        except BaseException as error:
            proof[0] = {
                "status": "failed",
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
    dist.broadcast_object_list(proof, src=0, group=data_world.process_group)
    if proof[0] is None or proof[0].get("status") != "passed":
        raise AssertionError(f"rank 0 DCP metadata inspection failed: {proof[0]}")
    return expected, proof[0]["metadata"]


def dcp_load(loader: Any, data_world: Any, checkpoint_path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    import torch.distributed.checkpoint as dcp

    dcp.load(
        {"dataloader": loader},
        checkpoint_id=str(checkpoint_path),
        process_group=data_world.process_group,
    )
    actual = state_payload(loader)
    if actual != expected:
        raise AssertionError(f"DCP restore target differs from serialized scalar state: {actual} != {expected}")
    return actual


def run_probe_child(evidence_dir: Path, namespace: str) -> None:
    import numpy as np
    import torch
    import torch.distributed as dist

    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        timeout=timedelta(seconds=DIST_TIMEOUT_SECONDS),
    )
    torch.set_num_threads(1)
    mds = load_g1_helper()
    result_path = evidence_dir / f"rank{dist.get_rank()}.json"
    loader = None
    try:
        runtime = runtime_identity()
        mds_inputs = validate_mds_inputs(mds)
        loader, tokenizer, data_world = make_loader(
            mds,
            prefetch_batches=0,
            persistent_workers=True,
            namespace=namespace,
        )
        if dist.get_world_size() != 1 or dist.get_backend(data_world.process_group) != "gloo":
            raise AssertionError("pipeline probe must be a one-owner Gloo world")
        iterator = iter(loader)
        inputs, labels, worker_id, worker_pid = next(iterator)
        input_ids = inputs["input"]
        positions = inputs["positions"]
        if tuple(input_ids.shape) != (2, 4096) or tuple(labels.shape) != (2, 4096):
            raise AssertionError("world1 probe returned unexpected input/label shapes")
        if input_ids.dtype is not torch.int64 or labels.dtype is not torch.int64:
            raise AssertionError("world1 probe changed int64 input/label dtypes")
        if positions.dtype is not torch.int32:
            raise AssertionError("world1 probe changed int32 position dtype")
        if not torch.equal(labels[:, :-1], input_ids[:, 1:]) or not bool(
            torch.all(labels[:, -1] == -100)
        ):
            raise AssertionError("world1 probe changed the causal target/ignore contract")
        for row_index in range(2):
            if not np.array_equal(
                positions[row_index].numpy(),
                mds.independent_positions(input_ids[row_index].numpy(), tokenizer.eos_id),
            ):
                raise AssertionError("world1 probe position IDs differ from actual token resets")
        expected_hashes = set(
            json.loads(mds.MANIFEST_PATH.read_text())["data_contract"]["row_sha256_by_sample_id"]
        )
        row_hashes = [mds.row_hash(input_ids[row_index]) for row_index in range(2)]
        if any(value not in expected_hashes for value in row_hashes):
            raise AssertionError("world1 probe tokens do not match the fixed real-MDS manifest")
        record = {
            "worker_id": int(worker_id),
            "worker_pid": int(worker_pid),
            "input_row_hashes": row_hashes,
            "labels_sha256": tensor_hash(labels),
            "positions_sha256": tensor_hash(positions),
        }
        if loader._prefetch_iterator is not None or loader.num_prefetch_batches != 0:
            raise AssertionError("default-zero loader unexpectedly allocated a producer")
        write_json(
            result_path,
            {
                "status": "passed",
                "rank": 0,
                "backend": "gloo",
                "world_size": 1,
                "runtime": runtime,
                "mds": mds_inputs,
                "batch": record,
                "default_zero_has_no_prefetch_iterator": True,
                "config": {
                    "local_batch_size": mds.LOCAL_BATCH_SIZE,
                    "sequence_length": mds.SEQ_LEN,
                    "num_workers": 2,
                    "persistent_workers": True,
                    "pin_memory": False,
                    "num_prefetch_batches": 0,
                },
            },
        )
    except BaseException as error:
        write_json(
            evidence_dir / f"failure_rank{dist.get_rank()}.json",
            {"error": repr(error), "traceback": traceback.format_exc()},
        )
        raise
    finally:
        try:
            if loader is not None:
                loader.close()
        finally:
            dist.destroy_process_group()


def run_study_child(
    evidence_dir: Path,
    checkpoint_root: Path,
    *,
    persistent_workers: bool,
    namespace_prefix: str,
) -> None:
    import torch
    import torch.distributed as dist

    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        timeout=timedelta(seconds=DIST_TIMEOUT_SECONDS),
    )
    torch.set_num_threads(1)
    mds = load_g1_helper()
    loaders: list[Any] = []
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    try:
        runtime = runtime_identity()
        mds_inputs = validate_mds_inputs(mds)
        if world_size != 8 or dist.get_backend(dist.group.WORLD) != "gloo":
            raise AssertionError(f"study requires eight explicit Gloo owners, got {world_size}")
        worker_mode = "persistent" if persistent_workers else "nonpersistent"
        manifest = json.loads(mds.MANIFEST_PATH.read_text())

        off_loader, tokenizer, off_world = make_loader(
            mds,
            prefetch_batches=0,
            persistent_workers=persistent_workers,
            namespace=f"{namespace_prefix}_{worker_mode}_off",
        )
        loaders.append(off_loader)
        if off_world.process_group is not dist.group.WORLD:
            raise AssertionError("OFF loader lacks the explicit initialized Gloo group")
        off_iterator = iter(off_loader)
        off_records: list[dict[str, Any]] = []
        off_edges = {
            "eos_to_bos_pairs": 0,
            "eos_to_eos_pairs": 0,
            "rows_starting_with_eos": 0,
            "rows_starting_mid_document": 0,
        }
        off_state_12: dict[str, Any] | None = None
        off_state_55: dict[str, Any] | None = None
        off_state_56: dict[str, Any] | None = None
        off_state_58: dict[str, Any] | None = None
        skew_pause_state: dict[str, Any] | None = None
        skew_pause_record: dict[str, Any] | None = None
        for batch_index in range(DATA_BATCHES):
            record, edges = validate_batch(
                mds,
                next(off_iterator),
                rank=rank,
                batch_index=batch_index,
                tokenizer=tokenizer,
            )
            off_records.append(record)
            if batch_index < mds.EPOCH_BATCHES:
                off_edges = {key: off_edges[key] + edges[key] for key in off_edges}
            if batch_index + 1 == 12:
                off_state_12 = state_payload(off_loader)
            if batch_index + 1 == 55:
                off_state_55 = state_payload(off_loader)
            if batch_index + 1 == mds.EPOCH_BATCHES:
                off_state_56 = state_payload(off_loader)
            if batch_index + 1 == 58:
                off_state_58 = state_payload(off_loader)
        if off_loader._prefetch_iterator is not None:
            raise AssertionError("default-zero path allocated a prefetch iterator")
        if (
            off_state_12 is None
            or off_state_55 is None
            or off_state_56 is None
            or off_state_58 is None
        ):
            raise AssertionError("offline control cursor snapshots are missing")
        if not persistent_workers:
            # Nonpersistent worker PIDs are expected to change on the finite epoch reset.
            pass
        off_loader.close()
        loaders.remove(off_loader)

        candidate, candidate_tokenizer, data_world = make_loader(
            mds,
            prefetch_batches=QUEUE_DEPTH,
            persistent_workers=persistent_workers,
            namespace=f"{namespace_prefix}_{worker_mode}_on",
        )
        loaders.append(candidate)
        if candidate_tokenizer.eos_id != tokenizer.eos_id:
            raise AssertionError("OFF and ON loaders resolved different tokenizers")
        iterator = iter(candidate)
        initial_records: list[dict[str, Any]] = []
        for batch_index in range(12):
            record, _ = validate_batch(
                mds,
                next(iterator),
                rank=rank,
                batch_index=batch_index,
                tokenizer=candidate_tokenizer,
            )
            initial_records.append(record)
        if [batch_content(item) for item in initial_records] != [
            batch_content(item) for item in off_records[:12]
        ]:
            raise AssertionError("OFF/ON initial batch, labels, positions, or order differ")
        queue_ahead = wait_for_queue_ahead(candidate)
        on_state_12 = state_payload(candidate)
        if on_state_12 != off_state_12:
            raise AssertionError("prefetch checkpoint cursor is not the delivered 12-batch cursor")
        assert_prefetch_cursor_ahead(on_state_12, queue_ahead["last_ready_cursor"])

        checkpoint_paths = {
            name: checkpoint_root / worker_mode / name
            for name in ("delivered12", "epoch_end56", "chained58")
        }
        checkpoint_proofs: dict[str, dict[str, Any]] = {}
        saved12, checkpoint_proofs["delivered12"] = dcp_save(
            candidate, data_world, checkpoint_paths["delivered12"]
        )
        if saved12 != on_state_12:
            raise AssertionError("DCP save target does not match the delivered cursor")

        # Move past the epoch boundary before restoring the older delivered cursor.
        candidate_records = list(initial_records)
        boundary_indices_consumed = False

        def consume_candidate_index(expected_index: int) -> dict[str, int]:
            started_ns = time.monotonic_ns()
            record, _ = validate_batch(
                mds,
                next(iterator),
                rank=rank,
                batch_index=expected_index,
                tokenizer=candidate_tokenizer,
            )
            ended_ns = time.monotonic_ns()
            candidate_records.append(record)
            if batch_content(record) != batch_content(off_records[expected_index]):
                raise AssertionError(f"OFF/ON mismatch at boundary batch {expected_index}")
            return {
                "batch_index": expected_index,
                "started_monotonic_ns": started_ns,
                "ended_monotonic_ns": ended_ns,
            }

        for batch_index in range(12, 58):
            if boundary_indices_consumed and batch_index in (55, 56):
                continue
            record, _ = validate_batch(
                mds,
                next(iterator),
                rank=rank,
                batch_index=batch_index,
                tokenizer=candidate_tokenizer,
            )
            candidate_records.append(record)
            if batch_content(record) != batch_content(off_records[batch_index]):
                raise AssertionError(f"OFF/ON mismatch before old-state restore at batch {batch_index}")
            if batch_index == 54:
                boundary_record: dict[str, Any] = {
                    "rank": rank,
                    "delivered_batches_before_pause": 55,
                    "process_group": "explicit Gloo data-world group for start broadcast",
                    "process_group_timeout_seconds": DIST_TIMEOUT_SECONDS,
                    "launcher_timeout_seconds": MAX_STUDY_SECONDS,
                }
                if rank == 0:
                    prefetcher = candidate._prefetch_iterator
                    if prefetcher is None:
                        raise AssertionError(
                            "prefetch iterator disappeared after 55 delivered batches"
                        )
                    queue_ahead_at_55 = wait_for_queue_ahead(candidate)
                    with prefetcher.paused() as (ready, finished):
                        boundary_record["pause_active_during_skew"] = bool(prefetcher._paused)
                        boundary_record["ready_items_held"] = len(ready)
                        boundary_record["source_finished"] = bool(finished)
                        skew_pause_state = {
                            "dp_rank_0": scalar_tree(candidate._committed_state)
                        }
                        if skew_pause_state != off_state_55:
                            raise AssertionError(
                                "rank 0 pause did not retain its 55-delivery cursor"
                            )
                        if not ready:
                            raise AssertionError(
                                "rank 0 pause did not retain the queued final old-epoch batch"
                            )
                        last_ready_cursor = scalar_tree(ready[-1][1])
                        if last_ready_cursor != off_state_56["dp_rank_0"]:
                            raise AssertionError(
                                "rank 0 did not retain the expected final old-epoch cursor"
                            )
                        boundary_record["committed_cursor_matches_off55"] = True
                        boundary_record["last_ready_cursor_matches_off56"] = True
                        boundary_record["queue_ahead_at_55"] = queue_ahead_at_55
                        boundary_record["rank0_pause_started_utc"] = datetime.now(
                            timezone.utc
                        ).isoformat()
                        pause_started_ns = time.monotonic_ns()
                        boundary_record["rank0_pause_started_monotonic_ns"] = pause_started_ns
                        start_signal: list[Any] = [
                            {"rank0_pause_started_monotonic_ns": pause_started_ns}
                        ]
                        dist.broadcast_object_list(
                            start_signal, src=0, group=data_world.process_group
                        )
                        boundary_record["sleep_started_utc"] = datetime.now(
                            timezone.utc
                        ).isoformat()
                        sleep_started = time.monotonic()
                        time.sleep(SKEW_SLEEP_SECONDS)
                        boundary_record["skew_sleep_seconds"] = time.monotonic() - sleep_started
                        boundary_record["sleep_ended_utc"] = datetime.now(
                            timezone.utc
                        ).isoformat()
                        pause_ended_ns = time.monotonic_ns()
                        boundary_record["rank0_pause_ended_monotonic_ns"] = pause_ended_ns
                    boundary_record["pause_released_utc"] = datetime.now(
                        timezone.utc
                    ).isoformat()
                    boundary_record["pause_released"] = not bool(prefetcher._paused)
                    rollover_advance = {
                        "rank": rank,
                        "old_epoch_last_batch": consume_candidate_index(55),
                        "new_epoch_first_batch": consume_candidate_index(56),
                    }
                else:
                    start_signal = [None]
                    dist.broadcast_object_list(
                        start_signal, src=0, group=data_world.process_group
                    )
                    if (
                        not start_signal
                        or not isinstance(start_signal[0], dict)
                        or "rank0_pause_started_monotonic_ns" not in start_signal[0]
                    ):
                        raise AssertionError("rank 0 did not publish its paused-boundary start")
                    pause_started_ns = int(
                        start_signal[0]["rank0_pause_started_monotonic_ns"]
                    )
                    boundary_record["rank0_pause_started_monotonic_ns"] = pause_started_ns
                    rollover_advance = {
                        "rank": rank,
                        "old_epoch_last_batch": consume_candidate_index(55),
                        "new_epoch_first_batch": consume_candidate_index(56),
                    }
                boundary_record["rollover_advance"] = rollover_advance
                skew_pause_record = boundary_record
                boundary_indices_consumed = True

        if [batch_content(item) for item in candidate_records] != [
            batch_content(item) for item in off_records[:58]
        ]:
            raise AssertionError("OFF/ON batch content differs through the first epoch wrap")
        off_wrap_summary = sequence_summary(off_records[:58])
        on_wrap_summary = sequence_summary(candidate_records)
        if off_wrap_summary != on_wrap_summary:
            raise AssertionError("OFF/ON content hashes differ through the first epoch wrap")
        stale_iterator = iterator
        restored12 = dcp_load(candidate, data_world, checkpoint_paths["delivered12"], saved12)
        try:
            next(stale_iterator)
        except StopIteration:
            stale_iterator_invalidated = True
        else:
            stale_iterator_invalidated = False
        if not stale_iterator_invalidated:
            raise AssertionError("restore left the superseded prefetch iterator usable")
        iterator = iter(candidate)
        for batch_index in range(12, mds.EPOCH_BATCHES):
            record, _ = validate_batch(
                mds,
                next(iterator),
                rank=rank,
                batch_index=batch_index,
                tokenizer=candidate_tokenizer,
            )
            if batch_content(record) != batch_content(off_records[batch_index]):
                raise AssertionError(f"same-instance old-state continuation differs at {batch_index}")
        end_state = state_payload(candidate)
        if end_state != off_state_56:
            raise AssertionError("same-instance replay did not reach the exact 56-batch epoch cursor")
        saved_end, checkpoint_proofs["epoch_end56"] = dcp_save(
            candidate, data_world, checkpoint_paths["epoch_end56"]
        )
        if saved_end != end_state:
            raise AssertionError("exact-epoch-end DCP save changed its serialized cursor")

        for batch_index in range(56, 58):
            record, _ = validate_batch(
                mds,
                next(iterator),
                rank=rank,
                batch_index=batch_index,
                tokenizer=candidate_tokenizer,
            )
            if batch_content(record) != batch_content(off_records[batch_index]):
                raise AssertionError(f"OFF/ON wrap continuation differs at batch {batch_index}")
        stale_epoch_end_iterator = iterator
        restored_end = dcp_load(candidate, data_world, checkpoint_paths["epoch_end56"], saved_end)
        try:
            next(stale_epoch_end_iterator)
        except StopIteration:
            epoch_end_iterator_invalidated = True
        else:
            epoch_end_iterator_invalidated = False
        if not epoch_end_iterator_invalidated:
            raise AssertionError("exact-end restore left the superseded prefetch iterator usable")
        iterator = iter(candidate)
        for batch_index in range(56, 58):
            record, _ = validate_batch(
                mds,
                next(iterator),
                rank=rank,
                batch_index=batch_index,
                tokenizer=candidate_tokenizer,
            )
            if batch_content(record) != batch_content(off_records[batch_index]):
                raise AssertionError(f"exact-end-epoch restore differs at batch {batch_index}")
        chain_state, checkpoint_proofs["chained58"] = dcp_save(
            candidate, data_world, checkpoint_paths["chained58"]
        )
        if chain_state != off_state_58:
            raise AssertionError("chained save does not reflect the current delivered cursor")

        for loader in loaders:
            loader.close()
        loaders.clear()

        fresh, fresh_tokenizer, fresh_world = make_loader(
            mds,
            prefetch_batches=QUEUE_DEPTH,
            persistent_workers=persistent_workers,
            namespace=f"{namespace_prefix}_{worker_mode}_fresh",
        )
        loaders.append(fresh)
        fresh_state = dcp_load(fresh, fresh_world, checkpoint_paths["chained58"], chain_state)
        fresh_iterator = iter(fresh)
        fresh_records = []
        for batch_index in range(58, 60):
            record, _ = validate_batch(
                mds,
                next(fresh_iterator),
                rank=rank,
                batch_index=batch_index,
                tokenizer=fresh_tokenizer,
            )
            if batch_content(record) != batch_content(off_records[batch_index]):
                raise AssertionError(f"fresh chained restore differs at batch {batch_index}")
            fresh_records.append(record)

        # Reconcile first-epoch coverage and row-boundary counts across all owners.
        local_ids = [sample for batch in off_records[: mds.EPOCH_BATCHES] for sample in batch["sample_ids"]]
        all_ids: list[Any] = [None] * world_size
        all_edges: list[Any] = [None] * world_size
        dist.all_gather_object(all_ids, local_ids, group=data_world.process_group)
        dist.all_gather_object(all_edges, off_edges, group=data_world.process_group)
        global_coverage: list[Any] = [None]
        if rank == 0:
            try:
                first_epoch_ids = validate_first_epoch_ids(all_ids, mds.NUM_ROWS)
                edge_totals = sum_owner_edge_counts(all_edges)
                expected_edges = manifest["data_contract"]["row_boundary_coverage"]
                if edge_totals != expected_edges:
                    raise AssertionError(
                        f"global MDS BOS/EOS boundary coverage changed: {edge_totals} != {expected_edges}"
                    )
                global_coverage[0] = {
                    "status": "passed",
                    "sample_count": len(first_epoch_ids),
                    "unique_sample_count": len(set(first_epoch_ids)),
                    "sample_ids_sha256": sha256_json(first_epoch_ids),
                    "row_boundary_counts": edge_totals,
                    "expected_row_boundary_counts": expected_edges,
                }
            except BaseException as error:
                global_coverage[0] = {
                    "status": "failed",
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                }
        dist.broadcast_object_list(global_coverage, src=0, group=data_world.process_group)
        if global_coverage[0] is None or global_coverage[0].get("status") != "passed":
            raise AssertionError(f"global real-MDS owner coverage failed: {global_coverage[0]}")

        if skew_pause_record is None:
            raise AssertionError("the skewed real Mosaic epoch boundary was not exercised")

        local_result = {
            "status": "passed",
            "rank": rank,
            "backend": dist.get_backend(data_world.process_group),
            "world_size": world_size,
            "data_world": {
                "rank": data_world.rank,
                "world_size": data_world.world_size,
                "local_rank": data_world.local_rank,
                "local_world_size": data_world.local_world_size,
                "explicit_group": data_world.process_group is not None,
                "shared_memory_namespace": data_world.shared_memory_namespace,
            },
            "runtime": runtime,
            "mds": mds_inputs,
            "config": {
                "local_batch_size": mds.LOCAL_BATCH_SIZE,
                "sequence_length": mds.SEQ_LEN,
                "num_workers": 2,
                "persistent_workers": persistent_workers,
                "pin_memory": False,
                "prefetch_factor": None,
                "prefetch_off": 0,
                "prefetch_on": QUEUE_DEPTH,
            },
            "delivered_batches_per_control": DATA_BATCHES,
            "off_sequence": sequence_summary(off_records),
            "on_initial_sequence": sequence_summary(initial_records),
            "off_sequence_through_wrap": off_wrap_summary,
            "on_sequence_through_wrap": on_wrap_summary,
            "off_on_content_hashes_equal_through_wrap": off_wrap_summary == on_wrap_summary,
            "initial_off_on_parity": [batch_content(item) for item in initial_records]
            == [batch_content(item) for item in off_records[:12]],
            "off_on_cursor_12_equal": off_state_12 == on_state_12,
            "off_on_exact_batch_parity_through_58": True,
            "all_epoch_sample_ids": len(local_ids),
            "first_epoch_boundary_coverage": off_edges,
            "global_first_epoch_coverage": global_coverage[0],
            "worker_counts": {
                str(worker): sum(item["worker_id"] == worker for item in off_records)
                for worker in range(2)
            },
            "worker_pids": {
                str(worker): sorted({item["worker_pid"] for item in off_records if item["worker_id"] == worker})
                for worker in range(2)
            },
            "queue_ahead_at_12_deliveries": queue_ahead,
            "rank0_boundary_pause": {
                "delivered_batches_before_pause": 55,
                "committed_cursor_matches_off55": rank != 0 or skew_pause_state == off_state_55,
                "details": skew_pause_record,
                "all_owner_deadlines_passed": False,
            },
            "delivered12_state": on_state_12,
            "same_instance_old_state_restored": restored12 == saved12,
            "stale_iterator_invalidated_after_old_cursor_restore": stale_iterator_invalidated,
            "exact_epoch_end56_state": end_state,
            "exact_end_state_restored": restored_end == saved_end,
            "stale_iterator_invalidated_after_exact_end_restore": epoch_end_iterator_invalidated,
            "chained58_state": chain_state,
            "fresh_chained_state_loaded": fresh_state == chain_state,
            "fresh_chained_sequence": sequence_summary(fresh_records),
            "dcp_checkpoints": checkpoint_proofs,
        }
        all_results: list[Any] = [None] * world_size
        dist.all_gather_object(all_results, local_result, group=data_world.process_group)
        skew_validation: list[Any] = [None]
        if rank == 0:
            try:
                if [item["rank"] for item in all_results] != list(range(world_size)):
                    raise AssertionError("all eight CPU owner summaries are not present")
                if not all(item["status"] == "passed" for item in all_results):
                    raise AssertionError("one or more CPU owner summaries failed")
                skew_validation[0] = {
                    "status": "passed",
                    "proof": validate_boundary_skew(all_results),
                }
            except BaseException as error:
                skew_validation[0] = {
                    "status": "failed",
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                }
        dist.broadcast_object_list(skew_validation, src=0, group=data_world.process_group)
        if skew_validation[0] is None or skew_validation[0].get("status") != "passed":
            raise AssertionError(f"eight-owner skew proof failed: {skew_validation[0]}")
        for item in all_results:
            item["rank0_boundary_pause"]["all_owner_deadlines_passed"] = True
        if rank == 0:
            receipt = {
                "status": "passed",
                "case": worker_mode,
                "backend": "gloo",
                "owner_count": world_size,
                "actual_data_world_group": "dist.group.WORLD",
                "ranks": all_results,
                "boundary_skew_proof": skew_validation[0]["proof"],
                "global_first_epoch_coverage": global_coverage[0],
                "global_first_epoch_sample_ids": global_coverage[0]["sample_count"],
                "loader_only_dcp": True,
                "dcp_metadata_keys_by_rank": {
                    str(item["rank"]): {
                        name: proof["state_dict_metadata_keys"]
                        for name, proof in item["dcp_checkpoints"].items()
                    }
                    for item in all_results
                },
            }
            write_json(evidence_dir / "result.json", receipt)
        dist.barrier(group=data_world.process_group)
    except BaseException as error:
        write_json(
            evidence_dir / f"failure_rank{rank}.json",
            {"error": repr(error), "traceback": traceback.format_exc()},
        )
        raise
    finally:
        close_errors = []
        active_exception = sys.exc_info()[0] is not None
        for loader in reversed(loaders):
            try:
                loader.close()
            except BaseException as error:
                close_errors.append(repr(error))
        if close_errors:
            write_json(
                evidence_dir / f"cleanup_failure_rank{rank}.json",
                {"errors": close_errors},
            )
        try:
            dist.destroy_process_group()
        finally:
            if close_errors and not active_exception:
                raise RuntimeError(f"loader cleanup failed: {close_errors}")


def worker_entrypoint(args: argparse.Namespace) -> None:
    evidence_dir = Path(args.evidence_dir).resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    if args.worker_mode == "probe":
        run_probe_child(evidence_dir, args.namespace)
    elif args.worker_mode == "study":
        run_study_child(
            evidence_dir,
            Path(args.checkpoint_root).resolve(),
            persistent_workers=args.persistent_workers == "true",
            namespace_prefix=args.namespace,
        )
    else:
        raise ValueError(f"unknown worker mode {args.worker_mode}")


def launcher_module() -> Any:
    return load_module(RUNNER, "v2_reviewed_campaign_runner")


def command_for_worker(
    *,
    world_size: int,
    mode: str,
    evidence_dir: Path,
    namespace: str,
    logs_dir: Path,
    persistent_workers: bool | None = None,
    checkpoint_root: Path | None = None,
) -> list[str]:
    command = [
        str(PYTHON),
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={world_size}",
        "--max_restarts=0",
        "--log_dir",
        str(logs_dir / "torchrun"),
        "--redirects=3",
        "--tee=3",
        str(Path(__file__).resolve()),
        "--worker-mode",
        mode,
        "--evidence-dir",
        str(evidence_dir),
        "--namespace",
        namespace,
    ]
    if persistent_workers is not None:
        command.extend(["--persistent-workers", str(persistent_workers).lower()])
    if checkpoint_root is not None:
        command.extend(["--checkpoint-root", str(checkpoint_root)])
    return command


def write_gate_receipts(path: Path, receipt: dict[str, Any]) -> None:
    write_json(path, receipt)
    compact = {
        "status": receipt["status"],
        "run_id": receipt["run_id"],
        "run_root": str(path.parent),
        "receipt_path": str(path),
        "receipt_sha256": sha256_file(path),
        "source_identity_before": receipt.get("source_identity_before"),
        "source_identity_after": receipt.get("source_identity_after"),
        "excluded_evidence_scope": receipt.get("excluded_evidence_scope", []),
        "checks": [
            {
                key: check.get(key)
                for key in (
                    "name",
                    "passed",
                    "timeout_seconds",
                    "launch_outcome",
                    "result_path",
                    "log_path",
                )
                if key in check
            }
            for check in receipt.get("checks", [])
        ],
        "remaining_scope": receipt.get("unresolved_gates", []),
    }
    write_json(TASK_DIR / "loader-gate.json", compact)


def write_failure_receipt(path: Path, receipt: dict[str, Any]) -> None:
    receipt["status"] = "failed"
    receipt.setdefault("source_identity_after", source_identity())
    receipt["source_identity_unchanged"] = (
        receipt.get("source_identity_after") == receipt.get("source_identity_before")
    )
    write_gate_receipts(path, receipt)


def orchestrate(reuse_world1_probe: Path | None = None) -> int:
    if not PYTHON.is_file():
        raise FileNotFoundError(f"authorized existing interpreter is missing: {PYTHON}")
    if not PREPARE_MDS.is_file() or not RUNNER.is_file():
        raise FileNotFoundError("the preserved historical MDS builder or launcher is missing")
    source_before = source_identity()
    if not GIFT_ROOT.is_dir():
        raise FileNotFoundError(f"current task GIFT source is missing: {GIFT_ROOT}")

    EVIDENCE_PARENT.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    run_root = EVIDENCE_PARENT / f"loader-check-{run_id}"
    run_root.mkdir(parents=True, exist_ok=False)
    helper_hash = sha256_file(Path(__file__))
    runner_hash = sha256_file(RUNNER)
    gift_files = {
        relative: sha256_file(GIFT_ROOT / relative)
        for relative in (
            "torchtitan/config/schema/datasets/streaming/core/dataloader.py",
            "torchtitan/datasets/streaming/core/dataloader.py",
            "torchtitan/datasets/streaming/core/prefetch.py",
            "torchtitan/datasets/streaming/text/dataloader.py",
            "torchtitan/components/dataloader.py",
            "torchtitan/components/validate.py",
            "torchtitan/trainer.py",
        )
    }
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONPATH": f"{GIFT_ROOT}:{CAMPAIGN_DIR}",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )

    receipt: dict[str, Any] = {
        "status": "running",
        "run_id": run_id,
        "task_root": str(TASK_ROOT),
        "task_run_directory": str(TASK_DIR),
        "historical_fixture_directory_read_only": str(CAMPAIGN_DIR),
        "source_integration_base": "3b6b8928ec4b15a529d07ee99e672f2fc75ef73a",
        "ported_v2_reference": "21c1891dda531a1222e6c350802f2765758794f9",
        "source_identity_before": source_before,
        "helper": {"path": str(Path(__file__).resolve()), "sha256": helper_hash},
        "reviewed_runner": {"path": str(RUNNER), "sha256": runner_hash},
        "task_tmpdir_policy": "Use the existing runner's short task/.tmp/run-xxxxxxxx directories, one per launcher, and remove each only after its exact process group stops.",
        "task_tmpdirs": [],
        "historical_mds_builder": {
            "path": str(PREPARE_MDS),
            "sha256": sha256_file(PREPARE_MDS),
        },
        "current_task_gift_file_hashes": gift_files,
        "authorized_scope": "CPU/Gloo real-MDS loader-only DCP; no GPU computation, native run, performance claim, installation, new allocation, or Git publication",
        "environment_overrides": {
            key: environment[key]
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "PYTHONPATH",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
        "finite_acceptance": [
            "a passed actual-MDS-record world-size-1 Gloo probe matching this fixture and source",
            "two sequential eight-owner cases compare prefetch 0 and 32 with two workers, persistent and nonpersistent, through 58 batches",
            "all 896 first-epoch real-MDS row IDs occur exactly once globally, with row-boundary counts summed across all owners",
            "rank 0 observes a producer-ahead cursor different from the delivered scalar cursor at batch 12",
            "same-instance DCP restore of a queue-ahead 12-batch delivered cursor after crossing an epoch wrap",
            "persisted DCP values are decoded on CPU and compared by complete scalar tree and keys with the saved cursor",
            "exact 56-batch epoch-end DCP save/load, stale iterator invalidation, and chained58 fresh-loader restore",
            "rank 0 holds PrefetchIterator.paused() after 55 deliveries with the final old-epoch item queued while peers start a real epoch-1 next() call during the recorded pause, all before process-group and launcher deadlines",
            "all tensor dtypes, label shifts, ignored final labels, positions, sample hashes, worker assignment, queue marker, loader-only DCP, and clean owner/worker/producer/process-group exit",
        ],
        "performance_review": {
            "scope": "CPU loader lifecycle behavior only; no TPS, latency, or speedup claim",
            "enabled_costs_expected": "one producer thread per loader/rank, queue condition push/pop, per-batch scalar Mosaic state snapshots, bounded ready batches, and DCP-only load coordination barriers",
            "memory_context": "depth 32 is at most 32 ready batches per owner before worker IPC, active read, and consumer references; actual RSS is not measured here",
            "default_zero": "source path should create no producer/queue; verified by runtime marker",
            "runtime_unaffected": "no model arithmetic, CUDA kernels, H2D transfers, optimizer, or training collective path is invoked",
            "pin_memory": "kept at canonical false; CUDA is hidden only from child processes",
        },
        "checks": [],
        "unresolved_gates": [
            "native CUDA/NCCL behavior and H2D interaction",
            "native numerical parity and performance/memory/RSS comparison",
        ],
        "excluded_evidence_scope": [
            "full-model checkpoint continuation, model/optimizer state, and trainer restart are outside this loader-only DCP gate"
        ],
    }
    write_gate_receipts(run_root / "receipt.json", receipt)

    runner = launcher_module()
    if reuse_world1_probe is not None:
        try:
            reused_probe = validate_reused_probe(reuse_world1_probe.resolve())
        except BaseException as error:
            receipt["checks"].append(
                {
                    "name": "world1_real_mds_record_probe_reused",
                    "passed": False,
                    "error": repr(error),
                }
            )
            write_failure_receipt(run_root / "receipt.json", receipt)
            return 1
        receipt["prior_accepted_run"] = {
            "run_root": str(reuse_world1_probe.resolve()),
            "reuse_provenance": reused_probe,
            "supersedes_claims": [
                "prior boundary skew did not show peers starting epoch-1 next() during rank-0 pause",
                "prior run did not persist paired 58-batch OFF/ON content hashes",
                "prior run removed loader-only DCP artifacts after inspection",
            ],
        }
        receipt["checks"].append(
            {
                "name": "world1_real_mds_record_probe_reused",
                "passed": True,
                **reused_probe,
            }
        )
        probe_passed = True
    else:
        try:
            probe_tmpdir, probe_tmp_capacity = runner._make_task_tmpdir()
        except BaseException as error:
            receipt["checks"].append(
                {"name": "world1_task_tmpdir_preflight", "passed": False, "error": repr(error)}
            )
            write_failure_receipt(run_root / "receipt.json", receipt)
            return 1
        base_env = environment.copy()
        base_env["TASK_LOCAL_TMPDIR"] = str(probe_tmpdir)
        probe_dir = run_root / "probe"
        probe_logs = probe_dir / "logs"
        probe_logs.mkdir(parents=True, exist_ok=False)
        probe_attempt = probe_dir / "attempt"
        probe_attempt.mkdir(parents=True, exist_ok=False)
        probe_namespace = f"titan_v2gloo_{run_id.replace('-', '_')}_probe"
        probe_command = command_for_worker(
            world_size=1,
            mode="probe",
            evidence_dir=probe_dir,
            namespace=probe_namespace,
            logs_dir=probe_logs,
        )
        probe_outcome = runner._launch(
            probe_command,
            base_env,
            {"logs": probe_logs, "attempt": probe_attempt},
            MAX_PROBE_SECONDS,
        )
        probe_tmp_cleanup: dict[str, Any] = {
            "path": str(probe_tmpdir),
            "capacity": probe_tmp_capacity,
            "removed": False,
            "preserved_because_process_group_survived": not probe_outcome[4],
        }
        if probe_outcome[4]:
            try:
                runner._cleanup_task_tmpdir(probe_tmpdir)
                probe_tmp_cleanup["removed"] = not probe_tmpdir.exists()
            except BaseException as error:
                probe_tmp_cleanup["cleanup_error"] = repr(error)
        receipt["task_tmpdirs"].append(probe_tmp_cleanup)
        probe_passed = (
            probe_outcome[0] == 0
            and not probe_outcome[1]
            and not probe_outcome[3]
            and probe_outcome[4]
            and probe_tmp_cleanup["removed"]
            and (probe_dir / "rank0.json").is_file()
        )
        receipt["checks"].append(
            {
                "name": "world1_real_mds_record_probe",
                "command": probe_command,
                "environment_overrides": receipt["environment_overrides"],
                "timeout_seconds": MAX_PROBE_SECONDS,
                "launch_outcome": {
                    "exit_code": probe_outcome[0],
                    "timed_out": probe_outcome[1],
                    "pid": probe_outcome[2],
                    "orphaned_process_group": probe_outcome[3],
                    "process_group_stopped": probe_outcome[4],
                },
                "result_path": str(probe_dir / "rank0.json"),
                "log_path": str(probe_logs / "launcher.log"),
                "task_tmpdir": probe_tmp_cleanup,
                "passed": probe_passed,
            }
        )
    if not probe_passed:
        write_failure_receipt(run_root / "receipt.json", receipt)
        return 1

    checkpoint_root = run_root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=False)
    for persistent in (True, False):
        case_name = "workers_persistent_true" if persistent else "workers_persistent_false"
        try:
            case_tmpdir, case_tmp_capacity = runner._make_task_tmpdir()
        except BaseException as error:
            receipt["checks"].append(
                {"name": f"{case_name}_task_tmpdir_preflight", "passed": False, "error": repr(error)}
            )
            write_failure_receipt(run_root / "receipt.json", receipt)
            return 1
        case_dir = run_root / case_name
        case_dir.mkdir(parents=True, exist_ok=False)
        logs_dir = case_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=False)
        attempt_dir = case_dir / "attempt"
        attempt_dir.mkdir(parents=True, exist_ok=False)
        case_env = environment.copy()
        case_env["TASK_LOCAL_TMPDIR"] = str(case_tmpdir)
        namespace = f"titan_v2gloo_{run_id.replace('-', '_')}_{'p' if persistent else 'np'}"
        command = command_for_worker(
            world_size=8,
            mode="study",
            evidence_dir=case_dir,
            namespace=namespace,
            logs_dir=logs_dir,
            persistent_workers=persistent,
            checkpoint_root=checkpoint_root,
        )
        outcome = runner._launch(
            command,
            case_env,
            {"logs": logs_dir, "attempt": attempt_dir},
            MAX_STUDY_SECONDS,
        )
        case_tmp_cleanup: dict[str, Any] = {
            "path": str(case_tmpdir),
            "capacity": case_tmp_capacity,
            "removed": False,
            "preserved_because_process_group_survived": not outcome[4],
        }
        if outcome[4]:
            try:
                runner._cleanup_task_tmpdir(case_tmpdir)
                case_tmp_cleanup["removed"] = not case_tmpdir.exists()
            except BaseException as error:
                case_tmp_cleanup["cleanup_error"] = repr(error)
        receipt["task_tmpdirs"].append(case_tmp_cleanup)
        child_result = case_dir / "result.json"
        child_receipt = json.loads(child_result.read_text()) if child_result.is_file() else {}
        passed = (
            outcome[0] == 0
            and not outcome[1]
            and not outcome[3]
            and outcome[4]
            and case_tmp_cleanup["removed"]
            and child_result.is_file()
            and child_receipt.get("status") == "passed"
            and not list(case_dir.glob("failure_rank*.json"))
            and not list(case_dir.glob("cleanup_failure_rank*.json"))
        )
        receipt["checks"].append(
            {
                "name": case_name,
                "command": command,
                "environment_overrides": receipt["environment_overrides"],
                "timeout_seconds": MAX_STUDY_SECONDS,
                "launch_outcome": {
                    "exit_code": outcome[0],
                    "timed_out": outcome[1],
                    "pid": outcome[2],
                    "orphaned_process_group": outcome[3],
                    "process_group_stopped": outcome[4],
                },
                "result_path": str(child_result),
                "rank_failure_paths": sorted(str(path) for path in case_dir.glob("failure_rank*.json")),
                "cleanup_failure_paths": sorted(str(path) for path in case_dir.glob("cleanup_failure_rank*.json")),
                "result_sha256": sha256_file(child_result) if child_result.is_file() else None,
                "summary": {
                    "owner_count": child_receipt.get("owner_count"),
                    "global_first_epoch_sample_ids": child_receipt.get("global_first_epoch_sample_ids"),
                    "loader_only_dcp": child_receipt.get("loader_only_dcp"),
                    "dcp_metadata_keys": child_receipt.get("dcp_metadata_keys_by_rank"),
                },
                "log_path": str(logs_dir / "launcher.log"),
                "task_tmpdir": case_tmp_cleanup,
                "passed": passed,
            }
        )
        if not passed:
            write_failure_receipt(run_root / "receipt.json", receipt)
            return 1

    # Retain the tiny loader-only DCP trees and rank-0 CPU-decoded torch saves.
    metadata_files: dict[str, Any] = {}
    for path in sorted(checkpoint_root.rglob(".metadata")):
        metadata_files[str(path.relative_to(run_root))] = sha256_file(path)
    dcp_data_files = {
        str(path.relative_to(run_root)): sha256_file(path)
        for path in sorted(checkpoint_root.rglob("*.distcp"))
    }
    decoded_files = {
        str(path.relative_to(run_root)): sha256_file(path)
        for path in sorted(checkpoint_root.rglob("*.decoded.pt"))
    }
    artifacts_retained = (
        len(metadata_files) == 6
        and len(dcp_data_files) == 48
        and len(decoded_files) == 6
    )
    receipt["dcp_artifacts_retained"] = {
        "metadata_sha256_by_path": metadata_files,
        "dcp_data_sha256_by_path": dcp_data_files,
        "decoded_torch_save_sha256_by_path": decoded_files,
        "loader_only_checkpoint_count": len(metadata_files),
        "dcp_rank_shard_count": len(dcp_data_files),
        "decoded_torch_save_count": len(decoded_files),
        "checkpoint_root": str(checkpoint_root),
    }
    receipt["checks"].append(
        {
            "name": "loader_only_dcp_artifacts_retained",
            "passed": artifacts_retained,
            "metadata_count": len(metadata_files),
            "dcp_data_file_count": len(dcp_data_files),
            "decoded_torch_save_count": len(decoded_files),
        }
    )
    if not artifacts_retained:
        write_failure_receipt(run_root / "receipt.json", receipt)
        return 1
    receipt["source_identity_after"] = source_identity()
    receipt["source_identity_unchanged"] = (
        receipt["source_identity_after"] == receipt["source_identity_before"]
    )
    receipt["checks"].append(
        {
            "name": "source_patch_identity_stable_during_gate",
            "passed": receipt["source_identity_unchanged"],
            "source_identity_before": receipt["source_identity_before"],
            "source_identity_after": receipt["source_identity_after"],
        }
    )
    tmpdirs_cleaned = all(
        item["removed"] and not Path(item["path"]).exists()
        for item in receipt["task_tmpdirs"]
    )
    receipt["checks"].append(
        {
            "name": "owned_task_tmpdirs_cleaned_after_process_exit",
            "passed": tmpdirs_cleaned,
            "paths": [item["path"] for item in receipt["task_tmpdirs"]],
        }
    )
    if not receipt["source_identity_unchanged"]:
        write_failure_receipt(run_root / "receipt.json", receipt)
        return 1
    if not tmpdirs_cleaned:
        write_failure_receipt(run_root / "receipt.json", receipt)
        return 1

    receipt["dcp_checkpoint_root_retained"] = checkpoint_root.is_dir()
    receipt["status"] = "passed" if all(item.get("passed") for item in receipt["checks"]) else "failed"
    write_gate_receipts(run_root / "receipt.json", receipt)
    return 0 if receipt["status"] == "passed" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-mode", choices=("probe", "study"))
    parser.add_argument("--evidence-dir")
    parser.add_argument("--namespace")
    parser.add_argument("--persistent-workers", choices=("true", "false"))
    parser.add_argument("--checkpoint-root")
    parser.add_argument("--reuse-world1-probe", type=Path)
    args = parser.parse_args()
    return args


def main() -> int:
    args = parse_args()
    if args.worker_mode is None:
        return orchestrate(args.reuse_world1_probe)
    if args.reuse_world1_probe is not None:
        raise ValueError("--reuse-world1-probe is only valid in controller mode")
    if not args.evidence_dir or not args.namespace:
        raise ValueError("worker mode requires --evidence-dir and --namespace")
    if args.worker_mode == "study" and (args.persistent_workers is None or not args.checkpoint_root):
        raise ValueError("study mode requires persistence mode and checkpoint root")
    worker_entrypoint(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
