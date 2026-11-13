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
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


CAMPAIGN_DIR = Path(__file__).resolve().parent
TASK_ROOT = CAMPAIGN_DIR.parents[1]
RAW_ROOT = CAMPAIGN_DIR / "artifacts" / "raw"
EVIDENCE_PARENT = RAW_ROOT / "v2-real-mds-gloo-dcp"
VARIANT_ROOT = RAW_ROOT / "source-variants" / "v2-prefetch"
VARIANT_GIFT = VARIANT_ROOT / "gift"
VARIANT_COMMIT = "21c1891dda531a1222e6c350802f2765758794f9"
MAIN_HEAD = "e9048f65f6a00b87c97b1e7e2cfb31fa66f56414"
PYTHON = Path("/opt/torchtitan-venv/bin/python3.12")
PREPARE_MDS = CAMPAIGN_DIR / "prepare_mds.py"
RUNNER = CAMPAIGN_DIR / "run.py"
EXPECTED_TORCH = "2.15.0.dev20260822+cu130"
EXPECTED_STREAMING = "0.13.0"
MAX_PROBE_SECONDS = 120
MAX_STUDY_SECONDS = 900
DIST_TIMEOUT_SECONDS = 120
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


def state_payload(loader: Any) -> dict[str, Any]:
    state = loader.state_dict()
    if set(state) != {"dp_rank_0"}:
        raise AssertionError(f"unexpected loader state keys: {sorted(state)}")
    scalar_tree(state)
    return copy.deepcopy(state)


def verify_variant_imports() -> dict[str, dict[str, str]]:
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
    root = VARIANT_GIFT.resolve()
    for name, module in modules.items():
        source = Path(module.__file__).resolve()
        if not source.is_relative_to(root):
            raise RuntimeError(f"{name} imported outside pinned variant: {source}")
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

    modules = verify_variant_imports()
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
        "mosaicml_streaming": streaming_version,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "variant_files": modules,
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
    return {
        "batch_count": len(records),
        "input_rows_sha256": sha256_json([item["input_row_hashes"] for item in records]),
        "labels_sha256": sha256_json([item["labels_sha256"] for item in records]),
        "positions_sha256": sha256_json([item["positions_sha256"] for item in records]),
        "sequence_sha256": sha256_json(records),
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
    return {
        "state_dict_metadata_keys": keys,
        "metadata_types": {
            key: type(value).__name__ for key, value in sorted(metadata.state_dict_metadata.items())
        },
        "metadata_file_sha256": sha256_file(path / ".metadata"),
        "expected_scalar_state": scalar_tree(expected_state),
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
        torch.set_num_threads(1)
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
    result_path = evidence_dir / f"rank{rank}.json"
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
        off_state_56: dict[str, Any] | None = None
        off_state_58: dict[str, Any] | None = None
        skew_pause_state: dict[str, Any] | None = None
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
            if batch_index + 1 == mds.EPOCH_BATCHES:
                off_state_56 = state_payload(off_loader)
            if batch_index + 1 == 58:
                off_state_58 = state_payload(off_loader)
        if off_loader._prefetch_iterator is not None:
            raise AssertionError("default-zero path allocated a prefetch iterator")
        if off_state_12 is None or off_state_56 is None or off_state_58 is None:
            raise AssertionError("offline control cursor snapshots are missing")
        if not persistent_workers:
            # Nonpersistent worker PIDs are expected to change on the finite epoch reset.
            pass

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
        if initial_records != off_records[:12]:
            raise AssertionError("OFF/ON initial batch, labels, positions, or order differ")
        queue_ahead = wait_for_queue_ahead(candidate)
        on_state_12 = state_payload(candidate)
        if on_state_12 != off_state_12:
            raise AssertionError("prefetch checkpoint cursor is not the delivered 12-batch cursor")
        if on_state_12["dp_rank_0"] == queue_ahead["last_ready_cursor"].get("dp_rank_0"):
            raise AssertionError("producer-ahead state was mistaken for the delivered cursor")

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
        for batch_index in range(12, 58):
            record, _ = validate_batch(
                mds,
                next(iterator),
                rank=rank,
                batch_index=batch_index,
                tokenizer=candidate_tokenizer,
            )
            if record != off_records[batch_index]:
                raise AssertionError(f"OFF/ON mismatch before old-state restore at batch {batch_index}")
            if batch_index == 55 and rank == 0:
                skew_pause_state = state_payload(candidate)
                if skew_pause_state != off_state_56:
                    raise AssertionError("boundary pause did not retain the delivered end-of-epoch cursor")
                time.sleep(0.5)
        restored12 = dcp_load(candidate, data_world, checkpoint_paths["delivered12"], saved12)
        iterator = iter(candidate)
        for batch_index in range(12, mds.EPOCH_BATCHES):
            record, _ = validate_batch(
                mds,
                next(iterator),
                rank=rank,
                batch_index=batch_index,
                tokenizer=candidate_tokenizer,
            )
            if record != off_records[batch_index]:
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
            if record != off_records[batch_index]:
                raise AssertionError(f"OFF/ON wrap continuation differs at batch {batch_index}")
        restored_end = dcp_load(candidate, data_world, checkpoint_paths["epoch_end56"], saved_end)
        iterator = iter(candidate)
        for batch_index in range(56, 58):
            record, _ = validate_batch(
                mds,
                next(iterator),
                rank=rank,
                batch_index=batch_index,
                tokenizer=candidate_tokenizer,
            )
            if record != off_records[batch_index]:
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
            if record != off_records[batch_index]:
                raise AssertionError(f"fresh chained restore differs at batch {batch_index}")
            fresh_records.append(record)

        # Reconcile complete first-epoch row coverage and retain compact hashes.
        if off_edges != manifest["data_contract"]["row_boundary_coverage"]:
            raise AssertionError(f"MDS BOS/EOS boundary coverage changed: {off_edges}")
        local_ids = [sample for batch in off_records[: mds.EPOCH_BATCHES] for sample in batch["sample_ids"]]
        all_ids: list[Any] = [None] * world_size
        dist.all_gather_object(all_ids, local_ids, group=data_world.process_group)
        if rank == 0 and sorted(sample for ids in all_ids for sample in ids) != list(range(mds.NUM_ROWS)):
            raise AssertionError("eight owners do not cover every MDS row exactly once per epoch")

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
            "initial_off_on_parity": initial_records == off_records[:12],
            "off_on_cursor_12_equal": off_state_12 == on_state_12,
            "off_on_exact_batch_parity_through_58": True,
            "all_epoch_sample_ids": len(local_ids),
            "first_epoch_boundary_coverage": off_edges,
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
                "batch_count": 56,
                "pause_seconds": 0.5 if rank == 0 else 0.0,
                "cursor_matches_off_control": rank != 0 or skew_pause_state == off_state_56,
                "all_owner_deadlines_passed": True,
            },
            "delivered12_state": on_state_12,
            "same_instance_old_state_restored": restored12 == saved12,
            "exact_epoch_end56_state": end_state,
            "exact_end_state_restored": restored_end == saved_end,
            "chained58_state": chain_state,
            "fresh_chained_state_loaded": fresh_state == chain_state,
            "fresh_chained_sequence": sequence_summary(fresh_records),
            "dcp_checkpoints": checkpoint_proofs,
        }
        all_results: list[Any] = [None] * world_size
        dist.all_gather_object(all_results, local_result, group=data_world.process_group)
        if rank == 0:
            if [item["rank"] for item in all_results] != list(range(world_size)):
                raise AssertionError("all eight CPU owner summaries are not present")
            if not all(item["status"] == "passed" for item in all_results):
                raise AssertionError("one or more CPU owner summaries failed")
            receipt = {
                "status": "passed",
                "case": worker_mode,
                "backend": "gloo",
                "owner_count": world_size,
                "actual_data_world_group": "dist.group.WORLD",
                "ranks": all_results,
                "global_first_epoch_sample_ids": mds.NUM_ROWS,
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


def write_failure_receipt(path: Path, receipt: dict[str, Any]) -> None:
    receipt["status"] = "failed"
    write_json(path, receipt)


def orchestrate() -> int:
    if not PYTHON.is_file():
        raise FileNotFoundError(f"authorized existing interpreter is missing: {PYTHON}")
    if git_output(TASK_ROOT, "rev-parse", "HEAD") != MAIN_HEAD:
        raise RuntimeError("frozen main task HEAD changed before V2 CPU study")
    if git_output(VARIANT_ROOT, "rev-parse", "HEAD") != VARIANT_COMMIT:
        raise RuntimeError("V2 candidate source pin changed before CPU study")
    if git_output(VARIANT_ROOT, "status", "--porcelain"):
        raise RuntimeError("V2 candidate source variant is not clean")

    EVIDENCE_PARENT.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    run_root = EVIDENCE_PARENT / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    helper_hash = sha256_file(Path(__file__))
    runner_hash = sha256_file(RUNNER)
    variant_files = {
        relative: sha256_file(VARIANT_GIFT / relative)
        for relative in (
            "torchtitan/config/schema/datasets/streaming/core/dataloader.py",
            "torchtitan/datasets/streaming/core/dataloader.py",
            "torchtitan/datasets/streaming/core/prefetch.py",
            "torchtitan/datasets/streaming/text/dataloader.py",
        )
    }
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONPATH": f"{VARIANT_GIFT}:{CAMPAIGN_DIR}",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )

    receipt: dict[str, Any] = {
        "status": "running",
        "run_id": run_id,
        "task_root": str(TASK_ROOT),
        "main_head": MAIN_HEAD,
        "variant_root": str(VARIANT_ROOT),
        "variant_branch": "vkropotin-v2-prefetch",
        "variant_commit": VARIANT_COMMIT,
        "variant_clean_before_run": True,
        "helper": {"path": str(Path(__file__).resolve()), "sha256": helper_hash},
        "reviewed_runner": {"path": str(RUNNER), "sha256": runner_hash},
        "variant_runtime_file_hashes": variant_files,
        "authorized_scope": "CPU/Gloo real-MDS loader-only DCP; no GPU, native run, performance claim, installation, job allocation, or Git publication",
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
            "one actual-MDS-record world-size-1 Gloo pipeline probe first",
            "eight actual data owners compare prefetch 0 and 32 for persistent and nonpersistent two-worker modes through 68 batches",
            "same-instance DCP restore of a queue-ahead 12-batch delivered cursor after crossing an epoch wrap",
            "exact 56-batch epoch-end DCP save/load and a fresh-loader chained restore",
            "all tensor dtypes, label shifts, ignored final labels, positions, sample hashes, actual queue marker, explicit Gloo group, and scalar DCP metadata/target",
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
            "full-model checkpoint continuation",
        ],
    }
    write_json(run_root / "receipt.json", receipt)

    runner = launcher_module()
    base_env = environment.copy()
    base_env["TASK_LOCAL_TMPDIR"] = str(run_root / "probe" / "tmp")
    (run_root / "probe" / "tmp").mkdir(parents=True, exist_ok=False)
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
    probe_passed = (
        probe_outcome[0] == 0
        and not probe_outcome[1]
        and probe_outcome[4]
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
        case_dir = run_root / case_name
        case_dir.mkdir(parents=True, exist_ok=False)
        (case_dir / "tmp").mkdir(parents=True, exist_ok=False)
        logs_dir = case_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=False)
        attempt_dir = case_dir / "attempt"
        attempt_dir.mkdir(parents=True, exist_ok=False)
        case_env = environment.copy()
        case_env["TASK_LOCAL_TMPDIR"] = str(case_dir / "tmp")
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
        child_result = case_dir / "result.json"
        child_receipt = json.loads(child_result.read_text()) if child_result.is_file() else {}
        passed = (
            outcome[0] == 0
            and not outcome[1]
            and not outcome[3]
            and outcome[4]
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
                "passed": passed,
            }
        )
        if not passed:
            write_failure_receipt(run_root / "receipt.json", receipt)
            return 1

    # Verify the saved metadata/target proofs, then remove only this task's tiny DCP files.
    checkpoint_files: dict[str, Any] = {}
    for path in sorted(checkpoint_root.rglob(".metadata")):
        checkpoint_files[str(path.relative_to(run_root))] = sha256_file(path)
    if len(checkpoint_files) != 6:
        receipt["checks"].append({"name": "dcp_artifact_cleanup", "passed": False, "reason": "no DCP metadata found"})
        write_failure_receipt(run_root / "receipt.json", receipt)
        return 1
    receipt["dcp_artifacts_before_cleanup"] = {
        "metadata_sha256_by_path": checkpoint_files,
        "loader_only_checkpoint_count": len(checkpoint_files),
    }
    receipt["status"] = "passed"
    write_json(run_root / "receipt.json", receipt)
    shutil.rmtree(checkpoint_root)
    receipt["checks"].append(
        {
            "name": "dcp_artifact_cleanup",
            "passed": not checkpoint_root.exists(),
            "removed_only_task_owned_loader_checkpoints": True,
        }
    )
    receipt["status"] = "passed" if all(item.get("passed") for item in receipt["checks"]) else "failed"
    receipt["final_main_head"] = git_output(TASK_ROOT, "rev-parse", "HEAD")
    receipt["final_variant_commit"] = git_output(VARIANT_ROOT, "rev-parse", "HEAD")
    receipt["final_variant_clean"] = not bool(git_output(VARIANT_ROOT, "status", "--porcelain"))
    if receipt["final_main_head"] != MAIN_HEAD or receipt["final_variant_commit"] != VARIANT_COMMIT or not receipt["final_variant_clean"]:
        receipt["status"] = "failed"
        receipt["source_freeze_invariant"] = False
    else:
        receipt["source_freeze_invariant"] = True
    write_json(run_root / "receipt.json", receipt)
    return 0 if receipt["status"] == "passed" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-mode", choices=("probe", "study"))
    parser.add_argument("--evidence-dir")
    parser.add_argument("--namespace")
    parser.add_argument("--persistent-workers", choices=("true", "false"))
    parser.add_argument("--checkpoint-root")
    args = parser.parse_args()
    return args


def main() -> int:
    args = parse_args()
    if args.worker_mode is None:
        return orchestrate()
    if not args.evidence_dir or not args.namespace:
        raise ValueError("worker mode requires --evidence-dir and --namespace")
    if args.worker_mode == "study" and (args.persistent_workers is None or not args.checkpoint_root):
        raise ValueError("study mode requires persistence mode and checkpoint root")
    worker_entrypoint(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
