#!/usr/bin/env python3
"""Prepare and verify the G1 full-length MDS data contract on CPU."""

from __future__ import annotations

import argparse
import collections
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import get_worker_info

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config.schema.datasets.streaming.text.configs import (
    StreamingTextDatasetConfig,
    TextStreamConfig,
)
from torchtitan.config.schema.datasets.streaming.text.dataloader import (
    StreamingTextDataLoaderConfig,
)
from torchtitan.datasets.streaming.core.dataloader import BaseStreamingDataLoader
from torchtitan.datasets.streaming.text.collators import TextPretrainCollator
from torchtitan.datasets.streaming.text.dataloader import StreamingTextDataLoader
from torchtitan.distributed.data_world import (
    DataWorld,
    get_dataloader_shared_memory_namespace,
)

import streaming
from streaming.base import StreamingDataLoader, StreamingDataset
from streaming.base.partition import get_partitions_for_worker
from streaming.base.partition.orig import get_partitions_orig


CAMPAIGN_DIR = Path(__file__).resolve().parent
ROOT = CAMPAIGN_DIR.parents[1]
GIFT_ROOT = ROOT / "gift"
RAW_DIR = CAMPAIGN_DIR / "artifacts" / "raw"
MDS_ROOT = RAW_DIR / "mds4096"
MDS_SPLIT = MDS_ROOT / "train"
MANIFEST_PATH = CAMPAIGN_DIR / "mds-data.json"
YAML_PATH = CAMPAIGN_DIR / "mds-dataloader.yaml"
SOURCE_PATH = GIFT_ROOT / "tests/assets/c4_test/data.json"
TOKENIZER_DIR = GIFT_ROOT / "tests/assets/tokenizer"
PACKER_PATH = GIFT_ROOT / "tests/assets/c4_test_mds/pack_test_dataset.py"

SEQ_LEN = 4096
NUM_ROWS = 896
SHARD_ROWS = 32
LOCAL_BATCH_SIZE = 2
WORLD_SIZE = 8
NUM_WORKERS = 2
EPOCH_BATCHES = 56
CHECK_BATCHES = 58
IGNORE_INDEX = -100
STREAMING_VERSION = "0.13.0"

EXPECTED_INPUT_HASHES = {
    "corpus": "07a274d2f6c903fd1c491260be302f99383930d31656f50de33ef467af6643b1",
    "tokenizer_json": "fe58d9f888054e03f0b202a00df9f6fc05559d8125fe188aa2f0ff5dee142b6c",
    "tokenizer_config": "1ccbfda3bb357c2706747d3a22bac82029baac0e4677c812058c7f4a763e8188",
}

FIM_RATES = {
    "vanilla_fim": 0.0,
    "inline_fim": 0.0,
    "multiline_fim": 0.0,
    "syntax_aware_fim": 0.0,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def relative_or_absolute(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def check_import_identity() -> dict[str, Any]:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"Python 3.12 required, got {platform.python_version()}")
    if Path(sys.prefix).resolve() != Path("/opt/torchtitan-venv").resolve():
        raise RuntimeError(f"Unexpected interpreter prefix: {sys.prefix}")

    installed_streaming_version = importlib.metadata.version("mosaicml-streaming")
    if installed_streaming_version != STREAMING_VERSION:
        raise RuntimeError(
            f"Expected installed mosaicml-streaming {STREAMING_VERSION}, "
            f"got {installed_streaming_version}"
        )

    streaming_path = Path(streaming.__file__).resolve()
    site_packages = Path(sys.prefix).resolve() / "lib/python3.12/site-packages"
    if not streaming_path.is_relative_to(site_packages):
        raise RuntimeError(
            f"Streaming must resolve to the installed package, got {streaming_path}"
        )
    bundled_streaming = GIFT_ROOT / "contrib/streaming"
    if streaming_path.is_relative_to(bundled_streaming.resolve()):
        raise RuntimeError(f"Bundled streaming import selected: {streaming_path}")

    from streaming.base import partition as partition_module
    from streaming.base import world as world_module
    from streaming.base.format.base import reader as reader_module
    from torchtitan.datasets.streaming.core import dataloader as core_loader_module
    from torchtitan.datasets.streaming.text import collators as collator_module
    from torchtitan.datasets.streaming.text import dataloader as text_loader_module
    from torchtitan.distributed import data_world as data_world_module

    objects = {
        "streaming_package": streaming,
        "streaming_dataset": StreamingDataset,
        "streaming_dataloader": StreamingDataLoader,
        "streaming_partition_dispatch": partition_module,
        "streaming_partition_orig": get_partitions_orig,
        "streaming_world": world_module,
        "streaming_mds_reader": reader_module,
        "titan_data_world": data_world_module,
        "titan_streaming_core_loader": core_loader_module,
        "titan_streaming_text_collator": collator_module,
        "titan_streaming_text_loader": text_loader_module,
        "tokenizer": HuggingFaceTokenizer,
    }
    files: dict[str, dict[str, str]] = {}
    for name, value in objects.items():
        source = inspect.getsourcefile(value)
        if source is None:
            raise RuntimeError(f"Cannot identify active source for {name}")
        source_path = Path(source).resolve()
        files[name] = {
            "path": relative_or_absolute(source_path),
            "sha256": sha256_file(source_path),
        }

    files["packer"] = {
        "path": relative_or_absolute(PACKER_PATH),
        "sha256": sha256_file(PACKER_PATH),
    }
    files["helper"] = {
        "path": relative_or_absolute(Path(__file__)),
        "sha256": sha256_file(Path(__file__)),
    }
    return {
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_prefix": str(Path(sys.prefix).resolve()),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "tokenizers": importlib.metadata.version("tokenizers"),
        "mosaicml_streaming": installed_streaming_version,
        "streaming_import": str(streaming_path),
        "titan_import": str(Path(inspect.getsourcefile(StreamingTextDataLoader)).resolve()),
        "active_source_files": files,
    }


def encode_rows(tokenizer: HuggingFaceTokenizer) -> tuple[list[np.ndarray], int]:
    rows: list[np.ndarray] = []
    token_buffer: list[int] = []
    docs_consumed = 0
    with SOURCE_PATH.open() as source:
        for line in source:
            docs_consumed += 1
            document = json.loads(line)["text"]
            token_buffer.extend(
                tokenizer.encode(document, add_bos=True, add_eos=True)
            )
            while len(token_buffer) >= SEQ_LEN and len(rows) < NUM_ROWS:
                rows.append(np.asarray(token_buffer[:SEQ_LEN], dtype="<u4"))
                del token_buffer[:SEQ_LEN]
            if len(rows) == NUM_ROWS:
                break
    if len(rows) != NUM_ROWS:
        raise ValueError(f"Encoded only {len(rows)} rows, expected {NUM_ROWS}")
    return rows, docs_consumed


def row_hash(row: np.ndarray | torch.Tensor) -> str:
    if isinstance(row, torch.Tensor):
        values = row.detach().cpu().numpy().astype("<u4", copy=False)
    else:
        values = np.asarray(row, dtype="<u4")
    return hashlib.sha256(values.tobytes()).hexdigest()


def row_edge_counts(
    rows: list[np.ndarray], *, eos_id: int, bos_id: int
) -> dict[str, int]:
    counts = {
        "eos_to_bos_pairs": 0,
        "eos_to_eos_pairs": 0,
        "rows_starting_with_eos": 0,
        "rows_starting_mid_document": 0,
    }
    for row in rows:
        counts["eos_to_bos_pairs"] += int(
            np.count_nonzero((row[:-1] == eos_id) & (row[1:] == bos_id))
        )
        counts["eos_to_eos_pairs"] += int(
            np.count_nonzero((row[:-1] == eos_id) & (row[1:] == eos_id))
        )
        counts["rows_starting_with_eos"] += int(row[0] == eos_id)
        counts["rows_starting_mid_document"] += int(row[0] not in (bos_id, eos_id))
    return counts


def make_partition_manifest() -> dict[str, Any]:
    partitions = get_partitions_orig(
        NUM_ROWS,
        num_canonical_nodes=1,
        num_physical_nodes=1,
        ranks_per_node=WORLD_SIZE,
        workers_per_rank=NUM_WORKERS,
        batch_size=LOCAL_BATCH_SIZE,
    )
    expected_shape = (1, WORLD_SIZE, NUM_WORKERS, 28, LOCAL_BATCH_SIZE)
    if partitions.shape != expected_shape:
        raise AssertionError(f"partition shape {partitions.shape} != {expected_shape}")

    partition_ids: dict[str, list[int]] = {}
    ids_by_rank_worker: dict[str, dict[str, Any]] = {}
    observed_global_ids: list[int] = []
    for rank in range(WORLD_SIZE):
        for worker in range(NUM_WORKERS):
            installed = partitions[0, rank, worker].reshape(-1)
            optimized = get_partitions_for_worker(
                "orig",
                NUM_ROWS,
                1,
                1,
                WORLD_SIZE,
                NUM_WORKERS,
                LOCAL_BATCH_SIZE,
                target_node=0,
                target_rank=rank,
                target_worker=worker,
            )
            if not np.array_equal(installed, optimized):
                raise AssertionError(
                    f"installed worker partition differs at rank={rank}, worker={worker}"
                )
            ids = [int(sample_id) for sample_id in installed]
            if len(ids) != 56 or any(sample_id < 0 for sample_id in ids):
                raise AssertionError(
                    f"rank={rank}, worker={worker} has unexpected IDs: {ids}"
                )
            ids_array = np.asarray(ids, dtype="<i8")
            partition_ids[f"{rank}:{worker}"] = ids
            ids_by_rank_worker[f"{rank}:{worker}"] = {
                "count": len(ids),
                "first_ids": ids[:6],
                "last_ids": ids[-6:],
                "sha256": hashlib.sha256(ids_array.tobytes()).hexdigest(),
            }
            observed_global_ids.extend(ids)

    expected_all_ids = list(range(NUM_ROWS))
    if sorted(observed_global_ids) != expected_all_ids:
        raise AssertionError("rank/worker partitions do not cover each row exactly once")

    for rank in range(WORLD_SIZE):
        for batch_idx in range(EPOCH_BATCHES):
            worker = batch_idx % NUM_WORKERS
            pair_offset = (batch_idx // NUM_WORKERS) * LOCAL_BATCH_SIZE
            expected_pair = [16 * batch_idx + rank, 16 * batch_idx + rank + 8]
            got_pair = partition_ids[f"{rank}:{worker}"][
                pair_offset : pair_offset + LOCAL_BATCH_SIZE
            ]
            if got_pair != expected_pair:
                raise AssertionError(
                    f"partition row IDs mismatch rank={rank}, batch={batch_idx}: "
                    f"{got_pair} != {expected_pair}"
                )

    return {
        "algorithm": "orig",
        "num_canonical_nodes": 1,
        "num_physical_nodes": 1,
        "ranks_per_node": WORLD_SIZE,
        "workers_per_rank": NUM_WORKERS,
        "batch_size": LOCAL_BATCH_SIZE,
        "shape": list(partitions.shape),
        "ids_by_rank_worker": ids_by_rank_worker,
        "all_worker_ids_sha256": hashlib.sha256(
            np.asarray(partitions, dtype="<i8").tobytes()
        ).hexdigest(),
        "epoch_batches_per_rank": EPOCH_BATCHES,
        "sample_ids_per_rank_per_epoch": 112,
        "expected_sample_ids_per_rank_batch": "[16 * (batch % 56) + rank, + 8]",
        "expected_worker_per_rank_batch": "batch % 2",
        "global_epoch_id_count": len(expected_all_ids),
        "global_epoch_ids_sha256": hashlib.sha256(
            np.asarray(expected_all_ids, dtype="<i8").tobytes()
        ).hexdigest(),
    }


def describe_mds_output() -> dict[str, Any]:
    index_path = MDS_SPLIT / "index.json"
    index = json.loads(index_path.read_text())
    shard_records = []
    for record in index["shards"]:
        basename = record["raw_data"]["basename"]
        shard_path = MDS_SPLIT / basename
        if not shard_path.is_file():
            raise AssertionError(f"MDS index references missing shard: {shard_path}")
        shard_records.append(
            {
                "name": basename,
                "samples": int(record["samples"]),
                "bytes": shard_path.stat().st_size,
                "sha256": sha256_file(shard_path),
            }
        )
    expected_files = {"index.json", *(record["name"] for record in shard_records)}
    actual_files = {path.name for path in MDS_SPLIT.iterdir()}
    if actual_files != expected_files:
        raise AssertionError(
            f"unexpected MDS output files: {sorted(actual_files ^ expected_files)}"
        )
    total_samples = sum(record["samples"] for record in shard_records)
    if total_samples != NUM_ROWS:
        raise AssertionError(f"MDS has {total_samples} rows, expected {NUM_ROWS}")
    return {
        "path": relative_or_absolute(MDS_ROOT),
        "split": "train",
        "index_sha256": sha256_file(index_path),
        "row_count": total_samples,
        "shard_count": len(shard_records),
        "shards": shard_records,
    }


def verify_file_identities(manifest: dict[str, Any]) -> None:
    actual_inputs = {
        "corpus": sha256_file(SOURCE_PATH),
        "tokenizer_json": sha256_file(TOKENIZER_DIR / "tokenizer.json"),
        "tokenizer_config": sha256_file(TOKENIZER_DIR / "tokenizer_config.json"),
    }
    if actual_inputs != manifest["input_hashes"]:
        raise AssertionError(f"input file hashes changed: {actual_inputs}")
    if actual_inputs != EXPECTED_INPUT_HASHES:
        raise AssertionError(f"fixture input hashes differ from investigation: {actual_inputs}")

    if check_import_identity() != manifest["runtime"]:
        raise AssertionError("interpreter, import path, or active source hash changed")

    current_mds = describe_mds_output()
    if current_mds != manifest["mds_output"]:
        raise AssertionError("MDS index or shard hashes changed after preparation")


def yaml_fragment() -> str:
    local_path = json.dumps(str(MDS_ROOT.resolve()))
    return f'''dataloader:
  type: streaming_text
  dataset:
    streams:
      - name: c4_perf896
        local: {local_path}
        split: train
    fim_rates:
      vanilla_fim: 0.0
      inline_fim: 0.0
      multiline_fim: 0.0
      syntax_aware_fim: 0.0
    enable_syntax_aware_fim: false
    shuffle: false
    num_canonical_nodes: 1
    epoch_size: null
    partition_algo: orig
    legacy: true
  drop_last: true
  num_workers: 2
  persistent_workers: true
  pin_memory: false
  prefetch_factor: null
  infinite: true
'''


def prepare() -> None:
    if not MDS_ROOT.resolve().is_relative_to(RAW_DIR.resolve()):
        raise AssertionError(f"MDS output is outside campaign raw: {MDS_ROOT}")
    if MDS_ROOT.exists():
        raise FileExistsError(
            f"Refusing to run the destructive packer on existing output: {MDS_ROOT}"
        )
    if MANIFEST_PATH.exists() or YAML_PATH.exists():
        raise FileExistsError("Refusing to overwrite an existing G1 receipt or YAML")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    input_hashes = {
        "corpus": sha256_file(SOURCE_PATH),
        "tokenizer_json": sha256_file(TOKENIZER_DIR / "tokenizer.json"),
        "tokenizer_config": sha256_file(TOKENIZER_DIR / "tokenizer_config.json"),
    }
    if input_hashes != EXPECTED_INPUT_HASHES:
        raise AssertionError(f"fixture input hashes differ from investigation: {input_hashes}")

    tokenizer = HuggingFaceTokenizer(tokenizer_path=str(TOKENIZER_DIR))
    if (tokenizer.bos_id, tokenizer.eos_id, tokenizer.get_vocab_size()) != (2002, 2003, 2009):
        raise AssertionError(
            f"unexpected tokenizer IDs/vocabulary: {tokenizer.bos_id}, "
            f"{tokenizer.eos_id}, {tokenizer.get_vocab_size()}"
        )
    rows, documents_consumed = encode_rows(tokenizer)
    hashes = [row_hash(row) for row in rows]
    multiplicities = collections.defaultdict(list)
    for sample_id, digest in enumerate(hashes):
        multiplicities[digest].append(sample_id)
    duplicate_groups = [
        {"row_sha256": digest, "sample_ids": sample_ids, "multiplicity": len(sample_ids)}
        for digest, sample_ids in sorted(multiplicities.items())
        if len(sample_ids) > 1
    ]
    edges = row_edge_counts(rows, eos_id=tokenizer.eos_id, bos_id=tokenizer.bos_id)
    if not edges["eos_to_bos_pairs"]:
        raise AssertionError("fixture lacks an in-row EOS-to-BOS boundary")
    if not edges["rows_starting_with_eos"]:
        raise AssertionError("fixture lacks a row starting with EOS")
    if not edges["rows_starting_mid_document"]:
        raise AssertionError("fixture lacks a mid-document row start")

    packer_spec = importlib.util.spec_from_file_location("g1_mds_packer", PACKER_PATH)
    if packer_spec is None or packer_spec.loader is None:
        raise ImportError(f"Cannot import existing packer: {PACKER_PATH}")
    packer = importlib.util.module_from_spec(packer_spec)
    packer_spec.loader.exec_module(packer)
    packer.pack_mds_dataset(
        SOURCE_PATH,
        MDS_SPLIT,
        tokenizer_path=TOKENIZER_DIR,
        seq_len=SEQ_LEN,
        num_rows=NUM_ROWS,
        shard_rows=SHARD_ROWS,
    )

    output = describe_mds_output()
    partition = make_partition_manifest()
    source_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    manifest = {
        "schema_version": 1,
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": source_revision,
        "scope": "G1 MDS data preparation and CPU loader contract; no training",
        "commands": {
            "prepare": (
                "source /etc/profile.d/wm.sh && "
                f"PYTHONPATH={GIFT_ROOT} /opt/torchtitan-venv/bin/python3.12 "
                "runs/overnight-streaming-perf/prepare_mds.py prepare"
            ),
            "checks": (
                "source /etc/profile.d/wm.sh && "
                f"PYTHONPATH={GIFT_ROOT} /opt/torchtitan-venv/bin/torchrun "
                f"--standalone --nnodes=1 --nproc_per_node={WORLD_SIZE} "
                "runs/overnight-streaming-perf/prepare_mds.py check "
                "--pass-name <pass1|pass2> > "
                "runs/overnight-streaming-perf/artifacts/raw/"
                "mds-data-check-<pass1|pass2>.log 2>&1"
            ),
        },
        "input_hashes": input_hashes,
        "inputs": {
            "corpus_path": relative_or_absolute(SOURCE_PATH),
            "corpus_documents": 2000,
            "documents_consumed_to_fill_rows": documents_consumed,
            "tokenizer_path": relative_or_absolute(TOKENIZER_DIR),
            "tokenizer_bos_id": tokenizer.bos_id,
            "tokenizer_eos_id": tokenizer.eos_id,
            "tokenizer_vocabulary_size": tokenizer.get_vocab_size(),
        },
        "data_contract": {
            "num_rows": NUM_ROWS,
            "seq_len": SEQ_LEN,
            "shard_rows_argument": SHARD_ROWS,
            "row_dtype": "uint32",
            "encoded_token_count": NUM_ROWS * SEQ_LEN,
            "token_min": min(int(row.min()) for row in rows),
            "token_max": max(int(row.max()) for row in rows),
            "row_sha256_by_sample_id": hashes,
            "duplicate_hash_multiplicities": duplicate_groups,
            "duplicate_hash_group_count": len(duplicate_groups),
            "row_boundary_coverage": edges,
        },
        "mds_output": output,
        "partition": partition,
        "loader_settings": {
            "type": "streaming_text",
            "data_world_size": WORLD_SIZE,
            "local_world_size": WORLD_SIZE,
            "replication": 1,
            "num_workers": NUM_WORKERS,
            "persistent_workers": True,
            "pin_memory": False,
            "prefetch_factor": None,
            "fim_rates": FIM_RATES,
            "enable_syntax_aware_fim": False,
            "shuffle": False,
            "infinite": True,
            "num_canonical_nodes": 1,
            "epoch_size": None,
            "partition_algo": "orig",
            "legacy": True,
            "drop_last": True,
            "local_batch_size": LOCAL_BATCH_SIZE,
            "token_field": "tokens",
        },
        "runtime": check_import_identity(),
        "checks": {
            "status": "prepared; 8-rank behavior pending",
            "passes": [],
            "failed_checks": [],
            "skipped_checks": [
                "GPU/NCCL training was outside this deliverable; no training launched"
            ],
        },
    }
    write_json(MANIFEST_PATH, manifest)
    print(
        json.dumps(
            {
                "status": "prepared",
                "manifest": relative_or_absolute(MANIFEST_PATH),
                "MDS_rows": output["row_count"],
                "MDS_shards": output["shard_count"],
                "shard_samples": [shard["samples"] for shard in output["shards"]],
                "duplicate_hash_groups": len(duplicate_groups),
                "source_documents_consumed": documents_consumed,
            },
            sort_keys=True,
        )
    )


class WorkerMetadataCollator:
    """Call the production collator and append CPU worker identity for this check."""

    def __init__(self, delegate: TextPretrainCollator) -> None:
        self.delegate = delegate

    def __call__(self, samples: list[dict[str, torch.Tensor]]):
        batch = self.delegate(samples)
        worker = get_worker_info()
        if worker is None:
            raise RuntimeError("data check expected collation inside a DataLoader worker")
        return batch[0], batch[1], worker.id, os.getpid()


def build_loader() -> tuple[StreamingTextDataLoader, HuggingFaceTokenizer, DataWorld]:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    if (world_size, local_world_size) != (WORLD_SIZE, WORLD_SIZE):
        raise RuntimeError(
            f"expected one-node 8-rank run, got world={world_size}, "
            f"local_world={local_world_size}"
        )
    data_world = DataWorld(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        local_world_size=local_world_size,
        replication=1,
        shared_memory_namespace=get_dataloader_shared_memory_namespace(
            "g1_mds_contract"
        ),
        process_group=dist.group.WORLD,
    )
    tokenizer = HuggingFaceTokenizer(tokenizer_path=str(TOKENIZER_DIR))
    dataset = StreamingTextDatasetConfig(
        streams=[
            TextStreamConfig(
                name="c4_perf896",
                local=str(MDS_ROOT.resolve()),
                split="train",
            )
        ],
        fim_rates=dict(FIM_RATES),
        enable_syntax_aware_fim=False,
        shuffle=False,
        num_canonical_nodes=1,
        epoch_size=None,
        partition_algo="orig",
        legacy=True,
    )
    config = StreamingTextDataLoaderConfig(
        dataset=dataset,
        drop_last=True,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        pin_memory=False,
        prefetch_factor=None,
        infinite=True,
    )
    loader = StreamingTextDataLoader(
        config,
        data_world=data_world,
        tokenizer=tokenizer,
        seq_len=SEQ_LEN,
        local_batch_size=LOCAL_BATCH_SIZE,
    )
    loader._loader.collate_fn = WorkerMetadataCollator(loader._loader.collate_fn)
    return loader, tokenizer, data_world


def independent_positions(tokens: np.ndarray, eos_id: int) -> np.ndarray:
    expected = np.empty(SEQ_LEN, dtype=np.int32)
    last_start = 0
    for column in range(SEQ_LEN):
        if column == 0 or int(tokens[column - 1]) == eos_id:
            last_start = column
        expected[column] = column - last_start
    return expected


def check_pass(pass_name: str) -> None:
    if pass_name not in {"pass1", "pass2"}:
        raise ValueError(f"pass-name must be pass1 or pass2, got {pass_name}")
    if not dist.is_available() or not dist.is_gloo_available():
        raise RuntimeError("Gloo process group is unavailable")
    if not torch.distributed.is_initialized():
        dist.init_process_group(backend="gloo", init_method="env://")
    torch.set_num_threads(1)
    rank = dist.get_rank()
    if int(os.environ.get("WORLD_SIZE", "0")) != WORLD_SIZE:
        raise RuntimeError(f"torchrun WORLD_SIZE must be {WORLD_SIZE}")

    manifest = json.loads(MANIFEST_PATH.read_text())
    verify_file_identities(manifest)
    row_hashes = manifest["data_contract"]["row_sha256_by_sample_id"]
    if len(row_hashes) != NUM_ROWS:
        raise AssertionError(f"manifest has {len(row_hashes)} row hashes")
    if manifest["loader_settings"]["persistent_workers"] is not True:
        raise AssertionError("persistent_workers must be enabled for this contract")

    loader, tokenizer, data_world = build_loader()
    if loader.data_world != data_world:
        raise AssertionError("loader did not retain the configured DataWorld")

    row_digest = hashlib.sha256()
    worker_counts: collections.Counter[int] = collections.Counter()
    worker_pids: dict[int, int] = {}
    worker_sequence: list[int] = []
    batch_hashes: list[list[str]] = []
    feature_counts = {
        "eos_to_bos_pairs": 0,
        "eos_to_eos_pairs": 0,
        "rows_starting_with_eos": 0,
        "rows_starting_mid_document": 0,
    }
    observed_token_min = sys.maxsize
    observed_token_max = -1
    iterator = iter(loader)
    for batch_idx in range(CHECK_BATCHES):
        inputs, labels, worker_id, worker_pid = next(iterator)
        expected_worker = batch_idx % NUM_WORKERS
        if worker_id != expected_worker:
            raise AssertionError(
                f"rank={rank} batch={batch_idx} came from worker={worker_id}, "
                f"expected {expected_worker}"
            )
        if worker_id in worker_pids and worker_pids[worker_id] != worker_pid:
            raise AssertionError(
                f"rank={rank} worker {worker_id} changed PID across epoch wrap: "
                f"{worker_pids[worker_id]} -> {worker_pid}"
            )
        worker_pids[worker_id] = worker_pid
        worker_counts[worker_id] += 1
        worker_sequence.append(worker_id)

        input_ids = inputs["input"]
        positions = inputs["positions"]
        expected_shape = (LOCAL_BATCH_SIZE, SEQ_LEN)
        if tuple(input_ids.shape) != expected_shape or tuple(labels.shape) != expected_shape:
            raise AssertionError(
                f"rank={rank} batch={batch_idx} input/label shapes "
                f"{tuple(input_ids.shape)}, {tuple(labels.shape)}"
            )
        if tuple(positions.shape) != expected_shape:
            raise AssertionError(
                f"rank={rank} batch={batch_idx} positions shape {tuple(positions.shape)}"
            )
        if input_ids.dtype != torch.int64 or labels.dtype != torch.int64:
            raise AssertionError(
                f"rank={rank} input/label dtypes {input_ids.dtype}, {labels.dtype}"
            )
        if positions.dtype != torch.int32:
            raise AssertionError(f"rank={rank} positions dtype {positions.dtype}")
        if not torch.equal(labels[:, :-1], input_ids[:, 1:]):
            raise AssertionError(f"rank={rank} batch={batch_idx} label shift mismatch")
        if not bool(torch.all(labels[:, -1] == IGNORE_INDEX)):
            raise AssertionError(f"rank={rank} batch={batch_idx} final labels not ignored")
        valid_labels = int((labels != IGNORE_INDEX).sum().item())
        if valid_labels != LOCAL_BATCH_SIZE * (SEQ_LEN - 1):
            raise AssertionError(
                f"rank={rank} batch={batch_idx} has {valid_labels} valid labels"
            )

        batch_hash_list = []
        for row_idx in range(LOCAL_BATCH_SIZE):
            tokens = input_ids[row_idx].numpy()
            row_min = int(tokens.min())
            row_max = int(tokens.max())
            observed_token_min = min(observed_token_min, row_min)
            observed_token_max = max(observed_token_max, row_max)
            if row_min < 0 or row_max >= tokenizer.get_vocab_size():
                raise AssertionError(
                    f"rank={rank} batch={batch_idx} token range [{row_min}, {row_max}] "
                    f"outside vocabulary {tokenizer.get_vocab_size()}"
                )
            digest = row_hash(input_ids[row_idx])
            batch_hash_list.append(digest)
            row_digest.update(bytes.fromhex(digest))

            sample_id = 16 * (batch_idx % EPOCH_BATCHES) + rank + 8 * row_idx
            expected_digest = row_hashes[sample_id]
            if digest != expected_digest:
                raise AssertionError(
                    f"rank={rank} batch={batch_idx} row={row_idx} hash mismatch "
                    f"for expected sample ID {sample_id}"
                )
            row_features = row_edge_counts(
                [tokens.astype("<u4", copy=False)],
                eos_id=tokenizer.eos_id,
                bos_id=tokenizer.bos_id,
            )
            if batch_idx < EPOCH_BATCHES:
                for feature, count in row_features.items():
                    feature_counts[feature] += count
            expected_pos = independent_positions(tokens, tokenizer.eos_id)
            if not np.array_equal(positions[row_idx].numpy(), expected_pos):
                raise AssertionError(
                    f"rank={rank} batch={batch_idx} row={row_idx} position reset mismatch"
                )
        batch_hashes.append(batch_hash_list)
        if batch_idx >= EPOCH_BATCHES and batch_hash_list != batch_hashes[batch_idx - EPOCH_BATCHES]:
            raise AssertionError(
                f"rank={rank} replay after batch {EPOCH_BATCHES} differs at batch {batch_idx}"
            )

    if worker_counts != {0: 29, 1: 29}:
        raise AssertionError(f"rank={rank} worker batch counts: {dict(worker_counts)}")
    if set(worker_pids) != {0, 1} or len(set(worker_pids.values())) != NUM_WORKERS:
        raise AssertionError(f"rank={rank} workers did not have distinct stable PIDs")
    if observed_token_min != manifest["data_contract"]["token_min"]:
        raise AssertionError(f"rank={rank} observed token minimum differs from fixture")
    if observed_token_max != manifest["data_contract"]["token_max"]:
        raise AssertionError(f"rank={rank} observed token maximum differs from fixture")

    sample_ids = [
        16 * batch_idx + rank + 8 * row_idx
        for batch_idx in range(EPOCH_BATCHES)
        for row_idx in range(LOCAL_BATCH_SIZE)
    ]
    summary = {
        "rank": rank,
        "batches_checked": CHECK_BATCHES,
        "worker_counts": {str(worker): count for worker, count in sorted(worker_counts.items())},
        "worker_pids_stable_across_wrap": True,
        "worker_pids": {str(worker): pid for worker, pid in sorted(worker_pids.items())},
        "worker_sequence_sha256": sha256_json(worker_sequence),
        "row_hash_sequence_sha256": row_digest.hexdigest(),
        "first_epoch_sample_ids_sha256": sha256_json(sample_ids),
        "row_boundary_coverage": feature_counts,
        "observed_token_range": [observed_token_min, observed_token_max],
        "wrap_replay_verified": True,
    }

    gathered: list[dict[str, Any] | None] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered, summary)
    if rank == 0:
        all_summaries = [item for item in gathered if item is not None]
        if [item["rank"] for item in all_summaries] != list(range(WORLD_SIZE)):
            raise AssertionError("all-rank result set is incomplete or out of order")
        if any(item["batches_checked"] != CHECK_BATCHES for item in all_summaries):
            raise AssertionError("not every rank checked 58 batches")
        worker_pids_all = [
            int(pid)
            for item in all_summaries
            for pid in item["worker_pids"].values()
        ]
        if len(set(worker_pids_all)) != WORLD_SIZE * NUM_WORKERS:
            raise AssertionError("worker process IDs are not unique across ranks")

        combined_edges = collections.Counter()
        for item in all_summaries:
            combined_edges.update(item["row_boundary_coverage"])
            if not item["wrap_replay_verified"]:
                raise AssertionError(f"rank {item['rank']} did not verify epoch wrap replay")
        expected_edges = manifest["data_contract"]["row_boundary_coverage"]
        if dict(combined_edges) != expected_edges:
            raise AssertionError(
                f"observed row-boundary coverage {dict(combined_edges)} != {expected_edges}"
            )

        all_expected_ids = [
            16 * batch_idx + rank_id + 8 * row_idx
            for rank_id in range(WORLD_SIZE)
            for batch_idx in range(EPOCH_BATCHES)
            for row_idx in range(LOCAL_BATCH_SIZE)
        ]
        if sorted(all_expected_ids) != list(range(NUM_ROWS)):
            raise AssertionError("observed expected-ID formula misses or repeats sample IDs")

        receipt = json.loads(MANIFEST_PATH.read_text())
        completed_names = {result["pass_name"] for result in receipt["checks"]["passes"]}
        if pass_name in completed_names:
            raise FileExistsError(f"receipt already contains {pass_name}")
        previous_pass = next(
            (result for result in receipt["checks"]["passes"] if result["pass_name"] == "pass1"),
            None,
        )
        if pass_name == "pass2":
            if previous_pass is None:
                raise AssertionError("pass2 cannot be recorded before pass1")
            for first, second in zip(previous_pass["ranks"], all_summaries, strict=True):
                if first["row_hash_sequence_sha256"] != second["row_hash_sequence_sha256"]:
                    raise AssertionError(f"pass replay changed rank {second['rank']} row order")
                if first["worker_sequence_sha256"] != second["worker_sequence_sha256"]:
                    raise AssertionError(f"pass replay changed rank {second['rank']} worker order")

        tempdir = os.environ.get("TMPDIR")
        tempdir_prefix = f"TMPDIR={tempdir} " if tempdir else ""
        check_command = (
            "source /etc/profile.d/wm.sh && "
            f"{tempdir_prefix}PYTHONPATH={GIFT_ROOT} "
            "/opt/torchtitan-venv/bin/torchrun "
            f"--standalone --nnodes=1 --nproc_per_node={WORLD_SIZE} "
            "runs/overnight-streaming-perf/prepare_mds.py "
            f"check --pass-name {pass_name} > "
            f"runs/overnight-streaming-perf/artifacts/raw/"
            f"mds-data-check-{pass_name}.log 2>&1"
        )
        pass_result = {
            "pass_name": pass_name,
            "status": "passed",
            "command": check_command,
            "stdout_stderr_log": relative_or_absolute(
                RAW_DIR / f"mds-data-check-{pass_name}.log"
            ),
            "launcher_run_id": os.environ.get("TORCHELASTIC_RUN_ID"),
            "effective_environment": {
                "TMPDIR": tempdir,
                "PYTHONPATH": os.environ.get("PYTHONPATH"),
                "LOCAL_WORLD_SIZE": os.environ.get("LOCAL_WORLD_SIZE"),
            },
            "backend": "gloo",
            "world_size": WORLD_SIZE,
            "batches_checked_per_rank": CHECK_BATCHES,
            "epoch_batches_per_rank": EPOCH_BATCHES,
            "all_rank_results": True,
            "global_epoch_sample_id_union": NUM_ROWS,
            "global_first_epoch_valid_labels": NUM_ROWS * (SEQ_LEN - 1),
            "ranks": all_summaries,
        }
        receipt["checks"]["passes"].append(pass_result)
        if {result["pass_name"] for result in receipt["checks"]["passes"]} == {
            "pass1",
            "pass2",
        }:
            yaml_text = yaml_fragment()
            YAML_PATH.write_text(yaml_text)
            receipt["yaml_fragment"] = {
                "path": relative_or_absolute(YAML_PATH),
                "sha256": sha256_file(YAML_PATH),
                "canonical_config_promoted": False,
            }
            receipt["checks"]["status"] = "passed"
        else:
            receipt["checks"]["status"] = "one fresh-process pass passed; second pending"
        receipt["checks"]["failed_checks"] = []
        write_json(MANIFEST_PATH, receipt)
        print(
            json.dumps(
                {
                    "status": "passed",
                    "pass_name": pass_name,
                    "rank_count": WORLD_SIZE,
                    "batches_per_rank": CHECK_BATCHES,
                    "rows": NUM_ROWS,
                    "worker_pids": worker_pids_all,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    del iterator
    del loader
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare", help="pack the new 896-row MDS dataset")
    check_parser = subparsers.add_parser("check", help="check the 8-rank loader contract")
    check_parser.add_argument("--pass-name", required=True, choices=("pass1", "pass2"))
    args = parser.parse_args()
    if args.command == "prepare":
        if dist.is_initialized():
            raise RuntimeError("prepare must run as one process, not under torchrun")
        check_import_identity()
        prepare()
    elif args.command == "check":
        check_pass(args.pass_name)


if __name__ == "__main__":
    main()
