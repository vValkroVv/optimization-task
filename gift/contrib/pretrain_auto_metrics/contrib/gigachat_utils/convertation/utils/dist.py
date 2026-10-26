import os, re
import shutil
import time
from pathlib import Path

import torch
from collections import OrderedDict
from typing import Dict, List, Tuple

from logger import logger


def _get_rank_world(args) -> tuple[int, int]:
    rank = getattr(args, "global_rank", None)
    world = getattr(args, "world_size", None)
    if rank is None:
        rank = int(os.environ.get("RANK", "0"))
    if world is None:
        world = int(os.environ.get("WORLD_SIZE", "1"))
    return int(rank), int(world)


def clear_barrier_state(
    sync_dir: Path,
    rank: int,
    *,
    save_llm_path: Path | None = None,
    poll_s: float = 2.0,
) -> None:
    """Rank 0 clears sync (and optional LLM) dirs, then signals setup.done; others wait."""
    if rank == 0:
        # Clear previous sync markers so a re-run cannot see stale done files.
        if sync_dir.exists():
            shutil.rmtree(sync_dir, ignore_errors=True)
        sync_dir.mkdir(parents=True, exist_ok=True)
        (sync_dir / "setup.done").touch()
    else:
        while not (sync_dir / "setup.done").exists():
            time.sleep(poll_s)


def _file_barrier(sync_dir: Path, rank: int, world: int, *, tag: str = "llm_stream", poll_s: float = 2.0) -> None:
    """Shared-FS barrier: each rank touches a done file; all ranks wait for every rank."""
    sync_dir.mkdir(parents=True, exist_ok=True)
    done_path = sync_dir / f"{tag}.rank{rank:02d}.done"
    done_path.touch()
    if world <= 1:
        return
    expected = [sync_dir / f"{tag}.rank{r:02d}.done" for r in range(world)]
    while not all(p.exists() for p in expected):
        time.sleep(poll_s)
    if rank == 0:
        logger.info(f"[rank 0] file barrier '{tag}' passed ({world} ranks)")


_LAYER_ID_RE = re.compile(r"\blayers\.(\d+)\b")
_COMPONENT_GROUPS = ("language_model", "encoders", "adapters")

def _layer_id_from_raw(key: str) -> int | None:
    m = _LAYER_ID_RE.search(key)
    return int(m.group(1)) if m else None

def _component_group(key: str) -> str:
    for group in _COMPONENT_GROUPS:
        if group in key:
            return group
    return "language_model"

def _split_keys_by_component(keys: List[str]) -> "OrderedDict[str, List[str]]":
    groups: "OrderedDict[str, List[str]]" = OrderedDict((g, []) for g in _COMPONENT_GROUPS)
    for k in keys:
        if k == "model.meta":
            continue
        groups[_component_group(k)].append(k)
    return groups

def _make_bundles_simple(keys: List[str], *, put_global_first: bool = True) -> List[List[str]]:
    """
       - LAYER:<N>  -> 'layers.N'
       - GLOBAL     -> (embed_tokens, lm_head, mtp., ...)
    """
    groups: "OrderedDict[str, List[str]]" = OrderedDict()
    global_bucket: List[str] = []

    for k in keys:
        if k == "model.meta":
            continue
        lid = _layer_id_from_raw(k)
        if lid is None:
            global_bucket.append(k)
        else:
            groups.setdefault(f"LAYER:{lid}", []).append(k)

    layer_keys = sorted(groups.keys(), key=lambda t: int(t.split(":")[1]))  # ["LAYER:0","LAYER:1",...]

    bundles: List[List[str]] = []
    if put_global_first and global_bucket:
        bundles.append(global_bucket)
    bundles.extend(groups[t] for t in layer_keys)
    if not put_global_first and global_bucket:
        bundles.append(global_bucket)

    return bundles

def _select_keys_for_rank_by_bundles(
    all_keys: List[str],
    chunk_layers: int,
    rank: int,
    world: int,
    *,
    put_global_first: bool = True,
) -> List[str]:
    selected: List[str] = []
    for group_keys in _split_keys_by_component(all_keys).values():
        if not group_keys:
            continue
        bundles = _make_bundles_simple(group_keys, put_global_first=put_global_first)
        bundle_chunks = [bundles[i:i+chunk_layers] for i in range(0, len(bundles), chunk_layers)]
        my_bundles = [b for i, ch in enumerate(bundle_chunks) if (i % world) == rank for b in ch]
        selected.extend(k for b in my_bundles for k in b)
    return selected

def _workplan_from_state_by_bundles(
    state_model: Dict[str, torch.Tensor],
    chunk_layers: int,
    *,
    put_global_first: bool = True,
) -> List[Tuple[int, List[str]]]:
    raw_keys = [k for k in state_model.keys() if k != "model.meta"]
    work_plan: List[Tuple[int, List[str]]] = []
    chunk_idx = 0
    for group_keys in _split_keys_by_component(raw_keys).values():
        if not group_keys:
            continue
        bundles = _make_bundles_simple(group_keys, put_global_first=put_global_first)
        bundle_chunks = [bundles[i:i+chunk_layers] for i in range(0, len(bundles), chunk_layers)]
        for ch in bundle_chunks:
            work_plan.append((chunk_idx, [k for grp in ch for k in grp]))
            chunk_idx += 1
    return work_plan
