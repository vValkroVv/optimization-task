import os
import glob
import json
import shutil
from typing import Dict, Tuple


def _load_rank_index(rank_dir: str) -> Tuple[Dict[str, str], int]:
    local_index_path = os.path.join(rank_dir, "rank-index.json")
    weight_map: Dict[str, str] = {}
    total_size = 0

    if os.path.isfile(local_index_path):
        with open(local_index_path, "r", encoding="utf-8") as fh:
            part = json.load(fh)
        weight_map.update(part.get("weight_map", {}))
        total_size += int(part.get("metadata", {}).get("total_size", 0))
    else:
        pattern = os.path.join(rank_dir, "shard-r*-c*.weight_map.json")
        files = sorted(glob.glob(pattern))
        for f in files:
            with open(f, "r", encoding="utf-8") as fh:
                part = json.load(fh)
            weight_map.update(part.get("weight_map", {}))
            total_size += int(part.get("total_size", 0))

    return weight_map, total_size


def _resolve_safetensors_path(rank_dir: str, rel_or_name: str) -> str:
    if os.path.isabs(rel_or_name):
        p = rel_or_name
    else:
        p = os.path.join(rank_dir, rel_or_name)
        if not os.path.isfile(p):
            p = os.path.join(rank_dir, os.path.basename(rel_or_name))
    return os.path.realpath(p)


def _remove_empty_dirs(dir_path: str):
    if not os.path.isdir(dir_path):
        return
    with os.scandir(dir_path) as it:
        for entry in list(it):
            if entry.is_dir(follow_symlinks=False):
                _remove_empty_dirs(entry.path)
    try:
        os.rmdir(dir_path)
    except OSError:
        pass


def flatten_move_and_reindex(
    base_dir: str,
    out_index_name: str = "model.safetensors.index.json",
):
    base_dir = os.path.realpath(base_dir)
    rank_dirs = sorted([p for p in glob.glob(os.path.join(base_dir, "rank*")) if os.path.isdir(p)])
    if not rank_dirs:
        raise FileNotFoundError(f"Not found subfolders rank*/ in: {base_dir}")

    merged_tensor_to_src: Dict[str, str] = {}
    total_size = 0

    for rd in rank_dirs:
        wm_local, sz = _load_rank_index(rd)
        for tensor_name, rel_path in wm_local.items():
            src_path = _resolve_safetensors_path(rd, rel_path)
            if not os.path.isfile(src_path):
                raise FileNotFoundError(f"File from index not found: {src_path} (tensor '{tensor_name}')")
            merged_tensor_to_src.setdefault(tensor_name, src_path)
        total_size += sz

    unique_src_paths = sorted(set(merged_tensor_to_src.values()))
    basenames = [os.path.basename(p) for p in unique_src_paths]
    if len(basenames) != len(set(basenames)):
        dups = []
        seen = {}
        for p in unique_src_paths:
            b = os.path.basename(p)
            if b in seen and os.path.realpath(seen[b]) != os.path.realpath(p):
                dups.append(b)
            else:
                seen[b] = p
        dups = sorted(set(dups))
        if dups:
            raise AssertionError("Name collisions when moving to root: " + ", ".join(dups))

    for bname in set(basenames):
        dst = os.path.join(base_dir, bname)
        if os.path.exists(dst):
            raise AssertionError(f"There is already a file at the root: {dst}.")

    moved = 0
    for src in set(merged_tensor_to_src.values()):
        dst = os.path.join(base_dir, os.path.basename(src))
        os.makedirs(base_dir, exist_ok=True)
        shutil.move(src, dst)
        moved += 1

    for rd in rank_dirs:
        _remove_empty_dirs(rd)

    new_weight_map: Dict[str, str] = {
        tensor_name: os.path.basename(old_src)
        for tensor_name, old_src in merged_tensor_to_src.items()
    }

    index = {
        "metadata": {"total_size": int(total_size)},
        "weight_map": new_weight_map,
    }

    out_path = os.path.join(base_dir, out_index_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    print(
        f"[ok] Transfer complete.\n"
        f" Files moved: {moved}\n"
        f" Index: {out_path}\n"
        f" Tensors: {len(new_weight_map)}, total_size={total_size} bytes"
    )
