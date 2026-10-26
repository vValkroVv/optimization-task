#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, shutil, os, sys
from pathlib import Path
from typing import Dict, Tuple

IDX_NAME = "model.safetensors.index.json"

def load_index(dir_path: Path) -> Tuple[Dict, Dict[str, str]]:
    idx_path = dir_path / IDX_NAME
    if not idx_path.exists():
        raise FileNotFoundError(f"Index not found: {idx_path}")
    with open(idx_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    meta = data.get("metadata", {}) or {}
    wm = data.get("weight_map", {}) or {}
    return meta, wm

def load_index_if_exists(dir_path: Path) -> Tuple[Dict, Dict[str, str]]:
    idx_path = dir_path / IDX_NAME
    if not idx_path.exists():
        return {}, {}
    with open(idx_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return (data.get("metadata", {}) or {}, data.get("weight_map", {}) or {})

def copy_shards_with_prefix(src: Path, dst: Path, prefix: str) -> Dict[str, str]:
    dst.mkdir(parents=True, exist_ok=True)
    rename_map: Dict[str, str] = {}
    for p in sorted(src.glob("*.safetensors")):
        new_name = f"{prefix}{p.name}"
        out = dst / new_name
        if out.exists():
            rename_map[p.name] = new_name
            continue
        shutil.copy2(p, out)
        rename_map[p.name] = new_name
    return rename_map

def maybe_copy_static_files(src: Path, dst: Path, copy_config_and_tokenizer: bool):
    if not copy_config_and_tokenizer:
        return
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json", "generation_config.json",
                 "spiece.model", "merges.txt", "vocab.json"):
        sp = src / name
        dp = dst / name
        if sp.exists() and not dp.exists():
            shutil.copy2(sp, dp)

def copy_non_safetensors_from_src1(src: Path, dst: Path):
    for root, dirs, files in os.walk(src):
        root_p = Path(root)
        rel = root_p.relative_to(src)
        dst_root = dst / rel
        dst_root.mkdir(parents=True, exist_ok=True)

        for fname in files:
            if fname.endswith(".safetensors"):
                continue
            if fname == IDX_NAME:
                continue
            sp = root_p / fname
            dp = dst_root / fname
            if not dp.exists():
                shutil.copy2(sp, dp)

def merge_into_base_weight_map(
    base_wm: Dict[str, str],
    wm_add: Dict[str, str],
    rename_map: Dict[str, str],
    tag: str,
    prefer_existing: bool = True,
    verbose: bool = True,
) -> Dict[str, str]:
    out = dict(base_wm)
    for tensor_key, shard_name in wm_add.items():
        new_name = rename_map.get(shard_name)
        if not new_name:
            raise KeyError(
                f"[{tag}] Shard '{shard_name}' is missing from the renamed files (rename_map)."
            )
        if tensor_key in out:
            if prefer_existing:
                if verbose:
                    print(f"[warn] skip duplicate tensor key from {tag}: {tensor_key} "
                          f"(keep existing -> {out[tensor_key]})")
                continue
            else:
                if verbose:
                    print(f"[info] overwrite tensor key from {tag}: {tensor_key} "
                          f"{out[tensor_key]} -> {new_name}")
        out[tensor_key] = new_name
    return out

def recompute_total_size(dst: Path, weight_map: Dict[str, str]) -> int:
    seen = set(weight_map.values())
    total = 0
    for fname in seen:
        fp = dst / fname
        try:
            total += fp.stat().st_size
        except FileNotFoundError:
            print(f"[warn] referenced shard not found in dst: {fname}", file=sys.stderr)
    return total

def parse_args():
    ap = argparse.ArgumentParser(
        description="Merge two HF checkpoint folders into one with prefixed shard filenames (incremental, skip duplicates)."
    )
    ap.add_argument("--src1", required=True, help="Path to the first hf_checkpoint folder (.../ema_1/)")
    ap.add_argument("--src2", required=True, help="Path to the second hf_checkpoint folder (.../ema_2/)")
    ap.add_argument("--dst",  required=True, help="New/existing target hf_checkpoint folder (will be created if missing)")
    ap.add_argument("--prefix1", default="ema1-", help="Prefix for shards from src1 (default: 'ema1-')")
    ap.add_argument("--prefix2", default="ema2-", help="Prefix for shards from src2 (default: 'ema2-')")
    ap.add_argument("--copy-static", action="store_true",
                    help="Copy config/tokenizer files from src1 if they are missing in dst.")
    ap.add_argument("--overwrite-duplicates", action="store_true",
                    help="Overwrite existing weight_map entries on collisions (disabled by default).")
    return ap.parse_args()

def main():
    args = parse_args()
    src1 = Path(args.src1).resolve()
    src2 = Path(args.src2).resolve()
    dst  = Path(args.dst).resolve()

    if not src1.exists() or not src2.exists():
        raise FileNotFoundError("One of source folders does not exist.")

    dst.mkdir(parents=True, exist_ok=True)

    base_meta, base_wm = load_index_if_exists(dst)

    meta1, wm1 = load_index(src1)
    meta2, wm2 = load_index(src2)

    copy_non_safetensors_from_src1(src1, dst)

    maybe_copy_static_files(src1, dst, args.copy_static)

    rename1 = copy_shards_with_prefix(src1, dst, args.prefix1)
    rename2 = copy_shards_with_prefix(src2, dst, args.prefix2)

    prefer_existing = not args.overwrite_duplicates
    merged_wm = merge_into_base_weight_map(base_wm, wm1, rename1, tag="src1", prefer_existing=prefer_existing)
    merged_wm = merge_into_base_weight_map(merged_wm, wm2, rename2, tag="src2", prefer_existing=prefer_existing)

    total_size = recompute_total_size(dst, merged_wm)
    merged = {"metadata": {"total_size": int(total_size)}, "weight_map": merged_wm}

    out_path = dst / IDX_NAME
    tmp_path = dst / (IDX_NAME + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, out_path)

    print(f"[OK] Wrote merged index to: {out_path}")
    print(f"[OK] Total tensors: {len(merged_wm)}, total_size={merged['metadata']['total_size']}")
    print(f"[OK] Unique shards referenced: {len(set(merged_wm.values()))}")
    print(f"[OK] Files in dst: {len(list(dst.glob('*.safetensors')))}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
