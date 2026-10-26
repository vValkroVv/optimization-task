#!/usr/bin/env python3
"""Append extra MTP heads to a safetensors checkpoint by replicating an existing head.

Works on both naming conventions:
  * local / monolith: ``model.mtp_block.mtp_layers.{i}.*`` + ``model.mtp_block.mtp_norms.{i}.*``
  * HF DeepSeek-style: ``model.layers.{num_hidden_layers + i}.*``

Only the replicated head is written out (a couple of small extra shards); every other shard is
reused as-is, so the cost does not depend on the size of the checkpoint. By default the result is
a new directory of symlinks + the new shards + a patched index/config; ``--in-place`` instead
appends the new shards next to the originals and rewrites the index there. Every file ``--in-place``
overwrites (index, ``config.json``, ``composer_state.pt``, any patched ``*.yaml``) gets a one-time
``.bak`` copy first.

Example:
    python append_mtp_heads.py --checkpoint-dir /path/to/ckpt --num-new 2 \
        --output-dir /path/to/ckpt_mtp3
"""
import argparse
import copy
import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

INDEX_CANDIDATES = ("model.safetensors.index.json", "index.json", "rank-index.json")

_LOCAL_LAYER_RE = re.compile(r"mtp_block\.mtp_layers\.(\d+)\.")
_LOCAL_NORM_RE = re.compile(r"mtp_block\.mtp_norms\.(\d+)\.")
_HF_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")

# Small files copied next to the shards when building a new output directory.
SIDECAR_SUFFIXES = (".json", ".py", ".yaml", ".yml", ".txt", ".model")


def find_index(ckpt_dir: Path, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        return p if p.is_absolute() else ckpt_dir / explicit
    for name in INDEX_CANDIDATES:
        if (ckpt_dir / name).exists():
            return ckpt_dir / name
    raise FileNotFoundError(
        f"No index found in {ckpt_dir}; tried {INDEX_CANDIDATES}. Pass --index explicitly."
    )


def load_config(ckpt_dir: Path) -> dict:
    cfg_path = ckpt_dir / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return json.load(f)
    return {}


def detect_layout(weight_map: Dict[str, str]) -> str:
    if any(_LOCAL_LAYER_RE.search(k) or _LOCAL_NORM_RE.search(k) for k in weight_map):
        return "local"
    return "hf"


def local_head_ids(weight_map: Dict[str, str]) -> List[int]:
    ids = set()
    for k in weight_map:
        for rx in (_LOCAL_LAYER_RE, _LOCAL_NORM_RE):
            m = rx.search(k)
            if m:
                ids.add(int(m.group(1)))
    return sorted(ids)


def hf_head_ids(weight_map: Dict[str, str], num_hidden_layers: int) -> List[int]:
    ids = set()
    for k in weight_map:
        m = _HF_LAYER_RE.match(k)
        if m and int(m.group(1)) >= num_hidden_layers:
            ids.add(int(m.group(1)) - num_hidden_layers)
    return sorted(ids)


def plan_local(weight_map: Dict[str, str], src: int, new_ids: List[int]) -> Dict[str, str]:
    """Return {new_key: src_key} for the local/monolith naming."""
    src_keys = [
        k for k in weight_map
        if f"mtp_block.mtp_layers.{src}." in k or f"mtp_block.mtp_norms.{src}." in k
    ]
    if not src_keys:
        raise ValueError(f"No keys found for MTP head {src} (mtp_layers.{src}/mtp_norms.{src}).")
    mapping: Dict[str, str] = {}
    for new in new_ids:
        for k in src_keys:
            nk = k.replace(f"mtp_layers.{src}.", f"mtp_layers.{new}.")
            nk = nk.replace(f"mtp_norms.{src}.", f"mtp_norms.{new}.")
            if nk == k:
                raise ValueError(f"Failed to rename key {k} for head {new}")
            mapping[nk] = k
    return mapping


def plan_hf(
    weight_map: Dict[str, str], src: int, new_ids: List[int], num_hidden_layers: int
) -> Dict[str, str]:
    """Return {new_key: src_key} for the HF DeepSeek-style naming."""
    src_prefix = f"model.layers.{num_hidden_layers + src}."
    src_keys = [k for k in weight_map if k.startswith(src_prefix)]
    if not src_keys:
        raise ValueError(f"No keys found under prefix '{src_prefix}'.")
    mapping: Dict[str, str] = {}
    for new in new_ids:
        dst_prefix = f"model.layers.{num_hidden_layers + new}."
        for k in src_keys:
            mapping[dst_prefix + k[len(src_prefix):]] = k
    return mapping


def write_new_shards(
    ckpt_dir: Path,
    out_dir: Path,
    weight_map: Dict[str, str],
    mapping: Dict[str, str],
    shard_gb: float,
    shard_prefix: str,
) -> Tuple[Dict[str, str], int]:
    """Stream the source tensors into fresh shards. Returns (new weight map entries, bytes)."""
    # Group by source shard so every source file is opened exactly once.
    groups: Dict[str, List[Tuple[str, str]]] = {}
    for new_k, src_k in mapping.items():
        groups.setdefault(weight_map[src_k], []).append((src_k, new_k))

    shard_bytes = int(shard_gb * 1024 ** 3)
    new_map: Dict[str, str] = {}
    total = 0
    cur: Dict[str, torch.Tensor] = {}
    cur_size = 0
    chunk_idx = 0

    def flush():
        nonlocal cur, cur_size, chunk_idx, total
        if not cur:
            return
        fname = f"{shard_prefix}-c{chunk_idx:05d}.safetensors"
        save_file({k: v.contiguous() for k, v in cur.items()}, str(out_dir / fname))
        size = sum(t.numel() * t.element_size() for t in cur.values())
        for k in cur:
            new_map[k] = fname
        total += size
        print(f"  wrote {fname}: {len(cur)} tensors, {size / 1e9:.3f} GB")
        cur = {}
        cur_size = 0
        chunk_idx += 1

    for src_shard, pairs in sorted(groups.items()):
        with safe_open(str(ckpt_dir / src_shard), framework="pt", device="cpu") as f:
            for src_k, new_k in pairs:
                t = f.get_tensor(src_k).clone()  # clone: never keep mmap-backed storage
                nb = t.numel() * t.element_size()
                if cur and cur_size + nb > shard_bytes:
                    flush()
                cur[new_k] = t
                cur_size += nb
    flush()
    return new_map, total


def link_existing(ckpt_dir: Path, out_dir: Path, index_path: Path, link_mode: str) -> None:
    """Populate out_dir with links to the original shards and copies of the small sidecar files."""
    for src in sorted(ckpt_dir.iterdir()):
        if src.is_dir() or src.name == index_path.name:
            continue
        dst = out_dir / src.name
        if dst.exists() or dst.is_symlink():
            continue
        if src.suffix == ".safetensors":
            if link_mode == "hard":
                os.link(src, dst)
            else:
                dst.symlink_to(src.resolve())
        elif src.suffix in SIDECAR_SUFFIXES or src.name.startswith("tokenizer"):
            shutil.copy2(src, dst)


def patch_config(cfg: dict, total_heads: int) -> Tuple[dict, List[str]]:
    out = copy.deepcopy(cfg)
    touched = []
    for key in ("num_nextn_predict_layers", "mtp_predictor_num"):
        if key in out:
            out[key] = total_heads
            touched.append(key)
    return out, touched


def backup_once(path: Path, backup: bool) -> None:
    """Keep a one-time .bak of a file about to be overwritten in place.

    Only needed for --in-place: in the --output-dir flow every file we rewrite is
    already a copy, and the originals are never touched."""
    if not backup or not path.exists():
        return
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"backed up  {path.name} -> {bak.name}")


def patch_yaml_configs(out_dir: Path, total_heads: int, backup: bool = False) -> List[str]:
    """Best-effort update of mtp_predictor_num inside llm_foundry_config.yaml-style files.

    safe_dump drops comments and reflows the file, so in place we keep a .bak first."""
    try:
        import yaml
    except ImportError:
        return []
    touched = []
    for path in sorted(out_dir.glob("*.y*ml")):
        with open(path) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, (dict, list)):
            continue
        hits = [0]

        def walk(node):
            if isinstance(node, dict):
                for k, v in node.items():
                    if k in ("mtp_predictor_num", "num_nextn_predict_layers") and isinstance(v, int):
                        node[k] = total_heads
                        hits[0] += 1
                    else:
                        walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)

        walk(data)
        if hits[0]:
            backup_once(path, backup)
            with open(path, "w") as f:
                yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
            touched.append(f"{path.name} ({hits[0]} field(s))")
    return touched


def patch_composer_state(ckpt_dir: Path, out_dir: Path, total_heads: int,
                        backup: bool = False) -> str:
    """Patch the embedded HF config inside a small composer_state.pt (config-only, as written by
    the MTP mappers). Full training states are left alone — they are huge and are not what a
    converted checkpoint carries."""
    src = ckpt_dir / "composer_state.pt"
    if not src.exists():
        return "state:     no composer_state.pt"

    def keep_original(reason: str) -> str:
        dst = out_dir / src.name
        if out_dir != ckpt_dir and not (dst.exists() or dst.is_symlink()):
            dst.symlink_to(src.resolve())
        return f"state:     composer_state.pt {reason} — not patched"

    size_mb = src.stat().st_size / 1e6
    if size_mb > 100:
        return keep_original(f"is {size_mb:.0f} MB (full training state)")
    state = torch.load(str(src), map_location="cpu", weights_only=False)
    try:
        content = state["state"]["integrations"]["huggingface"]["model"]["config"]["content"]
    except (KeyError, TypeError):
        return keep_original("has no embedded HF config")
    new_content, touched = patch_config(content, total_heads)
    if not touched:
        return keep_original("config has no MTP-count field")
    state["state"]["integrations"]["huggingface"]["model"]["config"]["content"] = new_content
    backup_once(out_dir / "composer_state.pt", backup)
    torch.save(state, str(out_dir / "composer_state.pt"))
    return f"state:     composer_state.pt {touched} -> {total_heads}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint-dir", required=True, help="Directory with the shards + index")
    ap.add_argument("--num-new", type=int, default=2, help="How many extra MTP heads to append")
    ap.add_argument("--src-head", type=int, default=0, help="0-indexed MTP head to replicate")
    ap.add_argument("--output-dir", help="Where to write the result (links to the original shards)")
    ap.add_argument("--in-place", action="store_true", help="Append shards into --checkpoint-dir itself")
    ap.add_argument("--index", help="Index filename/path (default: auto-detect)")
    ap.add_argument("--shard-gb", type=float, default=4.0, help="Max size of a new shard")
    ap.add_argument("--link-mode", choices=("sym", "hard"), default="sym",
                    help="How to reference the original shards in --output-dir")
    ap.add_argument("--dry-run", action="store_true", help="Print the plan and exit")
    args = ap.parse_args()

    ckpt_dir = Path(args.checkpoint_dir).resolve()
    if bool(args.output_dir) == bool(args.in_place):
        ap.error("pass exactly one of --output-dir / --in-place")
    out_dir = ckpt_dir if args.in_place else Path(args.output_dir).resolve()

    index_path = find_index(ckpt_dir, args.index)
    with open(index_path) as f:
        index = json.load(f)
    weight_map: Dict[str, str] = index["weight_map"]

    cfg = load_config(ckpt_dir)
    layout = detect_layout(weight_map)

    if layout == "local":
        existing = local_head_ids(weight_map)
    else:
        if "num_hidden_layers" not in cfg:
            ap.error(f"HF-style checkpoint needs num_hidden_layers in {ckpt_dir}/config.json")
        existing = hf_head_ids(weight_map, int(cfg["num_hidden_layers"]))
    if not existing:
        ap.error("No MTP head found in the index — nothing to replicate.")
    if args.src_head not in existing:
        ap.error(f"--src-head {args.src_head} not in existing heads {existing}")

    new_ids = list(range(max(existing) + 1, max(existing) + 1 + args.num_new))
    if layout == "local":
        mapping = plan_local(weight_map, args.src_head, new_ids)
    else:
        mapping = plan_hf(weight_map, args.src_head, new_ids, int(cfg["num_hidden_layers"]))

    total_heads = len(existing) + args.num_new
    print(f"index:     {index_path}")
    print(f"layout:    {layout}")
    print(f"heads:     existing {existing} + new {new_ids}  ->  {total_heads} total")
    print(f"tensors:   {len(mapping)} new keys replicated from head {args.src_head}")
    print(f"output:    {out_dir}{' (in place)' if args.in_place else ''}")
    sample = sorted(mapping)[:3]
    for k in sample:
        print(f"           {mapping[k]}  ->  {k}")
    if args.dry_run:
        print("dry run — nothing written")
        return

    collisions = sorted(set(mapping) & set(weight_map))
    if collisions:
        ap.error(f"{len(collisions)} target keys already exist, e.g. {collisions[:3]}")

    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.in_place:
        link_existing(ckpt_dir, out_dir, index_path, args.link_mode)

    new_entries, added_bytes = write_new_shards(
        ckpt_dir, out_dir, weight_map, mapping, args.shard_gb, "mtp-extra-r00"
    )

    new_index = {
        "metadata": {**index.get("metadata", {}),
                     "total_size": int(index.get("metadata", {}).get("total_size", 0)) + added_bytes},
        "weight_map": {**weight_map, **new_entries},
    }
    out_index = out_dir / index_path.name
    backup_once(index_path, args.in_place)
    with open(out_index, "w") as f:
        json.dump(new_index, f, indent=2)
    print(f"index:     {out_index} ({len(new_index['weight_map'])} tensors, "
          f"+{added_bytes / 1e9:.3f} GB)")

    if cfg:
        new_cfg, touched = patch_config(cfg, total_heads)
        if touched:
            backup_once(out_dir / "config.json", args.in_place)
            with open(out_dir / "config.json", "w") as f:
                json.dump(new_cfg, f, indent=2)
            print(f"config:    config.json {touched} -> {total_heads}")
        else:
            print("config:    no MTP-count field in config.json — set it manually")
    for msg in patch_yaml_configs(out_dir, total_heads, backup=args.in_place):
        print(f"config:    {msg} -> {total_heads}")
    print(patch_composer_state(ckpt_dir, out_dir, total_heads, backup=args.in_place))

    print("done")


if __name__ == "__main__":
    main()
