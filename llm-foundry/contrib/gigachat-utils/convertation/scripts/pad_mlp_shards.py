#!/usr/bin/env python3
import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file


TARGET_SUBSTRINGS = (
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def ceil_to_multiple(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


def tensor_nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def load_safetensors_file(path: Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, str] | None]:
    tensors = {}
    metadata = None
    with safe_open(str(path), framework="pt", device="cpu") as f:
        try:
            metadata = f.metadata()
        except Exception:
            metadata = None
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    return tensors, metadata


def save_safetensors_file(path: Path, tensors: Dict[str, torch.Tensor], metadata: Dict[str, str] | None):
    for k, v in tensors.items():
        if not v.is_contiguous():
            tensors[k] = v.contiguous()
    save_file(tensors, str(path), metadata=metadata)


def infer_pad_axis(name: str) -> int:
    # Hugging Face Linear weight is usually [out_features, in_features]
    # gate_proj / up_proj: hidden -> intermediate, so weight shape [intermediate, hidden]
    # down_proj: intermediate -> hidden, so weight shape [hidden, intermediate]
    if "mlp.gate_proj.weight" in name:
        return 0
    if "mlp.up_proj.weight" in name:
        return 0
    if "mlp.down_proj.weight" in name:
        return 1
    raise ValueError(f"Unsupported tensor name for padding axis inference: {name}")


def is_target_tensor(name: str) -> bool:
    return name.endswith(".weight") and any(s in name for s in TARGET_SUBSTRINGS)


def pad_tensor(t: torch.Tensor, axis: int, new_size: int) -> torch.Tensor:
    old_size = t.shape[axis]
    if new_size < old_size:
        raise ValueError(f"new_size={new_size} < old_size={old_size}")
    if new_size == old_size:
        return t

    pad_amount = new_size - old_size
    if axis == 0:
        # pad rows
        pad_shape = (pad_amount, *t.shape[1:])
        pad_tensor = torch.zeros(pad_shape, dtype=t.dtype, device=t.device)
        return torch.cat([t, pad_tensor], dim=0)
    elif axis == 1:
        # pad cols
        pad_shape = (t.shape[0], pad_amount, *t.shape[2:])
        pad_tensor = torch.zeros(pad_shape, dtype=t.dtype, device=t.device)
        return torch.cat([t, pad_tensor], dim=1)
    else:
        raise ValueError(f"Only axis 0/1 supported, got axis={axis}")


def backup_file(path: Path):
    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(path, backup)


def update_config_json(
    config_path: Path,
    old_intermediate_size: int,
    new_intermediate_size: int,
    dry_run: bool,
) -> None:
    if not config_path.exists():
        print(f"[WARN] config.json not found: {config_path}")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    changed = False
    current = cfg.get("intermediate_size", None)
    if current is None:
        print("[WARN] config.json has no intermediate_size; skipping config update")
    elif current != old_intermediate_size:
        print(
            f"[WARN] config.json intermediate_size={current}, expected {old_intermediate_size}. "
            "Skipping automatic update."
        )
    else:
        cfg["intermediate_size"] = new_intermediate_size
        changed = True

    if changed:
        print(f"[INFO] Updating config.json: intermediate_size {old_intermediate_size} -> {new_intermediate_size}")
        if not dry_run:
            backup_file(config_path)
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
                f.write("\n")


def recompute_total_size_from_index(index_data: dict, checkpoint_dir: Path) -> int:
    # Recompute from actual tensors in shards referenced by index.
    shard_names = sorted(set(index_data["weight_map"].values()))
    total = 0
    for shard_name in shard_names:
        shard_path = checkpoint_dir / shard_name
        tensors, _ = load_safetensors_file(shard_path)
        total += sum(tensor_nbytes(t) for t in tensors.values())
    return total


def main():
    parser = argparse.ArgumentParser(description="Pad selected MLP weights inside a sharded HF safetensors checkpoint.")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Directory containing model.safetensors.index.json and shard files",
    )
    parser.add_argument(
        "--index-file",
        type=str,
        default="model.safetensors.index.json",
        help="Index filename inside checkpoint dir",
    )
    parser.add_argument(
        "--multiple-of",
        type=int,
        default=128,
        help="Pad target dimension to the nearest multiple of this value",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=None,
        help="Explicit padded intermediate size. If omitted, computed from old size and --multiple-of",
    )
    parser.add_argument(
        "--only-if-size",
        type=int,
        default=None,
        help="Only modify tensors whose relevant dimension equals this size (e.g. 10944)",
    )
    parser.add_argument(
        "--update-config",
        action="store_true",
        help="Update config.json intermediate_size if it matches old size",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned changes without rewriting files",
    )
    args = parser.parse_args()

    ckpt_dir = args.checkpoint_dir
    index_path = ckpt_dir / args.index_file
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    with open(index_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)

    weight_map: Dict[str, str] = index_data["weight_map"]

    target_tensors: List[str] = [name for name in weight_map if is_target_tensor(name)]
    if not target_tensors:
        print("[INFO] No matching tensors found.")
        return

    # Group targeted tensors by shard
    shard_to_tensor_names: Dict[str, List[str]] = {}
    for name in target_tensors:
        shard_to_tensor_names.setdefault(weight_map[name], []).append(name)

    print(f"[INFO] Found {len(target_tensors)} target tensors in {len(shard_to_tensor_names)} shard(s).")

    inferred_old_sizes = set()
    planned_changes = []

    # First pass: inspect shapes and determine target size
    for shard_name, tensor_names in sorted(shard_to_tensor_names.items()):
        shard_path = ckpt_dir / shard_name
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for name in sorted(tensor_names):
                shape = tuple(f.get_slice(name).get_shape())
                axis = infer_pad_axis(name)
                old_size = shape[axis]

                if args.only_if_size is not None and old_size != args.only_if_size:
                    continue

                inferred_old_sizes.add(old_size)
                new_size = args.target_size if args.target_size is not None else ceil_to_multiple(old_size, args.multiple_of)

                if new_size == old_size:
                    continue

                planned_changes.append((shard_name, name, shape, axis, old_size, new_size))

    if not planned_changes:
        print("[INFO] Nothing to change after filtering.")
        return

    if len(inferred_old_sizes) > 1 and args.target_size is None:
        raise ValueError(
            f"Found multiple source sizes on target axes: {sorted(inferred_old_sizes)}. "
            "Use --only-if-size or --target-size to disambiguate."
        )

    print("[INFO] Planned changes:")
    for shard_name, name, shape, axis, old_size, new_size in planned_changes:
        print(f"  - {name}: shard={shard_name}, shape={shape}, axis={axis}, {old_size} -> {new_size}")

    if args.dry_run:
        return

    changed_shards = sorted({x[0] for x in planned_changes})

    # Rewrite only changed shards
    for shard_name in changed_shards:
        shard_path = ckpt_dir / shard_name
        print(f"[INFO] Rewriting shard: {shard_path}")
        backup_file(shard_path)

        tensors, metadata = load_safetensors_file(shard_path)

        for planned_shard_name, name, shape, axis, old_size, new_size in planned_changes:
            if planned_shard_name != shard_name:
                continue
            t = tensors[name]
            actual_old = t.shape[axis]
            if actual_old != old_size:
                raise ValueError(
                    f"Tensor {name} shape changed between scan and rewrite: "
                    f"expected axis {axis} size {old_size}, got {actual_old}"
                )
            tensors[name] = pad_tensor(t, axis=axis, new_size=new_size)

        save_safetensors_file(shard_path, tensors, metadata=metadata)

    # Backup and update index metadata.total_size
    backup_file(index_path)
    total_size = recompute_total_size_from_index(index_data, ckpt_dir)
    index_data.setdefault("metadata", {})
    index_data["metadata"]["total_size"] = total_size

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    # Optionally update config.json
    if args.update_config:
        if len(inferred_old_sizes) == 1:
            old_intermediate_size = next(iter(inferred_old_sizes))
            new_intermediate_size = args.target_size if args.target_size is not None else ceil_to_multiple(
                old_intermediate_size, args.multiple_of
            )
            update_config_json(
                ckpt_dir / "config.json",
                old_intermediate_size=old_intermediate_size,
                new_intermediate_size=new_intermediate_size,
                dry_run=False,
            )
        else:
            print(
                "[WARN] Not updating config.json automatically because multiple source sizes were detected: "
                f"{sorted(inferred_old_sizes)}"
            )

    print("[INFO] Done.")


if __name__ == "__main__":
    main()