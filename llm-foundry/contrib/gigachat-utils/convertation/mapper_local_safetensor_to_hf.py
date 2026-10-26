#!/usr/bin/env python
# coding: utf-8
"""
Convert local training safetensors checkpoint → HuggingFace DeepSeek-V3 format.

Reads safetensors + llm_foundry_config.yaml (or config.json) with local naming,
applies DeepseekV3Mapper (forward direction), writes HF safetensors with
rank-based sharding.

Multi-node compatible: each rank processes a subset of layers.
After all ranks finish, merge with merge_safetensors_ranks.py.

Usage:
  python mapper_local_safetensor_to_hf.py \\
      --input /path/to/local_checkpoint \\
      --output /path/to/hf_checkpoint \\
      --global-rank 0 --world-size 4
"""
import argparse
import gc
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import safetensors
from safetensors.torch import save_file as safe_save_file
from tqdm import tqdm

try:
    import yaml
except ImportError:
    yaml = None


# Matches both old format (model.layers.X.) and new format with Composer prefix (model.model.layers.X.)
_LAYER_RE = re.compile(r"(?:model\.)?model\.layers\.(\d+)\.")
_MTP_LAYER_RE = re.compile(r"(?:model\.)?model\.mtp_block\.mtp_layers\.(\d+)\.")
_MTP_NORMS_RE = re.compile(r"(?:model\.)?model\.mtp_block\.mtp_norms\.(\d+)\.")


# ═══════════════════════════════════════════════════════════════════════════
# Config loading
# ═══════════════════════════════════════════════════════════════════════════

_DEEPSEEK_REQUIRED_KEYS = {
    "hidden_size", "num_attention_heads", "num_hidden_layers",
    "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim",
}


def _validate_deepseek_config(cfg: Dict[str, Any], source: str) -> None:
    """Check that config has the required DeepSeek-V3 parameters."""
    missing = _DEEPSEEK_REQUIRED_KEYS - set(cfg.keys())
    if missing:
        raise ValueError(
            f"Config from {source} is missing required DeepSeek parameters: {missing}. "
            f"Available keys: {sorted(cfg.keys())}"
        )


def load_local_model_config(input_dir: str) -> Dict[str, Any]:
    """
    Load model config from a local checkpoint directory.

    Tries in order:
      1. llm_foundry_config.yaml → model.config_overrides
      2. config.json (validated against DeepSeek required fields)
      3. config_old.json
      4. composer_state.pt → state.integrations.huggingface.model.config.content
    """
    # 1) llm_foundry_config.yaml
    llmf_path = os.path.join(input_dir, "llm_foundry_config.yaml")
    if os.path.exists(llmf_path):
        assert yaml is not None, "PyYAML is required to read llm_foundry_config.yaml"
        with open(llmf_path) as f:
            llmf = yaml.safe_load(f)
        config_overrides = llmf.get("model", {}).get("config_overrides", {})
        if config_overrides:
            _validate_deepseek_config(config_overrides, llmf_path)
            print(f"Loaded model config from {llmf_path} (model.config_overrides)")
            return config_overrides

    # 2) config.json
    cfg_path = os.path.join(input_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        _validate_deepseek_config(cfg, cfg_path)
        print(f"Loaded model config from {cfg_path}")
        return cfg

    # 3) composer_state.pt
    cs_path = os.path.join(input_dir, "composer_state.pt")
    if os.path.exists(cs_path):
        import torch
        cs = torch.load(cs_path, map_location="cpu", weights_only=False)
        try:
            cfg = cs["state"]["integrations"]["huggingface"]["model"]["config"]["content"]
            _validate_deepseek_config(cfg, cs_path)
            print(f"Loaded model config from {cs_path} (composer_state)")
            return cfg
        except (KeyError, TypeError):
            pass

    raise FileNotFoundError(
        f"No model config found in {input_dir}. "
        f"Tried: llm_foundry_config.yaml, config.json, config_old.json, composer_state.pt."
    )


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _group_keys_by_layer(
    all_keys: List[str],
    num_hidden_layers: int = 0,
) -> Tuple[List[str], Dict[int, List[str]]]:
    global_keys: List[str] = []
    layer_keys: Dict[int, List[str]] = {}
    for k in all_keys:
        m = _LAYER_RE.match(k)
        if m:
            layer_keys.setdefault(int(m.group(1)), []).append(k)
            continue
        m = _MTP_LAYER_RE.match(k)
        if m and num_hidden_layers > 0:
            layer_keys.setdefault(num_hidden_layers + int(m.group(1)), []).append(k)
            continue
        m = _MTP_NORMS_RE.match(k)
        if m and num_hidden_layers > 0:
            layer_keys.setdefault(num_hidden_layers + int(m.group(1)), []).append(k)
            continue
        global_keys.append(k)
    return global_keys, layer_keys


def _assign_layers_to_rank(
    num_layers: int, rank: int, world_size: int, chunk_layers: int = 1,
) -> List[int]:
    all_layers = list(range(num_layers))
    chunks = [all_layers[i:i + chunk_layers] for i in range(0, len(all_layers), chunk_layers)]
    my_layers = []
    for idx, ch in enumerate(chunks):
        if idx % world_size == rank:
            my_layers.extend(ch)
    return my_layers


def _load_keys_from_safetensors(
    model_dir: str, keys_to_load: List[str], weight_map: Dict[str, str],
) -> Dict[str, torch.Tensor]:
    shard_to_keys: Dict[str, List[str]] = {}
    for k in keys_to_load:
        sf = weight_map.get(k)
        if sf:
            shard_to_keys.setdefault(sf, []).append(k)
    result: Dict[str, torch.Tensor] = {}
    for sf, skeys in shard_to_keys.items():
        spath = os.path.join(model_dir, sf)
        with safetensors.safe_open(spath, framework="pt", device="cpu") as f:
            for k in skeys:
                result[k] = f.get_tensor(k)
    return result


def _save_chunk(tensors: Dict[str, torch.Tensor], out_path: str) -> int:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cpu = {k: v.detach().cpu().contiguous() for k, v in tensors.items()}
    safe_save_file(cpu, out_path)
    return sum(t.numel() * t.element_size() for t in cpu.values())


def _write_weight_map(out_rank_dir, rank, chunk_idx, mapped, shard_file, shard_size):
    os.makedirs(out_rank_dir, exist_ok=True)
    wm = {k: shard_file for k in mapped.keys()}
    wm_path = os.path.join(out_rank_dir, f"shard-r{rank:02d}-c{chunk_idx:05d}.weight_map.json")
    with open(wm_path, "w") as fh:
        json.dump({"weight_map": wm, "total_size": shard_size}, fh)


def _write_rank_index(out_rank_dir: str) -> None:
    import glob as glob_mod
    pattern = os.path.join(out_rank_dir, "shard-r*-c*.weight_map.json")
    files = sorted(glob_mod.glob(pattern))
    weight_map: Dict[str, str] = {}
    total_size = 0
    for f in files:
        with open(f) as fh:
            part = json.load(fh)
        for k, v in part.get("weight_map", {}).items():
            weight_map.setdefault(k, v)
        total_size += int(part.get("total_size", 0))
    local_index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(os.path.join(out_rank_dir, "rank-index.json"), "w") as fh:
        json.dump(local_index, fh, indent=2)
    print(f"[{os.path.basename(out_rank_dir)}] rank-index.json: "
          f"{len(weight_map)} tensors, {total_size} bytes")


# ═══════════════════════════════════════════════════════════════════════════
# Main streaming function
# ═══════════════════════════════════════════════════════════════════════════

def stream_local_safetensor_to_hf(
    input_dir: str,
    output_dir: str,
    *,
    rank: int = 0,
    world_size: int = 1,
    chunk_layers: int = 1,
    local_config: Optional[Dict[str, Any]] = None,
    args=None,
) -> None:
    """
    Streaming multi-node conversion: local safetensors → HF DeepSeek-V3 format.

    Applies DeepseekV3Mapper (from base_mappers / transformers_utils/deepseek.py)
    in the forward direction (local → HF).

    Output: rank{NN}/shard-r{NN}-c{CCCCC}.safetensors + rank-index.json
    """
    from base_mappers import DeepseekV3Mapper

    os.makedirs(output_dir, exist_ok=True)

    # Load local config
    if local_config is None:
        local_config = load_local_model_config(input_dir)

    # Load safetensors index
    index_path = os.path.join(input_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            idx = json.load(f)
        weight_map = idx["weight_map"]
    else:
        weight_map = {}
        for sf in sorted(Path(input_dir).glob("*.safetensors")):
            with safetensors.safe_open(str(sf), framework="pt") as f:
                for k in f.keys():
                    weight_map[k] = sf.name

    all_keys = sorted(weight_map.keys())
    num_hidden_layers = local_config.get("num_hidden_layers", 0)
    global_keys, layer_keys_map = _group_keys_by_layer(all_keys, num_hidden_layers)
    num_layers = max(layer_keys_map.keys()) + 1 if layer_keys_map else 0

    my_layers = _assign_layers_to_rank(num_layers, rank, world_size, chunk_layers)
    handle_globals = (rank == 0)

    # Initialize the forward mapper
    mapper = DeepseekV3Mapper()
    mapper.ensure_config(local_config, args)

    # Rank 0: save HF config, tokenizer, runtime code
    if rank == 0:
        try:
            mapper.save_meta(local_config, args, output_dir)
            print(f"[rank {rank}] Saved HF config/tokenizer")
        except Exception as e:
            print(f"[rank {rank}] WARNING: Failed to save meta: {e}")

    out_rank_dir = os.path.join(output_dir, f"rank{rank:02d}")
    os.makedirs(out_rank_dir, exist_ok=True)

    chunk_idx = 0

    def _strip_composer_prefix(tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Strip outer 'model.' Composer prefix if present (model.model.xxx → model.xxx, model.lm_head.weight → lm_head.weight).
        New-format checkpoints have this extra prefix for papayaFSDP compatibility; DeepseekV3Mapper expects the old format."""
        if not any(k.startswith("model.model.") for k in tensors):
            return tensors  # old format already, no change needed
        return {re.sub(r"^model\.", "", k, count=1): v for k, v in tensors.items()}

    # Global tensors (rank 0)
    if handle_globals and global_keys:
        raw = _load_keys_from_safetensors(input_dir, global_keys, weight_map)
        raw = _strip_composer_prefix(raw)
        mapped = mapper.map(raw)
        mapped.update(mapper.flush())
        del raw

        shard_file = f"shard-r{rank:02d}-c{chunk_idx:05d}.safetensors"
        shard_path = os.path.join(out_rank_dir, shard_file)
        shard_size = _save_chunk(mapped, shard_path)
        _write_weight_map(out_rank_dir, rank, chunk_idx, mapped, shard_file, shard_size)
        mapped.clear()
        chunk_idx += 1
        print(f"[rank {rank}] Saved global tensors ({len(global_keys)} keys)")

    # Process layers
    layer_chunks = [my_layers[i:i + chunk_layers] for i in range(0, len(my_layers), chunk_layers)]

    for lc in tqdm(layer_chunks, desc=f"[rank {rank}] Converting layers"):
        layer_keys: List[str] = []
        for lid in lc:
            layer_keys.extend(layer_keys_map.get(lid, []))

        raw = _load_keys_from_safetensors(input_dir, layer_keys, weight_map)
        raw = _strip_composer_prefix(raw)
        mapped = mapper.map(raw)
        mapped.update(mapper.flush())
        del raw

        shard_file = f"shard-r{rank:02d}-c{chunk_idx:05d}.safetensors"
        shard_path = os.path.join(out_rank_dir, shard_file)
        shard_size = _save_chunk(mapped, shard_path)
        _write_weight_map(out_rank_dir, rank, chunk_idx, mapped, shard_file, shard_size)
        mapped.clear()
        chunk_idx += 1
        gc.collect()

    _write_rank_index(out_rank_dir)
    print(f"[rank {rank}] Done. Processed {len(my_layers)} layers, wrote {chunk_idx} shards.")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Convert local safetensors checkpoint → HF DeepSeek-V3 format (multi-node)"
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Path to local checkpoint dir (safetensors + llm_foundry_config.yaml or config.json)")
    parser.add_argument("--output", "-o", required=True,
                        help="Path to output HF directory")
    parser.add_argument("--global-rank", type=int, default=None)
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--chunk-layers", type=int, default=1)
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--torch-dtype", type=str, default="bfloat16")
    args = parser.parse_args()

    rank = args.global_rank if args.global_rank is not None else int(os.environ.get("RANK", "0"))
    world_size = args.world_size if args.world_size is not None else int(os.environ.get("WORLD_SIZE", "1"))

    print(f"[rank {rank}/{world_size}] input={args.input} output={args.output}")

    stream_local_safetensor_to_hf(
        input_dir=args.input,
        output_dir=args.output,
        rank=rank,
        world_size=world_size,
        chunk_layers=args.chunk_layers,
        args=args,
    )

    if rank == 0 and world_size == 1:
        from utils.merge_ranks import flatten_move_and_reindex
        print("\n=== Merging rank shards ===")
        flatten_move_and_reindex(args.output)

    print(f"[rank {rank}] All done!")


if __name__ == "__main__":
    main()
