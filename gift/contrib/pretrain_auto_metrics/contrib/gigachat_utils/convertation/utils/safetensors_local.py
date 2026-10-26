import glob
import json
import os
import torch
from safetensors.torch import save_file as save_safetensors
from logger import logger

from typing import Dict


def _save_chunk_as_safetensors(tensors: Dict[str, torch.Tensor], out_path: str) -> int:
    cpu_tensors = {k: (v.detach().cpu().contiguous()) for k, v in tensors.items()}
    save_safetensors(cpu_tensors, out_path)
    return sum(t.numel() * t.element_size() for t in cpu_tensors.values())


def _write_local_index(output_rank_dir: str):
    pattern = os.path.join(output_rank_dir, "shard-r*-c*.weight_map.json")
    files = sorted(glob.glob(pattern))
    weight_map: Dict[str, str] = {}
    total_size = 0
    for f in files:
        with open(f, "r") as fh:
            part = json.load(fh)
        wm = part.get("weight_map", {})
        sz = int(part.get("total_size", 0))
        for k, v in wm.items():
            weight_map.setdefault(k, v)
        total_size += sz
    local_index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(os.path.join(output_rank_dir, "rank-index.json"), "w") as fh:
        json.dump(local_index, fh, indent=2)
    logger.info(f"[{os.path.basename(output_rank_dir)}] Wrote local rank-index.json "
                f"with {len(weight_map)} tensors, total_size={total_size} bytes.")
