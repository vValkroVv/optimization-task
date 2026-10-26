from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

try:
    from utils.file_utils import _atomic_write_json
except BaseException:
    current_file = Path(__file__).resolve().parent
    import sys
    sys.path.insert(0, str(current_file))
    from file_utils import _atomic_write_json


HF_ASSEMBLY_MANIFEST = "hf_assembly_plan.json"


def write_hf_assembly_manifest(output_dir: str, args, *, model_family: str | None = None) -> None:
    manifest = {
        "target": "hf_checkpoint",
        "tokenizer_path": getattr(args, "tokenizer_path", None),
        "torch_dtype": getattr(args, "torch_dtype", None),
        "input_checkpoint_path": getattr(args, "input_checkpoint_path", None),
    }
    if model_family is not None:
        manifest["model_family"] = model_family
    _atomic_write_json(os.path.join(output_dir, HF_ASSEMBLY_MANIFEST), manifest)


def load_hf_assembly_manifest(checkpoint_dir: str) -> dict[str, Any]:
    manifest_path = Path(checkpoint_dir) / HF_ASSEMBLY_MANIFEST
    if not manifest_path.is_file():
        return {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def has_hf_assembly_manifest(checkpoint_dir: str) -> bool:
    manifest = load_hf_assembly_manifest(checkpoint_dir)
    return manifest.get("target") == "hf_checkpoint"
