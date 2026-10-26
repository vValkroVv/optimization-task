from __future__ import annotations

import importlib
import sys
from pathlib import Path


def ensure_gigarli_import_path() -> None:
    """Add the sibling `contrib/gigarli/src` path to `sys.path` if needed."""

    try:
        importlib.import_module("gigarli.client")
        return
    except ModuleNotFoundError:
        sys.modules.pop("gigarli", None)

    sibling_repo_root = Path(__file__).resolve().parents[2] / "gigarli"
    src_root = sibling_repo_root / "src"
    inner_package = src_root / "gigarli"
    if not src_root.is_dir() or not inner_package.is_dir():
        return

    path_str = str(src_root)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
        importlib.invalidate_caches()
