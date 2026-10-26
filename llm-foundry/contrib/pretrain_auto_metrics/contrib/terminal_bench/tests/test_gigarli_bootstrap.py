from __future__ import annotations

import importlib
import sys
from pathlib import Path

from terminal_bench.src.gigarli_bootstrap import ensure_gigarli_import_path


def test_bootstrap_adds_sibling_gigarli_repo_root(monkeypatch) -> None:
    contrib_root = Path(__file__).resolve().parents[2]
    sibling_src_root = contrib_root / "gigarli" / "src"

    filtered_path = [
        entry
        for entry in sys.path
        if Path(entry).resolve() != sibling_src_root.resolve()
    ]
    monkeypatch.setattr(sys, "path", filtered_path)

    for name in list(sys.modules):
        if name == "gigarli" or name.startswith("gigarli."):
            sys.modules.pop(name, None)

    ensure_gigarli_import_path()

    assert str(sibling_src_root) in sys.path
    module = importlib.import_module("gigarli.client")
    assert module.__name__ == "gigarli.client"


def test_bootstrap_is_idempotent(monkeypatch) -> None:
    contrib_root = Path(__file__).resolve().parents[2]
    sibling_src_root = contrib_root / "gigarli" / "src"
    path_str = str(sibling_src_root)

    monkeypatch.setattr(sys, "path", [path_str, *sys.path])

    before = list(sys.path)
    ensure_gigarli_import_path()

    assert sys.path == before
