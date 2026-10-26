"""Import blocker for the torch-free tests.

``run_blocked`` runs code in a fresh interpreter that refuses heavy runtime packages,
so a torch import fails even when the test process has torch loaded.
"""

import importlib.abc
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

BLOCKED = (
    "torch",
    "triton",
    "transformers",
    "datasets",
    "tokenizers",
    "fla",
    "deep_ep",
    "streaming",
    "torchft",
    "vllm",
    "monarch",
)


class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in BLOCKED:
            raise ModuleNotFoundError(
                f"{fullname} is blocked: this code must be torch-free", name=fullname
            )
        return None


def install() -> None:
    sys.meta_path.insert(0, Blocker())


def run_blocked(code: str, *, check: bool = True) -> subprocess.CompletedProcess:
    """Run *code* under the blocker in a subprocess started from the repo root."""
    prologue = "from tests.unit_tests.torch_free import install; install()\n"
    return subprocess.run(
        [sys.executable, "-c", prologue + code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=check,
    )
