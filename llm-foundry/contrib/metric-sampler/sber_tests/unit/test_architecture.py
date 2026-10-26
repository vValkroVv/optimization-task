from __future__ import annotations

import ast
from pathlib import Path

import pytest

import metric_sampler

pytestmark = pytest.mark.unit

# Layer -> the first-party packages and modules its modules may import.
_ALLOWED = {
    "metric_sampler.contract": ("metric_sampler.contract",),
    "metric_sampler.capture.kernels": ("metric_sampler.contract", "metric_sampler.capture.kernels"),
    "metric_sampler.capture": ("metric_sampler.contract", "metric_sampler.capture"),
    "metric_sampler.node": ("metric_sampler.contract", "metric_sampler.capture", "metric_sampler.node"),
    "metric_sampler.island": ("metric_sampler.contract", "metric_sampler.island"),
    "metric_sampler.cloud": ("metric_sampler.cloud",),
    "metric_sampler.logged": ("metric_sampler.contract",),
    "metric_sampler.rank_capture": ("metric_sampler.contract", "metric_sampler.capture", "metric_sampler.node"),
}
# The stages that run off the GPU.
_WITHOUT_TORCH = ("metric_sampler.contract", "metric_sampler.node", "metric_sampler.island", "metric_sampler.cloud",
                  "metric_sampler.logged")
_PACKAGE_ROOT = Path(metric_sampler.__file__).parent


def _within(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def _modules() -> dict[str, set[str]]:
    """Every module of the package with every module name it imports, lazy imports included."""
    modules = {}
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        parts = path.relative_to(_PACKAGE_ROOT.parent).with_suffix("").parts
        name = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        imported = set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                assert node.level == 0, f"{name} uses a relative import"
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
        modules[name] = imported
    return modules


def _layer(module: str) -> str | None:
    matches = [layer for layer in _ALLOWED if _within(module, layer)]
    return max(matches, key=len) if matches else None


def test_every_module_belongs_to_a_layer() -> None:
    unplaced = [module for module in _modules() if module != "metric_sampler" and _layer(module) is None]

    assert unplaced == []


def test_every_layer_imports_only_the_layers_below_it() -> None:
    violations = []
    for module, imported in _modules().items():
        layer = _layer(module)
        if layer is None:
            continue
        for name in imported:
            if _within(name, "metric_sampler") and not any(_within(name, allowed) for allowed in _ALLOWED[layer]):
                violations.append(f"{module} imports {name}")

    assert violations == []


def test_the_stages_off_the_gpu_never_import_torch() -> None:
    violations = [
        f"{module} imports {name}"
        for module, imported in _modules().items()
        if any(_within(module, prefix) for prefix in _WITHOUT_TORCH)
        for name in imported
        if _within(name, "torch") or _within(name, "triton")
    ]

    assert violations == []
