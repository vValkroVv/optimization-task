"""Fixtures shared by unit tests, plus the one import-time environment shim."""

import atexit
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import types

import pytest

# triton ships only on Linux and the MoE kernel imports reach it, so stub it to
# keep the config-level tests importable on macOS/Windows. Only when it is
# *genuinely* absent: stubbing over a real triton turns every kernel test into a
# silent skip. At module level because pytest imports conftest before any test
# module, which is the only point early enough to matter.
#
# A meta-path finder, not fixed sys.modules entries: once ``import triton``
# succeeds, torch imports arbitrary ``triton.*`` submodules and uses some
# attributes as types in isinstance checks (e.g. flop_counter's ``JITFunction``).
# So any ``triton.*`` name gets a module, CamelCase attributes get empty classes
# and the rest get pass-through decorators.


class _TritonStub(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "triton" or fullname.startswith("triton."):
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        module = types.ModuleType(spec.name)
        module.__path__ = []
        module.__version__ = "0.0.0"  # pyrefly: ignore [missing-attribute]
        module.__getattr__ = self._make_attr  # PEP 562, per-module
        return module

    def exec_module(self, module):
        pass

    @staticmethod
    def _make_attr(name):
        if name.startswith("__"):
            raise AttributeError(name)
        if name[0].isupper():
            # Used in isinstance checks and constructed with arbitrary args.
            return type(name, (), {"__init__": lambda self, *a, **k: None})
        return _AnyStub()


class _AnyStub:
    """Stand-in for lowercase triton attributes: pass-through decorator, callable,
    and every attribute is another stub."""

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return self

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _TritonStub._make_attr(name)


if importlib.util.find_spec("triton") is None and "triton" not in sys.modules:
    sys.meta_path.append(_TritonStub())


@pytest.fixture(autouse=True)
def _clean_streaming_shared_memory(request, monkeypatch):
    """Release shared memory between streaming test cases"""
    if not request.node.path.name.startswith("test_streaming_"):
        yield
        return

    from streaming.base.shared.memory import SharedMemory

    shared_memory_objects = []
    original_init = SharedMemory.__init__

    def tracking_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        shared_memory_objects.append(self)

    monkeypatch.setattr(SharedMemory, "__init__", tracking_init)
    try:
        yield
    finally:
        for shared_memory in reversed(shared_memory_objects):
            shared_memory.cleanup()
            atexit.unregister(shared_memory.cleanup)
