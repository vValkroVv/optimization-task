from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import ModuleType

import pytest

from torchtitan.experiments.cuda_graph_campaign import backend_scan_fix


def _write_source_tree(root: Path, *, header: bytes | None = None) -> Path:
    root.mkdir(parents=True)
    wrapper = root / "hybrid_ep_buffer.py"
    wrapper.write_text(
        "import os\n"
        "import scan_fix_native\n"
        "class HybridEPBuffer:\n"
        "    def __init__(self):\n"
        "        self.native = scan_fix_native.HybridEPBuffer(\n"
        "            os.path.dirname(os.path.abspath(__file__))\n"
        "        )\n"
    )
    backend = root / "backend"
    backend.mkdir()
    target_header = backend / "hybrid_ep_backend.cuh"
    target_header.write_bytes(
        header
        if header is not None
        else b"int32_t thread_local_all_block_local_experts_sum[NUM_OF_EXPERTS_PER_RANK - 1];\n"
    )
    (backend / "NCCL_LICENSE.txt").write_text("license\n")
    (backend / "helper.cuh").write_text("unchanged support header\n")
    wrapper.chmod(0o640)
    target_header.chmod(0o640)
    return wrapper


def _allow_fixture_hashes(monkeypatch: pytest.MonkeyPatch, wrapper: Path) -> None:
    monkeypatch.setattr(
        backend_scan_fix,
        "_WRAPPER_SHA256",
        hashlib.sha256(wrapper.read_bytes()).hexdigest(),
    )
    header = wrapper.parent / "backend" / "hybrid_ep_backend.cuh"
    monkeypatch.setattr(
        backend_scan_fix,
        "_HEADER_SHA256",
        hashlib.sha256(header.read_bytes()).hexdigest(),
    )


def test_staging_relocates_wrapper_and_factory_passes_staged_base_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_wrapper = _write_source_tree(tmp_path / "installed")
    _allow_fixture_hashes(monkeypatch, source_wrapper)
    runtime_wrapper, cache_root, manifest_path = backend_scan_fix._stage_runtime_copy(
        source_wrapper, tmp_path / "attempt", rank=4
    )

    original_header = source_wrapper.parent / "backend" / "hybrid_ep_backend.cuh"
    runtime_header = runtime_wrapper.parent / "backend" / "hybrid_ep_backend.cuh"
    expected = original_header.read_bytes().replace(
        backend_scan_fix._OLD_BOUND, backend_scan_fix._NEW_BOUND, 1
    )
    assert runtime_header.read_bytes() == expected
    assert runtime_wrapper.read_bytes() == source_wrapper.read_bytes()
    assert stat.S_IMODE(runtime_header.stat().st_mode) == 0o640
    assert (
        runtime_wrapper.parent / "backend" / "NCCL_LICENSE.txt"
    ).read_text() == "license\n"

    manifest = json.loads(manifest_path.read_text())
    assert (
        manifest["patch"]["source_sha256"]
        == hashlib.sha256(original_header.read_bytes()).hexdigest()
    )
    assert manifest["patch"]["runtime_sha256"] == hashlib.sha256(expected).hexdigest()
    assert (
        manifest["source_backend_tree_sha256"]
        != manifest["runtime_backend_tree_sha256"]
    )
    source_support_files = {
        entry["path"]: (entry["sha256"], entry["mode"])
        for entry in manifest["source_backend_files"]
        if entry["path"] != "hybrid_ep_backend.cuh"
    }
    runtime_support_files = {
        entry["path"]: (entry["sha256"], entry["mode"])
        for entry in manifest["runtime_backend_files"]
        if entry["path"] != "hybrid_ep_backend.cuh"
    }
    assert runtime_support_files == source_support_files
    assert manifest["native_cache_root"] == str(cache_root)
    assert manifest["native_cache_path"].endswith(f"proc-{manifest['pid']}")
    assert not list(cache_root.iterdir())

    native_module = ModuleType("scan_fix_native")
    native_module.HybridEPBuffer = lambda base_path: base_path
    monkeypatch.setitem(sys.modules, "scan_fix_native", native_module)
    relocated_module_name = "deep_ep.cuda_graph_campaign_scan_fix_rank_4_pid_test"
    relocated_class_module, relocated_class = backend_scan_fix._load_relocated_class(
        runtime_wrapper, relocated_module_name
    )
    original_class = type("HybridEPBuffer", (), {})
    original_class.__module__ = "deep_ep.hybrid_ep_buffer"
    target_module = ModuleType("torchtitan.distributed.deepep.hybridep")
    target_module._buffer = None
    target_module.HybridEPBuffer = original_class

    package = ModuleType("deep_ep")
    package.HybridEPBuffer = original_class
    monkeypatch.setitem(sys.modules, "deep_ep", package)
    try:
        backend_scan_fix._replace_factory_class(
            target_module, original_class, relocated_class
        )
        instance = target_module.HybridEPBuffer()
        assert instance.native == str(runtime_wrapper.parent)
        assert target_module.HybridEPBuffer is relocated_class
        assert package.HybridEPBuffer is original_class
    finally:
        sys.modules.pop(relocated_module_name, None)
        sys.modules.pop(relocated_class_module.__name__, None)


def test_staging_rejects_unsupported_wrapper_hash_before_writing(
    tmp_path: Path,
) -> None:
    source_wrapper = _write_source_tree(tmp_path / "installed")
    with pytest.raises(ValueError, match="Unsupported DeepEP HybridEP wrapper"):
        backend_scan_fix._stage_runtime_copy(
            source_wrapper, tmp_path / "attempt", rank=0
        )
    assert not (tmp_path / "attempt").exists()


def test_staging_rejects_unsupported_header_hash_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_wrapper = _write_source_tree(tmp_path / "installed")
    monkeypatch.setattr(
        backend_scan_fix,
        "_WRAPPER_SHA256",
        hashlib.sha256(source_wrapper.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError, match="Unsupported DeepEP HybridEP backend header"):
        backend_scan_fix._stage_runtime_copy(
            source_wrapper, tmp_path / "attempt", rank=0
        )
    assert not (tmp_path / "attempt").exists()


def test_staging_rejects_duplicate_patch_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    duplicate = (
        backend_scan_fix._OLD_BOUND + b"\n" + backend_scan_fix._OLD_BOUND + b"\n"
    )
    source_wrapper = _write_source_tree(tmp_path / "installed", header=duplicate)
    _allow_fixture_hashes(monkeypatch, source_wrapper)
    with pytest.raises(ValueError, match="exactly one expected bound"):
        backend_scan_fix._stage_runtime_copy(
            source_wrapper, tmp_path / "attempt", rank=0
        )
    assert not (tmp_path / "attempt").exists()


def test_staging_rejects_rank_directory_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_wrapper = _write_source_tree(tmp_path / "installed")
    _allow_fixture_hashes(monkeypatch, source_wrapper)
    attempt_root = tmp_path / "attempt"
    backend_scan_fix._stage_runtime_copy(source_wrapper, attempt_root, rank=2)
    with pytest.raises(RuntimeError, match="use a fresh attempt root"):
        backend_scan_fix._stage_runtime_copy(source_wrapper, attempt_root, rank=2)


def test_factory_rejects_non_native_wrapper() -> None:
    original_class = type("HybridEPBuffer", (), {})
    original_class.__module__ = "unsupported.wrapper"
    relocated_class = type("HybridEPBuffer", (), {})
    target_module = ModuleType("torchtitan.distributed.deepep.hybridep")
    target_module._buffer = None
    target_module.HybridEPBuffer = original_class

    with pytest.raises(
        RuntimeError, match="expected the installed DeepEP native wrapper"
    ):
        backend_scan_fix._replace_factory_class(
            target_module, original_class, relocated_class
        )


def test_factory_rejects_replacement_after_native_buffer_creation() -> None:
    original_class = type("HybridEPBuffer", (), {})
    original_class.__module__ = "deep_ep.hybrid_ep_buffer"
    relocated_class = type("HybridEPBuffer", (), {})
    target_module = ModuleType("torchtitan.distributed.deepep.hybridep")
    target_module._buffer = object()
    target_module.HybridEPBuffer = original_class

    with pytest.raises(RuntimeError, match="before buffer construction"):
        backend_scan_fix._replace_factory_class(
            target_module, original_class, relocated_class
        )


def test_factory_rejects_changed_factory_class() -> None:
    original_class = type("HybridEPBuffer", (), {})
    original_class.__module__ = "deep_ep.hybrid_ep_buffer"
    replacement_class = type("HybridEPBuffer", (), {})
    relocated_class = type("HybridEPBuffer", (), {})
    target_module = ModuleType("torchtitan.distributed.deepep.hybridep")
    target_module._buffer = None
    target_module.HybridEPBuffer = replacement_class

    with pytest.raises(RuntimeError, match="factory changed"):
        backend_scan_fix._replace_factory_class(
            target_module, original_class, relocated_class
        )


def test_install_requires_runtime_root_before_importing_gift_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GIFT_HYBRID_EP_SCAN_RUNTIME_DIR", raising=False)

    with pytest.raises(ValueError, match="GIFT_HYBRID_EP_SCAN_RUNTIME_DIR"):
        backend_scan_fix.install_hybrid_ep_scan_fix()


def _prepare_install_transaction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    prior_cache: str | None,
) -> tuple[ModuleType, type, str, Path, ModuleType, ModuleType, Path]:
    target_module = ModuleType("torchtitan.distributed.deepep.hybridep")
    target_module._buffer = None
    target_module.logger = ModuleType("test_logger")
    original_class = type("HybridEPBuffer", (), {})
    original_class.__module__ = "deep_ep.hybrid_ep_buffer"
    target_module.HybridEPBuffer = original_class

    source_wrapper = tmp_path / "installed" / "hybrid_ep_buffer.py"
    source_wrapper.parent.mkdir()
    source_wrapper.touch()
    source_module = ModuleType("deep_ep.hybrid_ep_buffer")
    source_module.__file__ = str(source_wrapper)
    source_module.HybridEPBuffer = original_class
    deep_ep_package = ModuleType("deep_ep")
    deep_ep_package.__path__ = [str(source_wrapper.parent)]
    deep_ep_package.HybridEPBuffer = original_class
    monkeypatch.setitem(sys.modules, "deep_ep", deep_ep_package)
    monkeypatch.setitem(sys.modules, "deep_ep.hybrid_ep_buffer", source_module)
    monkeypatch.setitem(
        sys.modules, "torchtitan.distributed.deepep.hybridep", target_module
    )

    runtime_root = tmp_path / "attempt"
    manifest_path = tmp_path / "runtime_manifest.json"
    module_name = f"deep_ep.cuda_graph_campaign_scan_fix_test_{tmp_path.name}"
    manifest_path.write_text(
        json.dumps(
            {
                "wrapper_module": module_name,
                "patch": {"runtime_sha256": "test-sha256"},
            }
        )
    )
    cache_root = tmp_path / "new-cache"

    def staged_copy(
        wrapper_path: Path, attempt_root: Path, rank: int
    ) -> tuple[Path, Path, Path]:
        assert wrapper_path == source_wrapper
        assert attempt_root == runtime_root
        assert rank == 0
        return tmp_path / "runtime" / "hybrid_ep_buffer.py", cache_root, manifest_path

    monkeypatch.setattr(backend_scan_fix, "_stage_runtime_copy", staged_copy)
    monkeypatch.setenv("GIFT_HYBRID_EP_SCAN_RUNTIME_DIR", str(runtime_root))
    monkeypatch.setenv("RANK", "0")
    if prior_cache is None:
        monkeypatch.delenv("HYBRID_EP_CACHE_DIR", raising=False)
    else:
        monkeypatch.setenv("HYBRID_EP_CACHE_DIR", prior_cache)
    return (
        target_module,
        original_class,
        module_name,
        cache_root,
        deep_ep_package,
        source_module,
        manifest_path,
    )


@pytest.mark.parametrize("prior_cache", [None, "", "previous-cache"])
@pytest.mark.parametrize("failure_point", ["loader", "factory"])
def test_install_rolls_back_failed_setup_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prior_cache: str | None,
    failure_point: str,
) -> None:
    (
        target_module,
        original_class,
        module_name,
        cache_root,
        deep_ep_package,
        source_module,
        manifest_path,
    ) = _prepare_install_transaction(monkeypatch, tmp_path, prior_cache)

    if failure_point == "loader":

        def fail_after_module_load(
            _wrapper_path: Path, name: str
        ) -> tuple[ModuleType, type]:
            monkeypatch.setitem(sys.modules, name, ModuleType(name))
            raise ImportError("injected wrapper import failure")

        monkeypatch.setattr(
            backend_scan_fix, "_load_relocated_class", fail_after_module_load
        )
        expected_error = ImportError
    else:
        relocated_class = type("HybridEPBuffer", (), {})
        relocated_class.__module__ = module_name

        def load_relocated_class(
            _wrapper_path: Path, name: str
        ) -> tuple[ModuleType, type]:
            module = ModuleType(name)
            module.HybridEPBuffer = relocated_class
            monkeypatch.setitem(sys.modules, name, module)
            return module, relocated_class

        replace_factory = backend_scan_fix._replace_factory_class

        def replace_then_fail(
            module: ModuleType, original: type, relocated: type
        ) -> None:
            replace_factory(module, original, relocated)
            raise RuntimeError("injected failure after factory replacement")

        monkeypatch.setattr(
            backend_scan_fix, "_load_relocated_class", load_relocated_class
        )
        monkeypatch.setattr(
            backend_scan_fix, "_replace_factory_class", replace_then_fail
        )
        expected_error = RuntimeError

    with pytest.raises(expected_error):
        backend_scan_fix.install_hybrid_ep_scan_fix()

    assert target_module.HybridEPBuffer is original_class
    if prior_cache is None:
        assert "HYBRID_EP_CACHE_DIR" not in os.environ
    else:
        assert os.environ["HYBRID_EP_CACHE_DIR"] == prior_cache
    assert module_name not in sys.modules
    assert sys.modules["deep_ep"] is deep_ep_package
    assert sys.modules["deep_ep.hybrid_ep_buffer"] is source_module
    assert manifest_path.is_file()
    assert cache_root == tmp_path / "new-cache"


def test_scan_fix_marker_is_visible_before_native_training_starts(
    tmp_path: Path,
) -> None:
    gift_root = Path(__file__).parents[2]
    script = """
import importlib.util
import sys
import types
from pathlib import Path

spec = importlib.util.find_spec("deep_ep")
assert spec is not None and spec.origin is not None
installed_wrapper = Path(spec.origin).parent / "hybrid_ep_buffer.py"
deep_ep = types.ModuleType("deep_ep")
deep_ep.__path__ = [str(installed_wrapper.parent)]
original_class = type("HybridEPBuffer", (), {})
original_class.__module__ = "deep_ep.hybrid_ep_buffer"
deep_ep.HybridEPBuffer = original_class
source_module = types.ModuleType("deep_ep.hybrid_ep_buffer")
source_module.__file__ = str(installed_wrapper)
source_module.HybridEPBuffer = original_class
sys.modules["deep_ep"] = deep_ep
sys.modules["deep_ep.hybrid_ep_buffer"] = source_module

from torchtitan.tools.logging import logger
target = types.ModuleType("torchtitan.distributed.deepep.hybridep")
target._buffer = None
target.HybridEPBuffer = original_class
target.logger = logger
sys.modules[target.__name__] = target

native_train = types.ModuleType("torchtitan.train")
def native_main(argv=None):
    assert argv == ["campaign.yaml"]
    assert target.HybridEPBuffer is not original_class
    print("NATIVE_MAIN_AFTER_INSTALL")
native_train.main = native_main
sys.modules[native_train.__name__] = native_train

from torchtitan.experiments.cuda_graph_campaign.train_scan_fix import main
main(["campaign.yaml"])
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(gift_root), env.get("PYTHONPATH", "")])
    env["GIFT_HYBRID_EP_SCAN_RUNTIME_DIR"] = str(tmp_path / "runtime")
    env["RANK"] = "0"
    env["GIFT_LOG_RANK"] = "all"
    env.pop("HYBRID_EP_CACHE_DIR", None)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=gift_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    activation = "Campaign HybridEP fused-scan bound fix active: rank=0"
    assert activation in result.stdout
    assert result.stdout.index(activation) < result.stdout.index(
        "NATIVE_MAIN_AFTER_INSTALL"
    )


def test_replaced_factory_is_used_by_gift_get_buffer_without_cuda() -> None:
    gift_root = Path(__file__).parents[2]
    script = """
import sys
import types
import torch

native_class = type("HybridEPBuffer", (), {})
native_class.__module__ = "deep_ep.hybrid_ep_buffer"
deep_ep = types.ModuleType("deep_ep")
deep_ep.HybridEPBuffer = native_class
sys.modules["deep_ep"] = deep_ep

from torchtitan.distributed.deepep import hybridep
from torchtitan.experiments.cuda_graph_campaign.backend_scan_fix import (
    _replace_factory_class,
)

calls = []
class RelocatedBuffer:
    def __init__(self, **kwargs):
        calls.append(kwargs)

class Group:
    def size(self):
        return 8

hybridep._buffer = None
_replace_factory_class(hybridep, native_class, RelocatedBuffer)
hybridep.get_buffer(Group(), hidden_dim=128, num_tokens=1, num_local_experts=4)
assert len(calls) == 1
assert calls[0]["max_num_of_tokens_per_rank"] == 64
assert calls[0]["load_cached_kernels"] is True
assert isinstance(hybridep._buffer, RelocatedBuffer)
assert not torch.cuda.is_initialized()
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(gift_root), env.get("PYTHONPATH", "")])
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=gift_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_train_entry_activates_fix_then_forwards_argv_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from torchtitan.experiments.cuda_graph_campaign import train_scan_fix

    events: list[object] = []
    argv = ["configs/campaign.yaml", "--training.steps", "10"]
    monkeypatch.setattr(
        train_scan_fix, "init_logger", lambda: events.append("init_logger")
    )
    monkeypatch.setattr(
        train_scan_fix, "install_hybrid_ep_scan_fix", lambda: events.append("install")
    )
    native_train = ModuleType("torchtitan.train")

    def native_main(received_argv: list[str] | None = None) -> None:
        events.append(received_argv)

    native_train.main = native_main
    monkeypatch.setitem(sys.modules, "torchtitan.train", native_train)
    train_scan_fix.main(argv)

    assert events == ["init_logger", "install", argv]
    assert events[2] is argv
