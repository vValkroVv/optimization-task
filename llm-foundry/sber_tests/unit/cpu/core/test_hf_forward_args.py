import functools
import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]


def _load_module(module_name: str, file_path: Path):
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module {module_name} from {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _load_model_wrapper_module():
    for contrib_path in (
        REPO_ROOT / "contrib/composer",
        REPO_ROOT / "contrib/streaming",
        REPO_ROOT / "contrib/gigafsdp",
    ):
        contrib_path_str = str(contrib_path)
        if contrib_path_str not in sys.path:
            sys.path.insert(0, contrib_path_str)

    llmfoundry_pkg = sys.modules.setdefault("llmfoundry", types.ModuleType("llmfoundry"))
    llmfoundry_pkg.__path__ = [str(REPO_ROOT / "llmfoundry")]

    models_pkg = sys.modules.setdefault("llmfoundry.models", types.ModuleType("llmfoundry.models"))
    models_pkg.__path__ = [str(REPO_ROOT / "llmfoundry/models")]

    hf_pkg = sys.modules.setdefault("llmfoundry.models.hf", types.ModuleType("llmfoundry.models.hf"))
    hf_pkg.__path__ = [str(REPO_ROOT / "llmfoundry/models/hf")]

    _load_module("llmfoundry.models.hf.hf_fsdp", REPO_ROOT / "llmfoundry/models/hf/hf_fsdp.py")
    return _load_module(
        "llmfoundry.models.hf.model_wrapper_for_test",
        REPO_ROOT / "llmfoundry/models/hf/model_wrapper.py",
    )


class KwOnlyForwardModel:
    def forward(
        self,
        *,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs,
    ):
        del kwargs
        return input_ids, attention_mask, labels


class MixedForwardModel:
    def forward(self, batch, *, labels=None, **kwargs):
        del kwargs
        return batch, labels


def passthrough(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper


class WrappedKwOnlyForwardModel:
    @passthrough
    def forward(self, *, input_ids=None, labels=None):
        return input_ids, labels


class TestHfForwardArgs:
    @pytest.mark.unit
    @pytest.mark.fast
    def test_kwonly_forward_args_are_preserved(self):
        model_wrapper = _load_model_wrapper_module()
        arg_names = model_wrapper._get_model_forward_args(KwOnlyForwardModel().forward)

        assert "input_ids" in arg_names
        assert "attention_mask" in arg_names
        assert "labels" in arg_names

    @pytest.mark.unit
    @pytest.mark.fast
    def test_positional_and_kwonly_args_are_combined(self):
        model_wrapper = _load_model_wrapper_module()
        arg_names = model_wrapper._get_model_forward_args(MixedForwardModel().forward)

        assert "batch" in arg_names
        assert "labels" in arg_names

    @pytest.mark.unit
    @pytest.mark.fast
    def test_wrapped_kwonly_forward_args_are_preserved(self):
        model_wrapper = _load_model_wrapper_module()
        arg_names = model_wrapper._get_model_forward_args(WrappedKwOnlyForwardModel().forward)

        assert "input_ids" in arg_names
        assert "labels" in arg_names
