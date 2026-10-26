import inspect
from contextlib import contextmanager
from enum import Enum
from functools import wraps
from typing import Any, Callable, Optional, Dict

import torch

try:
    import nvtx as _nvtx
except ImportError:
    _nvtx = None


class BackendType(str, Enum):
    NVTX = "nvtx"
    TORCH_NVTX = "torch_nvtx"
    RECORD_FUNCTION = "record_function"


class NvtxAnnotator:
    def __init__(self):
        self._stack: list[tuple[str, Any]] = []

    @property
    def available(self) -> bool:
        return _nvtx is not None

    @property
    def has_active_ranges(self) -> bool:
        return bool(self._stack)

    def push(self, msg: str) -> None:
        if not self.available:
            raise RuntimeError("nvtx backend is unavailable")
        ctx = _nvtx.annotate(message=msg)
        ctx.__enter__()
        self._stack.append((msg, ctx))

    def pop(self, msg: Optional[str] = None) -> None:
        if not self._stack:
            raise RuntimeError("Attempted to pop NVTX range from empty stack")
        last_msg, ctx = self._stack.pop()
        if msg is not None and msg != last_msg:
            raise ValueError(
                f"Attempted to pop NVTX range with msg={msg}, but last range has msg={last_msg}"
            )
        ctx.__exit__(None, None, None)


class TorchNvtxAnnotator:
    def __init__(self):
        self._stack: list[str] = []

    @property
    def available(self) -> bool:
        return torch.cuda.is_available() and hasattr(torch.cuda, "nvtx")

    @property
    def has_active_ranges(self) -> bool:
        return bool(self._stack)

    def push(self, msg: str) -> None:
        if not self.available:
            raise RuntimeError("torch.cuda.nvtx backend is unavailable")
        self._stack.append(msg)
        torch.cuda.nvtx.range_push(msg)

    def pop(self, msg: Optional[str] = None) -> None:
        if not self._stack:
            raise RuntimeError("Attempted to pop NVTX range from empty stack")
        last_msg = self._stack.pop()
        if msg is not None and msg != last_msg:
            raise ValueError(
                f"Attempted to pop NVTX range with msg={msg}, but last range has msg={last_msg}"
            )
        torch.cuda.nvtx.range_pop()


class RecordFunctionAnnotator:
    def __init__(self):
        self._msg_stack: list[str] = []
        self._ctx_stack: list[Any] = []

    @property
    def available(self) -> bool:
        return hasattr(torch, "profiler") and hasattr(torch.profiler, "record_function")

    @property
    def has_active_ranges(self) -> bool:
        return bool(self._msg_stack)

    def push(self, msg: str) -> None:
        if not self.available:
            raise RuntimeError("record_function is unavailable")
        ctx = torch.profiler.record_function(msg)
        ctx.__enter__()
        self._msg_stack.append(msg)
        self._ctx_stack.append(ctx)

    def pop(self, msg: Optional[str] = None) -> None:
        if not self._msg_stack:
            raise RuntimeError("Attempted to pop record_function range from empty stack")
        last_msg = self._msg_stack.pop()
        ctx = self._ctx_stack.pop()
        if msg is not None and msg != last_msg:
            raise ValueError(
                f"Attempted to pop record_function range with msg={msg}, but last range has msg={last_msg}"
            )
        ctx.__exit__(None, None, None)


class ProfilerAnnotation:
    _instance: Optional["ProfilerAnnotation"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        config_enabled: bool = False,
        backend: BackendType = BackendType.RECORD_FUNCTION,
    ):
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.config_enabled = config_enabled
        self._runtime_enabled = False
        self.backend = backend

        self.annotators = {
            BackendType.NVTX: NvtxAnnotator(),
            BackendType.TORCH_NVTX: TorchNvtxAnnotator(),
            BackendType.RECORD_FUNCTION: RecordFunctionAnnotator(),
        }
        self._initialized = True

    @property
    def enabled(self) -> bool:
        return self.config_enabled and self._runtime_enabled

    def enable_runtime(self) -> None:
        self._runtime_enabled = True

    def disable_runtime(self) -> None:
        self._runtime_enabled = False

    def configure(
        self,
        backend: BackendType,
        config_enabled: Optional[bool] = False,
    ) -> None:
        if not config_enabled:
            self.config_enabled = config_enabled
            return
        self._validate_backend(backend)
        self.backend = backend
        self.config_enabled = config_enabled

    def _validate_backend(self, backend: BackendType) -> None:
        if backend not in self.annotators.keys() or not self.annotators[backend].available:
            raise RuntimeError(f"Requested annotator backend `{backend}` is unavailable")
    
    def push(self, msg: str) -> None:
        if not self.enabled: return
        self.annotators[self.backend].push(msg=msg)

    def pop(self, msg: Optional[str] = None) -> None:
        if not self.enabled: return
        self.annotators[self.backend].pop(msg=msg)

    @contextmanager
    def annotate(self, message: str):
        self.push(msg=message)
        try:
            yield
        finally:
            self.pop(msg=message)

    def decorator(self, message: Optional[str] = None):
        def decorator_impl(func: Callable) -> Callable:
            @wraps(func)
            def wrapped(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)
                msg = message or self._decorator_get_func_path(func)
                with self.annotate(message=msg):
                    return func(*args, **kwargs)

            return wrapped
        return decorator_impl

    @staticmethod
    def _decorator_get_func_path(func: Callable) -> str:
        caller_func = func.__name__
        module = inspect.getmodule(func)
        return f"{module.__name__}.{caller_func}"


profiler_annotation = ProfilerAnnotation()


def decorator_forward_backward(message: Optional[str] = None):
    """Decorate ``nn.Module.forward`` and annotate both forward and module backward."""

    def decorator_impl(func: Callable) -> Callable:
        base_message = message or profiler_annotation._decorator_get_func_path(func)
        fwd_wrapped = profiler_annotation.decorator(message=f"{base_message}.forward")(func)

        @wraps(func)
        def wrapped(self, *args, **kwargs):
            if profiler_annotation.enabled:
                handles_attr = "_profiler_annotation_backward_handles_by_message"
                handles_by_message: Dict[str, Any] = getattr(self, handles_attr, {})

                if base_message not in handles_by_message:
                    if (
                        not hasattr(self, "register_full_backward_pre_hook")
                        or not hasattr(self, "register_full_backward_hook")
                    ):
                        raise TypeError(
                            "decorator_forward_backward can be used only on nn.Module.forward methods"
                        )

                    bwd_msg = f"{base_message}.backward"

                    def _backward_pre_hook(_module, _grad_output):
                        profiler_annotation.push(msg=bwd_msg)

                    def _backward_post_hook(_module, _grad_input, _grad_output):
                        profiler_annotation.pop(msg=bwd_msg)

                    handles_by_message[base_message] = (
                        self.register_full_backward_pre_hook(_backward_pre_hook),
                        self.register_full_backward_hook(_backward_post_hook),
                    )
                    setattr(self, handles_attr, handles_by_message)

            return fwd_wrapped(self, *args, **kwargs)

        return wrapped

    return decorator_impl
