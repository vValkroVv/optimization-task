"""Standalone vendor CE kernels under the immediate two-head GIFT VJP contract."""

import hashlib
from importlib import import_module, metadata
from pathlib import Path

import torch


IGNORE_INDEX = -100
LIGER_REVISION = "42dc23c41dcaf1f95d858837aed5301fd4faa27a"
VENDOR_VARIANTS = ("liger", "liger-fp32", "cute", "cute-fp32", "fla", "fla-centered")


class VendorUnavailable(RuntimeError):
    """The requested vendor module or runtime is unavailable on this host."""


def _import_vendor(name):
    try:
        return import_module(name)
    except (ImportError, OSError) as error:
        raise VendorUnavailable(f"Cannot load {name}: {error}") from error


def _source(module):
    path = Path(module.__file__).resolve()
    return {"module": module.__name__, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _version(distribution):
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _head_sums(losses):
    return tuple(head.sum(dtype=torch.float32) for head in losses.chunk(2))


class _LigerCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, kernel, fp32_scratch):
        flat = logits.detach().view(-1, logits.size(-1))
        scratch = flat.float() if fp32_scratch else flat
        # Explicit SUM mode leaves both denominator and per-head weights to
        # the caller. The vendor primitive exposes FP32 row losses in either dtype.
        losses, _, _, _, unscaled = kernel(
            scratch, labels.view(-1), None, IGNORE_INDEX, 0.0, 0.0, "sum", None,
            0, 0.0, 0.0, False, False, False, True,
        )
        if unscaled.data_ptr() != scratch.data_ptr() or losses.dtype != torch.float32:
            raise RuntimeError("Unexpected vendor CE loss buffer or scratch layout")
        if not fp32_scratch:
            torch.autograd.graph.increment_version(logits)
        ctx.save_for_backward(logits, unscaled)
        return _head_sums(losses)

    @staticmethod
    def backward(ctx, main_scale, mtp_scale):
        if torch.is_grad_enabled():
            raise RuntimeError("Vendor CE supports immediate first-order backward only")
        logits, unscaled = ctx.saved_tensors
        # Consume the buffer once. The FP32 scratch rows enter this multiply
        # before their first BF16 cast, using the actual upstream autograd scales.
        row_scales = torch.stack((main_scale, mtp_scale)).repeat_interleave(unscaled.size(0) // 2)
        torch.mul(unscaled, row_scales[:, None], out=logits.view_as(unscaled))
        return logits, None, None, None


def make_vendor_ce(name):
    """Return a callable that consumes logits and returns two raw FP32 sums.

    The caller applies weights and normalization and immediately runs first-order
    backward. The projected logits may be overwritten. Labels must have been
    validated by the dataset or fixture loader.
    """
    if name not in VENDOR_VARIANTS:
        raise ValueError(f"Unknown vendor CE variant {name!r}")

    if name.startswith(("liger", "cute")):
        cute = name.startswith("cute")
        if cute and (not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0)):
            raise VendorUnavailable("Liger CuTe DSL CE requires CUDA compute capability 9.0 or newer")
        backend = "cutedsl" if cute else "triton"
        adapter = _import_vendor(f"liger_kernel.ops.backends._{backend}.cross_entropy")
        kernel_module = _import_vendor(
            "liger_kernel.ops.cutedsl.ops.cross_entropy" if cute else "liger_kernel.ops.cross_entropy"
        )
        kernel = getattr(adapter, f"cross_entropy_loss_and_grad_{backend}", None)
        if kernel is None:
            raise VendorUnavailable(f"Liger {backend} CE primitive is missing at {adapter.__file__}")
        fp32_scratch = name.endswith("-fp32")

        def run(logits, labels):
            if cute and (logits.size(-1) % 8 or logits.data_ptr() % 16):
                raise ValueError("Liger CuTe CE requires 16-byte-aligned rows and a vocab divisible by eight")
            return _LigerCE.apply(logits, labels, kernel, fp32_scratch)

        receipt = {
            "backend": f"Liger {backend}",
            "api_reference_revision": LIGER_REVISION,
            "distribution_version": _version("liger-kernel"),
            "sources": [_source(adapter), _source(kernel_module)],
            "unscaled_dz_dtype": "float32" if fp32_scratch else "bfloat16",
            "late_scale": "actual FP32 row VJP times vendor gradient, single aten::mul into consumed BF16 logits",
        }
        if cute:
            receipt.update(
                cutlass_dsl_version=_version("nvidia-cutlass-dsl"),
                tvm_ffi_version=_version("apache-tvm-ffi"),
            )
    else:
        module = _import_vendor("fla.modules.fused_cross_entropy")
        centered = name.endswith("-centered")

        def run(logits, labels):
            flat = logits.view(-1, logits.size(-1))
            if centered:
                # A detached shift preserves the analytic CE VJP. Keep the
                # centered values in FP32 so BF16 does not change their spacing.
                flat = flat.float()
                flat.sub_(flat.detach().amax(dim=1, keepdim=True))
            rows, _ = module.CrossEntropyLossFunction.apply(
                flat, labels.view(-1), 0.0, 1.0, 0.0, None, IGNORE_INDEX, True, None,
            )
            return _head_sums(rows)

        receipt = {
            "backend": "FLA standalone Triton CE",
            "distribution_version": _version("flash-linear-attention"),
            "sources": [_source(module)],
            "fp32_centered_logits": centered,
            "late_scale": "FLA applies the actual FP32 row VJP within CE backward",
        }

    def ce(logits, labels, weights, denominator):
        del denominator
        if (
            len(weights) != 2
            or logits.ndim not in (2, 3)
            or not logits.size(0)
            or logits.size(0) % 2
            or logits.shape[:-1] != labels.shape
            or logits.dtype != torch.bfloat16
            or labels.dtype != torch.int64
            or not logits.is_cuda
            or labels.device != logits.device
            or not logits.is_contiguous()
            or not labels.is_contiguous()
            or not logits.requires_grad
        ):
            raise ValueError("Expected two contiguous folded CUDA BF16 heads and matching int64 labels")
        return run(logits, labels)

    ce.backend_receipt = receipt
    return ce
