"""Capture one real rank-zero microbatch in a standalone native diagnostic."""

from __future__ import annotations

from contextlib import ExitStack
import hashlib
import os
from pathlib import Path
import subprocess
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[2]


def _snapshot(tensor):
    """Preserve logical bytes and strides without retaining the GPU allocation."""
    source = tensor.detach()
    dtype = source.dtype
    if dtype == torch.float8_e4m3fn:
        source = source.view(torch.uint8)
    result = torch.empty_strided(
        source.shape, source.stride(), dtype=source.dtype, device="cpu"
    )
    result.copy_(source)
    return result.view(dtype) if source.dtype != dtype else result


def _layout(tensor):
    return {
        "shape": tuple(tensor.shape),
        "stride": tensor.stride(),
        "dtype": str(tensor.dtype),
        "storage_offset": tensor.storage_offset(),
        "address_mod_256": tensor.data_ptr() % 256,
    }


class _ExpertCapture:
    def __init__(self):
        self.active = False
        self.pending = {}
        self.projections = {}

    def install(self):
        from torchtitan.ops.float8 import _cudacherry as cudacherry
        from torchtitan.ops.float8 import direct_expert_dw as direct

        original = cudacherry.requantize_rowwise_to_columnwise

        @torch.library.custom_op(
            "gift_backward_capture::row2col",
            mutates_args=(),
            device_types="cuda",
            schema="(Tensor data, Tensor m_indices, Tensor sizes, Tensor scales) "
            "-> (Tensor, Tensor)",
        )
        def row2col(data, m_indices, sizes, scales):
            result = original(data, m_indices, sizes, scales)
            if self.active and len(self.projections) < 2:
                self.pending[self.operand_key(*result)] = {
                    "data": _snapshot(data),
                    "scales": _snapshot(scales),
                    "data_layout": _layout(data),
                    "scales_layout": _layout(scales),
                    "m_indices": _snapshot(m_indices),
                    "sizes": _snapshot(sizes),
                }
            return result

        @row2col.register_fake
        def fake_row2col(data, m_indices, sizes, scales):
            return original(data, m_indices, sizes, scales)

        @direct.write_direct_expert_dw.register_kernel("cuda")
        def write(
            a_data, a_scale, b_data, b_scale, slot, offset, ks, ks_cpu, ks_tensor
        ):
            if self.active and len(self.projections) < 2:
                lease = direct._get_lease(slot)
                w13_offset, w2_offset = lease._offsets
                if offset == w13_offset:
                    role, shape = "w13", direct._W13_SHAPE
                elif offset == w2_offset:
                    role, shape = "w2", direct._W2_SHAPE
                else:
                    raise RuntimeError(f"Unexpected expert gradient offset {offset}")
                self.collect_projection(
                    role,
                    shape,
                    offset,
                    slot,
                    a_data,
                    a_scale,
                    b_data,
                    b_scale,
                    ks_tensor,
                )
            direct._write_direct_expert_dw(
                a_data, a_scale, b_data, b_scale, slot, offset, ks, ks_cpu, ks_tensor
            )

        # Warmup imports this same callable. The runtime tap remains inactive
        # until the trainer enters the first real forward/backward call.
        cudacherry.requantize_rowwise_to_columnwise = row2col

    @staticmethod
    def operand_key(data, scales):
        # The direct writer reinterprets E4M3 data and transposes scale views.
        return (data.data_ptr(), scales.data_ptr())

    def collect_projection(
        self, role, shape, offset, slot, a_data, a_scale, b_data, b_scale, sizes
    ):
        if role in self.projections:
            raise RuntimeError(
                "First native expert group did not write both projections"
            )
        x = self.pending.pop(self.operand_key(b_data, b_scale), None)
        dy = self.pending.pop(self.operand_key(a_data, a_scale), None)
        if x is None or dy is None:
            raise RuntimeError(
                f"Cannot match the first direct {role} write to real CudaCherry operands"
            )
        self.projections[role] = {
            "x_data": x["data"],
            "x_scales": x["scales"],
            "dy_data": dy["data"],
            "dy_scales": dy["scales"],
            "aligned_sizes": _snapshot(sizes),
            "weight_shape": tuple(shape),
            "direct_dw_offset": offset,
            "slot_numel": slot.numel(),
            "storage_offsets": {
                "x_data": x["data_layout"]["storage_offset"],
                "x_scales": x["scales_layout"]["storage_offset"],
                "dy_data": dy["data_layout"]["storage_offset"],
                "dy_scales": dy["scales_layout"]["storage_offset"],
            },
            "address_mod_256": {
                "x_data": x["data_layout"]["address_mod_256"],
                "x_scales": x["scales_layout"]["address_mod_256"],
                "dy_data": dy["data_layout"]["address_mod_256"],
                "dy_scales": dy["scales_layout"]["address_mod_256"],
            },
            "m_indices": x["m_indices"],
        }
        if not (
            torch.equal(x["m_indices"], dy["m_indices"])
            and torch.equal(x["sizes"], dy["sizes"])
            and torch.equal(x["sizes"], self.projections[role]["aligned_sizes"])
        ):
            raise RuntimeError(
                f"CudaCherry operands and direct {role} dW used different routing"
            )
        if len(self.projections) == 2:
            first, second = self.projections["w13"], self.projections["w2"]
            if not all(
                torch.equal(first[k], second[k]) for k in ("m_indices", "aligned_sizes")
            ):
                raise RuntimeError("Captured expert projections used different routing")
            self.pending.clear()


def _metadata(trainer, head):
    config = trainer.config
    directory = Path(config.dump_folder)
    resolved = directory / "config/resolved_config.yaml"
    return {
        "rank": 0,
        "world_size": int(os.environ["WORLD_SIZE"]),
        "update": trainer.step,
        "microbatch": 1,
        "gradient_accumulation": trainer.gradient_accumulation_steps,
        "source_revision": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "capture_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "resolved_config": str(resolved),
        "resolved_config_sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        "cuda_device": torch.cuda.get_device_name(),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "compile_backend": config.compile.backend,
        "num_sms": config.fp8.sparse_deep_gemm_num_sms,
        "lm_head_class": f"{type(head).__module__}.{type(head).__qualname__}",
        "lm_head_bias": head.bias is not None,
        "lm_head_compiled_in_native": callable(
            getattr(head, "_compiled_call_impl", None)
        ),
        "diagnostic_only": True,
    }


def install_capture(output: Path):
    """Install on rank zero after setting up the production rank and CUDA device."""
    if int(os.environ["RANK"]) != 0:
        return

    from torchtitan.components.loss import IGNORE_INDEX, MTPLoss
    from torchtitan.trainer import Trainer

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fixture = output / "rank0.pt"
    if fixture.exists():
        raise FileExistsError(fixture)

    experts = _ExpertCapture()
    experts.install()
    native_forward_backward = Trainer.forward_backward_step

    def first_forward_backward(trainer, *args, **kwargs):
        Trainer.forward_backward_step = native_forward_backward
        loss = trainer.loss_fn
        if not isinstance(loss, MTPLoss) or loss.lm_head is None:
            raise RuntimeError("Expected the native chunked MTP loss and LM head")
        if trainer.gradient_accumulation_steps != 32 or trainer.step != 1:
            raise RuntimeError(
                "Capture requires the first native update with accumulation 32"
            )

        head = loss.lm_head
        chunks = []
        chunk_layouts = []
        ce = {
            "head_weights": [1.0, *loss.loss_weights],
            "n_chunks": loss.num_chunks,
            "ignore_index": IGNORE_INDEX,
        }
        native_loss = type(loss).__call__

        def capture_loss(
            instance, pred, packed, denominator=None, *loss_args, **loss_kwargs
        ):
            if instance is not loss or ce.get("labels") is not None:
                raise RuntimeError(
                    "Expected one native MTP loss call in the microbatch"
                )
            labels, positions = (
                (packed[:, 0], packed[:, 1]) if packed.ndim == 3 else (packed, None)
            )
            ce["labels"] = _snapshot(labels)
            ce["positions"] = None if positions is None else _snapshot(positions)
            ce["denominator"] = (
                _snapshot(denominator)
                if isinstance(denominator, torch.Tensor)
                else denominator
            )
            return native_loss(
                instance, pred, packed, denominator, *loss_args, **loss_kwargs
            )

        def capture_head(module, inputs):
            (hidden,) = inputs
            chunk_layouts.append(_layout(hidden))
            chunks.append(_snapshot(hidden))
            if "weight" not in ce:
                weight = module.weight
                if (
                    weight.shape != (128256, 4096)
                    or weight.dtype != torch.bfloat16
                    or hasattr(weight, "to_local")
                ):
                    raise RuntimeError(
                        "The GFSDP LM head must expose its full BF16 weight"
                    )
                ce["weight_layout"] = _layout(weight)
                ce["weight"] = _snapshot(weight)

        experts.active = True
        try:
            with ExitStack() as cleanup:
                cleanup.enter_context(
                    patch.object(type(loss), "__call__", capture_loss)
                )
                cleanup.callback(head.register_forward_pre_hook(capture_head).remove)
                result = native_forward_backward(trainer, *args, **kwargs)
        finally:
            experts.active = False

        if (
            len(chunks) != loss.num_chunks
            or "weight" not in ce
            or ce.get("denominator") is None
            or set(experts.projections) != {"w13", "w2"}
        ):
            raise RuntimeError(
                "First native microbatch did not capture both components"
            )
        ce["hidden_chunks_layout"] = chunk_layouts
        ce["hidden"] = torch.cat(chunks, dim=1).unflatten(
            0, (len(ce["head_weights"]), -1)
        )
        if (
            ce["hidden"].shape != (2, 2, 4096, 4096)
            or ce["hidden"].dtype != torch.bfloat16
        ):
            raise RuntimeError(
                "Expected both native BF16 heads with shape B2 S4096 H4096"
            )

        bundle = {
            "schema": 1,
            "metadata": _metadata(trainer, head),
            "ce": ce,
            "row2col": experts.projections,
        }
        temporary = fixture.with_suffix(".tmp")
        with temporary.open("xb") as writer:
            torch.save(bundle, writer)
        temporary.rename(fixture)
        print(f"backward_components_native_capture={fixture}", flush=True)
        experts.projections = {}
        return result

    Trainer.forward_backward_step = first_forward_backward
