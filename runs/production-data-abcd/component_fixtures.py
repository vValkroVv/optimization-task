"""Diagnostic first-native-microbatch capture; never enable in a timing process."""
from __future__ import annotations

from contextlib import AbstractContextManager
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
ENV = "ABCD_COMPONENT_FIXTURE_DIR"


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def map_tensors(value, fn):
    import torch

    if isinstance(value, torch.Tensor):
        if hasattr(value, "to_local"):
            raise TypeError("component capture requires native local tensors, not DTensor")
        return fn(value)
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, dict):
        return {k: map_tensors(v, fn) for k, v in value.items()}
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        if type(value).__name__ != "VarlenMetadata":
            raise TypeError(f"unsupported fixture metadata {type(value)}")
        return {"__varlen_metadata__": [map_tensors(v, fn) for v in value]}
    if isinstance(value, (tuple, list)):
        return type(value)(map_tensors(v, fn) for v in value)
    raise TypeError(f"unsupported fixture metadata {type(value)}; do not silently drop masks")


def restore(value, device):
    import torch

    if isinstance(value, dict) and set(value) == {"__varlen_metadata__"}:
        from torchtitan.models.common.attention.varlen_meta import VarlenMetadata
        return VarlenMetadata(*(restore(v, device) for v in value["__varlen_metadata__"]))
    if isinstance(value, dict):
        return {k: restore(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(restore(v, device) for v in value)
    return value.to(device) if isinstance(value, torch.Tensor) else value


class NativeCapture(AbstractContextManager):
    """Wrap call boundaries outside compiled blocks, preserving their callable.

    CPU clones and output gradient hooks are deliberately intrusive diagnostics.
    All wrappers are removed after the first full forward_backward_step, before
    the remaining accumulation microbatches. No weights or optimizer are saved.
    """
    def __init__(self, trainer, output, rank, world_size):
        self.trainer = trainer
        self.output = Path(output).resolve()
        if not self.output.is_relative_to(ROOT / "runs/production-data-abcd"):
            raise ValueError(f"{ENV} must be inside runs/production-data-abcd")
        self.rank = rank
        self.world_size = world_size
        self.records = {}
        self.restorations = []
        self.hooks = []
        self.complete = False

    def replace(self, owner, name, value):
        existed = name in vars(owner)
        old = getattr(owner, name)
        self.restorations.append((owner, name, old, existed))
        setattr(owner, name, value)
        return old

    def cleanup(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        for owner, name, old, existed in reversed(self.restorations):
            if existed:
                setattr(owner, name, old)
            else:
                delattr(owner, name)
        self.restorations.clear()

    def boundary(self, module, component):
        slot = "_compiled_call_impl" if callable(getattr(module, "_compiled_call_impl", None)) else "forward"
        original = getattr(module, slot)
        names = (("hidden", "rope_cache", "attention_masks", "positions", "token_mask")
                 if component == "layer0" else
                 ("hidden", "embedding", "rope_cache", "attention_masks", "positions", "token_mask"))

        def call(*args, **kwargs):
            if component in self.records:
                # SAC replay is legal only after the output gradient arrived.
                if "upstream" not in self.records[component]:
                    raise RuntimeError(f"duplicate native forward boundary: {component}")
                return original(*args, **kwargs)
            values = dict(zip(names, args))
            aliases = {"trunk_hidden": "hidden", "input_embeds": "embedding", "x": "hidden"}
            values.update({aliases.get(k, k): v for k, v in kwargs.items()})
            row = {name: map_tensors(values.get(name), lambda t: t.detach().cpu().clone())
                   for name in ("hidden", "embedding", "positions", "attention_masks", "token_mask")}
            if row["hidden"] is None:
                raise RuntimeError(f"missing native hidden input: {component}")
            self.records[component] = row
            out = original(*args, **kwargs)
            if not out.requires_grad:
                raise RuntimeError(f"native output does not require gradients: {component}")

            def gradient(grad):
                if "upstream" in row:
                    raise RuntimeError(f"duplicate native gradient boundary: {component}")
                if component == "mtp":
                    if grad.shape[0] != 1:
                        raise RuntimeError("component harness requires exactly one native MTP head")
                    grad = grad[0]
                row["upstream"] = grad.detach().cpu().clone()
                # Returning None preserves the exact incoming gradient.
            self.hooks.append(out.register_hook(gradient))
            return out
        self.replace(module, slot, call)

    def __enter__(self):
        if self.trainer.config.training.steps != 1 or self.trainer.gradient_accumulation_steps != 32:
            raise RuntimeError("component acquisition requires one complete update with native accumulation 32")
        if len(self.trainer.model_parts) != 1:
            raise RuntimeError("component capture requires the native non-PP root")
        # Each rank owns one fresh directory; pre-existing outputs are rejected.
        self.rank_dir = self.output / f"rank{self.rank}"
        self.rank_dir.mkdir(parents=True, exist_ok=False)
        model = self.trainer.model_parts[0]
        original_fb = self.trainer.forward_backward_step
        try:
            self.boundary(model.layers["0"], "layer0")
            self.boundary(model.mtp, "mtp")
            original_embedding = model.tok_embeddings.forward

            def embedding(*args, **kwargs):
                if "input_ids" in self.records:
                    raise RuntimeError("duplicate native embedding boundary")
                ids = args[0] if args else kwargs["input"]
                self.records["input_ids"] = ids.detach().cpu().clone()
                return original_embedding(*args, **kwargs)
            self.replace(model.tok_embeddings, "forward", embedding)

            def first_microbatch(*args, **kwargs):
                result = original_fb(*args, **kwargs)
                self.cleanup()
                self.write()
                self.complete = True
                return result
            self.replace(self.trainer, "forward_backward_step", first_microbatch)
        except BaseException:
            self.cleanup()
            raise
        return self

    def write(self):
        import torch

        if set(self.records) != {"layer0", "mtp", "input_ids"}:
            raise RuntimeError("missing native component capture boundary")
        ids = self.records["input_ids"]
        if ids.dtype != torch.int64 or ids.ndim != 2:
            raise RuntimeError("native input IDs must be rank-local B,S int64")
        for component in ("layer0", "mtp"):
            row = self.records[component]
            if "upstream" not in row or row["hidden"].shape != row["upstream"].shape:
                raise RuntimeError(f"missing or mismatched native upstream: {component}")
            if row["hidden"].shape[:2] != ids.shape:
                raise RuntimeError("native component and input-ID batch shapes differ")
        config = self.trainer.config.to_dict()
        input_provenance = {}
        assets = config.get("hf_assets_path")
        if assets:
            asset_dir = Path(assets).resolve(strict=True)
            input_provenance["tokenizer"] = {str(p): digest(p) for p in sorted(asset_dir.iterdir()) if p.is_file()}
        for stream in config.get("dataloader", {}).get("dataset", {}).get("streams", []):
            local = Path(stream["local"])
            index = (local / stream.get("split", "") / "index.json").resolve(strict=True)
            input_provenance[str(index)] = digest(index)
        path = self.rank_dir / "tensors.pt"
        torch.save(self.records, path)
        receipt = {
            "schema_version": 1, "rank": self.rank, "world_size": self.world_size,
            "microbatch": "first complete native forward_backward_step",
            "state": "native activations/upstream; component replay uses paired canonical harness initialization",
            "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / "gift/torchtitan").rglob("*.py"))},
            "capture_sha256": digest(__file__), "resolved_config": config,
            "input_provenance": input_provenance,
            "tensors_sha256": digest(path), "input_ids_shape": list(ids.shape),
            "input_ids_int64_sha256": hashlib.sha256(ids.contiguous().numpy().tobytes()).hexdigest(),
        }
        (self.rank_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, default=str) + "\n")
        self.records.clear()

    def __exit__(self, exc_type, exc, tb):
        self.cleanup()
        if exc_type is None and not self.complete:
            raise RuntimeError("native capture did not complete a microbatch")
        return False


def load_fixture(root, rank, *, production=True):
    import torch

    rank_dir = Path(root) / f"rank{rank}"
    receipt = json.loads((rank_dir / "receipt.json").read_text())
    path = rank_dir / "tensors.pt"
    if receipt["rank"] != rank or receipt["schema_version"] != 1 or digest(path) != receipt["tensors_sha256"]:
        raise ValueError("fixture identity/hash mismatch")
    tensors = torch.load(path, map_location="cpu", weights_only=True)
    ids = tensors["input_ids"]
    if ids.dtype != torch.int64 or (production and (list(ids.shape) != [2, 4096] or receipt["world_size"] != 8)):
        raise ValueError("captured fixture requires native B2 S4096 int64 on eight ranks")
    if hashlib.sha256(ids.contiguous().numpy().tobytes()).hexdigest() != receipt["input_ids_int64_sha256"]:
        raise ValueError("input IDs hash mismatch")
    if production:
        for component in ("layer0", "mtp"):
            row = tensors[component]
            if list(row["hidden"].shape) != [2, 4096, 4096] or row["hidden"].shape != row["upstream"].shape:
                raise ValueError("captured component shape mismatch")
            if component == "mtp" and (row["embedding"] is None or row["embedding"].shape != row["hidden"].shape):
                raise ValueError("missing native unshifted shared MTP embedding")
    return tensors, receipt
