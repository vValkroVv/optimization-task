"""Run the native EP8 trainer with the exact CUDA-13 consuming folded CE."""

from contextlib import contextmanager
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys


FUSED_DIR = Path(__file__).resolve().parent.parent / "fused-ce"
ARTIFACT_HASHES = {
    "fused.py": "e764c7168a49b8bc3bde27547ddc2915681285a6d3f0c7ed2e73bda04a0e28b9",
    "centered.so": "41357457bb454c721ee72b5b9e4151cff7056f5e3b473a1241b39c4161925a11",
}
METHOD = "cuda-13"
CHUNKS_ENV = "GIFT_EXPERIMENT_LM_HEAD_MATRIX_CHUNKS"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def staged_writer(chunks):
    if chunks != 32:
        yield
        return

    from torchtitan.distributed.gfsdp import _staged_head_dw as staged
    from torchtitan.distributed.gfsdp import _staged_head_dw_writer as writer

    original = staged.select_staged_head_dw
    lines, first_line = inspect.getsourcelines(original)
    source = "".join(lines)
    guard = 'expected_chunks not in ("2", "8")'
    if source.count(guard) != 1 or 32 in writer._LT_ALGOS:
        raise RuntimeError("Unexpected native staged dW selector or writer table")
    source = source.replace(guard, 'expected_chunks not in ("2", "8", "32")')
    source = source.replace("actual 2 or 8 LM-head chunks", "actual 2, 8, or 32 LM-head chunks")
    namespace = {}
    exec(
        compile("\n" * (first_line - 1) + source, inspect.getfile(original), "exec"),
        original.__globals__,
        namespace,
    )
    plan = writer._LT_ALGOS[8]
    if plan[:2] != ("5.5", False):
        raise RuntimeError("C32 requires the preflighted cuBLASLt 5.5 writer plan")
    writer._LT_ALGOS[32] = plan
    staged.select_staged_head_dw = namespace[original.__name__]
    try:
        yield
    finally:
        staged.select_staged_head_dw = original
        del writer._LT_ALGOS[32]


def run_exact_ce(train_argv, adapter, native_main):
    import torch
    from torchtitan.components.loss import MTPLoss
    from torchtitan.experiments.four_fusion import stable_ce

    rank = int(os.environ["RANK"])
    chunks = int(os.environ[CHUNKS_ENV])
    if chunks not in (2, 8, 32) or os.environ["GIFT_EXPERIMENT_LM_HEAD_MATRIX"] != "new":
        raise RuntimeError("Exact CE requires C2, C8, or C32 with the new LM-head arm")
    steps = int(adapter.options["expected_steps"])
    if steps != 20 or int(adapter.options["warmup_updates"]) != 10:
        raise RuntimeError("Exact CE requires 10 warmup and 10 measured updates")
    for name, expected in ARTIFACT_HASHES.items():
        if hashlib.sha256((FUSED_DIR / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Unexpected exact CE artifact {name}")
    if os.environ.get("FUSED_CE_BINARY", "centered.so") != "centered.so":
        raise RuntimeError("Exact CUDA-13 CE requires the pinned centered.so")
    fused = load_module(FUSED_DIR / "fused.py", "_lm_final_fused")
    launcher = fused.Launcher(METHOD)
    original_call = MTPLoss.__call__
    original_factory = stable_ce.make_consuming_folded_cross_entropy
    prepared = []
    scale_check = None
    active_scales = None
    cached_denominator = None
    cached_weights = None
    weights_tensor = None
    cached_scales = None

    def loss_call(self, pred, labels, global_valid_tokens=None, **kwargs):
        nonlocal active_scales, cached_denominator
        nonlocal cached_weights, weights_tensor, cached_scales
        if pred.ndim != 4:
            return original_call(self, pred, labels, global_valid_tokens, **kwargs)
        if (
            active_scales is not None
            or not isinstance(global_valid_tokens, float)
            or not global_valid_tokens > 0
        ):
            raise RuntimeError("Expected immediate folded backward and a positive DP-reduced float")

        weights = (1.0, *self.loss_weights)
        if weights != cached_weights:
            weights_tensor = pred.new_tensor(weights, dtype=torch.float32)
            cached_weights = weights
            cached_denominator = None
        # Trainer.train_step shares one denominator across all 32 microbatches.
        if global_valid_tokens != cached_denominator:
            cached_scales = weights_tensor / global_valid_tokens
            cached_denominator = global_valid_tokens

        active_scales = cached_scales
        try:
            return original_call(self, pred, labels, global_valid_tokens, **kwargs)
        finally:
            active_scales = None

    @torch.library.custom_op("lm_final_exact_ce::consume", mutates_args={"logits"})
    def consume(
        logits: torch.Tensor, labels: torch.Tensor, scales: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        losses = logits.new_empty((logits.shape[0],), dtype=torch.float32)
        launcher(logits, labels, losses, scales)
        main, mtp = losses.chunk(2)
        return main.sum(), mtp.sum()

    @consume.register_fake
    def fake(logits, labels, scales):
        return logits.new_empty((), dtype=torch.float32), logits.new_empty((), dtype=torch.float32)

    class PrecomputedGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx, logits, labels, scales, compiled):
            result = compiled(logits, labels, scales)
            torch.autograd.graph.increment_version(logits)
            ctx.save_for_backward(logits, scales)
            return result

        @staticmethod
        def backward(ctx, *incoming):
            nonlocal scale_check
            if torch.is_grad_enabled():
                raise RuntimeError("Exact consuming CE supports first-order backward only")
            logits, scales = ctx.saved_tensors
            if scale_check is None:
                if launcher.calls != 1:
                    raise RuntimeError("Expected to validate CE scales on the first warmup call")
                actual = torch.stack(incoming).float()
                if not torch.equal(actual, scales):
                    raise RuntimeError(f"Native CE scales differ: {actual.tolist()} != {scales.tolist()}")
                scale_check = {
                    "incoming_fp32_scales": actual.tolist(),
                    "prepared_fp32_scales": scales.tolist(),
                }
            return logits, None, None, None

    def factory(backend):
        if backend != "inductor":
            raise RuntimeError(f"Exact CE requires the inductor loss backend, got {backend}")
        prepared.append(backend)
        compiled = torch.compile(consume, backend=backend, fullgraph=True)

        def storage(logits, labels, num_heads):
            if (
                not logits.requires_grad
                or active_scales is None
                or num_heads != 2
                or logits.dtype != torch.bfloat16
                or not logits.is_contiguous()
                or not labels.is_contiguous()
            ):
                raise RuntimeError("Exact CE requires two contiguous folded BF16 heads")
            flat = logits.view(-1, logits.shape[-1])
            return PrecomputedGradient.apply(flat, labels.view(-1), active_scales, compiled)

        return storage

    identity = {
        "rank": rank,
        "chunks": chunks,
        "method": METHOD,
    }
    with staged_writer(chunks):
        MTPLoss.__call__ = loss_call
        stable_ce.make_consuming_folded_cross_entropy = factory
        try:
            print(
                "lm_final_exact_ce_install="
                + json.dumps({**identity, "artifact_sha256": ARTIFACT_HASHES}, sort_keys=True),
                flush=True,
            )
            result = native_main(train_argv, adapter)
        finally:
            MTPLoss.__call__ = original_call
            stable_ce.make_consuming_folded_cross_entropy = original_factory

    expected = steps * 32 * chunks
    if prepared != ["inductor"] or launcher.calls != expected or scale_check is None:
        raise RuntimeError(
            f"Exact CE on rank {rank}: backends={prepared}, calls={launcher.calls}/{expected}, "
            f"scales_checked={scale_check is not None}"
        )
    print(
        "lm_final_exact_ce_complete="
        + json.dumps(
            {**identity, **scale_check, "calls": launcher.calls, "backend": prepared[0]},
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def main():
    bridge = load_module(Path(os.environ["LM_FINAL_FLASH_MEASURE"]), "_lm_final_flash_measure")
    original = bridge._bootstrap_and_run

    def bootstrap(train_argv, adapter, native_main):
        def with_exact_ce(argv, installed_adapter):
            return run_exact_ce(argv, installed_adapter, native_main)

        return original(train_argv, adapter, with_exact_ce)

    bridge._bootstrap_and_run = bootstrap
    try:
        return bridge.main()
    finally:
        bridge._bootstrap_and_run = original


if __name__ == "__main__":
    main()
