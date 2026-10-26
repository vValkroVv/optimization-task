"""Compiled step -> eager logged step -> compiled step, on an Inductor model."""

import contextlib
import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from torchtitan.callbacks.builtin.monitoring import MonitoringCallback
from torchtitan.callbacks.sources.dispatcher_source import (
    DispatcherSource,
    MaxVioConfig,
)
from torchtitan.callbacks.sources.modelling_source import (
    ModellingSource,
    MoECaptureConfig,
)
from torchtitan.callbacks.sources.trainer_source import TrainerSource
from torchtitan.config import ActivationCheckpointConfig, CompileConfig
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.distributed.backends.fsdp2 import FSDP2Backend
from torchtitan.distributed.compile import apply_compile
from torchtitan.distributed.pipeline_parallel import metadata_probe
from torchtitan.models.common.moe import GroupedExperts, MoE, TokenChoiceTopKRouter
from torchtitan.models.common.nn_modules import Linear
from torchtitan.protocols.blocks import Block
from torchtitan.protocols.module import Module, ModuleDict
from tests.unit_tests.callbacks.trainer_stub import snapshot

DIM, HIDDEN, EXPERTS, TOP_K, TOKENS = 32, 64, 4, 2, 16
STEPS, LOG_INTERVAL = 4, 2


class _SingleRankDims:
    pp_enabled = False

    def get_optional_mesh(self, name: str):
        del name
        return None


class _Block(Block):
    def __init__(self):
        super().__init__()
        self.attention = Linear.Config(
            in_features=DIM, out_features=DIM, bias=False
        ).build()
        self.moe = MoE.Config(
            experts=GroupedExperts.Config(
                dim=DIM, hidden_dim=HIDDEN, num_experts=EXPERTS
            ),
            router=TokenChoiceTopKRouter.Config(
                gate=Linear.Config(in_features=DIM, out_features=EXPERTS, bias=False),
                top_k=TOP_K,
            ),
        ).build()

    def forward(self, x):
        return x + self.moe(self.attention(x).relu())


class _Model(Module):
    def __init__(self):
        super().__init__()
        self.layers = ModuleDict({str(i): _Block() for i in range(2)})
        self.head = Linear.Config(in_features=DIM, out_features=DIM, bias=False).build()

    def forward(self, x):
        for layer in self.layers.values():
            x = layer(x)
        return self.head(x)


def _callback() -> MonitoringCallback:
    return MonitoringCallback(
        MonitoringCallback.Config(
            log_interval=LOG_INTERVAL,
            sources={
                "modelling": ModellingSource.Config(capture_backward=True),
                "dispatcher": DispatcherSource.Config(
                    max_vio=[
                        MaxVioConfig(name="mb_avg", window="global", level="microbatch")
                    ]
                ),
            },
        ),
        sharding_backend=FSDP2Backend(),
    )


def _state(step: int, model: nn.Module, device: str):
    return snapshot(
        SimpleNamespace(
            step=step,
            model_parts=[model],
            parallel_dims=_SingleRankDims(),
            device=torch.device(device),
            gradient_accumulation_steps=1,
        )
    )


def _train(model, callback, *, device: str, seed: int):
    """``STEPS`` SGD steps on one fixed input stream; returns (losses, logged records)."""
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    generator = torch.Generator(device=device).manual_seed(seed)
    dispatcher = callback.find_source(DispatcherSource) if callback else None
    losses, logged = [], {}
    if callback is not None:
        callback.fit_start(_state(0, model, device))
    for step in range(1, STEPS + 1):
        state = _state(step, model, device)
        if callback is not None:
            callback.batch_start(state)
        x = torch.randn(1, TOKENS, DIM, device=device, generator=generator)
        # Step 1 compiles; afterwards a compiled step must hit the cache,
        # including the one right after the eager logged step.
        eager_step = callback is not None and step % LOG_INTERVAL == 0
        stance = (
            torch.compiler.set_stance("fail_on_recompile")
            if step > 1 and not eager_step
            else contextlib.nullcontext()
        )
        with stance:
            loss = model(x).float().pow(2).mean()
            loss.backward()
        if callback is not None:
            callback.before_optimizer_step(state)
        optimizer.step()
        optimizer.zero_grad()
        if callback is not None:
            callback.after_optimizer_step(state)
            # Each block dispatched once this step, recompute included.
            for name, accum in dispatcher._accum.items():
                assert int(accum.cursor) == 1, f"step {step}: {name} double-counted"
                assert int(accum.routed) == TOKENS * TOP_K, f"step {step}: {name}"
            callback.batch_end(state)
            logged[step] = list(state._records)
        losses.append(loss.item())
    if callback is not None:
        callback.close()
    return losses, logged


def _initialized_model(device: str) -> _Model:
    """``build()`` leaves the parameters unset; give them a small normal init."""
    torch.manual_seed(0)
    model = _Model().to(device)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.02)
        for buffer in model.buffers():
            buffer.zero_()
    return model


def _run(device: str, backend: str) -> None:
    torch._dynamo.reset()
    reference = _initialized_model(device)
    monitored = copy.deepcopy(reference)
    for model in (reference, monitored):
        apply_ac(
            model, ActivationCheckpointConfig(mode="full"), model_compile_enabled=True
        )
        apply_compile(model, CompileConfig(backend=backend))
    blocks = [m for m in monitored.modules() if getattr(m, "_compiled_call_impl", None)]
    assert len(blocks) == 2

    reference_losses, _ = _train(reference, None, device=device, seed=1)
    monitored_losses, logged = _train(monitored, _callback(), device=device, seed=1)

    for step in range(1, STEPS + 1):
        names = {record.name for record in logged.get(step, ())}
        if step % LOG_INTERVAL:
            assert not names, f"step {step} is not a logged step"
            continue
        assert any(n.startswith("activations_") for n in names), step
        assert any(n.startswith("activations_gradients_") for n in names), step
        assert "moe/token_dispatcher/layers.0.moe/routed_tokens" in names
    assert all(block._compiled_call_impl is not None for block in blocks)

    # The logged step ran eager: the only difference is the eager/Inductor
    # kernel drift, and the trajectories stay together.
    torch.testing.assert_close(
        torch.tensor(monitored_losses),
        torch.tensor(reference_losses),
        rtol=1e-4,
        atol=1e-5,
    )
    for (name, a), (_, b) in zip(
        monitored.named_parameters(), reference.named_parameters()
    ):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5, msg=name)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Inductor on a CUDA device")
def test_compiled_eager_compiled_with_activation_checkpointing():
    _run("cuda", "inductor")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Inductor on a CUDA device")
def test_metadata_probe_does_not_record_metrics():
    torch._dynamo.reset()
    model = _initialized_model("cuda")
    apply_ac(model, ActivationCheckpointConfig(mode="full"), model_compile_enabled=True)
    apply_compile(model, CompileConfig(backend="inductor"))
    callback = MonitoringCallback(
        MonitoringCallback.Config(
            log_interval=LOG_INTERVAL,
            sources={
                "modelling": ModellingSource.Config(
                    capture_backward=True,
                    moe=MoECaptureConfig(per_expert_tokens=True),
                ),
                "trainer": TrainerSource.Config(
                    topk_accuracy=[1], logits_module_pattern="^head$"
                ),
            },
        ),
        sharding_backend=FSDP2Backend(),
    )
    state = _state(LOG_INTERVAL, model, "cuda")
    callback.fit_start(state)
    callback.batch_start(state)
    modelling = callback.find_source(ModellingSource)
    trainer = callback.find_source(TrainerSource)
    x = torch.randn(1, TOKENS, DIM, device="cuda")
    try:
        with metadata_probe(model):
            loss = model(x).float().square().mean()
        assert not modelling._capture_accum
        assert not modelling._router_accum
        assert not trainer._topk_chunks
        with metadata_probe(model):
            loss.backward()
        assert not modelling._capture_accum
        assert not modelling._router_accum
        assert not trainer._topk_chunks

        model.zero_grad(set_to_none=True)
        model(x).float().square().mean().backward()
        assert modelling._capture_accum
        assert modelling._router_accum
        assert len(trainer._topk_chunks) == 1
        assert any("activations_gradients" in name for name in modelling._capture_accum)
    finally:
        callback.close()
