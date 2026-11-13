from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "gift"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from numerics import NativeNumericalReceipt, _tensor_digest  # noqa: E402
import numerics  # noqa: E402
from torchtitan.callbacks.base import Event  # noqa: E402
from torchtitan.callbacks.manager import CallbackManager  # noqa: E402
from torchtitan.components.optimizer import OptimizersContainer  # noqa: E402
from torchtitan.distributed import utils as dist_utils  # noqa: E402


@pytest.fixture(autouse=True)
def preserve_cpu_rng_state():
    state = torch.get_rng_state().clone()
    try:
        yield
    finally:
        torch.set_rng_state(state)


class _Handler:
    def emit(self, event, trainer, **fields):
        return SimpleNamespace(**fields)


def _callback_manager() -> CallbackManager:
    manager = object.__new__(CallbackManager)
    manager.handler = _Handler()
    return manager


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 3)


class _TinyLoss:
    def __init__(self) -> None:
        self._components: dict[str, torch.Tensor] = {}
        self.components_calls = 0

    def begin_step(self) -> None:
        self._components = {}

    def add(self, name: str, value: torch.Tensor) -> None:
        detached = value.detach()
        previous = self._components.get(name)
        self._components[name] = detached if previous is None else previous + detached

    def components(self) -> dict[str, torch.Tensor]:
        self.components_calls += 1
        return dict(self._components)


class _TinyAuxLoss:
    def __init__(self) -> None:
        self._states = [
            SimpleNamespace(
                names=("router/load",),
                totals=torch.tensor([0.25], dtype=torch.float32),
                scale=torch.tensor(1.0, dtype=torch.float32),
            )
        ]

    def begin_step(self) -> None:
        self._states[0].totals.zero_()
        self._states[0].scale.fill_(1.0)


class _TinySchedulers:
    def __init__(self, optimizer: OptimizersContainer) -> None:
        self.optimizer = optimizer
        self.steps = 0

    def step(self) -> None:
        self.steps += 1
        for group in self.optimizer.optimizers[0].param_groups:
            group["lr"] *= 0.99

    def state_dict(self) -> dict[str, object]:
        return {"steps": self.steps, "learning_rate": self.optimizer.optimizers[0].param_groups[0]["lr"]}


class _TinyTrainer:
    def __init__(
        self,
        *,
        optimizer_container_class: type[OptimizersContainer] = OptimizersContainer,
        mutate_gradient: bool = False,
        counter_delta: int = 0,
        fail_forward: bool = False,
        global_valid_tokens: torch.Tensor | float | int | None = None,
    ) -> None:
        self.model_parts = [_TinyModel()]
        config = optimizer_container_class.Config(
            lr=0.01,
            weight_decay=0.0,
            implementation="for-loop",
        )
        self.optimizers = optimizer_container_class(config, model_parts=self.model_parts)
        self.lr_schedulers = _TinySchedulers(self.optimizers)
        self.callbacks = _callback_manager()
        self.loss_fn = _TinyLoss()
        self.aux_loss = _TinyAuxLoss()
        self.step = 0
        self.ntokens_seen = torch.tensor(2**31 + 7, dtype=torch.int64)
        self.mutate_gradient = mutate_gradient
        self.counter_delta = counter_delta
        self.fail_forward = fail_forward
        self.global_valid_tokens = global_valid_tokens
        self.losses: list[torch.Tensor] = []
        self.last_norm: torch.Tensor | None = None

    def forward_backward_step(
        self,
        *,
        input_dict: dict[str, torch.Tensor],
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor | float | int,
    ) -> torch.Tensor:
        if self.fail_forward:
            raise RuntimeError("sentinel forward failure")
        inputs = input_dict["input"]
        logits_main = self.model_parts[0].linear(inputs)
        logits_mtp = self.model_parts[0].linear(inputs * 0.5 + 0.25)
        ce = F.cross_entropy(logits_main, labels, reduction="sum") / global_valid_tokens
        mtp = F.cross_entropy(logits_mtp, labels.roll(1), reduction="sum") / global_valid_tokens
        loss = ce + 0.3 * mtp
        self.loss_fn.add("loss_ce", ce)
        self.loss_fn.add("loss_mtp", mtp)
        self.aux_loss._states[0].totals.add_(0.125)
        loss.backward()
        return loss

    def run(self, updates: int, *, microbatches: int = 2) -> None:
        self.callbacks.emit(Event.after_load, self)
        input_base = torch.arange(6, dtype=torch.float32).reshape(2, 3).transpose(0, 1)
        labels_base = torch.tensor([0, 2, 1], dtype=torch.int64)
        denominator = self.global_valid_tokens
        if denominator is None:
            denominator = torch.tensor(microbatches * labels_base.numel(), dtype=torch.int64)
        for update in range(1, updates + 1):
            self.step = update
            self.optimizers.zero_grad()
            self.loss_fn.begin_step()
            self.aux_loss.begin_step()
            update_loss = torch.zeros((), dtype=torch.float32)
            batches = []
            for index in range(microbatches):
                inputs = input_base + index * 0.125 + update * 0.01
                state = self.callbacks.emit(
                    Event.after_dataloader,
                    self,
                    input_dict={"input": inputs},
                    labels=labels_base,
                )
                batches.append((state.input_dict, state.labels))
            for input_dict, labels in batches:
                loss = self.forward_backward_step(
                    input_dict=input_dict,
                    labels=labels,
                    global_valid_tokens=denominator,
                )
                self.losses.append(loss.detach().clone())
                update_loss.add_(loss.detach())
            if self.mutate_gradient and update == 1:
                first = next(self.model_parts[0].parameters())
                first.grad.view(-1)[0].add_(1e-7)
            self.last_norm = dist_utils.clip_grad_norm_(
                [param for model in self.model_parts for param in model.parameters()],
                1e6,
                foreach=True,
            )
            self.optimizers.step()
            self.lr_schedulers.step()
            self.ntokens_seen.add_(int(denominator))
            if self.counter_delta:
                self.ntokens_seen.add_(self.counter_delta)
            self.callbacks.emit(
                Event.batch_end,
                self,
                loss=update_loss,
                grad_norm=self.last_norm,
                global_valid_tokens=denominator,
            )


def _receipt_run(
    tmp_path: Path,
    *,
    updates: int = 2,
    capture_updates: tuple[int, ...] = (1, 2),
    expected_microbatches: int = 2,
    mutate_gradient: bool = False,
    counter_delta: int = 0,
    fail_forward: bool = False,
    optimizer_container_class: type[OptimizersContainer] = OptimizersContainer,
) -> tuple[_TinyTrainer, Path, torch.Tensor]:
    trainer = _TinyTrainer(
        optimizer_container_class=optimizer_container_class,
        mutate_gradient=mutate_gradient,
        counter_delta=counter_delta,
        fail_forward=fail_forward,
    )
    torch_rng_before = torch.get_rng_state().clone()
    receipt_path = tmp_path / "native-numerics-rank0.json"
    observer = NativeNumericalReceipt(
        trainer_class=_TinyTrainer,
        optimizer_container_class=optimizer_container_class,
        callback_manager_class=CallbackManager,
        dist_utils_module=dist_utils,
        output_dir=tmp_path,
        rank=0,
        capture_updates=capture_updates,
        expected_microbatches=expected_microbatches,
    )
    with observer:
        trainer.run(updates, microbatches=2)
    return trainer, receipt_path, torch_rng_before


def _observations(path: Path) -> list[dict[str, object]]:
    return json.loads(path.read_text())["observations"]


def _find(records: list[dict[str, object]], *, update: int, phase: str, name: str):
    matches = [
        record
        for record in records
        if record["update"] == update and record["phase"] == phase and record["name"] == name
    ]
    assert len(matches) == 1
    return matches[0]


def _training_snapshot(trainer: _TinyTrainer) -> dict[str, object]:
    optimizer = trainer.optimizers.optimizers[0]
    model = trainer.model_parts[0]
    return {
        "losses": [loss.clone() for loss in trainer.losses],
        "gradients": [None if p.grad is None else p.grad.clone() for p in model.parameters()],
        "parameters": [p.detach().clone() for p in model.parameters()],
        "optimizer_state": [
            {
                key: value.detach().clone() if isinstance(value, torch.Tensor) else value
                for key, value in optimizer.state.get(param, {}).items()
            }
            for param in model.parameters()
        ],
        "optimizer_step": trainer.step,
        "tokens_seen": trainer.ntokens_seen.clone(),
    }


def test_observation_is_neutral_for_two_update_two_chunk_adamw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(703)
    baseline = _TinyTrainer()
    baseline.run(2)
    baseline_rng = torch.get_rng_state().clone()
    expected = _training_snapshot(baseline)

    torch.manual_seed(703)
    observed = _TinyTrainer()
    torch_rng_before = torch.get_rng_state().clone()
    original_forward = _TinyTrainer.forward_backward_step
    original_step = OptimizersContainer.step
    original_emit = CallbackManager.emit
    original_clip = dist_utils.clip_grad_norm_
    output_dir = tmp_path / "neutrality"
    observer = NativeNumericalReceipt(
        trainer_class=_TinyTrainer,
        optimizer_container_class=OptimizersContainer,
        callback_manager_class=CallbackManager,
        dist_utils_module=dist_utils,
        output_dir=output_dir,
        rank=0,
        capture_updates=(1, 2),
        expected_microbatches=2,
    )
    digest_cpu_flags = []
    original_digest = numerics._tensor_digest

    def check_deferred_digest(value, rank, *, require_cpu=False):
        assert observer._forward_wrapper_depth == 0
        digest_cpu_flags.append(require_cpu)
        return original_digest(value, rank, require_cpu=require_cpu)

    monkeypatch.setattr(numerics, "_tensor_digest", check_deferred_digest)
    with observer:
        observed.run(2)
    assert _TinyTrainer.forward_backward_step is original_forward
    assert OptimizersContainer.step is original_step
    assert CallbackManager.emit is original_emit
    assert dist_utils.clip_grad_norm_ is original_clip
    observed_rng = torch.get_rng_state().clone()
    actual = _training_snapshot(observed)
    for key in ("losses", "gradients", "parameters"):
        for left, right in zip(expected[key], actual[key], strict=True):
            assert torch.equal(left, right)
    for expected_state, actual_state in zip(
        expected["optimizer_state"], actual["optimizer_state"], strict=True
    ):
        assert expected_state.keys() == actual_state.keys()
        for key, left in expected_state.items():
            right = actual_state[key]
            assert torch.equal(left, right) if isinstance(left, torch.Tensor) else left == right
    assert expected["optimizer_step"] == actual["optimizer_step"]
    assert torch.equal(expected["tokens_seen"], actual["tokens_seen"])
    assert torch.equal(torch_rng_before, torch.get_rng_state())
    assert torch.equal(baseline_rng, observed_rng)
    assert observed.loss_fn.components_calls == 2
    assert digest_cpu_flags.count(True) == 8
    records = _observations(output_dir / "native-numerics-rank0.json")
    assert any(
        record["update"] == 1
        and record["phase"] == "microbatch_output"
        and record["name"] == "loss.local_normalized"
        and record["present"]
        for record in records
    )
    assert any(
        record["update"] == 1
        and record["phase"] == "callback_state"
        and record["name"] == "callbacks.batch_end.grad_norm"
        for record in records
    )
    assert any(
        record["update"] == 1
        and record["phase"] == "loss_components"
        and record["microbatch"] is None
        for record in records
    )
    assert any(record["phase"] == "optimizer_state" and record["name"].endswith("optimizer.exp_avg") for record in records)


def test_gradient_hash_detects_one_element_change_with_same_rounded_norm(tmp_path: Path) -> None:
    torch.manual_seed(811)
    base = _TinyTrainer()
    baseline_norm = None
    original_clip = dist_utils.clip_grad_norm_

    def capture_base_norm(*args, **kwargs):
        nonlocal baseline_norm
        baseline_norm = original_clip(*args, **kwargs)
        return baseline_norm

    dist_utils.clip_grad_norm_ = capture_base_norm
    try:
        base.run(1)
    finally:
        dist_utils.clip_grad_norm_ = original_clip

    torch.manual_seed(811)
    changed = _TinyTrainer(mutate_gradient=True)
    changed_dir = tmp_path / "changed"
    observer = NativeNumericalReceipt(
        trainer_class=_TinyTrainer,
        optimizer_container_class=OptimizersContainer,
        callback_manager_class=CallbackManager,
        dist_utils_module=dist_utils,
        output_dir=changed_dir,
        rank=0,
        capture_updates=(1,),
        expected_microbatches=2,
    )
    with observer:
        changed.run(1)

    assert baseline_norm is not None and changed.last_norm is not None
    assert f"{baseline_norm.item():.5g}" == f"{changed.last_norm.item():.5g}"
    baseline_grad = next(base.model_parts[0].parameters()).grad
    changed_record = _find(
        _observations(changed_dir / "native-numerics-rank0.json"),
        update=1,
        phase="grad_preclip",
        name="model.0.linear.weight.grad",
    )
    baseline_hash = hashlib.sha256(
        memoryview(baseline_grad.detach().contiguous().view(torch.uint8).numpy())
    ).hexdigest()
    assert changed_record["sha256"] != baseline_hash


def test_step_and_int64_token_mutation_are_recorded_exactly(tmp_path: Path) -> None:
    trainer, path, _ = _receipt_run(
        tmp_path,
        updates=1,
        capture_updates=(1,),
        counter_delta=5,
    )
    records = _observations(path)
    token_record = _find(
        records,
        update=1,
        phase="batch_end",
        name="trainer.ntokens_seen",
    )
    expected = trainer.ntokens_seen.contiguous()
    digest = hashlib.sha256(
        memoryview(expected.reshape(-1).view(torch.uint8).numpy())
    ).hexdigest()
    assert token_record["dtype"] == "torch.int64"
    assert token_record["sha256"] == digest
    unmutated_count = torch.tensor(2**31 + 13, dtype=torch.int64)
    unmutated_digest = hashlib.sha256(
        memoryview(unmutated_count.reshape(-1).view(torch.uint8).numpy())
    ).hexdigest()
    assert token_record["sha256"] != unmutated_digest
    step_state = trainer.optimizers.optimizers[0].state[
        next(trainer.model_parts[0].parameters())
    ]["step"]
    step_record = _find(
        records,
        update=1,
        phase="optimizer_state",
        name="model.0.linear.weight.optimizer.step",
    )
    step_digest = hashlib.sha256(
        memoryview(
            step_state.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
        )
    ).hexdigest()
    assert step_state.item() == 1.0
    assert step_record["sha256"] == step_digest
    assert json.loads(path.read_text())["coverage"]["1"] == {
        "input_batches": 2,
        "microbatches": 2,
        "summary_calls": 1,
        "clip_calls": 1,
        "optimizer_calls": 1,
        "batch_end_calls": 1,
    }


def test_unselected_updates_are_not_captured(tmp_path: Path) -> None:
    _, path, _ = _receipt_run(
        tmp_path,
        updates=3,
        capture_updates=(1, 2),
    )
    records = _observations(path)
    assert all(record["update"] != 3 for record in records)


def test_actual_dist_sum_python_float_denominator_is_captured_without_tensor_conversion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    native_denominator = dist_utils.dist_sum(torch.tensor(262080, dtype=torch.int64))
    assert type(native_denominator) is float
    # This is the tensor-only metadata operation the old forward wrapper used.
    with pytest.raises(TypeError, match="unsupported tensor metadata type"):
        numerics._tensor_shape_dtype(native_denominator)

    torch.manual_seed(1907)
    baseline = _TinyTrainer(global_valid_tokens=native_denominator)
    baseline.run(1)
    baseline_rng = torch.get_rng_state().clone()
    expected = _training_snapshot(baseline)

    torch.manual_seed(1907)
    trainer = _TinyTrainer(global_valid_tokens=native_denominator)
    output = tmp_path / "python-denominator"
    observer = NativeNumericalReceipt(
        trainer_class=_TinyTrainer,
        optimizer_container_class=OptimizersContainer,
        callback_manager_class=CallbackManager,
        dist_utils_module=dist_utils,
        output_dir=output,
        rank=0,
        capture_updates=(1,),
        expected_microbatches=2,
    )
    original_digest = numerics._tensor_digest

    def reject_python_values(value, rank, *, require_cpu=False):
        assert type(value) not in (int, float)
        return original_digest(value, rank, require_cpu=require_cpu)

    monkeypatch.setattr(numerics, "_tensor_digest", reject_python_values)
    original_tensor = torch.tensor

    def reject_denominator_tensor_cast(data, *args, **kwargs):
        assert not (
            type(data) in (int, float) and data == native_denominator
        ), "native Python denominator must not be converted back to a tensor"
        return original_tensor(data, *args, **kwargs)

    monkeypatch.setattr(torch, "tensor", reject_denominator_tensor_cast)
    with observer:
        trainer.run(1)
    observed_rng = torch.get_rng_state().clone()

    actual = _training_snapshot(trainer)
    for key in ("losses", "gradients", "parameters"):
        for left, right in zip(expected[key], actual[key], strict=True):
            assert torch.equal(left, right)
    for expected_state, actual_state in zip(
        expected["optimizer_state"], actual["optimizer_state"], strict=True
    ):
        assert expected_state.keys() == actual_state.keys()
        for key, left in expected_state.items():
            right = actual_state[key]
            assert torch.equal(left, right) if isinstance(left, torch.Tensor) else left == right
    assert expected["optimizer_step"] == actual["optimizer_step"]
    assert torch.equal(expected["tokens_seen"], actual["tokens_seen"])
    assert torch.equal(baseline_rng, observed_rng)
    records = _observations(output / "native-numerics-rank0.json")
    denominator_records = [
        record
        for record in records
        if record["name"] in {
            "global_valid_tokens",
            "callbacks.batch_end.global_valid_tokens",
        }
    ]
    assert len(denominator_records) == 2
    assert all(record["python_type"] == "float" for record in denominator_records)
    assert all(record["dtype"] == "python.float" for record in denominator_records)
    assert all(record["value"] == 262080.0 for record in denominator_records)
    assert all(record["sha256"] is None for record in denominator_records)
    microbatch_metadata = [
        record
        for record in records
        if record["phase"] == "microbatch_metadata"
        and record["name"] == "trainer.forward_backward_step"
    ]
    assert len(microbatch_metadata) == 2
    assert all(
        record["global_valid_tokens"]["python_type"] == "float"
        and record["global_valid_tokens"]["value"] == 262080.0
        for record in microbatch_metadata
    )


def test_python_integer_denominator_keeps_its_type_and_exact_value() -> None:
    observer = NativeNumericalReceipt(
        trainer_class=_TinyTrainer,
        optimizer_container_class=OptimizersContainer,
        callback_manager_class=CallbackManager,
        dist_utils_module=dist_utils,
        output_dir="unused",
        rank=0,
    )
    observer._observe_denominator(
        2**40 + 3,
        name="global_valid_tokens",
        phase="update_summary",
        update=1,
    )
    record = observer._observations[0]
    assert record["python_type"] == "int"
    assert record["dtype"] == "python.int"
    assert record["value"] == 2**40 + 3


def test_incomplete_capture_rejects_receipt_and_restores_wrappers(tmp_path: Path) -> None:
    trainer = _TinyTrainer()
    original_forward = _TinyTrainer.forward_backward_step
    original_clip = dist_utils.clip_grad_norm_
    original_emit = CallbackManager.emit
    output = tmp_path / "incomplete"
    observer = NativeNumericalReceipt(
        trainer_class=_TinyTrainer,
        optimizer_container_class=OptimizersContainer,
        callback_manager_class=CallbackManager,
        dist_utils_module=dist_utils,
        output_dir=output,
        rank=0,
        capture_updates=(1,),
        expected_microbatches=3,
    )
    with pytest.raises(RuntimeError, match="microbatches; expected 3"):
        with observer:
            trainer.run(1, microbatches=2)
    assert _TinyTrainer.forward_backward_step is original_forward
    assert dist_utils.clip_grad_norm_ is original_clip
    assert CallbackManager.emit is original_emit
    assert not (output / "native-numerics-rank0.json").exists()


def test_wrapper_restoration_and_no_receipt_after_forward_error(tmp_path: Path) -> None:
    trainer = _TinyTrainer(fail_forward=True)
    original_forward = _TinyTrainer.forward_backward_step
    original_clip = dist_utils.clip_grad_norm_
    original_emit = CallbackManager.emit
    output = tmp_path / "error"
    observer = NativeNumericalReceipt(
        trainer_class=_TinyTrainer,
        optimizer_container_class=OptimizersContainer,
        callback_manager_class=CallbackManager,
        dist_utils_module=dist_utils,
        output_dir=output,
        rank=0,
        capture_updates=(1,),
        expected_microbatches=2,
    )
    with pytest.raises(RuntimeError, match="sentinel forward failure"):
        with observer:
            trainer.run(1)
    assert _TinyTrainer.forward_backward_step is original_forward
    assert dist_utils.clip_grad_norm_ is original_clip
    assert CallbackManager.emit is original_emit
    assert not (output / "native-numerics-rank0.json").exists()


def test_hash_metadata_distinguishes_none_empty_layout_dtype_and_large_int64() -> None:
    empty = torch.empty((0,), dtype=torch.float32)
    empty_meta, empty_hash, empty_finite = _tensor_digest(empty, rank=3)
    transposed = torch.arange(6, dtype=torch.float32).reshape(2, 3).transpose(0, 1)
    transposed_meta, transposed_hash, _ = _tensor_digest(transposed, rank=3)
    contiguous_meta, contiguous_hash, _ = _tensor_digest(transposed.contiguous(), rank=3)
    large_int = torch.tensor([2**31 + 17], dtype=torch.int64)
    int_meta, int_hash, int_finite = _tensor_digest(large_int, rank=3)
    same_value_float = torch.tensor([float(2**31 + 17)], dtype=torch.float64)
    float_meta, float_hash, float_finite = _tensor_digest(same_value_float, rank=3)

    absent = NativeNumericalReceipt(
        trainer_class=_TinyTrainer,
        optimizer_container_class=OptimizersContainer,
        callback_manager_class=CallbackManager,
        dist_utils_module=dist_utils,
        output_dir="unused",
        rank=3,
    )
    absent._observe_value(None, name="missing", phase="test", update=1)
    none_record = absent._observations[0]
    assert empty_meta["local_shape"] == [0]
    assert empty_hash == hashlib.sha256(b"").hexdigest()
    assert empty_finite is True
    assert none_record["present"] is False and none_record["sha256"] is None
    assert none_record["dtype"] is None
    assert transposed_meta["local_stride"] == [1, 3]
    assert transposed_hash == contiguous_hash
    assert contiguous_meta["dtype"] == "torch.float32"
    assert int_meta["dtype"] == "torch.int64" and int_finite is None
    assert float_meta["dtype"] == "torch.float64" and float_finite is True
    assert int_hash != float_hash
    assert large_int.item() == 2**31 + 17

    class UnexpectedShardLike:
        def __init__(self) -> None:
            self.local_calls = 0
            self.full_calls = 0

        def to_local(self):
            self.local_calls += 1
            raise AssertionError("unexpected wrapper must be rejected before local access")

        def full_tensor(self):
            self.full_calls += 1
            raise AssertionError("receipt must never gather a full tensor")

    unsupported = UnexpectedShardLike()
    with pytest.raises(TypeError, match="unsupported tensor type"):
        _tensor_digest(unsupported, rank=3)
    assert unsupported.local_calls == unsupported.full_calls == 0


def test_ragged_shard_receipt_hashes_local_elements_without_gather(tmp_path: Path) -> None:
    import torch.distributed as dist
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
    from torchtitan.distributed.gfsdp import RaggedShardDTensor
    from torchtitan.distributed.gfsdp._tensor import RaggedShard

    rendezvous = tmp_path / "gloo-rendezvous"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=0,
        world_size=1,
    )
    original_grad_mode = torch.is_grad_enabled()
    try:
        mesh = DeviceMesh("cpu", torch.tensor([0]), mesh_dim_names=("shard",))
        local = torch.tensor([1.25, -2.5], dtype=torch.float32)
        spec = DTensorSpec(
            mesh,
            (RaggedShard(local_numel=2, global_offset=0, shard_numels=(2,)),),
            tensor_meta=TensorMeta(
                shape=torch.Size([2]), stride=(1,), dtype=torch.float32
            ),
        )
        ragged = RaggedShardDTensor(local, spec, requires_grad=False)
        original_full_tensor = RaggedShardDTensor.full_tensor

        def fail_on_gather(self):
            raise AssertionError("numerical receipt must not gather a RaggedShardDTensor")

        RaggedShardDTensor.full_tensor = fail_on_gather
        try:
            metadata, digest, finite = _tensor_digest(ragged, rank=0)
        finally:
            RaggedShardDTensor.full_tensor = original_full_tensor

        expected = hashlib.sha256(
            memoryview(local.reshape(-1).view(torch.uint8).numpy())
        ).hexdigest()
        assert torch.is_grad_enabled() is original_grad_mode
        assert metadata["dtype"] == "torch.float32"
        assert metadata["global_shape"] == [2]
        assert metadata["local_shape"] == [2]
        assert metadata["mesh"]["device_type"] == "cpu"
        assert metadata["placements"][0]["type"] == "RaggedShard"
        assert metadata["placements"][0]["local_numel"] == 2
        assert metadata["placements"][0]["global_offset"] == 0
        assert metadata["placements"][0]["shard_numels"] == [2]
        assert digest == expected
        assert finite is True
    finally:
        dist.destroy_process_group()


class _ColdOptimizerContainer(OptimizersContainer):
    def step(self, closure=None):
        return super().step(closure)


def test_cold_optimizer_hook_setup_survives_wrapper_restoration(tmp_path: Path) -> None:
    raw_step = _ColdOptimizerContainer.__dict__["step"]
    output = tmp_path / "cold"
    observer = NativeNumericalReceipt(
        trainer_class=_TinyTrainer,
        optimizer_container_class=_ColdOptimizerContainer,
        callback_manager_class=CallbackManager,
        dist_utils_module=dist_utils,
        output_dir=output,
        rank=0,
        capture_updates=(1,),
        expected_microbatches=2,
    )
    with observer:
        trainer = _TinyTrainer(optimizer_container_class=_ColdOptimizerContainer)
        trainer.run(1)
    restored = _ColdOptimizerContainer.__dict__["step"]
    assert getattr(restored, "hooked", False)
    assert getattr(restored, "__wrapped__", None) is raw_step
