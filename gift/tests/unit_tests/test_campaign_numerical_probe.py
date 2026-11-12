from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn


def _load_subject():
    test_path = Path(__file__).resolve()
    staged_gift = test_path.parents[2]
    repo_gift = next(
        (
            parent / "gift"
            for parent in test_path.parents
            if (parent / "gift/torchtitan/trainer.py").is_file()
        ),
        next(
            (
                parent
                for parent in test_path.parents
                if (parent / "torchtitan/trainer.py").is_file()
            ),
            None,
        ),
    )
    if repo_gift is None:
        raise RuntimeError("could not locate the GIFT source package for scoped tests")
    if staged_gift.resolve() == repo_gift.resolve():
        return importlib.import_module(
            "torchtitan.experiments.cuda_graph_campaign.numerical_probe"
        )

    sys.path.insert(0, str(repo_gift))
    import torchtitan.config.schema.experiments.cuda_graph_campaign as schema_package
    import torchtitan.experiments.cuda_graph_campaign as campaign_package

    staged_schema = (
        staged_gift / "torchtitan/config/schema/experiments/cuda_graph_campaign"
    )
    staged_campaign = staged_gift / "torchtitan/experiments/cuda_graph_campaign"
    schema_package.__path__.insert(0, str(staged_schema))
    campaign_package.__path__.insert(0, str(staged_campaign))

    def load(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load staged module {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    load(
        "torchtitan.config.schema.experiments.cuda_graph_campaign.numerical_probe",
        staged_schema / "numerical_probe.py",
    )
    load(
        "torchtitan.experiments.cuda_graph_campaign.continuation",
        staged_campaign / "continuation.py",
    )
    return load(
        "torchtitan.experiments.cuda_graph_campaign.numerical_probe",
        staged_campaign / "numerical_probe.py",
    )


probe_module = _load_subject()


class _CountingBudget:
    def __init__(self):
        self.reserved = 0

    def reserve(self, byte_count: int) -> None:
        self.reserved += byte_count

    def reconcile(self) -> int:
        return self.reserved


def _patch_local_distributed(monkeypatch):
    monkeypatch.setattr(probe_module, "_rank_and_world", lambda: (0, 1))
    monkeypatch.setattr(probe_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(probe_module.dist, "get_world_size", lambda: 1)

    def all_gather_object(output, value, **kwargs):
        output[0] = value

    monkeypatch.setattr(probe_module.dist, "all_gather_object", all_gather_object)
    monkeypatch.setattr(probe_module.dist, "barrier", lambda **kwargs: None)


def _trainer_and_probe(tmp_path: Path, monkeypatch):
    from torchtitan.components.aux_loss import AuxLossTracker

    _patch_local_distributed(monkeypatch)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([1.0, -2.0]))

    model = Model()
    optimizer = torch.optim.AdamW([model.weight], lr=0.1, weight_decay=0.01, fused=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    model.weight.grad = torch.tensor([0.25, -0.5])
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()

    trainer = SimpleNamespace(
        config=SimpleNamespace(
            dump_folder=str(tmp_path),
            checkpoint=SimpleNamespace(initial_load_path=str(tmp_path / "step-1")),
        ),
        optimizers=SimpleNamespace(optimizers=[optimizer]),
        lr_schedulers=SimpleNamespace(
            schedulers=[scheduler], state_dict=scheduler.state_dict
        ),
        dataloader=SimpleNamespace(state_dict=lambda: {"position": 4}),
        callbacks=SimpleNamespace(stateful_objects=lambda: {}),
        model_parts=(model,),
        aux_loss=AuxLossTracker(
            [model],
            parallel_dims=SimpleNamespace(pp_enabled=False),
            device=torch.device("cpu"),
        ),
        step=1,
    )
    trainer.state_dict = lambda: {"step": trainer.step}
    config = SimpleNamespace(mode="observe", output_dir="probe", updates=1)
    rng_owner = SimpleNamespace()
    observer = probe_module.CampaignNumericalProbe(config, rng_continuation=rng_owner)
    observer.bind_trainer(trainer, rng_owner)
    observer.output_dir.mkdir(parents=True)
    observer.budget = _CountingBudget()
    observer._rng_restore_checked = True
    identity = "0:weight"
    observer._prior_p_hashes[identity] = probe_module._logical_digest(
        model.weight,
        probe_module._hash_header(
            identity=identity,
            family="P",
            phase="parameter_endpoint",
            tensor=model.weight,
        ),
    )
    observer._storage_info = {"cpu_fixture": True}
    phases = []

    def publish(phase, status, *, complete=False):
        observer._phase_history.append({"phase": phase, **status})
        if observer.rank == 0:
            root = {
                "mode": config.mode,
                "phase": phase,
                "complete": complete and status["ok"],
                "ranks": [status],
                "phase_history_by_rank": [observer._phase_history],
                "storage": observer._storage_info,
                "updates": config.updates,
            }
            probe_module._atomic_json(observer.output_dir / "manifest.json", root)
        phases.append((phase, status, complete))

    observer._publish_phase = publish
    return model, optimizer, scheduler, trainer, observer, phases


def _state(
    model, optimizer, trainer, *, step, microbatch_idx=None, loss=None, norm=None
):
    from torchtitan.callbacks.base import ObserverState

    return ObserverState(
        event=None,
        step=step,
        model_parts=(model,),
        parallel_dims=None,
        device=torch.device("cpu"),
        config=trainer.config,
        loss=loss,
        grad_norm=norm,
        global_valid_tokens=torch.tensor(4),
        microbatch_idx=microbatch_idx,
        ntokens_seen=step * 4,
        gradient_accumulation_steps=4,
        _optimizers=(optimizer,),
    )


def _run_cpu_reference(tmp_path: Path, monkeypatch, *, updates: int, wrong_order=False):
    from torchtitan.distributed.utils import clip_grad_norm_

    model, optimizer, scheduler, trainer, observer, phases = _trainer_and_probe(
        tmp_path, monkeypatch
    )
    observer.config.updates = updates
    expected_by_step = {}
    max_norm = 0.5
    for step in range(2, updates + 2):
        trainer.step = step
        start = _state(model, optimizer, trainer, step=step)
        observer.batch_start(start)
        optimizer.zero_grad(set_to_none=True)
        loss_values = []
        for microbatch_index in range(4):
            microbatch = _state(
                model,
                optimizer,
                trainer,
                step=step,
                microbatch_idx=microbatch_index,
            )
            observer.microbatch_start(microbatch)
            prepared = (
                torch.tensor([microbatch_index], dtype=torch.int64),
                torch.tensor([microbatch_index], dtype=torch.int64),
                {},
                {"attention_masks": None},
            )
            observer.capture_prepared_batch(
                step=step,
                microbatch_index=microbatch_index,
                prepared=prepared,
            )
            trainer.aux_loss.weigh(
                valid_tokens=torch.tensor(1),
                global_valid_tokens=torch.tensor(4),
            )
            loss = (model.weight * (microbatch_index + 1)).sum()
            loss_values.append(loss.detach())
            loss.backward()
            observer.after_forward(
                _state(
                    model,
                    optimizer,
                    trainer,
                    step=step,
                    microbatch_idx=microbatch_index,
                    loss=loss.detach(),
                )
            )
        raw_gradient = model.weight.grad.detach().clone()
        if wrong_order and step == 2:
            norm = clip_grad_norm_(
                [model.weight], max_norm, foreach=True, pp_mesh=None, ep_enabled=False
            )
            observer.before_optimizer_step(start)
        else:
            observer.before_optimizer_step(start)
            norm = clip_grad_norm_(
                [model.weight], max_norm, foreach=True, pp_mesh=None, ep_enabled=False
            )
        clipped_gradient = model.weight.grad.detach().clone()
        used_lr = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()
        post_update = _state(
            model,
            optimizer,
            trainer,
            step=step,
            loss=torch.stack(loss_values).sum(),
            norm=norm,
        )
        observer.after_optimizer_step(post_update)
        observer.batch_end(post_update)
        expected_by_step[step] = {
            "raw_gradient": raw_gradient,
            "clipped_gradient": clipped_gradient,
            "norm": norm.detach().clone(),
            "parameter": model.weight.detach().clone(),
            "exp_avg": optimizer.state[model.weight]["exp_avg"].detach().clone(),
            "exp_avg_sq": optimizer.state[model.weight]["exp_avg_sq"].detach().clone(),
            "step_tensor": optimizer.state[model.weight]["step"].detach().clone(),
            "used_lr": used_lr,
            "next_lr": optimizer.param_groups[0]["lr"],
        }
    return model, optimizer, observer, phases, expected_by_step, wrong_order


def test_aux_loss_scale_restore_uses_native_pre_forward_weigh(tmp_path, monkeypatch):
    from torchtitan.components.aux_loss import AuxLossTracker
    from torchtitan.models.common.regularization import declare_aux_loss

    _patch_local_distributed(monkeypatch)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            self.runtime_tensor = torch.tensor([1.0])
            declare_aux_loss(self, "router_aux_loss")

    saved_model = Model()
    saved_tracker = AuxLossTracker(
        [saved_model],
        parallel_dims=SimpleNamespace(pp_enabled=False),
        device=torch.device("cpu"),
    )
    saved_tracker.weigh(
        valid_tokens=torch.tensor(8), global_valid_tokens=torch.tensor(256)
    )
    saved_inventory = probe_module._buffer_inventory((saved_model,))
    checkpoint = saved_model.state_dict()
    assert "aux_loss_scale" not in checkpoint
    assert saved_model.aux_loss_scale.item() == 1 / 32

    restored_model = Model()
    restored_tracker = AuxLossTracker(
        [restored_model],
        parallel_dims=SimpleNamespace(pp_enabled=False),
        device=torch.device("cpu"),
    )
    restored_model.load_state_dict(checkpoint)
    restored_inventory = probe_module._buffer_inventory((restored_model,))
    assert restored_model.aux_loss_scale.item() == 1
    assert saved_inventory != restored_inventory
    assert probe_module._same_loaded_buffer_inventory(
        saved_inventory, restored_inventory
    )
    restored_model.runtime_tensor.fill_(2)
    assert not probe_module._same_loaded_buffer_inventory(
        saved_inventory, probe_module._buffer_inventory((restored_model,))
    )
    restored_model.runtime_tensor.fill_(1)

    optimizer = torch.optim.AdamW([restored_model.weight], fused=True)
    trainer = SimpleNamespace(
        config=SimpleNamespace(
            dump_folder=str(tmp_path),
            checkpoint=SimpleNamespace(initial_load_path=str(tmp_path / "step-1")),
        ),
        model_parts=(restored_model,),
        optimizers=SimpleNamespace(optimizers=[optimizer]),
        aux_loss=restored_tracker,
    )
    rng_owner = SimpleNamespace()
    observer = probe_module.CampaignNumericalProbe(
        SimpleNamespace(mode="observe", output_dir="probe", updates=1),
        rng_continuation=rng_owner,
    )
    observer.bind_trainer(trainer, rng_owner)
    observer.output_dir.mkdir(parents=True)
    observer.budget = _CountingBudget()
    observer._current = {"step": 2, "microbatches": [{"index": 0}]}
    observer._current_microbatch_index = 0
    try:
        restored_tracker.weigh(
            valid_tokens=torch.tensor(8), global_valid_tokens=torch.tensor(256)
        )
        assert restored_model.aux_loss_scale is restored_tracker._states[0].scale
        assert restored_model.aux_loss_scale.item() == 1 / 32
        capture = observer._current["microbatches"][0]["native_aux_loss_scale"]
        assert capture["producer"] == "AuxLossTracker.weigh"
        assert capture["boundary"] == "after_weigh_before_forward"
        assert capture["parts"][0]["module_names"] == ["<root>"]
        assert capture["parts"][0]["value"] == 1 / 32
        assert probe_module._buffer_inventory((restored_model,)) == saved_inventory
    finally:
        observer.close()
    assert "weigh" not in restored_tracker.__dict__


@pytest.mark.parametrize("updates", [1, 10])
def test_native_cpu_accumulation_records_and_reads_every_update(
    tmp_path, monkeypatch, updates
):
    _, optimizer, observer, phases, expected, _ = _run_cpu_reference(
        tmp_path, monkeypatch, updates=updates
    )
    assert [phase for phase, _, _ in phases] == [
        f"step_{step}" for step in range(2, updates + 2)
    ]
    assert phases[-1][2] is True
    assert not optimizer._optimizer_step_pre_hooks

    for step, values in expected.items():
        report_path = observer.output_dir / f"rank-00.step-{step:05d}.json"
        report = json.loads(report_path.read_text())
        assert all(
            microbatch["native_aux_loss_scale"]["boundary"]
            == "after_weigh_before_forward"
            for microbatch in report["microbatches"]
        )
        updated = report["updated_parameters"]["0:weight"]
        for family in ("P", "exp_avg", "exp_avg_sq", "step"):
            assert updated["hashes"][family] == updated["payload"][family]["sha256"]
        with probe_module._TensorPackReader(
            observer.output_dir / report["payload_file"], report["payload_entries"]
        ) as reader:
            assert reader.verify_all() == report["readback"]["verified_entries"]
            prefix = "0:weight"
            torch.testing.assert_close(
                reader.read(f"{prefix}:grad_preclip"), values["raw_gradient"]
            )
            torch.testing.assert_close(
                reader.read(f"{prefix}:grad_postclip"), values["clipped_gradient"]
            )
            torch.testing.assert_close(reader.read("native:total_norm"), values["norm"])
            torch.testing.assert_close(reader.read(f"{prefix}:P"), values["parameter"])
            torch.testing.assert_close(
                reader.read(f"{prefix}:exp_avg"), values["exp_avg"]
            )
            torch.testing.assert_close(
                reader.read(f"{prefix}:exp_avg_sq"), values["exp_avg_sq"]
            )
            torch.testing.assert_close(
                reader.read(f"{prefix}:step"), values["step_tensor"]
            )
        postclip = report["postclip"]["0:weight"]
        assert postclip["lr_used"] == values["used_lr"]
        assert report["next_groups"][0]["hparams"]["lr"] == values["next_lr"]
        assert report["payload_sha256"] == report["readback"]["payload_sha256"]
    manifest = json.loads((observer.output_dir / "manifest.json").read_text())
    assert manifest["complete"] is True


def test_wrong_preclip_order_is_a_failing_control(tmp_path, monkeypatch):
    _, _, observer, _, expected, _ = _run_cpu_reference(
        tmp_path, monkeypatch, updates=1, wrong_order=True
    )
    report = json.loads((observer.output_dir / "rank-00.step-00002.json").read_text())
    with probe_module._TensorPackReader(
        observer.output_dir / report["payload_file"], report["payload_entries"]
    ) as reader:
        wrong_preclip = reader.read("0:weight:grad_preclip")
    with pytest.raises(AssertionError):
        torch.testing.assert_close(wrong_preclip, expected[2]["raw_gradient"])


def test_tensor_pack_corruption_and_partial_write_fail(tmp_path):
    budget = _CountingBudget()
    final_path = tmp_path / "pack.bin"
    writer = probe_module._TensorPackWriter(final_path, budget)
    source = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
    writer.add("view", source, identity="p", family="P", phase="test")
    writer.add("none", None, identity="p", family="gradient", phase="test")
    writer.add(
        "zero",
        torch.zeros(2, dtype=torch.float32),
        identity="p",
        family="gradient",
        phase="test",
    )
    writer.close()
    entry = writer.entries["view"]
    assert entry["shape"] == [4, 3]
    assert entry["stride"] == [1, 4]
    reader = probe_module._TensorPackReader(final_path, writer.entries)
    assert reader.verify_all() == 3
    torch.testing.assert_close(reader.read("view"), source)
    assert reader.read("none") is None
    torch.testing.assert_close(reader.read("zero"), torch.zeros(2))
    reader.close()

    with final_path.open("r+b") as stream:
        original = stream.read(1)
        stream.seek(0)
        stream.write(bytes([original[0] ^ 1]))
    corrupted = probe_module._TensorPackReader(final_path, writer.entries)
    with pytest.raises(ValueError, match="hash mismatch"):
        corrupted.verify_all()
    corrupted.close()

    partial = probe_module._TensorPackWriter(tmp_path / "partial.bin", budget)
    partial.add("x", torch.ones(1), identity="x", family="P", phase="test")
    partial.abort()
    assert (tmp_path / "partial.bin.partial").is_file()
    assert not (tmp_path / "partial.bin").exists()


def test_complex_rope_cache_inventory_and_tensor_pack_roundtrip(tmp_path):
    from torchtitan.models.common.embeddings.rope import ComplexRoPE

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.rope = ComplexRoPE.Config(dim=8, max_seq_len=16).build()

    model = Model()
    cache = model.rope.cache
    assert cache.dtype == torch.complex64
    assert cache.layout == torch.strided
    assert cache.is_contiguous()
    assert not cache.is_conj()
    for unsupported_view in (cache.t(), cache.conj()):
        with pytest.raises(ValueError, match="unsupported tensor layout/dtype"):
            probe_module._logical_cpu(unsupported_view)

    inventory = probe_module._buffer_inventory((model,))
    rope_cache = next(item for item in inventory if item["name"] == "rope.cache")
    assert rope_cache["layout"]["dtype"] == "torch.complex64"

    model.rope.unsupported_cache = torch.ones(2, dtype=torch.complex128)
    with pytest.raises(ValueError, match=r"rope\.unsupported_cache.*torch\.complex128"):
        probe_module._buffer_inventory((model,))
    del model.rope.unsupported_cache

    pack_path = tmp_path / "complex-rope.bin"
    writer = probe_module._TensorPackWriter(pack_path, _CountingBudget())
    entry = writer.add(
        "rope.cache",
        cache,
        identity="rope.cache",
        family="buffer",
        phase="inventory",
    )
    header = probe_module._hash_header(
        identity="rope.cache", family="buffer", phase="inventory", tensor=cache
    )
    assert entry["dtype"] == "torch.complex64"
    assert entry["stride"] == list(cache.stride())
    assert entry["sha256"] == probe_module._logical_digest(cache, header)
    writer.close()

    with probe_module._TensorPackReader(pack_path, writer.entries) as reader:
        assert reader.verify_all() == 1
        restored = reader.read("rope.cache")
    assert restored.dtype == cache.dtype
    assert restored.layout == cache.layout
    assert restored.stride() == cache.stride()
    assert not restored.is_conj()
    assert torch.equal(restored, cache)

    unsupported = probe_module._TensorPackWriter(
        tmp_path / "unsupported-complex.bin", _CountingBudget()
    )
    try:
        with pytest.raises(ValueError, match="torch.complex128"):
            unsupported.add(
                "complex128",
                torch.ones(2, dtype=torch.complex128),
                identity="unsupported",
                family="buffer",
                phase="inventory",
            )
    finally:
        unsupported.abort()


def test_payload_budget_enforces_cap_and_free_reserve(tmp_path, monkeypatch):
    monkeypatch.setattr(probe_module, "_CAP_BYTES", 32)
    monkeypatch.setattr(probe_module, "_FREE_RESERVE_BYTES", 8)
    monkeypatch.setattr(
        probe_module.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=100, total=100, used=0),
    )
    output = tmp_path / "probe"
    output.mkdir()
    budget = probe_module._PayloadBudget(output, (tmp_path,))
    budget.reserve(24)
    with pytest.raises(OSError, match="cap exceeded"):
        budget.reserve(9)

    second = tmp_path / "second"
    second.mkdir()
    monkeypatch.setattr(
        probe_module.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=10, total=10, used=0),
    )
    reserve_budget = probe_module._PayloadBudget(second, (second,))
    with pytest.raises(OSError, match="less than 8 free bytes"):
        reserve_budget.reserve(3)


def test_pinned_flex_blockmask_inventory_is_tensorwise(tmp_path):
    from torch.nn.attention.flex_attention import BlockMask, and_masks
    from torchtitan.models.common import get_causal_mask_mod

    block_counts = torch.ones((1, 1, 1), dtype=torch.int32)
    block_indices = torch.zeros((1, 1, 1, 1), dtype=torch.int32)
    mask = BlockMask(
        seq_lengths=(8, 8),
        kv_num_blocks=block_counts,
        kv_indices=block_indices,
        full_kv_num_blocks=torch.zeros_like(block_counts),
        full_kv_indices=torch.zeros_like(block_indices),
        q_num_blocks=block_counts.clone(),
        q_indices=block_indices.clone(),
        full_q_num_blocks=torch.zeros_like(block_counts),
        full_q_indices=torch.zeros_like(block_indices),
        BLOCK_SIZE=(128, 128),
        mask_mod=and_masks(get_causal_mask_mod()),
    )
    writer = probe_module._TensorPackWriter(tmp_path / "mask.bin", _CountingBudget())
    record = probe_module._capture_block_mask(
        writer, mask, key="mask", identity="mask", phase="input"
    )
    writer.close()
    assert record["type"].endswith("BlockMask")
    assert record["fields"]["mask_mod"]["interpretation"].startswith("metadata_only")
    with probe_module._TensorPackReader(
        tmp_path / "mask.bin", writer.entries
    ) as reader:
        assert reader.verify_all() == len(writer.entries)


def test_inventory_alias_collision_and_optimizer_refusal():
    class Aliases(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Parameter(torch.ones(2))
            self.second = self.first

    alias = Aliases()
    records = probe_module._named_parameters((alias,))
    assert len(records) == 1
    assert set(records[0]["names"]) == {"first", "second"}

    class Collision(nn.Module):
        def __init__(self):
            super().__init__()
            wrapped = nn.Module()
            wrapped.p = nn.Parameter(torch.ones(1))
            self._orig_mod = wrapped
            self.p = nn.Parameter(torch.zeros(1))

    with pytest.raises(ValueError, match="canonical parameter-name collision"):
        probe_module._named_parameters((Collision(),))

    param = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([param], fused=False)
    from torchtitan.callbacks.base import ObserverState

    state = ObserverState(
        event=None,
        step=1,
        model_parts=(nn.Module(),),
        parallel_dims=None,
        device=torch.device("cpu"),
        config=None,
        _optimizers=(optimizer,),
    )
    with pytest.raises(ValueError, match="unsupported native AdamW options"):
        probe_module._group_map(state)


def test_partial_bind_failure_removes_installed_hooks(tmp_path, monkeypatch):
    from torchtitan.components.aux_loss import AuxLossTracker

    _patch_local_distributed(monkeypatch)
    first_param = nn.Parameter(torch.ones(1))
    first = torch.optim.AdamW([first_param], fused=True)
    second = torch.optim.SGD([nn.Parameter(torch.ones(1))], lr=0.1)
    model = nn.Module()
    trainer = SimpleNamespace(
        config=SimpleNamespace(
            dump_folder=str(tmp_path),
            checkpoint=SimpleNamespace(initial_load_path=str(tmp_path / "step-1")),
        ),
        optimizers=SimpleNamespace(optimizers=[first, second]),
        model_parts=(model,),
        aux_loss=AuxLossTracker(
            [model],
            parallel_dims=SimpleNamespace(pp_enabled=False),
            device=torch.device("cpu"),
        ),
    )
    config = SimpleNamespace(mode="observe", output_dir="probe", updates=1)
    rng_owner = SimpleNamespace()
    observer = probe_module.CampaignNumericalProbe(config, rng_continuation=rng_owner)
    with pytest.raises(TypeError, match="unsupported native optimizer"):
        observer.bind_trainer(trainer, rng_owner)
    assert not first._optimizer_step_pre_hooks
    assert "weigh" not in trainer.aux_loss.__dict__


def test_s0_failure_record_is_best_effort_and_publishes(tmp_path, monkeypatch):
    observer = object.__new__(probe_module.CampaignNumericalProbe)
    observer.config = SimpleNamespace(mode="create_s0")
    observer.rank = 0
    observer.output_dir = tmp_path / "probe"
    observer.budget = _CountingBudget()
    observer._hook_handles = []
    observer._aux_loss_weigh_wrapped = False
    observer._rng_wrapped = False
    observer._active_writer = None
    observer._rng_serialized = []
    observer._ensure_output = lambda **kwargs: None
    published = []
    observer._publish_phase = lambda phase, status, **kwargs: published.append(
        (phase, status)
    )

    def fail_failure_record(*args, **kwargs):
        raise OSError("secondary failure-record write failure")

    monkeypatch.setattr(probe_module, "_atomic_json", fail_failure_record)
    expected = tmp_path / "checkpoints" / "step-1"
    state = SimpleNamespace(
        step=1,
        last_saved_checkpoint=str(expected),
        config=SimpleNamespace(
            dump_folder=str(tmp_path),
            checkpoint=SimpleNamespace(folder="checkpoints"),
        ),
        device=SimpleNamespace(type="cpu"),
    )
    with pytest.raises(RuntimeError, match="native S0 serialization must call"):
        observer.after_checkpoint_save(state)
    assert len(published) == 1
    assert published[0][0] == "s0"
    assert "native S0 serialization must call" in published[0][1]["error"]


def test_load_failure_record_is_best_effort_and_publishes(tmp_path, monkeypatch):
    _patch_local_distributed(monkeypatch)
    observer = object.__new__(probe_module.CampaignNumericalProbe)
    observer.config = SimpleNamespace(mode="observe", updates=1)
    observer.rank = 0
    observer.world_size = 1
    observer.output_dir = tmp_path / "probe"
    observer.budget = None
    observer._hook_handles = []
    observer._aux_loss_weigh_wrapped = False
    observer._rng_wrapped = False
    observer._active_writer = None
    published = []
    observer._publish_phase = lambda phase, status, **kwargs: published.append(
        (phase, status)
    )

    writes = []

    def fail_unbudgeted_write(*args, **kwargs):
        writes.append((args, kwargs))
        raise OSError("unbudgeted failure-record write should be skipped")

    monkeypatch.setattr(probe_module, "_atomic_json", fail_unbudgeted_write)
    state = SimpleNamespace(
        step=0,
        config=SimpleNamespace(
            checkpoint=SimpleNamespace(
                initial_load_path=str(tmp_path / "checkpoints" / "step-1")
            ),
            dump_folder=str(tmp_path),
        ),
    )
    with pytest.raises(RuntimeError, match="observe requires restored step 1"):
        observer.after_load(state)
    assert len(published) == 1
    assert published[0][0] == "s0_verified"
    assert "observe requires restored step 1" in published[0][1]["error"]
    assert writes == []


def test_batch_failure_record_is_best_effort_and_publishes(tmp_path):
    class Writer:
        def __init__(self):
            self.aborted = False

        def add(self, *args, **kwargs):
            return {"status": "tensor", "sha256": "unused"}

        def abort(self):
            self.aborted = True

    observer = object.__new__(probe_module.CampaignNumericalProbe)
    observer.config = SimpleNamespace(mode="observe", updates=1)
    observer.rank = 0
    observer.output_dir = tmp_path

    def fail_reservation(byte_count):
        raise OSError("secondary failure-record quota failure")

    observer.budget = SimpleNamespace(reserve=fail_reservation)
    observer._current = {
        "step": 2,
        "rng_restored_checked": True,
        "updated_parameters": {"p": {}},
    }
    writer = Writer()
    observer._active_writer = writer
    observer._hook_handles = []
    observer._aux_loss_weigh_wrapped = False
    observer._rng_wrapped = False
    observer._native_state = lambda: {}

    def fail_native_state(*args, **kwargs):
        raise OSError("primary native-state ENOSPC failure")

    observer._save_small_state = fail_native_state
    published = []

    def publish(phase, status, **kwargs):
        published.append((phase, status))
        if not status["ok"]:
            raise RuntimeError(status["error"])

    observer._publish_phase = publish

    state = SimpleNamespace(
        step=2,
        loss=torch.tensor(1.0),
        ntokens_seen=4,
        global_valid_tokens=torch.tensor(4),
    )
    with pytest.raises(RuntimeError, match="primary native-state ENOSPC failure"):
        observer.batch_end(state)
    assert published[0][0] == "step_2"
    assert "primary native-state ENOSPC failure" in published[0][1]["error"]
    assert writer.aborted
    assert observer._active_writer is None
    assert observer._current is None


def test_phase_failure_without_budget_still_reaches_collectives(tmp_path, monkeypatch):
    observer = object.__new__(probe_module.CampaignNumericalProbe)
    observer.config = SimpleNamespace(mode="observe", updates=1)
    observer.rank = 0
    observer.world_size = 1
    observer.output_dir = tmp_path / "probe"
    observer.budget = None
    observer._storage_info = {}
    observer._phase_history = []
    gathered = []
    barriers = []
    writes = []

    def all_gather(output, value, **kwargs):
        gathered.append(value)
        output[0] = value

    def fail_unbudgeted_write(*args, **kwargs):
        writes.append((args, kwargs))
        raise OSError("unbudgeted phase write should be skipped")

    monkeypatch.setattr(probe_module.dist, "all_gather_object", all_gather)
    monkeypatch.setattr(
        probe_module.dist, "barrier", lambda **kwargs: barriers.append(1)
    )
    monkeypatch.setattr(probe_module, "_atomic_json", fail_unbudgeted_write)
    with pytest.raises(RuntimeError) as error:
        observer._publish_phase(
            "step_2", {"rank": 0, "ok": False, "error": "primary storage failure"}
        )
    assert "primary storage failure" in str(error.value)
    assert "storage budget unavailable" in str(error.value)
    assert len(gathered) == 3
    assert barriers == [1]
    assert writes == []


def test_pending_rng_accessor_returns_copies():
    from torchtitan.experiments.cuda_graph_campaign.continuation import (
        RNGContinuationState,
    )

    state = object.__new__(RNGContinuationState)
    state._pending_restore = {
        "torch_cpu_state": torch.tensor([1, 2], dtype=torch.uint8),
        "torch_cuda_state": torch.tensor([3, 4], dtype=torch.uint8),
        "scalar": 5,
    }
    snapshot = state.pending_restore_snapshot()
    snapshot["torch_cpu_state"][0] = 9
    assert state._pending_restore["torch_cpu_state"][0].item() == 1
    assert state.pending_restore_snapshot()["scalar"] == 5
    state._pending_restore = None
    assert state.pending_restore_snapshot() is None
