from __future__ import annotations

import gc
import hashlib
import io
import json
import sys
import weakref
from pathlib import Path
from types import FunctionType, ModuleType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from torchtitan.config.errors import ConfigFieldError
from torchtitan.config.schema.experiments.cuda_graph_campaign.combine_bwd_completion import (
    CampaignCombineBwdCompletionConfig,
)
from torchtitan.experiments.cuda_graph_campaign import combine_bwd_completion as diagnostic
from torchtitan.experiments.cuda_graph_campaign import (
    train_combine_bwd_completion as diagnostic_entrypoint,
)


def _selected_function(path: Path, handle=None, *, firstlineno: int = 651):
    def selected():
        return handle

    code = selected.__code__.replace(
        co_name="<lambda>",
        co_filename=str(path),
        co_firstlineno=firstlineno,
    )
    fn = FunctionType(code, selected.__globals__, "<lambda>", None, selected.__closure__)
    fn.__module__ = diagnostic._SELECTED_MODULE
    fn.__qualname__ = diagnostic._SELECTED_QUALNAME
    return fn


def _adapter_state(*, step: int = 21, microbatch_idx: int = 0, armed: bool = True):
    module = ModuleType(diagnostic._HYBRIDEP_MODULE)
    module.__file__ = str(diagnostic._expected_hybridep_path())
    calls = []

    def original(fn, *produced_on_compute):
        calls.append((fn, produced_on_compute))
        return object()

    state = diagnostic._AdapterState(module=module, original=original)
    state.fit_started = True
    state.training_context = True
    state.step = step
    state.microbatch_idx = microbatch_idx
    state.armed = armed
    state.rank = 0
    state.output = io.StringIO()
    return state, calls


def _install_fake_hybridep(monkeypatch, tmp_path: Path):
    path = tmp_path / "hybridep.py"
    source = "\n" * 650 + "    lambda: _buffer.combine_with_unpermute(\n"
    path.write_text(source, encoding="utf-8")
    module = ModuleType(diagnostic._HYBRIDEP_MODULE)
    module.__file__ = str(path)
    module._buffer = None

    def original(fn, *produced_on_compute):
        return fn()

    original.__code__ = original.__code__.replace(
        co_filename=str(path), co_firstlineno=121
    )
    original.__module__ = module.__name__
    original.__name__ = diagnostic._HELPER_NAME
    original.__qualname__ = diagnostic._HELPER_NAME
    module._run_on_comm_stream = original
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(diagnostic.importlib, "import_module", lambda _: module)
    monkeypatch.setattr(diagnostic, "_expected_hybridep_path", lambda: path)
    monkeypatch.setattr(
        diagnostic,
        "_HYBRIDEP_SHA256",
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(diagnostic, "_ACTIVE_STATE", None)
    return module, original, path


def _valid_observer_state(tmp_path: Path, *, step: int = 0, microbatch_idx=None):
    config = SimpleNamespace(
        model_spec=SimpleNamespace(name="gift_moe", flavor="3B"),
        parallelism=SimpleNamespace(
            pipeline_parallel_degree=1,
            expert_parallel_degree=8,
            fsdp_backend="fsdp2",
        ),
        training=SimpleNamespace(
            local_batch_size=8,
            global_batch_size=256,
            seq_len=4096,
            steps=60,
        ),
        dataloader=SimpleNamespace(dataset="c4_test"),
        activation_checkpoint=SimpleNamespace(mode="selective"),
        compile=SimpleNamespace(
            enable=True,
            components=["model", "loss"],
            backend="inductor",
        ),
        fp8=SimpleNamespace(
            dense_use_shared_quantizer=True,
            sparse_use_grouped_gemm=True,
        ),
        checkpoint=SimpleNamespace(enable=False),
        comm=SimpleNamespace(mode="default"),
        dump_folder=str(tmp_path),
    )
    parallel_dims = SimpleNamespace(world_size=8, pp=1, ep=8)
    return SimpleNamespace(
        step=step,
        microbatch_idx=microbatch_idx,
        model_parts=(),
        parallel_dims=parallel_dims,
        device=torch.device("cuda", 0),
        config=config,
        gradient_accumulation_steps=4,
    )


def _patch_target_dist(monkeypatch, *, rank=0, world_size=8):
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: rank)
    monkeypatch.setattr(dist, "get_world_size", lambda: world_size)


class _FakeEvent:
    def __init__(self, order, *, record_error=None, wait_error=None, alive=None, **kwargs):
        self.order = order
        self.record_error = record_error
        self.wait_error = wait_error
        self.alive = alive
        assert kwargs == {"enable_timing": False}
        self.order.append("event_create")

    def record(self, stream):
        assert stream == "compute-stream"
        self.order.append("event_record")
        if self.record_error is not None:
            raise self.record_error

    def synchronize(self):
        self.order.append("event_synchronize")
        if self.alive is not None:
            gc.collect()
            assert all(reference() is not None for reference in self.alive)
        if self.wait_error is not None:
            raise self.wait_error


def _patch_event_api(monkeypatch, order, *, record_error=None, wait_error=None, alive=None):
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda: order.append("current_stream") or "compute-stream",
    )
    monkeypatch.setattr(
        torch.cuda,
        "Event",
        lambda **kwargs: _FakeEvent(
            order,
            record_error=record_error,
            wait_error=wait_error,
            alive=alive,
            **kwargs,
        ),
    )


def test_expected_hybridep_path_is_the_existing_task_source():
    expected = (
        Path(__file__).resolve().parents[2]
        / "torchtitan/distributed/deepep/hybridep.py"
    )
    assert diagnostic._expected_hybridep_path() == expected
    assert expected.is_file()
    assert hashlib.sha256(expected.read_bytes()).hexdigest() == diagnostic._HYBRIDEP_SHA256


def test_selected_call_waits_after_native_join_and_keeps_objects_live(monkeypatch):
    order = []
    state, _ = _adapter_state()

    class Payload:
        pass

    handle = Payload()
    compute_input = Payload()
    fn = _selected_function(diagnostic._expected_hybridep_path(), handle)
    references = [weakref.ref(fn), weakref.ref(compute_input), weakref.ref(handle)]
    external = {"fn": fn, "input": compute_input}
    returned = []

    def original(fn_arg, *produced_on_compute):
        assert fn_arg is references[0]()
        assert produced_on_compute[0] is references[1]()
        order.append("native_enter")
        external.clear()
        output = Payload()
        returned.append(weakref.ref(output))
        order.append("native_join_return")
        return output

    state.original = original
    class SpyLog(io.StringIO):
        def flush(self):
            order.append("log_flush")
            super().flush()

    state.output = SpyLog()
    _patch_event_api(monkeypatch, order, alive=[*references, lambda: returned[0]()])
    del fn, compute_input, handle
    real_write = diagnostic._write_record

    def write(state_arg, record_type, **fields):
        order.append(f"log_{record_type}")
        real_write(state_arg, record_type, **fields)

    monkeypatch.setattr(diagnostic, "_write_record", write)

    output = diagnostic._call_adapter(state, external["fn"], external["input"])

    assert output is returned[0]()
    assert order == [
        "native_enter",
        "native_join_return",
        "log_checkpoint_enter",
        "log_flush",
        "current_stream",
        "event_create",
        "event_record",
        "event_synchronize",
        "log_checkpoint_complete",
        "log_flush",
    ]
    assert order.count("native_enter") == 1
    assert state.global_call_ordinal == 1
    assert state.per_update_call_ordinal == 1
    assert [json.loads(line)["record_type"] for line in state.output.getvalue().splitlines()] == [
        "checkpoint_enter",
        "checkpoint_complete",
    ]


def test_unselected_and_inactive_calls_pass_through_without_cuda(monkeypatch):
    state, calls = _adapter_state(armed=False)
    sentinel = object()
    fn = lambda: sentinel
    state.original = lambda call, *args: calls.append((call, args)) or sentinel
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda: pytest.fail("inactive call touched CUDA"),
    )
    output = diagnostic._call_adapter(state, fn, sentinel)
    assert output is sentinel
    assert calls[0][0] is fn
    assert calls[0][1] == (sentinel,)
    assert state.global_call_ordinal == 0

    state.original = lambda call, *args: sentinel
    selected = _selected_function(diagnostic._expected_hybridep_path())
    assert diagnostic._call_adapter(state, selected) is sentinel
    assert state.global_call_ordinal == 1
    assert state.output.getvalue() == ""


def test_selected_call_preserves_none_output(monkeypatch):
    order = []
    state, _ = _adapter_state()
    state.original = lambda *_: None
    _patch_event_api(monkeypatch, order)
    assert (
        diagnostic._call_adapter(
            state,
            _selected_function(diagnostic._expected_hybridep_path()),
        )
        is None
    )


def test_selected_call_without_training_context_calls_native_once_then_fails():
    state, calls = _adapter_state()
    state.training_context = False
    with pytest.raises(RuntimeError, match="no valid native training context"):
        diagnostic._call_adapter(
            state,
            _selected_function(diagnostic._expected_hybridep_path()),
        )
    assert len(calls) == 1
    record = json.loads(state.output.getvalue().splitlines()[0])
    assert record["record_type"] == "invalid_training_context"


@pytest.mark.parametrize(
    ("stage", "record_type"),
    [
        ("native", "native_call_error"),
        ("record", "event_record_error"),
        ("wait", "event_wait_error"),
    ],
)
def test_stage_errors_are_recorded_and_reraised(monkeypatch, stage, record_type):
    order = []
    state, _ = _adapter_state()
    failure = RuntimeError(f"{stage} failed")
    selected = _selected_function(diagnostic._expected_hybridep_path())
    if stage == "native":
        state.original = lambda *_: (_ for _ in ()).throw(failure)
    else:
        state.original = lambda *_: object()
        _patch_event_api(
            monkeypatch,
            order,
            record_error=failure if stage == "record" else None,
            wait_error=failure if stage == "wait" else None,
        )

    with pytest.raises(RuntimeError, match=f"{stage} failed"):
        diagnostic._call_adapter(state, selected)

    records = [json.loads(line) for line in state.output.getvalue().splitlines()]
    assert records[-1]["record_type"] == record_type
    assert records[-1]["exception_text"] == f"{stage} failed"


def test_logging_failure_fails_before_event_record(monkeypatch):
    order = []
    state, _ = _adapter_state()
    state.original = lambda *_: object()
    _patch_event_api(monkeypatch, order)

    def fail_enter(*_args, **_kwargs):
        raise OSError("log flush failed")

    monkeypatch.setattr(diagnostic, "_write_record", fail_enter)
    with pytest.raises(OSError, match="log flush failed"):
        diagnostic._call_adapter(
            state, _selected_function(diagnostic._expected_hybridep_path())
        )
    assert order == []


def test_selector_source_mismatch_is_an_adapter_error(monkeypatch):
    state, calls = _adapter_state()
    selected = _selected_function(diagnostic._expected_hybridep_path(), firstlineno=650)
    with pytest.raises(RuntimeError, match="source location changed"):
        diagnostic._call_adapter(state, selected)
    assert calls == []
    assert json.loads(state.output.getvalue().splitlines()[0])["record_type"] == "selector_error"


def test_installer_pins_helper_and_restores_idempotently(monkeypatch, tmp_path):
    module, original, _ = _install_fake_hybridep(monkeypatch, tmp_path)

    restore = diagnostic.install_combine_bwd_completion()
    adapter = module._run_on_comm_stream
    assert adapter is not original
    with pytest.raises(RuntimeError, match="already installed"):
        diagnostic.install_combine_bwd_completion()

    restore()
    restore()
    assert module._run_on_comm_stream is original
    assert diagnostic._ACTIVE_STATE is None


def test_installer_rejects_buffer_and_invalid_source_pins(monkeypatch, tmp_path):
    module, _, path = _install_fake_hybridep(monkeypatch, tmp_path)
    expected_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match="not the task source"):
        diagnostic._verify_install_target(
            module,
            expected_path=path.with_name("other.py"),
            expected_sha256=expected_hash,
        )
    with pytest.raises(RuntimeError, match="source hash"):
        diagnostic._verify_install_target(
            module,
            expected_path=path,
            expected_sha256="0" * 64,
        )
    module._buffer = object()
    with pytest.raises(RuntimeError, match="before HybridEP buffer construction"):
        diagnostic._verify_install_target(
            module,
            expected_path=path,
            expected_sha256=expected_hash,
        )


def test_installer_rejects_unpinned_helper_signature(monkeypatch, tmp_path):
    module, _, path = _install_fake_hybridep(monkeypatch, tmp_path)

    def replacement(fn):
        return fn()

    replacement.__code__ = replacement.__code__.replace(
        co_filename=str(path), co_firstlineno=121
    )
    replacement.__module__ = module.__name__
    replacement.__name__ = diagnostic._HELPER_NAME
    replacement.__qualname__ = diagnostic._HELPER_NAME
    module._run_on_comm_stream = replacement

    with pytest.raises(RuntimeError, match="communication helper"):
        diagnostic._verify_install_target(
            module,
            expected_path=path,
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )


def test_installer_rejects_a_preexisting_helper_wrapper(monkeypatch, tmp_path):
    module, original, path = _install_fake_hybridep(monkeypatch, tmp_path)
    original.__wrapped__ = lambda *_: None

    with pytest.raises(RuntimeError, match="communication helper"):
        diagnostic._verify_install_target(
            module,
            expected_path=path,
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )


def test_restorer_does_not_overwrite_unexpected_replacement(monkeypatch, tmp_path):
    module, _, _ = _install_fake_hybridep(monkeypatch, tmp_path)
    restore = diagnostic.install_combine_bwd_completion()
    replacement = lambda *_: None
    module._run_on_comm_stream = replacement

    with pytest.raises(RuntimeError, match="left untouched"):
        restore()
    restore()
    assert module._run_on_comm_stream is replacement
    assert diagnostic._ACTIVE_STATE is None


@pytest.mark.parametrize("failure_stage", ["native", "record", "wait"])
def test_restore_action_returns_original_helper_after_call_error(
    monkeypatch, tmp_path, failure_stage
):
    module, original, _ = _install_fake_hybridep(monkeypatch, tmp_path)
    restore = diagnostic.install_combine_bwd_completion()
    adapter_state = diagnostic._ACTIVE_STATE
    adapter_state.output = io.StringIO()
    adapter_state.fit_started = True
    adapter_state.training_context = True
    adapter_state.step = 21
    adapter_state.microbatch_idx = 0
    adapter_state.armed = True
    failure = RuntimeError(f"{failure_stage} failed")
    if failure_stage == "native":
        adapter_state.original = lambda *_: (_ for _ in ()).throw(failure)
    else:
        adapter_state.original = lambda *_: object()
        _patch_event_api(
            monkeypatch,
            [],
            record_error=failure if failure_stage == "record" else None,
            wait_error=failure if failure_stage == "wait" else None,
        )

    try:
        with pytest.raises(RuntimeError, match=f"{failure_stage} failed"):
            diagnostic._call_adapter(
                adapter_state,
                _selected_function(diagnostic._expected_hybridep_path()),
            )
    finally:
        restore()
    assert module._run_on_comm_stream is original
    assert diagnostic._ACTIVE_STATE is None


def test_callback_pairing_fails_without_entrypoint_state():
    with pytest.raises(RuntimeError, match="dedicated training entrypoint"):
        diagnostic.CampaignCombineBwdCompletion(CampaignCombineBwdCompletionConfig())


def test_callback_pairing_rejects_a_second_observer(monkeypatch):
    adapter_state = diagnostic._AdapterState(
        module=ModuleType(diagnostic._HYBRIDEP_MODULE),
        original=lambda fn, *_: fn(),
    )
    monkeypatch.setattr(diagnostic, "_ACTIVE_STATE", adapter_state)
    diagnostic.CampaignCombineBwdCompletion(CampaignCombineBwdCompletionConfig())
    with pytest.raises(RuntimeError, match="exactly one"):
        diagnostic.CampaignCombineBwdCompletion(CampaignCombineBwdCompletionConfig())


def test_entrypoint_requires_exactly_one_callback_before_scan_install(monkeypatch):
    class Loaded:
        trainer_config = SimpleNamespace(callbacks={}, comm=SimpleNamespace(mode="default"))

    called = []
    monkeypatch.setattr(diagnostic_entrypoint, "init_logger", lambda: None)
    monkeypatch.setattr(diagnostic_entrypoint, "get_argv", lambda _: ["config.yaml"])
    monkeypatch.setattr(
        diagnostic_entrypoint,
        "load_trainer_config",
        lambda *_args, **_kwargs: Loaded(),
    )
    monkeypatch.setattr(
        diagnostic_entrypoint,
        "install_hybrid_ep_scan_fix",
        lambda: called.append("scan"),
    )

    with pytest.raises(ValueError, match="exactly one"):
        diagnostic_entrypoint.main([])
    assert called == []


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_entrypoint_restores_after_native_failure_without_masking_it(
    monkeypatch, cleanup_fails
):
    callback = CampaignCombineBwdCompletionConfig()
    trainer_config = SimpleNamespace(
        callbacks={"completion": callback},
        comm=SimpleNamespace(mode="default"),
        checkpoint=SimpleNamespace(create_seed_checkpoint=False),
    )
    loaded = SimpleNamespace(trainer_config=trainer_config)
    monkeypatch.setattr(diagnostic_entrypoint, "init_logger", lambda: None)
    monkeypatch.setattr(diagnostic_entrypoint, "get_argv", lambda _: ["config.yaml"])
    monkeypatch.setattr(
        diagnostic_entrypoint,
        "load_trainer_config",
        lambda *_args, **_kwargs: loaded,
    )
    monkeypatch.setattr(diagnostic_entrypoint, "install_hybrid_ep_scan_fix", lambda: None)
    cleanup_calls = []
    primary = RuntimeError("native training failed")

    def restore():
        cleanup_calls.append("restore")
        if cleanup_fails:
            raise RuntimeError("restore failed")

    monkeypatch.setattr(
        diagnostic,
        "install_combine_bwd_completion",
        lambda: restore,
    )
    train_module = ModuleType("torchtitan.train")

    def native_main(_argv):
        raise primary

    train_module.main = native_main
    monkeypatch.setitem(sys.modules, "torchtitan.train", train_module)

    with pytest.raises(RuntimeError, match="native training failed") as error:
        diagnostic_entrypoint.main([])
    assert error.value is primary
    assert cleanup_calls == ["restore"]
    if cleanup_fails:
        assert any("restore failed" in note for note in primary.__notes__)


@pytest.mark.parametrize("value", ["", "   ", ".", "../escape", "a/../b", "bad\x00path"])
def test_schema_rejects_invalid_output_paths(value):
    with pytest.raises(ConfigFieldError) as error:
        CampaignCombineBwdCompletionConfig(output_dir=value)
    assert error.value.field == "output_dir"


@pytest.mark.parametrize(
    ("native_steps", "accepted"),
    [(40, False), (60, True), (61, False)],
)
def test_fit_start_requires_native_stop_at_window_end(
    monkeypatch, tmp_path, native_steps, accepted
):
    _patch_target_dist(monkeypatch)
    monkeypatch.setenv("TORCHINDUCTOR_CUDAGRAPHS", "0")
    adapter_state = diagnostic._AdapterState(
        module=ModuleType(diagnostic._HYBRIDEP_MODULE),
        original=lambda fn, *_: object(),
    )
    adapter_state.module.__file__ = str(diagnostic._expected_hybridep_path())
    monkeypatch.setattr(diagnostic, "_ACTIVE_STATE", adapter_state)
    observer = diagnostic.CampaignCombineBwdCompletion(
        CampaignCombineBwdCompletionConfig()
    )
    state = _valid_observer_state(tmp_path)
    state.config.training.steps = native_steps

    if accepted:
        observer.fit_start(state)
        observer.close()
        activation = json.loads(
            (tmp_path / "combine_bwd_completion" / "rank-00000.jsonl")
            .read_text()
            .splitlines()[0]
        )
        assert activation["window"] == {"first_update": 21, "last_update": 60}
    else:
        with pytest.raises(RuntimeError, match="native training.steps=60"):
            observer.fit_start(state)


def test_observer_records_fixed_boundaries_and_waits_only_in_window(
    monkeypatch, tmp_path
):
    _patch_target_dist(monkeypatch)
    monkeypatch.setenv("TORCHINDUCTOR_CUDAGRAPHS", "0")
    adapter_state = diagnostic._AdapterState(
        module=ModuleType(diagnostic._HYBRIDEP_MODULE),
        original=lambda fn, *_: object(),
    )
    adapter_state.module.__file__ = str(diagnostic._expected_hybridep_path())
    monkeypatch.setattr(diagnostic, "_ACTIVE_STATE", adapter_state)
    observer = diagnostic.CampaignCombineBwdCompletion(
        CampaignCombineBwdCompletionConfig(output_dir="completion")
    )
    start_state = _valid_observer_state(tmp_path)
    observer.fit_start(start_state)

    order = []
    _patch_event_api(monkeypatch, order)
    monkeypatch.setattr(dist, "barrier", lambda: pytest.fail("diagnostic used a barrier"))
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda *_args, **_kwargs: pytest.fail("diagnostic used device-wide sync"),
    )

    call_ordinals = {}
    for step in (20, 21, 48, 60, 61):
        batch_state = _valid_observer_state(tmp_path, step=step)
        observer.batch_start(batch_state)
        for microbatch_idx in range(4):
            microbatch_state = _valid_observer_state(
                tmp_path, step=step, microbatch_idx=microbatch_idx
            )
            observer.microbatch_start(microbatch_state)
            if (step, microbatch_idx) in {
                (20, 0),
                (21, 2),
                (48, 0),
                (60, 3),
                (61, 0),
            }:
                selected = _selected_function(diagnostic._expected_hybridep_path())
                diagnostic._call_adapter(adapter_state, selected)
                call_ordinals[(step, microbatch_idx)] = adapter_state.global_call_ordinal
        observer.after_optimizer_step(batch_state)
        observer.batch_end(batch_state)

    observer.close()
    records = [
        json.loads(line)
        for line in (tmp_path / "completion" / "rank-00000.jsonl").read_text().splitlines()
    ]
    checkpoint_steps = [
        record["step"]
        for record in records
        if record["record_type"] == "checkpoint_complete"
    ]
    assert checkpoint_steps == [21, 48, 60]
    assert call_ordinals == {
        (20, 0): 1,
        (21, 2): 2,
        (48, 0): 3,
        (60, 3): 4,
        (61, 0): 5,
    }
    assert [
        record["selected_calls"]
        for record in records
        if record["record_type"] == "microbatch_end" and record["step"] == 60
    ] == [0, 0, 0, 1]
    assert all(
        record["record_type"] != "checkpoint_enter"
        or record["step"] in {21, 48, 60}
        for record in records
    )
    update_start = {
        record["step"]: record["armed"]
        for record in records
        if record["record_type"] == "update_start"
    }
    assert update_start == {20: False, 21: True, 48: True, 60: True, 61: False}


def test_observer_requires_runtime_call_activation_for_complete_result(
    monkeypatch, tmp_path
):
    _patch_target_dist(monkeypatch)
    monkeypatch.setenv("TORCHINDUCTOR_CUDAGRAPHS", "0")
    adapter_state = diagnostic._AdapterState(
        module=ModuleType(diagnostic._HYBRIDEP_MODULE),
        original=lambda fn, *_: object(),
    )
    adapter_state.module.__file__ = str(diagnostic._expected_hybridep_path())
    monkeypatch.setattr(diagnostic, "_ACTIVE_STATE", adapter_state)
    observer = diagnostic.CampaignCombineBwdCompletion(CampaignCombineBwdCompletionConfig())
    observer.fit_start(_valid_observer_state(tmp_path))

    for step in range(21, 61):
        batch_state = _valid_observer_state(tmp_path, step=step)
        observer.batch_start(batch_state)
        for microbatch_idx in range(4):
            observer.microbatch_start(
                _valid_observer_state(
                    tmp_path, step=step, microbatch_idx=microbatch_idx
                )
            )
        observer.after_optimizer_step(batch_state)
        observer.batch_end(batch_state)

    with pytest.raises(RuntimeError, match="activation evidence is incomplete"):
        observer.fit_end(_valid_observer_state(tmp_path, step=60))
    observer.close()
    records = [
        json.loads(line)
        for line in (tmp_path / "combine_bwd_completion" / "rank-00000.jsonl")
        .read_text()
        .splitlines()
    ]
    result = [record for record in records if record["record_type"] == "diagnostic_result"]
    assert result[0]["status"] == "invalid"


def test_observer_invalidates_active_microbatch_without_selected_calls(
    monkeypatch, tmp_path
):
    _patch_target_dist(monkeypatch)
    monkeypatch.setenv("TORCHINDUCTOR_CUDAGRAPHS", "0")
    adapter_state = diagnostic._AdapterState(
        module=ModuleType(diagnostic._HYBRIDEP_MODULE),
        original=lambda fn, *_: object(),
    )
    adapter_state.module.__file__ = str(diagnostic._expected_hybridep_path())
    monkeypatch.setattr(diagnostic, "_ACTIVE_STATE", adapter_state)
    observer = diagnostic.CampaignCombineBwdCompletion(CampaignCombineBwdCompletionConfig())
    observer.fit_start(_valid_observer_state(tmp_path))
    order = []
    _patch_event_api(monkeypatch, order)

    for step in range(21, 61):
        batch_state = _valid_observer_state(tmp_path, step=step)
        observer.batch_start(batch_state)
        for microbatch_idx in range(4):
            observer.microbatch_start(
                _valid_observer_state(
                    tmp_path, step=step, microbatch_idx=microbatch_idx
                )
            )
            if step == 21 and microbatch_idx == 0:
                diagnostic._call_adapter(
                    adapter_state,
                    _selected_function(diagnostic._expected_hybridep_path()),
                )
        observer.after_optimizer_step(batch_state)
        observer.batch_end(batch_state)

    with pytest.raises(RuntimeError, match="activation evidence is incomplete"):
        observer.fit_end(_valid_observer_state(tmp_path, step=60))
    observer.close()
    result = [
        json.loads(line)
        for line in (tmp_path / "combine_bwd_completion" / "rank-00000.jsonl")
        .read_text()
        .splitlines()
        if json.loads(line)["record_type"] == "diagnostic_result"
    ][0]
    assert result["status"] == "invalid"
    assert result["zero_call_microbatches"]
    assert {"step": 21, "microbatch_idx": 1} in result["zero_call_microbatches"]


def test_observer_completes_when_every_active_microbatch_is_observed(
    monkeypatch, tmp_path
):
    _patch_target_dist(monkeypatch)
    monkeypatch.setenv("TORCHINDUCTOR_CUDAGRAPHS", "0")
    adapter_state = diagnostic._AdapterState(
        module=ModuleType(diagnostic._HYBRIDEP_MODULE),
        original=lambda fn, *_: object(),
    )
    adapter_state.module.__file__ = str(diagnostic._expected_hybridep_path())
    monkeypatch.setattr(diagnostic, "_ACTIVE_STATE", adapter_state)
    observer = diagnostic.CampaignCombineBwdCompletion(CampaignCombineBwdCompletionConfig())
    observer.fit_start(_valid_observer_state(tmp_path))
    _patch_event_api(monkeypatch, [])

    for step in range(21, 61):
        batch_state = _valid_observer_state(tmp_path, step=step)
        observer.batch_start(batch_state)
        for microbatch_idx in range(4):
            observer.microbatch_start(
                _valid_observer_state(
                    tmp_path, step=step, microbatch_idx=microbatch_idx
                )
            )
            diagnostic._call_adapter(
                adapter_state,
                _selected_function(diagnostic._expected_hybridep_path()),
            )
        observer.after_optimizer_step(batch_state)
        observer.batch_end(batch_state)

    observer.fit_end(_valid_observer_state(tmp_path, step=60))
    observer.close()
    result = [
        json.loads(line)
        for line in (tmp_path / "combine_bwd_completion" / "rank-00000.jsonl")
        .read_text()
        .splitlines()
        if json.loads(line)["record_type"] == "diagnostic_result"
    ][0]
    assert result["status"] == "complete"
    assert result["active_selected_calls"] == 160
    assert result["active_completed_checkpoints"] == 160


@pytest.mark.parametrize("primary_active", [False, True])
def test_close_closes_log_even_when_write_and_flush_fail(
    monkeypatch, tmp_path, primary_active
):
    _patch_target_dist(monkeypatch)
    monkeypatch.setenv("TORCHINDUCTOR_CUDAGRAPHS", "0")
    adapter_state = diagnostic._AdapterState(
        module=ModuleType(diagnostic._HYBRIDEP_MODULE),
        original=lambda fn, *_: object(),
    )
    adapter_state.module.__file__ = str(diagnostic._expected_hybridep_path())
    monkeypatch.setattr(diagnostic, "_ACTIVE_STATE", adapter_state)
    observer = diagnostic.CampaignCombineBwdCompletion(CampaignCombineBwdCompletionConfig())
    observer.fit_start(_valid_observer_state(tmp_path))
    adapter_state.output.close()

    class BrokenLog:
        closed = False

        def write(self, _text):
            raise OSError("write failed")

        def flush(self):
            raise OSError("flush failed")

        def close(self):
            self.closed = True

    broken_log = BrokenLog()
    adapter_state.output = broken_log
    if primary_active:
        primary = RuntimeError("native failure")
        with pytest.raises(RuntimeError, match="native failure") as error:
            try:
                raise primary
            except RuntimeError:
                observer.close()
                raise
        assert error.value is primary
        assert any("write failed" in note for note in primary.__notes__)
    else:
        with pytest.raises(OSError, match="write failed"):
            observer.close()
    assert broken_log.closed
    assert adapter_state.output is None
    assert adapter_state.closed


def test_test_custom_op_resolves_replaced_module_global_in_eager_and_aot_backward():
    prior_fake_calls = list(_combine_bwd_test_fake_calls)
    _combine_bwd_test_fake_calls.clear()
    runtime_calls = []

    def original_helper(fn, *_produced_on_compute):
        return fn()

    def recording_helper(fn, *produced_on_compute):
        runtime_calls.append((fn, produced_on_compute))
        return original_helper(fn, *produced_on_compute)

    module = sys.modules[__name__]
    prior_helper = module._combine_bwd_test_helper

    def run_with_helper(helper, *, compiled: bool):
        module._combine_bwd_test_helper = helper
        value = torch.tensor([1.0, -3.0, 5.0], requires_grad=True)
        if compiled:
            loss = torch.compile(
                _combine_bwd_test_loss,
                backend="aot_eager",
                fullgraph=True,
            )(value)
        else:
            loss = _combine_bwd_test_loss(value)
        loss.backward()
        return loss.detach(), value.grad.detach().clone()

    try:
        eager_baseline = run_with_helper(original_helper, compiled=False)
        assert runtime_calls == []
        torch.testing.assert_close(eager_baseline[1], torch.full((3,), 2.0))

        compiled_baseline = run_with_helper(original_helper, compiled=True)
        assert _combine_bwd_test_fake_calls
        assert runtime_calls == []

        runtime_calls.clear()
        compiled_actual = run_with_helper(recording_helper, compiled=True)
        assert runtime_calls
        torch.testing.assert_close(compiled_actual[0], compiled_baseline[0])
        torch.testing.assert_close(compiled_actual[1], compiled_baseline[1])

        eager_actual = run_with_helper(recording_helper, compiled=False)
        assert len(runtime_calls) >= 2
        torch.testing.assert_close(eager_actual[0], eager_baseline[0])
        torch.testing.assert_close(eager_actual[1], eager_baseline[1])
    finally:
        module._combine_bwd_test_helper = prior_helper
        _combine_bwd_test_fake_calls[:] = prior_fake_calls


_COMBINE_BWD_TEST_NAMESPACE = "campaign_combine_bwd_completion_test"
_combine_bwd_test_fake_calls: list[str] = []
_combine_bwd_test_helper = lambda fn, *_produced_on_compute: fn()


@torch.library.custom_op(
    f"{_COMBINE_BWD_TEST_NAMESPACE}::forward",
    mutates_args=(),
)
def _combine_bwd_test_forward(value: torch.Tensor) -> torch.Tensor:
    return value.clone()


@_combine_bwd_test_forward.register_fake
def _combine_bwd_test_forward_fake(value: torch.Tensor) -> torch.Tensor:
    _combine_bwd_test_fake_calls.append("forward")
    return torch.empty_like(value)


@torch.library.custom_op(
    f"{_COMBINE_BWD_TEST_NAMESPACE}::backward",
    mutates_args=(),
)
def _combine_bwd_test_backward(gradient: torch.Tensor) -> torch.Tensor:
    return _combine_bwd_test_helper(lambda: gradient * 2, gradient)


@_combine_bwd_test_backward.register_fake
def _combine_bwd_test_backward_fake(gradient: torch.Tensor) -> torch.Tensor:
    _combine_bwd_test_fake_calls.append("backward")
    return torch.empty_like(gradient)


def _combine_bwd_test_setup_context(ctx, inputs, output) -> None:
    del ctx, inputs, output


def _combine_bwd_test_autograd(ctx, gradient: torch.Tensor):
    del ctx
    return torch.ops.campaign_combine_bwd_completion_test.backward(gradient)


_combine_bwd_test_forward.register_autograd(
    _combine_bwd_test_autograd,
    setup_context=_combine_bwd_test_setup_context,
)


def _combine_bwd_test_loss(value: torch.Tensor) -> torch.Tensor:
    return torch.ops.campaign_combine_bwd_completion_test.forward(value).sum()
