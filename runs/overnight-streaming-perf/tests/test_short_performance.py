"""CPU checks for the monitoring-free short performance protocol."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import measure
import parse_results
import run as runner
from test_compilation_window import _parser_records
from test_measure import _adapter


@pytest.mark.parametrize(
    "option",
    [
        "--native-numerics",
        "--native-reference-output",
        "--native-reference-input",
        "--native-capture-update",
    ],
)
def test_removed_modes_are_rejected(option: str, tmp_path: Path) -> None:
    value = (
        []
        if option == "--native-numerics"
        else ["1" if option == "--native-capture-update" else str(tmp_path)]
    )
    with pytest.raises(SystemExit):
        runner._parse_args(["--run-id", "removed", option, *value])
    with pytest.raises(SystemExit):
        measure._parse_args(
            [
                "--run-id",
                "removed",
                "--measurement-dir",
                str(tmp_path),
                "--steps",
                "20",
                "--warmup",
                "10",
                "--window",
                "10",
                option,
                *value,
                "--",
                "experiment.yaml",
            ]
        )


def test_short_command_resolves_preserved_recipe_without_monitoring(
    tmp_path: Path,
) -> None:
    from torchtitan.config import run_pre_submit
    from torchtitan.config.build import prepare_document
    from torchtitan.config.cli import load_document
    from torchtitan.config.schema.callbacks.builtin.ema import EMAConfig

    args = runner._parse_args(["--run-id", "short"])
    assert (args.steps, args.warmup, args.window) == (20, 10, 10)
    assert args.observe_compiler_cache is False
    canonical = run_pre_submit(str(runner.BASE_CONFIG), world_size=8)
    assert canonical.ok, canonical.report.errors
    assert set(canonical.data["callbacks"]) == {"ema"}

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    experiment = config_dir / "experiment.yaml"
    runner._write_experiment_config(experiment)
    paths = {
        "config": config_dir,
        "native": tmp_path / "native",
        "measurement": tmp_path / "measurement",
        "logs": tmp_path / "logs",
    }
    command = runner._command(args, paths)
    helper_start = command.index(str(runner.HELPER_DIR / "measure.py")) + 1
    helper_args, native_args = measure._parse_args(command[helper_start:])
    assert (helper_args.steps, helper_args.warmup, helper_args.window) == (20, 10, 10)
    assert helper_args.monitor_interval == 0
    assert not helper_args.observe_compiler_cache
    assert not helper_args.capture_microbatches
    pairs = list(zip(native_args[1::2], native_args[2::2]))
    checked = run_pre_submit(
        native_args[0], [(key[2:], value) for key, value in pairs], world_size=8
    )
    assert checked.ok, checked.report.errors
    actual = checked.data
    expected = dict(canonical.data)
    # Only these sections contain intentional run-local overrides.
    for section in (
        "training",
        "profiler",
        "dump_folder",
        "hf_assets_path",
        "dataloader",
    ):
        expected.pop(section, None)
    assert {key: actual[key] for key in expected} == expected
    assert actual["training"] == {**canonical.data["training"], "steps": 20}
    assert actual["lr_scheduler"]["total_steps"] == 65000
    assert actual["lr_scheduler"]["warmup_steps"] == 2000
    assert actual["debug"]["seed"] == 2048
    assert actual["training"]["local_batch_size"] == 2
    assert actual["training"]["global_batch_size"] == 512
    assert actual["training"]["seq_len"] == 4096
    assert 512 // (2 * 8) == 32
    assert set(actual["callbacks"]) == {"ema"}
    assert actual["loss"]["num_chunks"] == 2
    assert actual["loss"].get("num_chunks_on_monitoring_steps") is None
    assert actual["loss"]["type"] == "mtp"
    assert actual["loss"]["loss_weights"] == [0.3]
    assert actual["checkpoint"]["enable"] is False
    assert actual["validator"]["enable"] is False
    assert actual["profiler"]["enable_profiling"] is False
    assert actual["profiler"]["enable_memory_snapshot"] is False
    document, _ = load_document(
        native_args[0], [(key[2:], value) for key, value in pairs]
    )
    config = prepare_document(document).trainer_config
    assert isinstance(config.callbacks["ema"], EMAConfig)
    assert (
        config.callbacks["ema"].decay,
        config.callbacks["ema"].update_period,
        config.callbacks["ema"].start_step,
    ) == (0.99, 10, 10000)
    native_config = paths["native"] / "config"
    native_config.mkdir(parents=True)
    from torchtitan.config.save import dump_yaml, get_typed_document

    prepared = prepare_document(document)
    (native_config / "resolved_config.yaml").write_text(
        dump_yaml(get_typed_document(config, prepared.model_section))
    )
    assert runner._verify_resolved_config(paths, args)


def test_short_adapter_counts_optimizer_updates_without_monitoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "8")
    trainer_class, adapter, _, syncs = _adapter(
        tmp_path,
        steps=20,
        warmup=10,
        window=10,
        monitor_interval=0,
        capture_microbatches=True,
    )
    with adapter:
        trainer_class().train()
    import json

    record = json.loads((tmp_path / "measurement/rank-00.json").read_text())
    assert record["completed_steps"] == 20
    assert record["microbatch_calls_by_step"] == {
        str(step): 32 for step in range(1, 21)
    }
    assert record["measurement"]["monitoring_updates"] == []
    assert not any(item["monitoring_step"] for item in record["cadence"])
    assert syncs == [1, 1]
    assert "compilation_observer" not in record["measurement"]
    assert "compiler_guard" not in record


@pytest.mark.parametrize("window", [1, 10, 12])
def test_short_parser_accepts_positive_windows_without_cache_gate(window: int) -> None:
    records, metadata = _parser_records()
    metadata.update(
        steps=10 + window, warmup_updates=10, window_updates=window, monitor_interval=0
    )
    metadata.pop("deep_gemm_compilation_guard")
    for record in records:
        record.update(expected_steps=10 + window, completed_steps=10 + window)
        record.pop("compiler_guard")
        measurement = record["measurement"]
        measurement.pop("compilation_observer")
        measurement.update(
            warmup_updates=10,
            window_updates=window,
            monitoring_updates=[],
            physical_input_tokens_local=window
            * parse_results.LOCAL_PHYSICAL_TOKENS_PER_UPDATE,
            supervised_tokens_global=window
            * parse_results.GLOBAL_SUPERVISED_TOKENS_PER_UPDATE,
        )
    summary = parse_results.summarize_records(records, metadata)
    assert summary["status"] == "complete", summary["errors"]
    assert summary["compilation_observer"]["status"] == "disabled"
    assert summary["deep_gemm_compilation_guard"]["status"] == "disabled"


def test_native_step_builds_only_ema_and_uses_ordinary_mtp_chunks() -> None:
    import torch
    from torchtitan.callbacks.manager import CallbackManager
    from torchtitan.callbacks.builtin.ema import EMA
    from torchtitan.callbacks.builtin.monitoring import MonitoringCallback
    from torchtitan.components.loss import MTPLoss
    from torchtitan.config import load_raw_document
    from torchtitan.config.build import prepare_document
    from torchtitan.trainer import Trainer

    config = prepare_document(load_raw_document(str(runner.BASE_CONFIG))).trainer_config
    store = SimpleNamespace(
        use_ema=False, holds_average=False, updates=[], seed=lambda: None
    )
    store.update = lambda weight: store.updates.append(weight)
    backend = SimpleNamespace(ema_store=lambda parts: store)
    head = torch.nn.Linear(3, 5)
    manager = CallbackManager(
        config.callbacks, model_parts=[head], sharding_backend=backend
    )
    assert manager.find(MonitoringCallback) is None
    assert isinstance(manager.find(EMA), EMA)
    assert len(manager.handler.callbacks) == 1
    assert not head._forward_hooks and not head._backward_hooks

    loss_fn = MTPLoss(config.loss)  # Small CPU fixture exercises the eager native loss.
    loss_fn.set_lm_head(head)
    native_optimizer = torch.optim.AdamW(head.parameters(), lr=0.001)
    trainer = object.__new__(Trainer)
    trainer.config = config
    trainer.callbacks = manager
    trainer.model_parts = [head]
    trainer.loss_fn = loss_fn
    trainer.step = 50  # This used to select the monitoring-specific 32 chunks.
    trainer.device = torch.device("cpu")
    trainer.gradient_accumulation_steps = 32
    trainer.ntokens_seen = torch.zeros((), dtype=torch.int64)
    trainer.parallel_dims = SimpleNamespace(
        dp_enabled=False,
        pp_enabled=False,
        ep_enabled=False,
        dp_cp_enabled=False,
        get_optional_mesh=lambda _: None,
    )
    trainer.optimizers = SimpleNamespace(
        optimizers=[native_optimizer],
        zero_grad=native_optimizer.zero_grad,
        step=native_optimizer.step,
    )
    trainer.lr_schedulers = SimpleNamespace(
        schedulers=[SimpleNamespace(get_last_lr=lambda: [0.001])], step=lambda: None
    )
    trainer.checkpointer = SimpleNamespace(
        maybe_wait_for_staging=lambda: None, last_saved_path=None
    )
    logs = []
    trainer.metrics_processor = SimpleNamespace(
        ntokens_since_last_log=0,
        should_log=lambda _: True,
        log_extra_metrics=lambda *_args, **_kwargs: None,
        log=lambda *args, **kwargs: logs.append((args, kwargs)),
    )
    trainer.aux_loss = SimpleNamespace(
        begin_step=loss_fn.reset_components, metrics=lambda: {}
    )
    seen_chunks = []

    def forward_backward(**kwargs):
        seen_chunks.append(loss_fn.num_chunks)
        hidden = torch.ones(2, 1, 4, 3, requires_grad=True)
        loss = loss_fn(hidden, kwargs["labels"], kwargs["global_valid_tokens"])
        loss.backward()
        return loss

    trainer.forward_backward_step = forward_backward
    # Keep the native trainer's batch mutation, loss routing, accumulation,
    # optimizer, EMA dispatch and native metric logging on the exercised path.
    batches = iter(
        [
            (
                {"tokens": torch.ones(1, 4, dtype=torch.int64)},
                torch.tensor([[0, 1, 2, 3]]),
            )
            for _ in range(32)
        ]
    )
    Trainer.train_step(trainer, batches)
    assert seen_chunks == [2] * 32
    assert loss_fn.loss_weights == [0.3]
    assert loss_fn.num_chunks == 2
    assert int(trainer.ntokens_seen) == 128
    assert trainer.metrics_processor.ntokens_since_last_log == 128
    assert len(logs) == 1 and torch.isfinite(torch.tensor(logs[0][0][1]))
    assert set(loss_fn.declared_components()) == {"loss_ce", "loss_mtp"}
    assert store.updates == []  # The preserved EMA start_step is 10000.
    trainer.step = 10000
    Trainer.train_step(
        trainer,
        iter(
            [
                (
                    {"tokens": torch.ones(1, 4, dtype=torch.int64)},
                    torch.tensor([[0, 1, 2, 3]]),
                )
                for _ in range(32)
            ]
        ),
    )
    assert store.updates == [pytest.approx(0.01)]
    assert not head._forward_hooks and not head._backward_hooks
    manager.close()
