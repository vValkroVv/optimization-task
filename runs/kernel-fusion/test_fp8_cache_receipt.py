"""Exercise production rank dispatch and the native receipt with CPU doubles."""

from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    ("fp8_enabled", "cache_state", "error"),
    [
        (False, "active", None),
        (False, "missing", "BF16 expert-weight cache is inactive"),
        (True, "active", None),
        (True, "legacy", "BF16 expert-weight cache is inactive"),
        (True, "missing", "Requested FP8 expert cache is inactive"),
        (True, "unprepared", "FP8 expert cache has no prepared and updated state"),
        (True, "unrequested", "FP8 expert cache was not requested"),
    ],
)
def test_production_rank_cache_receipt(
    tmp_path, monkeypatch, fp8_enabled, cache_state, error
):
    runner_path = Path(__file__).with_name("run.py")
    root = runner_path.resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "runs/overnight-streaming-perf/tests"))
    helpers = import_module("test_measure")
    bf16_helpers = import_module("test_bf16_cache_runner")
    spec = spec_from_file_location("_kernel_fusion_test_runner", runner_path)
    runner = module_from_spec(spec)
    spec.loader.exec_module(runner)

    baseline = runner.load(
        root / "runs/compile-baseline/run.py", "_compile_receipt_test"
    )
    bridge = runner.load(
        root / "runs/schedule-embedding/measure.py", "_schedule_receipt_test"
    )
    flash = runner.load(root / "runs/flash-qla/measure.py", "_flash_receipt_test")
    monkeypatch.setattr(runner, "load", lambda *args: baseline)
    monkeypatch.setattr(baseline, "load", lambda *args: bridge)
    if cache_state == "legacy":
        monkeypatch.setattr(runner, "install_fp8_rank_measurement", lambda *_, **__: None)
    stage_selections = []
    monkeypatch.setattr(
        bridge,
        "_record_stage_selection",
        lambda *args, **kwargs: stage_selections.append("S"),
    )

    events = []
    group = SimpleNamespace(
        _module_fqn="layers.0[efsdp]",
        _fp8_cache_requested=cache_state != "unrequested",
        _fp8_cache=None,
    )

    def iter_groups(part, *, recurse):
        assert recurse is True
        assert events == ["train returned"]
        return [part.group]

    monkeypatch.setitem(
        sys.modules,
        "torchtitan.distributed.gfsdp.fully_shard",
        SimpleNamespace(_iter_fsdp_param_groups=iter_groups),
    )
    argv = ["--", "config.yaml", "--dump_folder", str(tmp_path)]
    argv.extend((runner.FP8_CACHE_FLAG, str(fp8_enabled).lower()))
    monkeypatch.setattr(
        sys, "argv", [str(runner_path), "rank", runner.PRODUCTION, *argv]
    )
    for name in runner.SWITCHES.values():
        monkeypatch.delenv(name, raising=False)
    for name in runner.variant_switches(runner.PRODUCTION):
        monkeypatch.setenv(name, "1")
    monkeypatch.delenv(runner.DIRECT_DW_BUILD_ENV, raising=False)
    monkeypatch.setenv("TORCHDYNAMO_DISABLE", "0")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "8")

    def train_rank(train_argv):
        assert train_argv == argv
        bridge._patch_flash_measure(flash, None, {}, {}, "S")
        monkeypatch.setattr(helpers, "measure", flash._load_base_measure())
        trainer_class, adapter, _, sync_calls = helpers._adapter(
            tmp_path,
            steps=2,
            warmup=1,
            window=1,
            monitor_interval=0,
            capture_microbatches=False,
        )
        adapter.window_options["gfsdp_reuse_bf16_expert_weights"] = True
        original_train = trainer_class.train

        def train(trainer):
            original_train(trainer)
            assert sync_calls == [1, 1]
            if fp8_enabled and cache_state in {"active", "legacy", "unprepared"}:
                group._fp8_cache = SimpleNamespace(
                    generation=2,
                    prepared_generation=-1 if cache_state == "unprepared" else 1,
                    ready_event=None if cache_state == "unprepared" else object(),
                    writer_event=object(),
                )
            events.append("train returned")

        monkeypatch.setattr(trainer_class, "train", train)
        trainer = trainer_class()
        trainer.config = SimpleNamespace(
            parallelism=SimpleNamespace(gfsdp_fp8_expert_weights=fp8_enabled)
        )
        part = bf16_helpers.CacheStatsPart(trainer.model_parts[0].parameters())
        part.group = group
        part.active = not fp8_enabled and cache_state == "active"
        trainer.model_parts = [part]
        with adapter:
            trainer.train()

    monkeypatch.setattr(bridge, "main", train_rank)
    if error is None:
        assert runner.main() == 0
    else:
        with pytest.raises(RuntimeError, match=error):
            runner.main()
    assert stage_selections == ["S"]
    record = json.loads((tmp_path / "measurement/rank-00.json").read_text())
    assert record["completed_steps"] == 2
    if error is not None:
        assert record["measurement_status"] == "incomplete"
        assert error in record["training_error"]
        return

    assert record["measurement_status"] == "complete"
    assert record["measurement"]["cuda_synchronize_calls"] == 2
    (snapshot,) = record["bf16_expert_cache"]["snapshots"]
    assert snapshot["boundary"] == "after_train"
    assert any(row["enabled"] for row in snapshot["groups"]) is not fp8_enabled
    receipts = record["training_boundary_receipts"]
    assert receipts["before_train"] == receipts["after_train"]
    assert receipts["before_window"]["epoch"] == receipts["after_train"]
    assert receipts["after_train"]["dataset_available"] is False
    if fp8_enabled:
        assert receipts["after_train_fp8_expert_cache"] == {
            "update": 2,
            "groups": [
                {
                    "model_part": 0,
                    "fqn": "layers.0[efsdp]",
                    "generation": 2,
                    "prepared_generation": 1,
                }
            ],
        }
    else:
        assert "after_train_fp8_expert_cache" not in receipts

    records = [{**record, "rank": rank} for rank in range(8)]
    summary = helpers.parse_results.summarize_records(
        records, {"steps": 2, "warmup_updates": 1, "window_updates": 1}
    )
    assert summary["status"] == "complete", summary["errors"]
