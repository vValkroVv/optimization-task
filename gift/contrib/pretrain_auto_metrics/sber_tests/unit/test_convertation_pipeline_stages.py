import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file

from node_runner.src.config_model import (
    ConvertationConfig,
    PreparationConfig,
    QuantizationConfig,
)
from node_runner.src.convertation_manager import (
    ConvertationContext,
    ConvertationManager,
    MergeHFStage,
    MultiNodeConvertationStage,
    PipelineStage,
    PreparationStage,
    QuantizeStage,
    SingleNodeConvertationStage,
)

pytestmark = pytest.mark.unit


def _context(tmp_path: Path, **overrides) -> ConvertationContext:
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    artifacts_path = tmp_path / "artifacts"
    artifacts_path.mkdir(parents=True, exist_ok=True)
    ctx = ConvertationContext(
        convertation_config_path=artifacts_path / "convertation_config.yaml",
        hf_checkpoint_path=artifacts_path / "hf_checkpoint",
        checkpoint_path=checkpoint_path,
        artifacts_path=artifacts_path,
        prepared_checkpoint_path=artifacts_path / "prepared_checkpoint",
        tokenizer_path=artifacts_path / "tokenizer",
        log_file=artifacts_path / "convertation.log",
        originaly_hf=False,
        originally_fsdp=False,
        mappers=["map_fsdp_to_hf"],
        global_rank=0,
        world_size=2,
        convertation_config=ConvertationConfig(mappers=["map_fsdp_to_hf"]),
        preparation_config=PreparationConfig(skip=False),
        quantization_config=None,
        orkestration_manager=type(
            "OM", (), {"wait_barrier": lambda self, stage: None}
        )(),
        is_worker=False,
        is_master=True,
        skip_preparation=False,
    )
    for key, value in overrides.items():
        setattr(ctx, key, value)
    return ctx


def _patch_utils_convertation_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "node_runner.src.convertation_manager.PipelineStage._get_path_to_utils_convertation",
        lambda self: tmp_path,
    )


def _patch_sleep_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    sleeps: list[int] = []
    monkeypatch.setattr(
        "node_runner.src.convertation_manager.time.sleep", sleeps.append
    )
    return sleeps


def _patch_rmtree_calls(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    removed: list[Path] = []
    monkeypatch.setattr(
        "node_runner.src.convertation_manager.shutil.rmtree",
        lambda p, **kwargs: removed.append(Path(p)),
    )
    return removed


def _create_fp8_hf_checkpoint(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(
        '{"model_type":"llama","quantization_config":{"quant_method":"fp8"}}'
    )
    (path / "model.safetensors").write_text("fp8")


class TestConvertationManagerCore:
    def test_read_hf_config_missing_and_invalid_json(self, tmp_path: Path):
        missing = tmp_path / "missing_config"
        missing.mkdir()
        assert ConvertationManager._read_hf_config(missing) is None

        invalid = tmp_path / "invalid_config"
        invalid.mkdir()
        (invalid / "config.json").write_text("{")
        assert ConvertationManager._read_hf_config(invalid) is None

    def test_setup_paths_hf_and_non_hf(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        manager = ConvertationManager()
        ckpt = tmp_path / "ckpt"
        artifacts = tmp_path / "artifacts"
        ckpt.mkdir(parents=True, exist_ok=True)
        artifacts.mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(manager, "is_hf_checkpoint", lambda p: True)
        manager._setup_paths(ckpt, artifacts, ["map_gigafsdp_to_hf"])
        assert manager.originaly_hf is True
        assert manager.hf_checkpoint_path == ckpt
        assert manager.prepared_checkpoint_path == ckpt

        monkeypatch.setattr(manager, "is_hf_checkpoint", lambda p: False)
        manager._setup_paths(ckpt, artifacts, ["map_gigafsdp_to_hf"])
        assert manager.originaly_hf is False
        assert manager.hf_checkpoint_path == artifacts / "hf_checkpoint"
        assert manager.prepared_checkpoint_path == artifacts / "prepared_checkpoint"

    def test_dump_convertation_config_master_and_worker(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        manager = ConvertationManager()
        config = ConvertationConfig()
        path = tmp_path / "config.yaml"
        manager._dump_convertation_config(config, path, is_master=True)
        assert path.exists()
        sleeps = _patch_sleep_calls(monkeypatch)
        manager._dump_convertation_config(config, path, is_master=False)
        assert sleeps == [3]

    def test_build_context_and_pipeline(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        manager = ConvertationManager()
        manager.convertation_config_path = tmp_path / "c.yaml"
        manager.hf_checkpoint_path = tmp_path / "hf"
        manager.prepared_checkpoint_path = tmp_path / "prep"
        manager.tokenizer_path = tmp_path / "tok"
        manager.originaly_hf = False
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        ctx = manager._build_convertation_context(
            checkpoint_path=tmp_path / "ckpt",
            artifacts_path=tmp_path / "artifacts",
            convertation_config=ConvertationConfig(mappers=["map_fsdp_to_hf"]),
            nnodes=2,
            node_rank=1,
            mappers=["map_fsdp_to_hf"],
            convertation_log_file=tmp_path / "log.txt",
            preparation_config=PreparationConfig(skip=True),
            quantization_config=QuantizationConfig(),
            orkestration_manager=type(
                "OM", (), {"wait_barrier": lambda self, stage: None}
            )(),
            is_worker=True,
            is_master=False,
            skip_preparation=True,
        )
        manager.convertation_context = ctx
        assert ctx.quantization_config is not None
        stages = manager._build_pipeline()
        assert [type(s) for s in stages] == [
            PreparationStage,
            SingleNodeConvertationStage,
            MultiNodeConvertationStage,
            MergeHFStage,
            QuantizeStage,
        ]

    def test_run_pipeline_calls_stage_methods(self):
        manager = ConvertationManager()
        calls = []

        class _Stage:
            def run(self):
                calls.append("run")

            def barrier_after_run(self):
                calls.append("barrier")

        manager.pipeline = [_Stage(), _Stage()]
        manager.convertation_context = SimpleNamespace(
            hf_checkpoint_path=Path("/tmp/hf_checkpoint"),
            tokenizer_path=Path("/tmp/tokenizer"),
        )
        manager.run_pipeline()
        assert calls == ["run", "barrier", "run", "barrier"]


class TestPipelineStageCommonBehavior:
    def test_pipeline_stage_path_resolution(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        ctx = _context(tmp_path)
        original_get_path = PipelineStage._get_path_to_utils_convertation
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        stage = SingleNodeConvertationStage(ctx)
        assert stage.cwd == tmp_path

        monkeypatch.setattr(
            "node_runner.src.convertation_manager.PipelineStage._get_path_to_utils_convertation",
            original_get_path,
        )
        monkeypatch.setattr("sys.path", ["/a/b", "/x/utils/convertation"])
        assert (
            stage._get_path_to_utils_convertation()
            .as_posix()
            .endswith("utils/convertation")
        )

        monkeypatch.setattr("sys.path", ["/a/b"])
        with pytest.raises(ValueError, match="Utils/convertation not found"):
            stage._get_path_to_utils_convertation()

    def test_pipeline_stage_helpers_and_barrier(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        calls = []
        ctx = _context(
            tmp_path,
            mappers=["map_gigafsdp_to_hf_safetensor"],
            orkestration_manager=type(
                "OM", (), {"wait_barrier": lambda self, stage: calls.append(stage)}
            )(),
        )
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        stage = SingleNodeConvertationStage(ctx)
        assert stage.is_map_gigafsdp_to_hf_safetensor() is True
        assert stage.is_map_gigafsdp_to_hf() is False
        assert stage.is_map_mixed_gigafsdp_to_fsdp() is False
        stage.barrier_after_run()
        assert calls == ["SINGLENODE_CONVERTATION_DONE"]

        mtp_ctx = _context(
            tmp_path / "mtp",
            mappers=["map_mixed_gigafsdp_to_mtp_safetensors"],
        )
        mtp_stage = SingleNodeConvertationStage(mtp_ctx)
        assert mtp_stage.is_map_mixed_gigafsdp_to_mtp_safetensors() is True
        assert mtp_stage.is_fsdp_output_mapper() is True

    def test_update_and_save_convertation_config(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        ctx = _context(tmp_path)
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        stage = MultiNodeConvertationStage(ctx)
        stage.update_and_save_convertation_config({"x": 2})
        assert ctx.convertation_config.model_dump().get("x") == 2
        assert ctx.convertation_config_path.exists()

    def test_pipeline_stage_run_skip_and_execute(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        ctx = _context(tmp_path)
        _patch_utils_convertation_path(monkeypatch, tmp_path)

        class _DummyStage(PipelineStage):
            def __init__(self, context, do_skip=False):
                self._do_skip = do_skip
                self.after_called = False
                super().__init__("DUMMY", context)

            def skip(self):
                return self._do_skip

            def build_cmd_to_run(self):
                self.cmd_to_run = ["echo", "x"]

            def build_cwd(self):
                self.cwd = tmp_path

            def before_run(self):
                return None

            def after_run(self):
                self.after_called = True

        skipped = _DummyStage(ctx, do_skip=True)
        skipped.run()
        assert skipped.after_called is False

        calls = {"start": 0, "wait": 0, "raise": 0}

        class _SP:
            def __init__(self, cmd, log_file, cwd):
                assert cmd == ["echo", "x"]
                assert cwd == tmp_path

            def start(self):
                calls["start"] += 1

            def wait_for_completion(self):
                calls["wait"] += 1

            def raise_for_exit_code(self):
                calls["raise"] += 1

        monkeypatch.setattr(
            "node_runner.src.convertation_manager.SubprocessManager", _SP
        )
        executed = _DummyStage(ctx, do_skip=False)
        executed.run()
        assert calls == {"start": 1, "wait": 1, "raise": 1}
        assert executed.after_called is True

    def test_pipeline_stage_run_skips_for_original_hf(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        ctx = _context(
            tmp_path / "run_skip_original_hf",
            originaly_hf=True,
            originally_fsdp=False,
        )

        class _DummyStage(PipelineStage):
            def __init__(self, context):
                self.before_called = False
                self.after_called = False
                super().__init__("DUMMY_ORIGINAL_HF", context)

            def skip(self):
                return False

            def build_cmd_to_run(self):
                self.cmd_to_run = ["echo", "x"]

            def build_cwd(self):
                self.cwd = tmp_path

            def before_run(self):
                self.before_called = True

            def after_run(self):
                self.after_called = True

        stage = _DummyStage(ctx)
        stage.run()
        assert stage.before_called is False
        assert stage.after_called is False


class TestConcretePipelineStages:
    def test_stage_skip_methods_and_before_run(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        ctx = _context(tmp_path)
        (ctx.hf_checkpoint_path).mkdir(parents=True, exist_ok=True)
        (ctx.prepared_checkpoint_path).mkdir(parents=True, exist_ok=True)
        sleeps = _patch_sleep_calls(monkeypatch)
        removed = _patch_rmtree_calls(monkeypatch)

        prep = PreparationStage(ctx)
        assert prep.skip() is False
        prep_skip = PreparationStage(
            _context(tmp_path / "prep_skip", skip_preparation=True)
        )
        assert prep_skip.skip() is True
        prep.before_run()
        assert ctx.prepared_checkpoint_path in removed

        worker_ctx = _context(tmp_path / "worker", is_master=False, is_worker=True)
        worker_ctx.prepared_checkpoint_path.mkdir(parents=True, exist_ok=True)
        prep_worker = PreparationStage(worker_ctx)
        prep_worker.before_run()
        assert 60 in sleeps

        multi = MultiNodeConvertationStage(ctx)
        monkeypatch.setattr(multi, "is_hf_checkpoint", lambda p: False)
        assert multi.skip() is False
        multi.before_run()
        assert any(p == ctx.prepared_checkpoint_path for p in removed)

        merge = MergeHFStage(_context(tmp_path, is_worker=True))
        assert merge.skip() is True
        single = SingleNodeConvertationStage(_context(tmp_path, is_worker=True))
        assert single.skip() is True

    def test_preparation_before_run_noop_when_prepared_dir_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        removed = []
        monkeypatch.setattr(
            "node_runner.src.convertation_manager.shutil.rmtree",
            lambda p: removed.append(Path(p)),
        )
        ctx = _context(tmp_path / "prep_missing", is_master=True)
        # ensure prepared_checkpoint_path does not exist -> line 300 false branch
        prep = PreparationStage(ctx)
        prep.before_run()
        assert removed == []

    def test_single_node_helpers(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir(parents=True, exist_ok=True)
        (ckpt / "0").mkdir()
        (ckpt / "1").mkdir()
        assert SingleNodeConvertationStage._get_path_to_0_rank(ckpt).name == "1"

        _patch_utils_convertation_path(monkeypatch, tmp_path)
        ctx = _context(tmp_path)
        stage = SingleNodeConvertationStage(ctx)
        assert "--input-config-path" in stage.cmd_to_run


class TestConvertationLifecycleScenarios:
    def test_manager_setup_propagates_flags_and_builds_pipeline(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "node_runner.src.convertation_manager.time.sleep", lambda _seconds: None
        )
        manager = ConvertationManager()
        checkpoint_path = tmp_path / "checkpoint"
        artifacts_path = tmp_path / "artifacts"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        artifacts_path.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(manager, "is_hf_checkpoint", lambda _path: False)

        convertation_cfg = ConvertationConfig(mappers=["map_gigafsdp_to_hf"])
        preparation_cfg = PreparationConfig(skip=True)
        manager.setup(
            checkpoint_path=checkpoint_path,
            artifacts_path=artifacts_path,
            convertation_config=convertation_cfg,
            preparation_config=preparation_cfg,
            quantization_config=QuantizationConfig(),
            nnodes=2,
            node_rank=1,
            convertation_log_file=artifacts_path / "convertation.log",
            orkestration_manager=type(
                "OM", (), {"wait_barrier": lambda self, stage: None}
            )(),
            is_worker=True,
            is_master=False,
        )

        assert manager.convertation_context is not None
        assert manager.convertation_context.mappers == ["map_gigafsdp_to_hf"]
        assert manager.convertation_context.skip_preparation is True
        assert manager.convertation_context.quantization_config is not None
        assert [type(stage) for stage in manager.pipeline] == [
            PreparationStage,
            SingleNodeConvertationStage,
            MultiNodeConvertationStage,
            MergeHFStage,
            QuantizeStage,
        ]

    def test_preparation_skip_branches(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)

        map_ctx = _context(tmp_path / "map", mappers=["map_gigafsdp_to_hf"])
        assert PreparationStage(map_ctx).skip() is True

        skip_ctx = _context(tmp_path / "skip", skip_preparation=True)
        assert PreparationStage(skip_ctx).skip() is True

        existing_ctx = _context(tmp_path / "existing")
        stage = PreparationStage(existing_ctx)
        monkeypatch.setattr(stage, "skip_if_converted_hf_exists", lambda: True)
        assert stage.skip() is True

    def test_multinode_skip_branches(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)

        map_ctx = _context(tmp_path / "multi_map", mappers=["map_gigafsdp_to_hf"])
        assert MultiNodeConvertationStage(map_ctx).skip() is True

        mtp_ctx = _context(
            tmp_path / "multi_mtp",
            mappers=["map_mixed_gigafsdp_to_mtp_safetensors"],
        )
        mtp_stage = MultiNodeConvertationStage(mtp_ctx)
        monkeypatch.setattr(mtp_stage, "skip_if_converted_hf_exists", lambda: False)
        assert mtp_stage.skip() is False

        existing_ctx = _context(tmp_path / "multi_existing")
        stage = MultiNodeConvertationStage(existing_ctx)
        monkeypatch.setattr(stage, "skip_if_converted_hf_exists", lambda: True)
        assert stage.skip() is True

    def test_merge_skip_branches(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        _patch_utils_convertation_path(monkeypatch, tmp_path)

        map_ctx = _context(tmp_path / "merge_map", mappers=["map_gigafsdp_to_hf"])
        assert MergeHFStage(map_ctx).skip() is True

        worker_ctx = _context(tmp_path / "merge_worker", is_worker=True)
        assert MergeHFStage(worker_ctx).skip() is True

        existing_ctx = _context(tmp_path / "merge_existing")
        stage = MergeHFStage(existing_ctx)
        monkeypatch.setattr(stage, "skip_if_converted_hf_exists", lambda: True)
        assert stage.skip() is True

    def test_single_skip_branches(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)

        map_ctx = _context(
            tmp_path / "single_map",
            mappers=["map_gigafsdp_to_hf_safetensor"],
        )
        assert SingleNodeConvertationStage(map_ctx).skip() is True

        worker_ctx = _context(tmp_path / "single_worker", is_worker=True)
        assert SingleNodeConvertationStage(worker_ctx).skip() is True

        existing_ctx = _context(tmp_path / "single_existing")
        stage = SingleNodeConvertationStage(existing_ctx)
        monkeypatch.setattr(stage, "skip_if_converted_hf_exists", lambda: True)
        assert stage.skip() is True

    @pytest.mark.parametrize(
        ("prepared_exists", "expected_sleep"),
        [
            (True, 20),
            (False, 5),
        ],
    )
    def test_multinode_before_run_worker_wait_policy(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        prepared_exists: bool,
        expected_sleep: int,
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        sleeps = _patch_sleep_calls(monkeypatch)
        ctx = _context(tmp_path / "multi_worker", is_master=False, is_worker=True)
        if prepared_exists:
            ctx.prepared_checkpoint_path.mkdir(parents=True, exist_ok=True)
        stage = MultiNodeConvertationStage(ctx)
        stage.before_run()
        assert expected_sleep in sleeps

    def test_single_before_run_worker_waits_for_master(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        sleeps = _patch_sleep_calls(monkeypatch)
        ctx = _context(tmp_path / "single_worker", is_master=False, is_worker=False)
        stage = SingleNodeConvertationStage(ctx)
        stage.before_run()
        assert sleeps == [10]

    def test_stage_noop_hooks_are_safe_and_do_not_mutate_config(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        ctx = _context(tmp_path / "noop")
        before = ctx.convertation_config.model_dump()

        PreparationStage(ctx).after_run()
        MultiNodeConvertationStage(ctx).after_run()
        merge = MergeHFStage(ctx)
        merge.before_run()
        merge.after_run()
        quantize = QuantizeStage(ctx)
        quantize.build_cmd_to_run()
        SingleNodeConvertationStage(ctx).after_run()

        assert ctx.convertation_config.model_dump() == before

    def test_merge_skip_is_false_for_master_when_no_other_skip_reasons(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        stage = MergeHFStage(
            _context(
                tmp_path / "merge_run",
                is_worker=False,
                mappers=["map_fsdp_to_hf"],
            )
        )
        monkeypatch.setattr(stage, "skip_if_converted_hf_exists", lambda: False)
        assert stage.skip() is False

    def test_quantize_skip_branches(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)

        disabled_ctx = _context(tmp_path / "quant_disabled")
        assert QuantizeStage(disabled_ctx).skip() is True

        skip_not_set_ctx = _context(
            tmp_path / "quant_skip_not_set",
            quantization_config=QuantizationConfig(),
        )
        assert QuantizeStage(skip_not_set_ctx).skip() is True

        skip_ctx = _context(
            tmp_path / "quant_skip",
            quantization_config=QuantizationConfig(skip=True),
        )
        assert QuantizeStage(skip_ctx).skip() is True

        worker_ctx = _context(
            tmp_path / "quant_worker",
            is_worker=True,
            convertation_config=ConvertationConfig(mappers=["map_fsdp_to_hf"]),
            quantization_config=QuantizationConfig(skip=False),
        )
        assert QuantizeStage(worker_ctx).skip() is True

        fsdp_ctx = _context(
            tmp_path / "quant_fsdp",
            mappers=["map_gigafsdp_to_fsdp_safetensor"],
            convertation_config=ConvertationConfig(
                mappers=["map_gigafsdp_to_fsdp_safetensor"]
            ),
            quantization_config=QuantizationConfig(skip=False),
        )
        assert QuantizeStage(fsdp_ctx).skip() is True

        master_ctx = _context(
            tmp_path / "quant_master",
            is_worker=False,
            is_master=True,
            mappers=["map_fsdp_to_hf"],
            convertation_config=ConvertationConfig(mappers=["map_fsdp_to_hf"]),
            quantization_config=QuantizationConfig(skip=False),
        )
        assert QuantizeStage(master_ctx).skip() is False

        hf_to_local_ctx = _context(
            tmp_path / "quant_hf_to_local",
            mappers=["map_hf_safetensor_to_local"],
            convertation_config=ConvertationConfig(
                mappers=["map_hf_safetensor_to_local"]
            ),
            quantization_config=QuantizationConfig(skip=False),
        )
        assert QuantizeStage(hf_to_local_ctx).skip() is True

        local_to_hf_ctx = _context(
            tmp_path / "quant_local_to_hf",
            mappers=["map_local_safetensor_to_hf"],
            convertation_config=ConvertationConfig(
                mappers=["map_local_safetensor_to_hf"]
            ),
            quantization_config=QuantizationConfig(skip=False),
        )
        assert QuantizeStage(local_to_hf_ctx).skip() is True

    def test_quantize_helper_paths(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        disabled_stage = QuantizeStage(_context(tmp_path / "quant_no_section"))
        assert disabled_stage._get_quantization_section() == {}

        ctx = _context(
            tmp_path / "quant_paths",
            quantization_config=QuantizationConfig(),
        )
        stage = QuantizeStage(ctx)

        assert stage._get_quantization_config_path() == (
            tmp_path / "configs" / "quantization_config.yaml"
        )
        assert stage._get_quantization_script_path() == (
            tmp_path
            / "quantization"
            / "fine_grained_fp8_quantization_layer_by_layer.py"
        )
        assert (
            stage._get_quantized_checkpoint_path()
            == ctx.artifacts_path / "hf_checkpoint_fp8"
        )

    def test_quantize_build_cmd_uses_fp8_script(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        ctx = _context(
            tmp_path / "quant_cmd",
            mappers=["map_fsdp_to_hf"],
            convertation_config=ConvertationConfig(mappers=["map_fsdp_to_hf"]),
            quantization_config=QuantizationConfig(),
        )
        stage = QuantizeStage(ctx)

        assert stage.cmd_to_run[0] == "/home/user/convertation/bin/python"
        assert stage.cmd_to_run[1].endswith(
            "quantization/fine_grained_fp8_quantization_layer_by_layer.py"
        )
        assert stage.cwd.as_posix().endswith("quantization")
        assert stage.cwd.parent == tmp_path

    def test_quantize_before_run_writes_quantization_yaml(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        config_path = tmp_path / "quantization_config.yaml"
        script_path = tmp_path / "custom_quantization.py"
        ctx = _context(
            tmp_path / "quant_config",
            mappers=["map_fsdp_to_hf"],
            convertation_config=ConvertationConfig(mappers=["map_fsdp_to_hf"]),
            quantization_config=QuantizationConfig(
                script_path=str(script_path),
                config_path=str(config_path),
                device_map="auto",
                quantization_config={
                    "weight_block_size": [128, 128],
                    "modules_to_not_convert": [],
                },
            ),
        )
        tmp_output = ctx.artifacts_path / "hf_checkpoint_fp8"
        tmp_output.mkdir(parents=True, exist_ok=True)
        removed = _patch_rmtree_calls(monkeypatch)
        stage = QuantizeStage(ctx)

        stage.before_run()

        assert tmp_output in removed
        assert config_path.exists()
        cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        assert cfg["input_checkpoint_path"] == str(ctx.hf_checkpoint_path)
        assert cfg["output_checkpoint_path"] == str(tmp_output)
        assert cfg["device_map"] == "auto"
        assert cfg["config_path"] == str(config_path)
        assert cfg["script_path"] == str(script_path)
        assert cfg["output_checkpoint_name"] == "hf_checkpoint_fp8"
        assert "skip" not in cfg
        assert cfg["quantization_config"]["weight_block_size"] == [128, 128]

    def test_quantize_before_run_writes_yaml_when_output_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        config_path = tmp_path / "missing_output" / "quantization_config.yaml"
        ctx = _context(
            tmp_path / "quant_config_no_output",
            mappers=["map_fsdp_to_hf"],
            convertation_config=ConvertationConfig(mappers=["map_fsdp_to_hf"]),
            quantization_config=QuantizationConfig(
                config_path=str(config_path),
            ),
        )
        stage = QuantizeStage(ctx)

        stage.before_run()

        assert config_path.exists()
        cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        assert cfg["input_checkpoint_path"] == str(ctx.hf_checkpoint_path)
        assert cfg["output_checkpoint_path"] == str(
            ctx.artifacts_path / "hf_checkpoint_fp8"
        )

    def test_quantize_runs_for_original_hf_bf16_checkpoint(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        ctx = _context(
            tmp_path / "quant_original_hf",
            originaly_hf=True,
            mappers=["map_fsdp_to_hf"],
            convertation_config=ConvertationConfig(mappers=["map_fsdp_to_hf"]),
            quantization_config=QuantizationConfig(
                skip=False,
                config_path="quantization_config.yaml",
            ),
        )
        ctx.hf_checkpoint_path = ctx.checkpoint_path
        ctx.tokenizer_path = ctx.checkpoint_path
        (ctx.checkpoint_path / "config.json").write_text(
            '{"model_type":"llama","dtype":"bfloat16"}'
        )
        (ctx.checkpoint_path / "model.safetensors").write_text("bf16")
        assert QuantizeStage(ctx).skip() is False

    def test_quantize_does_not_assume_original_hf_without_dtype_is_bf16(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        ctx = _context(
            tmp_path / "quant_original_hf_unknown_dtype",
            originaly_hf=True,
            mappers=["map_fsdp_to_hf"],
            convertation_config=ConvertationConfig(mappers=["map_fsdp_to_hf"]),
            quantization_config=QuantizationConfig(skip=False),
        )
        ctx.hf_checkpoint_path = ctx.checkpoint_path
        ctx.tokenizer_path = ctx.checkpoint_path
        (ctx.checkpoint_path / "config.json").write_text('{"model_type":"llama"}')
        (ctx.checkpoint_path / "model.safetensors").write_text("weights")

        assert QuantizeStage(ctx).skip() is False

    def test_quantize_skips_original_fp8_checkpoint(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        ctx = _context(
            tmp_path / "quant_original_fp8",
            originaly_hf=True,
            mappers=["map_fsdp_to_hf"],
            convertation_config=ConvertationConfig(mappers=["map_fsdp_to_hf"]),
            quantization_config=QuantizationConfig(skip=False),
        )
        ctx.hf_checkpoint_path = ctx.checkpoint_path
        ctx.tokenizer_path = ctx.checkpoint_path
        _create_fp8_hf_checkpoint(ctx.checkpoint_path)
        calls = {"start": 0}

        class _SP:
            def __init__(self, cmd, log_file, cwd):
                return None

            def start(self):
                calls["start"] += 1

            def wait_for_completion(self):
                return None

            def raise_for_exit_code(self):
                return None

        monkeypatch.setattr(
            "node_runner.src.convertation_manager.SubprocessManager", _SP
        )

        assert QuantizeStage(ctx).skip() is True
        QuantizeStage(ctx).run()
        assert calls["start"] == 0
        assert ctx.hf_checkpoint_path == ctx.checkpoint_path

    def test_quantize_does_not_run_for_original_fsdp_checkpoint(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        ctx = _context(
            tmp_path / "quant_original_fsdp",
            originaly_hf=False,
            originally_fsdp=True,
            mappers=["map_fsdp_to_hf"],
            convertation_config=ConvertationConfig(mappers=["map_fsdp_to_hf"]),
            quantization_config=QuantizationConfig(),
        )
        calls = {"start": 0}

        class _SP:
            def __init__(self, cmd, log_file, cwd):
                return None

            def start(self):
                calls["start"] += 1

            def wait_for_completion(self):
                return None

            def raise_for_exit_code(self):
                return None

        monkeypatch.setattr(
            "node_runner.src.convertation_manager.SubprocessManager", _SP
        )
        assert QuantizeStage(ctx).skip() is True
        QuantizeStage(ctx).run()
        assert calls["start"] == 0

    def test_quantize_after_run_switches_context_to_fp8_checkpoint(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        ctx = _context(
            tmp_path / "quant_replace",
            mappers=["map_fsdp_to_hf"],
            convertation_config=ConvertationConfig(mappers=["map_fsdp_to_hf"]),
            quantization_config=QuantizationConfig(),
        )
        ctx.hf_checkpoint_path.mkdir(parents=True, exist_ok=True)
        (ctx.hf_checkpoint_path / "old.txt").write_text("bf16")
        source_path = ctx.artifacts_path / "hf_checkpoint_fp8"
        source_path.mkdir(parents=True, exist_ok=True)
        (source_path / "new.txt").write_text("fp8")

        QuantizeStage(ctx).after_run()

        assert source_path.exists()
        assert (ctx.artifacts_path / "hf_checkpoint" / "old.txt").exists()
        assert (ctx.hf_checkpoint_path / "new.txt").exists()
        assert ctx.hf_checkpoint_path == source_path
        assert ctx.tokenizer_path == source_path

    def test_quantize_after_run_without_existing_target(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        ctx = _context(
            tmp_path / "quant_no_target",
            mappers=["map_fsdp_to_hf"],
            convertation_config=ConvertationConfig(mappers=["map_fsdp_to_hf"]),
            quantization_config=QuantizationConfig(),
        )
        source_path = ctx.artifacts_path / "hf_checkpoint_fp8"
        source_path.mkdir(parents=True, exist_ok=True)
        (source_path / "new.txt").write_text("fp8")

        QuantizeStage(ctx).after_run()

        assert ctx.hf_checkpoint_path == source_path
        assert (ctx.hf_checkpoint_path / "new.txt").exists()

    def test_quantize_after_run_copies_jd_data_to_fp8_checkpoint(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        ctx = _context(
            tmp_path / "quant_jd_data",
            quantization_config=QuantizationConfig(),
        )
        (ctx.artifacts_path / "jd_data.json").write_text(
            json.dumps(
                {
                    "checkpoint_info": {"id": "checkpoint-id"},
                    "history": [{"action": "convertation"}],
                }
            )
        )
        fp8_path = ctx.artifacts_path / "hf_checkpoint_fp8"
        fp8_path.mkdir(parents=True)

        QuantizeStage(ctx).after_run()

        jd_data = json.loads((fp8_path / "artifacts" / "jd_data.json").read_text())
        assert jd_data["checkpoint_info"]["nfs_path"] == str(fp8_path)
        assert jd_data["checkpoint_info"]["parent_id"] == "checkpoint-id"

    def test_quantize_after_run_raises_when_output_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        ctx = _context(
            tmp_path / "quant_missing_target",
            mappers=["map_fsdp_to_hf"],
            convertation_config=ConvertationConfig(mappers=["map_fsdp_to_hf"]),
            quantization_config=QuantizationConfig(),
        )

        with pytest.raises(
            FileNotFoundError, match="Quantized checkpoint was not created"
        ):
            QuantizeStage(ctx).after_run()

    def test_quantize_worker_barrier_switches_to_existing_fp8_checkpoint(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        barriers = []
        ctx = _context(
            tmp_path / "quant_worker_barrier",
            is_worker=True,
            is_master=False,
            mappers=["map_fsdp_to_hf"],
            convertation_config=ConvertationConfig(mappers=["map_fsdp_to_hf"]),
            quantization_config=QuantizationConfig(skip=False),
            orkestration_manager=type(
                "OM", (), {"wait_barrier": lambda self, stage: barriers.append(stage)}
            )(),
        )
        source_path = ctx.artifacts_path / "hf_checkpoint_fp8"
        stage = QuantizeStage(ctx)
        monkeypatch.setattr(stage, "is_hf_checkpoint", lambda path: path == source_path)

        stage.barrier_after_run()

        assert barriers == ["QUANTIZE_FP8_DONE"]
        assert ctx.hf_checkpoint_path == source_path
        assert ctx.tokenizer_path == source_path

    def test_quantize_barrier_noop_when_quantization_not_requested(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        barriers = []
        ctx = _context(
            tmp_path / "quant_barrier_noop",
            is_worker=True,
            is_master=False,
            quantization_config=None,
            orkestration_manager=type(
                "OM", (), {"wait_barrier": lambda self, stage: barriers.append(stage)}
            )(),
        )
        original_hf_path = ctx.hf_checkpoint_path

        QuantizeStage(ctx).barrier_after_run()

        assert barriers == ["QUANTIZE_FP8_DONE"]
        assert ctx.hf_checkpoint_path == original_hf_path

    def test_quantize_skip_uses_existing_fp8_checkpoint(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        ctx = _context(
            tmp_path / "quant_existing_fp8",
            mappers=["map_fsdp_to_hf"],
            convertation_config=ConvertationConfig(mappers=["map_fsdp_to_hf"]),
            quantization_config=QuantizationConfig(skip=False),
        )
        source_path = ctx.artifacts_path / "hf_checkpoint_fp8"
        _create_fp8_hf_checkpoint(source_path)

        stage = QuantizeStage(ctx)

        assert stage.skip() is True
        assert ctx.hf_checkpoint_path == source_path
        assert ctx.tokenizer_path == source_path

    def test_pre_quant_padding_command_uses_default_script_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        default_script = tmp_path / "scripts" / "pad_mlp_shards.py"
        default_script.parent.mkdir(parents=True, exist_ok=True)
        default_script.write_text("print('ok')\n")
        ctx = _context(
            tmp_path / "quant_custom_pad_cmd",
            mappers=["map_fsdp_to_hf"],
            convertation_config=ConvertationConfig(mappers=["map_fsdp_to_hf"]),
            quantization_config=QuantizationConfig(
                skip=False,
                pre_quant_padding={
                    "enabled": True,
                    "only_if_mlp_intermediate_sizes": [10944],
                },
            ),
        )
        ctx.hf_checkpoint_path.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                "model.layers.0.mlp.gate_proj.weight": torch.zeros(
                    (10944, 7168), dtype=torch.bfloat16
                )
            },
            str(ctx.hf_checkpoint_path / "model-00001-of-00001.safetensors"),
        )
        stage = QuantizeStage(ctx)

        calls = {"cmd": None, "cwd": None}

        class _PadSP:
            def __init__(self, cmd, log_file, cwd):
                calls["cmd"] = cmd
                calls["cwd"] = cwd

            def start(self):
                return None

            def wait_for_completion(self):
                return None

            def raise_for_exit_code(self):
                return None

        monkeypatch.setattr(
            "node_runner.src.convertation_manager.SubprocessManager", _PadSP
        )
        stage._run_pre_quantization_padding_if_needed()

        assert calls["cmd"] is not None
        assert calls["cmd"][0] == "/home/user/convertation/bin/python"
        assert calls["cmd"][1] == str(default_script)
        assert "--checkpoint-dir" in calls["cmd"]
        assert calls["cwd"] == default_script.parent

    def test_pre_quant_padding_skips_size_outside_allow_list(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        ctx = _context(
            tmp_path / "quant_pad_size_not_allowed",
            mappers=["map_fsdp_to_hf"],
            convertation_config=ConvertationConfig(mappers=["map_fsdp_to_hf"]),
            quantization_config=QuantizationConfig(
                skip=False,
                pre_quant_padding={
                    "enabled": True,
                    "only_if_mlp_intermediate_sizes": [10944],
                },
            ),
        )
        ctx.hf_checkpoint_path.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                "model.layers.0.mlp.gate_proj.weight": torch.zeros(
                    (18432, 7168), dtype=torch.bfloat16
                )
            },
            str(ctx.hf_checkpoint_path / "model-00001-of-00001.safetensors"),
        )
        stage = QuantizeStage(ctx)

        calls = {"start": 0}

        class _PadSP:
            def __init__(self, cmd, log_file, cwd):
                return None

            def start(self):
                calls["start"] += 1

            def wait_for_completion(self):
                return None

            def raise_for_exit_code(self):
                return None

        monkeypatch.setattr(
            "node_runner.src.convertation_manager.SubprocessManager", _PadSP
        )
        stage._run_pre_quantization_padding_if_needed()

        assert calls["start"] == 0

    def test_pre_quant_padding_extracts_down_projection_size(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        ctx = _context(
            tmp_path / "quant_pad_down_proj",
            quantization_config=QuantizationConfig(skip=False),
        )
        ctx.hf_checkpoint_path.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                "model.layers.0.self_attn.q_proj.weight": torch.zeros(
                    (7168, 7168), dtype=torch.bfloat16
                )
            },
            str(ctx.hf_checkpoint_path / "model-00000-of-00002.safetensors"),
        )
        save_file(
            {
                "model.layers.0.mlp.down_proj.weight": torch.zeros(
                    (7168, 10944), dtype=torch.bfloat16
                )
            },
            str(ctx.hf_checkpoint_path / "model-00001-of-00002.safetensors"),
        )

        stage = QuantizeStage(ctx)

        assert (
            stage._extract_mlp_intermediate_size_from_safetensors_header(
                ctx.hf_checkpoint_path
            )
            == 10944
        )

    def test_pre_quant_padding_ignores_unexpected_mlp_tensor_shapes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        ctx = _context(
            tmp_path / "quant_pad_unexpected_shapes",
            quantization_config=QuantizationConfig(skip=False),
        )
        ctx.hf_checkpoint_path.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                "model.layers.0.mlp.gate_proj.weight": torch.zeros(
                    (10944,), dtype=torch.bfloat16
                ),
                "model.layers.1.mlp.down_proj.weight": torch.zeros(
                    (10944,), dtype=torch.bfloat16
                ),
                "model.layers.2.self_attn.q_proj.weight": torch.zeros(
                    (7168, 7168), dtype=torch.bfloat16
                ),
            },
            str(ctx.hf_checkpoint_path / "model-00001-of-00001.safetensors"),
        )

        stage = QuantizeStage(ctx)

        assert (
            stage._extract_mlp_intermediate_size_from_safetensors_header(
                ctx.hf_checkpoint_path
            )
            is None
        )

    def test_pre_quant_padding_skip_conditions(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)

        disabled_ctx = _context(
            tmp_path / "quant_pad_disabled",
            quantization_config=QuantizationConfig(
                skip=False,
                pre_quant_padding={"enabled": False},
            ),
        )
        assert (
            QuantizeStage(disabled_ctx)._should_run_pre_quantization_padding(
                10945,
                disabled_ctx.quantization_config,
            )
            is False
        )

        no_quant_stage = QuantizeStage(_context(tmp_path / "quant_pad_no_config"))
        no_quant_stage._run_pre_quantization_padding_if_needed()

        monkeypatch.setattr(
            QuantizeStage,
            "_extract_mlp_intermediate_size_from_safetensors_header",
            lambda self, path: pytest.fail(
                "header scan must be skipped when padding is disabled"
            ),
        )
        QuantizeStage(disabled_ctx)._run_pre_quantization_padding_if_needed()

        configured_ctx = _context(
            tmp_path / "quant_pad_configured_multiple",
            quantization_config=QuantizationConfig(
                skip=False,
                pre_quant_padding={"multiple_of": 256},
            ),
        )
        configured_stage = QuantizeStage(configured_ctx)
        assert (
            configured_stage._get_pre_quantization_padding_multiple(
                configured_ctx.quantization_config,
            )
            == 256
        )
        assert (
            configured_stage._should_run_pre_quantization_padding(
                1024,
                configured_ctx.quantization_config,
            )
            is False
        )

    def test_pre_quant_padding_can_run_on_copied_checkpoint_when_not_inplace(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        default_script = tmp_path / "scripts" / "pad_mlp_shards.py"
        default_script.parent.mkdir(parents=True, exist_ok=True)
        default_script.write_text("print('ok')\n")
        ctx = _context(
            tmp_path / "quant_pad_copy_mode",
            mappers=["map_fsdp_to_hf"],
            convertation_config=ConvertationConfig(mappers=["map_fsdp_to_hf"]),
            quantization_config=QuantizationConfig(
                skip=False,
                output_checkpoint_name="hf_checkpoint_fp8",
                pre_quant_padding={
                    "inplace": False,
                    "enabled": True,
                    "only_if_mlp_intermediate_sizes": [10944],
                    "update_config": False,
                },
            ),
        )
        ctx.hf_checkpoint_path.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                "model.layers.0.mlp.gate_proj.weight": torch.zeros(
                    (10944, 7168), dtype=torch.bfloat16
                )
            },
            str(ctx.hf_checkpoint_path / "model-00001-of-00001.safetensors"),
        )
        (ctx.hf_checkpoint_path / "config.json").write_text(
            '{"intermediate_size": 10944}'
        )
        stage = QuantizeStage(ctx)

        calls = {"cmd": None}

        class _PadSP:
            def __init__(self, cmd, log_file, cwd):
                calls["cmd"] = cmd

            def start(self):
                return None

            def wait_for_completion(self):
                return None

            def raise_for_exit_code(self):
                return None

        monkeypatch.setattr(
            "node_runner.src.convertation_manager.SubprocessManager", _PadSP
        )
        original_hf = ctx.hf_checkpoint_path
        stale_copied_dir = ctx.artifacts_path / "hf_checkpoint_fp8_prepad"
        stale_copied_dir.mkdir(parents=True, exist_ok=True)
        (stale_copied_dir / "stale.txt").write_text("old")
        stage._run_pre_quantization_padding_if_needed()

        assert calls["cmd"] is not None
        copied_dir = ctx.artifacts_path / "hf_checkpoint_fp8_prepad"
        assert calls["cmd"][calls["cmd"].index("--checkpoint-dir") + 1] == str(
            copied_dir
        )
        assert "--update-config" not in calls["cmd"]
        assert original_hf.exists()
        assert copied_dir.exists()
        assert not (copied_dir / "stale.txt").exists()
        assert ctx.hf_checkpoint_path == copied_dir

    def test_pre_quant_padding_never_runs_inplace_on_original_hf_checkpoint(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        default_script = tmp_path / "scripts" / "pad_mlp_shards.py"
        default_script.parent.mkdir(parents=True, exist_ok=True)
        default_script.write_text("print('ok')\n")
        ctx = _context(
            tmp_path / "quant_pad_original_hf_guard",
            originaly_hf=True,
            mappers=["map_fsdp_to_hf"],
            convertation_config=ConvertationConfig(mappers=["map_fsdp_to_hf"]),
            quantization_config=QuantizationConfig(
                skip=False,
                output_checkpoint_name="hf_checkpoint_fp8",
                pre_quant_padding={
                    "inplace": True,
                    "enabled": True,
                    "only_if_mlp_intermediate_sizes": [10944],
                },
            ),
        )
        ctx.hf_checkpoint_path = ctx.checkpoint_path
        save_file(
            {
                "model.layers.0.mlp.gate_proj.weight": torch.zeros(
                    (10944, 7168), dtype=torch.bfloat16
                )
            },
            str(ctx.hf_checkpoint_path / "model-00001-of-00001.safetensors"),
        )
        stage = QuantizeStage(ctx)

        calls = {"cmd": None}

        class _PadSP:
            def __init__(self, cmd, log_file, cwd):
                calls["cmd"] = cmd

            def start(self):
                return None

            def wait_for_completion(self):
                return None

            def raise_for_exit_code(self):
                return None

        monkeypatch.setattr(
            "node_runner.src.convertation_manager.SubprocessManager", _PadSP
        )
        stage._run_pre_quantization_padding_if_needed()

        copied_dir = ctx.artifacts_path / "hf_checkpoint_fp8_prepad"
        assert calls["cmd"] is not None
        assert calls["cmd"][calls["cmd"].index("--checkpoint-dir") + 1] == str(
            copied_dir
        )
        assert copied_dir.exists()
        assert ctx.hf_checkpoint_path == copied_dir

    def test_quantize_after_run_removes_prepad_copy(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        ctx = _context(
            tmp_path / "quant_prepad_cleanup",
            mappers=["map_fsdp_to_hf"],
            convertation_config=ConvertationConfig(mappers=["map_fsdp_to_hf"]),
            quantization_config=QuantizationConfig(),
        )
        quantized_path = ctx.artifacts_path / "hf_checkpoint_fp8"
        quantized_path.mkdir(parents=True, exist_ok=True)
        prepad_path = ctx.artifacts_path / "hf_checkpoint_fp8_prepad"
        prepad_path.mkdir(parents=True, exist_ok=True)
        (prepad_path / "model.safetensors").write_text("padded")

        QuantizeStage(ctx).after_run()

        assert ctx.hf_checkpoint_path == quantized_path
        assert not prepad_path.exists()

    def test_output_dir_name(self):
        assert (
            ConvertationManager._output_dir_name(["map_gigafsdp_to_hf"])
            == "hf_checkpoint"
        )
        assert (
            ConvertationManager._output_dir_name(["map_gigafsdp_to_fsdp"])
            == "fsdp_checkpoint"
        )
        assert (
            ConvertationManager._output_dir_name(["map_gigafsdp_to_fsdp_safetensor"])
            == "fsdp_checkpoint"
        )
        assert (
            ConvertationManager._output_dir_name(
                ["map_mixed_gigafsdp_to_mtp_safetensors"]
            )
            == "fsdp_checkpoint"
        )

    def test_setup_paths_fsdp_source(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        manager = ConvertationManager()
        ckpt = tmp_path / "ckpt"
        artifacts = tmp_path / "artifacts"
        ckpt.mkdir(parents=True, exist_ok=True)
        artifacts.mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(manager, "is_hf_checkpoint", lambda p: False)
        monkeypatch.setattr(manager, "is_fsdp_checkpoint", lambda p: True)
        manager._setup_paths(ckpt, artifacts, ["map_gigafsdp_to_hf"])
        assert manager.originaly_hf is False
        assert manager.originally_fsdp is True
        assert manager.hf_checkpoint_path == ckpt

    def test_setup_paths_fsdp_output(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        manager = ConvertationManager()
        ckpt = tmp_path / "ckpt"
        artifacts = tmp_path / "artifacts"
        ckpt.mkdir(parents=True, exist_ok=True)
        artifacts.mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(manager, "is_hf_checkpoint", lambda p: False)
        monkeypatch.setattr(manager, "is_fsdp_checkpoint", lambda p: False)
        manager._setup_paths(ckpt, artifacts, ["map_gigafsdp_to_fsdp"])
        assert manager.originaly_hf is False
        assert manager.hf_checkpoint_path == artifacts / "fsdp_checkpoint"

    def test_single_node_skip_fsdp_safetensor(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        ctx = _context(tmp_path, mappers=["map_gigafsdp_to_fsdp_safetensor"])
        stage = SingleNodeConvertationStage(ctx)
        assert stage.skip() is True

    def test_single_node_skip_mixed_gigafsdp_mtp_safetensors(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        ctx = _context(
            tmp_path,
            mappers=["map_mixed_gigafsdp_to_mtp_safetensors"],
        )
        stage = SingleNodeConvertationStage(ctx)
        assert stage.skip() is True

    def test_merge_skip_fsdp_output(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        ctx = _context(
            tmp_path,
            mappers=["map_gigafsdp_to_fsdp_safetensor"],
            is_worker=False,
        )
        stage = MergeHFStage(ctx)
        assert stage.skip() is True

        mtp_ctx = _context(
            tmp_path / "mtp_merge",
            mappers=["map_mixed_gigafsdp_to_mtp_safetensors"],
            is_worker=False,
        )
        assert MergeHFStage(mtp_ctx).skip() is True

    def test_gfsdp2_dcp_mapper_stage_routing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        ctx = _context(tmp_path, mappers=["map_gfsdp2_dcp_to_hf"])

        assert PreparationStage(ctx).skip() is True
        assert SingleNodeConvertationStage(ctx).skip() is True

        merge = MergeHFStage(ctx)
        assert merge.cmd_to_run == [
            "/home/user/convertation/bin/python",
            "map-fsdp-to-fsdp.py",
            "--input-config-path",
            str(ctx.convertation_config_path),
            "--finalize-only",
            "--world-size",
            "2",
        ]
        assert merge.cwd == tmp_path

    def test_setup_paths_safetensors_graft(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        # graft branch: HF in, HF out — output redirected to artifacts, tokenizer
        # and prepared_checkpoint stay at the input base; output_name override honoured.
        manager = ConvertationManager()
        ckpt = tmp_path / "ckpt"
        artifacts = tmp_path / "artifacts"
        ckpt.mkdir(parents=True, exist_ok=True)
        artifacts.mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(manager, "is_hf_checkpoint", lambda p: True)
        monkeypatch.setattr(manager, "is_fsdp_checkpoint", lambda p: False)

        # The graft always writes to a distinct dir so it never clobbers the base
        # checkpoint (conventionally artifacts/hf_checkpoint).
        manager._setup_paths(
            ckpt,
            artifacts,
            ["map_safetensors_graft_mtp_safetensors"],
        )
        assert manager.originaly_hf is True
        assert manager.hf_checkpoint_path == artifacts / "hf_checkpoint_mtp"
        # tokenizer + prepared point at the base, so MultiNode.before_run does NOT
        # rm-rf the input (graft skips that branch via is_map_safetensors_graft).
        assert manager.tokenizer_path == ckpt
        assert manager.prepared_checkpoint_path == ckpt

    def test_skip_if_originaly_hf_graft_carve_out(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        # Graft must NOT be blanket-skipped even when input is an HF checkpoint —
        # output is redirected, so the stage needs to actually run.
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        ctx = _context(
            tmp_path / "graft_hf_in",
            mappers=["map_safetensors_graft_mtp_safetensors"],
            originaly_hf=True,
        )
        stage = MultiNodeConvertationStage(ctx)
        assert stage.skip_if_originaly_hf() is False

        # Sanity: non-graft HF-input case still skips.
        non_graft_ctx = _context(
            tmp_path / "non_graft_hf_in",
            mappers=["map_gigafsdp_to_hf"],
            originaly_hf=True,
        )
        assert MultiNodeConvertationStage(non_graft_ctx).skip_if_originaly_hf() is True

    def test_single_node_skip_safetensors_graft(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        ctx = _context(tmp_path, mappers=["map_safetensors_graft_mtp_safetensors"])
        stage = SingleNodeConvertationStage(ctx)
        assert stage.skip() is True

    def test_skip_if_originaly_fsdp(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        ctx = _context(tmp_path, originaly_hf=False, originally_fsdp=True)
        stage = MultiNodeConvertationStage(ctx)
        assert stage.skip_if_originaly_hf() is True

    def test_skip_if_converted_hf_exists_fsdp(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        ctx = _context(tmp_path)
        stage = MultiNodeConvertationStage(ctx)
        monkeypatch.setattr(stage, "is_hf_checkpoint", lambda p: False)
        monkeypatch.setattr(stage, "is_fsdp_checkpoint", lambda p: True)
        assert stage.skip_if_converted_hf_exists() is True

    def test_single_node_before_run_master_updates_and_recreates_hf(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        stage = SingleNodeConvertationStage(
            _context(
                tmp_path / "single_run",
                is_worker=False,
                is_master=True,
                mappers=["map_fsdp_to_hf"],
            )
        )
        monkeypatch.setattr(stage, "skip_if_converted_hf_exists", lambda: False)
        assert stage.skip() is False

    def test_single_before_run_master_writes_config_and_recreates_output_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        ctx = _context(tmp_path / "single_master", is_master=True, is_worker=False)
        (ctx.checkpoint_path / "0").mkdir(parents=True, exist_ok=True)
        (ctx.checkpoint_path / "1").mkdir(parents=True, exist_ok=True)
        ctx.hf_checkpoint_path.mkdir(parents=True, exist_ok=True)

        stage = SingleNodeConvertationStage(ctx)
        updates = {}
        removed = []
        monkeypatch.setattr(
            stage,
            "update_and_save_convertation_config",
            lambda payload: updates.update(payload),
        )
        monkeypatch.setattr(
            "node_runner.src.convertation_manager.shutil.rmtree",
            lambda p: removed.append(Path(p)),
        )

        stage.before_run()

        assert updates["input_checkpoint_path"].endswith("/1")
        assert updates["tokenizer_path"].endswith("/tokenizer")
        assert updates["output_checkpoint_path"].endswith("/hf_checkpoint")
        assert ctx.hf_checkpoint_path in removed
        assert ctx.hf_checkpoint_path.exists()

    def test_single_before_run_mixed_gigafsdp_uses_parent_checkpoint_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        ctx = _context(
            tmp_path / "mixed_single_master",
            is_master=True,
            is_worker=False,
            mappers=["map_mixed_gigafsdp_to_fsdp", "map_fsdp_to_hf"],
            convertation_config=ConvertationConfig(
                mappers=["map_mixed_gigafsdp_to_fsdp", "map_fsdp_to_hf"]
            ),
        )
        (ctx.checkpoint_path / "artifacts").mkdir(parents=True, exist_ok=True)
        (ctx.checkpoint_path / "ep0-ba14000-tp_rank0-ep_rank0").mkdir(
            parents=True, exist_ok=True
        )
        ctx.hf_checkpoint_path.mkdir(parents=True, exist_ok=True)

        stage = SingleNodeConvertationStage(ctx)
        updates = {}
        removed = []
        monkeypatch.setattr(
            stage,
            "update_and_save_convertation_config",
            lambda payload: updates.update(payload),
        )
        monkeypatch.setattr(
            "node_runner.src.convertation_manager.shutil.rmtree",
            lambda p: removed.append(Path(p)),
        )

        stage.before_run()

        assert updates["input_checkpoint_path"] == str(ctx.checkpoint_path)
        assert updates["tokenizer_path"].endswith("/tokenizer")
        assert updates["output_checkpoint_path"].endswith("/hf_checkpoint")
        assert ctx.hf_checkpoint_path in removed
        assert ctx.hf_checkpoint_path.exists()

    def test_preparation_before_run_handles_exists_false_when_path_reports_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        sleeps = _patch_sleep_calls(monkeypatch)
        removed = _patch_rmtree_calls(monkeypatch)

        class _FakePreparedPath:
            def is_dir(self):
                return True

            def exists(self):
                return False

        ctx = _context(tmp_path / "prep_fake_exists_false", is_master=True)
        ctx.prepared_checkpoint_path = _FakePreparedPath()
        PreparationStage(ctx).before_run()
        assert sleeps == []
        assert removed == []

    def test_multinode_before_run_master_creates_hf_dir_when_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        removed = _patch_rmtree_calls(monkeypatch)
        ctx = _context(tmp_path / "multi_missing_hf", is_master=True, is_worker=False)
        stage = MultiNodeConvertationStage(ctx)
        stage.before_run()
        assert ctx.hf_checkpoint_path.exists()
        assert not any(path == ctx.hf_checkpoint_path for path in removed)

    def test_single_before_run_master_creates_hf_dir_when_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        ctx = _context(tmp_path / "single_missing", is_master=True, is_worker=False)
        (ctx.checkpoint_path / "0").mkdir(parents=True, exist_ok=True)
        (ctx.checkpoint_path / "1").mkdir(parents=True, exist_ok=True)
        stage = SingleNodeConvertationStage(ctx)
        monkeypatch.setattr(
            stage, "update_and_save_convertation_config", lambda _d: None
        )
        assert not ctx.hf_checkpoint_path.exists()
        stage.before_run()
        assert ctx.hf_checkpoint_path.exists()


class TestMapHfSafetensorToLocal:
    """Branches specific to the map_hf_safetensor_to_local mapper."""

    def test_setup_sets_local_checkpoint_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        manager = ConvertationManager()
        ckpt = tmp_path / "checkpoint"
        artifacts = tmp_path / "artifacts"
        ckpt.mkdir(parents=True, exist_ok=True)
        artifacts.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(manager, "is_hf_checkpoint", lambda _p: True)
        monkeypatch.setattr(manager, "is_fsdp_checkpoint", lambda _p: False)
        cfg = ConvertationConfig(mappers=["map_hf_safetensor_to_local"])
        manager.setup(
            checkpoint_path=ckpt,
            artifacts_path=artifacts,
            convertation_config=cfg,
            preparation_config=PreparationConfig(skip=True),
            quantization_config=None,
            nnodes=1,
            node_rank=0,
            convertation_log_file=artifacts / "convertation.log",
            orkestration_manager=type(
                "OM", (), {"wait_barrier": lambda self, stage: None}
            )(),
            is_worker=False,
            is_master=True,
        )
        expected = artifacts / "local_checkpoint"
        assert manager.local_checkpoint_path == expected
        assert manager.convertation_context.local_checkpoint_path == expected

    def test_skip_if_originaly_hf_is_false_for_hf_to_local_mapper(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        ctx = _context(
            tmp_path,
            mappers=["map_hf_safetensor_to_local"],
            originaly_hf=True,
            local_checkpoint_path=tmp_path / "local_checkpoint",
        )
        assert PreparationStage(ctx).skip_if_originaly_hf() is False

    def test_existing_converted_hf_does_not_skip_the_hf_to_local_mapper(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        ctx = _context(tmp_path, mappers=["map_hf_safetensor_to_local"])

        assert PreparationStage(ctx).skip_if_converted_hf_exists() is False

    def test_preparation_skips_for_hf_to_local_mapper(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        ctx = _context(tmp_path, mappers=["map_hf_safetensor_to_local"])

        assert PreparationStage(ctx).skip() is True

    def test_single_skip_for_hf_to_local_mapper(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        ctx = _context(tmp_path, mappers=["map_hf_safetensor_to_local"])
        assert SingleNodeConvertationStage(ctx).skip() is True

    def test_single_skip_for_local_to_hf_mapper(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        ctx = _context(tmp_path, mappers=["map_local_safetensor_to_hf"])
        assert SingleNodeConvertationStage(ctx).skip() is True

    def test_multinode_before_run_master_uses_local_checkpoint_as_output(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        local_ckpt = tmp_path / "artifacts" / "local_checkpoint"
        ctx = _context(
            tmp_path,
            mappers=["map_hf_safetensor_to_local"],
            is_master=True,
            local_checkpoint_path=local_ckpt,
        )
        ctx.prepared_checkpoint_path.mkdir(parents=True, exist_ok=True)
        updates = {}
        monkeypatch.setattr(
            "node_runner.src.convertation_manager.shutil.rmtree",
            lambda _p, **kw: None,
        )
        stage = MultiNodeConvertationStage(ctx)
        monkeypatch.setattr(
            stage, "update_and_save_convertation_config", lambda d: updates.update(d)
        )
        stage.before_run()
        assert updates["output_checkpoint_path"] == str(local_ckpt)
        assert local_ckpt.is_dir()

    def test_merge_build_cmd_uses_local_checkpoint_base_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        _patch_utils_convertation_path(monkeypatch, tmp_path)
        local_ckpt = tmp_path / "artifacts" / "local_checkpoint"
        ctx = _context(
            tmp_path,
            mappers=["map_hf_safetensor_to_local"],
            local_checkpoint_path=local_ckpt,
        )
        stage = MergeHFStage(ctx)
        idx = stage.cmd_to_run.index("--base-dir")
        assert stage.cmd_to_run[idx + 1] == str(local_ckpt)
