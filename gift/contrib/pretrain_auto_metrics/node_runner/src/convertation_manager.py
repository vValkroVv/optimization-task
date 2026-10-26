from __future__ import annotations

import json
import os
import shutil
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf
from safetensors import safe_open
from transformers import AutoTokenizer

from .config_model import (
    ConvertationConfig,
    PreparationConfig,
    QuantizationConfig,
)
from .job_dispatcher import JD
from .logger import Logger
from .orkestration_manager import FastApiOrkestrationManager
from .subprocess_manager import SubprocessManager


@dataclass
class ConvertationContext:
    convertation_config_path: Path
    hf_checkpoint_path: Path
    checkpoint_path: Path
    artifacts_path: Path
    prepared_checkpoint_path: Path
    tokenizer_path: Path
    log_file: Path

    originaly_hf: bool
    originally_fsdp: bool
    mappers: list[str]
    global_rank: int
    world_size: int
    convertation_config: ConvertationConfig
    preparation_config: PreparationConfig
    orkestration_manager: FastApiOrkestrationManager
    is_worker: bool
    is_master: bool
    skip_preparation: bool
    quantization_config: Optional[QuantizationConfig] = None
    local_checkpoint_path: Optional[Path] = None


class ConvertationUtils:
    @staticmethod
    def _read_hf_config(dir_path: Path) -> Optional[dict]:
        config_file = dir_path / "config.json"
        if not config_file.is_file():
            return None
        try:
            return json.loads(config_file.read_text())
        except Exception:
            return None

    @staticmethod
    def is_hf_checkpoint(dir_path: Path) -> bool:
        if not dir_path.is_dir():
            Logger.error(f"[HF checker]: {dir_path} is not a directory")
            return False
        Logger.info(f"[HF checker]: {dir_path} is a directory")

        config_file = dir_path / "config.json"
        if not config_file.is_file():
            Logger.error(f"[HF checker]: {config_file} is not a file")
            return False

        Logger.info(f"[HF checker]: {config_file} is a file")

        try:
            config = json.loads(config_file.read_text())
            Logger.info(f"[HF checker]: {config_file} is a valid JSON file")
        except Exception:
            Logger.error(f"[HF checker]: {config_file} is not a valid JSON file")
            return False

        if "model_type" not in config:
            Logger.error(f"[HF checker]: {config_file} is not a valid HF checkpoint")
            return False

        weight_files = list(dir_path.glob("*.safetensors")) + list(
            dir_path.glob("pytorch_model*.bin")
        )

        if not weight_files:
            Logger.error(f"[HF checker]: {dir_path} does not contain any weight files")
            return False
        Logger.info(f"[HF checker]: {dir_path} contains weight files")
        try:
            loaded_tokenizer = AutoTokenizer.from_pretrained(
                dir_path, trust_remote_code=True
            )
            Logger.info("[HF checker]: tokenizer loaded successfully")
        except Exception as e:
            Logger.warning(
                f"[HF checker]: tokenizer loading failed: {e}, but checkpoint structure is valid"
            )
        else:
            if not loaded_tokenizer:
                Logger.warning(
                    "[HF checker]: tokenizer is None, but checkpoint structure is valid"
                )
        Logger.info("[HF checker]: checkpoint is valid")
        return True

    @staticmethod
    def is_fsdp_checkpoint(dir_path: Path) -> bool:
        if not dir_path.is_dir():
            Logger.error(f"[FSDP checker]: {dir_path} is not a directory")
            return False
        Logger.info(f"[FSDP checker]: {dir_path} is a directory")
        if not (dir_path / "composer_state.pt").is_file():
            Logger.error(f"[FSDP checker]: {dir_path} missing composer_state.pt")
            return False
        Logger.info(f"[FSDP checker]: {dir_path} contains composer_state.pt")
        config_file = dir_path / "config.json"
        if not config_file.is_file():
            Logger.error(f"[FSDP checker]: {config_file} is not a file")
            return False
        try:
            config = json.loads(config_file.read_text())
            Logger.info(f"[FSDP checker]: {config_file} is a valid JSON file")
        except Exception:
            Logger.error(f"[FSDP checker]: {config_file} is not a valid JSON file")
            return False
        if "model_type" not in config:
            Logger.error(f"[FSDP checker]: {config_file} missing model_type field")
            return False
        weight_files = list(dir_path.glob("*.safetensors")) + list(
            dir_path.glob("pytorch_model*.bin")
        )
        if not weight_files:
            Logger.error(
                f"[FSDP checker]: {dir_path} does not contain any weight files"
            )
            return False
        Logger.info(f"[FSDP checker]: {dir_path} contains weight files")
        Logger.info("[FSDP checker]: checkpoint is valid")
        return True


class ConvertationManager(ConvertationUtils):
    def __init__(self):
        self._pipeline_stages: list[PipelineStage] = []
        self.originaly_hf: bool = False
        self.originally_fsdp: bool = False
        self.hf_checkpoint_path: Path = None
        self.local_checkpoint_path: Optional[Path] = None
        self.tokenizer_path: Path = None
        self.convertation_config_path: Path = None
        self.prepared_checkpoint_path: Path = None
        self.log_file: Path = None
        self.global_rank: int = None
        self.world_size: int = None
        self.convertation_context: Optional[ConvertationContext] = None

    @staticmethod
    def _output_dir_name(mappers: list[str]) -> str:
        if any(
            m.lower().startswith("map_gigafsdp_to_fsdp")
            or m.lower() == "map_mixed_gigafsdp_to_mtp_safetensors"
            for m in mappers
        ):
            return "fsdp_checkpoint"
        return "hf_checkpoint"

    @staticmethod
    def _is_safetensors_graft_mapper(mappers: list[str]) -> bool:
        return any(
            m.lower() == "map_safetensors_graft_mtp_safetensors" for m in mappers
        )

    def _setup_paths(
        self,
        checkpoint_path: Path,
        artifacts_path: Path,
        mappers: list[str],
    ) -> None:
        self.originaly_hf = self.is_hf_checkpoint(checkpoint_path)
        self.originally_fsdp = self.is_fsdp_checkpoint(checkpoint_path)
        is_graft = self._is_safetensors_graft_mapper(mappers)
        if is_graft:
            self.hf_checkpoint_path = Path(artifacts_path, "hf_checkpoint_mtp")
            self.tokenizer_path = checkpoint_path
            self.prepared_checkpoint_path = checkpoint_path
        elif self.originaly_hf or self.originally_fsdp:
            self.hf_checkpoint_path = checkpoint_path
            self.tokenizer_path = checkpoint_path
            self.prepared_checkpoint_path = checkpoint_path
        else:
            output_dir = self._output_dir_name(mappers)
            self.hf_checkpoint_path = Path(artifacts_path, output_dir)
            self.tokenizer_path = Path(artifacts_path, "tokenizer")
            self.prepared_checkpoint_path = Path(artifacts_path, "prepared_checkpoint")

        self.convertation_config_path = Path(artifacts_path, "convertation_config.yaml")

    def _adjust_paths_for_hf_to_local(self, artifacts_path: Path) -> None:
        """For map_hf_safetensor_to_local: output goes to a separate local_checkpoint dir."""
        self.local_checkpoint_path = Path(artifacts_path, "local_checkpoint")

    def _dump_convertation_config(
        self,
        convertation_config: ConvertationConfig,
        config_path: Path,
        is_master: bool,
    ) -> None:
        if is_master:
            OmegaConf.save(
                config=convertation_config.model_dump(exclude_none=True), f=config_path
            )
        else:
            time.sleep(3)

    def _build_convertation_context(
        self,
        checkpoint_path: Path,
        artifacts_path: Path,
        convertation_config: ConvertationConfig,
        nnodes: int,
        node_rank: int,
        mappers: list[str],
        convertation_log_file: Path,
        preparation_config: PreparationConfig,
        orkestration_manager: FastApiOrkestrationManager,
        is_worker: bool,
        is_master: bool,
        skip_preparation: bool,
        quantization_config: Optional[QuantizationConfig] = None,
    ) -> ConvertationContext:
        return ConvertationContext(
            convertation_config_path=self.convertation_config_path,
            hf_checkpoint_path=self.hf_checkpoint_path,
            checkpoint_path=checkpoint_path,
            artifacts_path=artifacts_path,
            prepared_checkpoint_path=self.prepared_checkpoint_path,
            tokenizer_path=self.tokenizer_path,
            log_file=convertation_log_file,
            originaly_hf=self.originaly_hf,
            originally_fsdp=self.originally_fsdp,
            mappers=mappers,
            global_rank=node_rank,
            world_size=nnodes,
            convertation_config=convertation_config,
            preparation_config=preparation_config,
            quantization_config=quantization_config,
            orkestration_manager=orkestration_manager,
            is_worker=is_worker,
            is_master=is_master,
            skip_preparation=skip_preparation,
            local_checkpoint_path=self.local_checkpoint_path,
        )

    def _build_pipeline(self) -> list[PipelineStage]:
        return [
            PreparationStage(self.convertation_context),
            SingleNodeConvertationStage(self.convertation_context),
            MultiNodeConvertationStage(self.convertation_context),
            MergeHFStage(self.convertation_context),
            QuantizeStage(self.convertation_context),
        ]

    def run_pipeline(self) -> None:
        for stage in self.pipeline:
            stage.run()
            stage.barrier_after_run()
        self.hf_checkpoint_path = self.convertation_context.hf_checkpoint_path
        self.tokenizer_path = self.convertation_context.tokenizer_path

    def setup(
        self,
        checkpoint_path: Path,
        artifacts_path: Path,
        convertation_config: ConvertationConfig,
        preparation_config: PreparationConfig,
        nnodes: int,
        node_rank: int,
        convertation_log_file: Path,
        orkestration_manager: FastApiOrkestrationManager,
        is_worker: bool,
        is_master: bool,
        quantization_config: Optional[QuantizationConfig] = None,
    ) -> None:
        mappers = convertation_config.mappers
        self._setup_paths(checkpoint_path, artifacts_path, mappers)

        # For HF→local reverse conversion, redirect output to artifacts
        if any(m.lower() == "map_hf_safetensor_to_local" for m in mappers):
            self._adjust_paths_for_hf_to_local(artifacts_path)
            Logger.info(
                f"[ConvertationManager] adjusted hf_checkpoint_path={self.hf_checkpoint_path}"
            )

        self._dump_convertation_config(
            convertation_config, self.convertation_config_path, is_master
        )
        skip_preparation = preparation_config.skip

        self.convertation_context = self._build_convertation_context(
            checkpoint_path,
            artifacts_path,
            convertation_config,
            nnodes,
            node_rank,
            mappers,
            convertation_log_file,
            preparation_config,
            orkestration_manager,
            is_worker,
            is_master,
            skip_preparation,
            quantization_config,
        )
        self.pipeline = self._build_pipeline()


class PipelineStage(ABC, ConvertationUtils):
    def __init__(self, name: str, convertation_context: ConvertationContext) -> None:
        self.name = name
        self.convertation_context = convertation_context
        self.cmd_to_run = []
        self.cwd = Path(".")
        self.build_cwd()
        self.build_cmd_to_run()

    def _get_path_to_utils_convertation(self) -> Path:
        for path in sys.path:
            if path.endswith("utils/convertation"):
                return Path(path)
        raise ValueError(
            "Utils/convertation not found in path, please updated PYTHONPATH"
        )

    def skip_if_originaly_hf(self) -> bool:
        # For HF→local conversion the input IS an HF checkpoint — don't blanket-skip
        if self.is_map_hf_safetensor_to_local():
            return False
        if self.is_map_safetensors_graft_mtp_safetensors():
            return False
        return (
            self.convertation_context.originaly_hf
            or self.convertation_context.originally_fsdp
        )

    def is_map_gigafsdp_to_hf(self) -> bool:
        return any(
            mapper.lower() == "map_gigafsdp_to_hf"
            for mapper in self.convertation_context.mappers
        )

    def is_map_gigafsdp_to_hf_safetensor(self) -> bool:
        return any(
            mapper.lower().replace("_omni", "") == "map_gigafsdp_to_hf_safetensor"
            for mapper in self.convertation_context.mappers
        )

    def is_map_hf_safetensor_to_local(self) -> bool:
        return any(
            mapper.lower() == "map_hf_safetensor_to_local"
            for mapper in self.convertation_context.mappers
        )

    def is_map_local_safetensor_to_hf(self) -> bool:
        return any(
            mapper.lower() == "map_local_safetensor_to_hf"
            for mapper in self.convertation_context.mappers
        )

    def is_map_gigafsdp_to_fsdp(self) -> bool:
        return any(
            mapper.lower() == "map_gigafsdp_to_fsdp"
            for mapper in self.convertation_context.mappers
        )

    def is_map_mixed_gigafsdp_to_fsdp(self) -> bool:
        return any(
            mapper.lower() == "map_mixed_gigafsdp_to_fsdp"
            for mapper in self.convertation_context.mappers
        )

    def is_map_mixed_gigafsdp_lora_to_hf_merged_model(self) -> bool:
        return any(
            mapper.lower() == "map_mixed_gigafsdp_lora_to_hf_merged_model"
            for mapper in self.convertation_context.mappers
        )

    def is_map_gigafsdp_to_fsdp_safetensor(self) -> bool:
        return any(
            mapper.lower() == "map_gigafsdp_to_fsdp_safetensor"
            for mapper in self.convertation_context.mappers
        )

    def is_map_mixed_gigafsdp_to_mtp_safetensors(self) -> bool:
        return any(
            mapper.lower() == "map_mixed_gigafsdp_to_mtp_safetensors"
            for mapper in self.convertation_context.mappers
        )

    def is_map_safetensors_graft_mtp_safetensors(self) -> bool:
        return any(
            mapper.lower() == "map_safetensors_graft_mtp_safetensors"
            for mapper in self.convertation_context.mappers
        )

    def is_map_gfsdp2_dcp_to_hf(self) -> bool:
        return any(
            mapper.lower() == "map_gfsdp2_dcp_to_hf"
            for mapper in self.convertation_context.mappers
        )

    def is_single_node_mapper(self) -> bool:
        return self.is_map_gigafsdp_to_hf() or self.is_map_gigafsdp_to_fsdp()

    def is_fsdp_output_mapper(self) -> bool:
        return (
            self.is_map_gigafsdp_to_fsdp()
            or self.is_map_gigafsdp_to_fsdp_safetensor()
            or self.is_map_mixed_gigafsdp_to_mtp_safetensors()
        )

    def update_and_save_convertation_config(self, dict_for_update: dict) -> None:
        self.convertation_context.convertation_config = (
            self.convertation_context.convertation_config.model_copy(
                update=dict_for_update
            )
        )

        OmegaConf.save(
            config=self.convertation_context.convertation_config.model_dump(
                exclude_none=True
            ),
            f=self.convertation_context.convertation_config_path,
        )

    def run(self) -> None:
        if self.skip_if_originaly_hf() or self.skip():
            Logger.info(f"Skipping {self.name} stage")
            return
        Logger.info(
            f"Running {self.name} stage with setup: {self.cmd_to_run} || cwd: {self.cwd}"
        )
        self.before_run()
        self.subprocess_manager = SubprocessManager(
            self.cmd_to_run,
            self.convertation_context.log_file,
            self.cwd,
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            path
            for path in env.get("PYTHONPATH", "").split(os.pathsep)
            if "contrib/lm_evaluation_harness" not in path
        )
        self.subprocess_manager.env = env
        self.subprocess_manager.start()
        self.subprocess_manager.wait_for_completion()
        self.subprocess_manager.raise_for_exit_code()
        self.after_run()
        Logger.info(f"{self.name} stage completed")

    def barrier_after_run(self) -> None:
        self.convertation_context.orkestration_manager.wait_barrier(
            f"{self.name}_DONE",
        )

    def skip_if_converted_hf_exists(self) -> bool:
        if self.is_map_hf_safetensor_to_local():
            return False
        path = self.convertation_context.hf_checkpoint_path
        return self.is_hf_checkpoint(path) or self.is_fsdp_checkpoint(path)

    @abstractmethod
    def skip(self) -> bool:
        pass  # pragma: no cover

    @abstractmethod
    def build_cmd_to_run(self) -> None:
        pass  # pragma: no cover

    @abstractmethod
    def build_cwd(self) -> None:
        pass  # pragma: no cover

    @abstractmethod
    def before_run(self) -> None:
        pass  # pragma: no cover

    @abstractmethod
    def after_run(self) -> None:
        pass  # pragma: no cover


class PreparationStage(PipelineStage):
    def __init__(self, convertation_context: ConvertationContext) -> None:
        super().__init__("PREPARATION", convertation_context)

    def skip(self) -> bool:
        if self.is_map_gigafsdp_to_hf():
            return True
        if self.is_map_hf_safetensor_to_local():
            return True
        if self.is_map_gfsdp2_dcp_to_hf():
            return True
        if self.convertation_context.skip_preparation:
            return True
        if self.skip_if_converted_hf_exists():
            return True
        return False

    def before_run(self) -> None:
        if self.convertation_context.prepared_checkpoint_path.is_dir():
            if self.convertation_context.is_master:
                if self.convertation_context.prepared_checkpoint_path.exists():
                    shutil.rmtree(self.convertation_context.prepared_checkpoint_path)
            else:
                time.sleep(60)

    def build_cmd_to_run(self) -> None:
        self.cmd_to_run = [
            "/home/user/convertation/bin/python",
            "distcp_ema_bf16_filter.py",
            "--src",
            str(self.convertation_context.checkpoint_path),
            "--dst",
            str(self.convertation_context.prepared_checkpoint_path),
            "--node-index",
            str(self.convertation_context.global_rank),
            "--num-nodes",
            str(self.convertation_context.world_size),
            *self.convertation_context.preparation_config.to_bash_list(),
        ]

    def build_cwd(self) -> None:
        self.cwd = self._get_path_to_utils_convertation() / "utils_script"

    def after_run(self) -> None:
        pass


class MultiNodeConvertationStage(PipelineStage):
    def __init__(self, convertation_context: ConvertationContext) -> None:
        super().__init__("MULTINODE_CONVERTATION", convertation_context)

    def skip(self) -> bool:
        if self.is_single_node_mapper():
            return True
        if self.skip_if_converted_hf_exists():
            return True
        return False

    def before_run(self) -> None:
        input_checkpoint_path = self.convertation_context.checkpoint_path

        skip_prepared_cleanup = (
            self.is_map_hf_safetensor_to_local()
            or self.is_map_safetensors_graft_mtp_safetensors()
        )
        if not skip_prepared_cleanup:
            if self.convertation_context.prepared_checkpoint_path.exists():
                if self.convertation_context.is_master:
                    shutil.rmtree(self.convertation_context.prepared_checkpoint_path)
                else:
                    time.sleep(20)

        if self.convertation_context.is_master:
            # For HF→local, output goes to local_checkpoint_path; otherwise to hf_checkpoint_path
            if (
                self.is_map_hf_safetensor_to_local()
                and self.convertation_context.local_checkpoint_path
            ):
                output_path = self.convertation_context.local_checkpoint_path
            else:
                output_path = self.convertation_context.hf_checkpoint_path
            self.update_and_save_convertation_config(
                {
                    "input_checkpoint_path": str(input_checkpoint_path),
                    "tokenizer_path": str(self.convertation_context.tokenizer_path),
                    "output_checkpoint_path": str(output_path),
                }
            )
            if output_path.exists():
                shutil.rmtree(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            time.sleep(5)

    def build_cmd_to_run(self) -> None:
        self.cmd_to_run = [
            "/home/user/convertation/bin/python",
            "map-fsdp-to-fsdp.py",
            "--input-config-path",
            str(self.convertation_context.convertation_config_path),
            "--global-rank",
            str(self.convertation_context.global_rank),
            "--world-size",
            str(self.convertation_context.world_size),
        ]

    def build_cwd(self) -> None:
        self.cwd = self._get_path_to_utils_convertation()

    def after_run(self) -> None:
        pass


class MergeHFStage(PipelineStage):
    def __init__(self, convertation_context: ConvertationContext) -> None:
        super().__init__("MERGE_HF", convertation_context)

    def skip(self) -> bool:
        if self.is_single_node_mapper():
            return True
        if self.is_fsdp_output_mapper():
            return True
        if self.convertation_context.is_worker:
            return True
        if self.skip_if_converted_hf_exists():
            return True
        return False

    def before_run(self) -> None:
        pass

    def build_cmd_to_run(self) -> None:
        if self.is_map_gfsdp2_dcp_to_hf():
            self.cmd_to_run = [
                "/home/user/convertation/bin/python",
                "map-fsdp-to-fsdp.py",
                "--input-config-path",
                str(self.convertation_context.convertation_config_path),
                "--finalize-only",
                "--world-size",
                str(self.convertation_context.world_size),
            ]
            return
        # For HF→local, merge goes to local_checkpoint_path
        if (
            self.is_map_hf_safetensor_to_local()
            and self.convertation_context.local_checkpoint_path
        ):
            base_dir = str(self.convertation_context.local_checkpoint_path)
        else:
            base_dir = str(self.convertation_context.hf_checkpoint_path)
        self.cmd_to_run = [
            "/home/user/convertation/bin/python",
            "merge_safetensors_ranks.py",
            "--base-dir",
            base_dir,
        ]

    def build_cwd(self) -> None:
        if self.is_map_gfsdp2_dcp_to_hf():
            self.cwd = self._get_path_to_utils_convertation()
            return
        self.cwd = self._get_path_to_utils_convertation() / "scripts"

    def after_run(self) -> None:
        pass


class QuantizeStage(PipelineStage):
    def __init__(self, convertation_context: ConvertationContext) -> None:
        super().__init__("QUANTIZE_FP8", convertation_context)

    @staticmethod
    def _extract_mlp_intermediate_size_from_safetensors_header(
        checkpoint_path: Path,
    ) -> Optional[int]:
        candidate_files = sorted(checkpoint_path.glob("*.safetensors"))
        for shard_path in candidate_files:
            with safe_open(str(shard_path), framework="pt", device="cpu") as f:
                for tensor_name in f.keys():
                    if tensor_name.endswith(
                        ".mlp.gate_proj.weight"
                    ) or tensor_name.endswith(".mlp.up_proj.weight"):
                        shape = tuple(f.get_slice(tensor_name).get_shape())
                        if len(shape) == 2 and isinstance(shape[0], int):
                            return shape[0]
                    if tensor_name.endswith(".mlp.down_proj.weight"):
                        shape = tuple(f.get_slice(tensor_name).get_shape())
                        if len(shape) == 2 and isinstance(shape[1], int):
                            return shape[1]
        return None

    def _should_run_pre_quantization_padding(
        self,
        mlp_intermediate_size: Optional[int],
        quantization_cfg: QuantizationConfig,
    ) -> bool:
        pre_pad_cfg = quantization_cfg.pre_quant_padding
        if not pre_pad_cfg.enabled:
            Logger.info("Skipping pre-quantization padding: disabled in config")
            return False

        if mlp_intermediate_size is None:
            Logger.info(
                "Could not detect model size from safetensors header; "
                "skipping pre-quantization padding stage"
            )
            return False

        target_sizes = pre_pad_cfg.only_if_mlp_intermediate_sizes
        if target_sizes is not None and mlp_intermediate_size not in target_sizes:
            Logger.info(
                "Skipping pre-quantization padding: "
                f"mlp_intermediate_size={mlp_intermediate_size} "
                f"is not in configured allow-list {target_sizes}"
            )
            return False

        padding_multiple = self._get_pre_quantization_padding_multiple(quantization_cfg)
        if mlp_intermediate_size % padding_multiple == 0:
            Logger.info(
                "Skipping pre-quantization padding: "
                f"mlp_intermediate_size={mlp_intermediate_size} already divisible by "
                f"{padding_multiple}"
            )
            return False

        return True

    def _get_pre_quantization_padding_multiple(
        self,
        quantization_cfg: QuantizationConfig,
    ) -> int:
        pre_pad_cfg = quantization_cfg.pre_quant_padding
        if pre_pad_cfg.multiple_of is not None:
            return pre_pad_cfg.multiple_of

        block_size = quantization_cfg.quantization_config.get(
            "weight_block_size", [128, 128]
        )
        return block_size[1]

    def _get_prepad_checkpoint_path(self) -> Path:
        return (
            self.convertation_context.artifacts_path
            / f"{self._get_quantized_checkpoint_path().name}_prepad"
        )

    def _prepare_checkpoint_for_pre_quantization_padding(
        self,
        quantization_cfg: QuantizationConfig,
    ) -> Path:
        input_hf_checkpoint_path = self.convertation_context.hf_checkpoint_path
        inplace = quantization_cfg.pre_quant_padding.inplace
        if inplace and (
            self.convertation_context.originaly_hf
            or self.convertation_context.originally_fsdp
        ):
            Logger.warning(
                "Pre-quantization padding: refusing to pad the source checkpoint "
                f"inplace ({input_hf_checkpoint_path}); falling back to a copy"
            )
            inplace = False

        if inplace:
            return input_hf_checkpoint_path

        checkpoint_dir_for_padding = self._get_prepad_checkpoint_path()
        if checkpoint_dir_for_padding.exists():
            shutil.rmtree(checkpoint_dir_for_padding)
        artifacts_dir_name = self.convertation_context.artifacts_path.name
        shutil.copytree(
            input_hf_checkpoint_path,
            checkpoint_dir_for_padding,
            ignore=shutil.ignore_patterns(artifacts_dir_name),
        )
        Logger.info(
            "Pre-quantization padding will run on a copied checkpoint "
            f"(inplace={inplace}): {checkpoint_dir_for_padding}"
        )
        return checkpoint_dir_for_padding

    def _run_pre_quantization_padding(
        self,
        *,
        checkpoint_dir: Path,
        script_path: Path,
        quantization_cfg: QuantizationConfig,
    ) -> None:
        cmd = [
            "/home/user/convertation/bin/python",
            str(script_path),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--multiple-of",
            str(self._get_pre_quantization_padding_multiple(quantization_cfg)),
        ]
        if quantization_cfg.pre_quant_padding.update_config:
            cmd.append("--update-config")
        Logger.info(
            f"Running pre-quantization padding before quantization: {' '.join(cmd)}"
        )
        pad_subprocess = SubprocessManager(
            cmd,
            self.convertation_context.log_file,
            script_path.parent,
        )
        pad_subprocess.start()
        pad_subprocess.wait_for_completion()
        pad_subprocess.raise_for_exit_code()

    def _run_pre_quantization_padding_if_needed(self) -> None:
        quantization_cfg = self.convertation_context.quantization_config
        if quantization_cfg is None:
            return
        if not quantization_cfg.pre_quant_padding.enabled:
            Logger.info("Skipping pre-quantization padding: disabled in config")
            return

        mlp_intermediate_size = (
            self._extract_mlp_intermediate_size_from_safetensors_header(
                self.convertation_context.hf_checkpoint_path
            )
        )
        if not self._should_run_pre_quantization_padding(
            mlp_intermediate_size,
            quantization_cfg,
        ):
            return

        input_hf_checkpoint_path = self.convertation_context.hf_checkpoint_path
        pad_script_path = (
            self._get_path_to_utils_convertation() / "scripts" / "pad_mlp_shards.py"
        )
        checkpoint_dir_for_padding = (
            self._prepare_checkpoint_for_pre_quantization_padding(quantization_cfg)
        )
        self._run_pre_quantization_padding(
            checkpoint_dir=checkpoint_dir_for_padding,
            script_path=pad_script_path,
            quantization_cfg=quantization_cfg,
        )

        if checkpoint_dir_for_padding != input_hf_checkpoint_path:
            self.convertation_context.hf_checkpoint_path = checkpoint_dir_for_padding
            Logger.info(
                "Switched quantization input checkpoint to padded copy: "
                f"{checkpoint_dir_for_padding}"
            )

    def skip_if_originaly_hf(self) -> bool:
        if (
            self.convertation_context.originaly_hf
            and not self.convertation_context.originally_fsdp
        ):
            return False
        return super().skip_if_originaly_hf()

    def _get_default_quantization_script_path(self) -> Path:
        return (
            self._get_path_to_utils_convertation()
            / "quantization"
            / "fine_grained_fp8_quantization_layer_by_layer.py"
        )

    def _get_default_quantization_config_path(self) -> Path:
        return (
            self._get_path_to_utils_convertation()
            / "configs"
            / "quantization_config.yaml"
        )

    def _get_quantization_section(self) -> dict:
        quantization = self.convertation_context.quantization_config
        if quantization is None:
            return {}
        return quantization.model_dump(exclude_none=True)

    def _get_quantization_script_path(self) -> Path:
        quantization_cfg = self._get_quantization_section()
        if quantization_cfg.get("script_path"):
            return Path(quantization_cfg["script_path"])
        return self._get_default_quantization_script_path()

    def _get_quantization_config_path(self) -> Path:
        quantization_cfg = self._get_quantization_section()
        if quantization_cfg.get("config_path"):
            return Path(quantization_cfg["config_path"])
        return self._get_default_quantization_config_path()

    def _get_quantized_checkpoint_path(self) -> Path:
        quantization_cfg = self._get_quantization_section()
        output_name = quantization_cfg.get(
            "output_checkpoint_name", "hf_checkpoint_fp8"
        )
        return self.convertation_context.artifacts_path / output_name

    def _set_hf_checkpoint_path(self, path: Path) -> None:
        self.convertation_context.hf_checkpoint_path = path
        self.convertation_context.tokenizer_path = path

    def _quantization_requested(self) -> bool:
        if self.convertation_context.quantization_config is None:
            return False
        if self.convertation_context.quantization_config.skip is not False:
            return False
        if self.is_fsdp_output_mapper():
            return False
        if self.is_map_hf_safetensor_to_local():
            return False
        if self.is_map_local_safetensor_to_hf():
            return False
        return True

    def _is_originally_fp8_hf(self) -> bool:
        if not self.convertation_context.originaly_hf:
            return False
        config = self._read_hf_config(self.convertation_context.hf_checkpoint_path)
        quantization_config = (config or {}).get("quantization_config")
        if not isinstance(quantization_config, dict):
            return False
        return str(quantization_config.get("quant_method", "")).lower() == "fp8"

    def _skip_if_originally_fp8(self) -> bool:
        if not self._is_originally_fp8_hf():
            return False

        Logger.info("Skipping QUANTIZE_FP8 stage: source checkpoint is already FP8")
        return True

    def _switch_to_existing_quantized_checkpoint(self) -> bool:
        quantized_checkpoint_path = self._get_quantized_checkpoint_path()
        if not self.is_hf_checkpoint(quantized_checkpoint_path):
            return False

        self._set_hf_checkpoint_path(quantized_checkpoint_path)
        Logger.info(
            "Skipping QUANTIZE_FP8 stage: using existing FP8 checkpoint "
            f"{quantized_checkpoint_path}"
        )
        return True

    def skip(self) -> bool:
        if not self._quantization_requested():
            return True
        if self._skip_if_originally_fp8():
            return True
        if self._switch_to_existing_quantized_checkpoint():
            return True
        if self.convertation_context.is_worker:
            return True
        return False

    def barrier_after_run(self) -> None:
        super().barrier_after_run()
        if (
            self._quantization_requested()
            and self.convertation_context.is_worker
            and not self._skip_if_originally_fp8()
        ):
            self._switch_to_existing_quantized_checkpoint()

    def before_run(self) -> None:
        self._run_pre_quantization_padding_if_needed()
        output_path = self._get_quantized_checkpoint_path()
        config_path = self._get_quantization_config_path()
        quantization_config = self.convertation_context.quantization_config.model_copy(
            update={
                "input_checkpoint_path": str(
                    self.convertation_context.hf_checkpoint_path
                ),
                "output_checkpoint_path": str(output_path),
                "script_path": str(self._get_quantization_script_path()),
                "config_path": str(config_path),
            }
        )

        if output_path.exists():
            shutil.rmtree(output_path)

        config_path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(
            config=quantization_config.model_dump(exclude_none=True),
            f=config_path,
        )

    def build_cmd_to_run(self) -> None:
        self.cmd_to_run = [
            "/home/user/convertation/bin/python",
            str(self._get_quantization_script_path()),
            "--config_path",
            str(self._get_quantization_config_path()),
        ]

    def build_cwd(self) -> None:
        self.cwd = self._get_quantization_script_path().parent

    def after_run(self) -> None:
        source_path = self._get_quantized_checkpoint_path()
        if not source_path.exists():
            raise FileNotFoundError(
                f"Quantized checkpoint was not created: {source_path}"
            )
        self._set_hf_checkpoint_path(source_path)
        JD.copy_jd_data_for_quantized_checkpoint(
            self.convertation_context.artifacts_path, source_path
        )

        prepad_path = self._get_prepad_checkpoint_path()
        if prepad_path.exists():
            shutil.rmtree(prepad_path)
            Logger.info(f"Removed temporary pre-padded checkpoint copy: {prepad_path}")


class SingleNodeConvertationStage(PipelineStage):
    def __init__(self, convertation_context: ConvertationContext) -> None:
        super().__init__("SINGLENODE_CONVERTATION", convertation_context)

    @staticmethod
    def _get_path_to_0_rank(checkpoint_path: Path) -> Path:
        list_dirs = os.listdir(checkpoint_path)
        list_dirs.sort()
        return Path(checkpoint_path, list_dirs[1])

    def skip(self) -> bool:
        if self.is_map_gigafsdp_to_hf_safetensor():
            return True
        if self.is_map_hf_safetensor_to_local():
            return True
        if self.is_map_local_safetensor_to_hf():
            return True
        if self.is_map_gigafsdp_to_fsdp_safetensor():
            return True
        if self.is_map_mixed_gigafsdp_to_mtp_safetensors():
            return True
        if self.is_map_safetensors_graft_mtp_safetensors():
            return True
        if self.is_map_gfsdp2_dcp_to_hf():
            return True
        if self.convertation_context.is_worker:
            return True
        if self.skip_if_converted_hf_exists():
            return True
        return False

    def before_run(self) -> None:
        if self.convertation_context.is_master:
            if (
                self.is_map_mixed_gigafsdp_to_fsdp()
                or self.is_map_mixed_gigafsdp_lora_to_hf_merged_model()
            ):
                input_checkpoint_path = self.convertation_context.checkpoint_path
            else:
                input_checkpoint_path = self._get_path_to_0_rank(
                    self.convertation_context.checkpoint_path
                )
            self.update_and_save_convertation_config(
                {
                    "input_checkpoint_path": str(input_checkpoint_path),
                    "tokenizer_path": str(self.convertation_context.tokenizer_path),
                    "output_checkpoint_path": str(
                        self.convertation_context.hf_checkpoint_path
                    ),
                }
            )
            if self.convertation_context.hf_checkpoint_path.exists():
                shutil.rmtree(self.convertation_context.hf_checkpoint_path)
            Path(self.convertation_context.hf_checkpoint_path).mkdir(
                parents=True, exist_ok=True
            )
        else:
            time.sleep(10)

    def build_cmd_to_run(self) -> None:
        self.cmd_to_run = [
            "/home/user/convertation/bin/python",
            "map-fsdp-to-fsdp.py",
            "--input-config-path",
            str(self.convertation_context.convertation_config_path),
        ]

    def build_cwd(self) -> None:
        self.cwd = self._get_path_to_utils_convertation()

    def after_run(self) -> None:
        pass
