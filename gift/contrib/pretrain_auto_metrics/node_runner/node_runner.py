import shutil
import time
import traceback
from pathlib import Path

from src.arg_parser import parse_arguments
from src.config_manager import ConfigManager
from src.convertation_manager import ConvertationManager
from src.inference.engine import InferenceEngine
from src.inference.inference_manager import InferenceManager
from src.job_dispatcher import JD
from src.logger import Logger
from src.metrics.metrics_manager import AutoMetricsManager
from src.orkestration_manager import FastApiOrkestrationManager
from src.trace import EventStatus, Trace


class NodeRunner:
    def __init__(
        self,
        checkpoint_path: str | None,
        config_path: str | None,
        mode: str,
        config_manager: ConfigManager,
        logger: Logger,
        orkestration_manager: FastApiOrkestrationManager,
        auto_metrics_manager: AutoMetricsManager,
        convertation_manager: ConvertationManager,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.config_path = Path(config_path) if config_path else None
        self.mode = mode
        self.config_manager = config_manager
        self.logger = logger
        self.orkestration_manager = orkestration_manager
        self.auto_metrics_manager = auto_metrics_manager
        self.convertation_manager = convertation_manager
        self.base_inference_engine: InferenceEngine | None = None

    def _setup_managers(self):
        self.config_manager.setup(
            self.checkpoint_path,
            config_path=self.config_path,
        )
        JD.reports_to_jd = not self.config_manager.config.is_replay_mode

        self.logger.setup(
            self.config_manager.artifacts_path,
            self.config_manager.node_name,
            self.config_manager.is_master,
            self.config_manager.is_slurm,
        )
        Trace.setup(self.config_manager)
        self.orkestration_manager.setup(
            self.config_manager.node_name,
            self.config_manager.config.autometrics.globals.orkestration_port,
            self.config_manager.global_nnodes,
            self.config_manager.global_node_rank,
            self.config_manager.global_dist_init_host,
        )

        if self.config_manager.config.runs_without_checkpoint:
            self.convertation_manager.hf_checkpoint_path = (
                self.config_manager.artifacts_path
            )
            return

        self.convertation_manager.setup(
            checkpoint_path=self.checkpoint_path,
            artifacts_path=self.config_manager.artifacts_path,
            convertation_config=self.config_manager.config.autometrics.convertation,
            preparation_config=self.config_manager.config.autometrics.preparation,
            quantization_config=self.config_manager.config.autometrics.quantization,
            nnodes=self.config_manager.global_nnodes,
            node_rank=self.config_manager.global_node_rank,
            convertation_log_file=self.logger.convertation_log_file,
            orkestration_manager=self.orkestration_manager,
            is_worker=self.orkestration_manager.is_worker,
            is_master=self.orkestration_manager.is_master,
        )

    def _setup_runtime_managers(self):
        self._setup_inference_manager()
        if self.mode == "inference":
            self.base_inference_engine = InferenceManager.setup_inference_engine()
        if self.mode == "autometrics":
            self.auto_metrics_manager.setup(
                config_manager=self.config_manager,
                run_config=self.config_manager.config,
                orkestration_manager=self.orkestration_manager,
                convertation_manager=self.convertation_manager,
            )

    def _setup_inference_manager(self) -> None:
        autometrics_config = self.config_manager.config.autometrics
        InferenceManager.setup(
            inference_config=autometrics_config.inference,
            global_node_rank=self.config_manager.global_node_rank,
            gpus_per_node=self.config_manager.gpus_per_node,
            globals_config=autometrics_config.globals,
            hf_checkpoint_path=self.convertation_manager.hf_checkpoint_path,
            router_config=autometrics_config.router,
            obm_proxy_config=autometrics_config.obm_proxy,
            registered_nodes=self.orkestration_manager.get_registered_nodes(),
            secondary_inference=autometrics_config.secondary_inference,
        )

    def _start_base_inference(self) -> None:
        engine = self.base_inference_engine
        om = self.orkestration_manager
        engine_units = engine.units()
        runs_on_nodes = any(unit.ranks for unit in engine_units)
        if om.is_master or runs_on_nodes:
            engine.start()
        om.wait_barrier("INFERENCE_READY")
        if om.is_master:
            engine.attach(engine_units)
        om.wait_barrier("ROUTER_START")
        Logger.info("Base inference engine started (inference mode)")

    def _cleanup_artifacts(self):
        if self.config_manager.config.runs_without_checkpoint:
            Logger.info("Artifact cleanup skipped: nothing was converted")
            return
        if self.orkestration_manager.is_worker:
            Logger.info("Artifact cleanup: is worker, skipping")
            return
        if (
            self.convertation_manager.originaly_hf
            or self.convertation_manager.originally_fsdp
        ):
            Logger.info(
                "Artifact cleanup skipped: source checkpoint is already HF/FSDP"
            )
            return
        if not self.config_manager.config.autometrics.globals.delete_converted_hf:
            Logger.info("Artifact cleanup skipped: delete_converted_hf is disabled")
            return
        if self.mode == "convertation":
            Logger.info("Artifact cleanup skipped: convertation mode")
            return
        if self.convertation_manager.hf_checkpoint_path.exists():
            shutil.rmtree(self.convertation_manager.hf_checkpoint_path)
            Logger.info("Artifact cleanup: removed converted HF checkpoint")

    def run(self):
        self._setup_managers()
        om = self.orkestration_manager
        om.wait_barrier("REGISTRATION")
        if not self.config_manager.config.runs_without_checkpoint:
            self.convertation_manager.run_pipeline()
        self._setup_runtime_managers()
        om.wait_barrier("INFERENCE_START")
        if self.mode == "inference":
            self._start_base_inference()
            if om.is_worker:
                om.wait_barrier("INFERENCE_STOP")
                return
            while True:
                time.sleep(3600)
        if self.mode == "autometrics":
            self.auto_metrics_manager.run_pipeline()
        if self.mode == "convertation":
            Logger.info("Convertation done, stopping node runner...")
        om.wait_barrier("CLEANUP")

    def cleanup(self, status: EventStatus = "ok"):
        if self.orkestration_manager.is_master:
            time.sleep(5)
        try:
            self._cleanup_artifacts()
            self.orkestration_manager.cleanup()
        finally:
            # The scrapers write until their engine stops, so the completion
            # marker is only true once every engine is down.
            self.auto_metrics_manager.cleanup()
            if self.base_inference_engine is not None:
                try:
                    self.base_inference_engine.close()
                    Logger.info("Cleanup: inference engine stopped")
                except Exception:
                    pass
            Trace.close(status)
        Logger.info("Cleanup finished")


def main():
    checkpoint_path, mode, config_path = parse_arguments()
    node_runner = NodeRunner(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        mode=mode,
        config_manager=ConfigManager(),
        logger=Logger(),
        orkestration_manager=FastApiOrkestrationManager(),
        auto_metrics_manager=AutoMetricsManager(),
        convertation_manager=ConvertationManager(),
    )
    status: EventStatus = "ok"
    try:
        node_runner.run()
    except Exception:
        status = "error"
        Logger.error(f"Node runner crashed:\n{traceback.format_exc()}")
        raise
    finally:
        try:
            node_runner.cleanup(status)
        except Exception:
            pass


if __name__ == "__main__":  # pragma: no cover
    main()
