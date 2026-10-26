import os
from pathlib import Path
from typing import Any

from ..config_model import (
    ExternalApiInferenceConfig,
    GlobalsConfig,
    InferenceConfig,
    ObmProxyConfig,
    RouterConfig,
    SglangInferenceConfig,
)
from ..logger import Logger
from .engine import InferenceEngine
from .external_api.engine import ExternalApiEngine
from .generations.dummy_engine import DummyEngine
from .sglang.sglang_engine import WORKER_PORT_OFFSET, SglangEngine
from .sglang.sglang_pd_engine import SglangPdEngine


class InferenceManager:
    _inference_config: InferenceConfig
    _global_node_rank: int
    _gpus_per_node: int
    _globals_config: GlobalsConfig
    _hf_checkpoint_path: Path
    _router_config: RouterConfig
    _obm_proxy_config: ObmProxyConfig
    _registered_nodes: dict[int, str]
    _secondary_inference: dict[str, dict[str, Any]]

    @classmethod
    def setup(
        cls,
        *,
        inference_config: InferenceConfig,
        global_node_rank: int,
        gpus_per_node: int,
        globals_config: GlobalsConfig,
        hf_checkpoint_path: Path | None = None,
        router_config: RouterConfig,
        obm_proxy_config: ObmProxyConfig,
        registered_nodes: dict[int, str],
        secondary_inference: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        cls._inference_config = inference_config
        cls._global_node_rank = global_node_rank
        cls._gpus_per_node = gpus_per_node
        cls._globals_config = globals_config
        cls._hf_checkpoint_path = hf_checkpoint_path
        cls._router_config = router_config
        cls._obm_proxy_config = obm_proxy_config
        cls._registered_nodes = registered_nodes
        cls._secondary_inference = secondary_inference or {}

    @classmethod
    def setup_inference_engine(cls) -> InferenceEngine:
        return cls._setup_inference_engine(
            inference_config=cls._inference_config,
            hf_checkpoint_path=cls._hf_checkpoint_path,
            obm_proxy_config=cls._obm_proxy_config,
            log_profile="evaluated",
            router_config=cls._router_config,
        )

    @classmethod
    def setup_preset_inference_engine(
        cls,
        *,
        model_name: str,
        judge_index: int,
    ) -> InferenceEngine:
        ports_per_engine = WORKER_PORT_OFFSET + 1
        engine_slot = judge_index + 1
        router_config = cls._router_config.model_copy(
            update={
                "port": cls._router_config.port + ports_per_engine * engine_slot,
                "prometheus_port": cls._router_config.prometheus_port + engine_slot,
            }
        )
        is_external_api = ":" in model_name
        if is_external_api:
            platform, _, external_model = model_name.partition(":")
            external_config = ExternalApiInferenceConfig.for_secondary_platform(
                platform, external_model, cls._secondary_inference
            )
            external_config = external_config.model_copy(
                update={"port": router_config.port}
            )
            return cls._setup_inference_engine(
                inference_config=external_config,
                hf_checkpoint_path=None,
                obm_proxy_config=None,
                log_profile=model_name,
                model_name=model_name,
                router_config=router_config,
            )

        if cls._globals_config.replay_generations_dir is not None:
            hf_checkpoint_path = None
        else:
            pam_models_path_value = os.getenv("PAM_MODELS_PATH")
            if pam_models_path_value is None:
                raise ValueError("PAM_MODELS_PATH is required when model_name is set")
            pam_models_path = Path(pam_models_path_value)
            assert pam_models_path.exists(), (
                f"Checkpoint path {pam_models_path} does not exist"
            )
            hf_checkpoint_path = pam_models_path / model_name
            assert hf_checkpoint_path.exists(), (
                f"Model checkpoint path {hf_checkpoint_path} does not exist"
            )
        inference_config = SglangInferenceConfig.for_secondary_preset(
            model_name, cls._secondary_inference
        )
        return cls._setup_inference_engine(
            inference_config=inference_config,
            hf_checkpoint_path=hf_checkpoint_path,
            obm_proxy_config=None,
            log_profile=model_name,
            model_name=model_name,
            router_config=router_config,
        )

    @classmethod
    def _setup_inference_engine(
        cls,
        *,
        inference_config: InferenceConfig,
        hf_checkpoint_path: Path | None,
        obm_proxy_config: ObmProxyConfig | None,
        log_profile: str,
        router_config: RouterConfig,
        model_name: str | None = None,
    ) -> InferenceEngine:
        if inference_config.engine_type == "external_api":
            return ExternalApiEngine(
                inference_config=inference_config,
                obm_proxy_config=obm_proxy_config,
                hf_checkpoint_path=hf_checkpoint_path,
                log_profile=log_profile,
                model_name=model_name,
            )

        if cls._globals_config.replay_generations_dir is not None:
            return DummyEngine(
                replay_generations_dir=cls._globals_config.replay_generations_dir,
                port=router_config.port,
                obm_proxy_config=obm_proxy_config,
                hf_checkpoint_path=hf_checkpoint_path,
                log_profile=log_profile,
                model_name=model_name,
            )

        if hf_checkpoint_path is None:
            raise ValueError(
                "hf_checkpoint_path is required for non-external-api inference"
            )

        if inference_config.engine_type == "sglang_pd":
            engine_cls = SglangPdEngine
            side_configs = (inference_config.prefill, inference_config.decode)
        elif inference_config.engine_type == "sglang":
            engine_cls = SglangEngine
            side_configs = (inference_config,)
        else:
            raise ValueError(
                f"Unsupported inference engine: {inference_config.engine_type}"
            )
        for side_config in side_configs:
            cls._maybe_update_tokenizer_path_from_checkpoint(
                inference_config=side_config,
                hf_checkpoint_path=hf_checkpoint_path,
            )
        return engine_cls(
            hf_checkpoint_path=hf_checkpoint_path,
            nodes=cls._registered_nodes,
            inference_config=inference_config,
            global_node_rank=cls._global_node_rank,
            gpus_per_node=cls._gpus_per_node,
            router_config=router_config,
            obm_proxy_config=obm_proxy_config,
            log_profile=log_profile,
            model_name=model_name,
            record_generations_dir=cls._globals_config.record_generations_dir,
        )

    @classmethod
    def _maybe_update_tokenizer_path_from_checkpoint(
        cls,
        inference_config: InferenceConfig,
        hf_checkpoint_path: Path,
    ) -> None:
        processor_candidates = [
            (
                inference_config.processor_path,
                "vision processor was detected at processor_path",
            ),
            (
                hf_checkpoint_path / inference_config.processor_internal_path,
                "vision processor was found in converted checkpoint",
            ),
        ]
        for processor_path, reason in processor_candidates:
            if processor_path is None or not Path(processor_path).exists():
                continue
            processor_path = str(processor_path)
            if inference_config.tokenizer_path != processor_path:
                Logger.warning(
                    f"Tokenizer path was changed because {reason}: {processor_path}"
                )
            inference_config.tokenizer_path = processor_path
            return
