from pathlib import Path

import pytest

from node_runner.src.config_model import (
    MODEL_NAME_PARAMS_MAPPER,
    ExternalApiInferenceConfig,
    GlobalsConfig,
    InferenceConfig,
    ObmProxyConfig,
    RouterConfig,
    SglangInferenceConfig,
    SglangPdInferenceConfig,
)
from node_runner.src.inference.external_api.engine import ExternalApiEngine
from node_runner.src.inference.generations.dummy_engine import DummyEngine
from node_runner.src.inference.inference_manager import InferenceManager
from node_runner.src.inference.sglang.sglang_inference import SglangInference

pytestmark = pytest.mark.unit


def _setup_manager(
    *,
    inference_config: InferenceConfig | None = None,
    hf_checkpoint_path: Path | None = Path("/tmp/model"),
    global_node_rank: int = 0,
    gpus_per_node: int = 8,
    globals_config: GlobalsConfig | None = None,
    router_config: RouterConfig | None = None,
    registered_nodes: dict[int, str] | None = None,
    secondary_inference: dict[str, dict] | None = None,
) -> None:
    InferenceManager.setup(
        inference_config=inference_config or SglangInferenceConfig(tp=1),
        hf_checkpoint_path=hf_checkpoint_path,
        global_node_rank=global_node_rank,
        gpus_per_node=gpus_per_node,
        globals_config=globals_config or GlobalsConfig(),
        router_config=router_config or RouterConfig(port=30000),
        obm_proxy_config=ObmProxyConfig(),
        registered_nodes=registered_nodes or {0: "127.0.0.1"},
        secondary_inference=secondary_inference,
    )


class TestInferenceManager:
    def test_setup_inference_engine_guards_against_non_sglang_runtime_value(self):
        inference_cfg = InferenceConfig.model_construct(
            engine_type="runtime-override",
            tp=1,
            pp=1,
            ep=1,
            dp=1,
            dtype="auto",
            reasoning_parser=None,
            tool_call_parser=None,
            mem_fraction_static=0.7,
            trust_remote_code=True,
            allow_auto_truncate=True,
            max_running_requests=1,
            chat_template=None,
        )
        _setup_manager(inference_config=inference_cfg)

        with pytest.raises(ValueError, match="Unsupported inference engine"):
            InferenceManager.setup_inference_engine()

    def test_setup_inference_engine_unsupported_type(self):
        inference_cfg = InferenceConfig.model_construct(engine_type="unknown")
        _setup_manager(inference_config=inference_cfg)
        with pytest.raises(ValueError, match="Unsupported inference engine"):
            InferenceManager.setup_inference_engine()

    def test_setup_inference_engine_sglang(self, monkeypatch: pytest.MonkeyPatch):
        created = {}

        class _FakeEngine:
            def __init__(self, **kwargs):
                created.update(kwargs)

        monkeypatch.setattr(
            "node_runner.src.inference.inference_manager.SglangEngine",
            _FakeEngine,
        )
        inference_cfg = SglangInferenceConfig(tp=8)
        router_cfg = RouterConfig(port=30000)
        _setup_manager(
            inference_config=inference_cfg,
            hf_checkpoint_path=Path("/tmp/model"),
            global_node_rank=0,
            gpus_per_node=4,
            globals_config=GlobalsConfig(),
            router_config=router_cfg,
            registered_nodes={0: "127.0.0.1", 1: "127.0.0.2"},
        )
        engine = InferenceManager.setup_inference_engine()
        assert isinstance(engine, _FakeEngine)
        assert created["hf_checkpoint_path"] == Path("/tmp/model")
        side_config = created["inference_config"]
        assert created["nodes"] == {0: "127.0.0.1", 1: "127.0.0.2"}
        assert side_config.tp == 8
        assert "engine_type" not in side_config.to_dict()

    def test_setup_inference_engine_sglang_pd(self, monkeypatch: pytest.MonkeyPatch):
        created = {}

        class _FakeEngine:
            def __init__(self, **kwargs):
                created.update(kwargs)

        monkeypatch.setattr(
            "node_runner.src.inference.inference_manager.SglangPdEngine",
            _FakeEngine,
        )
        inference_cfg = SglangPdInferenceConfig(
            prefill={"tp": 8},
            decode={"tp": 8},
        )
        _setup_manager(
            inference_config=inference_cfg,
            hf_checkpoint_path=Path("/tmp/model"),
            gpus_per_node=8,
            registered_nodes={0: "127.0.0.1", 1: "127.0.0.2"},
        )
        engine = InferenceManager.setup_inference_engine()
        assert isinstance(engine, _FakeEngine)
        assert created["nodes"] == {0: "127.0.0.1", 1: "127.0.0.2"}
        assert created["inference_config"] is inference_cfg

    def test_setup_inference_engine_maps_processor_path_to_tokenizer(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        created = {}

        class _FakeEngine:
            def __init__(self, **kwargs):
                created.update(kwargs)

        monkeypatch.setattr(
            "node_runner.src.inference.inference_manager.SglangEngine",
            _FakeEngine,
        )
        inference_cfg = SglangInferenceConfig.model_validate(
            {
                "engine_type": "sglang",
                "tp": 1,
                "processor_path": "/tmp/gigavision-processor",
                "tokenizer_path": "/tmp/tokenizer",
            }
        )

        _setup_manager(
            inference_config=inference_cfg,
            hf_checkpoint_path=Path("/tmp/model"),
        )
        InferenceManager.setup_inference_engine()

        normalized_cfg = created["inference_config"]
        assert normalized_cfg.tokenizer_path == "/tmp/tokenizer"
        assert "processor_path" not in normalized_cfg.to_dict()

    def test_setup_inference_engine_uses_processor_from_checkpoint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        model_path = tmp_path / "model"
        processor_path = model_path / "processors" / "GigaVisionProcessor"
        processor_path.mkdir(parents=True, exist_ok=True)
        created = {}

        class _FakeEngine:
            def __init__(self, **kwargs):
                created.update(kwargs)

        monkeypatch.setattr(
            "node_runner.src.inference.inference_manager.SglangEngine",
            _FakeEngine,
        )

        _setup_manager(
            inference_config=SglangInferenceConfig(tp=1),
            hf_checkpoint_path=model_path,
        )
        InferenceManager.setup_inference_engine()

        side_config = created["inference_config"]
        assert side_config.tokenizer_path == processor_path.as_posix()

    def test__maybe_update_tokenizer_path_from_checkpoint_uses_processor_from_checkpoint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        model_path = tmp_path / "model"
        processor_path = model_path / "processors" / "GigaVisionProcessor"
        processor_path.mkdir(parents=True, exist_ok=True)
        inference_cfg = SglangInferenceConfig(tp=1)
        warnings = []
        monkeypatch.setattr(
            "node_runner.src.inference.inference_manager.Logger.warning",
            lambda message: warnings.append(message),
        )

        InferenceManager._maybe_update_tokenizer_path_from_checkpoint(
            inference_config=inference_cfg,
            hf_checkpoint_path=model_path,
        )

        assert inference_cfg.tokenizer_path == processor_path.as_posix()
        assert len(warnings) == 1
        assert "vision processor was found in converted checkpoint" in warnings[0]

    def test__maybe_update_tokenizer_path_from_checkpoint_uses_configured_processor_internal_path(
        self, tmp_path: Path
    ):
        model_path = tmp_path / "model"
        processor_path = model_path / "processors" / "CustomVisionProcessor"
        processor_path.mkdir(parents=True, exist_ok=True)
        inference_cfg = SglangInferenceConfig(
            tp=1,
            processor_internal_path="processors/CustomVisionProcessor",
        )

        InferenceManager._maybe_update_tokenizer_path_from_checkpoint(
            inference_config=inference_cfg,
            hf_checkpoint_path=model_path,
        )

        assert inference_cfg.tokenizer_path == processor_path.as_posix()

    def test__maybe_update_tokenizer_path_from_checkpoint_keeps_tokenizer_when_processor_is_missing(
        self, tmp_path: Path
    ):
        inference_cfg = SglangInferenceConfig(
            tp=1,
            tokenizer_path="/tmp/explicit-tokenizer",
        )

        InferenceManager._maybe_update_tokenizer_path_from_checkpoint(
            inference_config=inference_cfg,
            hf_checkpoint_path=tmp_path / "model",
        )

        assert inference_cfg.tokenizer_path == "/tmp/explicit-tokenizer"

    def test__maybe_update_tokenizer_path_from_checkpoint_uses_existing_explicit_processor_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        explicit_processor_path = tmp_path / "explicit" / "gigavision-processor"
        explicit_processor_path.mkdir(parents=True, exist_ok=True)
        inference_cfg = SglangInferenceConfig.model_validate(
            {
                "engine_type": "sglang",
                "tp": 1,
                "processor_path": explicit_processor_path.as_posix(),
            }
        )
        warnings = []
        monkeypatch.setattr(
            "node_runner.src.inference.inference_manager.Logger.warning",
            lambda message: warnings.append(message),
        )
        InferenceManager._maybe_update_tokenizer_path_from_checkpoint(
            inference_config=inference_cfg,
            hf_checkpoint_path=tmp_path / "model",
        )
        assert inference_cfg.tokenizer_path == explicit_processor_path.as_posix()
        assert "processor_path" not in inference_cfg.to_dict()
        assert len(warnings) == 1
        assert "vision processor was detected at processor_path" in warnings[0]

    def test__maybe_update_tokenizer_path_from_checkpoint_skips_warning_when_explicit_processor_matches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        explicit_processor_path = tmp_path / "explicit" / "gigavision-processor"
        explicit_processor_path.mkdir(parents=True, exist_ok=True)
        inference_cfg = SglangInferenceConfig.model_validate(
            {
                "engine_type": "sglang",
                "tp": 1,
                "processor_path": explicit_processor_path.as_posix(),
                "tokenizer_path": explicit_processor_path.as_posix(),
            }
        )
        warnings = []
        monkeypatch.setattr(
            "node_runner.src.inference.inference_manager.Logger.warning",
            lambda message: warnings.append(message),
        )

        InferenceManager._maybe_update_tokenizer_path_from_checkpoint(
            inference_config=inference_cfg,
            hf_checkpoint_path=tmp_path / "model",
        )

        assert inference_cfg.tokenizer_path == explicit_processor_path.as_posix()
        assert warnings == []

    def test__maybe_update_tokenizer_path_from_checkpoint_skips_warning_when_checkpoint_processor_matches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        model_path = tmp_path / "model"
        processor_path = model_path / "processors" / "GigaVisionProcessor"
        processor_path.mkdir(parents=True, exist_ok=True)
        inference_cfg = SglangInferenceConfig(
            tp=1,
            tokenizer_path=processor_path.as_posix(),
        )
        warnings = []
        monkeypatch.setattr(
            "node_runner.src.inference.inference_manager.Logger.warning",
            lambda message: warnings.append(message),
        )

        InferenceManager._maybe_update_tokenizer_path_from_checkpoint(
            inference_config=inference_cfg,
            hf_checkpoint_path=model_path,
        )

        assert inference_cfg.tokenizer_path == processor_path.as_posix()
        assert warnings == []

    def test_setup_inference_engine_mapped_model_missing_paths(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setenv("PAM_MODELS_PATH", str(tmp_path / "no_models"))
        _setup_manager()
        with pytest.raises(AssertionError, match="does not exist"):
            InferenceManager.setup_preset_inference_engine(
                model_name="deepseek-v3", judge_index=0
            )

        models_root = tmp_path / "models"
        models_root.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("PAM_MODELS_PATH", str(models_root))
        with pytest.raises(AssertionError, match="Model checkpoint path"):
            InferenceManager.setup_preset_inference_engine(
                model_name="deepseek-v3", judge_index=0
            )

    def test_setup_inference_engine_mapped_model_happy_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        models_root = tmp_path / "models"
        model_name = "deepseek-v3"
        (models_root / model_name).mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("PAM_MODELS_PATH", str(models_root))
        created = {}

        class _FakeEngine:
            def __init__(self, **kwargs):
                created.update(kwargs)

        monkeypatch.setattr(
            "node_runner.src.inference.inference_manager.SglangEngine",
            _FakeEngine,
        )
        _setup_manager(
            global_node_rank=1,
            gpus_per_node=16,
            globals_config=GlobalsConfig(),
            router_config=RouterConfig(port=30000),
            registered_nodes={0: "127.0.0.1", 1: "127.0.0.2"},
        )
        engine = InferenceManager.setup_preset_inference_engine(
            model_name=model_name, judge_index=1
        )
        assert isinstance(engine, _FakeEngine)
        assert created["hf_checkpoint_path"] == models_root / model_name
        side_config = created["inference_config"]
        assert side_config.tp == MODEL_NAME_PARAMS_MAPPER[model_name]["tp"]
        assert created["router_config"] == RouterConfig(
            port=30004, prometheus_port=29002
        )

    def test_setup_inference_engine_requires_config_when_no_model_name(self):
        _setup_manager(hf_checkpoint_path=None)
        with pytest.raises(ValueError, match="hf_checkpoint_path is required"):
            InferenceManager.setup_inference_engine()

    def test_setup_inference_engine_unknown_mapped_model_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        models_root = tmp_path / "models"
        model_name = "custom-judge-model"
        (models_root / model_name).mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("PAM_MODELS_PATH", str(models_root))

        _setup_manager()
        with pytest.raises(ValueError, match="unknown SGLang preset"):
            InferenceManager.setup_preset_inference_engine(
                model_name=model_name, judge_index=0
            )

    def test_setup_preset_inference_engine_requires_models_env(self, monkeypatch):
        monkeypatch.delenv("PAM_MODELS_PATH", raising=False)
        _setup_manager()

        with pytest.raises(ValueError, match="PAM_MODELS_PATH is required"):
            InferenceManager.setup_preset_inference_engine(
                model_name="deepseek-v3", judge_index=0
            )

    def test_setup_preset_inference_engine_routes_colon_name_to_external_api(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # external secondary must not require PAM_MODELS_PATH or a local preset
        monkeypatch.delenv("PAM_MODELS_PATH", raising=False)
        created = {}

        class _FakeExternalApiEngine:
            def __init__(self, **kwargs):
                created.update(kwargs)

        monkeypatch.setattr(
            "node_runner.src.inference.inference_manager.ExternalApiEngine",
            _FakeExternalApiEngine,
        )
        _setup_manager(
            secondary_inference={"openrouter": {"api_key": "sk-or-test"}},
        )

        engine = InferenceManager.setup_preset_inference_engine(
            model_name="openrouter:qwen/qwen3-235b", judge_index=0
        )

        assert isinstance(engine, _FakeExternalApiEngine)
        cfg = created["inference_config"]
        assert cfg.model == "qwen/qwen3-235b"
        assert cfg.api_key == "sk-or-test"
        assert cfg.port == 30002
        assert created["hf_checkpoint_path"] is None
        assert created["model_name"] == "openrouter:qwen/qwen3-235b"

    def test_setup_inference_engine_external_api(self, monkeypatch: pytest.MonkeyPatch):
        created = {}

        class _FakeExternalApiEngine:
            def __init__(self, **kwargs):
                created.update(kwargs)

        monkeypatch.setattr(
            "node_runner.src.inference.inference_manager.ExternalApiEngine",
            _FakeExternalApiEngine,
        )
        inference_cfg = ExternalApiInferenceConfig.model_validate(
            {
                "formatting": "openai",
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-4o-mini",
                "api_key": "key",
            }
        )
        _setup_manager(
            inference_config=inference_cfg,
            hf_checkpoint_path=None,
            global_node_rank=1,
            registered_nodes={0: "127.0.0.1", 1: "127.0.0.2"},
        )

        engine = InferenceManager.setup_inference_engine()

        assert isinstance(engine, _FakeExternalApiEngine)
        assert created["inference_config"] is inference_cfg


class TestReplayEngine:
    def test_a_replay_dump_puts_a_dummy_engine_in_front_of_the_shooters(self):
        _setup_manager(
            globals_config=GlobalsConfig(replay_generations_dir="/tmp/generations")
        )

        engine = InferenceManager.setup_inference_engine()

        assert isinstance(engine, DummyEngine)
        assert engine.generations_manager.bash_cmd[-3:] == [
            "/tmp/generations",
            "--profile",
            "evaluated",
        ]

    def test_a_replayed_judge_needs_neither_models_path_nor_weights(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("PAM_MODELS_PATH", raising=False)
        _setup_manager(
            globals_config=GlobalsConfig(replay_generations_dir="/tmp/generations")
        )

        engine = InferenceManager.setup_preset_inference_engine(
            model_name="deepseek-v3", judge_index=0
        )

        assert isinstance(engine, DummyEngine)
        assert engine.log_profile == "deepseek-v3"
        assert engine.base_url == "http://127.0.0.1:30002"


@pytest.mark.parametrize("mode", ["record", "replay"])
def test_judge_generations_are_dumped_and_replayed_under_its_model_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str
):
    (tmp_path / "gpt-oss-120b").mkdir()
    monkeypatch.setenv("PAM_MODELS_PATH", str(tmp_path))
    monkeypatch.setattr(
        SglangInference, "_setup_deepgemm_cache_dir", lambda self, nfs, local: None
    )
    _setup_manager(
        globals_config=GlobalsConfig(**{f"{mode}_generations_dir": "/tmp/generations"})
    )

    engine = InferenceManager.setup_preset_inference_engine(
        model_name="gpt-oss-120b", judge_index=0
    )

    cmd = engine.generations_manager.bash_cmd
    assert cmd[cmd.index("--profile") + 1] == "gpt-oss-120b"
    assert cmd[cmd.index("--dump-dir") + 1] == "/tmp/generations"


def _port(url: str) -> int:
    return int(url.rsplit(":", 1)[1])


class TestJudgePorts:
    def test_judges_listen_on_ports_no_other_engine_uses(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        (tmp_path / "gpt-oss-120b").mkdir()
        monkeypatch.setenv("PAM_MODELS_PATH", str(tmp_path))
        monkeypatch.setattr(
            SglangInference, "_setup_deepgemm_cache_dir", lambda self, nfs, local: None
        )
        _setup_manager(
            router_config=RouterConfig(port=30000, prometheus_port=29000),
            globals_config=GlobalsConfig(record_generations_dir="/tmp/generations"),
        )
        engines = [InferenceManager.setup_inference_engine()] + [
            InferenceManager.setup_preset_inference_engine(
                model_name="gpt-oss-120b", judge_index=judge_index
            )
            for judge_index in range(3)
        ]

        ports = [
            (
                engine.router_port,
                _port(engine.worker_urls[0]),
                engine.router_config.prometheus_port,
                _port(engine.direct_base_url),
            )
            for engine in engines
        ]

        assert ports == [
            (30000, 30001, 29000, 35000),
            (30002, 30003, 29001, 35002),
            (30004, 30005, 29002, 35004),
            (30006, 30007, 29003, 35006),
        ]
        every_port = [port for engine_ports in ports for port in engine_ports]
        assert len(set(every_port)) == len(every_port)

    @pytest.mark.parametrize(
        ("judge_index", "expected_port"), [(0, 30002), (1, 30004), (2, 30006)]
    )
    def test_an_external_judge_serves_on_its_judge_port_without_weights(
        self, monkeypatch: pytest.MonkeyPatch, judge_index: int, expected_port: int
    ):
        monkeypatch.delenv("PAM_MODELS_PATH", raising=False)
        _setup_manager(secondary_inference={"openrouter": {"api_key": "sk-or-test"}})

        engine = InferenceManager.setup_preset_inference_engine(
            model_name="openrouter:qwen/qwen3", judge_index=judge_index
        )

        assert isinstance(engine, ExternalApiEngine)
        assert engine.base_url == f"http://127.0.0.1:{expected_port}"
        assert engine.inference_config.model == "qwen/qwen3"
        assert engine.log_profile == "openrouter:qwen/qwen3"
        assert engine.obm_proxy_manager is None

    def test_a_replayed_judge_answers_on_its_judge_port(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("PAM_MODELS_PATH", raising=False)
        _setup_manager(
            globals_config=GlobalsConfig(replay_generations_dir="/tmp/generations")
        )

        engine = InferenceManager.setup_preset_inference_engine(
            model_name="deepseek-v3", judge_index=1
        )

        assert isinstance(engine, DummyEngine)
        assert engine.base_url == "http://127.0.0.1:30004"
