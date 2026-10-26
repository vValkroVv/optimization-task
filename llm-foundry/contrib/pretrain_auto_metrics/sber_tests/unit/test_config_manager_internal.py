import base64
import os
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from node_runner.src.config_manager import ConfigManager
from node_runner.src.config_model import (
    DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE,
    BaseObmEvaluatorConfig,
    ConvertationConfig,
    DevicesObmEvaluatorConfig,
    DevicesV2ObmEvaluatorConfig,
    EvaluatorsConfig,
    GlobalsConfig,
    LmEvalEvaluatorConfig,
    LmEvalInstructTaskConfig,
    LmEvalPretrainTaskConfig,
    LmmsEvalBenchmarksConfig,
    LmmsEvalEvaluatorConfig,
    LmmsEvalRunConfig,
    MeraEvaluatorConfig,
    MeraTaskConfig,
    NodeRunnerConfigModel,
    ObmAllConfig,
    ObmProxyConfig,
    OpenaiObmEvaluatorConfig,
    RouterConfig,
    SglangInferenceConfig,
)
from sber_tests.test_support import set_env, setup_tmp_checkpoint

pytestmark = pytest.mark.unit


class TestConfigManagerInternals:
    def test_get_environment_info_unsupported(self, monkeypatch: pytest.MonkeyPatch):
        manager = ConfigManager()
        manager.config = type("_Config", (), {"runs_without_checkpoint": False})()
        monkeypatch.setattr(manager, "_is_cloud_env", lambda: False)
        monkeypatch.setattr(manager, "_is_slurm_env", lambda: False)
        with pytest.raises(ValueError, match="Environment not supported"):
            manager._get_environment_info()

    def test_infer_cloud_rank_worker(self):
        assert (
            ConfigManager._infer_cloud_rank("mpiworker-3", ["x", "mpiworker", "3"]) == 4
        )

    def test_setup_calls_pipeline_in_expected_order(self, tmp_path: Path):
        manager = ConfigManager()
        checkpoint_path = tmp_path / "checkpoint"
        calls: list[str] = []
        captured = {"set_paths_arg": None, "load_config_arg": None}

        def _set_paths(cp):
            captured["set_paths_arg"] = cp
            calls.append("set_paths")

        def _load_config(config_path, checkpoint_path):
            captured["load_config_arg"] = (config_path, checkpoint_path)
            calls.append("load_config")

        manager._set_paths = _set_paths
        manager._load_config = _load_config
        manager._get_environment_info = lambda: calls.append("get_environment_info")
        manager._decode_jinja_to_ram = lambda: calls.append("decode_jinja_to_ram")
        manager._decode_default_devsystem_kwarg = lambda: calls.append(
            "decode_default_devsystem_kwarg"
        )

        manager.setup(checkpoint_path)
        assert calls == [
            "load_config",
            "set_paths",
            "get_environment_info",
            "decode_jinja_to_ram",
            "decode_default_devsystem_kwarg",
        ]
        assert captured["set_paths_arg"] == checkpoint_path
        assert captured["load_config_arg"] == (None, checkpoint_path)


class TestPydanticConfigModels:
    # -- InferenceConfig -------------------------------------------------------

    def test_inference_config_alias_priority_and_defaults(self):
        cfg = SglangInferenceConfig.model_validate(
            {"tp": 2, "tp_size": 4, "pp-size": 3, "ep_size": 5}
        )
        assert cfg.tp == 2
        assert cfg.pp == 3
        assert cfg.ep == 5
        assert cfg.dp == 1
        assert cfg.processor_internal_path == "processors/GigaVisionProcessor"

    def test_inference_config_preserves_existing_fields(self):
        cfg = SglangInferenceConfig(
            dtype="fp16",
            mem_fraction_static=0.9,
            trust_remote_code=False,
            allow_auto_truncate=False,
            max_running_requests=128,
        )
        assert cfg.engine_type == "sglang"
        assert cfg.dtype == "fp16"
        assert cfg.mem_fraction_static == 0.9
        assert cfg.trust_remote_code is False
        assert cfg.allow_auto_truncate is False
        assert cfg.max_running_requests == 128

    # -- GlobalsConfig ---------------------------------------------------------

    def test_globals_config_uses_defaults_and_respects_overrides(self):
        defaults = GlobalsConfig()
        assert defaults.dist_init_path == "/etc/volcano/mpimaster.host"
        assert defaults.master_num_file_path == "/etc/volcano/VC_MPIMASTER_NUM"
        assert defaults.worker_num_file_path == "/etc/volcano/VC_MPIWORKER_NUM"
        assert defaults.orkestration_port == 61574
        assert defaults.delete_converted_hf is False

        custom = GlobalsConfig(
            dist_init_path="/tmp/custom_dist",
            master_num_file_path="/tmp/custom_master",
            worker_num_file_path="/tmp/custom_worker",
            orkestration_port=62000,
            delete_converted_hf=True,
        )
        assert custom.dist_init_path == "/tmp/custom_dist"
        assert custom.master_num_file_path == "/tmp/custom_master"
        assert custom.worker_num_file_path == "/tmp/custom_worker"
        assert custom.orkestration_port == 62000
        assert custom.delete_converted_hf is True

    def test_globals_config_rejects_unknown_keys(self):
        # sglang_communication_port moved to autometrics.inference; a stale
        # key in globals must fail validation instead of being silently ignored.
        with pytest.raises(ValidationError):
            GlobalsConfig(sglang_communication_port=61000)

    # -- RouterConfig / InferenceConfig chat-template --------------------------

    def test_router_config_defaults_and_inference_chat_template(self):
        router = RouterConfig()
        assert router.host == "0.0.0.0"

        inference = SglangInferenceConfig.model_validate(
            {"chat-template": "/nfs/old_template.jinja"}
        )
        assert inference.chat_template == "/nfs/old_template.jinja"

    @pytest.mark.parametrize(
        ("raw", "policy"),
        [({}, "round_robin"), ({"policy": "cache_aware"}, "cache_aware")],
    )
    def test_router_policy_defaults_to_round_robin_and_queue_waits_as_requests(
        self, raw, policy
    ):
        args = RouterConfig.model_validate(raw).to_bash_list()

        passed = args[args.index("--policy") + 1] if "--policy" in args else None
        assert passed == policy
        assert (
            args[args.index("--queue-timeout-secs") + 1]
            == args[args.index("--request-timeout-secs") + 1]
        )

    # -- ConvertationConfig ----------------------------------------------------

    def test_convertation_config_defaults(self):
        cfg = ConvertationConfig()
        assert cfg.output_tp_size == 1
        assert cfg.skip_optimizer is False
        assert cfg.print_final_model_structure is True
        assert cfg.chunk_layers == 5
        assert cfg.use_ema is True

    def test_convertation_config_preserves_overrides(self):
        cfg = ConvertationConfig(
            output_tp_size=8,
            skip_optimizer=True,
            print_final_model_structure=False,
            chunk_layers=9,
            use_ema=False,
        )
        assert cfg.output_tp_size == 8
        assert cfg.skip_optimizer is True
        assert cfg.print_final_model_structure is False
        assert cfg.chunk_layers == 9
        assert cfg.use_ema is False

    # -- LmEvalEvaluatorConfig -------------------------------------------------

    def test_lm_eval_apply_defaults_fills_model_args(self):
        cfg = LmEvalEvaluatorConfig(
            pretrain={
                "longbench": LmEvalPretrainTaskConfig(),
                "bbh": LmEvalPretrainTaskConfig(model_args={"base_url": "http://x"}),
            }
        )
        cfg.apply_defaults(
            inference_instances=7,
            max_seq_len=4096,
            hf_checkpoint_path=Path("/tmp/tokenizer"),
            base_url="http://127.0.0.1:29999",
        )
        assert cfg.pretrain["longbench"].batch_size == 1
        assert cfg.pretrain["bbh"].batch_size == 1
        assert (
            cfg.pretrain["bbh"].model_args["num_concurrent"]
            == 7 * DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE
        )
        assert cfg.pretrain["bbh"].model_args["tokenized_requests"] is True
        assert cfg.pretrain["bbh"].model_args["max_retries"] == 20

    def test_lm_eval_apply_defaults_empty_config(self):
        cfg = LmEvalEvaluatorConfig()
        cfg.apply_defaults(
            inference_instances=4,
            max_seq_len=4096,
            hf_checkpoint_path=Path("/tmp/tok"),
            base_url="http://127.0.0.1:29999",
        )
        assert cfg.pretrain == {}
        assert cfg.instruct == {}

    def test_lm_eval_apply_defaults_fills_base_url(self):
        cfg = LmEvalEvaluatorConfig(
            pretrain={"bbh": LmEvalPretrainTaskConfig(model_args={})}
        )
        cfg.apply_defaults(
            inference_instances=4,
            max_seq_len=4096,
            hf_checkpoint_path=Path("/tmp/tok"),
            base_url="http://127.0.0.1:29999",
        )
        assert (
            cfg.pretrain["bbh"].model_args["base_url"]
            == "http://127.0.0.1:29999/generate"
        )
        assert (
            cfg.pretrain["bbh"].model_args["num_concurrent"]
            == 4 * DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE
        )

    def test_lm_eval_apply_defaults_preserves_existing_model_args(self):
        cfg = LmEvalEvaluatorConfig(
            pretrain={
                "bbh": LmEvalPretrainTaskConfig(
                    model_args={
                        "base_url": "http://x",
                        "num_concurrent": 3,
                        "tokenized_requests": False,
                        "max_length": 2048,
                        "timeout": 10,
                        "max_retries": 2,
                    }
                ),
            }
        )
        cfg.apply_defaults(
            inference_instances=6,
            max_seq_len=4096,
            hf_checkpoint_path=Path("/tmp/tok"),
            base_url="http://127.0.0.1:29999",
        )
        assert cfg.pretrain["bbh"].model_args["base_url"] == "http://x"
        assert cfg.pretrain["bbh"].model_args["num_concurrent"] == 3
        assert cfg.pretrain["bbh"].model_args["tokenized_requests"] is False
        assert cfg.pretrain["bbh"].model_args["timeout"] == 10
        assert cfg.pretrain["bbh"].model_args["max_retries"] == 2

    def test_lm_eval_apply_defaults_sets_lcb_process_results_workers(self):
        cfg = LmEvalEvaluatorConfig(
            pretrain={
                "lcb_code_gen_lite": LmEvalPretrainTaskConfig(),
                "bbh": LmEvalPretrainTaskConfig(),
            },
            instruct={
                "lcb_code_gen_lite_instruct": LmEvalInstructTaskConfig(),
                "chat_task": LmEvalInstructTaskConfig(process_results_workers=8),
            },
        )
        cfg.apply_defaults(
            inference_instances=4,
            max_seq_len=4096,
            hf_checkpoint_path=Path("/tmp/tok"),
            base_url="http://127.0.0.1:29999",
        )
        assert cfg.pretrain["lcb_code_gen_lite"].process_results_workers == 16
        assert cfg.instruct["lcb_code_gen_lite_instruct"].process_results_workers == 16
        assert cfg.pretrain["bbh"].process_results_workers == 1
        assert cfg.instruct["chat_task"].process_results_workers == 8

    def test_lm_eval_apply_defaults_overrides_explicit_lcb_process_results_workers(
        self,
    ):
        cfg = LmEvalEvaluatorConfig(
            pretrain={
                "lcb_code_gen_lite": LmEvalPretrainTaskConfig(process_results_workers=8)
            },
            instruct={
                "lcb_code_gen_lite_instruct": LmEvalInstructTaskConfig(
                    process_results_workers=1
                )
            },
        )
        cfg.apply_defaults(
            inference_instances=4,
            max_seq_len=4096,
            hf_checkpoint_path=Path("/tmp/tok"),
            base_url="http://127.0.0.1:29999",
        )
        assert cfg.pretrain["lcb_code_gen_lite"].process_results_workers == 16
        assert cfg.instruct["lcb_code_gen_lite_instruct"].process_results_workers == 16

    def test_lm_eval_apply_defaults_sets_lcb_pass_at_k_filter_workers(self):
        cfg = LmEvalEvaluatorConfig(
            pretrain={
                "lcb_code_gen_lite_32": LmEvalPretrainTaskConfig(),
                "lcb_code_gen_lite": LmEvalPretrainTaskConfig(),
                "cruxeval_32": LmEvalPretrainTaskConfig(),
                "spellcheck_punctuation_ru_32": LmEvalPretrainTaskConfig(),
            },
            instruct={
                "lcb_code_gen_lite_instruct_32": LmEvalInstructTaskConfig(
                    filter_workers=8
                ),
                "chat_task": LmEvalInstructTaskConfig(),
            },
        )
        cfg.apply_defaults(
            inference_instances=4,
            max_seq_len=4096,
            hf_checkpoint_path=Path("/tmp/tok"),
            base_url="http://127.0.0.1:29999",
        )
        assert cfg.pretrain["lcb_code_gen_lite_32"].filter_workers == 16
        assert cfg.pretrain["cruxeval_32"].filter_workers == 16
        assert cfg.pretrain["spellcheck_punctuation_ru_32"].filter_workers == 16
        assert cfg.instruct["lcb_code_gen_lite_instruct_32"].filter_workers == 16
        assert cfg.pretrain["lcb_code_gen_lite"].filter_workers == 1
        assert cfg.instruct["chat_task"].filter_workers == 1

    def test_lm_eval_apply_defaults_uses_custom_api_port_for_base_urls(self):
        cfg = LmEvalEvaluatorConfig(
            pretrain={"bbh": LmEvalPretrainTaskConfig(model_args={})},
            instruct={"chat_task": LmEvalInstructTaskConfig(model_args={})},
        )
        cfg.apply_defaults(
            inference_instances=4,
            max_seq_len=4096,
            hf_checkpoint_path=Path("/tmp/tok"),
            base_url="http://127.0.0.1:31234",
        )
        assert (
            cfg.pretrain["bbh"].model_args["base_url"]
            == "http://127.0.0.1:31234/generate"
        )
        assert (
            cfg.instruct["chat_task"].model_args["base_url"]
            == "http://127.0.0.1:31234/v1/chat/completions"
        )

    def test_lm_eval_apply_defaults_keeps_explicit_batch_size(self):
        cfg = LmEvalEvaluatorConfig(
            pretrain={"bbh": LmEvalPretrainTaskConfig(batch_size=13)}
        )
        cfg.apply_defaults(
            inference_instances=6,
            max_seq_len=4096,
            hf_checkpoint_path=Path("/tmp/tok"),
            base_url="http://127.0.0.1:29999",
        )
        assert cfg.pretrain["bbh"].batch_size == 13

    def test_lmms_eval_apply_defaults_keeps_gigavision_defaults_external(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("CLUSTER_KEY", "SR004")
        monkeypatch.delenv("HF_HOME", raising=False)
        monkeypatch.delenv("TMPDIR", raising=False)
        monkeypatch.delenv("NLTK_DATA", raising=False)

        cfg = LmmsEvalEvaluatorConfig.model_validate(
            {
                "chatml_version": "v4.1",
                "benchmarks": {"tasks": ["chartqa_giga"]},
                "runs": {
                    "all": {
                        "benchmarks": {
                            "categories": ["all"],
                        },
                    },
                },
            }
        )
        cfg.apply_defaults(
            inference_instances=2,
            base_url="http://127.0.0.1:31234",
            direct_base_url="http://127.0.0.1:30000",
            inference_config=SglangInferenceConfig(),
        )

        assert cfg.chatml_version == "v4.1"
        assert cfg.benchmarks is not None
        assert cfg.benchmarks.tasks == ["chartqa_giga"]
        assert cfg.runs["all"].benchmarks is not None
        assert cfg.runs["all"].benchmarks.categories == ["all"]
        target = cfg.require_runtime_target()
        assert target.model_source == "api"
        assert target.model_overrides["api"]["obm_proxy"]["api"]["base_url"] == (
            "http://127.0.0.1:31234"
        )
        assert target.model_overrides["api"]["obm_proxy"]["parallel"] == 100
        assert "HF_HOME" not in os.environ
        assert "TMPDIR" not in os.environ
        assert "NLTK_DATA" not in os.environ

    def test_lmms_eval_chatml_version_is_runtime_dependent(self):
        assert LmmsEvalEvaluatorConfig.model_validate({}).chatml_version is None

    def test_lmms_eval_rejects_legacy_sglang_schema(self):
        with pytest.raises(ValueError):
            LmmsEvalEvaluatorConfig.model_validate(
                {
                    "chatml_version": "v4.1",
                    "tasks": {
                        "chartqa_giga": {},
                    },
                }
            )

    def test_lmms_eval_apply_defaults_unknown_cluster_keeps_process_env_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("CLUSTER_KEY", "UNKNOWN")
        monkeypatch.delenv("HF_HOME", raising=False)
        monkeypatch.delenv("TMPDIR", raising=False)
        monkeypatch.delenv("NLTK_DATA", raising=False)

        cfg = LmmsEvalEvaluatorConfig.model_validate({"chatml_version": "v4.1"})
        cfg.apply_defaults(
            inference_instances=2,
            base_url="http://127.0.0.1:31234",
            direct_base_url="http://127.0.0.1:30000",
            inference_config=SglangInferenceConfig(),
        )

        assert "HF_HOME" not in os.environ
        assert "TMPDIR" not in os.environ
        assert "NLTK_DATA" not in os.environ

    def test_lmms_eval_resolved_runs_uses_default_when_runs_missing(self):
        cfg = LmmsEvalEvaluatorConfig(
            chatml_version="v4.1",
            benchmarks=LmmsEvalBenchmarksConfig(tasks=["chartqa_giga"]),
        )
        assert list(cfg.resolved_runs) == ["default"]
        assert cfg.resolved_runs["default"].benchmarks is not None
        assert cfg.resolved_runs["default"].benchmarks.tasks == ["chartqa_giga"]

    def test_lmms_eval_resolved_runs_preserves_explicit_runs(self):
        cfg = LmmsEvalEvaluatorConfig(
            chatml_version="v4.1",
            benchmarks=LmmsEvalBenchmarksConfig(tasks=["chartqa_giga"]),
            runs={
                "all": LmmsEvalRunConfig(
                    benchmarks=LmmsEvalBenchmarksConfig(categories=["all"]),
                ),
            },
        )
        assert list(cfg.resolved_runs) == ["all"]
        assert cfg.resolved_runs["all"].benchmarks is not None
        assert cfg.resolved_runs["all"].benchmarks.categories == ["all"]

    # -- MeraEvaluatorConfig ---------------------------------------------------

    def test_mera_apply_defaults_fills_model_args(self):
        cfg = MeraEvaluatorConfig(
            tasks={
                "simplear": MeraTaskConfig(),
                "rummlu_gen": MeraTaskConfig(model_args={"base_url": "http://x"}),
            }
        )
        cfg.apply_defaults(
            inference_instances=5,
            base_url="http://127.0.0.1:29999",
        )
        assert cfg.tasks["simplear"].batch_size == 1
        assert (
            cfg.tasks["rummlu_gen"].model_args["num_concurrent"]
            == 5 * DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE
        )
        assert cfg.tasks["rummlu_gen"].model_args["timeout"] == 21600
        assert cfg.tasks["rummlu_gen"].model_args["max_retries"] == 20

    def test_mera_apply_defaults_fills_base_url(self):
        cfg = MeraEvaluatorConfig(tasks={"simplear": MeraTaskConfig(model_args={})})
        cfg.apply_defaults(
            inference_instances=4,
            base_url="http://127.0.0.1:29999",
        )
        assert (
            cfg.tasks["simplear"].model_args["base_url"]
            == "http://127.0.0.1:29999/v1/chat/completions"
        )

    def test_mera_apply_defaults_uses_custom_api_port_for_base_url(self):
        cfg = MeraEvaluatorConfig(tasks={"simplear": MeraTaskConfig(model_args={})})
        cfg.apply_defaults(
            inference_instances=4,
            base_url="http://127.0.0.1:31234",
        )
        assert (
            cfg.tasks["simplear"].model_args["base_url"]
            == "http://127.0.0.1:31234/v1/chat/completions"
        )

    def test_mera_apply_defaults_preserves_existing_fields(self):
        cfg = MeraEvaluatorConfig(
            tasks={
                "simplear": MeraTaskConfig(
                    batch_size=8,
                    model_args={
                        "base_url": "http://x",
                        "num_concurrent": 9,
                        "timeout": 77,
                    },
                ),
                "bps_gen": MeraTaskConfig(
                    batch_size=4,
                    model_args={
                        "base_url": "http://y",
                        "num_concurrent": 5,
                        "timeout": 88,
                    },
                ),
            }
        )
        cfg.apply_defaults(
            inference_instances=4,
            base_url="http://127.0.0.1:29999",
        )
        assert cfg.tasks["simplear"].batch_size == 8
        assert cfg.tasks["simplear"].model_args["num_concurrent"] == 9
        assert cfg.tasks["simplear"].model_args["timeout"] == 77
        assert cfg.tasks["bps_gen"].batch_size == 4
        assert cfg.tasks["bps_gen"].model_args["num_concurrent"] == 5
        assert cfg.tasks["bps_gen"].model_args["timeout"] == 88

    # -- ObmEvaluatorConfig ----------------------------------------------------

    def test_obm_evaluator_config_invalid_type_raises(self):
        with pytest.raises(ValueError):
            DevicesObmEvaluatorConfig(
                obm_config="bad", all=ObmAllConfig(groups=["open"])
            )

    def test_obm_openai_config_shooter(self):
        cfg = OpenaiObmEvaluatorConfig(all=ObmAllConfig(groups=["open"]), n_threads=8)
        shooter = cfg.shooter_config
        assert "openai" in shooter
        assert shooter["openai"]["api"]["base_url"] == "None/v1"
        assert shooter["openai"]["keys"] == ["empty"]
        assert shooter["openai"]["model"] == "<model>"
        assert shooter["openai"]["limit"]["rpm"] == 1_000_000_000

    def test_obm_devices_config_shooter(self):
        cfg = DevicesObmEvaluatorConfig(all=ObmAllConfig(groups=["open"]), n_threads=8)
        shooter = cfg.shooter_config
        assert "devices" in shooter
        assert shooter["devices"]["api"]["base_url"] is None
        assert shooter["devices"]["api"]["route_chat"] == "/v1/chat/completions"

    def test_obm_devices_v2_config_shooter(self):
        cfg = DevicesV2ObmEvaluatorConfig(
            all=ObmAllConfig(groups=["open"]), n_threads=8
        )
        shooter = cfg.shooter_config
        assert "devices" in shooter
        assert shooter["devices"]["api"]["base_url"] is None
        assert shooter["devices"]["api"]["route_chat"] == "/v2/chat/completions"
        assert shooter["devices"]["api"]["version"] == "v2"

    def test_obm_shooter_uses_custom_api_port_after_apply_defaults(self):
        openai_cfg = OpenaiObmEvaluatorConfig(
            all=ObmAllConfig(groups=["open"]), n_threads=8
        )
        devices_cfg = DevicesObmEvaluatorConfig(
            all=ObmAllConfig(groups=["open"]), n_threads=8
        )

        openai_cfg.apply_defaults(
            inference_instances=1,
            base_url="http://127.0.0.1:31234",
        )
        devices_cfg.apply_defaults(
            inference_instances=1,
            base_url="http://127.0.0.1:31234",
        )

        assert openai_cfg.shooter_config["openai"]["api"]["base_url"] == (
            "http://127.0.0.1:31234/v1"
        )
        assert devices_cfg.shooter_config["devices"]["api"]["base_url"] == (
            "http://127.0.0.1:31234"
        )

    def test_obm_all_config_default_n_workers(self):
        cfg = DevicesObmEvaluatorConfig(all=ObmAllConfig(groups=["open"]), n_threads=8)
        assert cfg.all.n_workers == 10

    def test_obm_all_config_custom_n_workers(self):
        cfg = DevicesObmEvaluatorConfig(
            all=ObmAllConfig(groups=["open"], n_workers=11), n_threads=8
        )
        assert cfg.all.n_workers == 11
        assert cfg.all.groups == ["open"]

    def test_obm_evaluator_apply_defaults_sets_n_threads(self):
        cfg = DevicesObmEvaluatorConfig(all=ObmAllConfig(groups=["open"]))
        cfg.apply_defaults(
            inference_instances=4,
            base_url="http://127.0.0.1:29999",
        )
        assert cfg.n_threads == 4 * DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE

    def test_obm_evaluator_apply_defaults_keeps_explicit_n_threads(self):
        cfg = DevicesObmEvaluatorConfig(all=ObmAllConfig(groups=["open"]), n_threads=9)
        cfg.apply_defaults(
            inference_instances=4,
            base_url="http://127.0.0.1:29999",
        )
        assert cfg.n_threads == 9

    # -- ObmProxyConfig --------------------------------------------------------

    def test_obm_proxy_config_defaults(self):
        cfg = ObmProxyConfig()
        assert cfg.sglang_port == 30000
        assert cfg.api_port == 29999
        assert cfg.workers == 16
        assert cfg.reasoning_token_budget == 0
        assert cfg.chat_template is None
        assert cfg.chat_template_kwargs == {}
        assert cfg.tokenizer_path is None

    def test_obm_proxy_config_preserves_overrides(self):
        cfg = ObmProxyConfig(
            workers=4,
            sglang_port=31000,
            api_port=32000,
            chat_handlers_mode="custom",
            chat_template="/tmp/custom.j2",
            chat_template_kwargs={"use_devsystem": False},
            reasoning_token_budget=1,
            tokenizer_path="/tmp/tokenizer",
        )
        assert cfg.workers == 4
        assert cfg.sglang_port == 31000
        assert cfg.api_port == 32000
        assert cfg.chat_handlers_mode == "custom"
        assert cfg.chat_template == "/tmp/custom.j2"
        assert cfg.chat_template_kwargs == {"use_devsystem": False}
        assert cfg.reasoning_token_budget == 1
        assert cfg.tokenizer_path == "/tmp/tokenizer"

    # -- ObmProxyConfig decode methods -----------------------------------------

    def test_decode_chat_template_slurm_and_local(self, tmp_path: Path):
        src_template = tmp_path / "template.j2"
        src_template.write_text(base64.b64encode(b"{{ x }}").decode("utf-8"))

        obm_proxy = ObmProxyConfig(chat_template=str(src_template))
        target_path = tmp_path / "chat_template.jinja"
        obm_proxy.decode_chat_template_to_ram(is_slurm=True, ram_path=str(target_path))
        assert target_path.read_text() == "{{ x }}"
        assert obm_proxy.chat_template == target_path.absolute().as_posix()

        obm_proxy2 = ObmProxyConfig(chat_template=str(src_template))
        obm_proxy2.decode_chat_template_to_ram(is_slurm=False)
        assert obm_proxy2.chat_template == src_template.absolute().as_posix()

    @pytest.mark.parametrize("raw_value", ["not_base64", "abcd"])
    def test_decode_chat_template_invalid_base64_raises(
        self, tmp_path: Path, raw_value: str
    ):
        src_template = tmp_path / "template.j2"
        src_template.write_text(raw_value)
        obm_proxy = ObmProxyConfig(chat_template=str(src_template))
        with pytest.raises(ValueError, match="Chat template is not encoded in base64"):
            obm_proxy.decode_chat_template_to_ram(
                is_slurm=True, ram_path=str(tmp_path / "out.jinja")
            )

    def test_check_if_encoded_and_decode(self):
        decode = ObmProxyConfig._check_if_encoded_and_decode
        assert decode("aGVsbG8=") == b"hello"
        assert decode("aGVsbG8=\n") == b"hello"
        assert decode(b"aGVsbG8=") == b"hello"
        assert decode("not_base64") is None
        assert decode(b"not_base64") is None
        assert decode("abcd") is None

    @pytest.mark.parametrize("kwargs", [None, {}])
    def test_decode_default_devsystem_noop_branches(self, kwargs: dict | None):
        obm_proxy = ObmProxyConfig(chat_template_kwargs=kwargs or {})
        obm_proxy.decode_default_devsystem(is_slurm=True)
        assert obm_proxy.chat_template_kwargs == (kwargs or {})

    @pytest.mark.parametrize(
        ("raw_value", "should_raise", "expected_value"),
        [
            (
                base64.b64encode(b"devsystem").decode("utf-8"),
                False,
                "devsystem",
            ),
            ("not_base64", True, None),
            ("abcd", True, None),
        ],
    )
    def test_decode_default_devsystem_slurm(
        self,
        raw_value: str,
        should_raise: bool,
        expected_value: str | None,
    ):
        obm_proxy = ObmProxyConfig(
            chat_template_kwargs={"default_devsystem": raw_value}
        )
        if should_raise:
            with pytest.raises(
                ValueError,
                match="Default devsystem is not encoded in base64",
            ):
                obm_proxy.decode_default_devsystem(is_slurm=True)
            return
        obm_proxy.decode_default_devsystem(is_slurm=True)
        assert obm_proxy.chat_template_kwargs["default_devsystem"] == expected_value

    def test_decode_default_devsystem_non_slurm_keeps_value(self):
        obm_proxy = ObmProxyConfig(chat_template_kwargs={"default_devsystem": "plain"})
        obm_proxy.decode_default_devsystem(is_slurm=False)
        assert obm_proxy.chat_template_kwargs["default_devsystem"] == "plain"


def _slurm_env() -> dict[str, str]:
    return {
        "SLURM_JOB_NUM_NODES": "2",
        "SLURM_LAUNCH_NODE_IPADDR": "127.0.0.1",
        "SLURM_PROCID": "0",
        "SLURM_JOB_ID": "job-1",
        "JS_GPUS_PER_NODE": "8",
        "JD_SYNC_DIR": "/tmp/jd-sync",
    }


def _base_node_runner_config() -> dict:
    return {
        "run_name": "run_name",
        "tensorboard_path": "/tmp/tb",
        "max_seq_len": 4096,
        "autometrics": {
            "globals": {
                "dist_init_path": "/tmp/non-existent-dist-init",
            },
            "router": {},
            "inference": {
                "tp": 1,
                "pp": 1,
                "ep": 1,
                "dp": 1,
            },
            "obm_proxy": {},
            "evaluators": {},
            "preparation": {},
            "convertation": {
                "mappers": ["map_fsdp_to_hf"],
            },
            "model_info": {
                "model_class": "sft",
                "size": "lite",
                "username": "user",
            },
        },
    }


def _dump_config(config_path: Path, config: dict) -> None:
    with open(config_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file)


def _load_dumped_model(tmp_path: Path, config: dict) -> NodeRunnerConfigModel:
    config_path = tmp_path / "pam_config.yaml"
    _dump_config(config_path, config)
    return NodeRunnerConfigModel.load(config_path)


def _apply_evaluator_defaults(
    model: NodeRunnerConfigModel,
    *,
    inference_instances: int,
    hf_checkpoint_path: Path,
) -> None:
    base_url = f"http://127.0.0.1:{model.autometrics.obm_proxy.api_port}"
    direct_base_url = f"http://127.0.0.1:{model.autometrics.router.port}"
    for _, evaluator_config in model.autometrics.evaluators:
        evaluator_config.apply_defaults(
            inference_instances=inference_instances,
            base_url=base_url,
            direct_base_url=direct_base_url,
            inference_config=model.autometrics.inference,
            max_seq_len=model.max_seq_len,
            hf_checkpoint_path=hf_checkpoint_path,
        )


class TestConfigManagerAndModelFromDumpedYaml:
    @pytest.mark.parametrize(
        ("inference_patch", "obm_proxy_patch", "expected_mode"),
        [
            ({"reasoning-parser": "r1"}, {}, "opensource"),
            ({}, {"chat-template": "/tmp/missing-template.jinja"}, "custom"),
            ({}, {}, "pretrain"),
        ],
    )
    def test_config_manager_setup_sets_chat_handlers_mode_from_loaded_yaml(
        self,
        tmp_path: Path,
        inference_patch: dict,
        obm_proxy_patch: dict,
        expected_mode: str,
    ):
        checkpoint_path, _artifacts_path, config_path = setup_tmp_checkpoint(tmp_path)
        config = _base_node_runner_config()
        config["autometrics"]["inference"].update(inference_patch)
        config["autometrics"]["obm_proxy"].update(obm_proxy_patch)
        _dump_config(config_path, config)

        manager = ConfigManager()
        with set_env(**_slurm_env()):
            manager.setup(checkpoint_path)

        assert manager.config is not None
        assert manager.config.autometrics.obm_proxy.chat_handlers_mode == expected_mode

    def test_quantization_skip_is_opt_in_and_not_dumped(self, tmp_path: Path):
        config = _base_node_runner_config()
        config["autometrics"]["quantization"] = {
            "device_map": "auto",
            "quantization_config": {
                "weight_block_size": [128, 128],
                "modules_to_not_convert": [],
            },
        }

        model = _load_dumped_model(tmp_path, config)

        assert model.autometrics.quantization is not None
        assert model.autometrics.quantization.skip is None
        quantization_dump = model.autometrics.quantization.model_dump(exclude_none=True)
        assert "skip" not in quantization_dump
        assert quantization_dump["device_map"] == "auto"
        assert quantization_dump["quantization_config"]["weight_block_size"] == [
            128,
            128,
        ]

    @pytest.mark.parametrize(
        "method_name",
        ["_decode_jinja_to_ram", "_decode_default_devsystem_kwarg"],
    )
    def test_decode_methods_noop_without_loaded_config(self, method_name: str):
        manager = ConfigManager()
        getattr(manager, method_name)()
        assert manager.config is None

    @pytest.mark.parametrize(
        ("inference_patch", "obm_proxy_patch", "error_match"),
        [
            (
                {"reasoning-parser": "r1"},
                {"chat-template": "/tmp/custom.jinja"},
                "Conflicting chat formatting configuration",
            ),
            (
                {},
                {"chat-template-kwargs": {"default_devsystem": "abc"}},
                "chat_template_kwargs were provided without a chat formatting source",
            ),
            (
                {},
                {},
                "lm_eval.instruct requires chat formatting",
            ),
            (
                {},
                {},
                "harbor_rli is an agentic, tool-calling",
            ),
        ],
    )
    def test_node_runner_config_load_rejects_invalid_chat_formatting_sources(
        self,
        tmp_path: Path,
        inference_patch: dict,
        obm_proxy_patch: dict,
        error_match: str,
    ):
        config = _base_node_runner_config()
        config["autometrics"]["inference"].update(inference_patch)
        config["autometrics"]["obm_proxy"].update(obm_proxy_patch)
        if "lm_eval.instruct requires chat formatting" in error_match:
            config["autometrics"]["evaluators"] = {
                "lm_eval": {
                    "instruct": {
                        "chat_task": {"model_args": {}},
                    }
                }
            }
        if "harbor_rli is an agentic" in error_match:
            config["autometrics"]["evaluators"] = {
                "harbor_rli": {"benchmark": "harness-bench-lite"},
            }

        config_path = tmp_path / "invalid_config.yaml"
        _dump_config(config_path, config)

        with pytest.raises(ValueError, match=error_match):
            NodeRunnerConfigModel.load(config_path)

    def test_node_runner_config_load_syncs_obm_proxy_tokenizer_path_to_inference(
        self, tmp_path: Path
    ):
        config = _base_node_runner_config()
        config["autometrics"]["obm_proxy"]["tokenizer-path"] = "/tmp/tokenizer"

        model = _load_dumped_model(tmp_path, config)

        assert model.autometrics.obm_proxy.tokenizer_path == "/tmp/tokenizer"
        assert model.autometrics.inference.tokenizer_path == "/tmp/tokenizer"

    def test_node_runner_config_load_syncs_inference_tokenizer_path_to_obm_proxy(
        self, tmp_path: Path
    ):
        config = _base_node_runner_config()
        config["autometrics"]["inference"]["tokenizer-path"] = "/tmp/tokenizer"

        model = _load_dumped_model(tmp_path, config)

        assert model.autometrics.inference.tokenizer_path == "/tmp/tokenizer"
        assert model.autometrics.obm_proxy.tokenizer_path == "/tmp/tokenizer"

    def test_node_runner_config_load_keeps_equal_tokenizer_paths(self, tmp_path: Path):
        config = _base_node_runner_config()
        config["autometrics"]["inference"]["tokenizer-path"] = "/tmp/tokenizer"
        config["autometrics"]["obm_proxy"]["tokenizer-path"] = "/tmp/tokenizer"

        model = _load_dumped_model(tmp_path, config)

        assert model.autometrics.obm_proxy.tokenizer_path == "/tmp/tokenizer"
        assert model.autometrics.inference.tokenizer_path == "/tmp/tokenizer"

    def test_node_runner_config_load_rejects_conflicting_tokenizer_paths(
        self, tmp_path: Path
    ):
        config = _base_node_runner_config()
        config["autometrics"]["inference"]["tokenizer-path"] = (
            "/tmp/inference-tokenizer"
        )
        config["autometrics"]["obm_proxy"]["tokenizer-path"] = "/tmp/obm-tokenizer"
        config_path = tmp_path / "invalid_tokenizer_config.yaml"
        _dump_config(config_path, config)

        with pytest.raises(ValueError, match="Conflicting tokenizer configuration"):
            NodeRunnerConfigModel.load(config_path)

    def test_apply_defaults_updates_all_evaluators_for_loaded_yaml(
        self, tmp_path: Path
    ):
        config = _base_node_runner_config()
        config["autometrics"]["obm_proxy"]["api-port"] = 31234
        config["autometrics"]["inference"]["chat-template"] = "/tmp/chat_template.jinja"
        config["autometrics"]["evaluators"] = {
            "obm": {
                "obm_config": "devices",
                "all": {"groups": ["open"]},
            },
            "lm_eval": {
                "pretrain": {
                    "longbench": {"model_args": {}},
                },
                "instruct": {
                    "chat_task": {"model_args": {}},
                },
            },
            "mera": {
                "tasks": {
                    "simplear": {"model_args": {}},
                },
            },
            "lmms_eval": {
                "chatml_version": "v4.1",
                "benchmarks": {
                    "tasks": ["chartqa_giga"],
                },
            },
            "harbor_rli": {
                "benchmark": "harness-bench-lite",
                "benchmark-kwargs": {"recursion-limit": 80},
            },
        }
        model = _load_dumped_model(tmp_path, config)
        _apply_evaluator_defaults(
            model,
            inference_instances=2,
            hf_checkpoint_path=tmp_path / "hf-model",
        )

        obm = model.autometrics.evaluators.obm
        assert obm is not None
        assert obm.n_threads == 2 * DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE

        lm_eval = model.autometrics.evaluators.lm_eval
        assert lm_eval is not None
        assert lm_eval.pretrain["longbench"].batch_size == 1
        assert (
            lm_eval.pretrain["longbench"].model_args["base_url"]
            == "http://127.0.0.1:31234/generate"
        )
        assert (
            lm_eval.instruct["chat_task"].model_args["base_url"]
            == "http://127.0.0.1:31234/v1/chat/completions"
        )

        mera = model.autometrics.evaluators.mera
        assert mera is not None
        assert (
            mera.tasks["simplear"].model_args["base_url"]
            == "http://127.0.0.1:31234/v1/chat/completions"
        )

        lmms_eval = model.autometrics.evaluators.lmms_eval
        assert lmms_eval is not None
        assert lmms_eval.chatml_version == "v4.1"
        assert lmms_eval.benchmarks is not None
        assert lmms_eval.benchmarks.tasks == ["chartqa_giga"]
        assert (
            lmms_eval.require_runtime_target().model_overrides["api"]["obm_proxy"][
                "api"
            ]["base_url"]
            == "http://127.0.0.1:31234"
        )
        assert (
            obm.shooter_config["devices"]["api"]["base_url"] == "http://127.0.0.1:31234"
        )

        harbor_rli = model.autometrics.evaluators.harbor_rli
        assert harbor_rli is not None
        assert harbor_rli.model_api_base == "http://127.0.0.1:31234/v1"
        assert harbor_rli.n_concurrent == 16

        evaluator_names = [name for name, _cfg in model.autometrics.evaluators]
        assert evaluator_names == [
            "lm_eval",
            "lmms_eval",
            "mera",
            "obm",
            "harbor_rli",
        ]
        assert model.autometrics.has_lm_eval_instruct is True

    def test_evaluators_config_iter_skips_empty_evaluators(self):
        evaluators = EvaluatorsConfig(
            obm=DevicesObmEvaluatorConfig(
                all=ObmAllConfig(groups=["open"]),
                n_threads=8,
            ),
            bfcl=None,
        )

        assert list(evaluators) == [("obm", evaluators.obm)]

    def test_evaluators_config_forbids_obm_with_new_obm(self):
        obm_cfg = DevicesObmEvaluatorConfig(
            all=ObmAllConfig(groups=["open"]),
            n_threads=8,
        )

        with pytest.raises(
            ValidationError, match="obm and new_obm cannot run together"
        ):
            EvaluatorsConfig(obm=obm_cfg, new_obm=obm_cfg)

    def test_evaluators_config_allows_new_obm_without_obm(self):
        evaluators = EvaluatorsConfig(
            new_obm=DevicesObmEvaluatorConfig(
                all=ObmAllConfig(groups=["open"]),
                n_threads=8,
            ),
        )

        assert list(evaluators) == [("new_obm", evaluators.new_obm)]

    def test_obm_all_requires_groups_or_tasks_in_loaded_yaml(self, tmp_path: Path):
        config = _base_node_runner_config()
        config["autometrics"]["evaluators"] = {
            "obm": {
                "obm_config": "devices",
                "all": {},
            }
        }
        config_path = tmp_path / "invalid_obm.yaml"
        _dump_config(config_path, config)
        with pytest.raises(
            ValueError,
            match="At least one of 'groups' or 'tasks' must be specified",
        ):
            NodeRunnerConfigModel.load(config_path)

    def test_obm_runner_config_keeps_n_threads_none_before_defaults(
        self, tmp_path: Path
    ):
        config = _base_node_runner_config()
        config["autometrics"]["evaluators"] = {
            "obm": {
                "obm_config": "devices",
                "all": {"groups": ["open"]},
            }
        }
        model = _load_dumped_model(tmp_path, config)
        obm = model.autometrics.evaluators.obm
        assert obm is not None
        assert obm.runner_config["n_threads"] is None

    def test_apply_defaults_noop_when_evaluators_section_is_empty(self, tmp_path: Path):
        model = _load_dumped_model(tmp_path, _base_node_runner_config())
        _apply_evaluator_defaults(
            model,
            inference_instances=2,
            hf_checkpoint_path=tmp_path / "hf-model",
        )
        assert model.autometrics.evaluators.lm_eval is None
        assert model.autometrics.evaluators.mera is None
        assert model.autometrics.evaluators.obm is None

    def test_apply_defaults_sets_gigametric_semaphore_from_inference_instances(
        self, tmp_path: Path
    ):
        config = _base_node_runner_config()
        config["autometrics"]["evaluators"] = {
            "gigametric": {
                "base_dir": str(tmp_path / "gm"),
                "local_model_id": "local",
                "validation_config": {
                    "validation_name": "x",
                    "models": [
                        {"id": "local", "llm_config": {"model": "openai"}},
                    ],
                },
            }
        }
        model = _load_dumped_model(tmp_path, config)
        _apply_evaluator_defaults(
            model,
            inference_instances=4,
            hf_checkpoint_path=tmp_path / "hf-model",
        )
        gm = model.autometrics.evaluators.gigametric
        assert gm is not None
        assert (
            gm.validation_config["models"][0]["llm_config"]["semaphore"]
            == 4 * DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE
        )

    def test_apply_defaults_caps_planbench_parallel_by_inference_instances(
        self, tmp_path: Path
    ):
        config = _base_node_runner_config()
        config["autometrics"]["evaluators"] = {
            "planbench": {
                "engine": "planbench_chat",
                "domains": ["blocksworld", "logistics", "mystery_blocksworld"],
            }
        }
        model = _load_dumped_model(tmp_path, config)
        _apply_evaluator_defaults(
            model,
            inference_instances=2,
            hf_checkpoint_path=tmp_path / "hf-model",
        )
        pb = model.autometrics.evaluators.planbench
        assert pb is not None
        assert pb.base_url == (
            f"http://127.0.0.1:{model.autometrics.obm_proxy.api_port}/v1"
        )
        assert pb.max_parallel_domains == 2


class TestConfigModelContracts:
    def test_base_obm_shooter_config_is_abstract_contract(self):
        class _BaseObmImpl(BaseObmEvaluatorConfig):
            pass

        cfg = _BaseObmImpl(
            obm_config="base", all=ObmAllConfig(groups=["open"]), n_threads=1
        )
        with pytest.raises(NotImplementedError):
            _ = cfg.shooter_config

    @pytest.mark.parametrize(
        ("raw_value", "expected_tasks"),
        [
            (None, {}),
            ({"simplear": "none"}, {"simplear"}),
        ],
    )
    def test_mera_wrap_flat_mapping_handles_none_and_flat_mapping(
        self, raw_value, expected_tasks
    ):
        cfg = MeraEvaluatorConfig.model_validate(raw_value)
        if isinstance(expected_tasks, set):
            assert set(cfg.tasks) == expected_tasks
            return
        assert cfg.tasks == expected_tasks

    def test_mera_wrap_flat_mapping_passes_through_non_dict(self):
        with pytest.raises(ValidationError):
            MeraEvaluatorConfig.model_validate(["not", "a", "dict"])

    def test_mera_flat_mapping_extracts_pool_max_parallel(self):
        cfg = MeraEvaluatorConfig.model_validate(
            {"pool_max_parallel": 4, "simplear": "none"}
        )
        assert cfg.pool_max_parallel == 4
        assert set(cfg.tasks) == {"simplear"}

    def test_mera_flat_mapping_splits_system_instruction_from_tasks(self):
        cfg = MeraEvaluatorConfig.model_validate(
            {
                "system_instruction": "do the task",
                "simplear": "none",
                "rucodeeval": {"gen_kwargs": '{"do_sample": true}'},
            }
        )
        assert cfg.system_instruction == "do the task"
        assert set(cfg.tasks) == {"simplear", "rucodeeval"}
        assert all(isinstance(t, MeraTaskConfig) for t in cfg.tasks.values())

    def test_mera_system_instruction_none_string_normalized(self):
        cfg = MeraEvaluatorConfig.model_validate(
            {"system_instruction": "none", "simplear": "none"}
        )
        assert cfg.system_instruction is None

    def test_mera_system_instruction_none_string_normalized_structured(self):
        cfg = MeraEvaluatorConfig.model_validate(
            {"system_instruction": "none", "tasks": {"simplear": "none"}}
        )
        assert cfg.system_instruction is None
        assert set(cfg.tasks) == {"simplear"}

    def test_mera_task_gen_kwargs_default_and_override(self):
        cfg = MeraEvaluatorConfig.model_validate(
            {
                "simplear": "none",
                "rucodeeval": {"gen_kwargs": '{"do_sample": true}'},
            }
        )
        assert cfg.tasks["simplear"].gen_kwargs == '{"until": []}'
        assert cfg.tasks["rucodeeval"].gen_kwargs == '{"do_sample": true}'

    def test_mera_system_instruction_excluded_from_dump(self):
        cfg = MeraEvaluatorConfig.model_validate(
            {"system_instruction": "hidden", "simplear": "none"}
        )
        assert "system_instruction" not in cfg.model_dump()

    def test_mera_task_declares_system_instruction_and_fewshot_fields(self):
        task = MeraTaskConfig()
        assert "system_instruction" in MeraTaskConfig.model_fields
        assert "fewshot_as_multiturn" in MeraTaskConfig.model_fields
        assert task.system_instruction is None
        assert task.fewshot_as_multiturn is True
