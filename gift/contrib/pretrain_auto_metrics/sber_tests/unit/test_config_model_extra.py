from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from pydantic import ValidationError

from node_runner.src.config_model import (
    AutometricsConfig,
    GlobalsConfig,
    BfclEvaluatorConfig,
    DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE,
    DreddEvaluatorConfig,
    ExternalApiInferenceConfig,
    ForgeEvaluatorConfig,
    GigaMetricJudgedEvaluatorConfig,
    InferenceConfig,
    LmEvalPretrainTaskConfig,
    MeraTaskConfig,
    MetricsEvaluatorConfig,
    Nl2RepoEvaluatorConfig,
    NodeRunnerConfigModel,
    ObmAllConfig,
    OpenaiObmEvaluatorConfig,
    OpenaiObmJudgedEvaluatorConfig,
    Tau2EvaluatorConfig,
)
from sber_tests.test_support import judge_endpoints

pytestmark = pytest.mark.unit


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


def _load_dumped_model(tmp_path: Path, config: dict) -> NodeRunnerConfigModel:
    config_path = tmp_path / "pam_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    return NodeRunnerConfigModel.load(config_path)


def _apply_evaluator_defaults(
    model: NodeRunnerConfigModel,
    *,
    inference_instances: int,
    hf_checkpoint_path: Path,
) -> None:
    base_url = f"http://127.0.0.1:{model.autometrics.obm_proxy.api_port}"
    for _, evaluator_config in model.autometrics.evaluators:
        evaluator_config.judge_endpoints = judge_endpoints(
            {
                name: "http://127.0.0.1:31000"
                for name in evaluator_config.secondary_model_names
            }
        )
        evaluator_config.apply_defaults(
            inference_instances=inference_instances,
            base_url=base_url,
            hf_checkpoint_path=hf_checkpoint_path,
        )


class TestTau2EvaluatorConfig:
    def test_invalid_domain_raises(self):
        with pytest.raises(ValueError, match="Invalid tau2 domain"):
            Tau2EvaluatorConfig(domains=["nope"], user_model_name="gpt-oss-120b")

    def test_apply_defaults_sets_agent_and_user_api_bases(self):
        cfg = Tau2EvaluatorConfig(domains=["airline"], user_model_name="gpt-oss-120b")
        cfg.judge_endpoints = judge_endpoints(
            {"gpt-oss-120b": "http://127.0.0.1:31000"}
        )
        cfg.apply_defaults(
            inference_instances=1,
            base_url="http://127.0.0.1:31234",
        )
        assert cfg.agent_api_base == "http://127.0.0.1:31234/v1"
        assert cfg.user_api_base == "http://127.0.0.1:31000/v1"

    def test_build_cli_run_requires_apply_defaults_user_api_base(self):
        cfg = Tau2EvaluatorConfig(domains=["airline"], user_model_name="gpt-oss-120b")
        with pytest.raises(
            ValueError,
            match="agent_api_base must be set via apply_defaults",
        ):
            cfg.build_cli_run(domain="airline", save_to="run_airline_1")

    def test_build_cli_run_requires_user_api_base_after_agent_base_is_set(self):
        cfg = Tau2EvaluatorConfig(domains=["airline"], user_model_name="gpt-oss-120b")
        cfg.agent_api_base = "http://127.0.0.1:29999/v1"

        with pytest.raises(ValueError, match="user_api_base must be set"):
            cfg.build_cli_run(domain="airline", save_to="run_airline_1")

    def test_secondary_model_names_requires_non_empty_user_model(self):
        cfg = Tau2EvaluatorConfig.model_construct(
            domains=["airline"],
            user_model_name=" ",
        )

        with pytest.raises(ValueError, match="user_model_name must not be empty"):
            _ = cfg.secondary_model_names


class TestNl2RepoEvaluatorConfig:
    def test_serializes_public_benchmark_fields(self) -> None:
        cfg = Nl2RepoEvaluatorConfig.model_validate(
            {
                "pass_at_k": 3,
                "n_tasks": 5,
                "include_task_name": "aiofiles",
                "exclude_task_name": ["tqdm"],
                "verifier_timeout": 1200,
                "max_retries": 2,
                "agent_env_version": "1.0.3",
                "verifier_env_version": "1.0.1",
                "enable_contamination_filter": True,
            }
        )

        payload = cfg.to_dict()

        assert payload["pass-at-k"] == 3
        assert payload["n-tasks"] == 5
        assert payload["task-name"] == "aiofiles"
        assert payload["exclude-task-name"] == ["tqdm"]
        assert payload["verifier-timeout"] == 1200
        assert payload["max-retries"] == 2
        assert payload["agent-env-version"] == "1.0.3"
        assert payload["verifier-env-version"] == "1.0.1"
        assert payload["enable-contamination-filter"] is True
        assert payload["max-input-tokens"] == 98304
        for removed in (
            "tasks-root",
            "trials",
            "expected-tasks",
            "include",
            "exclude",
            "limit",
            "agent-timeout-sec",
            "verifier-timeout-sec",
            "task-cpus",
            "task-memory-mb",
            "task-storage-mb",
        ):
            assert removed not in payload

    def test_normalizes_legacy_env_version_as_verifier_version(self) -> None:
        cfg = Nl2RepoEvaluatorConfig.model_validate({"env-version": "1.0.1"})

        payload = cfg.to_dict()

        assert payload["agent-env-version"] == "1.0.2"
        assert payload["verifier-env-version"] == "1.0.1"
        assert "env-version" not in payload

    def test_applies_model_endpoint_and_parallel_default(self) -> None:
        cfg = Nl2RepoEvaluatorConfig.model_validate({})

        assert cfg.agent == "openhands-sdk"
        assert cfg.enable_contamination_filter is True
        assert cfg.n_concurrent is None

        cfg.apply_defaults(
            inference_instances=4,
            base_url="http://127.0.0.1:31234",
        )

        assert cfg.model_api_base == "http://127.0.0.1:31234/v1"
        assert cfg.n_concurrent == 32

    def test_preserves_explicit_model_endpoint_and_parallelism(self) -> None:
        cfg = Nl2RepoEvaluatorConfig.model_validate(
            {
                "model-api-base": "https://model.example/v1",
                "n-concurrent": 7,
            }
        )

        cfg.apply_defaults(
            inference_instances=4,
            base_url="http://127.0.0.1:31234",
        )

        assert cfg.model_api_base == "https://model.example/v1"
        assert cfg.n_concurrent == 7


class TestBfclEvaluatorConfigNormalization:
    def test_normalize_test_category_joins_list(self):
        cfg = BfclEvaluatorConfig.model_validate(
            {
                "model": "gigachat-FC",
                "test-category": ["live_simple", "web_search", " ", ""],
            }
        )
        assert cfg.test_category == "live_simple,web_search"

    def test_normalize_model_joins_list(self):
        cfg = BfclEvaluatorConfig.model_validate(
            {
                "model": ["gigachat-FC", "another-model", " "],
                "test_category": "web_search",
            }
        )
        assert cfg.model == "gigachat-FC,another-model"


class TestBfclEvaluatorConfigToBashList:
    def test_to_bash_list_expands_lora_modules_list(self):
        cfg = BfclEvaluatorConfig.model_validate(
            {
                "model": "gigachat-FC",
                "test_category": "web_search",
                "lora-modules": ["adapter-a=/tmp/a", "adapter-b=/tmp/b"],
            }
        )
        cfg.apply_defaults(
            inference_instances=1,
            base_url="http://127.0.0.1:31234",
        )
        bash = cfg.to_bash_list()
        # Each lora module is emitted as its own --lora-modules argument.
        lora_indices = [i for i, x in enumerate(bash) if x == "--lora-modules"]
        assert len(lora_indices) == 2
        for idx in lora_indices:
            assert bash[idx + 1] in ("adapter-a=/tmp/a", "adapter-b=/tmp/b")
        # timeout travels via env (OPENAI_TIMEOUT), never as a CLI flag.
        assert "--timeout" not in bash

    def test_to_bash_list_dumps_dict_values_as_json(self, monkeypatch):
        cfg = BfclEvaluatorConfig.model_validate(
            {
                "model": "gigachat-FC",
                "test_category": "web_search",
            }
        )
        cfg.apply_defaults(
            inference_instances=1,
            base_url="http://127.0.0.1:31234",
        )

        original_dump = cfg.model_dump

        def _patched_dump(*args, **kwargs):
            dump = original_dump(*args, **kwargs)
            # Inject a dict-typed field to exercise the json.dumps fallback path.
            dump["custom-json"] = {"nested": [1, 2]}
            return dump

        monkeypatch.setattr(cfg, "model_dump", _patched_dump)
        bash = cfg.to_bash_list()
        idx = bash.index("--custom-json")
        assert bash[idx + 1] == '{"nested": [1, 2]}'

    def test_apply_defaults_keeps_existing_num_threads(self):
        cfg = BfclEvaluatorConfig(
            model="gigachat-FC",
            test_category="web_search",
            num_threads=3,
        )

        cfg.apply_defaults(inference_instances=8, base_url="http://127.0.0.1:29999")

        assert cfg.num_threads == 3
        assert cfg.base_url == "http://127.0.0.1:29999/v1"


class TestExternalApiInferenceConfigAuth:
    @pytest.mark.parametrize(
        ("formatting", "base_url", "expected", "auth"),
        [
            (
                "openai",
                "https://api.openai.com/v1",
                "https://api.openai.com",
                {"api_key": "key"},
            ),
            (
                "openai",
                "https://api.openai.com/v1/",
                "https://api.openai.com",
                {"api_key": "key"},
            ),
            (
                "openai",
                "https://api.openai.com/v1/chat/completions",
                "https://api.openai.com",
                {"api_key": "key"},
            ),
            (
                "openai",
                "https://openrouter.ai/api/v1",
                "https://openrouter.ai/api",
                {"api_key": "key"},
            ),
            (
                "openai",
                "https://openrouter.ai/api/v1/chat/completions",
                "https://openrouter.ai/api",
                {"api_key": "key"},
            ),
            (
                "openai",
                "https://openrouter.ai/api",
                "https://openrouter.ai/api",
                {"api_key": "key"},
            ),
            (
                "ift",
                "https://gigachat.ift.sberdevices.ru/v1",
                "https://gigachat.ift.sberdevices.ru",
                {"user_pass": "user:pass"},
            ),
            (
                "ift",
                "https://gigachat.ift.sberdevices.ru/v2",
                "https://gigachat.ift.sberdevices.ru",
                {"user_pass": "user:pass"},
            ),
            (
                "ift",
                "https://gigachat.ift.sberdevices.ru/v2/chat/completions",
                "https://gigachat.ift.sberdevices.ru",
                {"user_pass": "user:pass"},
            ),
        ],
    )
    def test_normalizes_base_url_v1_suffix(
        self, formatting: str, base_url: str, expected: str, auth: dict[str, str]
    ):
        cfg = ExternalApiInferenceConfig.model_validate(
            {
                "formatting": formatting,
                "base_url": base_url,
                "model": "gpt-4o-mini",
                **auth,
            }
        )

        assert cfg.base_url == expected

    def test_openai_formatting_requires_api_key(self):
        with pytest.raises(ValueError, match="api_key is required"):
            ExternalApiInferenceConfig.model_validate(
                {
                    "formatting": "openai",
                    "base_url": "https://api.openai.com/v1",
                    "model": "gpt-4o-mini",
                }
            )

    def test_openai_formatting_rejects_user_pass(self):
        with pytest.raises(ValueError, match="user_pass is supported only"):
            ExternalApiInferenceConfig.model_validate(
                {
                    "formatting": "openai",
                    "base_url": "https://api.openai.com/v1",
                    "model": "gpt-4o-mini",
                    "api_key": "key",
                    "user_pass": "user:pass",
                }
            )

    def test_ift_formatting_accepts_user_pass_without_api_key(self):
        cfg = ExternalApiInferenceConfig.model_validate(
            {
                "formatting": "ift",
                "base_url": "https://gigachat.ift.sberdevices.ru",
                "model": "model",
                "user_pass": "user:pass",
            }
        )

        assert cfg.api_key is None
        assert cfg.user_pass == "user:pass"

    def test_ift_formatting_accepts_blank_user_pass_when_api_key_is_set(self):
        cfg = ExternalApiInferenceConfig.model_validate(
            {
                "formatting": "ift",
                "base_url": "https://gigachat.ift.sberdevices.ru",
                "model": "model",
                "api_key": "key",
                "user_pass": "...",
            }
        )

        assert cfg.api_key == "key"
        assert cfg.user_pass == "..."

    @pytest.mark.parametrize("user_pass", ["...:...", "userpass"])
    def test_user_pass_must_have_user_and_password(self, user_pass: str):
        with pytest.raises(ValueError, match="USER:PASS format"):
            ExternalApiInferenceConfig.model_validate(
                {
                    "formatting": "ift",
                    "base_url": "https://gigachat.ift.sberdevices.ru",
                    "model": "model",
                    "user_pass": user_pass,
                }
            )

    def test_ift_formatting_requires_api_key_or_user_pass(self):
        with pytest.raises(ValueError, match=r"api_key or user_pass"):
            ExternalApiInferenceConfig.model_validate(
                {
                    "formatting": "ift",
                    "base_url": "https://gigachat.ift.sberdevices.ru",
                    "model": "model",
                    "api_key": "...",
                }
            )

    def test_uses_external_api_property_is_true(self):
        cfg = ExternalApiInferenceConfig.model_validate(
            {
                "formatting": "openai",
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-4o-mini",
                "api_key": "key",
            }
        )

        assert cfg.uses_external_api is True

    def test_request_customization_cannot_override_protected_keys(self):
        with pytest.raises(ValueError, match="protected request keys: model"):
            ExternalApiInferenceConfig.model_validate(
                {
                    "formatting": "openai",
                    "base_url": "https://api.openai.com/v1",
                    "model": "gpt-4o-mini",
                    "api_key": "key",
                    "request_customization": {"model": "other"},
                }
            )


class TestNodeRunnerConfigApplyDefaultsBfclAndTau2:
    def test_apply_defaults_propagates_obm_proxy_port_to_bfcl_and_tau2(
        self, tmp_path: Path
    ):
        config = _base_node_runner_config()
        config["autometrics"]["obm_proxy"]["api-port"] = 31234
        config["autometrics"]["evaluators"] = {
            "bfcl": {
                "model": "gigachat-FC",
                "test_category": "web_search",
            },
            "tau2": {
                "domains": ["airline"],
                "user_model_name": "gpt-oss-120b",
            },
        }
        model = _load_dumped_model(tmp_path, config)
        _apply_evaluator_defaults(
            model,
            inference_instances=2,
            hf_checkpoint_path=tmp_path / "hf-model",
        )

        assert model.autometrics.evaluators.bfcl is not None
        assert model.autometrics.evaluators.bfcl.base_url == (
            "http://127.0.0.1:31234/v1"
        )
        assert model.autometrics.evaluators.tau2 is not None
        assert model.autometrics.evaluators.tau2.agent_api_base == (
            "http://127.0.0.1:31234/v1"
        )


class TestForgeEvaluatorConfig:
    def test_duplicate_recipe_in_groups_raises(self):
        with pytest.raises(ValueError, match="multiple forge groups"):
            ForgeEvaluatorConfig(
                judge="j-a",
                groups=[
                    {"recipes": ["camo"], "judge": "j-b"},
                    {"recipes": ["camo"], "judge": "j-c"},
                ],
            )

    def test_apply_defaults_sets_attack_api_base_and_judge_urls(self):
        cfg = ForgeEvaluatorConfig(judge="j-a")
        cfg.judge_endpoints = judge_endpoints({"j-a": "http://127.0.0.1:31000"})
        cfg.apply_defaults(
            inference_instances=1,
            base_url="http://127.0.0.1:31234",
        )
        assert cfg.attack_api_base == "http://127.0.0.1:31234/v1"
        assert cfg.build_meta_override("camo")["judge"]["base_url"] == (
            "http://127.0.0.1:31000/v1"
        )

    def test_build_meta_override_uses_url_of_recipe_judge(self):
        cfg = ForgeEvaluatorConfig(
            judge="j-a", groups=[{"recipes": ["camo"], "judge": "j-b"}]
        )
        cfg.judge_endpoints = judge_endpoints(
            {"j-a": "http://a:31000", "j-b": "http://b:31001"}
        )
        cfg.apply_defaults(
            inference_instances=1,
            base_url="http://127.0.0.1:29999",
        )

        assert cfg.build_meta_override("camo")["judge"]["base_url"] == (
            "http://b:31001/v1"
        )
        assert cfg.build_meta_override("other")["judge"]["base_url"] == (
            "http://a:31000/v1"
        )

    def test_build_meta_override_uses_group_overrides(self):
        cfg = ForgeEvaluatorConfig(
            judge="j-a",
            temperature=0.1,
            groups=[
                {
                    "recipes": ["camo"],
                    "judge": "j-b",
                    "temperature": 0.9,
                    "max_tokens": 256,
                    "judge_prompt_template": "builtin:reasoning",
                }
            ],
        )
        cfg.judge_endpoints = judge_endpoints(
            {"j-a": "http://x:8000", "j-b": "http://x:8000"}
        )
        cfg.apply_defaults(
            inference_instances=1,
            base_url="http://127.0.0.1:29999",
        )
        override = cfg.build_meta_override("camo")
        assert override["model"]["temperature"] == 0.9
        assert override["model"]["max_tokens"] == 256
        assert override["judge"]["model"] == "j-b"
        assert override["judge"]["prompt_template"] == "builtin:reasoning"

    def test_build_meta_override_omits_timeout_when_none(self):
        cfg = ForgeEvaluatorConfig(judge="j-a", timeout=None)
        cfg.judge_endpoints = judge_endpoints({"j-a": "http://x:8000"})
        cfg.apply_defaults(
            inference_instances=1,
            base_url="http://127.0.0.1:29999",
        )

        override = cfg.build_meta_override("camo")

        assert "timeout" not in override["model"]
        assert "timeout" not in override["judge"]

    def test_secondary_model_names_rejects_empty_judge_names(self):
        cfg = ForgeEvaluatorConfig.model_construct(
            judge="judge-a",
            groups=[{"recipes": ["r1"], "judge": " "}],
        )

        with pytest.raises(ValueError, match="judge model names"):
            _ = cfg.secondary_model_names

    def test_secondary_model_names_deduplicates_group_judges(self):
        cfg = ForgeEvaluatorConfig(
            judge="judge-a",
            groups=[
                {"recipes": ["r1"], "judge": "judge-b"},
                {"recipes": ["r2"], "judge": "judge-b"},
            ],
        )

        assert cfg.secondary_model_names == ["judge-a", "judge-b"]


class TestNodeRunnerApplyDefaultsForge:
    def test_apply_defaults_propagates_obm_proxy_port_to_forge(self, tmp_path: Path):
        config = _base_node_runner_config()
        config["autometrics"]["obm_proxy"]["api-port"] = 31234
        config["autometrics"]["evaluators"] = {"forge": {"judge": "pollux-judge-32b"}}
        model = _load_dumped_model(tmp_path, config)
        _apply_evaluator_defaults(
            model,
            inference_instances=2,
            hf_checkpoint_path=tmp_path / "hf-model",
        )
        assert model.autometrics.evaluators.forge is not None
        assert model.autometrics.evaluators.forge.attack_api_base == (
            "http://127.0.0.1:31234/v1"
        )


def _make_judged_cfg(**overrides) -> OpenaiObmJudgedEvaluatorConfig:
    kwargs = dict(
        obm_config="openai",
        all=ObmAllConfig(tasks=["t"]),
        judges={"ds1": "j-a"},
        n_threads=8,
    )
    kwargs.update(overrides)
    return OpenaiObmJudgedEvaluatorConfig(**kwargs)


class TestObmRunnerConfig:
    def test_includes_shooter_params_and_task_overrides(self):
        task_override = {
            "dataset": {"max_items": 4},
            "benchmark": {
                "rollouts": {
                    "count_per_item": 3,
                    "aggregation": {"mode": "flat"},
                }
            },
        }
        cfg = OpenaiObmEvaluatorConfig(
            all=ObmAllConfig(tasks=["ifbench_official"]),
            shooter_params={"temperature": 0.0, "max_tokens": 4096},
            task_overrides={"ifbench_official": task_override},
        )
        cfg.apply_defaults(
            inference_instances=1,
            base_url="http://127.0.0.1:29999",
        )

        runner_config = cfg.runner_config

        assert runner_config["benchmark"]["runner"]["shooter"]["openai"]["params"] == {
            "temperature": 0.0,
            "max_tokens": 4096,
        }
        assert runner_config["ifbench_official"] == task_override

    @pytest.mark.parametrize(
        ("field", "invalid_value"),
        [
            ("task_overrides", {"ifbench_official": []}),
            ("shooter_params", []),
        ],
    )
    def test_rejects_non_mapping_overrides(self, field, invalid_value):
        with pytest.raises(ValidationError):
            OpenaiObmEvaluatorConfig.model_validate(
                {
                    "all": {"tasks": ["ifbench_official"]},
                    field: invalid_value,
                }
            )

    def test_task_overrides_cannot_replace_runner_config(self):
        cfg = OpenaiObmEvaluatorConfig(
            all=ObmAllConfig(tasks=["ifbench_official"]),
            task_overrides={"logging": {"enabled": False}},
        )

        assert cfg.runner_config["logging"] == {
            "enabled": True,
            "file": "obm.log",
            "level": "INFO",
        }


class TestObmJudgedMixinApplyDefaults:
    def test_sets_n_threads(self):
        cfg = _make_judged_cfg(n_threads=None)
        cfg.judge_endpoints = judge_endpoints({"ds1": "http://x:8000"})
        cfg.apply_defaults(
            inference_instances=2,
            base_url="http://127.0.0.1:29999",
        )
        assert cfg.n_threads == 2 * DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE

    def test_keeps_existing_n_threads(self):
        cfg = _make_judged_cfg(n_threads=42)
        cfg.judge_endpoints = judge_endpoints({"ds1": "http://x:8000"})
        cfg.apply_defaults(
            inference_instances=2,
            base_url="http://127.0.0.1:29999",
        )
        assert cfg.n_threads == 42

    def test_rejects_empty_judge_name(self):
        with pytest.raises(ValueError, match="non-empty model names"):
            _make_judged_cfg(judges={"": ["task"]})

    def test_normalize_judges_leaves_non_dict_values_to_pydantic(self):
        assert OpenaiObmJudgedEvaluatorConfig._normalize_judges(["judge-a"]) == [
            "judge-a"
        ]


class TestObmJudgedJudgesValidation:
    """Судьи ходят в один dump-dir, а кэш OBM судью не различает: пересечение таск-листов
    привело бы к тому, что второй судья публикует вердикты первого."""

    def test_rejects_task_shared_by_two_judges(self):
        with pytest.raises(ValueError, match="'hle' assigned to both"):
            _make_judged_cfg(
                judges={"pollux-judge-32b": ["pollux", "hle"], "qwen-235b": ["hle"]}
            )

    def test_accepts_disjoint_task_lists(self):
        judges = {
            "pollux-judge-32b": ["pollux"],
            "qwen-235b": ["simpleqa", "imo_answer_bench", "hle"],
        }

        cfg = _make_judged_cfg(judges=judges)

        assert cfg.judges == judges


class TestObmJudgedMixinRunnerConfig:
    def test_judged_runner_config_sets_judge_shooter_per_judge(self):
        cfg = _make_judged_cfg(judges={"j-a": ["t1"], "j-b": ["t2"]})
        cfg.judge_endpoints = judge_endpoints(
            {"j-a": "http://a:31000", "j-b": "http://b:31001"}
        )
        cfg.apply_defaults(
            inference_instances=1,
            base_url="http://127.0.0.1:29999",
        )

        urls = {
            judge: cfg.judged_runner_config(judge)["benchmark"]["runner"][
                "llm_judge_shooter"
            ]["openai"]["api"]["base_url"]
            for judge in ("j-a", "j-b")
        }

        assert urls == {"j-a": "http://a:31000/v1", "j-b": "http://b:31001/v1"}

    def test_judged_runner_config_adds_llm_judge_shooter(self):
        cfg = _make_judged_cfg()
        cfg.judge_endpoints = judge_endpoints({"ds1": "http://x:8000"})
        cfg.apply_defaults(
            inference_instances=1,
            base_url="http://127.0.0.1:29999",
        )
        rc = cfg.judged_runner_config("ds1")
        assert (
            rc["benchmark"]["runner"]["llm_judge_shooter"]["openai"]["api"]["base_url"]
            == "http://x:8000/v1"
        )
        # Regular shooter remains pointed at the local obm_proxy api_port.
        assert rc["benchmark"]["runner"]["shooter"]["openai"]["api"][
            "base_url"
        ].startswith("http://127.0.0.1:")
        # Large default request timeout reaches both shooters.
        assert rc["benchmark"]["runner"]["shooter"]["openai"]["api"]["timeout"] == 21600
        assert (
            rc["benchmark"]["runner"]["llm_judge_shooter"]["openai"]["api"]["timeout"]
            == 21600
        )


class TestNodeRunnerApplyDefaultsObmJudged:
    def test_apply_defaults_propagates_obm_proxy_port_to_obm_judged(
        self, tmp_path: Path
    ):
        config = _base_node_runner_config()
        config["autometrics"]["obm_proxy"]["api-port"] = 31234
        config["autometrics"]["evaluators"] = {
            "obm_judged": {
                "obm_config": "openai",
                "all": {"tasks": ["t1"]},
                "judges": {"pollux-judge-32b": ["t1"]},
            },
        }
        model = _load_dumped_model(tmp_path, config)
        _apply_evaluator_defaults(
            model,
            inference_instances=2,
            hf_checkpoint_path=tmp_path / "hf-model",
        )
        assert model.autometrics.evaluators.obm_judged is not None
        assert model.autometrics.evaluators.obm_judged.base_url == (
            "http://127.0.0.1:31234"
        )
        assert (
            model.autometrics.evaluators.obm_judged.n_threads
            == 2 * DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE
        )


class TestBaseConfigContracts:
    def test_metrics_evaluator_config_defaults_are_noops(self):
        cfg = MetricsEvaluatorConfig()

        assert cfg.apply_defaults(inference_instances=1, base_url="http://x") is None
        assert cfg.secondary_model_names == []

    def test_dredd_config_typed_fields_and_serialization(self):
        cfg = DreddEvaluatorConfig.model_validate(
            {
                "judge": {"model_name": "judge-v1"},
                "model": {"api_base": "http://localhost:29999/v1"},
            }
        )

        assert cfg.judge.model_name == "judge-v1"
        assert cfg.secondary_model_names == ["judge-v1"]
        # datasets stay absent in the payload so dredd keeps its default arenas
        assert cfg.to_dict() == {
            "judge": {
                "model_name": "judge-v1",
                "request_timeout": 21600,
                "max_tokens": 16384,
                "pam_mode": True,
            },
            "model": {
                "request_timeout": 21600,
                "api_base": "http://localhost:29999/v1",
            },
        }
        with pytest.raises(ValueError):
            DreddEvaluatorConfig.model_validate("raw")

    def test_dredd_apply_defaults_scales_parallel_api_calls_and_sets_urls(self):
        endpoints = judge_endpoints({"j": "http://judge"}, {"j": 3})
        cfg = DreddEvaluatorConfig.model_validate({"judge": {"model_name": "j"}})
        cfg.judge_endpoints = endpoints
        cfg.apply_defaults(inference_instances=2, base_url="http://x")
        assert (
            cfg.model.parallel_api_calls == 2 * DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE
        )
        assert cfg.model.api_base == "http://x/v1"
        assert cfg.judge.api_base == "http://judge/v1"
        assert (
            cfg.judge.parallel_api_calls == 3 * DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE
        )

        explicit = DreddEvaluatorConfig.model_validate(
            {
                "judge": {"model_name": "j", "api_base": "http://judge-gw/v1"},
                "model": {"parallel_api_calls": 16, "api_base": "http://model/v1"},
            }
        )
        explicit.judge_endpoints = endpoints
        explicit.apply_defaults(inference_instances=2, base_url="http://x")
        assert explicit.model.parallel_api_calls == 16
        assert explicit.model.api_base == "http://model/v1"
        assert explicit.judge.api_base == "http://judge-gw/v1"

    def test_inference_config_base_properties_are_false(self):
        cfg = InferenceConfig()

        assert cfg.uses_external_api is False
        assert cfg.has_reasoning_parser is False
        assert cfg.has_tool_call_parser is False
        assert cfg.has_chat_template is False


class TestAutometricsConfigValidation:
    @staticmethod
    def _build(**overrides) -> AutometricsConfig:
        payload = {
            "globals": {},
            "router": {},
            "inference": {"tp": 1},
            "obm_proxy": {},
            "evaluators": {},
            "preparation": {},
            "convertation": {"mappers": ["map_fsdp_to_hf"]},
            "model_info": {
                "model_class": "sft",
                "size": "lite",
                "username": "user",
            },
        }
        payload.update(overrides)
        return AutometricsConfig.model_validate(payload)

    def test_external_api_resolves_opensource_mode_and_skips_tokenizer_sync(self):
        cfg = self._build(
            inference={
                "engine_type": "external_api",
                "formatting": "openai",
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-4o-mini",
                "api_key": "key",
            },
            obm_proxy={"tokenizer_path": "/tmp/tokenizer"},
        )

        assert cfg.resolved_chat_handlers_mode == "opensource"
        assert cfg.obm_proxy.chat_handlers_mode == "opensource"
        assert cfg.obm_proxy.tokenizer_path == "/tmp/tokenizer"

    def test_external_api_allows_lmms_eval_without_chatml_version(self):
        cfg = self._build(
            inference={
                "engine_type": "external_api",
                "formatting": "openai",
                "base_url": "https://openrouter.ai/api/v1",
                "model": "anthropic/claude-sonnet-4",
                "api_key": "key",
            },
            evaluators={
                "lmms_eval": {
                    "benchmarks": {"tasks": ["ai2d"], "limit": 5},
                },
            },
        )

        assert cfg.evaluators.lmms_eval is not None
        assert cfg.evaluators.lmms_eval.chatml_version is None

    def test_openrouter_vision_example_loads(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-key")
        config_path = (
            Path(__file__).parents[2] / "configs/vision/lmms-eval-openrouter.yaml"
        )

        cfg = NodeRunnerConfigModel.load(config_path)

        assert cfg.autometrics.inference.uses_external_api is True
        assert cfg.autometrics.inference.base_url == "https://openrouter.ai/api"
        assert cfg.autometrics.inference.model == "anthropic/claude-sonnet-4"
        assert cfg.autometrics.inference.max_concurrency == 10
        assert cfg.autometrics.inference.max_retries == 5
        assert cfg.autometrics.inference.timeout == 7200
        assert cfg.autometrics.inference.logging_level == "info"
        assert cfg.autometrics.inference.request_customization == {}
        assert cfg.autometrics.globals.delete_converted_hf is False
        lmms_eval = cfg.autometrics.evaluators.lmms_eval
        assert lmms_eval is not None
        assert lmms_eval.chatml_version is None
        assert lmms_eval.timeout == 7200
        assert lmms_eval.gigavision == {}
        assert list(lmms_eval.resolved_runs) == ["smoke"]
        assert lmms_eval.resolved_runs["smoke"].benchmarks is not None
        assert lmms_eval.resolved_runs["smoke"].benchmarks.tasks == ["ai2d"]
        assert lmms_eval.resolved_runs["smoke"].benchmarks.limit == 5

    def test_local_lmms_eval_requires_chatml_version(self):
        with pytest.raises(ValueError, match="lmms_eval.chatml_version is required"):
            self._build(evaluators={"lmms_eval": {}})

    def test_chat_template_kwargs_require_formatting_source(self):
        with pytest.raises(ValueError, match="chat_template_kwargs"):
            self._build(
                obm_proxy={"chat_template_kwargs": {"reasoning": True}},
            )

    def test_lm_eval_instruct_requires_chat_formatting(self, tmp_path: Path):
        config = _base_node_runner_config()
        config["autometrics"]["evaluators"] = {"lm_eval": {"instruct": {"mmlu": {}}}}

        with pytest.raises(ValueError, match="lm_eval.instruct requires"):
            _load_dumped_model(tmp_path, config)

    @pytest.mark.parametrize("benchmark", ["harness-bench-lite", "harness-bench-medium"])
    def test_harness_bench_requires_chat_formatting(self, benchmark):
        with pytest.raises(ValueError, match="harbor_rli"):
            self._build(
                evaluators={"harbor_rli": {"benchmark": benchmark}},
            )

    def test_external_api_rejects_unsupported_evaluators(self):
        with pytest.raises(ValueError, match="lm_eval.pretrain"):
            self._build(
                inference={
                    "engine_type": "external_api",
                    "formatting": "openai",
                    "base_url": "https://api.openai.com/v1",
                    "model": "gpt-4o-mini",
                    "api_key": "key",
                },
                evaluators={
                    "lm_eval": {"pretrain": {"bbh": {}}},
                    "unknown_eval": {},
                },
            )

    def test_external_api_dredd_requires_local_judge_model_name(self):
        with pytest.raises(ValueError, match="judge.model_name"):
            self._build(
                inference={
                    "engine_type": "external_api",
                    "formatting": "openai",
                    "base_url": "https://api.openai.com/v1",
                    "model": "gpt-4o-mini",
                    "api_key": "key",
                },
                evaluators={"dredd": {"judge": {"api_base": "http://judge"}}},
            )

    def test_external_api_tau2_requires_user_model_name(self):
        with pytest.raises(ValueError, match="tau2.user_model_name must not be empty"):
            self._build(
                inference={
                    "engine_type": "external_api",
                    "formatting": "openai",
                    "base_url": "https://api.openai.com/v1",
                    "model": "gpt-4o-mini",
                    "api_key": "key",
                },
                evaluators={
                    "tau2": Tau2EvaluatorConfig.model_construct(
                        domains=["airline"],
                        user_model_name=" ",
                    )
                },
            )

    def test_external_api_dredd_with_model_name_allows_other_aux_checks(self):
        cfg = self._build(
            inference={
                "engine_type": "external_api",
                "formatting": "openai",
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-4o-mini",
                "api_key": "key",
            },
            evaluators={
                "dredd": {"judge": {"model_name": "pollux-judge-32b"}},
            },
        )

        assert cfg.evaluators.dredd.judge.model_name == "pollux-judge-32b"

    def test_judge_with_unknown_sglang_preset_rejected(self):
        with pytest.raises(ValueError, match="unknown SGLang preset 'no-such-judge'"):
            self._build(
                evaluators={
                    "obm_judged": {
                        "obm_config": "openai",
                        "all": {"tasks": ["t1"]},
                        "judges": {"no-such-judge": ["t1"]},
                    },
                },
            )

    def test_external_judge_without_credentials_rejected(self):
        with pytest.raises(ValueError, match="api_key is required"):
            self._build(
                evaluators={
                    "obm_judged": {
                        "obm_config": "openai",
                        "all": {"tasks": ["t1"]},
                        "judges": {"openrouter:qwen/qwen3-235b": ["t1"]},
                    },
                },
            )

    def test_lmms_eval_is_supported_by_external_api(self):
        cfg = AutometricsConfig.model_construct(
            evaluators=SimpleNamespace(
                lm_eval=None,
                lmms_eval=object(),
                acc_len=None,
                model_extra={},
            )
        )

        assert cfg._unsupported_external_api_evaluators() == []

    def test_unsupported_external_api_evaluators_reports_acc_len(self):
        cfg = AutometricsConfig.model_construct(
            evaluators=SimpleNamespace(
                lm_eval=None,
                lmms_eval=None,
                acc_len=object(),
                model_extra={},
            )
        )

        assert cfg._unsupported_external_api_evaluators() == [
            "autometrics.evaluators.acc_len"
        ]


def _make_gigametric_judged_cfg(**overrides) -> GigaMetricJudgedEvaluatorConfig:
    kwargs = dict(
        validation_config={
            "benchmarks": {"hle": {"name": "hle-adapter"}},
            "models": [
                {
                    "id": "local",
                    "llm_config": {"model": "openai"},
                    "benchmarks": ["hle"],
                }
            ],
        },
        base_dir="/tmp/gigametric_runs",
        local_model_id="local",
        judges={"judge-a": ["hle"]},
    )
    kwargs.update(overrides)
    return GigaMetricJudgedEvaluatorConfig(**kwargs)


def _apply_judged_defaults(
    cfg: GigaMetricJudgedEvaluatorConfig,
    secondary_base_url: str,
    inference_instances: int = 1,
) -> None:
    cfg.judge_endpoints = judge_endpoints(
        {name: secondary_base_url for name in cfg.judges}
    )
    cfg.apply_defaults(
        inference_instances=inference_instances,
        base_url="http://localhost:29999",
    )


class TestGigaMetricJudgedApplyDefaults:
    def test_sets_semaphore(self):
        cfg = _make_gigametric_judged_cfg()
        _apply_judged_defaults(cfg, "http://x:8000", inference_instances=2)
        llm_config = cfg.validation_config["models"][0]["llm_config"]
        assert llm_config["semaphore"] == 2 * DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE


class TestGigaMetricJudgedSecondaryModelNames:
    def test_judges_are_keys_in_declaration_order(self):
        cfg = _make_gigametric_judged_cfg(
            validation_config={
                "benchmarks": {
                    "hle": {"name": "hle-adapter"},
                    "bbh": {"name": "bbh-adapter"},
                    "ragqa": {"name": "ragqa-adapter"},
                },
                "models": [
                    {
                        "id": "local",
                        "llm_config": {"model": "openai"},
                        "benchmarks": ["hle", "bbh", "ragqa"],
                    }
                ],
            },
            judges={"judge-a": ["hle", "ragqa"], "judge-b": ["bbh"]},
        )
        assert cfg.secondary_model_names == ["judge-a", "judge-b"]

    def test_scalar_task_normalized_to_list(self):
        cfg = _make_gigametric_judged_cfg(judges={"judge-a": "hle"})
        assert cfg.judges == {"judge-a": ["hle"]}

    def test_normalize_judges_leaves_non_dict_values_to_pydantic(self):
        # не-dict проходит насквозь — тип отвергнет pydantic, а не нормализатор.
        assert GigaMetricJudgedEvaluatorConfig._normalize_judges(["judge-a"]) == [
            "judge-a"
        ]

    def test_empty_judges_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            _make_gigametric_judged_cfg(judges={})

    def test_blank_judge_model_name_raises(self):
        with pytest.raises(ValueError, match="non-empty model names"):
            _make_gigametric_judged_cfg(judges={" ": ["hle"]})


class TestGigaMetricJudgedJudgeLlmConfig:
    @pytest.mark.parametrize("scope", ["defaults", "benchmark"])
    def test_replaces_user_model_with_local_judge(self, scope):
        cfg = _make_gigametric_judged_cfg()
        _apply_judged_defaults(cfg, "http://localhost:8000")
        external_user = {
            "model": "external",
            "api_key": "${oc.env:EXTERNAL_USER_TOKEN}",
            "parameters": {"temperature": 0.7},
        }
        base_config = {
            "defaults": {"benchmark_config": {}},
            "benchmarks": {"tau2": {}},
        }
        entry = (
            base_config["defaults"]["benchmark_config"]
            if scope == "defaults"
            else base_config["benchmarks"]["tau2"]
        )
        entry["parameters"] = {"user_llm_config": external_user, "num_trials": 1}

        result = cfg.judged_validation_config(base_config, "judge-a", ["tau2"])

        parameters = (
            result["defaults"]["benchmark_config"]["parameters"]
            if scope == "defaults"
            else result["benchmarks"]["tau2"]["parameters"]
        )
        assert parameters["user_llm_config"] == {
            "model": "openai",
            "name": "judge-a",
            "api_url": "http://localhost:8000/v1",
            "api_key": "not-needed",
            "max_retries": 20,
            "retry_wait_sec": 10,
            "parameters": {"timeout": 21600},
        }
        assert parameters["num_trials"] == 1
        assert entry["parameters"]["user_llm_config"] == external_user

    def test_each_judge_gets_its_own_url(self):
        cfg = _make_gigametric_judged_cfg(
            judges={"judge-a": ["hle"], "judge-b": ["tau2"]}
        )
        cfg.judge_endpoints = judge_endpoints(
            {
                "judge-a": "http://a:31000",
                "judge-b": "http://b:31001",
            }
        )
        cfg.apply_defaults(
            inference_instances=1,
            base_url="http://localhost:29999",
        )

        assert cfg.judge_llm_config("judge-a")["api_url"] == "http://a:31000/v1"
        assert cfg.judge_llm_config("judge-b")["api_url"] == "http://b:31001/v1"

    def test_builds_openai_config_with_v1_suffix(self):
        # apply_defaults передаёт чистый base_url (без /v1) — метод конкатенирует /v1.
        cfg = _make_gigametric_judged_cfg()
        _apply_judged_defaults(cfg, "http://x:8000")
        judge = cfg.judge_llm_config("judge-a")
        assert judge["api_url"] == "http://x:8000/v1"
        assert judge["model"] == "openai"
        assert judge["name"] == "judge-a"
        assert judge["parameters"] == {"timeout": 21600}


class TestNodeRunnerConfigGigaMetricJudged:
    def test_loads_from_node_runner_config_and_applies_defaults(self, tmp_path: Path):
        config = _base_node_runner_config()
        config["autometrics"]["evaluators"] = {
            "gigametric_judged": {
                "base_dir": str(tmp_path / "runs"),
                "local_model_id": "local",
                "judges": {"pollux-judge-32b": ["hle"]},
                "validation_config": {
                    "benchmarks": {"hle": {"name": "hle-adapter"}},
                    "models": [
                        {
                            "id": "local",
                            "llm_config": {"model": "openai"},
                            "benchmarks": ["hle"],
                        }
                    ],
                },
            },
        }
        model = _load_dumped_model(tmp_path, config)
        judged = model.autometrics.evaluators.gigametric_judged
        assert judged is not None
        assert judged.secondary_model_names == ["pollux-judge-32b"]
        judged.judge_endpoints = judge_endpoints(
            {"pollux-judge-32b": "http://10.0.0.1:8000"}
        )
        judged.apply_defaults(
            inference_instances=2,
            base_url="http://localhost:29999",
        )
        llm_config = judged.validation_config["models"][0]["llm_config"]
        assert llm_config["semaphore"] == 2 * DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE


class TestEvalTaskConfigNumFewshot:
    @pytest.mark.parametrize(
        ("num_fewshot", "expected"),
        [
            (None, None),
            (0, 0),
            (5, 5),
            ([0, 5], [0, 5]),
            ([5, 0, 5], [5, 0]),
        ],
    )
    def test_accepted_values(self, num_fewshot, expected):
        task = LmEvalPretrainTaskConfig(num_fewshot=num_fewshot)
        assert task.num_fewshot == expected

    def test_empty_list_is_rejected(self):
        with pytest.raises(ValidationError, match="must not be an empty list"):
            LmEvalPretrainTaskConfig(num_fewshot=[])

    @pytest.mark.parametrize("num_fewshot", (-1, [0, -2]))
    def test_negative_values_are_rejected(self, num_fewshot):
        with pytest.raises(ValidationError):
            LmEvalPretrainTaskConfig(num_fewshot=num_fewshot)

    def test_mera_still_rejects_a_list(self):
        # MERA reuses EvalTaskConfig for CLI serialization but has no fan-out,
        # so a list would reach lm_eval as `--num_fewshot [0, 5]`.
        with pytest.raises(ValidationError):
            MeraTaskConfig(num_fewshot=[0, 5])

        assert MeraTaskConfig(num_fewshot=5).num_fewshot == 5


class TestGenerationsMode:
    @pytest.mark.parametrize(
        "field", ["record_generations_dir", "replay_generations_dir"]
    )
    def test_one_mode_at_a_time_is_fine(self, field):
        config = GlobalsConfig(**{field: "/tmp/generations"})

        assert getattr(config, field) == "/tmp/generations"

    def test_recording_a_replay_is_refused(self):
        with pytest.raises(ValidationError, match="mutually exclusive"):
            GlobalsConfig(
                record_generations_dir="/tmp/out", replay_generations_dir="/tmp/in"
            )

    @pytest.mark.parametrize(
        ("replay_dir", "replay_mode"), [("/tmp/generations", True), (None, False)]
    )
    def test_a_replay_is_what_makes_the_checkpoint_optional(
        self, replay_dir, replay_mode, tmp_path
    ):
        config = NodeRunnerConfigModel.model_validate(_measurement_config(replay_dir))

        assert config.is_replay_mode is replay_mode
        assert config.runs_without_checkpoint is replay_mode
        assert config.is_external_api_mode is False

    @pytest.mark.parametrize(
        "alias", ["record-generations-dir", "replay-generations-dir"]
    )
    def test_the_yaml_spelling_is_accepted(self, alias):
        config = GlobalsConfig.model_validate({alias: "/tmp/generations"})

        assert "/tmp/generations" in config.model_dump(by_alias=True).values()


def _measurement_config(replay_dir: str | None) -> dict:
    config_path = (
        Path(__file__).parent.parent / "test_data" / "configs" / "10b_test.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["user"] = "test_user"
    config["tensorboard_path"] = "/tmp/tb"
    if replay_dir is not None:
        config["autometrics"]["globals"]["replay-generations-dir"] = replay_dir
    return config
