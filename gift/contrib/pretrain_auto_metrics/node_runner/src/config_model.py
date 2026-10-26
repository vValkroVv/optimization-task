from __future__ import annotations

import copy
import json
import re
import sys
from base64 import b64decode
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, ClassVar, Iterator, Literal, Optional
from urllib.parse import urlsplit, urlunsplit

from omegaconf import OmegaConf
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)

from .logger import Logger
from .pam_proxies.models.sglang import SamplingParams

METRICS_FOR_BS_1 = (
    "longbench",
    "longbench2",
    "ruler",
)

SGLANG_DEFAULT_ROUTER_PORT = 30000
SGLANG_DEFAULT_DISAGGREGATION_BOOTSTRAP_PORT = 8998
SGLANG_ROUTER_PROMETHEUS_PORT = 29000

MEAN_VAR_INFERENCE_DEFAULTS = {
    "pp-size": 1,
    "dtype": "bfloat16",
    "is-embedding": True,
    "allow-auto-truncate": False,
    "chunked-prefill-size": -1,
    "disable-radix-cache": True,
    "disable-cuda-graph": True,
    "attention-backend": "triton",
}

MODEL_NAME_PARAMS_MAPPER: dict[str, dict[str, Any]] = {
    "meanvar": MEAN_VAR_INFERENCE_DEFAULTS,
    "gemma4-meanvar": {
        **MEAN_VAR_INFERENCE_DEFAULTS,
        "tp": 2,
        "dp": 4,
        "context-length": 98304,
        "max-running-requests": 12,
        "max-prefill-tokens": 16384,
        "max-total-tokens": 100352,
        "mem-fraction-static": 0.90,
    },
    "gpt-oss-120b": {
        "tp": 4,
        "pp-size": 1,
        "dp": 2,
        "trust-remote-code": True,
        "allow-auto-truncate": True,
        "max-running-requests": 512,
        "reasoning-parser": "gpt-oss",
        "tool-call-parser": "gpt-oss",
        "mem_fraction_static": 0.75,
        "context-length": 32768,
    },
    "deepseek-v3": {
        "tp": 16,
        "pp-size": 1,
        "ep": 16,
        "dp": 1,
        "trust-remote-code": True,
        "allow-auto-truncate": True,
        "max-running-requests": 1024,
        "dtype": "auto",
    },
    "deepseek-v3.1": {
        "tp": 16,
        "pp-size": 1,
        "ep": 16,
        "dp": 1,
        "trust-remote-code": True,
        "allow-auto-truncate": True,
        "max-running-requests": 1024,
    },
    "deepseek-v3.1-terminus": {
        "tp": 16,
        "pp-size": 1,
        "ep": 16,
        "dp": 1,
        "trust-remote-code": True,
        "allow-auto-truncate": True,
        "max-running-requests": 1024,
    },
    "minimax-m2.7": {
        "tp": 8,
        "pp-size": 1,
        "ep": 8,
        "dp": 1,
        "trust-remote-code": True,
        "allow-auto-truncate": True,
        "max-running-requests": 1024,
        "reasoning-parser": "minimax-append-think",
    },
    "Qwen3-235B-A22B-Instruct-2507-FP8": {
        "tp": 8,
        "pp-size": 1,
        "ep": 8,
        "dp": 1,
        "trust-remote-code": True,
        "allow-auto-truncate": True,
        "max-running-requests": 128,
        "tool-call-parser": "qwen",
        "mem_fraction_static": 0.8,
        "context-length": 131072,
    },
    "pollux-judge-32b": {
        "tp": 2,
        "pp-size": 1,
        "dp": 4,
        "trust-remote-code": True,
        "allow-auto-truncate": True,
        "max-running-requests": 1024,
        "context-length": 8192,
    },
    "Qwen3.5-397B-A17B-FP8": {
        "tp": 8,
        "pp-size": 1,
        "ep": 8,
        "trust-remote-code": True,
        "allow-auto-truncate": True,
        "max-running-requests": 1024,
        "reasoning-parser": "qwen3",
        "tool-call-parser": "qwen3_coder",
        "mem_fraction_static": 0.8,
        "context-length": 131072,
    },
}

EXTERNAL_API_PLATFORM_DEFAULTS: dict[str, dict[str, Any]] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "formatting": "openai",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "formatting": "openai",
    },
    "chatwm_openrouter": {
        "base_url": "https://chatwm.ru/f47db70ec76d1d333be922d9587fe825/openrouter/api",
        "formatting": "openai",
    },
    "ift": {
        "base_url": "https://gigachat.ift.sberdevices.ru",
        "formatting": "ift",
    },
    "prod": {
        "base_url": "https://gigachat.sberdevices.ru",
        "formatting": "ift",
    },
}
LMMS_EVAL_DEFAULT_VENV_PYTHON = "/home/user/lmms_eval/bin/python"
LMMS_EVAL_DEFAULT_VERSION_FILE = "/home/user/lmms_eval_version.txt"
LMMS_EVAL_DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE = 50
DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE = 128
LM_EVAL_PROCESS_RESULTS_WORKERS_DEFAULT = 16
LM_EVAL_PROCESS_RESULTS_WORKERS_TASKS = {
    "bigcodebench",
    "lcb_code_gen_lite",
    "lcb_code_gen_lite_instruct",
}
LM_EVAL_FILTER_WORKERS_DEFAULT = 16
LM_EVAL_FILTER_WORKERS_TASKS = {
    "lcb_code_gen_lite_32",
    "lcb_code_gen_lite_instruct_32",
    "cruxeval_32",
    "spellcheck_punctuation_ru_32",
}
BFCL_DEFAULT_MAX_RETRIES = 5
TAU2_DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE = 32
TERMINAL_BENCH_DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE = 8
HARBOR_RLI_DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE = 8
SAFETY_UNIFIED_DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE = 16
SAFETY_UNIFIED_DEFAULT_JUDGE_PARALLEL_PER_INFERENCE_INSTANCE = 8
ACC_LEN_DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE = 8
SWE_BENCH_DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE = 8
SKILLS_BENCH_DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE = 8
NL2REPO_DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE = 8
CLAWEVAL_DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE = 16
TOKEN_NLL_DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE = 16
CLAWEVAL_DEFAULT_JUDGE_PARALLEL_PER_INFERENCE_INSTANCE = 48
CLAWEVAL_DEFAULT_LLM_RETRIES = 20
DEFAULT_REQUEST_TIMEOUT = 21600


class BaseConfigModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True, exclude_none=True)

    def to_bash_list(self) -> list[str]:
        dump = self.model_dump(by_alias=True, exclude_none=True)
        bash_cmd_list = []
        for key, value in dump.items():
            if isinstance(value, bool):
                if value:
                    bash_cmd_list.append(f"--{key}")
            else:
                if isinstance(value, dict):
                    value = json.dumps(value)
                else:
                    value = str(value)
                bash_cmd_list.append(f"--{key}")
                bash_cmd_list.append(value)

        return bash_cmd_list


class ModelInfoConfig(BaseConfigModel):
    model_class: str
    size: str
    username: str


@dataclass(frozen=True)
class JudgeEndpoint:
    base_url: str
    instances: int


class MetricsEvaluatorConfig(BaseConfigModel):
    judge_endpoints: dict[str, JudgeEndpoint] = Field(
        default_factory=dict, exclude=True
    )

    def apply_defaults(
        self,
        *,
        inference_instances: int,
        base_url: str,
        **_: Any,
    ) -> None:
        pass

    def require_base_url(self) -> str:
        base_url = getattr(self, "base_url", None)
        if base_url is None:
            raise ValueError(
                f"{type(self).__name__}.base_url must be set via "
                "apply_defaults() before run"
            )
        return base_url

    @property
    def secondary_model_names(self) -> list[str]:
        return []


class JudgedEvaluatorConfig(MetricsEvaluatorConfig):
    judges: dict[str, list[str]]

    @field_validator("judges", mode="before")
    @classmethod
    def _normalize_judges(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        if not value:
            raise ValueError("judges must not be empty")
        if any(not name.strip() for name in value):
            raise ValueError("judges must contain non-empty model names")
        return {
            judge: [tasks] if isinstance(tasks, str) else tasks
            for judge, tasks in value.items()
        }

    @property
    def secondary_model_names(self) -> list[str]:
        return list(self.judges)


class DreddJudgeConfig(BaseConfigModel):
    model_name: str
    parallel_api_calls: int | None = Field(default=None, ge=1)
    request_timeout: int = Field(default=DEFAULT_REQUEST_TIMEOUT, gt=0)
    max_tokens: int = Field(default=16384, ge=1)
    api_base: str | None = None
    pam_mode: bool = True


class DreddModelConfig(BaseConfigModel):
    parallel_api_calls: int | None = Field(default=None, ge=1)
    request_timeout: int = Field(default=DEFAULT_REQUEST_TIMEOUT, gt=0)
    api_base: str | None = None


class DreddBaselineConfig(BaseConfigModel):
    name: str | None = None


class DreddDatasetConfig(BaseConfigModel):
    name: str
    baseline: DreddBaselineConfig | None = None


class DreddEvaluatorConfig(MetricsEvaluatorConfig):
    judge: DreddJudgeConfig
    model: DreddModelConfig = Field(default_factory=DreddModelConfig)
    baseline: DreddBaselineConfig | None = None
    datasets: list[DreddDatasetConfig] | None = None

    def apply_defaults(
        self,
        *,
        inference_instances: int,
        base_url: str,
        **_: Any,
    ) -> None:
        if self.model.parallel_api_calls is None:
            self.model.parallel_api_calls = (
                inference_instances * DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE
            )
        if self.model.api_base is None:
            self.model.api_base = f"{base_url}/v1"
        judge = self.judge_endpoints[self.judge.model_name]
        if self.judge.parallel_api_calls is None:
            self.judge.parallel_api_calls = (
                judge.instances * DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE
            )
        if self.judge.api_base is None:
            self.judge.api_base = f"{judge.base_url}/v1"

    @property
    def secondary_model_names(self) -> list[str]:
        return [self.judge.model_name]


class GlobalsConfig(BaseConfigModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    dist_init_path: str = "/etc/volcano/mpimaster.host"
    master_num_file_path: str = "/etc/volcano/VC_MPIMASTER_NUM"
    worker_num_file_path: str = "/etc/volcano/VC_MPIWORKER_NUM"
    orkestration_port: int = Field(default=61574, ge=1, le=65535)
    delete_converted_hf: bool = False
    full_metrics_recalculate: bool = False
    send_metrics_at_end: bool = True
    user_api_port: int = Field(default=8000, ge=1, le=65535)
    record_generations_dir: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "record-generations-dir",
            "record_generations_dir",
        ),
        serialization_alias="record-generations-dir",
    )
    replay_generations_dir: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "replay-generations-dir",
            "replay_generations_dir",
        ),
        serialization_alias="replay-generations-dir",
    )

    @model_validator(mode="after")
    def _validate_generations_mode(self) -> "GlobalsConfig":
        if self.record_generations_dir is not None and (
            self.replay_generations_dir is not None
        ):
            raise ValueError(
                "record-generations-dir and replay-generations-dir are mutually "
                "exclusive: there is nothing to record when answers are replayed"
            )
        return self


RouterPolicy = Literal[
    "random",
    "round_robin",
    "cache_aware",
    "power_of_two",
    "bucket",
    "manual",
    "consistent_hashing",
    "prefix_hash",
]


class RouterConfig(BaseConfigModel):
    host: str = "0.0.0.0"
    port: int = Field(default=SGLANG_DEFAULT_ROUTER_PORT, ge=1, le=65535)
    policy: RouterPolicy = "round_robin"
    prefill_policy: RouterPolicy | None = Field(
        default=None,
        validation_alias=AliasChoices("prefill-policy", "prefill_policy"),
        serialization_alias="prefill-policy",
    )
    decode_policy: RouterPolicy | None = Field(
        default=None,
        validation_alias=AliasChoices("decode-policy", "decode_policy"),
        serialization_alias="decode-policy",
    )
    max_concurrent_requests: int | None = Field(
        default=16384,
        ge=-1,
        validation_alias=AliasChoices(
            "max-concurrent-requests",
            "max_concurrent_requests",
        ),
        serialization_alias="max-concurrent-requests",
    )
    queue_size: int | None = Field(
        default=32768,
        ge=1,
        validation_alias=AliasChoices("queue-size", "queue_size"),
        serialization_alias="queue-size",
    )
    request_timeout_secs: int = Field(
        default=DEFAULT_REQUEST_TIMEOUT,
        gt=0,
        validation_alias=AliasChoices(
            "request-timeout-secs",
            "request_timeout_secs",
        ),
        serialization_alias="request-timeout-secs",
    )
    queue_timeout_secs: int = Field(
        default=DEFAULT_REQUEST_TIMEOUT,
        gt=0,
        validation_alias=AliasChoices("queue-timeout-secs", "queue_timeout_secs"),
        serialization_alias="queue-timeout-secs",
    )
    health_check_timeout_secs: int = Field(
        default=25,
        gt=0,
        validation_alias=AliasChoices(
            "health-check-timeout-secs",
            "health_check_timeout_secs",
        ),
        serialization_alias="health-check-timeout-secs",
    )
    health_failure_threshold: int = Field(
        default=7,
        gt=0,
        validation_alias=AliasChoices(
            "health-failure-threshold",
            "health_failure_threshold",
        ),
        serialization_alias="health-failure-threshold",
    )
    health_success_threshold: int = Field(
        default=1,
        gt=0,
        validation_alias=AliasChoices(
            "health-success-threshold",
            "health_success_threshold",
        ),
        serialization_alias="health-success-threshold",
    )
    prometheus_port: int = Field(
        default=SGLANG_ROUTER_PROMETHEUS_PORT,
        ge=1,
        le=65535,
        validation_alias=AliasChoices("prometheus-port", "prometheus_port"),
        serialization_alias="prometheus-port",
    )


class InferenceConfig(BaseConfigModel):
    engine_type: str = Field(
        default="sglang",
        validation_alias=AliasChoices(
            "engine_type",
            "engine-type",
            "inference_engine",
            "inference-engine",
        ),
        exclude=True,
    )

    @property
    def uses_external_api(self) -> bool:
        return False

    @property
    def has_reasoning_parser(self) -> bool:
        return False

    @property
    def has_tool_call_parser(self) -> bool:
        return False

    @property
    def has_chat_template(self) -> bool:
        return False


class SglangInferenceConfig(InferenceConfig):
    engine_type: Literal["sglang"] = Field(
        default="sglang",
        validation_alias=AliasChoices(
            "engine_type",
            "engine-type",
            "inference_engine",
            "inference-engine",
        ),
        exclude=True,
    )
    tp: int = Field(
        default=1,
        ge=1,
        validation_alias=AliasChoices("tp", "tp_size", "tp-size"),
        serialization_alias="tp-size",
    )
    pp: int = Field(
        default=1,
        ge=1,
        validation_alias=AliasChoices("pp", "pp_size", "pp-size"),
        serialization_alias="pp-size",
    )
    ep: int = Field(
        default=1,
        ge=1,
        validation_alias=AliasChoices("ep", "ep_size", "ep-size"),
        serialization_alias="ep-size",
    )
    dp: int = Field(
        default=1,
        ge=1,
        validation_alias=AliasChoices("dp", "dp_size", "dp-size"),
        serialization_alias="dp-size",
    )
    sglang_communication_port: int = Field(
        default=20000,
        ge=1,
        le=65535,
        validation_alias=AliasChoices(
            "sglang_communication_port",
            "sglang-communication-port",
        ),
        exclude=True,
    )
    dtype: str = "auto"
    reasoning_parser: str | None = Field(
        default=None,
        validation_alias=AliasChoices("reasoning-parser", "reasoning_parser"),
        serialization_alias="reasoning-parser",
    )
    tool_call_parser: str | None = Field(
        default=None,
        validation_alias=AliasChoices("tool-call-parser", "tool_call_parser"),
        serialization_alias="tool-call-parser",
    )
    mem_fraction_static: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("mem-fraction-static", "mem_fraction_static"),
        serialization_alias="mem-fraction-static",
    )
    chunked_prefill_size: int = Field(
        default=8192,
        ge=-1,
        validation_alias=AliasChoices("chunked-prefill-size", "chunked_prefill_size"),
        serialization_alias="chunked-prefill-size",
    )
    max_running_requests: int | None = Field(
        default=1024,
        ge=1,
        validation_alias=AliasChoices("max-running-requests", "max_running_requests"),
        serialization_alias="max-running-requests",
    )
    max_queued_requests: int | None = Field(
        default=8192,
        ge=1,
        validation_alias=AliasChoices("max-queued-requests", "max_queued_requests"),
        serialization_alias="max-queued-requests",
    )
    trust_remote_code: bool = Field(
        default=True,
        validation_alias=AliasChoices("trust-remote-code", "trust_remote_code"),
        serialization_alias="trust-remote-code",
    )
    enable_metrics: bool = Field(
        default=True,
        validation_alias=AliasChoices("enable-metrics", "enable_metrics"),
        serialization_alias="enable-metrics",
    )
    allow_auto_truncate: bool = Field(
        default=True,
        validation_alias=AliasChoices("allow-auto-truncate", "allow_auto_truncate"),
        serialization_alias="allow-auto-truncate",
    )
    chat_template: str | None = Field(
        default=None,
        validation_alias=AliasChoices("chat-template", "chat_template"),
        serialization_alias="chat-template",
    )
    tokenizer_path: str | None = Field(
        default=None,
        validation_alias=AliasChoices("tokenizer-path", "tokenizer_path"),
        serialization_alias="tokenizer-path",
    )
    gated_norm_impl: Literal["native", "torch_compile", "triton"] | None = Field(
        default=None,
        validation_alias=AliasChoices("gated-norm-impl", "gated_norm_impl"),
        serialization_alias="gated-norm-impl",
    )
    moe_runner_backend: str | None = Field(
        default=None,
        validation_alias=AliasChoices("moe-runner-backend", "moe_runner_backend"),
        serialization_alias="moe-runner-backend",
    )
    processor_path: str | None = Field(
        default=None,
        exclude=True,
        validation_alias=AliasChoices("processor-path", "processor_path"),
    )
    processor_internal_path: str = Field(
        default="processors/GigaVisionProcessor",
        exclude=True,
        validation_alias=AliasChoices("processor-name", "processor_internal_path"),
        serialization_alias="processor-name",
    )
    moe_a2a_backend: str | None = Field(
        default=None,
        validation_alias=AliasChoices("moe-a2a-backend", "moe_a2a_backend"),
        serialization_alias="moe-a2a-backend",
    )
    speculative_algorithm: str | None = Field(
        default=None,
        validation_alias=AliasChoices("speculative-algorithm", "speculative_algorithm"),
        serialization_alias="speculative-algorithm",
    )
    speculative_num_steps: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices("speculative-num-steps", "speculative_num_steps"),
        serialization_alias="speculative-num-steps",
    )
    speculative_eagle_topk: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices(
            "speculative-eagle-topk", "speculative_eagle_topk"
        ),
        serialization_alias="speculative-eagle-topk",
    )
    speculative_num_draft_tokens: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices(
            "speculative-num-draft-tokens", "speculative_num_draft_tokens"
        ),
        serialization_alias="speculative-num-draft-tokens",
    )
    auto_mtp: bool = Field(
        default=False,
        validation_alias=AliasChoices("auto-mtp", "auto_mtp"),
        exclude=True,
    )
    disaggregation_mode: Literal["prefill", "decode"] | None = Field(
        default=None,
        validation_alias=AliasChoices("disaggregation-mode", "disaggregation_mode"),
        serialization_alias="disaggregation-mode",
    )
    disaggregation_bootstrap_port: int | None = Field(
        default=None,
        ge=1,
        le=65535,
        validation_alias=AliasChoices(
            "disaggregation-bootstrap-port", "disaggregation_bootstrap_port"
        ),
        serialization_alias="disaggregation-bootstrap-port",
    )

    schedule_policy: Literal["fcfs", "lof"] = Field(
        default="fcfs",
        validation_alias=AliasChoices("schedule-policy", "schedule_policy"),
        serialization_alias="schedule-policy",
    )
    enable_priority_scheduling: Literal[True] = Field(
        default=True,
        validation_alias=AliasChoices(
            "enable-priority-scheduling", "enable_priority_scheduling"
        ),
        serialization_alias="enable-priority-scheduling",
    )
    disable_priority_preemption: Literal[True] = Field(
        default=True,
        validation_alias=AliasChoices(
            "disable-priority-preemption", "disable_priority_preemption"
        ),
        serialization_alias="disable-priority-preemption",
    )
    schedule_low_priority_values_first: Literal[False] = Field(
        default=False,
        validation_alias=AliasChoices(
            "schedule-low-priority-values-first", "schedule_low_priority_values_first"
        ),
        serialization_alias="schedule-low-priority-values-first",
    )
    default_priority_value: Literal[0] = Field(
        default=0,
        validation_alias=AliasChoices(
            "default-priority-value", "default_priority_value"
        ),
        serialization_alias="default-priority-value",
    )

    @classmethod
    def for_secondary_preset(
        cls,
        model_name: str,
        secondary_inference: dict[str, dict[str, Any]],
    ) -> "SglangInferenceConfig | SglangPdInferenceConfig":
        if (
            model_name not in MODEL_NAME_PARAMS_MAPPER
            and model_name not in secondary_inference
        ):
            raise ValueError(
                f"unknown SGLang preset {model_name!r}; "
                "add it to MODEL_NAME_PARAMS_MAPPER"
            )
        preset = MODEL_NAME_PARAMS_MAPPER.get(model_name, {})
        overrides = secondary_inference.get(model_name, {})
        if SglangPdInferenceConfig.wants_pd(overrides):
            return SglangPdInferenceConfig.model_validate(
                {
                    **overrides,
                    "shared_params": {
                        **preset,
                        **(
                            overrides.get("shared_params")
                            or overrides.get("shared-params")
                            or {}
                        ),
                    },
                }
            )
        return cls.model_validate({**preset, **overrides})

    @property
    def has_reasoning_parser(self) -> bool:
        return self.reasoning_parser is not None

    @property
    def has_tool_call_parser(self) -> bool:
        return self.tool_call_parser is not None

    @property
    def has_chat_template(self) -> bool:
        return self.chat_template is not None


class SglangPdInferenceConfig(InferenceConfig):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    engine_type: Literal["sglang_pd"] = Field(
        default="sglang_pd",
        validation_alias=AliasChoices(
            "engine_type",
            "engine-type",
            "inference_engine",
            "inference-engine",
        ),
        exclude=True,
    )
    prefill: SglangInferenceConfig
    decode: SglangInferenceConfig
    prefill_fraction: float = Field(
        default=0.5,
        gt=0.0,
        lt=1.0,
        validation_alias=AliasChoices("prefill_fraction", "prefill-fraction"),
        exclude=True,
    )

    @staticmethod
    def wants_pd(params: dict) -> bool:
        sides = {"prefill", "decode"} & params.keys()
        if len(sides) == 1:
            raise ValueError(
                f"PD inference needs both prefill and decode, got only {sides.pop()}"
            )
        return bool(sides)

    @model_validator(mode="before")
    @classmethod
    def _apply_shared_params(cls, value: Any) -> Any:
        if (
            not isinstance(value, dict)
            or "prefill" not in value
            or "decode" not in value
        ):
            return value
        shared = value.get("shared_params") or value.get("shared-params")
        stripped = {
            key: param
            for key, param in value.items()
            if key not in ("shared_params", "shared-params")
        }
        if not shared:
            return stripped
        return {
            **stripped,
            "prefill": {**shared, **(value["prefill"] or {})},
            "decode": {**shared, **(value["decode"] or {})},
        }

    @model_validator(mode="after")
    def _apply_disaggregation_roles(self) -> "SglangPdInferenceConfig":
        self.prefill.disaggregation_mode = "prefill"
        self.decode.disaggregation_mode = "decode"
        if self.prefill.disaggregation_bootstrap_port is None:
            self.prefill.disaggregation_bootstrap_port = (
                SGLANG_DEFAULT_DISAGGREGATION_BOOTSTRAP_PORT
            )
        if self.prefill.tokenizer_path != self.decode.tokenizer_path:
            raise ValueError(
                "prefill and decode must share tokenizer_path; "
                "set it once via shared_params"
            )
        return self

    @property
    def tokenizer_path(self) -> str | None:
        return self.decode.tokenizer_path

    @tokenizer_path.setter
    def tokenizer_path(self, value: str | None) -> None:
        self.prefill.tokenizer_path = value
        self.decode.tokenizer_path = value

    @property
    def has_reasoning_parser(self) -> bool:
        return self.decode.has_reasoning_parser

    @property
    def has_tool_call_parser(self) -> bool:
        return self.decode.has_tool_call_parser

    @property
    def has_chat_template(self) -> bool:
        return self.decode.has_chat_template


class ExternalApiInferenceConfig(InferenceConfig):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    REQUEST_CUSTOMIZATION_FORBIDDEN_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "assistant_id",
            "messages",
            "metadata",
            "model",
            "stream",
            "thread_id",
            "tools",
        }
    )
    BASE_URL_API_SUFFIXES: ClassVar[tuple[str, ...]] = (
        "/v1/chat/completions",
        "/v2/chat/completions",
        "/v1",
        "/v2",
    )

    engine_type: Literal["external_api"] = Field(
        default="external_api",
        validation_alias=AliasChoices(
            "engine_type",
            "engine-type",
            "inference_engine",
            "inference-engine",
        ),
        exclude=True,
    )
    formatting: Literal["openai", "ift"] = "openai"
    port: int = Field(
        default=SGLANG_DEFAULT_ROUTER_PORT,
        ge=1,
        le=65535,
        validation_alias=AliasChoices("port", "port"),
        serialization_alias="port",
    )
    base_url: str = Field(
        validation_alias=AliasChoices("base_url", "base-url"),
        serialization_alias="base-url",
    )
    model: str
    api_key: str | None = Field(
        default=None,
        exclude=True,
        validation_alias=AliasChoices("api_key", "api-key"),
        serialization_alias="api-key",
    )
    user_pass: str | None = Field(
        default=None,
        exclude=True,
        validation_alias=AliasChoices("user_pass", "user-pass"),
        serialization_alias="user-pass",
    )
    timeout: float = Field(default=DEFAULT_REQUEST_TIMEOUT, gt=0)
    logging_level: Literal["debug", "info", "warning", "error", "critical"] = Field(
        default="info",
        validation_alias=AliasChoices("logging-level", "logging_level"),
        serialization_alias="logging-level",
    )
    max_retries: int = Field(
        default=5,
        ge=0,
        validation_alias=AliasChoices("max_retries", "max-retries"),
        serialization_alias="max-retries",
    )
    retry_wait_sec: float = Field(
        default=1.0,
        gt=0,
        validation_alias=AliasChoices("retry_wait_sec", "retry-wait-sec"),
        serialization_alias="retry-wait-sec",
    )
    max_concurrency: int = Field(
        default=10,
        ge=1,
        validation_alias=AliasChoices("max_concurrency", "max-concurrency"),
        serialization_alias="max-concurrency",
    )
    request_customization: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices(
            "request_customization",
            "request-customization",
            "request_customisation",
            "request-customisation",
        ),
        serialization_alias="request-customization",
    )

    @classmethod
    def for_secondary_platform(
        cls,
        platform: str,
        model: str,
        secondary_inference: dict[str, dict[str, Any]],
    ) -> "ExternalApiInferenceConfig":
        if (
            platform not in EXTERNAL_API_PLATFORM_DEFAULTS
            and platform not in secondary_inference
        ):
            raise ValueError(
                f"unknown external API platform {platform!r}; "
                f"known platforms: {sorted(EXTERNAL_API_PLATFORM_DEFAULTS)}"
            )
        merged: dict[str, Any] = {
            **EXTERNAL_API_PLATFORM_DEFAULTS.get(platform, {}),
            **secondary_inference.get(platform, {}),
            "model": model,
        }
        return cls.model_validate(merged)

    @model_validator(mode="after")
    def _normalize_base_url(self) -> "ExternalApiInferenceConfig":
        self.base_url = self._without_api_suffix(self.base_url.rstrip("/"))
        return self

    @property
    def uses_external_api(self) -> bool:
        return True

    @model_validator(mode="after")
    def _validate_auth(self) -> "ExternalApiInferenceConfig":
        has_api_key = self._has_secret(self.api_key)
        has_user_pass = self._has_secret(self.user_pass)

        if self.formatting == "openai":
            if not has_api_key:
                raise ValueError(
                    "api_key is required for "
                    "external_api with openai formatting and must not be '...'"
                )
            if has_user_pass:
                raise ValueError(
                    "user_pass is supported only for external_api with ift formatting"
                )
            return self

        if not has_api_key and not has_user_pass:
            raise ValueError(
                "api_key or user_pass "
                "is required for external_api with ift formatting and must not be '...'"
            )
        return self

    @staticmethod
    def _has_secret(value: str | None) -> bool:
        return value is not None and value.strip() not in ("", "...")

    def _without_api_suffix(self, value: str) -> str:
        parsed = urlsplit(value)
        path = parsed.path.rstrip("/")
        for suffix in self.BASE_URL_API_SUFFIXES:
            if path == suffix:
                return urlunsplit(parsed._replace(path=""))
            if path.endswith(suffix):
                return urlunsplit(parsed._replace(path=path[: -len(suffix)]))

        return value

    @field_validator("user_pass")
    @classmethod
    def _validate_user_pass(cls, value: str | None) -> str | None:
        if value is None or value.strip() in ("", "..."):
            return value

        user, separator, password = value.partition(":")
        invalid_user = user.strip() in ("", "...")
        invalid_password = password.strip() in ("", "...")
        if separator == "" or invalid_user or invalid_password:
            raise ValueError(
                "user_pass must be in USER:PASS format and must not be '...:...'"
            )
        return value

    @model_validator(mode="after")
    def _validate_request_customization(self) -> "ExternalApiInferenceConfig":
        forbidden_keys = (
            self.REQUEST_CUSTOMIZATION_FORBIDDEN_KEYS
            & self.request_customization.keys()
        )
        if forbidden_keys:
            forbidden = ", ".join(sorted(forbidden_keys))
            raise ValueError(
                "request_customization must not override "
                f"protected request keys: {forbidden}"
            )
        return self


InferenceConfigUnion = (
    SglangInferenceConfig | SglangPdInferenceConfig | ExternalApiInferenceConfig
)


class PreparationConfig(BaseConfigModel):
    skip: bool = False
    pair_action: Literal["keep-ema", "keep-nonema", "drop-both", "none"] = Field(
        default="keep-nonema",
        validation_alias=AliasChoices("pair-action", "pair_action"),
        serialization_alias="pair-action",
    )
    cast_bf16: bool = Field(
        default=True,
        validation_alias=AliasChoices("cast-bf16", "cast_bf16"),
        serialization_alias="cast-bf16",
    )
    dry_run: bool = Field(
        default=False,
        validation_alias=AliasChoices("dry-run", "dry_run"),
        serialization_alias="dry-run",
    )
    drop_tokenizer_merges: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "drop-tokenizer-merges",
            "drop_tokenizer_merges",
        ),
        serialization_alias="drop-tokenizer-merges",
    )
    drop_tokenizer_vocab: bool = Field(
        default=True,
        validation_alias=AliasChoices("drop-tokenizer-vocab", "drop_tokenizer_vocab"),
        serialization_alias="drop-tokenizer-vocab",
    )
    split_into_two: bool = Field(
        default=True,
        validation_alias=AliasChoices("split-into-two", "split_into_two"),
        serialization_alias="split-into-two",
    )


class PreQuantPaddingConfig(BaseConfigModel):
    inplace: bool = True
    enabled: bool = True
    only_if_mlp_intermediate_sizes: list[int] | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "only_if_mlp_intermediate_sizes",
            "only-if-mlp-intermediate-sizes",
            "only_if_mlp_intermediate_size",
            "only-if-mlp-intermediate-size",
        ),
        serialization_alias="only-if-mlp-intermediate-sizes",
    )
    multiple_of: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices("multiple_of", "multiple-of"),
        serialization_alias="multiple-of",
    )
    update_config: bool = Field(
        default=True,
        validation_alias=AliasChoices("update_config", "update-config"),
        serialization_alias="update-config",
    )


class QuantizationConfig(BaseConfigModel):
    skip: bool | None = Field(default=None, exclude=True)
    script_path: str | None = None
    config_path: str | None = None
    output_checkpoint_name: str = "hf_checkpoint_fp8"
    input_checkpoint_path: str | None = None
    output_checkpoint_path: str | None = None
    pre_quant_padding: PreQuantPaddingConfig = Field(
        default_factory=PreQuantPaddingConfig,
        validation_alias=AliasChoices("pre_quant_padding", "pre-quant-padding"),
        serialization_alias="pre-quant-padding",
    )
    device_map: str = "auto"
    quantization_config: dict[str, Any] = Field(
        default_factory=lambda: {
            "weight_block_size": [128, 128],
            "modules_to_not_convert": [],
        }
    )


class ConvertationConfig(BaseConfigModel):
    mappers: list[str] = Field(default_factory=list)
    input_tp_size: int = Field(default=1, ge=1)
    output_tp_size: int = Field(default=1, ge=1)
    skip_optimizer: bool = False
    print_final_model_structure: bool = True
    chunk_layers: int = Field(default=5, ge=1)
    use_ema: bool = True
    input_checkpoint_path: str | None = None
    output_checkpoint_path: str | None = None
    tokenizer_path: str | None = None
    hf_config_path: str | None = None


class ObmSamplingParams(SamplingParams):
    model_config = ConfigDict(extra="forbid")

    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    min_tokens: int | None = Field(default=None, ge=0)
    seed: int | None = None
    sampling_seed: int | None = None


class ObmProxyConfig(BaseConfigModel):
    sampling_params: ObmSamplingParams = Field(
        default_factory=ObmSamplingParams,
        validation_alias=AliasChoices("sampling-params", "sampling_params"),
        serialization_alias="sampling-params",
    )
    workers: int = Field(default=16, ge=1)
    logging_level: Literal["debug", "info", "warning", "error", "critical"] = Field(
        default="info",
        validation_alias=AliasChoices("logging-level", "logging_level"),
        serialization_alias="logging-level",
    )
    sglang_port: int = Field(
        default=SGLANG_DEFAULT_ROUTER_PORT,
        ge=1,
        le=65535,
        validation_alias=AliasChoices("sglang-port", "sglang_port"),
        serialization_alias="sglang-port",
    )
    api_port: int = Field(
        default=29999,
        ge=1,
        le=65535,
        validation_alias=AliasChoices("api-port", "api_port"),
        serialization_alias="api-port",
    )
    chat_handlers_mode: Literal["opensource", "custom", "pretrain"] | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "chat-handlers-mode",
            "chat_handlers_mode",
        ),
        serialization_alias="chat-handlers-mode",
    )
    chat_template: str | None = Field(
        default=None,
        validation_alias=AliasChoices("chat-template", "chat_template"),
        serialization_alias="chat-template",
    )
    chat_template_kwargs: dict[str, Any] = Field(
        default_factory=lambda: {},
        validation_alias=AliasChoices("chat-template-kwargs", "chat_template_kwargs"),
        serialization_alias="chat-template-kwargs",
    )
    reasoning_token_budget: int = Field(
        default=0,
        ge=0,
        validation_alias=AliasChoices(
            "reasoning-token-budget",
            "reasoning_token_budget",
        ),
        serialization_alias="reasoning-token-budget",
    )
    template_version: str | None = Field(
        default=None,
        validation_alias=AliasChoices("template-version", "template_version"),
        serialization_alias="template-version",
    )
    tokenizer_path: str | None = Field(
        default=None,
        validation_alias=AliasChoices("tokenizer-path", "tokenizer_path"),
        serialization_alias="tokenizer-path",
    )

    @field_serializer("sampling_params")
    def serialize_sampling_params(self, value: ObmSamplingParams) -> dict[str, Any]:
        return value.model_dump(exclude_unset=True)

    @staticmethod
    def _check_if_encoded_and_decode(content: str | bytes) -> bytes | None:
        try:
            if isinstance(content, bytes):
                normalized = content.strip()
            else:
                normalized = content.strip().encode("ascii")
            decoded = b64decode(normalized, validate=True)
            decoded.decode("utf-8")
            return decoded
        except Exception:
            return None

    def decode_chat_template_to_ram(
        self,
        *,
        is_slurm: bool,
        ram_path: str = "/home/user/chat_template.jinja",
    ) -> str | None:
        if not self.chat_template:
            return None

        source_path = Path(self.chat_template)
        if not source_path.exists():
            return None

        if is_slurm:
            template_content = source_path.read_text(encoding="utf-8")
            decoded_template = self._check_if_encoded_and_decode(template_content)
            if not decoded_template:
                raise ValueError(
                    "Chat template is not encoded in base64, "
                    "please encode it in base64 AND remove unencoded jinja from NFS"
                )

            target_path = Path(ram_path).absolute()
            target_path.write_text(decoded_template.decode("utf-8"), encoding="utf-8")
            self.chat_template = target_path.as_posix()
            return self.chat_template

        self.chat_template = source_path.absolute().as_posix()
        return self.chat_template

    def decode_default_devsystem(self, *, is_slurm: bool) -> None:
        if not is_slurm:
            return

        default_devsystem = self.chat_template_kwargs.get("default_devsystem")
        if not default_devsystem:
            return

        decoded_default_devsystem = self._check_if_encoded_and_decode(default_devsystem)
        if not decoded_default_devsystem:
            raise ValueError(
                "Default devsystem is not encoded in base64, please encode it in base64"
            )
        self.chat_template_kwargs["default_devsystem"] = (
            decoded_default_devsystem.decode("utf-8")
        )


# OBM START
class ObmAllConfig(BaseConfigModel):
    groups: Optional[list[str]] = None
    tasks: Optional[list[str]] = None
    ignore_tasks: Optional[list[str]] = None
    restart: bool = False
    context: Optional[str] = None
    disable_few_shots: bool = False
    disable_suggested_params: bool = False
    n_workers: int = 10

    @model_validator(mode="after")
    def _validate_groups_or_tasks(self) -> "ObmAllConfig":
        if self.groups is None and self.tasks is None:
            raise ValueError("At least one of 'groups' or 'tasks' must be specified")
        return self


def _openai_obm_shooter(
    base_url: str | None,
    timeout: float,
    model: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "openai": {
            "model": model,
            "keys": ["empty"],
            "limit": {"rpm": 1_000_000_000, "tpm": 1_000_000_000_000},
            "params": params or {},
            "api": {
                "suppress_error": False,
                "retries_count": 20,
                "retry_min_wait": 5,
                "retry_max_wait": 60,
                "timeout": timeout,
                "base_url": f"{base_url}/v1",
            },
        }
    }


class BaseObmEvaluatorConfig(MetricsEvaluatorConfig):
    obm_config: str
    task_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)
    shooter_params: dict[str, Any] = Field(default_factory=dict)
    base_url: str | None = Field(default=None, exclude=True)
    timeout: float = Field(default=DEFAULT_REQUEST_TIMEOUT, gt=0, exclude=True)
    n_threads: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices("n-threads", "n_threads"),
        serialization_alias="n-threads",
    )
    all: ObmAllConfig

    @computed_field
    @property
    def shooter_config(self) -> dict[str, Any]:
        raise NotImplementedError

    @computed_field
    @property
    def runner_config(self) -> dict[str, Any]:
        return {
            **self.task_overrides,
            "benchmark": {
                "runner": {
                    "shooter": self.shooter_config,
                },
            },
            "n_threads": self.n_threads,
            "quota": {
                "resource": {
                    "shooter": self.n_threads,
                },
            },
            "logging": {
                "enabled": True,
                "file": "obm.log",
                "level": "INFO",
            },
        }

    def apply_defaults(
        self,
        *,
        inference_instances: int,
        base_url: str,
        **_: Any,
    ) -> None:
        if self.n_threads is None:
            self.n_threads = (
                inference_instances * DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE
            )
        self.base_url = base_url


class OpenaiObmEvaluatorConfig(BaseObmEvaluatorConfig):
    obm_config: Literal["openai"] = "openai"

    @computed_field
    @property
    def shooter_config(self) -> dict[str, Any]:
        return _openai_obm_shooter(
            self.base_url,
            self.timeout,
            "<model>",
            self.shooter_params,
        )


class DevicesObmEvaluatorConfig(BaseObmEvaluatorConfig):
    obm_config: Literal["devices"] = "devices"

    @computed_field
    @property
    def shooter_config(self) -> dict[str, Any]:
        return {
            "devices": {
                "model": "<model>",
                "need_auth": False,
                "limit": {"rpm": 1_000_000_000, "tpm": 1_000_000_000_000},
                "params": self.shooter_params,
                "api": {
                    "suppress_error": False,
                    "retries_count": 20,
                    "retry_min_wait": 5,
                    "retry_max_wait": 60,
                    "timeout": self.timeout,
                    "base_url": self.base_url,
                    "route_chat": "/v1/chat/completions",
                },
            }
        }


class DevicesV2ObmEvaluatorConfig(BaseObmEvaluatorConfig):
    obm_config: Literal["devices-v2"] = "devices-v2"

    @computed_field
    @property
    def shooter_config(self) -> dict[str, Any]:
        return {
            "devices": {
                "model": "<model>",
                "need_auth": False,
                "limit": {"rpm": 1_000_000_000, "tpm": 1_000_000_000_000},
                "params": self.shooter_params,
                "api": {
                    "base_url": self.base_url,
                    "route_chat": "/v2/chat/completions",
                    "version": "v2",
                    "suppress_error": False,
                    "retries_count": 20,
                    "retry_min_wait": 5,
                    "retry_max_wait": 60,
                    "timeout": self.timeout,
                },
            }
        }


ObmEvaluatorConfig = Annotated[
    OpenaiObmEvaluatorConfig | DevicesObmEvaluatorConfig | DevicesV2ObmEvaluatorConfig,
    Field(discriminator="obm_config"),
]


class BaseObmJudgedEvaluatorConfig(JudgedEvaluatorConfig, BaseObmEvaluatorConfig):
    @model_validator(mode="after")
    def _validate_judges_do_not_share_tasks(self) -> "BaseObmJudgedEvaluatorConfig":
        seen: dict[str, str] = {}
        for judge, tasks in self.judges.items():
            for task in tasks:
                if task in seen:
                    raise ValueError(
                        "autometrics.evaluators.obm_judged.judges: task "
                        f"'{task}' assigned to both '{seen[task]}' and '{judge}'; "
                        "one task must be judged by a single model"
                    )
                seen[task] = judge
        return self

    def judged_runner_config(self, judge_model_name: str) -> dict[str, Any]:
        cfg = self.runner_config
        judge_base_url = self.judge_endpoints[judge_model_name].base_url
        judge_shooter = _openai_obm_shooter(judge_base_url, self.timeout, "<judge>")
        cfg["benchmark"]["runner"]["llm_judge_shooter"] = judge_shooter
        cfg["quota"]["resource"]["llm_judge_shooter"] = self.n_threads
        return cfg


class OpenaiObmJudgedEvaluatorConfig(
    BaseObmJudgedEvaluatorConfig, OpenaiObmEvaluatorConfig
):
    obm_config: Literal["openai"] = "openai"


class DevicesObmJudgedEvaluatorConfig(
    BaseObmJudgedEvaluatorConfig, DevicesObmEvaluatorConfig
):
    obm_config: Literal["devices"] = "devices"


ObmJudgedEvaluatorConfig = Annotated[
    OpenaiObmJudgedEvaluatorConfig | DevicesObmJudgedEvaluatorConfig,
    Field(discriminator="obm_config"),
]
# OBM END


# LM EVAL START
class EvalTaskConfig(BaseConfigModel):
    trust_remote_code: bool = True
    confirm_run_unsafe_code: bool = True
    model_args: dict[str, Any] = Field(default_factory=dict)
    num_fewshot: (
        Annotated[int, Field(ge=0)] | list[Annotated[int, Field(ge=0)]] | None
    ) = None
    repeats: int | None = Field(default=None, ge=1)
    process_results_workers: int = Field(default=1, ge=1)
    filter_workers: int = Field(default=1, ge=1)
    metadata: str | None = None
    log_samples: bool = False

    @model_validator(mode="before")
    @classmethod
    def normalize_empty_task(cls, value: Any) -> Any:
        if value is None or value == "none":
            return {}
        return value

    @field_validator("num_fewshot")
    @classmethod
    def dedupe_num_fewshot(
        cls, value: int | list[int] | None
    ) -> int | list[int] | None:
        if not isinstance(value, list):
            return value
        if not value:
            raise ValueError("num_fewshot must not be an empty list")
        return list(dict.fromkeys(value))

    @property
    def model_args_cli(self) -> str:
        return ",".join(f"{key}={value}" for key, value in self.model_args.items())

    def to_bash_list(self) -> list[str]:
        dump = self.model_dump(by_alias=True, exclude_none=True)
        model_args = dump.pop("model_args", None)
        bash_cmd_list = []
        for key, value in dump.items():
            if isinstance(value, bool):
                if value:
                    bash_cmd_list.append(f"--{key}")
            else:
                if isinstance(value, dict):
                    value = json.dumps(value)
                else:
                    value = str(value)
                bash_cmd_list.append(f"--{key}")
                bash_cmd_list.append(value)

        if model_args:
            bash_cmd_list.append("--model_args")
            bash_cmd_list.append(self.model_args_cli)

        return bash_cmd_list


class LmEvalPretrainTaskConfig(EvalTaskConfig):
    batch_size: int = 1
    model: str = "sglang-generate"

    def apply_defaults(
        self,
        default_parallel: int,
        max_length: int,
        hf_checkpoint_path: Path,
        generate_url: str,
    ) -> None:
        self.model_args.setdefault("base_url", generate_url)
        self.model_args.setdefault("num_concurrent", default_parallel)
        self.model_args.setdefault("tokenized_requests", True)
        self.model_args.setdefault(
            "tokenizer", hf_checkpoint_path.absolute().as_posix()
        )
        self.model_args.setdefault("max_length", max_length - 32)
        self.model_args.setdefault("timeout", DEFAULT_REQUEST_TIMEOUT)
        self.model_args.setdefault("max_retries", 20)


class LmEvalInstructTaskConfig(EvalTaskConfig):
    model: str = "local-chat-completions"
    batch_size: int = 1
    apply_chat_template: bool = True

    def apply_defaults(
        self,
        *,
        default_parallel: int,
        chat_completions_url: str,
    ) -> None:
        self.model_args.setdefault("base_url", chat_completions_url)
        self.model_args.setdefault("num_concurrent", default_parallel)
        self.model_args.setdefault("timeout", DEFAULT_REQUEST_TIMEOUT)
        self.model_args.setdefault("max_retries", 20)


class LmEvalEvaluatorConfig(MetricsEvaluatorConfig):
    pool_max_parallel: int = Field(default=10, ge=1, exclude=True)
    restart: bool = Field(default=False, exclude=True)
    pretrain: dict[str, LmEvalPretrainTaskConfig] = Field(default_factory=dict)
    instruct: dict[str, LmEvalInstructTaskConfig] = Field(default_factory=dict)

    @staticmethod
    def _apply_task_name_defaults(
        task_name: str,
        task_cfg: EvalTaskConfig,
    ) -> None:
        if task_name in LM_EVAL_PROCESS_RESULTS_WORKERS_TASKS:
            task_cfg.process_results_workers = LM_EVAL_PROCESS_RESULTS_WORKERS_DEFAULT
        if task_name in LM_EVAL_FILTER_WORKERS_TASKS:
            task_cfg.filter_workers = LM_EVAL_FILTER_WORKERS_DEFAULT

    def apply_defaults(
        self,
        *,
        inference_instances: int,
        max_seq_len: int,
        hf_checkpoint_path: Path,
        base_url: str,
        **_: Any,
    ) -> None:
        default_parallel = inference_instances * DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE
        generate_url = f"{base_url}/generate"
        chat_completions_url = f"{base_url}/v1/chat/completions"

        for task_name, task_cfg in self.pretrain.items():
            task_cfg.apply_defaults(
                default_parallel=default_parallel,
                max_length=max_seq_len,
                hf_checkpoint_path=hf_checkpoint_path,
                generate_url=generate_url,
            )
            self._apply_task_name_defaults(task_name, task_cfg)

        for task_name, task_cfg in self.instruct.items():
            task_cfg.apply_defaults(
                default_parallel=default_parallel,
                chat_completions_url=chat_completions_url,
            )
            self._apply_task_name_defaults(task_name, task_cfg)


# LM EVAL END


# MERA START
class MeraTaskConfig(LmEvalInstructTaskConfig):
    # No fan-out over a shots list here, so a list would reach lm_eval as
    # `--num_fewshot [0, 5]`; keep rejecting it at config validation.
    num_fewshot: int | None = Field(default=None, ge=0)
    process_results_workers: int = Field(default=1, ge=1, exclude=True)
    filter_workers: int = Field(default=1, ge=1, exclude=True)
    include_path: str = "/home/user/mera_tasks"
    device: str = "cpu"
    log_samples: bool = True
    gen_kwargs: str = '{"until": []}'
    fewshot_as_multiturn: bool = True
    system_instruction: str | None = None


class MeraEvaluatorConfig(MetricsEvaluatorConfig):
    system_instruction: str | None = Field(default=None, exclude=True)
    pool_max_parallel: int = Field(default=10, ge=1, exclude=True)
    tasks: dict[str, MeraTaskConfig] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def wrap_flat_mapping(cls, value: Any) -> Any:
        if value is None:
            return {}
        if not isinstance(value, dict):
            return value
        if "tasks" in value:
            data = dict(value)
        else:
            tasks = dict(value)
            data = {
                "system_instruction": tasks.pop("system_instruction", None),
                "tasks": tasks,
            }
            if "pool_max_parallel" in tasks:
                data["pool_max_parallel"] = tasks.pop("pool_max_parallel")
        if data.get("system_instruction") == "none":
            data["system_instruction"] = None
        return data

    def apply_defaults(
        self,
        *,
        inference_instances: int,
        base_url: str,
        **_: Any,
    ) -> None:
        default_parallel = inference_instances * DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE
        chat_completions_url = f"{base_url}/v1/chat/completions"
        for _, task_cfg in self.tasks.items():
            task_cfg.apply_defaults(
                default_parallel=default_parallel,
                chat_completions_url=chat_completions_url,
            )


# MERA END


class LmmsEvalBenchmarksConfig(BaseConfigModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    mode: Literal["grouped", "sequential"] = "sequential"
    tasks: list[str] | None = None
    categories: list[str] | None = None
    limit: int | None = Field(default=None, ge=1)


class LmmsEvalRunConfig(BaseConfigModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    benchmarks: LmmsEvalBenchmarksConfig | None = None
    gigavision: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LmmsEvalRuntimeTarget:
    """Resolved GigaVision model backend for the active PAM inference runtime."""

    model_source: Literal["api", "lmms_eval"]
    model_overrides: dict[str, Any]

    @classmethod
    def from_inference_config(
        cls,
        inference_config: InferenceConfig,
        *,
        chatml_version: Literal["v4", "v4.1", "v5"] | None,
        base_url: str,
        direct_base_url: str,
        inference_instances: int,
        timeout: float,
    ) -> "LmmsEvalRuntimeTarget":
        if isinstance(
            inference_config,
            (SglangInferenceConfig, SglangPdInferenceConfig),
        ):
            if chatml_version is None:
                raise ValueError(
                    "lmms_eval.chatml_version is required for local inference"
                )
            return cls(
                model_source="api",
                model_overrides={
                    "chatml_version": chatml_version,
                    "api": {
                        "shooter": "obm_proxy",
                        "obm_proxy": {
                            "parallel": (
                                inference_instances
                                * LMMS_EVAL_DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE
                            ),
                            "api": {
                                "base_url": base_url,
                                "timeout": timeout,
                            },
                        },
                    },
                },
            )

        if isinstance(inference_config, ExternalApiInferenceConfig):
            model_args = ",".join(
                [
                    f"model={inference_config.model}",
                    f"base_url={direct_base_url}/v1",
                    "api_key=unused",
                    f"num_cpus={inference_config.max_concurrency}",
                    f"timeout={timeout}",
                    "max_retries=1",
                    "retry_backoff_s=0",
                ]
            )
            return cls(
                model_source="lmms_eval",
                model_overrides={
                    "lmms_eval": {
                        "model": "async_openai",
                        "model_args": model_args,
                    },
                },
            )

        raise TypeError(
            "lmms_eval does not support inference config "
            f"{type(inference_config).__name__}"
        )

    def apply(self, config: dict[str, Any]) -> None:
        """Apply this target to a generated GigaVision config."""
        config["model_source"] = self.model_source
        config["model"].update(copy.deepcopy(self.model_overrides))


class LmmsEvalEvaluatorConfig(MetricsEvaluatorConfig):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    CONFIGS_DIRNAME: ClassVar[Path] = Path("lmms_eval_configs")

    chatml_version: Literal["v4", "v4.1", "v5"] | None = None
    timeout: float = Field(default=DEFAULT_REQUEST_TIMEOUT, gt=0)
    benchmarks: LmmsEvalBenchmarksConfig | None = None
    runs: dict[str, LmmsEvalRunConfig] = Field(default_factory=dict)
    gigavision: dict[str, Any] = Field(default_factory=dict)
    lmms_eval_venv_python: str = Field(default=LMMS_EVAL_DEFAULT_VENV_PYTHON)
    lmms_eval_version_file: str = Field(default=LMMS_EVAL_DEFAULT_VERSION_FILE)
    _runtime_target: LmmsEvalRuntimeTarget | None = PrivateAttr(default=None)

    @property
    def resolved_runs(self) -> dict[str, LmmsEvalRunConfig]:
        if self.runs:
            return self.runs
        return {
            "default": LmmsEvalRunConfig(
                benchmarks=self.benchmarks,
            )
        }

    def apply_defaults(
        self,
        *,
        inference_instances: int,
        base_url: str,
        direct_base_url: str,
        inference_config: InferenceConfig,
        **_: Any,
    ) -> None:
        self._runtime_target = LmmsEvalRuntimeTarget.from_inference_config(
            inference_config,
            chatml_version=self.chatml_version,
            base_url=base_url,
            direct_base_url=direct_base_url,
            inference_instances=inference_instances,
            timeout=self.timeout,
        )

    def require_runtime_target(self) -> LmmsEvalRuntimeTarget:
        if self._runtime_target is None:
            raise ValueError(
                "LmmsEvalEvaluatorConfig runtime target must be set via "
                "apply_defaults() before run"
            )
        return self._runtime_target


class Tau2SamplingConfig(BaseConfigModel):
    temperature: float = 1.0
    top_p: float = 0.7
    reasoning_effort: str | None = None
    timeout: float = Field(default=DEFAULT_REQUEST_TIMEOUT, gt=0)
    num_retries: int = Field(default=20, ge=0)

    def to_llm_args(self, api_base: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "api_base": api_base,
            "api_key": "None",
            "timeout": self.timeout,
            "num_retries": self.num_retries,
        }
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
            payload["allowed_openai_params"] = ["reasoning_effort"]
        return payload


class Tau2EvaluatorConfig(MetricsEvaluatorConfig):
    VALID_DOMAINS: ClassVar[frozenset[str]] = frozenset(
        {"airline", "retail", "telecom"}
    )
    _CLI_FLAG_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "agent_llm",
            "user_llm",
            "nl_assertions_llm",
            "num_trials",
            "max_concurrency",
        }
    )

    domains: list[str] = Field(min_length=1)
    user_model_name: str = Field(exclude=True)
    num_trials: int = Field(default=1, ge=1, alias="num-trials")
    max_concurrency: int | None = Field(default=None, ge=1, alias="max-concurrency")
    agent_llm: str = Field(default="openai/my_agent_model", alias="agent-llm")
    user_llm: str = Field(default="openai/my_user_model", alias="user-llm")
    nl_assertions_llm: str = Field(
        default="openai/my_user_model", alias="nl-assertions-llm"
    )
    agent_llm_args: Tau2SamplingConfig = Field(
        default_factory=Tau2SamplingConfig, alias="agent-llm-args"
    )
    user_llm_args: Tau2SamplingConfig = Field(
        default_factory=lambda: Tau2SamplingConfig(
            temperature=0.7, top_p=0.8, reasoning_effort="high"
        ),
        alias="user-llm-args",
    )
    venv_python: str = Field(
        default="/home/user/tau2_eval/bin/python",
        exclude=True,
    )
    tau2_bench_root: str = Field(default="/home/user/tau2-bench", exclude=True)
    max_parallel_domains: int = Field(default=3, ge=1, exclude=True)
    agent_api_base: str | None = Field(default=None, exclude=True)
    user_api_base: str | None = Field(default=None, exclude=True)

    @field_validator("domains")
    @classmethod
    def _validate_domains(cls, domains: list[str]) -> list[str]:
        invalid = [d for d in domains if d not in cls.VALID_DOMAINS]
        if invalid:
            raise ValueError(
                f"Invalid tau2 domain(s) {invalid}. "
                f"Must be one of {sorted(cls.VALID_DOMAINS)}"
            )
        return domains

    def apply_defaults(
        self,
        *,
        inference_instances: int,
        base_url: str,
        **_: Any,
    ) -> None:
        self.agent_api_base = f"{base_url}/v1"
        user_model = self.judge_endpoints[self.user_model_name]
        self.user_api_base = f"{user_model.base_url}/v1"
        if self.max_concurrency is None:
            self.max_concurrency = (
                inference_instances * TAU2_DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE
            )

    @property
    def secondary_model_names(self) -> list[str]:
        if not self.user_model_name.strip():
            raise ValueError(
                "autometrics.evaluators.tau2.user_model_name must not be empty"
            )
        return [self.user_model_name]

    def build_cli_run(self, *, domain: str, save_to: str) -> list[str]:
        if self.agent_api_base is None:
            raise ValueError(
                "agent_api_base must be set via apply_defaults() before build_cli_run()"
            )
        if self.user_api_base is None:
            raise ValueError(
                "user_api_base must be set via apply_defaults() before build_cli_run()"
            )
        agent_json = json.dumps(self.agent_llm_args.to_llm_args(self.agent_api_base))
        user_json = json.dumps(self.user_llm_args.to_llm_args(self.user_api_base))
        bash: list[str] = []
        for key, value in self.model_dump(
            by_alias=True,
            exclude_none=True,
            include=self._CLI_FLAG_FIELDS,
        ).items():
            bash.extend([f"--{key}", str(value)])
        bash.extend(
            [
                "--agent-llm-args",
                agent_json,
                "--user-llm-args",
                user_json,
                "--nl-assertions-llm-args",
                user_json,
                "--domain",
                domain,
                "--save-to",
                save_to,
            ]
        )
        return bash


class Tau3EvaluatorConfig(Tau2EvaluatorConfig):
    VALID_DOMAINS: ClassVar[frozenset[str]] = frozenset(
        {"airline", "retail", "telecom", "banking_knowledge"}
    )

    domains: list[str] = Field(
        default_factory=lambda: ["airline", "retail", "telecom", "banking_knowledge"],
        min_length=1,
    )
    user_model_name: str = Field(
        default="Qwen3-235B-A22B-Instruct-2507-FP8", exclude=True
    )
    retrieval_config: str = Field(default="bm25", alias="retrieval-config")
    max_parallel_domains: int = Field(default=4, ge=1, exclude=True)

    def build_cli_run(self, *, domain: str, save_to: str) -> list[str]:
        bash = super().build_cli_run(domain=domain, save_to=save_to)
        if domain == "banking_knowledge":
            bash.extend(["--retrieval-config", self.retrieval_config])
        return bash


class ForgeEvaluatorConfig(MetricsEvaluatorConfig):
    attack_model_name: str = "served"
    attack_api_base: str | None = Field(default=None, exclude=True)
    temperature: float | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    timeout: float | None = Field(default=DEFAULT_REQUEST_TIMEOUT, gt=0)
    judge: str
    judge_prompt_template: str | None = None
    attacks: list[str] | None = None
    groups: list[dict[str, Any]] = Field(default_factory=list)
    workers: int | None = Field(default=None, ge=1)
    recipe_workers: int = Field(default=5, ge=1)
    limit: int | None = Field(default=None, ge=1)
    combinations: int | None = Field(default=None, ge=1)
    save_cache: bool = False
    fail_on_runtime_error: bool = True

    runtime_python: str = Field(
        default="/home/user/forge_lite_eval/bin/python", exclude=True
    )

    @field_validator("attacks", mode="before")
    @classmethod
    def _normalize_attacks(cls, value: Any) -> Any:
        if isinstance(value, str) and value.strip().lower() == "all":
            return None
        return value

    @model_validator(mode="after")
    def _validate_groups(self) -> "ForgeEvaluatorConfig":
        seen: set[str] = set()
        for group in self.groups:
            for recipe in group.get("recipes", []):
                if recipe in seen:
                    raise ValueError(
                        f"recipe '{recipe}' appears in multiple forge groups"
                    )
                seen.add(recipe)
        return self

    def apply_defaults(
        self,
        *,
        inference_instances: int,
        base_url: str,
        **_: Any,
    ) -> None:
        self.attack_api_base = f"{base_url}/v1"
        if self.workers is None:
            self.workers = inference_instances * DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE

    @property
    def secondary_model_names(self) -> list[str]:
        names = [self.judge]
        for group in self.groups:
            group_judge = group.get("judge")
            if group_judge and group_judge not in names:
                names.append(group_judge)
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError(
                "autometrics.evaluators.forge judge model names must not be empty"
            )
        return names

    def _group_for(self, recipe: str) -> dict[str, Any] | None:
        for group in self.groups:
            if recipe in group.get("recipes", []):
                return group
        return None

    def judge_for(self, recipe: str) -> str:
        group = self._group_for(recipe)
        if group is not None and group.get("judge"):
            return group["judge"]
        return self.judge

    def build_meta_override(self, recipe: str) -> dict[str, Any]:
        """Build the ``model`` / ``judge`` sections to merge into a recipe's
        ``meta.json``.
        """
        judge = self.judge_for(recipe)
        judge_api_base = f"{self.judge_endpoints[judge].base_url}/v1"
        group = self._group_for(recipe)

        def _resolve(field: str) -> Any:
            if group is not None and group.get(field) is not None:
                return group[field]
            return getattr(self, field)

        temperature = _resolve("temperature")
        max_tokens = _resolve("max_tokens")
        timeout = _resolve("timeout")
        prompt_template = _resolve("judge_prompt_template")

        model_section: dict[str, Any] = {
            "provider": "openai",
            "model": self.attack_model_name,
            "base_url": self.attack_api_base,
            "api_key_env": "EMPTY",
        }
        if temperature is not None:
            model_section["temperature"] = temperature
        if max_tokens is not None:
            model_section["max_tokens"] = max_tokens
        if timeout is not None:
            model_section["timeout"] = timeout

        judge_section: dict[str, Any] = {
            "provider": "openai",
            "model": judge,
            "base_url": judge_api_base,
            "api_key_env": "EMPTY",
        }
        if prompt_template is not None:
            judge_section["prompt_template"] = prompt_template
        if timeout is not None:
            judge_section["timeout"] = timeout

        return {"model": model_section, "judge": judge_section}


class BfclEvaluatorConfig(MetricsEvaluatorConfig):
    _PIPELINE_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"base_url", "venv_python", "bfcl_bench_root"}
    )
    _GENERATE_EXCLUDE_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"partial_eval", "score_dir"}
    )
    _EVALUATE_ONLY_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"test_category", "result_dir", "score_dir", "partial_eval"}
    )

    model: str
    test_category: str = Field(
        validation_alias=AliasChoices("test-category", "test_category"),
        serialization_alias="test-category",
    )
    skip_server_setup: bool = Field(default=True, alias="skip-server-setup")
    include_input_log: bool = Field(default=True, alias="include-input-log")
    num_threads: int | None = Field(default=None, ge=1, alias="num-threads")
    web_search: str = Field(default="sber", alias="web-search")
    allow_overwrite: bool = Field(default=False, alias="allow-overwrite")
    repair_json_max_iter: int = Field(default=5, ge=0, alias="repair-json-max-iter")
    temperature: float = Field(default=0.001, alias="temperature")
    partial_eval: bool = Field(default=False, alias="partial-eval")
    exclude_state_log: bool | None = Field(default=None, alias="exclude-state-log")
    num_gpus: int | None = Field(default=None, ge=1, alias="num-gpus")
    gpu_memory_utilization: float | None = Field(
        default=None, alias="gpu-memory-utilization"
    )
    backend: str | None = Field(default=None, alias="backend")
    local_model_path: str | None = Field(default=None, alias="local-model-path")
    run_ids: bool | None = Field(default=None, alias="run-ids")
    enable_lora: bool | None = Field(default=None, alias="enable-lora")
    max_lora_rank: int | None = Field(default=None, alias="max-lora-rank")
    lora_modules: list[str] | str | None = Field(default=None, alias="lora-modules")
    result_dir: str | None = Field(default=None, alias="result-dir")
    score_dir: str | None = Field(default=None, alias="score-dir")

    base_url: str | None = Field(default=None, exclude=True)
    timeout: float = Field(default=DEFAULT_REQUEST_TIMEOUT, gt=0, exclude=True)
    max_retries: int = Field(default=BFCL_DEFAULT_MAX_RETRIES, ge=0, exclude=True)
    venv_python: str = Field(
        default="/home/user/bfcl_eval/bin/python",
        exclude=True,
    )
    bfcl_bench_root: str = Field(
        default="contrib/gorilla/berkeley-function-call-leaderboard",
        exclude=True,
    )
    num_runs: int = Field(default=1, ge=1, alias="num-runs", exclude=True)
    max_parallel_runs: int = Field(
        default=1, ge=1, alias="max-parallel-runs", exclude=True
    )
    pass_at_k: list[int] = Field(default_factory=list, alias="pass-at-k", exclude=True)

    @field_validator("test_category", mode="before")
    @classmethod
    def _normalize_test_category(cls, value: Any) -> str:
        if isinstance(value, list):
            return ",".join(str(v).strip() for v in value if str(v).strip())
        return str(value).strip()

    @field_validator("model", mode="before")
    @classmethod
    def _normalize_model(cls, value: Any) -> str:
        if isinstance(value, list):
            return ",".join(str(v).strip() for v in value if str(v).strip())
        return str(value).strip()

    @model_validator(mode="after")
    def _validate_pass_at_k(self) -> "BfclEvaluatorConfig":
        for k in self.pass_at_k:
            if k < 1 or k > self.num_runs:
                raise ValueError(
                    f"pass_at_k values must be between 1 and num_runs ({self.num_runs}), got {k}"
                )
        return self

    def apply_defaults(
        self,
        *,
        inference_instances: int,
        base_url: str,
        **_: Any,
    ) -> None:
        if self.num_threads is None:
            self.num_threads = (
                inference_instances * DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE
            )
        self.base_url = f"{base_url}/v1"

    def to_bash_list(
        self,
        *,
        exclude: frozenset[str] | None = None,
        only: frozenset[str] | None = None,
    ) -> list[str]:
        skip_fields = self._PIPELINE_FIELDS | (exclude or frozenset())
        dump_kwargs: dict[str, Any] = {
            "by_alias": True,
            "exclude_none": True,
            "exclude": skip_fields,
        }
        if only is not None:
            dump_kwargs["include"] = only
        dump = self.model_dump(**dump_kwargs)

        bash_cmd_list: list[str] = []
        for key, value in dump.items():
            if key == "lora-modules" and isinstance(value, list):
                for item in value:
                    bash_cmd_list.extend(["--lora-modules", str(item)])
                continue
            if isinstance(value, bool):
                if value:
                    bash_cmd_list.append(f"--{key}")
            else:
                if isinstance(value, dict):
                    value = json.dumps(value)
                else:
                    value = str(value)
                bash_cmd_list.append(f"--{key}")
                bash_cmd_list.append(value)
        return bash_cmd_list

    def to_generate_bash_list(self, result_dir: Path) -> list[str]:
        cfg = self.model_copy(update={"result_dir": str(result_dir.resolve())})
        return cfg.to_bash_list(exclude=self._GENERATE_EXCLUDE_FIELDS)

    def to_evaluate_bash_list(self, result_dir: Path, score_dir: Path) -> list[str]:
        cfg = self.model_copy(
            update={
                "result_dir": str(result_dir.resolve()),
                "score_dir": str(score_dir.resolve()),
            }
        )
        return cfg.to_bash_list(only=self._EVALUATE_ONLY_FIELDS)


class GigaMetricEvaluatorConfig(MetricsEvaluatorConfig):
    _ANONYMIZED_ORG_NAME: ClassVar[str] = "papaya"
    _INTERNAL_ORG_NAME: ClassVar[str] = b64decode(b"c2Jlcg==").decode("utf-8")

    validation_config_path: Optional[Path] = None
    validation_config: Optional[dict[str, Any]] = None
    base_dir: Optional[Path] = None
    env: dict[str, str] = Field(default_factory=dict)
    runtime_python: str = "/home/user/papayametric_venv/bin/python"
    shooter_version: str = "image-baked"
    local_model_id: Optional[str] = None

    benchmarks: dict[str, dict[str, Any]] = Field(default_factory=dict)
    benchmark_catalog: Literal["pam", "gigametric"] = "gigametric"
    max_concurrent_requests: Optional[int] = Field(default=None, ge=1)
    inference_base_url: Optional[str] = Field(default=None, exclude=True)
    inference_instances: int = Field(default=1, ge=1, exclude=True)

    @field_validator("benchmarks", mode="before")
    @classmethod
    def _normalize_empty_benchmarks(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                benchmark: {} if override is None else override
                for benchmark, override in value.items()
            }
        return value

    @model_validator(mode="after")
    def _resolve_config_source(self) -> "GigaMetricEvaluatorConfig":
        has_path = self.validation_config_path is not None
        has_inline = self.validation_config is not None
        has_minimal = bool(self.benchmarks)
        if has_path and has_inline:
            raise ValueError(
                "gigametric: validation_config_path and validation_config are mutually exclusive"
            )
        if (has_path or has_inline) and has_minimal:
            raise ValueError(
                "gigametric: `benchmarks` is the minimal mode and can't be combined with "
                "validation_config / validation_config_path"
            )
        if not has_path and not has_inline and not has_minimal:
            raise ValueError(
                "gigametric: set `benchmarks` (minimal mode, e.g. {mmlu_pro: {}}) "
                "or one of validation_config / validation_config_path"
            )
        return self

    @model_validator(mode="after")
    def _reject_internal_benchmark_names(self) -> "GigaMetricEvaluatorConfig":
        if self._contains_internal_org_name(self.benchmarks):
            raise ValueError(
                "gigametric.benchmarks: replace internal names with PAPAYA aliases"
            )
        return self

    @classmethod
    def _contains_internal_org_name(cls, value: Any) -> bool:
        if isinstance(value, str):
            return cls._INTERNAL_ORG_NAME in value.casefold()
        if isinstance(value, dict):
            return any(
                cls._contains_internal_org_name(key)
                or cls._contains_internal_org_name(item)
                for key, item in value.items()
            )
        if isinstance(value, (list, tuple, set)):
            return any(cls._contains_internal_org_name(item) for item in value)
        return False

    @classmethod
    def _internal_preset_name(cls, preset: str) -> str:
        return preset.replace(cls._ANONYMIZED_ORG_NAME, cls._INTERNAL_ORG_NAME)

    @model_validator(mode="after")
    def _default_stream_off(self) -> "GigaMetricEvaluatorConfig":
        for model_entry in (self.validation_config or {}).get("models") or []:
            if isinstance(model_entry, dict) and isinstance(
                model_entry.get("llm_config"), dict
            ):
                model_entry["llm_config"].setdefault("parameters", {}).setdefault(
                    "stream", False
                )
        return self

    def apply_defaults(
        self,
        *,
        inference_instances: int,
        base_url: str,
        **_: Any,
    ) -> None:
        self.inference_base_url = base_url
        self.inference_instances = inference_instances
        if self.validation_config is None or self.local_model_id is None:
            return
        for model_entry in self.validation_config.get("models", []) or []:
            if not isinstance(model_entry, dict):
                continue
            if model_entry.get("id") != self.local_model_id:
                continue
            llm_config = model_entry.get("llm_config")
            if not isinstance(llm_config, dict):
                continue
            if "semaphore" not in llm_config:
                llm_config["semaphore"] = (
                    inference_instances * DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE
                )

    def _build_minimal_validation_config(
        self, *, example_full: Path, model_name: str
    ) -> dict[str, Any]:
        parallel_budget = (
            self.inference_instances * DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE
        )
        benchmark_concurrency = self.max_concurrent_requests
        if benchmark_concurrency is None:
            benchmark_concurrency = round(parallel_budget / 8)

        if not example_full.is_file():
            Logger.error(
                f"GigaMetric benchmark defaults not found at {example_full}; "
                "configured benchmarks will be skipped"
            )
            base: dict[str, Any] = {}
        else:
            loaded = OmegaConf.to_container(
                OmegaConf.create(
                    example_full.read_text(encoding="utf-8").replace(
                        self._ANONYMIZED_ORG_NAME,
                        self._INTERNAL_ORG_NAME,
                    )
                ),
                resolve=False,
            )
            if isinstance(loaded, dict):
                base = loaded
            else:
                Logger.error(
                    "GigaMetric benchmark defaults are not a mapping at "
                    f"{example_full}; configured benchmarks will be skipped"
                )
                base = {}

        defaults = base.get("defaults") or {}
        llm_defaults = defaults.get("llm_config") or {}
        llm_defaults["max_retries"] = 20
        llm_defaults["retry_wait_sec"] = 10
        llm_parameters = llm_defaults.get("parameters") or {}
        llm_parameters["timeout"] = 21600
        llm_defaults["parameters"] = llm_parameters
        defaults["llm_config"] = llm_defaults
        benchmark_defaults = defaults.get("benchmark_config") or {}
        benchmark_defaults["timeout"] = 21600
        benchmark_defaults.pop("judge_llm_config", None)
        parameters = benchmark_defaults.get("parameters") or {}
        parameters["max_concurrent_requests"] = benchmark_concurrency
        benchmark_defaults["parameters"] = parameters
        defaults["benchmark_config"] = benchmark_defaults
        base["defaults"] = defaults

        raw_catalog = base.get("benchmarks") or {}
        if isinstance(raw_catalog, dict):
            catalog = raw_catalog
        else:
            Logger.error(
                f"GigaMetric benchmark catalog is not a mapping in {example_full}; "
                "configured benchmarks will be skipped"
            )
            catalog = {}

        presets: dict[str, str] = {}
        for benchmark_id, override in self.benchmarks.items():
            public_preset = override.get("preset", benchmark_id)
            if not isinstance(public_preset, str):
                Logger.warning(
                    f"GigaMetric benchmark {benchmark_id!r} skipped: "
                    "preset must be a string"
                )
                continue
            preset = self._internal_preset_name(public_preset)
            if preset not in catalog:
                Logger.warning(
                    f"GigaMetric benchmark {benchmark_id!r} skipped: preset "
                    f"{public_preset!r} is absent from {example_full}"
                )
                continue
            presets[benchmark_id] = preset

        benchmarks: dict[str, Any] = {}
        for key, override in self.benchmarks.items():
            if key not in presets:
                continue
            entry = copy.deepcopy(catalog[presets[key]])
            entry_parameters = entry.get("parameters")
            if isinstance(entry_parameters, dict):
                entry_parameters.pop("max_concurrent_requests", None)

            for field, value in override.items():
                if field == "preset":
                    continue
                if field in {
                    "name",
                    "primary_metric",
                    "aggregation",
                    "credentials",
                    "parameters",
                }:
                    target = entry.get(field)
                    if isinstance(value, dict) and isinstance(target, dict):
                        entry[field] = OmegaConf.to_container(
                            OmegaConf.merge(target, value), resolve=False
                        )
                    else:
                        entry[field] = value
                else:
                    if not isinstance(entry.get("parameters"), dict):
                        entry["parameters"] = {}
                    entry["parameters"][field] = value
            benchmarks[key] = entry
        base["benchmarks"] = benchmarks

        if self.inference_base_url is None:
            raise ValueError(
                "inference_base_url must be set via apply_defaults() before "
                "materialising a minimal GigaMetric config"
            )
        inference_base_url = self.inference_base_url
        local_model_id = self.local_model_id or model_name
        self.local_model_id = local_model_id
        base["models"] = [
            {
                "id": local_model_id,
                "llm_config": {
                    "model": "openai",
                    "name": model_name,
                    "api_url": f"{inference_base_url.rstrip('/')}/v1",
                    "api_key": "not-needed",
                    "parameters": {"stream": False},
                    "semaphore": parallel_budget,
                },
                "benchmarks": list(benchmarks),
            }
        ]
        base.pop("model_refs", None)
        return base

    def materialise_validation_config(
        self, *, target: Path, example_full: Path, model_name: str
    ) -> Path:
        if self.validation_config_path is not None:
            return self.validation_config_path.expanduser().resolve()
        if self.validation_config is None:
            self.validation_config = self._build_minimal_validation_config(
                example_full=example_full,
                model_name=model_name,
            )
        defaults = self.validation_config.get("defaults") or {}
        llm_defaults = defaults.get("llm_config") or {}
        llm_defaults.setdefault("max_retries", 20)
        llm_defaults.setdefault("retry_wait_sec", 10)
        llm_parameters = llm_defaults.get("parameters") or {}
        llm_parameters.setdefault("timeout", 21600)
        llm_defaults["parameters"] = llm_parameters
        defaults["llm_config"] = llm_defaults
        benchmark_defaults = defaults.get("benchmark_config") or {}
        benchmark_defaults.setdefault("timeout", 21600)
        defaults["benchmark_config"] = benchmark_defaults
        self.validation_config["defaults"] = defaults
        OmegaConf.save(self.validation_config, target, resolve=False)
        return target


class PlanbenchEvaluatorConfig(MetricsEvaluatorConfig):
    engine: str = "planbench_chat"
    task: Literal["t1"] = "t1"
    domains: list[str] = Field(default_factory=lambda: ["blocksworld"], min_length=1)

    base_url: str | None = Field(default=None, exclude=True)
    venv_python: str = Field(
        default="/home/user/planbench_eval/bin/python",
        exclude=True,
    )
    planbench_root: str = Field(default="/home/user/planbench", exclude=True)
    max_parallel_domains: int = Field(default=1, ge=1, exclude=True)

    @field_validator("engine")
    @classmethod
    def _validate_engine(cls, engine: str) -> str:
        if not engine.endswith("_chat"):
            raise ValueError("engine must end with '_chat'")
        return engine

    def apply_defaults(
        self,
        *,
        inference_instances: int,
        base_url: str,
        **_: Any,
    ) -> None:
        self.base_url = f"{base_url}/v1"
        self.max_parallel_domains = min(inference_instances, len(self.domains))

    def build_cli_args(self, *, domain: str, work_dir: Path) -> list[str]:
        return [
            "--work-dir",
            str(work_dir.resolve()),
            "--task",
            self.task,
            "--config",
            domain,
            "--engine",
            self.engine,
        ]


class GigaMetricJudgedEvaluatorConfig(JudgedEvaluatorConfig, GigaMetricEvaluatorConfig):
    @model_validator(mode="after")
    def _reject_internal_judge_task_names(self) -> "GigaMetricJudgedEvaluatorConfig":
        if self._contains_internal_org_name(self.judges):
            raise ValueError(
                "gigametric_judged.judges: replace internal names with PAPAYA aliases"
            )
        return self

    def judge_llm_config(self, judge_model_name: str) -> dict[str, Any]:
        judge_base_url = self.judge_endpoints[judge_model_name].base_url
        return {
            "model": "openai",
            "name": judge_model_name,
            "api_url": f"{judge_base_url}/v1",
            "api_key": "not-needed",
            "max_retries": 20,
            "retry_wait_sec": 10,
            "parameters": {"timeout": 21600},
        }

    def unjudged_benchmarks(self, base_config: dict[str, Any]) -> list[str]:
        """Бенчи без назначенного судьи (их меряет несудейский gigametric)."""
        benchmarks = base_config.get("benchmarks") or {}
        judged = {b for tasks in self.judges.values() for b in tasks}
        return sorted(set(benchmarks) - judged)

    def judged_validation_config(
        self, base_config: dict[str, Any], judge_model_name: str, task_subset: list[str]
    ) -> dict[str, Any]:
        cfg = copy.deepcopy(base_config)
        defaults = cfg.get("defaults") or {}
        benchmark_config = defaults.get("benchmark_config") or {}
        subset = set(task_subset)
        benchmarks = cfg.get("benchmarks") or {}
        cfg["benchmarks"] = {b: benchmarks[b] for b in task_subset if b in benchmarks}
        jc = self.judge_llm_config(judge_model_name)
        for entry in [benchmark_config, *cfg["benchmarks"].values()]:
            entry.pop("judge_llm_config", None)
            params = entry.get("parameters") or {}
            for field in (
                "judge_llm_config",
                "judge_config",
                "grader_config",
                "judge_model",
                "judge_name",
                "judge_api_url",
                "judge_api_key",
                "judge_params",
                "judge_cert_file",
                "judge_cert_key_file",
            ):
                params.pop(field, None)
            if "user_llm_config" in params:
                params["user_llm_config"] = copy.deepcopy(jc)
            entry["parameters"] = params
        benchmark_config["judge_llm_config"] = jc
        # sber-llm-judge-adapter reads judge settings from parameters.
        params = benchmark_config.get("parameters") or {}
        params.setdefault("judge_api_url", jc["api_url"])
        params.setdefault("judge_api_key", jc["api_key"])
        params.setdefault("judge_model", jc["model"])
        params.setdefault("judge_name", jc["name"])
        benchmark_config["parameters"] = params
        defaults["benchmark_config"] = benchmark_config
        cfg["defaults"] = defaults
        models = []
        for model_entry in cfg.get("models") or []:
            kept = [b for b in (model_entry.get("benchmarks") or []) if b in subset]
            if not kept:
                continue
            model_entry["benchmarks"] = kept
            models.append(model_entry)
        cfg["models"] = models
        return cfg

    def materialise_judged_validation_config(
        self,
        *,
        target: Path,
        base_config: dict[str, Any],
        judge_model_name: str,
        task_subset: list[str],
    ) -> Path:
        cfg = self.judged_validation_config(base_config, judge_model_name, task_subset)
        OmegaConf.save(cfg, target, resolve=False)
        return target


class SafetyUnifiedEvaluatorConfig(JudgedEvaluatorConfig):
    model_config = ConfigDict(
        populate_by_name=True, extra="forbid", hide_input_in_errors=True
    )

    DEFAULT_NUM_RETRIES: ClassVar[int] = 20
    DEFAULT_MODEL: ClassVar[dict[str, Any]] = {
        "max_tokens": 2048,
        "temperature": 0.0,
        "num_retries": DEFAULT_NUM_RETRIES,
    }
    DEFAULT_JUDGE: ClassVar[dict[str, Any]] = {
        "max_tokens": 1536,
        "temperature": 0.0,
        "seed": 12345,
        "num_retries": DEFAULT_NUM_RETRIES,
    }
    ALL_DATASET_FILES: ClassVar[tuple[str, ...]] = (
        "unified_safety_dataset.jsonl",
        "unified_censor_old.jsonl",
        "unified_censor_new_full.jsonl",
        "unified_validation_basket.jsonl",
    )

    dataset_root: str = Field(default="/home/user/safety_unified/data", exclude=True)
    runner_root: str = Field(default="/home/user/safety_unified/runner", exclude=True)
    dataset_file: str | list[str] | None = None
    model: dict[str, Any] = Field(default_factory=dict)
    judge: dict[str, Any] = Field(default_factory=dict)
    color_weights: dict[str, float] = Field(default_factory=dict)
    domain_weights: dict[str, float] = Field(default_factory=dict)
    mutate: bool = False
    mutation_seed: int = 0
    judge_prompt_file: str = "judge_prompt_production.txt"
    judge_schema: Literal["production", "unsafe", "grade"] = "production"
    judge_prompt_files: dict[str, str] = Field(default_factory=dict)
    judge_chat_template_kwargs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    judge_pool: list[str | dict[str, Any]] = Field(default_factory=list)
    selection_manifest: str | None = None
    domain_rules_file: Literal["domain_rules.json"] | None = None
    publish_metrics: bool = True
    judge_min_agreement: float = Field(default=0.75, gt=0.0, le=1.0)
    llm_timeout: int = Field(default=DEFAULT_REQUEST_TIMEOUT, ge=1)
    parallel: int | None = Field(default=None, ge=1)
    judge_parallel: int | None = Field(default=None, ge=1)
    limit: int | None = Field(default=None, ge=1)
    answers_dir: str | None = None
    base_url: str | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _validate_selected_judges_share_datasets(self) -> SafetyUnifiedEvaluatorConfig:
        if self.selection_manifest is None:
            return self
        scopes = [self.dataset_files_for(name) for name in self.judges]
        if any(scope != scopes[0] for scope in scopes[1:]):
            raise ValueError(
                "autometrics.evaluators.safety_unified: selected judge passes "
                "require identical dataset files"
            )
        return self

    def apply_defaults(
        self,
        *,
        inference_instances: int,
        base_url: str,
        **_: Any,
    ) -> None:
        self.base_url = f"{base_url}/v1"
        if self.parallel is None:
            self.parallel = (
                inference_instances
                * SAFETY_UNIFIED_DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE
            )
        if self.judge_parallel is None:
            self.judge_parallel = (
                inference_instances
                * SAFETY_UNIFIED_DEFAULT_JUDGE_PARALLEL_PER_INFERENCE_INSTANCE
            )

    def dataset_file_for(self, judge_model_name: str) -> str | list[str]:
        tasks = self.judges[judge_model_name]
        if tasks == ["all"]:
            return "all"
        if tasks:
            return tasks
        return self.dataset_file or "all"

    def dataset_files_for(self, judge_model_name: str) -> list[str]:
        dataset_file = self.dataset_file_for(judge_model_name)
        if dataset_file == "all":
            return list(self.ALL_DATASET_FILES)
        if isinstance(dataset_file, str):
            return [dataset_file]
        return list(dataset_file)

    def dataset_scope_for(self, judge_model_name: str) -> str:
        dataset_file = self.dataset_file_for(judge_model_name)
        if isinstance(dataset_file, str):
            return Path(dataset_file).stem
        return "_".join(sorted(Path(name).stem for name in dataset_file))

    def build_overrides(
        self,
        *,
        judge_model_name: str,
        out_dir: Path,
        default_answers_dir: Path,
        checkpoint_model_name: str,
        checkpoint_path: Path,
    ) -> dict[str, Any]:
        if self.parallel is None or self.judge_parallel is None:
            raise ValueError(
                "safety_unified: parallel and judge_parallel "
                "must be set via apply_defaults()"
            )
        judge_base_url = self.judge_endpoints[judge_model_name].base_url
        model_overrides = {
            **self.DEFAULT_MODEL,
            "timeout": self.llm_timeout,
            **self.model,
        }
        model_overrides["api_base"] = self.require_base_url()
        model_overrides.setdefault("name", checkpoint_model_name or "model")
        model_overrides["checkpoint"] = str(checkpoint_path)

        judge_overrides: dict[str, Any] = {
            **self.DEFAULT_JUDGE,
            "timeout": self.llm_timeout,
            **self.judge,
        }
        judge_overrides["model_name"] = judge_model_name
        judge_overrides["api_base"] = f"{judge_base_url}/v1"
        chat_template_kwargs = self.judge_chat_template_kwargs.get(judge_model_name)
        if chat_template_kwargs:
            judge_overrides["chat_template_kwargs"] = chat_template_kwargs
        judge_overrides["parallel"] = self.judge_parallel

        payload = self.model_dump(
            exclude_none=True,
            exclude={
                "judges",
                "dataset_file",
                "model",
                "judge",
                "judge_prompt_files",
                "judge_chat_template_kwargs",
                "llm_timeout",
                "judge_parallel",
                "answers_dir",
                "publish_metrics",
            },
        )
        payload.update(
            {
                "out_dir": str(out_dir),
                "answers_dir": self.answers_dir or str(default_answers_dir),
                "dataset_root": self.dataset_root,
                "dataset_file": self.dataset_file_for(judge_model_name),
                "model": model_overrides,
                "judge": judge_overrides,
                "judge_prompt_file": self.judge_prompt_files.get(
                    judge_model_name, self.judge_prompt_file
                ),
            }
        )
        return payload

    def build_selection_overrides(
        self,
        *,
        metrics_dir: Path,
        checkpoint_model_name: str,
        checkpoint_path: Path,
    ) -> dict[str, Any]:
        result = self.build_overrides(
            judge_model_name=next(iter(self.judges)),
            out_dir=metrics_dir / "selected",
            default_answers_dir=metrics_dir / "answers",
            checkpoint_model_name=checkpoint_model_name,
            checkpoint_path=checkpoint_path,
        )
        result["source_runs"] = [
            {
                "model": model.partition(":")[2] if ":" in model else model,
                "judge_key": model,
                "root": str(metrics_dir / model.replace("/", "_")),
            }
            for model in self.judges
        ]
        return result


class AccLenDatasetConfig(BaseConfigModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    # Used as a filename and as a metric key, so no path separators.
    name: str = Field(pattern=r"^[\w.-]+$")
    path: Path
    dataset_format: Literal["custom", "sharegpt"] = Field(
        default="custom",
        validation_alias=AliasChoices("format", "dataset_format"),
    )
    max_new_tokens: int | None = Field(default=None, ge=4)
    apply_chat_template: bool = True
    temperature: float | None = Field(default=None, ge=0.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    repetition_penalty: float | None = Field(default=None, gt=0.0, le=2.0)

    @model_validator(mode="after")
    def _check_chat_template(self) -> "AccLenDatasetConfig":
        if (
            self.dataset_format == "sharegpt"
            and "apply_chat_template" in self.model_fields_set
        ):
            raise ValueError(
                f"acc_len dataset '{self.name}': sharegpt format always sends raw "
                "prompts, apply_chat_template has no effect"
            )
        return self

    @model_validator(mode="after")
    def _check_output_len(self) -> "AccLenDatasetConfig":
        if self.dataset_format == "custom" and self.max_new_tokens is None:
            raise ValueError(
                f"acc_len dataset '{self.name}': custom format requires max_new_tokens"
            )
        if self.dataset_format == "sharegpt" and self.max_new_tokens is not None:
            raise ValueError(
                f"acc_len dataset '{self.name}': sharegpt format keeps its reference "
                "output lengths, max_new_tokens has no effect"
            )
        return self


class AccLenEvaluatorConfig(MetricsEvaluatorConfig):
    """Config for the MTP acceptance-length evaluator (sglang.bench_serving)."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    datasets: list[AccLenDatasetConfig] = Field(min_length=1)
    num_prompts: int = Field(default=512, ge=1)
    seed: int = 1703
    warmup_requests: int = Field(default=10, ge=0)
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    repetition_penalty: float | None = Field(default=None, gt=0.0, le=2.0)
    request_seed: int | None = None
    apply_chat_template: bool = True
    reasoning: bool = False
    max_concurrency: int | None = Field(default=None, ge=1)
    request_timeout: int = Field(default=DEFAULT_REQUEST_TIMEOUT, gt=0)

    venv_python: str = Field(default="/home/user/sglang/bin/python", exclude=True)
    base_url: str | None = Field(default=None, exclude=True)
    worker_urls: list[str] = Field(default_factory=list, exclude=True)
    context_len: int | None = Field(default=None, exclude=True)

    def apply_defaults(
        self,
        *,
        inference_instances: int,
        direct_base_url: str,
        worker_urls: list[str],
        hf_checkpoint_path: Path | None,
        max_seq_len: int | None = None,
        **_: Any,
    ) -> None:
        self.base_url = direct_base_url
        self.worker_urls = worker_urls
        self.context_len = max_seq_len
        if self.max_concurrency is None:
            self.max_concurrency = (
                inference_instances * ACC_LEN_DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE
            )
        self.apply_chat_template = self.apply_chat_template and self._has_chat_template(
            hf_checkpoint_path
        )

    @staticmethod
    def _has_chat_template(hf_checkpoint_path: Path | None) -> bool:
        if hf_checkpoint_path is None:
            return False
        if any(
            (hf_checkpoint_path / name).exists()
            for name in ("chat_template.jinja", "chat_template.json")
        ):
            return True
        tokenizer_config_path = hf_checkpoint_path / "tokenizer_config.json"
        if not tokenizer_config_path.exists():
            return False
        try:
            with open(tokenizer_config_path) as tokenizer_config_file:
                return bool(json.load(tokenizer_config_file).get("chat_template"))
        except (OSError, json.JSONDecodeError):
            return False

    @model_validator(mode="after")
    def _default_request_seed(self) -> "AccLenEvaluatorConfig":
        if self.request_seed is None:
            self.request_seed = self.seed
        return self

    @model_validator(mode="after")
    def _unique_dataset_names(self) -> "AccLenEvaluatorConfig":
        names = [dataset.name for dataset in self.datasets]
        if len(names) != len(set(names)):
            raise ValueError(f"acc_len: dataset names must be unique, got {names}")
        return self

    def _sampling_params(self, dataset: AccLenDatasetConfig) -> dict[str, Any]:
        """Sampling params for one dataset; None inherits the evaluator value.

        Any temperature above 0 takes EAGLE off the greedy verify path into
        rejection sampling, whose acceptance coins are drawn with torch.rand_like
        from the global RNG and not from the request seed, so
        --enable-deterministic-inference does not make the acceptance length of
        such a dataset reproducible between runs.
        """
        params: dict[str, Any] = {
            "temperature": (
                self.temperature if dataset.temperature is None else dataset.temperature
            ),
            "top_p": self.top_p if dataset.top_p is None else dataset.top_p,
            "seed": self.request_seed,
        }
        repetition_penalty = (
            self.repetition_penalty
            if dataset.repetition_penalty is None
            else dataset.repetition_penalty
        )
        if repetition_penalty is not None:
            params["repetition_penalty"] = repetition_penalty
        return params

    def build_cli_args(
        self,
        *,
        dataset: AccLenDatasetConfig,
        output_path: Path,
    ) -> list[str]:
        args = [
            "--backend",
            "sglang-oai",
            "--base-url",
            self.require_base_url(),
            "--dataset-name",
            dataset.dataset_format,
            "--dataset-path",
            dataset.path.absolute().as_posix(),
            "--num-prompts",
            str(self.num_prompts),
            "--max-concurrency",
            str(self.max_concurrency),
            "--seed",
            str(self.seed),
            "--warmup-requests",
            str(self.warmup_requests),
            "--extra-request-body",
            json.dumps(self._sampling_params(dataset)),
            "--output-file",
            output_path.absolute().as_posix(),
            "--output-details",
            "--disable-ignore-eos",
            "--flush-cache",
        ]
        if self.context_len is not None:
            args += ["--sharegpt-context-len", str(self.context_len)]
        if dataset.dataset_format == "custom":
            args += ["--sharegpt-output-len", str(dataset.max_new_tokens)]
            if self.apply_chat_template and dataset.apply_chat_template:
                args.append("--apply-chat-template")
                if self.reasoning:
                    args += [
                        "--chat-template-kwargs",
                        json.dumps({"reasoning": True}),
                    ]
        return args


class RliConfig(BaseConfigModel):
    allow_internet_access: bool = Field(
        default=True,
        validation_alias=AliasChoices("allow-internet-access", "allow_internet_access"),
        serialization_alias="allow-internet-access",
    )
    n_concurrent: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices("n-concurrent", "n_concurrent"),
        serialization_alias="n-concurrent",
    )
    timeout_rli: int = Field(
        default=1620,
        ge=1,
        validation_alias=AliasChoices("timeout-rli", "timeout_rli"),
        serialization_alias="timeout-rli",
    )
    request_retries: int = Field(
        default=1,
        ge=0,
        validation_alias=AliasChoices("request-retries", "request_retries"),
        serialization_alias="request-retries",
    )
    transfer_retries: int = Field(
        default=10,
        ge=0,
        validation_alias=AliasChoices("transfer-retries", "transfer_retries"),
        serialization_alias="transfer-retries",
    )
    transfer_retry_backoff: float = Field(
        default=2.0,
        ge=0,
        validation_alias=AliasChoices(
            "transfer-retry-backoff",
            "transfer_retry_backoff",
        ),
        serialization_alias="transfer-retry-backoff",
    )
    session_ready_timeout: float = Field(
        default=600.0,
        gt=0,
        validation_alias=AliasChoices(
            "session-ready-timeout",
            "session_ready_timeout",
        ),
        serialization_alias="session-ready-timeout",
    )
    session_ready_poll_interval: float = Field(
        default=1.0,
        gt=0,
        validation_alias=AliasChoices(
            "session-ready-poll-interval",
            "session_ready_poll_interval",
        ),
        serialization_alias="session-ready-poll-interval",
    )
    enable_probe_warmup: bool = Field(
        default=True,
        validation_alias=AliasChoices("enable-probe-warmup", "enable_probe_warmup"),
        serialization_alias="enable-probe-warmup",
    )
    create_session_jitter_max_sec: float = Field(
        default=0.0,
        ge=0,
        validation_alias=AliasChoices(
            "create-session-jitter-max-sec",
            "create_session_jitter_max_sec",
        ),
        serialization_alias="create-session-jitter-max-sec",
    )
    probe_retry_backoff_sec: float = Field(
        default=2.0,
        ge=0,
        validation_alias=AliasChoices(
            "probe-retry-backoff-sec",
            "probe_retry_backoff_sec",
        ),
        serialization_alias="probe-retry-backoff-sec",
    )
    command_timeout: int = Field(
        default=600,
        ge=1,
        validation_alias=AliasChoices("command-timeout", "command_timeout"),
        serialization_alias="command-timeout",
    )


class TerminalBenchEvaluatorConfig(MetricsEvaluatorConfig, RliConfig):
    dataset: str = "/home/user/terminal-bench-2"
    agent: Literal["terminus-2", "mini-swe-agent"] = "terminus-2"
    model_api_base: str | None = Field(
        default=None,
        validation_alias=AliasChoices("model-api-base", "model_api_base"),
        serialization_alias="model-api-base",
    )
    max_tokens: int | None = Field(
        default=32768,
        ge=1,
        validation_alias=AliasChoices("max-tokens", "max_tokens"),
        serialization_alias="max-tokens",
    )
    max_total_output_tokens: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices(
            "max-total-output-tokens",
            "max_total_output_tokens",
        ),
        serialization_alias="max-total-output-tokens",
    )
    max_input_tokens: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices("max-input-tokens", "max_input_tokens"),
        serialization_alias="max-input-tokens",
    )
    agent_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("agent-kwargs", "agent_kwargs"),
        serialization_alias="agent-kwargs",
    )
    agent_timeout: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices("agent-timeout", "agent_timeout"),
        serialization_alias="agent-timeout",
    )
    llm_timeout: int = Field(
        default=DEFAULT_REQUEST_TIMEOUT,
        ge=1,
        validation_alias=AliasChoices("llm-timeout", "llm_timeout"),
        serialization_alias="llm-timeout",
    )
    pass_at_k: int = Field(
        default=1,
        ge=1,
        validation_alias=AliasChoices("pass-at-k", "pass_at_k"),
        serialization_alias="pass-at-k",
    )
    n_tasks: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices("n-tasks", "n_tasks"),
        serialization_alias="n-tasks",
    )
    task_name: str | list[str] | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "task-name",
            "task_name",
            "include-task-name",
            "include_task_name",
        ),
        serialization_alias="task-name",
    )
    exclude_task_name: str | list[str] | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "exclude-task-name",
            "exclude_task_name",
        ),
        serialization_alias="exclude-task-name",
    )
    env_prefix: str = Field(
        default="terminal-bench",
        validation_alias=AliasChoices("env-prefix", "env_prefix"),
        serialization_alias="env-prefix",
    )
    env_version: str = Field(
        default="1.0.1",
        validation_alias=AliasChoices("env-version", "env_version"),
        serialization_alias="env-version",
    )
    ttl_ms: int = Field(
        default=64_800_000,
        ge=1,
        validation_alias=AliasChoices("ttl-ms", "ttl_ms"),
        serialization_alias="ttl-ms",
    )
    create_session_jitter_max_sec: float = Field(
        default=180.0,
        ge=0,
        validation_alias=AliasChoices(
            "create-session-jitter-max-sec",
            "create_session_jitter_max_sec",
        ),
        serialization_alias="create-session-jitter-max-sec",
    )
    venv_python: str = Field(
        default="/home/user/terminal_bench_env/bin/python",
        exclude=True,
    )

    def apply_defaults(
        self,
        *,
        inference_instances: int,
        base_url: str,
        **_: Any,
    ) -> None:
        if self.model_api_base is None:
            self.model_api_base = f"{base_url}/v1"
        if self.n_concurrent is None:
            self.n_concurrent = (
                inference_instances
                * TERMINAL_BENCH_DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE
            )


class SkillsBenchEvaluatorConfig(MetricsEvaluatorConfig, RliConfig):
    agent: Literal["mini-swe-agent", "openhands-sdk", "openhands-cli", "terminus-2"] = (
        "openhands-cli"
    )
    conditions: list[Literal["no-skill", "with-skill"]] = Field(
        default_factory=lambda: ["no-skill", "with-skill"],
        min_length=1,
    )
    text_only: bool = Field(
        default=True,
        validation_alias=AliasChoices("text-only", "text_only"),
        serialization_alias="text-only",
    )
    model_api_base: str | None = Field(
        default=None,
        validation_alias=AliasChoices("model-api-base", "model_api_base"),
        serialization_alias="model-api-base",
    )
    max_tokens: int = Field(
        default=16384,
        ge=1,
        validation_alias=AliasChoices("max-tokens", "max_tokens"),
        serialization_alias="max-tokens",
    )
    max_total_output_tokens: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices(
            "max-total-output-tokens", "max_total_output_tokens"
        ),
        serialization_alias="max-total-output-tokens",
    )
    max_input_tokens: int = Field(
        default=98304,
        ge=1,
        validation_alias=AliasChoices("max-input-tokens", "max_input_tokens"),
        serialization_alias="max-input-tokens",
    )
    agent_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("agent-kwargs", "agent_kwargs"),
        serialization_alias="agent-kwargs",
    )
    agent_timeout: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices("agent-timeout", "agent_timeout"),
        serialization_alias="agent-timeout",
    )
    llm_timeout: int = Field(
        default=DEFAULT_REQUEST_TIMEOUT,
        ge=1,
        validation_alias=AliasChoices("llm-timeout", "llm_timeout"),
        serialization_alias="llm-timeout",
    )
    pass_at_k: int = Field(
        default=1,
        ge=1,
        validation_alias=AliasChoices("pass-at-k", "pass_at_k"),
        serialization_alias="pass-at-k",
    )
    n_tasks: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices("n-tasks", "n_tasks"),
        serialization_alias="n-tasks",
    )
    task_name: str | list[str] | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "task-name",
            "task_name",
            "include-task-name",
            "include_task_name",
        ),
        serialization_alias="task-name",
    )
    exclude_task_name: str | list[str] | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "exclude-task-name",
            "exclude_task_name",
        ),
        serialization_alias="exclude-task-name",
    )
    env_prefix: str = Field(
        default="skillsbench",
        validation_alias=AliasChoices("env-prefix", "env_prefix"),
        serialization_alias="env-prefix",
    )
    env_version: str = Field(
        default="1.0.1",
        validation_alias=AliasChoices("env-version", "env_version"),
        serialization_alias="env-version",
    )
    ttl_ms: int = Field(
        default=64_800_000,
        ge=1,
        validation_alias=AliasChoices("ttl-ms", "ttl_ms"),
        serialization_alias="ttl-ms",
    )
    create_session_jitter_max_sec: float = Field(
        default=450.0,
        ge=0,
        validation_alias=AliasChoices(
            "create-session-jitter-max-sec", "create_session_jitter_max_sec"
        ),
        serialization_alias="create-session-jitter-max-sec",
    )
    command_timeout: int = Field(
        default=180,
        ge=1,
        validation_alias=AliasChoices("command-timeout", "command_timeout"),
        serialization_alias="command-timeout",
    )
    venv_python: str = Field(
        default="/home/user/terminal_bench_env/bin/python",
        exclude=True,
    )
    force_recalc: bool = Field(
        default=False,
        validation_alias=AliasChoices("force-recalc", "force_recalc"),
        serialization_alias="force-recalc",
        exclude=True,
    )

    def apply_defaults(
        self,
        *,
        inference_instances: int,
        base_url: str,
        **_: Any,
    ) -> None:
        if self.model_api_base is None:
            self.model_api_base = f"{base_url}/v1"
        if self.n_concurrent is None:
            self.n_concurrent = (
                inference_instances
                * SKILLS_BENCH_DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE
            )


class HarborRliEvaluatorConfig(MetricsEvaluatorConfig, RliConfig):
    runs: list[HarborRliEvaluatorConfig] | None = Field(default=None, min_length=1)

    job_name: str | None = Field(
        default=None,
        min_length=1,
        validation_alias=AliasChoices("job-name", "job_name"),
        serialization_alias="job-name",
    )
    benchmark: str | None = Field(
        default=None, pattern=r"^[a-z0-9]+(?:[-_.][a-z0-9]+)*$"
    )
    benchmark_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("benchmark-kwargs", "benchmark_kwargs"),
        serialization_alias="benchmark-kwargs",
    )
    agent: str = Field(default="mini-swe-agent", min_length=1)
    model_api_base: str | None = Field(
        default=None,
        validation_alias=AliasChoices("model-api-base", "model_api_base"),
        serialization_alias="model-api-base",
    )
    max_tokens: int | None = Field(
        default=32768,
        ge=1,
        validation_alias=AliasChoices("max-tokens", "max_tokens"),
        serialization_alias="max-tokens",
    )
    max_total_output_tokens: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices(
            "max-total-output-tokens",
            "max_total_output_tokens",
        ),
        serialization_alias="max-total-output-tokens",
    )
    max_input_tokens: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices("max-input-tokens", "max_input_tokens"),
        serialization_alias="max-input-tokens",
    )
    agent_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("agent-kwargs", "agent_kwargs"),
        serialization_alias="agent-kwargs",
    )
    agent_timeout: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices("agent-timeout", "agent_timeout"),
        serialization_alias="agent-timeout",
    )
    llm_timeout: int = Field(
        default=DEFAULT_REQUEST_TIMEOUT,
        ge=1,
        validation_alias=AliasChoices("llm-timeout", "llm_timeout"),
        serialization_alias="llm-timeout",
    )
    pass_at_k: int = Field(
        default=1,
        ge=1,
        validation_alias=AliasChoices("pass-at-k", "pass_at_k"),
        serialization_alias="pass-at-k",
    )
    n_tasks: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices("n-tasks", "n_tasks"),
        serialization_alias="n-tasks",
    )
    task_name: str | list[str] | None = Field(
        default=None,
        validation_alias=AliasChoices("task-name", "task_name"),
        serialization_alias="task-name",
    )
    exclude_task_name: str | list[str] | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "exclude-task-name",
            "exclude_task_name",
        ),
        serialization_alias="exclude-task-name",
    )
    venv_python: str = Field(
        default="/home/user/harbor_rli_env/bin/python",
        exclude=True,
    )
    force_recalc: bool = Field(
        default=False,
        validation_alias=AliasChoices("force-recalc", "force_recalc"),
        serialization_alias="force-recalc",
        exclude=True,
    )

    @model_validator(mode="after")
    def _apply_network_defaults(self) -> HarborRliEvaluatorConfig:
        if (
            self.benchmark
            in {
                "swe-bench-verified",
                "swe-bench-multilingual",
                "harness-bench-lite",
                "harness-bench-medium",
            }
            and "allow_internet_access" not in self.model_fields_set
        ):
            self.allow_internet_access = False
        return self

    @model_validator(mode="after")
    def _validate_runs(self) -> HarborRliEvaluatorConfig:
        if self.runs is None:
            if self.benchmark is None:
                raise ValueError("harbor_rli: benchmark is required for a single run")
            reserved = {name.replace("_", "-") for name in type(self).model_fields}
            duplicates = reserved.intersection(
                key.replace("_", "-") for key in self.benchmark_kwargs
            )
            if duplicates:
                raise ValueError(
                    "harbor_rli: benchmark-kwargs duplicates common fields: "
                    + ", ".join(sorted(duplicates))
                )
            return self
        if self.model_fields_set != {"runs"}:
            raise ValueError("harbor_rli: runs cannot be combined with run parameters")
        benchmarks = set()
        for run in self.runs:
            if run.runs is not None:
                raise ValueError("harbor_rli: nested runs are not supported")
            if run.benchmark in benchmarks:
                raise ValueError("harbor_rli: each run must have a different benchmark")
            benchmarks.add(run.benchmark)
        return self

    def apply_defaults(
        self,
        *,
        inference_instances: int,
        base_url: str,
        **_: Any,
    ) -> None:
        if self.runs is not None:
            for run in self.runs:
                run.apply_defaults(
                    inference_instances=inference_instances, base_url=base_url
                )
            return
        if self.model_api_base is None:
            self.model_api_base = f"{base_url}/v1"
        if self.n_concurrent is None:
            self.n_concurrent = (
                inference_instances * HARBOR_RLI_DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE
            )


class PierRliEvaluatorConfig(HarborRliEvaluatorConfig):
    allow_internet_access: bool = Field(
        default=False,
        validation_alias=AliasChoices("allow-internet-access", "allow_internet_access"),
        serialization_alias="allow-internet-access",
    )
    runs: list[PierRliEvaluatorConfig] | None = Field(default=None, min_length=1)
    venv_python: str = Field(
        default="/home/user/pier_rli_env/bin/python",
        exclude=True,
    )


class SweBenchEvaluatorConfig(MetricsEvaluatorConfig, RliConfig):
    allow_internet_access: bool = Field(
        default=False,
        validation_alias=AliasChoices("allow-internet-access", "allow_internet_access"),
        serialization_alias="allow-internet-access",
    )
    dataset: str = "/home/user/swe-bench-verified/datasets/swebench-verified"
    agent: Literal[
        "mini-swe-agent",
        "agentless",
        "openhands",
    ] = "mini-swe-agent"
    model_api_base: str | None = Field(
        default=None,
        validation_alias=AliasChoices("model-api-base", "model_api_base"),
        serialization_alias="model-api-base",
    )
    max_tokens: int | None = Field(
        default=32768,
        ge=1,
        validation_alias=AliasChoices("max-tokens", "max_tokens"),
        serialization_alias="max-tokens",
    )
    max_total_output_tokens: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices(
            "max-total-output-tokens",
            "max_total_output_tokens",
        ),
        serialization_alias="max-total-output-tokens",
    )
    max_input_tokens: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices("max-input-tokens", "max_input_tokens"),
        serialization_alias="max-input-tokens",
    )
    agent_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("agent-kwargs", "agent_kwargs"),
        serialization_alias="agent-kwargs",
    )
    agent_timeout: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices("agent-timeout", "agent_timeout"),
        serialization_alias="agent-timeout",
    )
    llm_timeout: int = Field(
        default=DEFAULT_REQUEST_TIMEOUT,
        ge=1,
        validation_alias=AliasChoices("llm-timeout", "llm_timeout"),
        serialization_alias="llm-timeout",
    )
    pass_at_k: int = Field(
        default=1,
        ge=1,
        validation_alias=AliasChoices("pass-at-k", "pass_at_k"),
        serialization_alias="pass-at-k",
    )
    n_tasks: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices("n-tasks", "n_tasks"),
        serialization_alias="n-tasks",
    )
    task_name: str | list[str] | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "task-name",
            "task_name",
            "include-task-name",
            "include_task_name",
        ),
        serialization_alias="task-name",
    )
    exclude_task_name: str | list[str] | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "exclude-task-name",
            "exclude_task_name",
        ),
        serialization_alias="exclude-task-name",
    )
    env_prefix: str = Field(
        default="swe-bench",
        validation_alias=AliasChoices("env-prefix", "env_prefix"),
        serialization_alias="env-prefix",
    )
    env_version: str = Field(
        default="1.0.2",
        validation_alias=AliasChoices("env-version", "env_version"),
        serialization_alias="env-version",
    )
    ttl_ms: int = Field(
        default=64_800_000,
        ge=1,
        validation_alias=AliasChoices("ttl-ms", "ttl_ms"),
        serialization_alias="ttl-ms",
    )
    create_session_jitter_max_sec: float = Field(
        default=180.0,
        ge=0,
        validation_alias=AliasChoices(
            "create-session-jitter-max-sec",
            "create_session_jitter_max_sec",
        ),
        serialization_alias="create-session-jitter-max-sec",
    )
    command_timeout: int = Field(
        default=180,
        ge=1,
        validation_alias=AliasChoices("command-timeout", "command_timeout"),
        serialization_alias="command-timeout",
    )
    venv_python: str = Field(
        default="/home/user/terminal_bench_env/bin/python",
        exclude=True,
    )
    force_recalc: bool = Field(
        default=False,
        validation_alias=AliasChoices("force-recalc", "force_recalc"),
        serialization_alias="force-recalc",
        exclude=True,
    )
    post_reverify: bool = Field(
        default=False,
        validation_alias=AliasChoices("post-reverify", "post_reverify"),
        serialization_alias="post-reverify",
    )
    post_reverify_grader: str = Field(
        default="rli",
        validation_alias=AliasChoices(
            "post-reverify-grader",
            "post_reverify_grader",
        ),
        serialization_alias="post-reverify-grader",
    )

    def apply_defaults(
        self,
        *,
        inference_instances: int,
        base_url: str,
        **_: Any,
    ) -> None:
        if self.model_api_base is None:
            self.model_api_base = f"{base_url}/v1"
        if self.n_concurrent is None:
            self.n_concurrent = (
                inference_instances * SWE_BENCH_DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE
            )


class Nl2RepoEvaluatorConfig(MetricsEvaluatorConfig, RliConfig):
    job_name: str | None = Field(
        default=None,
        min_length=1,
        validation_alias=AliasChoices("job-name", "job_name"),
        serialization_alias="job-name",
    )
    agent: Literal[
        "mini-swe-agent",
        "openhands",
        "openhands-cli",
        "openhands-sdk",
    ] = "openhands-sdk"
    model_api_base: str | None = Field(
        default=None,
        validation_alias=AliasChoices("model-api-base", "model_api_base"),
        serialization_alias="model-api-base",
    )
    max_tokens: int | None = Field(
        default=16384,
        ge=1,
        validation_alias=AliasChoices("max-tokens", "max_tokens"),
        serialization_alias="max-tokens",
    )
    max_total_output_tokens: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices(
            "max-total-output-tokens",
            "max_total_output_tokens",
        ),
        serialization_alias="max-total-output-tokens",
    )
    max_input_tokens: int | None = Field(
        default=98304,
        ge=1,
        validation_alias=AliasChoices("max-input-tokens", "max_input_tokens"),
        serialization_alias="max-input-tokens",
    )
    agent_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("agent-kwargs", "agent_kwargs"),
        serialization_alias="agent-kwargs",
    )
    enable_contamination_filter: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "enable-contamination-filter",
            "enable_contamination_filter",
        ),
        serialization_alias="enable-contamination-filter",
    )
    agent_timeout: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices("agent-timeout", "agent_timeout"),
        serialization_alias="agent-timeout",
    )
    verifier_timeout: float | None = Field(
        default=None,
        gt=0,
        validation_alias=AliasChoices(
            "verifier-timeout",
            "verifier_timeout",
        ),
        serialization_alias="verifier-timeout",
    )
    llm_timeout: int = Field(
        default=DEFAULT_REQUEST_TIMEOUT,
        ge=1,
        validation_alias=AliasChoices("llm-timeout", "llm_timeout"),
        serialization_alias="llm-timeout",
    )
    pass_at_k: int = Field(
        default=1,
        ge=1,
        validation_alias=AliasChoices("pass-at-k", "pass_at_k"),
        serialization_alias="pass-at-k",
    )
    max_retries: int = Field(
        default=1,
        ge=0,
        validation_alias=AliasChoices("max-retries", "max_retries"),
        serialization_alias="max-retries",
    )
    n_tasks: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices("n-tasks", "n_tasks"),
        serialization_alias="n-tasks",
    )
    task_name: str | list[str] | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "task-name",
            "task_name",
            "include-task-name",
            "include_task_name",
        ),
        serialization_alias="task-name",
    )
    exclude_task_name: str | list[str] | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "exclude-task-name",
            "exclude_task_name",
        ),
        serialization_alias="exclude-task-name",
    )
    env_prefix: str = Field(
        default="nl2repo",
        validation_alias=AliasChoices("env-prefix", "env_prefix"),
        serialization_alias="env-prefix",
    )
    agent_env_version: str = Field(
        default="1.0.2",
        validation_alias=AliasChoices(
            "agent-env-version",
            "agent_env_version",
        ),
        serialization_alias="agent-env-version",
    )
    verifier_env_version: str = Field(
        default="1.0.0",
        validation_alias=AliasChoices(
            "verifier-env-version",
            "verifier_env_version",
            "env-version",
            "env_version",
        ),
        serialization_alias="verifier-env-version",
    )
    ttl_ms: int = Field(
        default=64_800_000,
        ge=1,
        validation_alias=AliasChoices("ttl-ms", "ttl_ms"),
        serialization_alias="ttl-ms",
    )
    create_session_jitter_max_sec: float = Field(
        default=450.0,
        ge=0,
        validation_alias=AliasChoices(
            "create-session-jitter-max-sec",
            "create_session_jitter_max_sec",
        ),
        serialization_alias="create-session-jitter-max-sec",
    )
    command_timeout: int = Field(
        default=180,
        ge=1,
        validation_alias=AliasChoices("command-timeout", "command_timeout"),
        serialization_alias="command-timeout",
    )
    venv_python: str = Field(
        default="/home/user/terminal_bench_env/bin/python",
        exclude=True,
    )
    force_recalc: bool = Field(
        default=False,
        validation_alias=AliasChoices("force-recalc", "force_recalc"),
        serialization_alias="force-recalc",
        exclude=True,
    )

    def apply_defaults(
        self,
        *,
        inference_instances: int,
        base_url: str,
        **_: Any,
    ) -> None:
        if self.model_api_base is None:
            self.model_api_base = f"{base_url}/v1"
        if self.n_concurrent is None:
            self.n_concurrent = (
                inference_instances * NL2REPO_DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE
            )


class AppWorldEvaluatorConfig(MetricsEvaluatorConfig):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    dataset: Literal["test_normal", "test_challenge"] = "test_normal"
    model_name: str = "served"
    num_processes: int | None = Field(default=None, ge=1)
    max_steps: int = Field(default=50, ge=1)
    request_timeout: float = Field(default=DEFAULT_REQUEST_TIMEOUT, gt=0)
    max_retries: int = Field(default=20, ge=1)
    base_url: str | None = Field(default=None, exclude=True)
    venv_python: str = Field(default="/home/user/appworld_env/bin/python", exclude=True)
    appworld_root: str = Field(default="/home/user/appworld", exclude=True)

    def apply_defaults(
        self,
        *,
        inference_instances: int,
        base_url: str,
        **_: Any,
    ) -> None:
        if self.num_processes is None:
            self.num_processes = (
                inference_instances * DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE
            )
        if self.base_url is None:
            self.base_url = f"{base_url}/v1"

    def to_override_json(self) -> str:
        return json.dumps(
            {
                "config": {
                    "agent": {
                        "model_config": {
                            "client_name": "openai",
                            "api_type": "chat_completions",
                            "name": self.model_name,
                            "base_url": self.require_base_url(),
                            "api_key_env_name": "OPENAI_API_KEY",
                            "timeout": self.request_timeout,
                            "max_retries": self.max_retries,
                            "use_cache": True,
                        },
                        "max_steps": self.max_steps,
                    }
                }
            }
        )


class ClawEvalModelConfig(BaseConfigModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    name: str | None = None
    api_base: str | None = None
    api_key: str | None = None
    timeout: int | None = Field(default=None, ge=1)
    max_retries: int | None = Field(default=None, ge=0)
    temperature: float | None = None
    context_window: int | None = Field(default=None, ge=1)
    extra_body: dict[str, Any] | None = None
    reasoning_effort: str | None = None
    input_modalities: list[str] | None = None


class ClawEvalJudgeConfig(BaseConfigModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    model_name: str = Field(default="minimax-m2.7", min_length=1)
    model_id: str | None = None
    api_base: str | None = None
    api_key: str | None = None
    timeout: int | None = Field(default=None, ge=1)
    max_retries: int | None = Field(default=None, ge=0)
    concurrency: int | None = Field(default=None, ge=1)


class ClawEvalUserModelConfig(BaseConfigModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    model_name: str | None = None
    api_base: str | None = None
    api_key: str | None = None
    timeout: int | None = Field(default=None, ge=1)
    max_retries: int | None = Field(default=None, ge=0)


class ClawEvalEvaluatorConfig(MetricsEvaluatorConfig):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    judge: ClawEvalJudgeConfig = Field(default_factory=ClawEvalJudgeConfig)
    model: ClawEvalModelConfig = Field(default_factory=ClawEvalModelConfig)
    user_model: ClawEvalUserModelConfig = Field(default_factory=ClawEvalUserModelConfig)
    upstream_root: str = "/home/user/claweval/upstream"
    runner_root: str = Field(default="/home/user/claweval/runner", exclude=True)
    splits: list[str] = Field(default_factory=lambda: ["general", "multi_turn"])
    skip_web_real: bool = True
    search_backend: Literal["serp", "sber"] = "sber"
    trials: int = 3
    llm_timeout: int = Field(
        default=DEFAULT_REQUEST_TIMEOUT,
        ge=1,
        validation_alias=AliasChoices("timeout", "llm_timeout"),
        exclude=True,
    )
    llm_retries: int = Field(
        default=CLAWEVAL_DEFAULT_LLM_RETRIES,
        ge=0,
        validation_alias=AliasChoices("retries", "llm_retries"),
        exclude=True,
    )
    parallel: Optional[int] = Field(default=None, ge=1)
    judge_parallel: Optional[int] = Field(default=None, ge=1, exclude=True)
    limit: Optional[int] = None
    env: dict[str, Any] = Field(default_factory=dict)
    python_path: str = "/home/user/claweval_env/bin/python"
    base_url: Optional[str] = Field(default=None, exclude=True)
    out_dir: Path | None = None

    @property
    def secondary_model_names(self) -> list[str]:
        return [self.judge.model_name]

    def apply_defaults(
        self,
        *,
        inference_instances: int,
        base_url: str,
        **_: Any,
    ) -> None:
        self.base_url = f"{base_url}/v1"
        self.model.api_base = self.base_url
        judge = self.judge_endpoints[self.judge.model_name]
        if self.judge.concurrency is None:
            self.judge.concurrency = self.judge_parallel or (
                judge.instances * CLAWEVAL_DEFAULT_JUDGE_PARALLEL_PER_INFERENCE_INSTANCE
            )
        if self.judge.api_base is None:
            self.judge.api_base = f"{judge.base_url}/v1"
        if self.model.timeout is None:
            self.model.timeout = self.llm_timeout
        if self.model.max_retries is None:
            self.model.max_retries = self.llm_retries
        if self.judge.timeout is None:
            self.judge.timeout = self.llm_timeout
        if self.judge.max_retries is None:
            self.judge.max_retries = self.llm_retries
        if self.user_model.api_base is not None:
            if self.user_model.timeout is None:
                self.user_model.timeout = self.llm_timeout
            if self.user_model.max_retries is None:
                self.user_model.max_retries = self.llm_retries
        if self.parallel is None:
            self.parallel = (
                inference_instances * CLAWEVAL_DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE
            )


class AaLcrRequestConfig(BaseConfigModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    temperature: float = Field(default=0.0, ge=0.0, le=2.0, strict=True)
    num_retries: int = Field(default=20, ge=1, strict=True)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0, strict=True)
    top_k: int | None = Field(default=None, ge=0, strict=True)


class AaLcrModelConfig(AaLcrRequestConfig):
    max_tokens: int = Field(default=16384, ge=1, strict=True)


class AaLcrJudgeConfig(AaLcrRequestConfig):
    model_name: str = Field(min_length=1)
    max_tokens: int = Field(default=4096, ge=1, strict=True)

    @field_validator("model_name")
    @classmethod
    def _validate_model_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("aa_lcr: judge model name can't be blank")
        return value


class AaLcrEvaluatorConfig(MetricsEvaluatorConfig):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    DEFAULT_DATASET_CSV: ClassVar[str] = "/home/user/aa_lcr/data/AA-LCR_Dataset.csv"
    DEFAULT_TEXT_DIR: ClassVar[str] = "/home/user/aa_lcr/data/extracted_text"
    DEFAULT_RUNNER_ROOT: ClassVar[str] = "/home/user/aa_lcr/runner"

    model: AaLcrModelConfig = Field(default_factory=AaLcrModelConfig)
    judge: AaLcrJudgeConfig
    dataset_csv: str = DEFAULT_DATASET_CSV
    text_dir: str = DEFAULT_TEXT_DIR
    runner_root: str = Field(default=DEFAULT_RUNNER_ROOT, exclude=True)
    n_repeats: int = Field(default=3, ge=1, strict=True)
    use_gold_as_candidate: bool = Field(default=False, strict=True)
    llm_timeout: int = Field(default=DEFAULT_REQUEST_TIMEOUT, ge=1, strict=True)
    parallel: int | None = Field(default=None, ge=1, strict=True)
    judge_parallel: int | None = Field(default=None, ge=1, strict=True)
    limit: int | None = Field(default=None, ge=1, strict=True)
    python_path: str | None = None
    base_url: str | None = Field(default=None, exclude=True)

    @field_validator("dataset_csv", "text_dir", "runner_root", "python_path")
    @classmethod
    def _validate_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("aa_lcr: paths can't be blank")
        return value

    @property
    def runner_python(self) -> str:
        return self.python_path or sys.executable

    @property
    def secondary_model_names(self) -> list[str]:
        return [self.judge.model_name]

    def apply_defaults(
        self,
        *,
        inference_instances: int,
        base_url: str,
        **_: Any,
    ) -> None:
        self.base_url = f"{base_url}/v1"
        if self.parallel is None:
            self.parallel = (
                inference_instances * DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE
            )
        judge = self.judge_endpoints[self.judge.model_name]
        if self.judge_parallel is None:
            self.judge_parallel = (
                judge.instances * DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE
            )

    def candidate_model_name(self, checkpoint_model_name: str) -> str:
        return checkpoint_model_name or "model"

    def build_overrides(
        self,
        *,
        out_dir: Path,
        checkpoint_model_name: str,
        phase: str,
    ) -> dict[str, Any]:
        if phase not in ("generation", "evaluation"):
            raise ValueError(f"aa_lcr: unsupported phase {phase!r}")
        model_overrides = self._model_overrides(checkpoint_model_name)
        overrides: dict[str, Any] = {
            "dataset_csv": self.dataset_csv,
            "text_dir": self.text_dir,
            "out_dir": str(out_dir),
            "n_repeats": self.n_repeats,
            "use_gold_as_candidate": self.use_gold_as_candidate,
            "model": model_overrides,
        }
        if self.limit is not None:
            overrides["limit"] = self.limit
        if phase == "evaluation":
            overrides["judge"] = self._judge_overrides()
        return overrides

    def _model_overrides(self, checkpoint_model_name: str) -> dict[str, Any]:
        if self.parallel is None:
            raise ValueError("aa_lcr: parallel must be set before run")
        base_url = self.require_base_url()
        overrides = self.model.model_dump(exclude_none=True)
        overrides.update(
            {
                "name": self.candidate_model_name(checkpoint_model_name),
                "api_base": base_url,
                "timeout": self.llm_timeout,
                "concurrency": self.parallel,
            }
        )
        return overrides

    def _judge_overrides(self) -> dict[str, Any]:
        if self.judge_parallel is None:
            raise ValueError("aa_lcr: apply_defaults() must be called before scoring")
        judge = self.judge_endpoints[self.judge.model_name]
        overrides = self.judge.model_dump(exclude_none=True)
        overrides.update(
            {
                "api_base": f"{judge.base_url}/v1",
                "timeout": self.llm_timeout,
                "concurrency": self.judge_parallel,
            }
        )
        return overrides

    def result_identity(self) -> dict[str, Any]:
        identity: dict[str, Any] = {
            "limit": self.limit,
            "n_repeats": 1 if self.use_gold_as_candidate else self.n_repeats,
            "use_gold_as_candidate": self.use_gold_as_candidate,
            "llm_timeout": self.llm_timeout,
            "judge": self.judge.model_dump(exclude_none=True),
        }
        if not self.use_gold_as_candidate:
            identity["model"] = self.model.model_dump(exclude_none=True)
        return identity


class TokenNllDatasetConfig(BaseConfigModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    path: str
    max_docs: int | None = Field(default=None, ge=1)

    @field_validator("path")
    @classmethod
    def _path_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("token_nll: dataset path can't be blank")
        return value


class TokenNllEvaluatorConfig(MetricsEvaluatorConfig):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    datasets: dict[str, TokenNllDatasetConfig]
    ref_lag_batches: int = Field(ge=1)
    max_docs: int = Field(default=500, ge=1)
    window_tokens: int = Field(ge=2)
    memorization_threshold: float = Field(default=0.01, gt=0)
    top_spans: int = Field(default=200, ge=1)
    cache_dir: str | None = None
    concurrency: int | None = Field(default=None, ge=1)
    request_timeout: int = Field(default=DEFAULT_REQUEST_TIMEOUT, ge=1)
    request_retries: int = Field(default=20, ge=1)
    base_url: str | None = Field(default=None, exclude=True)

    @field_validator("datasets")
    @classmethod
    def _datasets_not_empty(
        cls, value: dict[str, TokenNllDatasetConfig]
    ) -> dict[str, TokenNllDatasetConfig]:
        if not value:
            raise ValueError("token_nll: datasets can't be empty")
        return value

    def apply_defaults(
        self, *, inference_instances: int, base_url: str, **_: Any
    ) -> None:
        self.base_url = base_url
        if self.concurrency is None:
            self.concurrency = (
                inference_instances * TOKEN_NLL_DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE
            )


class CheckpointStatsEvaluatorConfig(MetricsEvaluatorConfig):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    device: str = Field(default="cpu", pattern=r"^(cpu|cuda(:\d+)?)$")
    num_threads: int = Field(default=16, ge=1)
    role_overrides: dict[str, str] = Field(default_factory=dict)
    topk_sigma: int = Field(default=16, ge=0)
    svd_niter: int = Field(default=2, ge=1)
    spectral_numel_cap: int = Field(default=20 * 1024**3 // 4, ge=1)
    include: str | None = None
    exclude: str | None = None
    hist_bins: int = Field(default=80, ge=1)
    hist_log_min: float = -8.0
    hist_log_max: float = 4.0

    @field_validator("include", "exclude")
    @classmethod
    def _valid_regex(cls, value: str | None) -> str | None:
        if value is not None:
            try:
                re.compile(value)
            except re.error as error:
                raise ValueError(f"invalid regex {value!r}: {error}") from error
        return value

    @field_validator("role_overrides")
    @classmethod
    def _valid_role_overrides(cls, value: dict[str, str]) -> dict[str, str]:
        for pattern, role in value.items():
            cls._valid_regex(pattern)
            if not re.fullmatch(r"[a-z][a-z0-9_]*", role):
                raise ValueError(
                    "role_overrides roles must use lowercase letters, digits and underscores"
                )
        return value

    @model_validator(mode="after")
    def _hist_range(self) -> "CheckpointStatsEvaluatorConfig":
        if self.hist_log_max <= self.hist_log_min:
            raise ValueError("hist_log_max must be greater than hist_log_min")
        return self


class EvaluatorsConfig(BaseConfigModel):
    lm_eval: LmEvalEvaluatorConfig | None = None
    lmms_eval: LmmsEvalEvaluatorConfig | None = None
    mera: MeraEvaluatorConfig | None = None
    obm: ObmEvaluatorConfig | None = None
    obm_judged: ObmJudgedEvaluatorConfig | None = None
    new_obm: ObmEvaluatorConfig | None = None
    new_obm_judged: ObmJudgedEvaluatorConfig | None = None
    bfcl: BfclEvaluatorConfig | None = None
    dredd: DreddEvaluatorConfig | None = None
    tau2: Tau2EvaluatorConfig | None = None
    tau3: Tau3EvaluatorConfig | None = None
    forge: ForgeEvaluatorConfig | None = None
    gigametric: GigaMetricEvaluatorConfig | None = None
    planbench: PlanbenchEvaluatorConfig | None = None
    gigametric_judged: GigaMetricJudgedEvaluatorConfig | None = None
    safety_unified: SafetyUnifiedEvaluatorConfig | None = None
    terminal_bench: TerminalBenchEvaluatorConfig | None = None
    harbor_rli: HarborRliEvaluatorConfig | None = None
    pier_rli: PierRliEvaluatorConfig | None = None
    acc_len: AccLenEvaluatorConfig | None = None
    skillsbench: SkillsBenchEvaluatorConfig | None = None
    swe_bench: SweBenchEvaluatorConfig | None = None
    nl2repo: Nl2RepoEvaluatorConfig | None = None
    appworld: AppWorldEvaluatorConfig | None = None
    claweval: ClawEvalEvaluatorConfig | None = None
    aa_lcr: AaLcrEvaluatorConfig | None = None
    token_nll: TokenNllEvaluatorConfig | None = None
    checkpoint_stats: CheckpointStatsEvaluatorConfig | None = None

    @model_validator(mode="after")
    def _forbid_obm_with_new_obm(self) -> "EvaluatorsConfig":
        if self.obm is not None and self.new_obm is not None:
            raise ValueError(
                "autometrics.evaluators: obm and new_obm cannot run together"
            )
        return self

    def __iter__(self) -> Iterator[tuple[str, MetricsEvaluatorConfig]]:
        for evaluator_name in type(self).model_fields:
            evaluator_cfg = getattr(self, evaluator_name)
            if evaluator_cfg is not None:
                yield evaluator_name, evaluator_cfg


class AutometricsConfig(BaseConfigModel):
    globals: GlobalsConfig
    router: RouterConfig
    inference: InferenceConfigUnion = Field(default_factory=SglangInferenceConfig)
    obm_proxy: ObmProxyConfig = Field(default_factory=ObmProxyConfig)
    evaluators: EvaluatorsConfig
    preparation: PreparationConfig
    quantization: QuantizationConfig | None = None
    convertation: ConvertationConfig
    model_info: ModelInfoConfig
    judge_evaluators: dict[str, Any] = Field(default_factory=dict)
    secondary_inference: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @field_validator("inference", mode="before")
    @classmethod
    def _detect_pd_inference(cls, value: Any) -> Any:
        if isinstance(value, dict) and SglangPdInferenceConfig.wants_pd(value):
            return {**value, "engine_type": "sglang_pd"}
        return value

    @field_serializer("secondary_inference")
    def _drop_secondary_inference_secrets(
        self,
        value: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        return {
            platform: {
                key: param
                for key, param in params.items()
                if key not in {"api_key", "api-key", "user_pass", "user-pass"}
            }
            for platform, params in value.items()
        }

    @property
    def uses_external_api(self) -> bool:
        return self.inference.uses_external_api

    @property
    def has_inference_reasoning_parser(self) -> bool:
        return self.inference.has_reasoning_parser

    @property
    def has_inference_tool_call_parser(self) -> bool:
        return self.inference.has_tool_call_parser

    @property
    def has_inference_chat_template(self) -> bool:
        return self.inference.has_chat_template

    @property
    def has_obm_proxy_chat_template(self) -> bool:
        return self.obm_proxy.chat_template is not None

    @property
    def has_obm_proxy_chat_template_kwargs(self) -> bool:
        return bool(self.obm_proxy.chat_template_kwargs)

    @property
    def wants_opensource_formatting(self) -> bool:
        return any(
            (
                self.has_inference_reasoning_parser,
                self.has_inference_tool_call_parser,
                self.has_inference_chat_template,
            )
        )

    @property
    def wants_custom_formatting(self) -> bool:
        return self.has_obm_proxy_chat_template

    @property
    def has_lm_eval_instruct(self) -> bool:
        lm_eval = self.evaluators.lm_eval
        return lm_eval is not None and bool(lm_eval.instruct)

    @property
    def has_harness_bench(self) -> bool:
        harbor_rli = self.evaluators.harbor_rli
        return harbor_rli is not None and any(
            run.benchmark in ("harness-bench-lite", "harness-bench-medium")
            for run in (harbor_rli.runs or [harbor_rli])
        )

    @property
    def resolved_chat_handlers_mode(
        self,
    ) -> Literal["opensource", "custom", "pretrain"]:
        if self.uses_external_api:
            return "opensource"
        if self.wants_opensource_formatting:
            return "opensource"
        if self.wants_custom_formatting:
            return "custom"
        return "pretrain"

    @model_validator(mode="after")
    def _sync_chat_handlers_mode(self) -> "AutometricsConfig":
        self.obm_proxy.chat_handlers_mode = self.resolved_chat_handlers_mode
        return self

    @model_validator(mode="after")
    def _sync_tokenizer_path(self) -> "AutometricsConfig":
        if self.uses_external_api:
            return self

        obm_proxy_tokenizer_path = self.obm_proxy.tokenizer_path
        inference_tokenizer_path = self.inference.tokenizer_path

        if obm_proxy_tokenizer_path is None and inference_tokenizer_path is None:
            return self

        if obm_proxy_tokenizer_path is None:
            self.obm_proxy.tokenizer_path = inference_tokenizer_path
            return self

        if inference_tokenizer_path is None:
            self.inference.tokenizer_path = obm_proxy_tokenizer_path
            return self

        if inference_tokenizer_path != obm_proxy_tokenizer_path:
            raise ValueError(
                "Conflicting tokenizer configuration: "
                "autometrics.obm_proxy.tokenizer_path and "
                "autometrics.inference.tokenizer_path differ. "
                "Use one value for both."
            )

        return self

    @model_validator(mode="after")
    def _validate_chat_formatting_strategy(self) -> "AutometricsConfig":
        if self.uses_external_api:
            return self

        if self.wants_opensource_formatting and self.wants_custom_formatting:
            raise ValueError(
                "Conflicting chat formatting configuration: "
                "opensource formatting was requested via autometrics.inference "
                "(reasoning_parser, tool_call_parser, or chat_template), "
                "but autometrics.obm_proxy.chat_template was also provided. "
                "Use exactly one strategy: either inference-based opensource formatting "
                "or obm_proxy-based custom formatting."
            )

        if (
            self.has_obm_proxy_chat_template_kwargs
            and not self.has_obm_proxy_chat_template
            and not self.wants_opensource_formatting
        ):
            raise ValueError(
                "autometrics.obm_proxy.chat_template_kwargs were provided without "
                "a chat formatting source. "
                "Provide autometrics.obm_proxy.chat_template for custom mode, "
                "or set inference-based opensource formatting "
                "(reasoning_parser/tool_call_parser/chat_template)."
            )

        if self.has_lm_eval_instruct and self.resolved_chat_handlers_mode == "pretrain":
            raise ValueError(
                "autometrics.evaluators.lm_eval.instruct requires chat formatting. "
                "Provide inference-based opensource formatting "
                "(reasoning_parser/tool_call_parser/chat_template) "
                "or provide autometrics.obm_proxy.chat_template for custom mode."
            )

        if self.has_harness_bench and self.resolved_chat_handlers_mode == "pretrain":
            raise ValueError(
                "autometrics.evaluators.harbor_rli is an agentic, tool-calling "
                "benchmark and requires chat formatting. "
                "Provide inference-based opensource formatting "
                "(reasoning_parser/tool_call_parser/chat_template) "
                "or provide autometrics.obm_proxy.chat_template for custom mode."
            )

        return self

    @model_validator(mode="after")
    def _validate_lmms_eval_chatml_version(self) -> "AutometricsConfig":
        lmms_eval = self.evaluators.lmms_eval
        if (
            lmms_eval is not None
            and not self.uses_external_api
            and lmms_eval.chatml_version is None
        ):
            raise ValueError(
                "autometrics.evaluators.lmms_eval.chatml_version is required "
                "for local inference"
            )
        return self

    @model_validator(mode="after")
    def _validate_external_api_scope(self) -> "AutometricsConfig":
        if not self.uses_external_api:
            return self

        unsupported_evaluators = self._unsupported_external_api_evaluators()
        if unsupported_evaluators:
            raise ValueError(
                "autometrics.inference.engine_type=external_api does not support "
                f"{', '.join(unsupported_evaluators)}"
            )
        return self

    @model_validator(mode="after")
    def _validate_secondary_models(self) -> "AutometricsConfig":
        for evaluator_name, evaluator_config in self.evaluators:
            for model_name in evaluator_config.secondary_model_names:
                self._validate_secondary_model(evaluator_name, model_name)
        return self

    def _validate_secondary_model(self, evaluator_name: str, model_name: str) -> None:
        platform, sep, model = model_name.partition(":")
        try:
            if sep:
                ExternalApiInferenceConfig.for_secondary_platform(
                    platform, model, self.secondary_inference
                )
            else:
                SglangInferenceConfig.for_secondary_preset(
                    model_name, self.secondary_inference
                )
        except ValueError as e:
            raise ValueError(
                f"autometrics.evaluators.{evaluator_name}: {model_name!r}: {e}"
            ) from e

    def _unsupported_external_api_evaluators(self) -> list[str]:
        unsupported = []
        if self.evaluators.lm_eval and self.evaluators.lm_eval.pretrain:
            unsupported.append("autometrics.evaluators.lm_eval.pretrain")
        # Reads the acceptance counters off the SGLang workers directly.
        if self.evaluators.acc_len:
            unsupported.append("autometrics.evaluators.acc_len")
        for evaluator_name in self.evaluators.model_extra:
            unsupported.append(f"autometrics.evaluators.{evaluator_name}")
        return unsupported


class NodeRunnerConfigModel(BaseConfigModel):
    run_name: str
    tensorboard_path: Path | None = None
    max_seq_len: int = Field(ge=1)
    autometrics: AutometricsConfig

    @classmethod
    def load(cls, config_path: Path) -> "NodeRunnerConfigModel":
        return cls.model_validate(
            OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        )

    @property
    def is_external_api_mode(self) -> bool:
        return self.autometrics.uses_external_api

    @property
    def is_replay_mode(self) -> bool:
        return self.autometrics.globals.replay_generations_dir is not None

    @property
    def runs_without_checkpoint(self) -> bool:
        return self.is_external_api_mode or self.is_replay_mode
