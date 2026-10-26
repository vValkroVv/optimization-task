from __future__ import annotations

from base64 import b64decode
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from job_submitter.asserts.base import BaseJobAssertor
from job_submitter.core.job_config import JobTemplate

if TYPE_CHECKING:
    from job_submitter.jobs.llmf.autometrics import AutometricsJobLLMF
    from job_submitter.jobs.pam.autometrics_external_api import (
        AutometricsJobPAMExternalAPI,
    )


class AutometricsExternalAPIAssertor(BaseJobAssertor):
    def _assert_non_empty_string(self, value: object, key: str) -> None:
        assert isinstance(value, str) and value.strip() and value.strip() != "...", (
            self._pretty_config_param_print({key: ["<non-empty string>"]})
        )

    def _assert_config_path(self, path_to_config: Optional[str]) -> Path:
        assert path_to_config, self._pretty_params_print(
            {"path_to_config": ["/valid/path/to/config.yaml"]}
        )
        config_path = Path(path_to_config)
        assert config_path.exists() and config_path.is_file(), (
            self._pretty_params_print(
                {"path_to_config": ["/valid/path/to/config.yaml"]}
            )
        )
        return config_path

    def _assert_inference(self, inference_params: dict) -> None:
        assert isinstance(inference_params, dict), self._pretty_config_param_print(
            {
                "autometrics.inference": [
                    {
                        "engine_type": "external_api",
                        "base_url": "https://gigachat.ift.sberdevices.ru",
                        "model": "<model>",
                        "api_key": "<api key>",
                        "user_pass": "<user:password>",
                    }
                ]
            }
        )
        assert "engine_type" in inference_params, self._pretty_print_missing_key(
            "autometrics.inference.engine_type"
        )
        assert inference_params["engine_type"] == "external_api", (
            self._pretty_config_param_print(
                {"autometrics.inference.engine_type": ["external_api"]}
            )
        )

        for key in ("base_url", "model"):
            assert key in inference_params, self._pretty_print_missing_key(
                f"autometrics.inference.{key}"
            )
            self._assert_non_empty_string(
                inference_params[key], f"autometrics.inference.{key}"
            )
        assert inference_params["base_url"].startswith(("http://", "https://")), (
            self._pretty_config_param_print(
                {
                    "autometrics.inference.base_url": [
                        "https://gigachat.ift.sberdevices.ru"
                    ]
                }
            )
        )

        if "formatting" in inference_params:
            self._assert_non_empty_string(
                inference_params["formatting"], "autometrics.inference.formatting"
            )

        for key in ("api_key", "user_pass"):
            if key in inference_params:
                self._assert_non_empty_string(
                    inference_params[key], f"autometrics.inference.{key}"
                )

        if inference_params.get("formatting") == "openai":
            assert "api_key" in inference_params, self._pretty_print_missing_key(
                "autometrics.inference.api_key"
            )
        else:
            assert "api_key" in inference_params or "user_pass" in inference_params, (
                self._pretty_config_param_print(
                    {
                        "autometrics.inference.api_key": ["<api key>"],
                        "autometrics.inference.user_pass": ["<user:password>"],
                    }
                )
            )

        if "max_concurrency" in inference_params:
            assert (
                isinstance(inference_params["max_concurrency"], int)
                and inference_params["max_concurrency"] > 0
            ), self._pretty_config_param_print(
                {"autometrics.inference.max_concurrency": [">0"]}
            )

    def _assert_external_api_config(self, config_path: Path) -> None:
        config = self._load_config(config_path)
        assert "run_name" in config, self._pretty_print_missing_key("run_name")
        assert "exp_name" in config, self._pretty_print_missing_key("exp_name")
        assert "user" in config, self._pretty_print_missing_key("user")
        assert "autometrics" in config, self._pretty_print_missing_key("autometrics")
        autometrics_params = config["autometrics"]
        assert "inference" in autometrics_params, self._pretty_print_missing_key(
            "autometrics.inference"
        )
        self._assert_inference(autometrics_params["inference"])
        assert "evaluators" in autometrics_params, self._pretty_print_missing_key(
            "autometrics.evaluators"
        )
        evaluators_params = autometrics_params["evaluators"]
        assert evaluators_params, self._pretty_config_param_print(
            {"autometrics.evaluators": ["<at least one evaluator>"]}
        )

    def assert_job(
        self,
        job: "AutometricsJobPAMExternalAPI",
        template: JobTemplate,
    ) -> None:
        super().assert_job(job, template)
        assert template.submission_type == "cloud", self._pretty_params_print(
            {"submission_type": ["cloud"]}
        )
        config_path = self._assert_config_path(job.path_to_config)
        self._assert_external_api_config(config_path)


class AutometricsJobAssertor(BaseJobAssertor):
    def _assert_checkpoint_dir(self, checkpoint_dir: str) -> None:
        assert checkpoint_dir, "Please provide a valid path to the checkpoint directory"
        assert Path(checkpoint_dir).exists(), (
            f"Checkpoint directory {checkpoint_dir} does not exist"
        )

    def _assert_mode(self, mode: str) -> None:
        assert mode in (
            "autometrics",
            "convertation",
            "inference",
        ), self._pretty_params_print(
            {"mode": ["autometrics", "convertation", "inference"]}
        )

    def _assert_globals(self, globals_params: dict) -> None:
        assert "delete_converted_hf" in globals_params, self._pretty_print_missing_key(
            "autometrics.globals.delete_converted_hf"
        )

    def _assert_router(self, router_params: dict) -> None:
        assert "host" in router_params, self._pretty_print_missing_key(
            "autometrics.router.host"
        )

    def _assert_convertation(self, convertation_params: dict) -> None:
        assert "mappers" in convertation_params, self._pretty_print_missing_key(
            "autometrics.convertation.mappers"
        )
        assert "input_tp_size" in convertation_params, self._pretty_print_missing_key(
            "autometrics.convertation.input_tp_size"
        )
        mappers = convertation_params["mappers"]
        allowed_single_mappers = (
            "map_gigafsdp_to_hf_safetensor",
            "map_gigafsdp_to_hf",
            "map_mixed_gigafsdp_to_mtp_safetensors",
            "map_safetensors_graft_mtp_safetensors",
            "map_gigafsdp_to_hf_omni_safetensor",
            "map_gfsdp2_dcp_to_hf",
            "map_hf_safetensor_to_local",
            "map_mixed_gigafsdp_lora_to_hf_merged_model",
            # LoRA omni runs: same streaming path as the plain omni mapper, but
            # merges the adapter into the base weights. Without it PAM would
            # evaluate the base model and silently report pre-LoRA numbers.
            "map_gigafsdp_lora_to_hf_omni_safetensor",
        )
        allowed_two_mappers = (
            "map_mixed_gigafsdp_to_fsdp",
            "map_fsdp_to_hf",
        )
        is_valid = (
            len(mappers) == 1 and mappers[0] in allowed_single_mappers
        ) or mappers == list(allowed_two_mappers)
        assert is_valid, self._pretty_config_param_print(
            {
                "mappers": [
                    ["map_gigafsdp_to_hf_safetensor"],
                    ["map_gigafsdp_to_hf"],
                    ["map_mixed_gigafsdp_to_mtp_safetensors"],
                    ["map_safetensors_graft_mtp_safetensors"],
                    ["map_gigafsdp_to_hf_omni_safetensor"],
                    ["map_hf_safetensor_to_local"],
                    ["map_mixed_gigafsdp_lora_to_hf_merged_model"],
                    ["map_gigafsdp_lora_to_hf_omni_safetensor"],
                    list(allowed_two_mappers),
                ]
            }
        )

        if mappers == ["map_safetensors_graft_mtp_safetensors"]:
            assert convertation_params.get("mtp_donor_path"), (
                self._pretty_print_missing_key(
                    "autometrics.convertation.mtp_donor_path"
                )
            )

    def _assert_preparation(self, preparation_params: dict) -> None:
        assert "skip" in preparation_params, self._pretty_print_missing_key(
            "autometrics.preparation.skip"
        )
        assert preparation_params["skip"] is True, "preparation.skip should be True"

    def _assert_model_info(self, model_info_params: dict) -> None:
        assert "model_class" in model_info_params, self._pretty_print_missing_key(
            "autometrics.model_info.model_class"
        )
        assert "size" in model_info_params, self._pretty_print_missing_key(
            "autometrics.model_info.size"
        )
        assert "username" in model_info_params, self._pretty_print_missing_key(
            "autometrics.model_info.username"
        )
        assert model_info_params["model_class"] in (
            "sft",
            "dpo",
            "vision",
            "audio",
            "grpo",
        ), self._pretty_config_param_print(
            {
                "autometrics.model_info.model_class": [
                    "sft",
                    "dpo",
                    "vision",
                    "audio",
                    "grpo",
                ]
            }
        )
        assert model_info_params["size"] in (
            "lite",
            "pro",
            "max",
            "ultra",
            "custom",
        ), self._pretty_config_param_print(
            {
                "autometrics.model_info.size": [
                    "lite",
                    "pro",
                    "max",
                    "ultra",
                    "custom",
                ]
            }
        )

    def _assert_obm_evaluator(
        self, obm_evaluator_params: Optional[dict], job: "AutometricsJobLLMF"
    ) -> None:
        if job.mode != "autometrics":
            return
        if obm_evaluator_params is None:
            return
        assert "obm_config" in obm_evaluator_params, self._pretty_print_missing_key(
            "autometrics.evaluators.obm.obm_config"
        )
        assert "all" in obm_evaluator_params, self._pretty_print_missing_key(
            "autometrics.evaluators.obm.all"
        )
        obm_config_params = obm_evaluator_params["obm_config"]
        if isinstance(obm_config_params, str):
            assert obm_config_params in (
                "devices",
                "devices-v2",
                "openai",
            ), self._pretty_config_param_print(
                {
                    "autometrics.evaluators.obm.obm_config": [
                        "devices",
                        "devices-v2",
                        "openai",
                    ]
                }
            )
        elif isinstance(obm_config_params, dict):
            assert (
                "devices" in obm_config_params
                or "devices-v2" in obm_config_params
                or "openai" in obm_config_params
            ), self._pretty_config_param_print(
                {
                    "autometrics.evaluators.obm.obm_config": [
                        "devices",
                        "devices-v2",
                        "openai",
                    ]
                }
            )
        else:
            assert False, "obm_config should be a string or a dictionary"

        obm_all_params = obm_evaluator_params.get("all", {})
        assert isinstance(obm_all_params, dict), self._pretty_config_param_print(
            {"autometrics.evaluators.obm.all.groups": [["assistant", "b2c"]]}
        )
        assert "group" not in obm_all_params, self._pretty_config_param_print(
            {"autometrics.evaluators.obm.all.groups": [["assistant", "b2c"]]},
            "legacy `group` key is not supported; use `groups` list",
        )

        groups = obm_all_params.get("groups", [])
        assert isinstance(groups, list), self._pretty_config_param_print(
            {"autometrics.evaluators.obm.all.groups": [["assistant", "b2c"]]}
        )
        for group in groups:
            assert isinstance(group, str), self._pretty_config_param_print(
                {"autometrics.evaluators.obm.all.groups": [["assistant", "b2c"]]},
                "all groups must be strings",
            )
            assert group == group.lower(), self._pretty_config_param_print(
                {"autometrics.evaluators.obm.all.groups": [[["assistant", "b2c"]]]},
                "all groups must be lowercase",
            )
        if job.submission_type == "slurm":
            forbidden_group = b64decode("U0JFUg==").decode().lower()
            assert forbidden_group not in groups, self._pretty_config_param_print(
                {"autometrics.evaluators.obm.all.groups": [groups]},
                "Please remove $B3R group from config and use papaya instead",
            )

    def _is_obm_config(self, obm_evaluator: Optional[dict], kind: str) -> bool:
        if obm_evaluator is None:
            return False
        obm_config = obm_evaluator["obm_config"]
        return obm_config == kind or (
            isinstance(obm_config, dict) and kind in obm_config
        )

    @staticmethod
    def _get_any_key(params: dict, keys: tuple[str, ...]):
        for key in keys:
            if key in params:
                return params[key]
        return None

    def _assert_obm_proxy(
        self,
        obm_proxy_params: dict,
        job: "AutometricsJobLLMF",
    ) -> None:
        if job.mode == "convertation":
            return

        chat_template = self._get_any_key(
            obm_proxy_params,
            ("chat-template", "chat_template"),
        )
        if chat_template is not None:
            assert isinstance(chat_template, str), self._pretty_config_param_print(
                {"autometrics.obm_proxy.chat-template": ["<path/to/template.base64>"]}
            )

            chat_template_path = Path(chat_template)
            assert chat_template_path.exists() and chat_template_path.is_file(), (
                self._pretty_config_param_print(
                    {
                        "autometrics.obm_proxy.chat-template": [
                            "<path/to/template.base64>"
                        ]
                    },
                    "chat-template must be a valid path",
                )
            )

        if job.submission_type != "slurm":
            return

        if chat_template is not None:
            assert self._is_encoded_in_base64(chat_template_path.read_bytes()), (
                self._pretty_config_param_print(
                    {
                        "autometrics.obm_proxy.chat-template": [
                            "<path/to/template.base64>"
                        ]
                    },
                    "chat-template file content must be encoded in base64 for Slurm submission, PLEASE ENCODE IT IN BASE64 AND REMOVE UNENCODED VERSION FROM NFS",
                )
            )

        chat_template_kwargs = self._get_any_key(
            obm_proxy_params,
            ("chat-template-kwargs", "chat_template_kwargs"),
        )
        if not chat_template_kwargs:
            return

        assert isinstance(chat_template_kwargs, dict), self._pretty_config_param_print(
            {
                "autometrics.obm_proxy.chat-template-kwargs": [
                    {"default_devsystem": "<some string in base64>"}
                ]
            }
        )

        default_devsystem = chat_template_kwargs.get("default_devsystem")
        if default_devsystem:
            assert self._is_encoded_in_base64(default_devsystem), (
                self._pretty_config_param_print(
                    {
                        "autometrics.obm_proxy.chat-template-kwargs.default_devsystem": [
                            "<some string in base64>"
                        ]
                    },
                    "default_devsystem must be encoded in base64 for Slurm submission, PLEASE ENCODE IT IN BASE64 AND REMOVE UNENCODED VERSION FROM NFS",
                )
            )

    def _assert_instruct_formatting_requirements(
        self,
        evaluators_params: dict,
        inference_params: dict,
        obm_proxy_params: dict,
    ) -> None:
        lm_eval_cfg = evaluators_params.get("lm_eval")
        has_lm_eval_instruct = isinstance(lm_eval_cfg, dict) and bool(
            lm_eval_cfg.get("instruct")
        )
        has_mera = bool(evaluators_params.get("mera"))
        has_dredd = evaluators_params.get("dredd") is not None
        has_obm = evaluators_params.get("obm") is not None
        has_instruct_metrics = has_lm_eval_instruct or has_mera or has_dredd or has_obm

        if not has_instruct_metrics:
            return

        reasoning_parser = self._get_any_key(
            inference_params,
            ("reasoning-parser", "reasoning_parser"),
        )
        tool_call_parser = self._get_any_key(
            inference_params,
            ("tool-call-parser", "tool_call_parser"),
        )
        has_inference_parsers = bool(reasoning_parser) or bool(tool_call_parser)

        obm_proxy_chat_template = self._get_any_key(
            obm_proxy_params,
            ("chat-template", "chat_template"),
        )
        has_obm_proxy_template = bool(obm_proxy_chat_template)

        assert has_inference_parsers or has_obm_proxy_template, (
            self._pretty_config_param_print(
                {
                    "autometrics.inference.reasoning-parser": [
                        "https://docs.sglang.io/advanced_features/server_arguments.html"
                    ],
                    "autometrics.inference.tool-call-parser": [
                        "https://docs.sglang.io/advanced_features/server_arguments.html"
                    ],
                    "autometrics.obm_proxy.chat-template": [
                        "/some/valid/path/to/jinja/template.jinja"
                    ],
                },
                "For instruct metrics (lm_eval.instruct/dredd/mera/obm), provide EITHER inference parsers (reasoning-parser or tool-call-parser) OR obm_proxy.chat-template",
            )
        )
        assert not (has_inference_parsers and has_obm_proxy_template), (
            self._pretty_config_param_print(
                {
                    "autometrics.inference.reasoning-parser": ["deepseek-r1"],
                    "autometrics.inference.tool-call-parser": ["qwen", "deepseekv3"],
                    "autometrics.obm_proxy.chat-template": [
                        "/some/valid/path/to/jinja/template.jinja"
                    ],
                },
                "Use only one formatting strategy for instruct metrics: inference parsers OR obm_proxy.chat-template",
            )
        )

        if has_obm_proxy_template:
            assert isinstance(obm_proxy_chat_template, str), (
                self._pretty_config_param_print(
                    {
                        "autometrics.obm_proxy.chat-template": [
                            "/some/valid/path/to/jinja/template.jinja"
                        ]
                    },
                    "obm_proxy.chat-template must be a string path",
                )
            )
            assert Path(obm_proxy_chat_template).exists(), (
                self._pretty_config_param_print(
                    {
                        "autometrics.obm_proxy.chat-template": [
                            "/some/valid/path/to/jinja/template.jinja"
                        ]
                    },
                    "obm_proxy.chat-template must be a valid path to a Jinja template",
                )
            )

        obm_evaluator = evaluators_params.get("obm")
        if obm_evaluator is None:
            return

        if has_inference_parsers:
            assert self._is_obm_config(obm_evaluator, "openai"), (
                self._pretty_config_param_print(
                    {"autometrics.evaluators.obm.obm_config": ["openai"]},
                    "obm_config must be `openai` when using inference parsers (reasoning-parser/tool-call-parser)",
                )
            )

    def _assert_inference_workers(
        self,
        valid_workers: list[int],
        job: "AutometricsJobLLMF",
        config: dict,
        source_job: str = "standalone",
    ) -> None:
        if "callbacks" in config:
            callbacks = config["callbacks"]
            assert "autometrics" in callbacks, self._pretty_print_missing_key(
                "autometrics"
            )
            autometrics = callbacks["autometrics"]
            assert "nnodes" in autometrics, self._pretty_print_missing_key(
                "autometrics.nnodes"
            )
            assert autometrics["nnodes"] in valid_workers, (
                self._pretty_config_param_print(
                    {"callbacks.autometrics.nnodes": valid_workers},
                )
            )
        else:
            if job.n_workers not in valid_workers:
                if source_job == "RayTrainJobVERL":
                    assert False, self._pretty_params_print(
                        {"pam_n_workers": valid_workers}, "RayTrainJobVERL"
                    )
                else:
                    assert False, self._pretty_params_print(
                        {"n_workers": valid_workers}
                    )

    def _assert_inference_tp_workers(
        self,
        total_gpus: int,
        job: "AutometricsJobLLMF",
        config: dict,
        source_job: str = "standalone",
    ) -> None:
        required_workers = total_gpus // job.n_gpus
        valid_workers = [
            required_workers * k for k in range(1, (128 // required_workers) + 1)
        ]
        self._assert_inference_workers(valid_workers, job, config, source_job)

    def _assert_inference_tp_and_forbidden_params(
        self,
        inference_params: dict,
        job: "AutometricsJobLLMF",
    ) -> None:
        valid_tp = (1, 2, 4, 8, 16, 32, 64)
        assert "tp" in inference_params, self._pretty_print_missing_key(
            "autometrics.inference.tp"
        )
        assert inference_params["tp"] in valid_tp, self._pretty_config_param_print(
            {"autometrics.inference.tp": list(valid_tp)}
        )
        assert "ep-size" not in inference_params, self._pretty_config_param_print(
            {"ep": list(valid_tp)}
        )
        assert "dp-size" not in inference_params, self._pretty_config_param_print(
            {"dp": [1, 2, 4, 8]}
        )
        assert "pp-size" not in inference_params, self._pretty_config_param_print(
            {"pp": [1, 2, 4, 8]}
        )
        assert "tp-size" not in inference_params, self._pretty_config_param_print(
            {"tp": list(valid_tp)}
        )
        if "ep" in inference_params and self._is_deepep_enabled(inference_params):
            assert inference_params["ep"] == inference_params["tp"], (
                self._pretty_config_param_print(
                    {"autometrics.inference.ep": list(valid_tp)},
                    "ep should be equal to tp when moe-a2a-backend is deepep",
                )
            )

    def _is_deepep_enabled(self, inference_params: dict) -> bool:
        moe_a2a_backend = self._get_any_key(
            inference_params,
            ("moe-a2a-backend", "moe_a2a_backend"),
        )
        return moe_a2a_backend == "deepep"

    def _assert_instance_parallelism(
        self,
        inference_params: dict,
        job: "AutometricsJobLLMF",
    ) -> None:
        tp = inference_params["tp"]
        dp_attention = bool(inference_params.get("enable-dp-attention"))
        if "dp" in inference_params:
            dp = inference_params["dp"]
            if dp_attention:
                assert tp % dp == 0, self._pretty_config_param_print(
                    {"autometrics.inference.dp": [1, 2, 4, 8, 16, 32, 64]},
                    f"dp({dp}) must divide tp({tp}) when enable-dp-attention is set",
                )
            else:
                assert "pp" not in inference_params, self._pretty_config_param_print(
                    {"pp": []}, "wrong param with dp"
                )
                assert dp in (
                    1,
                    2,
                    4,
                    8,
                ), self._pretty_config_param_print(
                    {"autometrics.inference.dp": [1, 2, 4, 8]}
                )
                assert tp * dp == job.n_gpus, self._pretty_config_param_print(
                    {"tp": [1, 2, 4, 8], "dp": [1, 2, 4, 8]},
                    f"tp*dp should be equal to n_gpus({job.n_gpus}) in AutometricsJob",
                )
        if "pp" in inference_params:
            assert inference_params["pp"] in (
                1,
                2,
                4,
                8,
            ), self._pretty_config_param_print(
                {"autometrics.inference.pp": [1, 2, 4, 8]}
            )
            if not dp_attention:
                assert "dp" not in inference_params, self._pretty_config_param_print(
                    {"dp": []}, "wrong param with pp"
                )
            assert job.n_gpus == 8, self._pretty_params_print({"n_gpus": [8]})
            gpus_needed = tp * inference_params["pp"]
            assert gpus_needed % job.n_gpus == 0, self._pretty_config_param_print(
                {"tp": [1, 2, 4, 8, 16, 32, 64], "pp": [1, 2, 4, 8]},
                f"tp*pp must be divisible by {job.n_gpus} (GPUs per node). "
                f"Got tp*pp={gpus_needed}.",
            )
        if tp > 8:
            assert job.n_gpus == 8, self._pretty_params_print({"n_gpus": [8]})
            assert tp % job.n_gpus == 0, self._pretty_config_param_print(
                {"tp": [1, 2, 4, 8, 16, 32, 64]},
                f"tp must be divisible by {job.n_gpus} (GPUs per node). Got tp={tp}.",
            )
        elif "dp" not in inference_params and "pp" not in inference_params:
            assert tp == job.n_gpus, self._pretty_config_param_print(
                {"tp": [1, 2, 4, 8, 16, 32, 64]},
                f"tp must be equal to {job.n_gpus} (GPUs per node). Got tp={tp}.",
            )

    def _assert_flat_worker_count(
        self,
        inference_params: dict,
        job: "AutometricsJobLLMF",
        config: dict,
        source_job: str = "standalone",
    ) -> None:
        tp = inference_params["tp"]
        if "pp" in inference_params:
            gpus_needed = tp * inference_params["pp"]
            assert gpus_needed % job.n_workers == 0, self._pretty_config_param_print(
                {"tp": [1, 2, 4, 8, 16, 32, 64], "pp": [1, 2, 4, 8]},
                f"tp*pp must be divisible by {job.n_workers} (workers). "
                f"Got tp*pp={gpus_needed}.",
            )
            self._assert_inference_tp_workers(gpus_needed, job, config, source_job)
        if tp > 8:
            self._assert_inference_tp_workers(tp, job, config, source_job)

    def _assert_inference_parallelism(
        self,
        inference_params: dict,
        job: "AutometricsJobLLMF",
        config: dict,
        source_job: str = "standalone",
    ) -> None:
        self._assert_instance_parallelism(inference_params, job)
        self._assert_flat_worker_count(inference_params, job, config, source_job)

    @staticmethod
    def _is_pd_inference(inference_params: dict) -> bool:
        return "prefill" in inference_params and "decode" in inference_params

    @staticmethod
    def _evaluator_secondary_model_names(name: str, params: dict) -> list:
        if name in ("obm_judged", "new_obm_judged", "gigametric_judged"):
            judges = params.get("judges")
            return list(judges) if isinstance(judges, dict) else []
        if name in ("tau2", "tau3"):
            if "user_model_name" not in params and name == "tau3":
                return ["tau3-default-local-judge"]
            return [params.get("user_model_name")]
        if name == "forge":
            names = [params.get("judge")]
            for group in params.get("groups") or []:
                if isinstance(group, dict):
                    names.append(group.get("judge"))
            return names
        return []

    def _has_local_secondary_models(self, evaluators_params: dict) -> bool:
        for name in (
            "tau2",
            "tau3",
            "obm_judged",
            "new_obm_judged",
            "gigametric_judged",
            "forge",
        ):
            params = evaluators_params.get(name)
            if not isinstance(params, dict):
                continue
            for model_name in self._evaluator_secondary_model_names(name, params):
                if isinstance(model_name, str) and ":" not in model_name:
                    return True
        return False

    def _merged_pd_side_params(self, inference_params: dict, side: str) -> dict:
        side_params = inference_params.get(side)
        assert isinstance(side_params, dict), self._pretty_config_param_print(
            {f"autometrics.inference.{side}": [{"tp": 16, "ep": 16}]}
        )
        shared_params = (
            self._get_any_key(inference_params, ("shared_params", "shared-params"))
            or {}
        )
        return {**shared_params, **side_params}

    def _instance_nnodes(
        self, inference_params: dict, job: "AutometricsJobLLMF"
    ) -> int:
        gpus_needed = inference_params["tp"] * inference_params.get("pp", 1)
        return max(1, gpus_needed // job.n_gpus)

    def _assert_judged_flat_workers(
        self,
        inference_params: dict,
        job: "AutometricsJobLLMF",
        config: dict,
        source_job: str = "standalone",
    ) -> None:
        instance_nnodes = self._instance_nnodes(inference_params, job)
        valid_workers = [
            2 * instance_nnodes * count
            for count in range(1, 128 // (2 * instance_nnodes) + 1)
        ]
        self._assert_inference_workers(valid_workers, job, config, source_job)

    def _assert_pd_inference(
        self,
        inference_params: dict,
        job: "AutometricsJobLLMF",
        config: dict,
        source_job: str = "standalone",
    ) -> None:
        prefill_fraction = self._get_any_key(
            inference_params,
            ("prefill_fraction", "prefill-fraction"),
        )
        if prefill_fraction is not None:
            assert (
                isinstance(prefill_fraction, (int, float)) and 0 < prefill_fraction < 1
            ), self._pretty_config_param_print(
                {"autometrics.inference.prefill_fraction": ["float in (0, 1)"]}
            )

        prefill_params = self._merged_pd_side_params(inference_params, "prefill")
        decode_params = self._merged_pd_side_params(inference_params, "decode")
        for side_params in (prefill_params, decode_params):
            self._assert_inference_tp_and_forbidden_params(side_params, job)
            self._assert_instance_parallelism(side_params, job)

        prefill_nnodes = self._instance_nnodes(prefill_params, job)
        decode_nnodes = self._instance_nnodes(decode_params, job)
        valid_workers = [
            n_workers
            for n_workers in range(prefill_nnodes + decode_nnodes, 129)
            if any(
                (n_workers - prefill_count * prefill_nnodes) % decode_nnodes == 0
                for prefill_count in range(
                    1, (n_workers - decode_nnodes) // prefill_nnodes + 1
                )
            )
        ]
        evaluators_params = config.get("autometrics", {}).get("evaluators") or {}
        if self._has_local_secondary_models(evaluators_params):
            valid_workers = [
                2 * n_workers for n_workers in valid_workers if 2 * n_workers <= 128
            ]
        self._assert_inference_workers(valid_workers, job, config, source_job)

    def _assert_inference(
        self,
        inference_params: dict,
        _obm_evaluator: Optional[dict],
        job: "AutometricsJobLLMF",
        config: dict,
        source_job: str = "standalone",
    ) -> None:
        if job.mode == "convertation":
            return

        if self._is_pd_inference(inference_params):
            self._assert_pd_inference(inference_params, job, config, source_job)
            return

        self._assert_inference_tp_and_forbidden_params(inference_params, job)
        self._assert_inference_parallelism(inference_params, job, config, source_job)
        evaluators_params = config.get("autometrics", {}).get("evaluators") or {}
        if self._has_local_secondary_models(evaluators_params):
            self._assert_judged_flat_workers(inference_params, job, config, source_job)

    def _assert_pam_config(
        self, config: dict, job: "AutometricsJobLLMF", source_job: str = "standalone"
    ) -> None:
        assert "run_name" in config, self._pretty_print_missing_key("run_name")
        assert "exp_name" in config, self._pretty_print_missing_key("exp_name")
        assert "user" in config, self._pretty_print_missing_key("user")
        assert "tensorboard_path" in config, self._pretty_print_missing_key(
            "tensorboard_path"
        )
        assert "max_seq_len" in config, self._pretty_print_missing_key("max_seq_len")
        assert "autometrics" in config, self._pretty_print_missing_key("autometrics")

        autometrics = config["autometrics"]
        assert "globals" in autometrics, self._pretty_print_missing_key(
            "autometrics.globals"
        )
        globals_params = autometrics["globals"]
        self._assert_globals(globals_params)
        assert "router" in autometrics, self._pretty_print_missing_key(
            "autometrics.router"
        )
        router_params = autometrics["router"]
        self._assert_router(router_params)
        assert "convertation" in autometrics, self._pretty_print_missing_key(
            "autometrics.convertation"
        )
        convertation_params = autometrics["convertation"]
        self._assert_convertation(convertation_params)
        assert "preparation" in autometrics, self._pretty_print_missing_key(
            "autometrics.preparation"
        )
        preparation_params = autometrics["preparation"]
        self._assert_preparation(preparation_params)
        model_info_params = autometrics["model_info"]
        self._assert_model_info(model_info_params)
        assert "evaluators" in autometrics, self._pretty_print_missing_key(
            "autometrics.evaluators"
        )
        evaluators_params = autometrics["evaluators"]
        obm_evaluator = evaluators_params.get("obm")
        self._assert_obm_evaluator(obm_evaluator, job)
        obm_proxy_params = autometrics.get("obm_proxy", {})
        assert isinstance(obm_proxy_params, dict), self._pretty_config_param_print(
            {
                "autometrics.obm_proxy": [
                    {
                        "chat-template": "/some/valid/path/to/jinja/template.jinja",
                        "chat-template-kwargs": {"default_devsystem": "<base64>"},
                    }
                ]
            }
        )
        self._assert_obm_proxy(obm_proxy_params, job)
        assert "inference" in autometrics, self._pretty_print_missing_key(
            "autometrics.inference"
        )
        inference_params = autometrics["inference"]
        self._assert_inference(inference_params, obm_evaluator, job, config, source_job)
        if job.mode == "autometrics":
            formatting_inference_params = inference_params
            if self._is_pd_inference(inference_params):
                formatting_inference_params = self._merged_pd_side_params(
                    inference_params, "decode"
                )
            self._assert_instruct_formatting_requirements(
                evaluators_params,
                formatting_inference_params,
                obm_proxy_params,
            )

    def _assert_autometrics_config(
        self,
        path_to_config: Optional[str],
        job: "AutometricsJobLLMF",
        template: JobTemplate,
        source_job: str = "standalone",
    ) -> None:
        if path_to_config is None:
            path_to_config = Path(job.checkpoint_dir) / "artifacts" / "pam_config.yaml"
        path_to_config: Path = Path(path_to_config)
        assert path_to_config.exists() and path_to_config.is_file(), (
            self._pretty_params_print(
                {"path_to_config": ["/valid/path/to/config.yaml"]}
            )
        )
        loaded_config = self._load_config(path_to_config)
        self._assert_pam_config(loaded_config, job, source_job)

    def assert_job(
        self,
        job: "AutometricsJobLLMF",
        template: JobTemplate,
        source_job: str = "standalone",
    ) -> None:
        super().assert_job(job, template)
        self._assert_mode(job.mode)
        self._assert_checkpoint_dir(job.checkpoint_dir)
        self._assert_autometrics_config(job.path_to_config, job, template, source_job)
