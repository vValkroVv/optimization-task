from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import socket
from copy import deepcopy
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential

import requests
import wandb
from composer.callbacks.checkpoint_saver import get_current_checkpoint_save
from composer.core import Callback, State, Time, TimeUnit
from composer.loggers import Logger
from composer.utils import dist
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 60

# Probe target used to detect outbound internet access, which decides what job-dispatcher url is used.
# If this address is unreachable, JOB_DISPATCHER_BASE_URL is considered unavailable.
INTERNET_CHECK_HOST = "1.1.1.1"
INTERNET_CHECK_PORT = 53

HEADERS = {"Authorization": os.environ.get("JOB_DISPATCHER_TOKEN")}

JOBS_DICT = {
    "bbh": {
        "model_type": "vllm",
        "model_args": "dtype=bfloat16,trust_remote_code=True,max_num_seqs=8,max_model_len=8192,tensor_parallel_size=8,gpu_memory_utilization=0.5",
        "tasks": "bbh",
        "num_fewshot": "3",
        "batch_size": "auto",
        "n_gpus": "8",
    },
    "arc": {
        "model_type": "vllm",
        "model_args": "dtype=bfloat16,trust_remote_code=True,max_num_seqs=8,max_model_len=8192,tensor_parallel_size=8,gpu_memory_utilization=0.5",
        "tasks": "arc_easy,arc_challenge",
        "num_fewshot": "25",
        "batch_size": "auto",
        "n_gpus": "8",
    },
    "truthfulqa": {
        "model_type": "vllm",
        "model_args": "dtype=bfloat16,trust_remote_code=True,max_num_seqs=8,max_model_len=8192,tensor_parallel_size=8,gpu_memory_utilization=0.5",
        "tasks": "truthfulqa_mc1,truthfulqa_mc2,truthfulqa_gen",
        "num_fewshot": "0",
        "batch_size": "auto",
        "n_gpus": "8",
    },
    "drop": {
        "model_type": "vllm",
        "model_args": "dtype=bfloat16,trust_remote_code=True,max_num_seqs=8,max_model_len=8192,tensor_parallel_size=8,gpu_memory_utilization=0.5",
        "tasks": "drop",
        "num_fewshot": "3",
        "batch_size": "auto",
        "n_gpus": "8",
    },
    "race": {
        "model_type": "vllm",
        "model_args": "dtype=bfloat16,trust_remote_code=True,max_num_seqs=8,max_model_len=8192,tensor_parallel_size=8,gpu_memory_utilization=0.5",
        "tasks": "race",
        "num_fewshot": "5",
        "batch_size": "auto",
        "n_gpus": "8",
    },
    "triviaqa": {
        "model_type": "vllm",
        "model_args": "dtype=bfloat16,trust_remote_code=True,max_num_seqs=8,max_model_len=8192,tensor_parallel_size=8,gpu_memory_utilization=0.5",
        "tasks": "triviaqa",
        "num_fewshot": "5",
        "batch_size": "auto",
        "n_gpus": "8",
    },
    "agieval": {
        "model_type": "vllm",
        "model_args": "dtype=bfloat16,trust_remote_code=True,max_num_seqs=8,max_model_len=8192,tensor_parallel_size=8,gpu_memory_utilization=0.5",
        "tasks": "agieval",
        "num_fewshot": "0",
        "batch_size": "auto",
        "n_gpus": "8",
    },
    "commonsense_qa": {
        "model_type": "vllm",
        "model_args": "dtype=bfloat16,trust_remote_code=True,max_num_seqs=8,max_model_len=8192,tensor_parallel_size=8,gpu_memory_utilization=0.5",
        "tasks": "commonsense_qa",
        "num_fewshot": "5",
        "batch_size": "auto",
        "n_gpus": "8",
    },
    "lambada": {
        "model_type": "vllm",
        "model_args": "dtype=bfloat16,trust_remote_code=True,max_num_seqs=8,max_model_len=8192,tensor_parallel_size=8,gpu_memory_utilization=0.5",
        "tasks": "lambada",
        "num_fewshot": "5",
        "batch_size": "auto",
        "n_gpus": "8",
    },
    "openbookqa": {
        "model_type": "vllm",
        "model_args": "dtype=bfloat16,trust_remote_code=True,max_num_seqs=8,max_model_len=8192,tensor_parallel_size=8,gpu_memory_utilization=0.5",
        "tasks": "openbookqa",
        "num_fewshot": "5",
        "batch_size": "auto",
        "n_gpus": "8",
    },
    "piqa": {
        "model_type": "vllm",
        "model_args": "dtype=bfloat16,trust_remote_code=True,max_num_seqs=8,max_model_len=8192,tensor_parallel_size=8,gpu_memory_utilization=0.5",
        "tasks": "piqa",
        "num_fewshot": "0",
        "batch_size": "auto",
        "n_gpus": "8",
    },
    "gsm8k": {
        "model_type": "vllm",
        "model_args": "dtype=bfloat16,trust_remote_code=True,max_num_seqs=8,max_model_len=8192,tensor_parallel_size=8,gpu_memory_utilization=0.5",
        "tasks": "gsm8k",
        "num_fewshot": "4",
        "batch_size": "auto",
        "n_gpus": "8",
    },
    "mathqa": {
        "model_type": "vllm",
        "model_args": "dtype=bfloat16,trust_remote_code=True,max_num_seqs=8,max_model_len=8192,tensor_parallel_size=8,gpu_memory_utilization=0.5",
        "tasks": "mathqa",
        "num_fewshot": "5",
        "batch_size": "auto",
        "n_gpus": "8",
    },
    "mgsm": {
        "model_type": "vllm",
        "model_args": "dtype=bfloat16,trust_remote_code=True,max_num_seqs=8,max_model_len=8192,tensor_parallel_size=8,gpu_memory_utilization=0.5",
        "tasks": "mgsm_direct,mgsm_cot_native",
        "num_fewshot": "8",
        "batch_size": "auto",
        "n_gpus": "8",
    },
    "minerva_math": {
        "model_type": "vllm",
        "model_args": "dtype=bfloat16,trust_remote_code=True,max_num_seqs=8,max_model_len=8192,tensor_parallel_size=8,gpu_memory_utilization=0.5",
        "tasks": "minerva_math",
        "num_fewshot": "4",
        "batch_size": "auto",
        "n_gpus": "8",
    },
    "gpqa": {
        "model_type": "vllm",
        "model_args": "dtype=bfloat16,trust_remote_code=True,max_num_seqs=8,max_model_len=8192,tensor_parallel_size=8,gpu_memory_utilization=0.5",
        "tasks": "gpqa",
        "num_fewshot": "5",
        "batch_size": "auto",
        "n_gpus": "8",
    },
    "humaneval": {
        "model_type": "vllm",
        "model_args": "dtype=bfloat16,trust_remote_code=True,max_num_seqs=8,max_model_len=8192,tensor_parallel_size=8,gpu_memory_utilization=0.5",
        "tasks": "humaneval",
        "num_fewshot": "0",
        "batch_size": "auto",
        "n_gpus": "8",
    },
    "mbpp": {
        "model_type": "vllm",
        "model_args": "dtype=bfloat16,trust_remote_code=True,max_num_seqs=8,max_model_len=8192,tensor_parallel_size=8,gpu_memory_utilization=0.5",
        "tasks": "mbpp",
        "num_fewshot": "0",
        "batch_size": "auto",
        "n_gpus": "8",
    },
    "xcopa": {
        "model_type": "vllm",
        "model_args": "dtype=bfloat16,trust_remote_code=True,max_num_seqs=8,max_model_len=8192,tensor_parallel_size=8,gpu_memory_utilization=0.5",
        "tasks": "xcopa",
        "num_fewshot": "5",
        "batch_size": "auto",
        "n_gpus": "8",
    },
    "xwinograd": {
        "model_type": "vllm",
        "model_args": "dtype=bfloat16,trust_remote_code=True,max_num_seqs=8,max_model_len=8192,tensor_parallel_size=8,gpu_memory_utilization=0.5",
        "tasks": "xwinograd",
        "num_fewshot": "5",
        "batch_size": "auto",
        "n_gpus": "8",
    },
    "xstorycloze": {
        "model_type": "vllm",
        "model_args": "dtype=bfloat16,trust_remote_code=True,max_num_seqs=8,max_model_len=8192,tensor_parallel_size=8,gpu_memory_utilization=0.5",
        "tasks": "xstorycloze",
        "num_fewshot": "0",
        "batch_size": "auto",
        "n_gpus": "8",
    },
    "pawsx": {
        "model_type": "vllm",
        "model_args": "dtype=bfloat16,trust_remote_code=True,max_num_seqs=8,max_model_len=8192,tensor_parallel_size=8,gpu_memory_utilization=0.5",
        "tasks": "pawsx",
        "num_fewshot": "5",
        "batch_size": "auto",
        "n_gpus": "8",
    },
    "longbench": {
        "max_length": "8192",
        "model_dtype": "fp16",
        "n_gpus": "8",
        "model_name": "gigachat-gigar-vllm",
    },
    "passkey": {"model_dtype": "bf16", "n_gpus": "8"},
    "sam": {
        "tasks": "sam",
        "pass_obm": False,  # Пропустить замер метрик OBM
        "pass_obb": False,  # Пропустить этап замера обб
        "pass_judge": False,  # Пропустить сбор бакетов Judge-метрик
        "obb_datasets": [],  # Метрики обб, если не указывать это поле, то будут замеряться дефолтные
        "obm_datasets": [],  # Метрики обб, если не указывать это поле, то будут замеряться дефолтные
        "obm_groups": [],
    },
}


class JobCallbackDispatcher(Callback):
    def __init__(self, all_kwargs: DictConfig):
        self.jdc_config = all_kwargs
        self.export_path = self._resolve_export_path()
        self._validate_export_path()
        self.eval_interval = Time.from_input(
            self.jdc_config.pop("eval_interval"), TimeUnit.BATCH
        )
        self.last_checkpoint = None

    def _resolve_export_path(self) -> str:
        """Resolves the export path from config or environment variable."""
        if "signal_files_path" in self.jdc_config:
            export_path = self.jdc_config.pop("signal_files_path")
        elif "JOB_DISPATCHER_CALLBACK_SIGNAL_FILES" in os.environ:
            export_path = os.environ["JOB_DISPATCHER_CALLBACK_SIGNAL_FILES"]
        else:
            raise ValueError("Can't find JOB_DISPATCHER_CALLBACK path!")
        print("EXPORTPATH_HERE", export_path)
        return export_path

    def _validate_export_path(self):
        """Validates that the export path exists."""
        if not Path(self.export_path).exists():
            raise FileNotFoundError(f"{self.export_path} path does not exist!")

    def after_checkpoint_save(self, state: State, logger: Logger):
        self._handle_checkpoint_update(state)

    def _handle_checkpoint_update(self, state: State):
        """Handles checkpoint updates and triggers export if necessary."""
        checkpoint_save = get_current_checkpoint_save(state)
        if not checkpoint_save:
            return

        saved_checkpoint = checkpoint_save["saved_path"]
        saved_checkpoint_batch_value = checkpoint_save["saved_checkpoint_batch_value"]

        if (
            saved_checkpoint != self.last_checkpoint
            and saved_checkpoint_batch_value % self.eval_interval.value == 0
        ):
            self.last_checkpoint = saved_checkpoint
            self.jdc_config.ckpt_path = str(Path(self.last_checkpoint).parent)
            if dist.get_global_rank() == 0:
                self.export_json()

    def export_json(self):
        filename = tempfile.mktemp(suffix=".json", dir=self.export_path)
        resulting_job_dict = dict()
        jobs_dict_config = om.to_container(self.jdc_config, resolve=True)
        print(jobs_dict_config, file=sys.stderr)
        if jobs_dict_config.get("job_params_override", None) is not None:
            resulting_job_dict.update(jobs_dict_config["job_params_override"])

        resulting_job_dict.update(jobs_dict_config["job_params_override"])
        customized_benchmarcks = set()
        for key, val in resulting_job_dict.items():
            if key in ["longbench", "passkey"]:
                customized_benchmarcks.add(key)
            else:
                if "tasks" in val:
                    customized_benchmarcks.update(val["tasks"].split(","))
        if jobs_dict_config.get("job_list", None) is not None:
            for job_group in jobs_dict_config["job_list"].values():
                for job in job_group.split():
                    if job not in JOBS_DICT:
                        raise ValueError(
                            f"Metric '{job}' has no default configuration and should not be in 'job_list' argument. Please provide custom configuration for '{job}' in 'job_params_override'."
                        )
                    if job not in customized_benchmarcks:
                        if job in ["longbench", "passkey"]:
                            resulting_job_dict[job] = JOBS_DICT[job]
                        else:
                            default = deepcopy(JOBS_DICT[job])
                            tasks = ",".join(
                                [
                                    task
                                    for task in default["tasks"].split(",")
                                    if task not in customized_benchmarcks
                                ]
                            )
                            if tasks:
                                default["tasks"] = tasks
                                if job != "sam":
                                    resulting_job_dict[
                                        "lm_eval_" + job + "_default_config"
                                    ] = default
                                else:
                                    resulting_job_dict[job] = default

        jobs_dict_config["jobs"] = resulting_job_dict
        if "job_list" in jobs_dict_config:
            del jobs_dict_config["job_list"]
        if "job_params_override" in jobs_dict_config:
            del jobs_dict_config["job_params_override"]

        jobs_dict_config["save_hf_in_ckpt_dir"] = self.jdc_config.get(
            "save_hf_in_ckpt_dir", False
        )
        num_attention_heads = jobs_dict_config["num_attention_heads"]
        if "longbench" in jobs_dict_config["jobs"]:
            for n_gpus in range(
                min(8, int(jobs_dict_config["jobs"]["longbench"]["n_gpus"])), 0, -1
            ):
                if num_attention_heads % n_gpus == 0:
                    jobs_dict_config["jobs"]["longbench"]["n_gpus"] = str(n_gpus)
                    break
        if "passkey" in jobs_dict_config["jobs"]:
            for n_gpus in range(
                min(8, int(jobs_dict_config["jobs"]["passkey"]["n_gpus"])), 0, -1
            ):
                if num_attention_heads % n_gpus == 0:
                    jobs_dict_config["jobs"]["passkey"]["n_gpus"] = str(n_gpus)
                    break
        for key, val in jobs_dict_config["jobs"].items():
            if (
                key not in ["longbench", "passkey"]
                and "n_gpus" in val
                and "model_args" in val
            ):
                for n_gpus in range(min(8, int(val["n_gpus"])), 0, -1):
                    if num_attention_heads % n_gpus == 0:
                        params = val["model_args"].split(",")
                        for i, param in enumerate(params):
                            if param.startswith("tensor_parallel_size="):
                                params[i] = "tensor_parallel_size=" + str(n_gpus)
                                break
                        else:
                            params.append("tensor_parallel_size=" + str(n_gpus))
                        break
                val["model_args"] = ",".join(params)

        # hash_digest = hashlib.md5(json.dumps(jobs_dict_config, sort_keys=True).encode('utf-8')).hexdigest()
        # filename = Path(self.export_path) / f"{hash_digest}.json"

        logger.info(f"Start writing callback .json config in {filename}")
        with Path(filename).open("w", encoding="utf-8") as f:
            json.dump(jobs_dict_config, f, ensure_ascii=False, indent=4)


class JobCallbackDispatcherWEB(Callback):
    def __init__(self, all_kwargs: DictConfig):
        self.jdcw_config = all_kwargs
        self.eval_interval = Time.from_input(
            self.jdcw_config.pop("eval_interval"), TimeUnit.BATCH
        )
        self.last_checkpoint = None

        self.run_uuid = None

    @staticmethod
    def _is_internet_available(timeout: int = 3) -> bool:
        try:
            with socket.create_connection((INTERNET_CHECK_HOST, INTERNET_CHECK_PORT), timeout=timeout):
                return True
        except OSError:
            return False

    @classmethod
    def get_jd_base_url(cls) -> Optional[str]:
        if cls._is_internet_available():
            return os.environ.get("JOB_DISPATCHER_BASE_URL")
        return os.environ.get("JOB_DISPATCHER_SLURM_BASE_URL")

    def fit_start(self, state: State, log: Logger):
        if dist.get_global_rank() != 0:
            return
        if os.environ.get("PIPELINE_ID"):
            return
        try:
            pipeline_id = self._notify_start()
            if pipeline_id is not None:
                os.environ["PIPELINE_ID"] = str(pipeline_id)
                os.environ["PIPELINE_STAGE_IDX"] = "0"
        except Exception:
            logger.error("Can't send pipeline create request to job dispatcher")

    def _notify_start(self):
        jd_base_url = self.get_jd_base_url()
        if not jd_base_url:
            logger.warning(
                "Skipping create pipeline: job-dispatcher is not available in this environment"
            )
            return

        pipeline_dict = {
            "ml_space_job": {
                "job_name": os.environ.get("MLSPACE_JOB_NAME"),
                "workspace": os.environ.get("MLS_JOB_WORKSPACE_NAME"),
            }
        }

        endpoint_path = "/pipeline/start_learning_pipeline"
        response = requests.post(
            url=f"{jd_base_url}{endpoint_path}",
            json=pipeline_dict,
            headers=HEADERS,
            verify=False,
            timeout=DEFAULT_TIMEOUT,
        )
        if response.status_code != 200:
            raise ValueError(
                f"job_dispatcher_callback_error {response.status_code}, {response.text}"
            )

        pipeline_id = response.json().get("pipeline_id")
        logger.info(f"Created learning pipeline: pipeline_id={pipeline_id}")
        return pipeline_id

    def after_checkpoint_save(self, state: State, logger: Logger):
        if dist.get_global_rank() != 0:
            return
        self._handle_checkpoint_update(state)

    def _handle_checkpoint_update(self, state: State):
        """Handles checkpoint updates and triggers export if necessary."""
        checkpoint_save = get_current_checkpoint_save(state)
        if not checkpoint_save:
            return

        saved_checkpoint = checkpoint_save["saved_path"]
        saved_checkpoint_batch_value = checkpoint_save["saved_checkpoint_batch_value"]

        if (
            saved_checkpoint != self.last_checkpoint
            and saved_checkpoint_batch_value % self.eval_interval.value == 0
        ):
            self.last_checkpoint = saved_checkpoint
            checkpoint_dir = str(Path(self.last_checkpoint).parent)
            try:
                self.export_json(checkpoint_dir)
            except Exception as e:
                logger.error(
                    f"Failed to export json in job dispatcher callback for path {checkpoint_dir}: {e}"
                )

    def _replace_checkpoint_placeholders(
        self, config_dict: dict, checkpoint_path: str, wandb_run=None
    ) -> None:
        checkpoint_dir = str(Path(checkpoint_path).parent)

        placeholders = {
            "{CHECKPOINT_PATH}": checkpoint_path,
            "{CHECKPOINT_DIR}": checkpoint_dir,
        }

        if wandb_run is not None:
            placeholders["{WANDB_ENTITY}"] = wandb_run.entity
            placeholders["{WANDB_PROJECT}"] = wandb_run.project
            placeholders["{WANDB_RUN_ID}"] = wandb_run.id

        def replace(obj):
            if isinstance(obj, str):
                for placeholder, replacement in placeholders.items():
                    obj = obj.replace(placeholder, replacement)
                return obj
            elif isinstance(obj, dict):
                return {k: replace(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace(v) for v in obj]
            return obj

        for key, value in config_dict.items():
            config_dict[key] = replace(value)

    def export_json(self, checkpoint_path: str):
        jobs_config = om.to_container(self.jdcw_config, resolve=True)
        jobs_config["checkpoint_path"] = checkpoint_path

        run = wandb.run  # текущий run в wandb

        self._replace_checkpoint_placeholders(
            jobs_config, checkpoint_path, wandb_run=run
        )

        if run is not None:
            jobs_config["wandb_entity"] = run.entity
            jobs_config["wandb_project"] = run.project
            jobs_config["wandb_run_id"] = run.id
        else:
            logger.info(
                "WandB is not initialized, skipping wandb-related configuration"
            )
        jobs_config["llmf_path"] = str(Path(__file__).parent.parent.parent)
        path = Path(checkpoint_path).parent / "artifacts/job-dispatcher"
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"CREATED PATH FOR ARTIFACTS {path}")

        jd_base_url = self.get_jd_base_url()
        if not jd_base_url:
            logger.warning(
                "Skipping run jobs: job-dispatcher is not available in this environment"
            )
            return

        endpoint_path = (
            "/jobs/run-custom-jobs"
            if "custom_pipeline" in jobs_config
            else "/jobs/run-jobs"
        )
        response = requests.post(
            url=f"{jd_base_url}{endpoint_path}",
            json=jobs_config,
            headers=HEADERS,
            verify=False,
            timeout=DEFAULT_TIMEOUT,
        )
        if response.status_code != 200:
            raise ValueError(
                f"job_dispatcher_callback_error {response.status_code}, {response.text}"
            )
        else:
            logger.info("Request sent to eval metrics via job_dispatcher")

    @classmethod
    def send_pam_job_name_if_pipeline_id(cls, job_name: str) -> None:
        pipeline_id = os.environ.get("PIPELINE_ID")
        if pipeline_id:
            try:
                cls._send_pam_job_name(int(pipeline_id), job_name)
            except Exception as e:
                logger.error(f"Can't send job name to jd: {e}")

    @classmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=15),
    )
    def _send_pam_job_name(
        cls,
        pipeline_id: int,
        job_name: str,
    ) -> None:
        body = {
            "name": job_name,
            "workspace": os.environ.get("MLS_JOB_WORKSPACE_NAME"),
            "stage_index": os.environ.get("PIPELINE_STAGE_IDX"),
        }

        jd_base_url = cls.get_jd_base_url()
        if not jd_base_url:
            logger.warning(
                "Skipping send pam job name: job-dispatcher is not available in this environment"
            )
            return

        res = requests.post(
            f"{jd_base_url}/ml_models/send_job_name/{pipeline_id}",
            json=body,
            headers=HEADERS,
            verify=False,
            timeout=DEFAULT_TIMEOUT,
        )
        res.raise_for_status()
