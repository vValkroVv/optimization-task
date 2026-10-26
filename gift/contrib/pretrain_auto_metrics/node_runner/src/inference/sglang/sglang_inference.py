import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from ...config_model import SglangInferenceConfig
from ...logger import Logger
from ...subprocess_manager import SubprocessManager

SGLANG_DEEPEP_DEFAULT_DISPATCH_TOKENS_PER_RANK = 128


class InferenceStartError(Exception):
    pass


class SglangInference:
    def __init__(
        self,
        hf_checkpoint_path: Path,
        inference_config: SglangInferenceConfig,
        log_profile: str = "evaluated",
        *,
        port: int,
        url: str,
        nnodes: int,
        node_rank: int,
        dist_init_address: str,
    ):
        self.hf_checkpoint_path = hf_checkpoint_path
        self.inference_config = inference_config
        self.log_profile = log_profile
        self.url = url
        self.subprocess_manager: Optional[SubprocessManager] = None
        self.inference_envs: dict[str, str] = {}
        Logger.info("Sglang inference initialized successfully")
        self._enable_auto_mtp()
        multinode_args: list[str] = []
        dist_init_args: list[str] = []
        if nnodes != 1:
            multinode_args = ["--nnodes", str(nnodes), "--node-rank", str(node_rank)]
            dist_init_args = ["--dist-init-addr", dist_init_address]
        self.bash_cmd_list = [
            "/home/user/sglang/bin/python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            str(hf_checkpoint_path),
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
            *multinode_args,
            *self.inference_config.to_bash_list(),
            *dist_init_args,
        ]
        self._setup_deepgemm_cache_dir(
            self._get_dg_cache_dir(), self._get_dg_cache_dir_local()
        )

    @retry(
        stop=stop_after_attempt(450),
        wait=wait_fixed(10),
        reraise=True,
        retry=retry_if_exception_type(InferenceStartError),
    )
    def wait_for_inference_to_initialize(self):
        if not self.subprocess_manager.is_alive():
            self.subprocess_manager.raise_for_exit_code()
        if self._check_sglang_is_ready(self.url):
            Logger.info("Inference started successfully")
            return

        raise InferenceStartError("Inference did not start in 75 minutes")

    def copy_dg_cache_to_nfs(self):
        self._copy_dg_cache_to_nfs_after_successfull_inference(
            self._get_dg_cache_dir(), self._get_dg_cache_dir_local()
        )

    def _enable_auto_mtp(self) -> None:
        if not self._enables_mtp(self.hf_checkpoint_path, self.inference_config):
            return
        Logger.info(f"MTP found in {self.hf_checkpoint_path}, enabling EAGLE")
        self.inference_config = self.inference_config.model_copy(
            update={
                "speculative_algorithm": "EAGLE",
                "speculative_num_steps": 2,
                "speculative_eagle_topk": 1,
                "speculative_num_draft_tokens": 3,
            }
        )

    @staticmethod
    def _enables_mtp(
        hf_checkpoint_path: Path, inference_config: SglangInferenceConfig
    ) -> bool:
        if (
            not inference_config.auto_mtp
            or inference_config.speculative_algorithm
            or inference_config.pp > 1
        ):
            return False
        index_path = hf_checkpoint_path / "model.safetensors.index.json"
        if not index_path.exists():
            return False
        weight_map = json.loads(index_path.read_text())["weight_map"]
        return any(
            ".eh_proj." in name or name.startswith("mtp.") for name in weight_map
        )

    @staticmethod
    def _check_sglang_is_ready(group_sglang_inference_address: str):
        url = f"{group_sglang_inference_address}/health_generate"
        try:
            response = requests.get(url, timeout=60)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def _get_dg_cache_dir(self) -> Path:
        return Path(os.getenv("PAM_DG_CACHE_DIR", "/tmp/dg_cache"))

    def _get_dg_cache_dir_local(self) -> Path:
        return Path("/home/user/dg_cache")

    def _setup_deepgemm_cache_dir(
        self, dg_cache_dir_nfs: Path, dg_cache_dir_local: Path
    ):
        dg_cache_dir_local.mkdir(parents=True, exist_ok=True)
        dg_cache_dir_nfs.mkdir(parents=True, exist_ok=True)

        dg_cache_dir_nfs_cache = dg_cache_dir_nfs / "cache"
        dg_cache_dir_local_cache = dg_cache_dir_local / "cache"
        if dg_cache_dir_local_cache.exists():
            shutil.rmtree(dg_cache_dir_local_cache)
        if dg_cache_dir_nfs_cache.exists():
            shutil.copytree(dg_cache_dir_nfs_cache, dg_cache_dir_local_cache)
        else:
            dg_cache_dir_local_cache.mkdir(parents=True, exist_ok=True)
        Logger.info(f"DeepGemm cache dir setup complete: {dg_cache_dir_local_cache}")
        self.inference_envs["SGLANG_DG_CACHE_DIR"] = (
            dg_cache_dir_local.absolute().as_posix()
        )

    @staticmethod
    def _atomic_copy_no_overwrite(src: Path, dst: Path) -> bool:
        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists():
            return False
        tmp = dst.with_name(f".{dst.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")

        try:
            shutil.copy2(src, tmp)
            os.link(tmp, dst)
            tmp.unlink(missing_ok=True)
            return True
        except FileExistsError:
            tmp.unlink(missing_ok=True)
            return False
        except Exception:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            raise

    def _copy_dg_cache_to_nfs_after_successfull_inference(
        self, dg_cache_dir_nfs: Path, dg_cache_dir_local: Path
    ):
        dg_cache_dir_local_cache = dg_cache_dir_local / "cache"
        dg_cache_dir_nfs_cache = dg_cache_dir_nfs / "cache"

        if not dg_cache_dir_local_cache.exists():
            Logger.info(
                f"DeepGemm cache not found locally, nothing to copy: {dg_cache_dir_local_cache}"
            )
            return

        dg_cache_dir_nfs_cache.mkdir(parents=True, exist_ok=True)

        copied_any = False
        for root, _, files in os.walk(dg_cache_dir_local_cache):
            src_root = Path(root)
            rel_root = src_root.relative_to(dg_cache_dir_local_cache)
            dst_root = dg_cache_dir_nfs_cache / rel_root
            dst_root.mkdir(parents=True, exist_ok=True)

            for file_name in files:
                src_file = src_root / file_name
                dst_file = dst_root / file_name

                if self._atomic_copy_no_overwrite(src_file, dst_file):
                    copied_any = True

        if copied_any:
            Logger.info(f"DeepGemm cache copied to NFS: {dg_cache_dir_nfs_cache}")
        else:
            Logger.info(
                f"DeepGemm cache already up to date on NFS: {dg_cache_dir_nfs_cache}"
            )

    @property
    def inference_log_file(self) -> Path:
        return Logger.inference_log_file_for(self.log_profile)

    def start_inference(self) -> None:
        Logger.info(f"Starting inference with command: {self.bash_cmd_list}")
        env = {**os.environ, **self.inference_envs}
        env.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
        env.setdefault("SGLANG_FUSED_MOE_CHUNK_SIZE", "16384")
        env.setdefault("SGLANG_TIMEOUT_KEEP_ALIVE", "75")
        draft_tokens = self.inference_config.speculative_num_draft_tokens
        if self.inference_config.moe_a2a_backend == "deepep" and draft_tokens:
            env.setdefault(
                "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK",
                str(SGLANG_DEEPEP_DEFAULT_DISPATCH_TOKENS_PER_RANK * draft_tokens),
            )
        self.subprocess_manager = SubprocessManager(
            self.bash_cmd_list,
            self.inference_log_file,
            None,
            "sglang-inference",
            env=env,
        )
        self.subprocess_manager.start()

    def stop(self):
        if self.subprocess_manager:
            self.subprocess_manager.graceful_stop()
