import logging
import os
import re
import subprocess as sp
import threading
import requests
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from composer.utils.job_dispatcher.base import JDBase

log = logging.getLogger(__name__)


class JDSendModel(JDBase):

    PIPE_TIMEOUT = 60 * 15

    ADD_MODEL_ENDPOINT = "/ml_models/add_model_with_step"

    CODE_ARCHIVE_NAME = "llm_foundry.tar"
    CONFIG_NAME = "llm_foundry_config.yaml"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._executor: Optional[ThreadPoolExecutor] = None
        self._send_model_future: Optional[Future] = None
        self._s3_run_index: Optional[int] = None

    def _is_s3_available(
        self,
        s3_dest: str,
        rclone_path: str,
        rclone_config: str,
    ) -> bool:
        if not s3_dest or not rclone_path or not rclone_config:
            log.warning("s3 destination/rclone settings are not configured; skipping s3 access")
            return False
        try:
            sp.run(
                [rclone_path, "--config", rclone_config, "lsf", "--max-depth", "1", s3_dest],
                capture_output=True,
                text=True,
                check=True,
                timeout=self.DEFAULT_TIMEOUT,
            )
            return True
        except Exception as e:
            log.warning(f"s3 is not reachable at {s3_dest}: {e}; skipping s3 access")
            return False

    def _delete_s3_file(
        self,
        s3_file_path: str,
        rclone_path: str,
        rclone_config: str,
    ) -> None:
        cleanup_cmd = [
            rclone_path,
            "--config", rclone_config,
            "deletefile",
            "--s3-no-check-bucket",
            s3_file_path,
        ]
        sp.run(
            cleanup_cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=self.DEFAULT_TIMEOUT,
        )

    def _upload_file_to_s3(
        self,
        filename: str,
        path: str,
        s3_dest: str,
        rclone_path: str,
        rclone_config: str,
    ):
        s3_file_path = f"{s3_dest}/{filename}"
        log.info(f"Uploading {path} to {s3_file_path}")
        script = [
            rclone_path,
            "--config", rclone_config,
            "-P", "copyto",
            "--s3-no-check-bucket",
            "--transfers", "16",
            "--buffer-size", "256Mi",
            "--multi-thread-streams", "16",
            "--s3-disable-checksum",
            "--ignore-checksum",
            "--fast-list",
            "--no-update-modtime",
            str(path),
            s3_file_path,
        ]
        try:
            sp.run(
                script,
                capture_output=True,
                text=True,
                check=True,
                timeout=self.PIPE_TIMEOUT,
            )
        except Exception:
            self._delete_s3_file(s3_file_path, rclone_path, rclone_config)
            raise

        log.info(f"Successfully uploaded {path} to {s3_file_path}")
        return s3_file_path

    def _tar_and_upload_llm_foundry_code(
        self,
        s3_dir: str,
        rclone_path: str,
        rclone_config: str,
    ) -> Optional[str]:
        s3_file_path = f"{s3_dir}/{self.CODE_ARCHIVE_NAME}"

        llm_foundry_path = Path(__file__).resolve().parents[5]
        if not llm_foundry_path.exists():
            log.warning(f"Could not infer llm-foundry path (tried: {llm_foundry_path})")
            return None

        log.info(f"Streaming tar of {llm_foundry_path} -> {s3_file_path}")

        tar_cmd = [
            "tar",
            "-cf", "-",
            "-C", str(llm_foundry_path.parent),
            llm_foundry_path.name,
        ]
        rclone_cmd = [
            rclone_path,
            "--config", rclone_config,
            "rcat",
            "--s3-no-check-bucket",
            "--s3-disable-checksum",
            "--ignore-checksum",
            s3_file_path,
        ]

        tar_proc = sp.Popen(tar_cmd, stdout=sp.PIPE, stderr=sp.PIPE)
        try:
            rclone_proc = sp.Popen(
                rclone_cmd,
                stdin=tar_proc.stdout,
                stdout=sp.PIPE,
                stderr=sp.PIPE,
            )
        except Exception:
            tar_proc.kill()
            tar_proc.wait()
            raise

        tar_proc.stdout.close()

        tar_err_chunks: list[bytes] = []
        def _drain_tar_stderr():
            try:
                assert tar_proc.stderr is not None
                for chunk in iter(lambda: tar_proc.stderr.read(4096), b""):
                    tar_err_chunks.append(chunk)
            except Exception:
                pass
        tar_stderr_drain = threading.Thread(
            target=_drain_tar_stderr,
            name="tar-stderr-drain",
            daemon=True,
        )
        tar_stderr_drain.start()

        try:
            _, rclone_err = rclone_proc.communicate(timeout=self.PIPE_TIMEOUT)
            tar_proc.wait(timeout=self.DEFAULT_TIMEOUT)
        except sp.TimeoutExpired:
            rclone_proc.kill()
            tar_proc.kill()
            rclone_proc.communicate()
            tar_proc.wait()
            tar_stderr_drain.join(timeout=5)
            self._delete_s3_file(s3_file_path, rclone_path, rclone_config)
            raise

        tar_stderr_drain.join(timeout=5)
        tar_err = b"".join(tar_err_chunks)

        if tar_proc.returncode != 0 or rclone_proc.returncode != 0:
            self._delete_s3_file(s3_file_path, rclone_path, rclone_config)
            raise sp.CalledProcessError(
                rclone_proc.returncode if rclone_proc.returncode != 0 else tar_proc.returncode,
                rclone_cmd if rclone_proc.returncode != 0 else tar_cmd,
                stderr=tar_err.decode(errors='replace'),
            )

        log.info(f"Successfully uploaded tar to {s3_file_path}")
        return s3_file_path

    def _upload_llm_foundry_config(
        self,
        artifact_folder: Path,
        s3_dir: str,
        rclone_path: str,
        rclone_config: str,
    ) -> Optional[str]:
        config_nfs_path = str(artifact_folder / self.CONFIG_NAME)

        return self._upload_file_to_s3(self.CONFIG_NAME, config_nfs_path, s3_dir, rclone_path, rclone_config)

    def _to_s3_relative(self, nfs_path: str, marker: Optional[str]) -> str:
        idx = nfs_path.find(marker) if marker else -1
        relative_path = nfs_path[idx + len(marker):] if idx >= 0 else nfs_path
        return relative_path.strip("/")

    def _get_next_s3_index(self, s3_model_dir: str, rclone_path: str, rclone_config: str) -> int:
        s3_parent_dir, _, model_name = s3_model_dir.rpartition("/")

        proc = sp.run(
            [rclone_path, "--config", rclone_config, "lsf", "--dirs-only", s3_parent_dir],
            capture_output=True,
            text=True,
            check=True,
            timeout=self.DEFAULT_TIMEOUT,
        )

        pattern = re.compile(rf"^{re.escape(model_name)}_(\d{{4}})/?$")

        max_index = 0
        for line in proc.stdout.splitlines():
            match = pattern.match(line.strip())
            if match:
                max_index = max(max_index, int(match.group(1)))
        return max_index + 1

    def _get_s3_dir(
        self,
        s3_dest: str,
        s3_path_marker: Optional[str],
        model_dir: str,
        checkpoint_path: Path,
        rclone_path: str,
        rclone_config: str,
    ) -> str:
        s3_model_dir = f"{s3_dest}/{self._to_s3_relative(model_dir, s3_path_marker)}"

        if self._s3_run_index is None:
            self._s3_run_index = self._get_next_s3_index(s3_model_dir, rclone_path, rclone_config)
            log.info(f"Using s3 run index {self._s3_run_index:04d} under {s3_model_dir}")

        s3_run_dir = f"{s3_model_dir}_{self._s3_run_index:04d}"

        return f"{s3_run_dir}/{checkpoint_path.name}"

    def _get_wandb_run(self):
        try:
            import wandb
        except ImportError:
            return None
        return wandb.run

    def send_model(self, artifact_folder) -> None:
        artifact_folder = Path(artifact_folder)

        config = self._resolve_model_config(artifact_folder)

        s3_dest = config.get("s3_dest")
        rclone_path = config.get("rclone_path")
        rclone_config = config.get("rclone_config")
        s3_path_marker = config.get("s3_path_marker")

        checkpoint_path = artifact_folder.parent
        model_dir, model_name, _epoch, step = self._parse_checkpoint_path(checkpoint_path)

        if self._is_s3_available(s3_dest, rclone_path, rclone_config):
            s3_dir = self._get_s3_dir(
                s3_dest, s3_path_marker, model_dir, checkpoint_path, rclone_path, rclone_config,
            )
            llmf_code_s3_path = self._tar_and_upload_llm_foundry_code(
                s3_dir, rclone_path, rclone_config,
            )
            llmf_config_s3_path = self._upload_llm_foundry_config(
                artifact_folder, s3_dir, rclone_path, rclone_config,
            )
        else:
            log.warning("Skipping s3 uploads of llm-foundry code/config: s3 is not available in this environment")
            llmf_code_s3_path = None
            llmf_config_s3_path = None

        model_dict = {
            "nfs_path": model_dir,
            "name": model_name,
            "size": config.get("size") or self.DEFAULT_SIZE,
            "model_class": config.get("model_class"),
            "username": config.get("username"),
            "region": os.environ.get("MLS_JOB_REGION_NAME"),
            "step": step,
            "llmf_code_s3_path": llmf_code_s3_path,
            "llmf_config_s3_path": llmf_config_s3_path,
            "ml_space_job": {
                "job_name": os.environ.get("MLSPACE_JOB_NAME"),
                "workspace": os.environ.get("MLS_JOB_WORKSPACE_NAME"),
            }
        }

        run = self._get_wandb_run()
        if run is not None:
            model_dict["wandb_entity"] = run.entity
            model_dict["wandb_project"] = run.project
            model_dict["wandb_run_id"] = run.id
        else:
            log.info("WandB is not initialized, skipping wandb-related configuration")

        jd_base_url = self.get_jd_base_url()
        if not jd_base_url:
            log.warning("Skipping add_model_with_step: job-dispatcher is not available in this environment")
            return

        response = requests.post(
            url=f"{jd_base_url}{self.ADD_MODEL_ENDPOINT}",
            json=model_dict,
            headers=self._jd_headers(),
            verify=False,
            timeout=self.JD_TIMEOUT,
        )
        if response.status_code != 200:
            log.error(f"job_dispatcher_send_model_error {response.status_code}, {response.text}")
            return

        log.info("Request sent to add model via job_dispatcher")

    def submit_send_model(self, artifact_folder) -> None:
        if self._send_model_future is not None and not self._send_model_future.done():
            log.info("Previous send_model is still running; skipping new submission")
            return

        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="job_dispatcher_send_model")

        future = self._executor.submit(self.send_model, artifact_folder)

        def _log_exc(f):
            exc = f.exception()
            if exc is not None:
                log.error(f"send_model failed: {exc}", exc_info=exc)

        future.add_done_callback(_log_exc)
        self._send_model_future = future
