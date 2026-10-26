import json
import logging
import os
import uuid
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from composer.utils.job_dispatcher.base import JDBase

log = logging.getLogger(__name__)


class JDData(JDBase):

    CHECKPOINT_HISTORY_ENDPOINT = "/checkpoint_history/create"

    def _resolve_parent_id(self, load_path) -> Optional[str]:
        if not load_path:
            return None

        load_path = Path(load_path)
        jd_data_path = load_path.parent / "artifacts" / "jd_data.json"
        if jd_data_path.is_file():
            try:
                with open(jd_data_path, encoding="utf-8") as f:
                    parent = json.load(f)
                return parent.get("checkpoint_info", {}).get("id")
            except Exception as e:
                log.error(f"Error occured while reading parent jd_data.json at {jd_data_path}: {e}")
                return None

        return None

    def _send_checkpoint_history(self, payload: dict) -> None:
        jd_base_url = self.get_jd_base_url()
        if not jd_base_url:
            log.warning("Skipping checkpoint_history: job-dispatcher is not available in this environment")
            return

        res = requests.post(
            f"{jd_base_url}{self.CHECKPOINT_HISTORY_ENDPOINT}",
            json=[payload],
            headers=self._jd_headers(),
            verify=False,
            timeout=self.JD_TIMEOUT,
        )
        res.raise_for_status()
        checkpoint_id = payload.get("checkpoint_info", {}).get("id")
        log.info(f"Sent checkpoint history to job dispatcher: {checkpoint_id}")

    def write_jd_data(self, artifact_folder, checkpoint_type="fsdp"):
        """Writes ``jd_data.json`` into ``artifact_folder`` and sends it to the job dispatcher."""
        from omegaconf import ListConfig

        checkpoint_path = Path(artifact_folder).parent
        _model_dir, model_name, _epoch, step = self._parse_checkpoint_path(checkpoint_path)
        workspace = os.environ.get("MLS_JOB_WORKSPACE_NAME") or os.environ.get("MLS_NTB_WORKSPACE_NAME")

        config = self._resolve_model_config(artifact_folder)

        parent_id = None
        if config.get("creation_type") != "pretrain":
            load_path = config.get("load_path")
            if isinstance(load_path, (list, tuple, ListConfig)):
                load_path = load_path[0] if load_path else None
            parent_id = self._resolve_parent_id(load_path)

        jd_data = {
            "checkpoint_info": {
                "id": str(uuid.uuid4()),
                "nfs_path": str(checkpoint_path),
                "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "step": step,
                "checkpoint_type": checkpoint_type,
                "checkpoint_creation_type": config.get("creation_type"),
                "workspace": workspace,
                "parent_id": parent_id,
            },
            "model_info": {
                "nfs_path": str(checkpoint_path.parent),
                "name": model_name,
                "size": config.get("size"),
                "model_class": config.get("model_class"),
                "username": config.get("username"),
                "region": os.environ.get("MLS_JOB_REGION_NAME"),
            },
            "ml_space_job": {
                "job_name": os.environ.get("MLSPACE_JOB_NAME"),
                "workspace": workspace,
            },
            "history": [],
        }

        jd_data_path = os.path.join(artifact_folder, "jd_data.json")
        serialized = json.dumps(jd_data, ensure_ascii=False, indent=4)
        with open(jd_data_path, "w", encoding="utf-8") as f:
            f.write(serialized)

        try:
            self._send_checkpoint_history(jd_data)
        except Exception as e:
            log.error(f"Error occured while sending checkpoint history to job dispatcher: {e}")
