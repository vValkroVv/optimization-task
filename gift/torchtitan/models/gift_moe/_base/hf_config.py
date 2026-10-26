"""Runtime entry points for the pure GIFT HuggingFace configuration contract."""

import os

from torchtitan.config.schema.models.gift_moe._base.hf_config import (
    build_hf_config,
    GIGACHAT_MODEL_TYPE,
)

__all__ = ["build_hf_config", "default_gigachat_remote_code_dir", "GIGACHAT_MODEL_TYPE"]

# PAM's checkout of the remote-code files, relative to the repository root.
_PAM_GIGACHAT_DIR = os.path.join(
    "contrib",
    "pretrain_auto_metrics",
    "contrib",
    "gigachat_utils",
    "convertation",
    "models",
    "gigachat3_5",
)


def default_gigachat_remote_code_dir() -> str:
    """``contrib/pretrain_auto_metrics/.../models/gigachat3_5`` under the repo root."""
    here = os.path.abspath(__file__)
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(here))))
    )
    return os.path.join(repo_root, _PAM_GIGACHAT_DIR)
