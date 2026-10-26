"""Tests for job_dispatcher checkpoint path parsing."""

from pathlib import Path

import pytest
from node_runner.src.job_dispatcher import JD

pytestmark = pytest.mark.unit

CHECKPOINT_PATH_EPOCH_BATCH = (
    "/home/jovyan/models/papayaR-10b-262k-sft-chatmlv4-template-512gbs-bf16-tp1-sp2-sft-v31_0-b2c-lr5e7/"
    "ep0_ba2576/artifacts/hf_checkpoint"
)
CHECKPOINT_PATH_GLOBAL_STEP = (
    "/mnt/nfs/example/example/rl/test/checkpoints/ckpts/"
    "examples/test_vllm_120b_16_nodes_prod_setup_test_from_hamster_user/"
    "global_step_1/actor/huggingface"
)
CHECKPOINT_PATH_SIMPLE = "/home/jovyan/models/deepseek-v3"

EXPECTED_MODEL_DIR_EPOCH = "/home/jovyan/models/papayaR-10b-262k-sft-chatmlv4-template-512gbs-bf16-tp1-sp2-sft-v31_0-b2c-lr5e7"
EXPECTED_MODEL_NAME_EPOCH = (
    "papayaR-10b-262k-sft-chatmlv4-template-512gbs-bf16-tp1-sp2-sft-v31_0-b2c-lr5e7"
)
EXPECTED_MODEL_NAME_GLOBAL_STEP = (
    "test_vllm_120b_16_nodes_prod_setup_test_from_hamster_user"
)
EXPECTED_MODEL_NAME_SIMPLE = "deepseek-v3"

TRANSFER_PATH_GLOBAL_STEP = (
    "/mnt/nfs/example/example/example/rl/test/checkpoints/ckpts/"
    "test_vllm_120b_16_nodes_prod_setup/global_step_2/actor/huggingface"
)
TRANSFER_PATH_EPOCH = (
    "/mnt/nfs/example/example/example/rl/test/checkpoints/ckpts/"
    "test_vllm_120b_16_nodes_prod_setup/ep0_ba2500/artifacts/hf_checkpoint"
)
TRANSFER_PATH_SIMPLE = (
    "/mnt/nfs/example/example/example/rl/test/checkpoints/ckpts/DeepSeek-V3-Base"
)

TRANSFER_PATH_GLOBAL_STEP_FP8 = (
    f"{TRANSFER_PATH_GLOBAL_STEP}/artifacts/hf_checkpoint_fp8"
)
TRANSFER_PATH_EPOCH_FP8 = TRANSFER_PATH_EPOCH.replace(
    "hf_checkpoint", "hf_checkpoint_fp8"
)
TRANSFER_PATH_SIMPLE_FP8 = f"{TRANSFER_PATH_SIMPLE}/artifacts/hf_checkpoint_fp8"

EXPECTED_TRANSFER_GLOBAL_STEP = "test_vllm_120b_16_nodes_prod_setup/global_step_2"
EXPECTED_TRANSFER_EPOCH = "test_vllm_120b_16_nodes_prod_setup/ep0_ba2500"
EXPECTED_TRANSFER_SIMPLE = "DeepSeek-V3-Base"


@pytest.mark.parametrize(
    (
        "hf_checkpoint_path",
        "expected_model_dir",
        "expected_model_name",
        "expected_epoch",
        "expected_step",
    ),
    [
        (
            CHECKPOINT_PATH_EPOCH_BATCH,
            EXPECTED_MODEL_DIR_EPOCH,
            EXPECTED_MODEL_NAME_EPOCH,
            0,
            2576,
        ),
        (
            CHECKPOINT_PATH_GLOBAL_STEP,
            None,
            EXPECTED_MODEL_NAME_GLOBAL_STEP,
            0,
            1,
        ),
        (
            CHECKPOINT_PATH_SIMPLE,
            CHECKPOINT_PATH_SIMPLE,
            EXPECTED_MODEL_NAME_SIMPLE,
            0,
            0,
        ),
        (
            f"{CHECKPOINT_PATH_SIMPLE}/artifacts/hf_checkpoint_fp8",
            CHECKPOINT_PATH_SIMPLE,
            EXPECTED_MODEL_NAME_SIMPLE,
            0,
            0,
        ),
    ],
)
def test_parse_checkpoint_path(
    hf_checkpoint_path: str,
    expected_model_dir: str | None,
    expected_model_name: str,
    expected_epoch: int,
    expected_step: int,
):
    model_dir, model_name, epoch, step = JD.parse_checkpoint_path(hf_checkpoint_path)

    if expected_model_dir is not None:
        assert model_dir == expected_model_dir
    assert model_name == expected_model_name
    assert epoch == expected_epoch
    assert step == expected_step


@pytest.mark.parametrize(
    ("checkpoint_path", "expected"),
    [
        (TRANSFER_PATH_GLOBAL_STEP, EXPECTED_TRANSFER_GLOBAL_STEP),
        (TRANSFER_PATH_EPOCH, EXPECTED_TRANSFER_EPOCH),
        (TRANSFER_PATH_SIMPLE, EXPECTED_TRANSFER_SIMPLE),
        (TRANSFER_PATH_GLOBAL_STEP_FP8, EXPECTED_TRANSFER_GLOBAL_STEP),
        (TRANSFER_PATH_EPOCH_FP8, EXPECTED_TRANSFER_EPOCH),
        (TRANSFER_PATH_SIMPLE_FP8, EXPECTED_TRANSFER_SIMPLE),
    ],
)
def test_parse_transfer_directory(checkpoint_path: str, expected: str) -> None:
    assert JD.parse_transfer_directory(Path(checkpoint_path)) == expected
