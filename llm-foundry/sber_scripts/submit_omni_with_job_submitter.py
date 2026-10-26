"""
Universal script to submit GigaVision/GigaOmni jobs to ML Space SR004 Cluster (a.k.a. wmtrain).

Includes additional comments and pre-submit validations.
"""

import sys

from pathlib import Path
from typing import Literal, Optional

from omegaconf import OmegaConf

ROOT_DIR = Path(__file__).parents[1]
sys.path.append(str(ROOT_DIR / "contrib" / "job_submitter"))

from job_submitter.jobs.llmf.train import TrainJobLLMF  # noqa: E402
from job_submitter.jobs.infinite import InfiniteJob  # noqa: E402
from job_submitter.jobs.gigachat_utils.test_convertation import (  # noqa: E402
    TestConvertationJobGigachatUtils,
)


# ---
# EDITABLE PARAMETERS START
# ---
# Only these parameters should be edited. All other parameters are automatically set by the script.
# ---
N_WORKERS: int = 0  # Number of nodes. Should be > 0. Used only for train job.
USER: str = "..."  # Your Mattermost login.
EXP_NAME: str = "..."  # Experiment name. Should be unique and not empty.
JOB_TYPE: Literal["train", "inf", "convert"] = (
    "train"  # "train" | "inf" | "convert". Train job is default.
)
CONFIG: str = ""  # Path to train or convertation config. Can be left empty only for infinity jobs.
QUEUE_NAME: Literal["default"] = (
    "default"  # Queue name. The only available queue on SR004 is "default".
)
MAX_RETRY: Optional[int] = (
    None  # Maximum number of retries. Cannot be 0, use None for no retries.
)
# EDITABLE PARAMETERS END
# ---

assert CONFIG is not None
assert N_WORKERS is not None
assert USER is not None
assert EXP_NAME is not None
assert JOB_TYPE is not None

assert JOB_TYPE in ["train", "inf", "convert"]
assert USER not in ["", "...", "your_mm_username"]
assert EXP_NAME not in ["", "...", "new_sota"]
assert N_WORKERS > 0
assert MAX_RETRY is None or MAX_RETRY > 0

wandb_env_path = ROOT_DIR / "sber_configs" / ".wandb.env"
assert wandb_env_path.is_file(), f".wandb.env does not exist: {wandb_env_path}"
wandb_env: dict[str, str] = {}
for raw_line in wandb_env_path.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, value = line.split("=", 1)
    wandb_env[key.strip()] = value.strip().strip('"').strip("'")
wandb_api_key = wandb_env.get("WANDB_API_KEY")
wandb_base_url = wandb_env.get("WANDB_BASE_URL")
assert wandb_api_key is not None, "WANDB_API_KEY is missing in sber_configs/.wandb.env"
assert wandb_api_key.strip() not in ("", "..."), (
    "WANDB_API_KEY must be a non-empty real key in sber_configs/.wandb.env"
)
assert wandb_base_url == "https://wandb.sberdevices.ru", (
    "WANDB_BASE_URL must be https://wandb.sberdevices.ru in "
    f"sber_configs/.wandb.env, got: {wandb_base_url!r}"
)

if JOB_TYPE == "train":
    config_path = Path(CONFIG)
    assert config_path.is_file(), f"Config does not exist: {config_path}"
    config = OmegaConf.load(config_path)
    assert config.get("omni_mode") is True, "Config must have omni_mode: true"
    assert isinstance(config.global_train_batch_size, int)
    assert config.global_train_batch_size > 0
    assert isinstance(config.max_duration, str)
    assert config.max_duration.endswith("ba")
    assert config.max_duration.removesuffix("ba").isdigit()
    assert isinstance(config.save_interval, str)
    assert config.save_interval.endswith("ba")
    assert config.save_interval.removesuffix("ba").isdigit()
    assert isinstance(config.t_warmup, str)
    assert config.t_warmup.endswith("ba")
    assert config.t_warmup.removesuffix("ba").isdigit()
    assert config.eval_interval == 0 or (
        isinstance(config.eval_interval, str)
        and config.eval_interval.endswith("ba")
        and config.eval_interval.removesuffix("ba").isdigit()
    ), "eval_interval must be 0 or have Nba format"
    assert config.user is not None
    assert str(config.user).strip() not in ("", "...")
    assert config.exp_name is not None
    assert str(config.exp_name).strip() not in ("", "...")
    assert config.wandb_project is not None
    assert str(config.wandb_project).strip() not in ("", "...")
    assert config.load_path is not None
    assert str(config.load_path).strip() not in ("", "...")
    assert config.ve_path is not None
    assert str(config.ve_path).strip() not in ("", "...")
    assert config.tokenizer_path is not None
    assert str(config.tokenizer_path).strip() not in ("", "...")
    if "language_model" in config.disable_grad:
        assert config.model.language_model.moe_router_bias_update_rate == 0.0, (
            "moe_router_bias_update_rate must be 0.0 when language_model is frozen"
        )
    if not config.icl_tasks:
        assert config.eval_interval == 0, (
            "eval_interval must be 0 when icl_tasks is empty"
        )
    assert len(config.model.encoders) == 1, "Exactly one vision encoder is required"
    assert len(config.model.adapters) == 1, "Exactly one vision adapter is required"
    assert len(config.model.modalities) == 1, "Exactly one vision modality is required"
    encoder = config.model.encoders[0]
    adapter = config.model.adapters[0]
    modality = config.model.modalities[0]
    assert encoder.pretrained_model_name_or_path == config.ve_path
    assert encoder.init_device == config.model.init_device
    assert adapter.init_device == config.model.init_device
    assert adapter.llm_hidden_size == config.model.language_model.hidden_size
    assert modality.name == "VisionModality"
    assert modality.modality_output.type == "adapter_outputs"
    assert modality.modality_output.key == "pixel_embeds"
    assert modality.modality_output.placeholder_token_id == config.image_token_id
    assert config.vision_processor.name == "GigaOmniVisionProcessor"
    assert config.vision_processor.image_processor == config.ve_path
    assert config.vision_processor.image_token_id == config.image_token_id
    assert len(config.train_loader.dataset.processors) == 1
    assert config.train_loader.dataset.processors[0] == config.vision_processor
    train_collators = config.train_loader.dataset.collators
    assert train_collators, "Train collators must not be empty"
    assert config.image_collator in train_collators
    for task in config.icl_tasks:
        assert len(task.sft_args.processors) == 1
        assert task.sft_args.processors[0] == config.vision_processor
        assert task.sft_args.collators, "Evaluation task collators must not be empty"
        assert config.image_collator in task.sft_args.collators

    if encoder.name == "InternViTForVisualLanguageModeling":
        assert adapter.name == "VisionPixelShuffleAdapter"
        assert adapter.ve_hidden_size == 1024
        assert adapter.downsample_ratio == 0.5
        assert modality.data_flow_preset == "vision_intern_vit"
        assert modality.modality_output.name == "pixel_shuffle_mlp.image"
        assert config.num_image_tokens_per_crop == 256
        assert config.vision_processor.image_aspect_ratio == "anyres"
        assert (
            config.vision_processor.num_image_tokens_per_crop
            == config.num_image_tokens_per_crop
        )
        assert (
            config.vision_processor.image_grid_pinpoints == config.image_grid_pinpoints
        )
        assert config.image_collator.name == "ImageCollator"
    else:
        assert encoder.name == "Qwen3ViTForVisualLanguageModeling", (
            f"Unsupported vision encoder: {encoder.name}"
        )
        assert adapter.name == "VisionQwen3Adapter"
        assert adapter.ve_hidden_size == 1152
        assert adapter.spatial_merge_size == 2
        assert adapter.initializer_range == 0.02
        assert modality.data_flow_preset == "vision_qwen3_vit"
        assert modality.modality_output.name == "qwen3_adapter.image"
        assert config.num_image_tokens_per_crop == 1
        assert config.qwen_max_image_tokens == 2048
        assert config.vision_processor.image_aspect_ratio == "qwen"
        assert (
            config.vision_processor.num_image_tokens_per_crop
            == config.num_image_tokens_per_crop
        )
        assert (
            config.vision_processor.qwen_max_image_tokens
            == config.qwen_max_image_tokens
        )
        assert config.image_collator.name == "QwenImageCollator"

    dataset = config.train_loader.dataset
    if dataset.shuffle is True:
        assert "partition_algo" not in dataset, (
            "partition_algo must not be set when shuffle is true"
        )
        assert "num_canonical_nodes" not in dataset, (
            "num_canonical_nodes must not be set when shuffle is true"
        )
    else:
        assert dataset.shuffle is False, "shuffle must be true or false"
        assert dataset.partition_algo == "orig", (
            "partition_algo must be orig when shuffle is false"
        )
        assert dataset.num_canonical_nodes == 1, (
            "num_canonical_nodes must be 1 when shuffle is false"
        )
    assert Path(config.load_path).exists(), (
        f"load_path does not exist: {config.load_path}"
    )
    assert Path(config.ve_path).exists(), f"ve_path does not exist: {config.ve_path}"
    assert Path(config.tokenizer_path).exists(), (
        f"tokenizer_path does not exist: {config.tokenizer_path}"
    )
    for stream_name, stream in config.train_loader.dataset.streams.items():
        assert Path(stream.local).exists(), (
            f"train stream {stream_name}.local does not exist: {stream.local}"
        )
        assert Path(stream.files_folder_path).exists(), (
            "train stream "
            f"{stream_name}.files_folder_path does not exist: "
            f"{stream.files_folder_path}"
        )
    for task in config.icl_tasks:
        assert Path(task.dataset_uri).exists(), (
            f"ICL task {task.label}.dataset_uri does not exist: {task.dataset_uri}"
        )

    job = TrainJobLLMF(
        user_login=USER,
        exp_name=EXP_NAME,
        path_to_config=CONFIG,
        script_type="pytorch2",
        use_clearml=False,
        script_args="",
        processes_per_worker=8,
        n_workers=N_WORKERS,
        queue_name=QUEUE_NAME,
        max_retry=MAX_RETRY,
    )
elif JOB_TYPE == "inf":
    job = InfiniteJob(
        user_login=USER,
        exp_name=f"{USER} | infinity job {EXP_NAME}",
        processes_per_worker=8,
        n_workers=1,
        queue_name=QUEUE_NAME,
        max_retry=MAX_RETRY,
    )
elif JOB_TYPE == "convert":
    convertation_script_path = (
        f"{ROOT_DIR}/contrib/gigachat-utils/convertation/map-fsdp-to-fsdp.py"
    )
    script = f"python {convertation_script_path} --input-config-path {CONFIG}"
    job = TestConvertationJobGigachatUtils(
        user_login=USER,
        exp_name=f"{USER} | convertation job {EXP_NAME}",
        script=script,
        processes_per_worker=1,
        n_workers=1,
        n_gpus=1,
        queue_name=QUEUE_NAME,
        max_retry=MAX_RETRY,
    )
else:
    raise ValueError(f"Invalid job type: {JOB_TYPE}")
result = job.submit()
print(result)
