from dataclasses import field
from typing import ClassVar, Literal

from torchtitan.config.configurable import config, Configurable
from torchtitan.config.errors import ConfigFieldError


@config(family="checkpoint", default="default", type="default")
class CheckpointManagerConfig(Configurable.Config):
    _owner_spec: ClassVar[str] = "torchtitan.components.checkpoint:CheckpointManager"

    def __post_init__(self) -> None:
        if type(self.interval) is not int or self.interval <= 0:
            raise ConfigFieldError("interval", "must be a positive integer")
        if (
            type(self.keep_latest_k) is not int
            or self.keep_latest_k < 0
            or self.keep_latest_k == 1
        ):
            raise ConfigFieldError(
                "keep_latest_k", "must be 0 (keep all) or at least 2"
            )
        self.validate_save_options()

    def validate_save_options(self) -> None:
        """Reject incompatible final-save options before runtime allocation."""
        if (
            self.hf_save_enabled
            and self.last_save_in_hf
            and not self.last_save_model_only
        ):
            raise ConfigFieldError(
                "last_save_in_hf", "requires checkpoint.last_save_model_only=true"
            )

    @property
    def hf_save_enabled(self) -> bool:
        """Whether this run will save a checkpoint in HF format."""
        return (
            self.enable
            and not self.load_only
            and (self.save_in_hf or self.last_save_in_hf)
        )

    enable: bool = False
    """Whether to enable checkpoint"""

    folder: str = "checkpoint"
    """
    The folder to store the checkpoints.
    When enable is set to true, checkpoints will be in {--dump_folder}/{--checkpoint.folder}.
    """

    interval: int = 500
    """Checkpointing interval in steps."""

    initial_load_path: str | None = None
    """
    This option specifies the path to the initial checkpoint to load, which is
    particularly useful for resuming training from a previous run with a
    different output path or when loading a checkpoint from a pre-trained model.
    If the checkpoint folder for the current run is not empty,
    located at {--dump_folder}/{--checkpoint.folder}, this option will be ignored.
    This feature allows users to load an initial checkpoint from a different folder and
    continue training, saving new checkpoints to the specified folder without affecting
    the existing ones.

    Note that the path should contain the full path to the checkpoint folder,
    including the step number, if any; for example,
    "//pre_train/checkpoints/llama3/llama3_8b/step_10000".
    """

    initial_load_model_only: bool = True
    """
    This option specifies if only the model should be loaded during the initial
    checkpoint load. The option is only used when `initial_load_path` is specified.
    If False, the checkpoint at `initial_load_path` is treated as a standard training
    checkpoint, including optimizer, lr scheduler, training states, etc.
    The default setting for this option is True. Note that you will have to use
    `--checkpoint.initial_load_model_only false` to override the default setting.
    """

    initial_load_in_hf: bool = False
    """
    Enable the use of HuggingFace's safetensors format for checkpointing. The option
    is only used when `initial_load_path` is specified. This will load checkpoints
    in HF's model definition and safetensors format instead of the default torchtitan
    model definition and DCP format, after necessary model state dict transformation.
    `initial_load_model_only` must be true because safetensors doesn't support saving
    non-tensors. The default value is False.
    """

    initial_load_in_hf_quantized: bool = False
    """
    Enable loading of HuggingFace's safetensors format with quantized state dict keys. The option
    is only used when `initial_load_path` and `initial_load_path_in_hf` is specified. This will load
    checkpoints in HF's model definition and dequantize on model weights if necessary. To support
    this parameter, the model need to define proper HuggingFaceStorageReader to perform dequantize.
    """

    last_save_model_only: bool = True
    """
    When last_save_model_only=True, only the model will be saved at the end of training,
    the last save.  With this, checkpoints can be loaded using `torch.load(..., weights_only=True)`
    after conversion.  When last_save_model_only=False, the full checkpoint will be saved.
    A full checkpoint includes model, optimizer and train_state, which can be used to resume training.
    The default value is True.
    """

    save_in_hf: bool = False
    """
    Also export a model-only HF safetensors copy on EVERY interval save, into
    `{checkpoint}/artifacts/hf_checkpoint`. `last_save_in_hf` only covers the
    final step; downstream evaluation needs HF for the intermediate
    checkpoints too. Costs one extra model-sized write per save.
    """

    last_save_in_hf: bool = False
    """
    Enable the use of Hugging Face's safetensors format for checkpointing. This will save the
    final checkpoints in safetensors format instead of the default DCP format, after necessary
    model state dict transformation. There will be a performance cost in using this as we need
    to consolidate the sharded tensors to full tensors as a separate step.
    last_save_model_only must be true because safetensors doesn't support saving
    non-tensors. On load, this argument isn't needed as we will detect whether the loaded
    checkpoint is in safetensors format or not. The default value is False.
    """

    last_save_ema_model_also: bool = False
    """
    Also save EMA weights as a separate model-only DCP checkpoint at the
    end of training. Requires the EMA callback to be enabled.
    """

    ema_folder: str = "ema_checkpoint"
    """
    Folder to store EMA model-only checkpoints under dump_folder.
    """

    hf_remote_code_dir: str = ""
    """
    Directory holding the remote-code files (``modeling_*.py`` /
    ``configuration_*.py``) an HF export must ship when its ``config.json``
    carries ``auto_map``. Which files, and where they live by default, the
    model's state dict adapter says (``hf_remote_code_files`` /
    ``default_hf_remote_code_dir``); empty means that default. Ignored for
    architectures ``transformers`` knows natively.
    """

    export_dtype: Literal["float16", "bfloat16", "float32"] = "float32"
    """
    Converts to the specified precision when training completes and last_save_model_only=true.
    """

    async_mode: Literal["disabled", "async", "async_with_pinned_mem"] = "disabled"
    """
    Which async checkpoint mode to use. Currently there are 3 different modes.

    - "disabled": synchronized checkpointing will be used.
    - "async": torch.distributed.checkpoint.async_save will be used.
    - "async_with_pinned_mem": this option utilizes a dedicated pinned memory space and creates a
      separate process for faster GPU->CPU transfer performance and eliminating GIL contention.
      The cost is increased CPU memory usage. If insufficient CPU memory is available, performance
      may degrade due to memory paging. For most users, "async" should suffice as the performance
      overhead is typically small (on the order of tens of seconds) compared to checkpointing
      frequency. This mode can be employed to pursue near-zero checkpointing times
      (e.g., < 1 second) given appropriate hardware support such as ample CPU memory and fast PCIe.

    "disabled" is the default mode.
    """

    keep_latest_k: int = 10
    """
    Keeps only the latest k checkpoints, and purging older ones. If 0, keep all checkpoints.
    K cannot be 1 as the last one may be in the process of being saved. As a result,
    the metadata of the last one may not be ready yet. The default value is 10 to avoid
    filling up the disk.
    """

    load_step: int = -1
    """Load the checkpoint at the specified step. If -1, load the latest checkpoint."""

    exclude_from_loading: list[str] = field(default_factory=list)
    """
    Exclude specific keys from being loaded from the checkpoint.
    Provide a comma-separated list of keys to exclude, e.g. 'optimizer,lr_scheduler,dataloader'.
    This will load the model only, excluding the specified keys.
    """

    enable_first_step_checkpoint: bool = False
    """
    Enable the checkpoint save at first step. This will save a checkpoint immediately
    after the first step to ensure checkpointing functions correctly. This is useful
    when running on a new cluster or storage to verify checkpointing without waiting
    for many steps or checkpointing too frequently. The default value is False.
    """

    create_seed_checkpoint: bool = False
    """
    Initializes the full model without applying parallelisms, and then saves it as a seed checkpoint.
    Note: requires user to call train.py without specifying any parallelisms, e.g. NGPU=1.
    Could be implemented as a separate script, but this way shares more code.
    """

    load_only: bool = False
    """
    In certain scenarios, you may only need to load checkpoints for verification or debugging
    purposes, without saving any new checkpoints. For example, you might use seed checkpoints
    to validate model correctness. Enabling this option allows checkpoints to be loaded
    without saving any during the training.
    """
