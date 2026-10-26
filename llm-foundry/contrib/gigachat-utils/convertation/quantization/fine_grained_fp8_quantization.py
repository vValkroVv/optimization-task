import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import safetensors
import torch
from huggingface_hub import split_torch_state_dict_into_shards
from omegaconf import OmegaConf as om
from safetensors.torch import save_file as safe_save_file
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    FineGrainedFP8Config,
    PreTrainedModel,
)
from transformers.modeling_utils import _load_parameter_into_model
from transformers.quantizers.quantizer_finegrained_fp8 import FineGrainedFP8HfQuantizer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def quantize_single_tensor(
    param_value: torch.Tensor,
    block_size_m: int,
    block_size_n: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a single weight tensor to FP8 using block-wise quantization with padding.

    The tensor is reshaped into blocks of shape (block_size_m, block_size_n)
    along the last two dimensions, quantized per block with its own scale,
    and then reshaped back. If necessary, padding is applied before
    quantization and removed afterwards.

    Args:
        param_value: Input tensor to quantize. Must have at least 2 dimensions.
        block_size_m: Block size along the second-to-last dimension.
        block_size_n: Block size along the last dimension.

    Returns:
        A tuple of:
            - quantized_param: The quantized tensor in torch.float8_e4m3fn.
            - scale: A tensor of inverse scales (weight_scale_inv) for each block.

    Raises:
        AssertionError: If block sizes are invalid or tensor rank < 2.
    """
    assert isinstance(param_value, torch.Tensor), "param_value must be a torch.Tensor"
    assert param_value.dim() >= 2, "param_value must have at least 2 dimensions"
    assert block_size_m > 0 and block_size_n > 0, "block sizes must be positive integers"

    # Get FP8 min/max values
    fp8_min = torch.finfo(torch.float8_e4m3fn).min
    fp8_max = torch.finfo(torch.float8_e4m3fn).max

    # Save original dimensions
    rows, cols = param_value.shape[-2:]

    assert rows > 0 and cols > 0, "Last two tensor dimensions must be non-zero"

    # Track padding
    padded_rows = rows
    padded_cols = cols
    row_padding = 0
    col_padding = 0

    # Check if padding is needed and apply it
    needs_row_padding = rows % block_size_m != 0
    needs_col_padding = cols % block_size_n != 0

    if needs_row_padding or needs_col_padding:
        row_padding = (block_size_m - rows % block_size_m) % block_size_m
        col_padding = (block_size_n - cols % block_size_n) % block_size_n

        if row_padding > 0 or col_padding > 0:
            param_value = torch.nn.functional.pad(
                param_value,
                (0, col_padding, 0, row_padding),
            )
            padded_rows = rows + row_padding
            padded_cols = cols + col_padding

    # Use padded dimensions for quantization
    rows_quant, cols_quant = padded_rows, padded_cols
    param_value_orig_shape = param_value.shape

    assert rows_quant % block_size_m == 0, "rows_quant must be divisible by block_size_m"
    assert cols_quant % block_size_n == 0, "cols_quant must be divisible by block_size_n"

    # Reorganize matrix into blocks for quantization
    param_value = param_value.reshape(
        -1,
        rows_quant // block_size_m,
        block_size_m,
        cols_quant // block_size_n,
        block_size_n,
    ).permute(0, 1, 3, 2, 4)

    # Calculate scaling factor for each block
    max_abs = torch.amax(torch.abs(param_value), dim=(-1, -2)).to(torch.float32)

    # Avoid division by zero: blocks that are all zeros just keep scale=1
    max_abs_safe = torch.where(max_abs == 0, torch.ones_like(max_abs), max_abs)
    scale = fp8_max / max_abs_safe
    scale_orig_shape = scale.shape
    scale = scale.unsqueeze(-1).unsqueeze(-1)

    # Quantize the weights
    quantized_param = torch.clamp(
        param_value * scale,
        min=fp8_min,
        max=fp8_max,
    ).to(torch.float8_e4m3fn)

    # Restore original shape
    quantized_param = quantized_param.permute(0, 1, 3, 2, 4)
    quantized_param = quantized_param.reshape(param_value_orig_shape)

    # Trim padding if it was applied
    if row_padding > 0 or col_padding > 0:
        slices = [slice(None)] * quantized_param.dim()
        slices[-2] = slice(0, rows)  # second to last dimension (rows)
        slices[-1] = slice(0, cols)  # last dimension (columns)
        quantized_param = quantized_param[tuple(slices)]

    # Reshape scale to match the number of blocks and invert (weight_scale_inv)
    scale = scale.reshape(scale_orig_shape).squeeze().reciprocal()

    return quantized_param, scale


def patched_create_quantized_param(
    self: FineGrainedFP8HfQuantizer,
    model: PreTrainedModel,
    param_value: torch.Tensor,
    param_name: str,
    target_device: torch.device,
    state_dict: Dict[str, Any],
    unexpected_keys: Optional[List[str]] = None,
) -> None:
    """
    Patched version of FineGrainedFP8HfQuantizer.create_quantized_param.

    Uses custom block-wise FP8 quantization with padding support to quantize
    a parameter and load both weight and weight_scale_inv into the model.

    Args:
        self: The quantizer instance.
        model: The target model to load quantized parameters into.
        param_value: The original (e.g., bf16) weight tensor.
        param_name: Name of the parameter in the model.
        target_device: Device where the parameter is expected to be loaded.
        state_dict: State dict used by HF loading utilities (unused here).
        unexpected_keys: Optional list of unexpected keys (unused here).
    """
    assert hasattr(self, "quantization_config"), "Quantizer must have quantization_config"
    assert hasattr(self.quantization_config, "weight_block_size"), (
        "quantization_config must define weight_block_size"
    )

    block_size_m, block_size_n = self.quantization_config.weight_block_size

    quantized_param, scale = quantize_single_tensor(
        param_value,
        block_size_m,
        block_size_n,
    )

    # Load quantized weight and inverse scale into the model
    _load_parameter_into_model(model, param_name, quantized_param)
    scale_name = param_name.rsplit(".", 1)[0] + ".weight_scale_inv"
    _load_parameter_into_model(model, scale_name, scale)


def find_missing_keys(
    bf16_index: Dict[str, Any],
    fp8_index: Dict[str, Any],
) -> Set[str]:
    """
    Find keys that exist in the bf16 model index but are missing in the fp8 index.

    Args:
        bf16_index: Index dictionary for the original (bf16) model.
        fp8_index: Index dictionary for the quantized (fp8) model.

    Returns:
        A set of parameter names that are present in bf16 but missing in fp8.
    """
    assert "weight_map" in bf16_index, "bf16_index must contain 'weight_map'"
    assert "weight_map" in fp8_index, "fp8_index must contain 'weight_map'"

    bf16_keys = set(bf16_index.get("weight_map", {}).keys())
    fp8_keys = set(fp8_index.get("weight_map", {}).keys())

    missing_keys = bf16_keys - fp8_keys

    logger.info("Total keys in bf16 model: %d", len(bf16_keys))
    logger.info("Total keys in fp8 model: %d", len(fp8_keys))
    logger.info("Found missing keys: %d", len(missing_keys))

    return missing_keys


def load_missing_weights(
    bf16_model_path: str,
    bf16_index: Dict[str, Any],
    missing_keys: Set[str],
) -> Dict[str, torch.Tensor]:
    """
    Load missing weights from the bf16 model according to the provided index.

    Only the tensors corresponding to missing_keys are loaded, grouped by file
    for efficiency.

    Args:
        bf16_model_path: Path to the original bf16 model directory.
        bf16_index: Index dictionary for the bf16 model.
        missing_keys: Set of parameter names that are missing in the fp8 model.

    Returns:
        A mapping from parameter name to torch.Tensor for the missing weights.
    """
    assert os.path.isdir(bf16_model_path), f"{bf16_model_path} is not a directory"
    assert "weight_map" in bf16_index, "bf16_index must contain 'weight_map'"

    weight_map = bf16_index.get("weight_map", {})
    missing_weights: Dict[str, torch.Tensor] = {}

    # Group keys by file for efficient loading
    file_to_keys: Dict[str, List[str]] = {}
    for key in missing_keys:
        file_name = weight_map.get(key)
        if file_name is not None:
            file_to_keys.setdefault(file_name, []).append(key)

    # Load weights from each file
    for file_name, keys in file_to_keys.items():
        file_path = os.path.join(bf16_model_path, file_name)
        logger.info("Loading %d weights from %s", len(keys), file_path)

        try:
            with safetensors.safe_open(file_path, framework="pt") as f:
                for key in keys:
                    if key in f.keys():
                        missing_weights[key] = f.get_tensor(key)
                    else:
                        logger.warning("Key %s not found in file %s", key, file_name)
        except Exception as exc:
            logger.error("Error loading file %s: %s", file_path, exc)

    return missing_weights


def quantize_missing_weights(
    missing_weights: Dict[str, torch.Tensor],
    quantization_config: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    """
    Quantize missing weights using the same block-wise FP8 scheme.

    For each tensor:
      * If it is 1D or matches any pattern from modules_to_not_convert,
        it is kept in original precision.
      * Otherwise it is quantized and an additional '<name>.weight_scale_inv'
        tensor is produced.

    Args:
        missing_weights: Mapping from parameter name to original bf16 tensors.
        quantization_config: Quantization configuration dictionary containing:
            - "weight_block_size": Tuple[int, int]
            - "modules_to_not_convert": Iterable of substrings for keys to skip.

    Returns:
        Mapping from tensor name to tensor (quantized weights + their scales).
    """
    assert "weight_block_size" in quantization_config, "quantization_config must have 'weight_block_size'"
    assert "modules_to_not_convert" in quantization_config, (
        "quantization_config must have 'modules_to_not_convert'"
    )

    block_size_m, block_size_n = quantization_config["weight_block_size"]
    modules_to_not_convert = quantization_config["modules_to_not_convert"]

    quantized_results: Dict[str, torch.Tensor] = {}

    for key, tensor in tqdm(missing_weights.items(), desc="Quantizing missing weights"):
        try:
            if tensor.dim() == 1 or any(pattern in key for pattern in modules_to_not_convert):
                # Keep original precision
                quantized_results[key] = tensor
                logger.info("Keeping original precision for key: %s", key)
            else:
                quantized_tensor, scale = quantize_single_tensor(
                    tensor,
                    block_size_m,
                    block_size_n,
                )
                scale_key = key.rsplit(".", 1)[0] + ".weight_scale_inv"
                quantized_results[key] = quantized_tensor
                quantized_results[scale_key] = scale

        except Exception as exc:
            logger.error("Error quantizing %s: %s", key, exc)

    return quantized_results


def get_metadata(quantized_results: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """
    Compute simple metadata (total_parameters, total_size) for a tensor dict.

    Args:
        quantized_results: Mapping from tensor name to tensor.

    Returns:
        A dictionary with:
            - "total_parameters": total number of elements.
            - "total_size": total size in bytes.
    """
    total_parameters = 0
    total_size = 0

    for tensor in quantized_results.values():
        total_parameters += tensor.numel()
        total_size += tensor.numel() * tensor.element_size()

    return {
        "total_parameters": int(total_parameters),
        "total_size": int(total_size),
    }


def update_checkpoint(
    index: Dict[str, Any],
    save_path: str,
    num_add_files: int,
) -> int:
    """
    Rename existing .safetensors shards so that the total number of shards becomes
    (old + num_add_files), and update the weight_map in the index accordingly.

    Supported cases:
      1) Existing files already follow HF naming convention:
         'model-00001-of-000NN.safetensors' — only the '-of-NN' part is updated.
      2) Existing files have arbitrary names (e.g. single 'model.safetensors') —
         we assume there are len(files) shards and rename them to
         'model-00001-of-XXXXX', 'model-00002-of-XXXXX', ..., updating index.

    Args:
        index: Index dictionary containing "weight_map".
        save_path: Directory with current safetensors shards.
        num_add_files: Number of new shards that will be added.

    Returns:
        The new total number of shard files (old + num_add_files).

    Raises:
        AssertionError: If num_add_files <= 0.
        RuntimeError: If no .safetensors files are found in save_path.
    """
    assert num_add_files > 0, "num_add_files must be positive"
    assert os.path.isdir(save_path), f"{save_path} is not a directory"
    assert "weight_map" in index, "index must contain 'weight_map'"

    suffix = ".safetensors"
    files = sorted(f for f in os.listdir(save_path) if f.endswith(suffix))

    if not files:
        raise RuntimeError(f"No .safetensors files found in {save_path}")

    # Try to detect HF naming pattern
    match = re.search(r"model-(\d+)-of-(\d+)\.safetensors$", files[0])
    if match:
        # Common case: update shard count
        num_old_files = int(match.group(2))
        num_new_files = num_old_files + num_add_files

        for file_name in files:
            new_file_name = change_safetensor_name(file_name, num_new_files)
            old_path = os.path.join(save_path, file_name)
            new_path = os.path.join(save_path, new_file_name)
            if new_file_name != file_name:
                shutil.move(old_path, new_path)
            for key, value in index["weight_map"].items():
                if value == file_name:
                    index["weight_map"][key] = new_file_name
    else:
        # Arbitrary names, e.g. single 'model.safetensors'
        num_old_files = len(files)
        num_new_files = num_old_files + num_add_files

        logger.info(
            "Existing safetensors do not follow HF naming pattern, "
            "renaming %d files to HF style with total shards=%d",
            num_old_files,
            num_new_files,
        )

        for i, file_name in enumerate(files, start=1):
            new_file_name = f"model-{i:05d}-of-{num_new_files:05d}.safetensors"
            old_path = os.path.join(save_path, file_name)
            new_path = os.path.join(save_path, new_file_name)
            shutil.move(old_path, new_path)

            for key, value in index["weight_map"].items():
                if value == file_name:
                    index["weight_map"][key] = new_file_name

    return num_new_files


def change_safetensor_name(
    name: str,
    pattern: int,
    position: int = 1,
) -> str:
    """
    Change either the 'of' part or the shard index in a HF-style safetensor name.

    Args:
        name: Original filename.
        pattern: New numeric pattern to insert (zero-padded to 5 digits).
        position: If 1, replace 'of' part, otherwise replace shard index
                  after 'model-'.

    Returns:
        Updated filename string.
    """
    if position == 1:
        # Update the 'of-XXXXX' part
        return re.sub(
            r"(?<=-of-)\d+(?=\.safetensors)",
            f"{pattern:05d}",
            name,
        )

    # Update the 'model-XXXXX-' part
    return re.sub(
        r"(?<=model-)\d+(?=-of-)",
        f"{pattern:05d}",
        name,
    )


def merge_indexes(
    index1: Dict[str, Any],
    index2: Dict[str, Any],
    index_path: str,
) -> None:
    """
    Merge two index dictionaries and write the result to disk.

    Metadata values are summed, and weight_map entries from index2 override
    or extend those in index1.

    Args:
        index1: Base index to be updated in-place.
        index2: Index whose metadata and weight_map should be added.
        index_path: Path where the merged index.json will be saved.
    """
    assert "metadata" in index1 and "metadata" in index2, "Both indexes must contain 'metadata'"
    assert "weight_map" in index1 and "weight_map" in index2, "Both indexes must contain 'weight_map'"

    for key in index1["metadata"].keys():
        index1["metadata"][key] += index2["metadata"].get(key, 0)

    for key, value in index2["weight_map"].items():
        index1["weight_map"][key] = value

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index1, f, indent=2)


def save_quantized_weights(
    quantized_results: Dict[str, torch.Tensor],
    output_path: str,
    fp8_index: Dict[str, Any],
    fp8_index_path: Optional[str],
) -> None:
    """
    Save quantized MTP weights into new safetensor shard files and update the index.

    If the fp8 index file did not previously exist, it will be created. Existing
    shards may be renamed to HF naming convention if necessary.

    Args:
        quantized_results: Mapping from tensor name to quantized tensors.
        output_path: Directory where the fp8 model checkpoint is stored.
        fp8_index: Current fp8 index dictionary (modified in-place).
        fp8_index_path: Path to the fp8 index file, or None if it does not exist.
    """
    assert quantized_results, "quantized_results must not be empty"
    assert os.path.isdir(output_path), f"{output_path} is not a directory"
    assert "weight_map" in fp8_index, "fp8_index must contain 'weight_map'"

    state_dict_split = split_torch_state_dict_into_shards(quantized_results)

    new_index = {
        "metadata": get_metadata(quantized_results),
    }
    num_add_files = len(state_dict_split.filename_to_tensors.keys())
    assert num_add_files > 0, "There must be at least one shard to save"

    # If index.json was missing, choose a default filename
    if fp8_index_path is None:
        fp8_index_path = os.path.join(output_path, "model.safetensors.index.json")

    # Rename existing shard files to align with new total shard count
    num_new_files = update_checkpoint(fp8_index, output_path, num_add_files)

    weight_map: Dict[str, str] = {}
    logger.info("Saving %d tensors to %s", num_add_files, output_path)

    for idx, (filename, tensors) in enumerate(
        state_dict_split.filename_to_tensors.items(),
        start=1,
    ):
        if num_add_files == 1:
            # Single new shard -> last shard
            filename = f"model-{num_new_files:05d}-of-{num_new_files:05d}.safetensors"
        else:
            # Multiple shards: fix both total shard count and shard index
            filename = change_safetensor_name(filename, num_new_files)  # update -of-XXXXX
            filename = change_safetensor_name(
                filename,
                num_new_files - num_add_files + idx,
                position=0,  # update model-XXXXX-
            )

        shard = {tensor_name: quantized_results[tensor_name] for tensor_name in tensors}
        safe_save_file(
            shard,
            os.path.join(output_path, filename),
            metadata={"format": "pt"},
        )
        for tensor_name in shard:
            weight_map[tensor_name] = filename

    new_index["weight_map"] = weight_map

    # Merge updated index into fp8_index and write to disk
    merge_indexes(fp8_index, new_index, fp8_index_path)

    logger.info("Updated index.json saved to %s", fp8_index_path)
    logger.info("Total keys in updated index.json: %d", len(fp8_index["weight_map"]))


def load_or_build_index(
    model_path: str,
    compute_metadata: bool = True,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Load an existing index.json or build an in-memory index for a single-safetensor model.

    Behavior:
      * If index.json exists in model_path, it is loaded and returned with its path.
      * If index.json does not exist, it is assumed that there is exactly one
        .safetensors file in model_path, and an index-like dictionary is built
        from its contents. The file is not written to disk here.

    Args:
        model_path: Path to the model directory.
        compute_metadata: If True, total_parameters and total_size are computed
            from tensor contents (only safe for small models).

    Returns:
        A tuple (index, index_path) where:
          - index: Index-like dictionary with "metadata" and "weight_map".
          - index_path: Path to existing index.json or None if it did not exist.

    Raises:
        RuntimeError: If no index.json exists and the directory does not contain
                      exactly one .safetensors file.
    """
    assert os.path.isdir(model_path), f"{model_path} is not a directory"

    # Try to find an existing index.json (HF often uses model.safetensors.index.json)
    index_files = [f for f in os.listdir(model_path) if f.endswith("index.json")]
    if index_files:
        index_path = os.path.join(model_path, index_files[0])
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        assert "weight_map" in index, "Loaded index.json must contain 'weight_map'"
        return index, index_path

    # Fallback: assume a small model with a single .safetensors file
    suffix = ".safetensors"
    safetensor_files = sorted(f for f in os.listdir(model_path) if f.endswith(suffix))

    if len(safetensor_files) != 1:
        raise RuntimeError(
            "Expected exactly one *.safetensors file when index.json is missing, "
            f"but found {len(safetensor_files)}: {safetensor_files}",
        )

    filename = safetensor_files[0]
    file_path = os.path.join(model_path, filename)

    weight_map: Dict[str, str] = {}
    metadata = {"total_parameters": 0, "total_size": 0}

    logger.info(
        "index.json not found in %s, building in-memory index from %s",
        model_path,
        filename,
    )

    with safetensors.safe_open(file_path, framework="pt") as f:
        for key in f.keys():
            weight_map[key] = filename
            if compute_metadata:
                tensor = f.get_tensor(key)
                metadata["total_parameters"] += tensor.numel()
                metadata["total_size"] += tensor.numel() * tensor.element_size()

    if not compute_metadata:
        metadata = {}

    index = {
        "metadata": metadata,
        "weight_map": weight_map,
    }

    # We do not write index.json here; for fp8 it will be written/updated later.
    return index, None


def main() -> None:
    """
    Entry point for bf16 -> fp8 quantization with optional MTP weight handling.

    Steps:
      1. Load quantization configuration from configs/quantization_config.yaml.
      2. Run HF FineGrained FP8 quantization via AutoModelForCausalLM.from_pretrained.
      3. Save the quantized model and tokenizer.
      4. Build/load indices for bf16 and fp8 checkpoints.
      5. Detect missing keys (e.g., MTP layers), load them from bf16,
         quantize them to FP8, save into new shards, and update fp8 index.
    """
    config_path = Path(__file__).parent.parent / "configs/quantization_config.yaml"
    assert config_path.is_file(), f"Config file not found at {config_path}"

    config = om.load(config_path)
    cfg: Dict[str, Any] = om.to_container(config, resolve=True)

    assert "input_checkpoint_path" in cfg, "Config must define 'input_checkpoint_path'"
    assert "output_checkpoint_path" in cfg, "Config must define 'output_checkpoint_path'"
    assert "quantization_config" in cfg, "Config must define 'quantization_config'"
    assert "device_map" in cfg, "Config must define 'device_map'"

    model_path = cfg["input_checkpoint_path"]
    save_path = cfg["output_checkpoint_path"]

    parent_dir = os.path.dirname(save_path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
        logger.info("%s was created!", parent_dir)

    quantization_config = FineGrainedFP8Config(**cfg["quantization_config"])

    # Patch the quantizer to use our custom block-wise quantization
    original_method = FineGrainedFP8HfQuantizer.create_quantized_param
    FineGrainedFP8HfQuantizer.create_quantized_param = patched_create_quantized_param

    try:
        logger.info("Starting model loading and base FP8 quantization...")

        quantized_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=cfg["device_map"],
            quantization_config=quantization_config,
            torch_dtype="auto",
            trust_remote_code=True,
        )

        logger.info("Saving quantized model to %s ...", save_path)
        quantized_model.save_pretrained(save_path)

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
            tokenizer.save_pretrained(save_path)
        except Exception as exc:
            logger.warning("The tokenizer was not found or could not be saved: %s", exc)

        logger.info("Base quantization completed successfully.")

    except Exception as exc:
        logger.error("Error during base quantization: %s", exc)
        raise
    finally:
        # Always restore the original method
        FineGrainedFP8HfQuantizer.create_quantized_param = original_method

    # --- Index handling for bf16 and fp8 models ---

    # Load/build bf16 index (metadata not needed -> save memory)
    bf16_index, _ = load_or_build_index(model_path, compute_metadata=False)
    # Load/build fp8 index (metadata needed for final index.json)
    fp8_index, fp8_index_path = load_or_build_index(save_path, compute_metadata=True)

    # Detect missing keys (e.g., MTP layers)
    missing_keys = find_missing_keys(bf16_index, fp8_index)

    if not missing_keys:
        logger.info("No missing keys detected between bf16 and fp8 indices.")
        return

    logger.info("Loading missing weights from bf16 checkpoint...")
    missing_weights = load_missing_weights(model_path, bf16_index, missing_keys)

    if not missing_weights:
        logger.warning("Failed to load any missing weights; aborting MTP quantization.")
        return

    quantization_config_dict: Dict[str, Any] = cfg["quantization_config"]
    quantized_results = quantize_missing_weights(missing_weights, quantization_config_dict)

    if not quantized_results:
        logger.warning("Failed to quantize missing weights; aborting.")
        return

    logger.info("Saving quantized missing weights and updating fp8 index.json...")
    save_quantized_weights(quantized_results, save_path, fp8_index, fp8_index_path)

    # Re-check missing keys after update
    remaining_missing_keys = find_missing_keys(bf16_index, fp8_index)
    remaining_count = len(remaining_missing_keys)
    logger.info("After update - missing keys: %d", remaining_count)

    if remaining_count == 0:
        logger.info("✅ All keys successfully added to fp8 model!")
    else:
        logger.warning("⚠️ %d keys are still missing: %s", remaining_count, remaining_missing_keys)

    logger.info("Missing weights quantization completed successfully.")


if __name__ == "__main__":
    main()