"""
Layer-by-layer BF16 -> FP8 quantization for gigachat3.5 checkpoints.

   gigachat3.5 <-> Qwen3-Next naming is derived from
   ``llm-foundry/contrib/gigachat-utils/convertation/transformers_utils/deepseek.py``:
   ``mtp.fc -> eh_proj``, ``mtp.norm -> shared_head.norm``,
   ``mtp.pre_fc_norm_embedding -> enorm``,
   ``mtp.pre_fc_norm_hidden -> hnorm``,
   ``linear_attn.* -> self_attn.*`` (on linear-attention layers),
   ``q_norm/k_norm -> q_a_layernorm/kv_a_layernorm``.

2. Add ``modules_to_not_convert`` into the output ``config.json``.
Usage with CLI args:
    python fine_grained_fp8_quantization_layer_by_layer.py \\
        --input_checkpoint_path /home/jovyan/nvokhotnikov/model_save_folder/700B_random_no_sigmoid/random_hf_checkpoint_uniform \\
        --output_checkpoint_path /home/jovyan/nvokhotnikov/model_save_folder/700B_random_no_sigmoid_fp8 \\
        --weight_block_size 128 128 \\
        --device cuda:0

Usage with config file (same format as quantization_config.yaml):
    python fine_grained_fp8_quantization_layer_by_layer.py \\
        --config_path /path/to/quantization_config.yaml

for hybrid architecture check DEFAULT_MODULES_TO_NOT_CONVERT !
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import safetensors
import torch
from safetensors.torch import save_file as safe_save_file
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Expanded fallback skip list for gigachat3.5. Kept in sync with
# ``artifacts/quantization_config.yaml``; see the module docstring and
# ``artifacts/config_qwen_next.yaml`` for the upstream Qwen3-Next reference.
DEFAULT_MODULES_TO_NOT_CONVERT: List[str] = [
    # Embeddings and heads.
    "lm_head",
    "embed_tokens",
    "shared_head.head",
    # MoE router (Gigachat35TopkRouter uses a plain nn.Parameter, not nn.Linear).
    "mlp.gate.weight",
    # MTP input projection (Qwen3-Next `mtp.fc`).
    "eh_proj",
    # Gigachat35GatedDeltaNetWrapper internals (Qwen3-Next `linear_attn.*`).
    "self_attn.A_log",
    "self_attn.conv1d",
    "self_attn.dt_bias",
    "self_attn.in_proj_ba",
    "self_attn.norm.weight",
    # Gated RMSNorm inner projections — new in gigachat3.5 vs DeepSeekV3.
    "gate_up_projection",
    "gate_down_projection",
    # Gigachat35Attention gated-attention output gate — new in gigachat3.5.
    "self_attn.gate_proj",
    # Gigachat35MoE shared-expert sigmoid gate — new in gigachat3.5
    # (Qwen3-Next equivalent: `mlp.shared_expert_gate`).
    "shared_expert_sigmoid",
    # DSA indexer routing weights are loaded as BF16 ReplicatedLinear in SGLang.
    "indexer.weights_proj",
]

# ---------------------------------------------------------------------------
# Block-wise FP8 quantization of a single tensor
# ---------------------------------------------------------------------------

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
        quantized_param = quantized_param[tuple(slices)].contiguous()

    # Reshape scale to match block layout and invert (weight_scale_inv).
    # Keep the two block dimensions even when one of them is 1: SGLang's
    # block-FP8 loader expects e.g. (1, num_col_blocks), not (num_col_blocks,).
    scale_shape = tuple(param_value_orig_shape[:-2]) + (
        rows_quant // block_size_m,
        cols_quant // block_size_n,
    )
    scale = scale.reshape(scale_shape).reciprocal()

    return quantized_param, scale


# ---------------------------------------------------------------------------
# Shard-by-shard processing
# ---------------------------------------------------------------------------


def _resolve_device(device: str) -> str:
    """Resolve ``auto`` to cuda when available; other values pass through."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _shard_device(device: str, shard_index: int) -> str:
    """Spread shards across all GPUs when no specific GPU index is pinned."""
    if device != "cuda":
        return device
    return f"cuda:{shard_index % torch.cuda.device_count()}"


def _should_skip_quantization(
    key: str,
    tensor: torch.Tensor,
    modules_to_not_convert: Sequence[str],
) -> bool:
    """Decide whether ``key`` must stay in original precision.

    * 1D tensors (scalars / biases / single-vector RMSNorms) are always kept.
    * Tensors whose key contains any substring from ``modules_to_not_convert``
      are kept. Substring matching mirrors HuggingFace's
      ``replace_with_fp8_linear`` / ``check_quantized_param``.
    """
    if tensor.dim() == 1:
        return True
    if any(pattern in key for pattern in modules_to_not_convert):
        return True
    return False


def process_single_shard(
    input_shard_path: str,
    output_shard_path: str,
    block_size_m: int,
    block_size_n: int,
    modules_to_not_convert: Sequence[str],
    device: str = "cpu",
) -> Dict[str, str]:
    """
    Process one safetensors shard: read tensors, quantize, write output shard.

    For each tensor in the input shard:
      - If it should skip quantization -> saved as-is (original dtype)
      - Otherwise -> quantized to FP8 + weight_scale_inv saved

    Returns:
        weight_map: dict mapping tensor_name -> shard_filename for all output tensors.
    """
    shard_filename = os.path.basename(output_shard_path)
    weight_map: Dict[str, str] = {}
    output_dict: Dict[str, torch.Tensor] = {}

    logger.info("Processing shard: %s", input_shard_path)

    with safetensors.safe_open(input_shard_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        logger.info("  %d tensors in shard", len(keys))

        for key in tqdm(keys, desc=f"  {shard_filename}", leave=False):
            tensor = f.get_tensor(key)

            if _should_skip_quantization(key, tensor, modules_to_not_convert):
                output_dict[key] = tensor
                weight_map[key] = shard_filename
            else:
                if device.startswith("cuda") and tensor.device.type == "cpu":
                    tensor = tensor.to(device)

                quantized_tensor, scale = quantize_single_tensor(
                    tensor,
                    block_size_m,
                    block_size_n,
                )

                scale_key = key.rsplit(".", 1)[0] + ".weight_scale_inv"

                output_dict[key] = quantized_tensor.cpu()
                output_dict[scale_key] = scale.cpu()
                weight_map[key] = shard_filename
                weight_map[scale_key] = shard_filename

            del tensor

    logger.info("  Saving output shard: %s", output_shard_path)
    safe_save_file(output_dict, output_shard_path, metadata={"format": "pt"})

    num_output_tensors = len(output_dict)
    logger.info("  Saved %d tensors to %s", num_output_tensors, output_shard_path)

    del output_dict
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return weight_map


# ---------------------------------------------------------------------------
# Config and metadata handling
# ---------------------------------------------------------------------------


def copy_non_safetensors_files(input_path: str, output_path: str) -> None:
    """Copy everything except safetensors shards and the tensor index."""
    skip_extensions = {".safetensors"}
    skip_files = {"model.safetensors.index.json"}

    for item in os.listdir(input_path):
        src = os.path.join(input_path, item)
        if not os.path.isfile(src):
            continue
        _, ext = os.path.splitext(item)
        if ext in skip_extensions or item in skip_files:
            continue
        dst = os.path.join(output_path, item)
        shutil.copy2(src, dst)
        logger.info("  Copied: %s", item)


def update_config_with_quantization(
    config_path: str,
    weight_block_size: Tuple[int, int],
    activation_scheme: str = "dynamic",
    modules_to_not_convert: Optional[Sequence[str]] = None,
) -> None:
    """Add quantization_config to config.json.

    {
        "quantization_config": {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
            "weight_block_size": [block_size_m, block_size_n],
            "ignored_layers": ["lm_head", "embed_tokens.weight", ...]
        }
    }
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    quant_config: Dict[str, Any] = {
        "activation_scheme": activation_scheme,
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": list(weight_block_size),
    }
    if modules_to_not_convert:
        quant_config["ignored_layers"] = list(modules_to_not_convert)

    config["quantization_config"] = quant_config

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    extra = (
        f" (incl. {len(modules_to_not_convert)} skip pattern(s))"
        if modules_to_not_convert
        else ""
    )
    logger.info("  Updated config.json with quantization_config%s", extra)


def build_and_save_index(
    weight_map: Dict[str, str],
    output_path: str,
) -> None:
    """Build and save ``model.safetensors.index.json`` from ``weight_map``."""
    total_size = 0
    seen_files = set(weight_map.values())
    for filename in seen_files:
        shard_path = os.path.join(output_path, filename)
        if os.path.exists(shard_path):
            total_size += os.path.getsize(shard_path)

    index = {
        "metadata": {
            "total_size": total_size,
        },
        "weight_map": weight_map,
    }

    index_path = os.path.join(output_path, "model.safetensors.index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
        f.write("\n")

    logger.info(
        "  Saved index: %d entries, %d bytes (%.2f GB)",
        len(weight_map),
        total_size,
        total_size / (1024 ** 3),
    )


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _apply_yaml_config(args: argparse.Namespace, config_path: str) -> None:
    """Overlay values from ``config_path`` onto ``args`` (in-place)."""
    try:
        from omegaconf import OmegaConf as om
    except ImportError:
        logger.warning(
            "omegaconf not installed, cannot load config file %s. Using CLI args.",
            config_path,
        )
        return

    cfg = om.to_container(om.load(config_path), resolve=True)
    assert isinstance(cfg, dict), f"Config at {config_path} must be a mapping"

    args.input_checkpoint_path = cfg.get(
        "input_checkpoint_path", args.input_checkpoint_path
    )
    args.output_checkpoint_path = cfg.get(
        "output_checkpoint_path", args.output_checkpoint_path
    )
    args.device = cfg.get("device_map", args.device)
    args.num_workers = cfg.get("num_workers", args.num_workers)

    qcfg = cfg.get("quantization_config") or {}
    if "weight_block_size" in qcfg:
        args.weight_block_size = list(qcfg["weight_block_size"])
    if "activation_scheme" in qcfg:
        args.activation_scheme = qcfg["activation_scheme"]
    if "modules_to_not_convert" in qcfg and qcfg["modules_to_not_convert"] is not None:
        args.modules_to_not_convert = list(qcfg["modules_to_not_convert"])


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Layer-by-layer BF16 -> FP8 quantization (low memory footprint)"
    )
    parser.add_argument(
        "--input_checkpoint_path",
        type=str,
        default=None,
        help="Path to the input BF16 HuggingFace checkpoint directory",
    )
    parser.add_argument(
        "--output_checkpoint_path",
        type=str,
        default=None,
        help="Path to the output FP8 HuggingFace checkpoint directory",
    )
    parser.add_argument(
        "--weight_block_size",
        type=int,
        nargs=2,
        default=[128, 128],
        metavar=("BLOCK_M", "BLOCK_N"),
        help="Block size for FP8 quantization (default: 128 128)",
    )
    parser.add_argument(
        "--activation_scheme",
        type=str,
        default="dynamic",
        choices=["dynamic", "static"],
        help="Activation quantization scheme (default: dynamic)",
    )
    parser.add_argument(
        "--modules_to_not_convert",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Substrings of parameter names to skip quantization. "
            "Default: the expanded gigachat3.5 list (see "
            "DEFAULT_MODULES_TO_NOT_CONVERT)."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help=(
            "Device for quantization computation (default: cpu). "
            "Use cuda:0 for GPU acceleration of per-tensor quantization."
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of processes quantizing shards in parallel (default: 8)",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help=(
            "Optional path to a quantization config YAML (same format as "
            "artifacts/quantization_config.yaml). Values from the YAML override "
            "CLI args."
        ),
    )
    args = parser.parse_args()

    # Optionally load config file (overrides CLI args)
    if args.config_path is not None:
        logger.info("Loading quantization config from %s", args.config_path)
        _apply_yaml_config(args, args.config_path)

    # Set default modules_to_not_convert for Gigachat3.5 if not specified
    if args.modules_to_not_convert is None:
        args.modules_to_not_convert = list(DEFAULT_MODULES_TO_NOT_CONVERT)

    # Validate inputs
    assert args.input_checkpoint_path, (
        "input_checkpoint_path is required (CLI flag or config YAML)"
    )
    assert args.output_checkpoint_path, (
        "output_checkpoint_path is required (CLI flag or config YAML)"
    )
    assert os.path.isdir(args.input_checkpoint_path), (
        f"Input path is not a directory: {args.input_checkpoint_path}"
    )
    assert args.weight_block_size[0] > 0 and args.weight_block_size[1] > 0, (
        "weight_block_size must be positive"
    )

    args.device = _resolve_device(args.device)
    assert args.num_workers > 0, "num_workers must be positive"

    block_size_m, block_size_n = args.weight_block_size

    os.makedirs(args.output_checkpoint_path, exist_ok=True)
    logger.info("Output directory: %s", args.output_checkpoint_path)
    logger.info(
        "Skip patterns (%d): %s",
        len(args.modules_to_not_convert),
        args.modules_to_not_convert,
    )

    # ------------------------------------------------------------------
    # Step 1: Copy non-safetensors files (config, tokenizer, modeling code)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 1: Copying non-safetensors files")
    logger.info("=" * 60)
    copy_non_safetensors_files(args.input_checkpoint_path, args.output_checkpoint_path)

    # ------------------------------------------------------------------
    # Step 2: Update config.json with quantization_config
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 2: Updating config.json with quantization_config")
    logger.info("=" * 60)
    config_json_path = os.path.join(args.output_checkpoint_path, "config.json")
    if os.path.exists(config_json_path):
        update_config_with_quantization(
            config_json_path,
            weight_block_size=(block_size_m, block_size_n),
            activation_scheme=args.activation_scheme,
            modules_to_not_convert=args.modules_to_not_convert,
        )
    else:
        logger.warning("config.json not found in input, skipping quantization_config update")

    # ------------------------------------------------------------------
    # Step 3: Process each safetensors shard one at a time
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 3: Processing safetensors shards (layer-by-layer)")
    logger.info("=" * 60)
    logger.info("  Block size: [%d, %d]", block_size_m, block_size_n)
    logger.info("  Device: %s", args.device)

    input_shards = sorted(
        f for f in os.listdir(args.input_checkpoint_path)
        if f.endswith(".safetensors")
    )

    if not input_shards:
        logger.error("No .safetensors files found in %s", args.input_checkpoint_path)
        sys.exit(1)

    logger.info(
        "Found %d input shard(s), processing with %d worker(s)",
        len(input_shards),
        args.num_workers,
    )

    combined_weight_map: Dict[str, str] = {}

    # ponytail: each worker holds one full output shard in RAM — with huge
    # shards lower num_workers if the node runs out of memory
    mp_context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=args.num_workers, mp_context=mp_context
    ) as pool:
        futures = [
            pool.submit(
                process_single_shard,
                os.path.join(args.input_checkpoint_path, shard_name),
                os.path.join(args.output_checkpoint_path, shard_name),
                block_size_m,
                block_size_n,
                args.modules_to_not_convert,
                device=_shard_device(args.device, shard_index),
            )
            for shard_index, shard_name in enumerate(input_shards)
        ]
        for future in futures:
            combined_weight_map.update(future.result())

    # ------------------------------------------------------------------
    # Step 4: Build and save index
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 4: Building and saving model.safetensors.index.json")
    logger.info("=" * 60)
    build_and_save_index(combined_weight_map, args.output_checkpoint_path)

    logger.info("=" * 60)
    logger.info("FP8 quantization complete!")
    logger.info("Output: %s", args.output_checkpoint_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
