# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import math
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import HuggingFaceStorageWriter
from torchtitan.components.checkpoint import ModelWrapper
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import TORCH_DTYPE_MAP
from torchtitan.components.checkpoint_conversion import (
    cast_export_state_dict,
    load_export_model_spec,
)


@torch.inference_mode()
def convert_to_hf(input_dir, output_dir, config_file, hf_assets_path, export_dtype):
    # Plain model, no converter (e.g. LoRA) wrappers: only the shapes matter.
    model_spec = load_export_model_spec(config_file)
    model_config = model_spec.model

    with torch.device("cpu"):
        model = model_config.build()
    model = ModelWrapper(model)

    sd_adapter = model_spec.state_dict_adapter(model_config, hf_assets_path)
    assert sd_adapter is not None, (
        "trying to convert checkpoint from DCP to HF safetensors format, but sd_adapter is not provided."
    )

    # allocate state dict memory with empty weights to load checkpoint
    state_dict = model._get_state_dict()
    # Representability needs only the config and shapes: fail before the load.
    # The checkpoint's step is unknown here: assume residual-scale warmups ended.
    if issues := model_config.validate_hf_export(step=math.inf):
        raise ValueError("; ".join(str(issue) for issue in issues))
    # Past that, only an unsaturated residual scale can still be reported.
    for issue in model_config.validate_hf_export(step=None):
        logger.warning(
            f"{issue}. The checkpoint step is unknown offline; if it "
            "was saved mid-warmup the export differs."
        )
    sd_adapter.check_hf_export(state_dict)
    dcp.load(
        state_dict,
        checkpoint_id=input_dir,
    )

    # convert state dict tt->hf
    hf_state_dict = sd_adapter.to_hf(state_dict)

    storage_writer = HuggingFaceStorageWriter(
        path=output_dir,
        save_distributed=True,
        fqn_to_index_mapping=sd_adapter.fqn_to_index_mapping,
        enable_consolidation=True,
        thread_count_consolidation=5,
    )

    # map and apply export dtype if needed
    target_dtype = TORCH_DTYPE_MAP[export_dtype]
    hf_state_dict = cast_export_state_dict(hf_state_dict, target_dtype)

    dcp.save(
        hf_state_dict,
        storage_writer=storage_writer,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DCP weights to HF format.")
    parser.add_argument(
        "input_dir", type=Path, help="Input directory with DCP weights."
    )
    parser.add_argument(
        "output_dir", type=Path, help="Output directory for HF checkpoint."
    )
    parser.add_argument(
        "--hf_assets_path",
        type=Path,
        help="Path to HF assets directory. This is used to get the model.safetensors.index.json mapping",
        default="./assets/hf/Llama-3.1-8B",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/llama3_8b.yaml",
        help="Experiment YAML the checkpoint was trained with. Prefer the run's "
        "<dump_folder>/config/resolved_config.yaml so the rebuilt model matches "
        "the checkpoint exactly.",
    )
    parser.add_argument(
        "--export_dtype",
        type=str,
        nargs="?",
        choices=["float16", "bfloat16", "float32"],
        default="float32",
        help="Export dtype for HF checkpoint (default: float32)",
    )
    args = parser.parse_args()

    convert_to_hf(
        args.input_dir,
        args.output_dir,
        args.config,
        args.hf_assets_path,
        args.export_dtype,
    )
