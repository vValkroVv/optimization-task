# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import HuggingFaceStorageReader
from torchtitan.components.checkpoint import ModelWrapper, warn_unmapped_hf_entries
from torchtitan.config.build import load_model_spec


@torch.inference_mode()
def convert_from_hf(input_dir, output_dir, config_file):
    # Plain model, no converter (e.g. LoRA) wrappers, to allocate the state dict.
    model_spec = load_model_spec(config_file, apply_converters=False)
    model_config = model_spec.model

    with torch.device("cpu"):
        model = model_config.build()
    model = ModelWrapper(model)

    sd_adapter = model_spec.state_dict_adapter(model_config, None)
    assert sd_adapter is not None, (
        "trying to convert checkpoint from HF to DCP safetensors format, but sd_adapter is not provided."
    )
    # get state dict in tt format with allocated memory
    state_dict = model._get_state_dict()
    # convert empty state dict to hf format so that hf weights can be loaded into it
    hf_state_dict = sd_adapter.to_hf(state_dict)
    dcp.load(
        hf_state_dict,
        storage_reader=HuggingFaceStorageReader(path=input_dir),
    )
    # convert state dict format back hf->tt and save
    loaded = sd_adapter.from_hf(hf_state_dict)
    warn_unmapped_hf_entries(
        state_dict,
        loaded,
        outcome="are not saved to the DCP, so a model with them cannot load it "
        "(missing keys); load the HF weights with checkpoint.initial_load_in_hf "
        "instead, which keeps their initial values",
    )
    dcp.save(
        loaded,
        checkpoint_id=output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HF checkpoint to DCP format.")
    parser.add_argument(
        "input_dir", type=Path, help="Input directory with HF checkpoint"
    )
    parser.add_argument("output_dir", type=Path, help="Output directory for DCP.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/llama3_8b.yaml",
        help="Experiment YAML the checkpoint was trained with. Prefer the run's "
        "<dump_folder>/config/resolved_config.yaml so the rebuilt model matches "
        "the checkpoint exactly.",
    )
    args = parser.parse_args()

    convert_from_hf(args.input_dir, args.output_dir, args.config)
