# FLUX model in torchtitan

## Overview
This directory contains the implementation of the [FLUX](https://github.com/black-forest-labs/flux/tree/main) model in torchtitan. In torchtitan, we showcase the pre-training process of text-to-image part of the FLUX model.

## Prerequisites
Install the required dependencies:
```bash
pip install -r requirements-flux.txt
```

## Usage
First, download the autoencoder model from HuggingFace with your own access token:
```bash
python scripts/download_hf_assets.py --repo_id black-forest-labs/FLUX.1-dev --additional_patterns ae.safetensors --hf_token <your_access_token>
```

This step will download the autoencoder model from HuggingFace and save it to the `assets/hf/FLUX.1-dev/ae.safetensors` file.

Run the following command to train the debug model on a single GPU:
```bash
NGPU=1 ./run_train.sh configs/experiments/flux_debug.yaml
```

`flux_debug.yaml` selects `trainer: {type: flux}` and carries the flux-only
`encoder:` and `inference:` sections. FLUX.1-dev and FLUX.1-schnell are arch
presets (`configs/presets/models/flux/{dev,schnell}.yaml`): to train one, copy
`flux_debug.yaml`, point its model import at the preset and adjust the training
sections. Sampling from a trained checkpoint uses the same experiment:
```bash
NGPU=8 ./torchtitan/models/flux/run_infer.sh configs/experiments/flux_debug.yaml --checkpoint.load_step 10
```


## Supported Features
- Parallelism: The model supports FSDP, HSDP, CP for training on multiple GPUs.
- Activation checkpointing: The model uses activation checkpointing to reduce memory usage during training.
- `torch.compile`: Per-block compilation for the Flux transformer (DoubleStreamBlock, SingleStreamBlock). See [torch.compile](#torchcompile) below.
- Distributed checkpointing and loading.
    - Notes on the current checkpointing implementation: To keep the model weights are sharded the same way as checkpointing, we need to shard the model weights before saving the checkpoint. This is done by checking each module at the end of evaluation, and sharding the weights of the module if it is a FSDPModule.
- CI for FLUX model. Supported periodically running integration tests on 8 GPUs, and unittests.

## torch.compile

The Flux model supports `torch.compile` for accelerating training. Compilation is applied per-block to the repeated DoubleStreamBlock and SingleStreamBlock layers in the main transformer.

To enable compilation, add the following flags:
```bash
./run_train.sh configs/experiments/flux_debug.yaml --compile.enable
```

By default, both the model and the loss function are compiled. You can control which components are compiled via `--compile.components`:
```bash
# Compile only the model (not the loss)
./run_train.sh configs/experiments/flux_debug.yaml --compile.enable --compile.components '["model"]'

# Compile only the loss
./run_train.sh configs/experiments/flux_debug.yaml --compile.enable --compile.components '["loss"]'
```

**Notes:**
- The Flux model blocks are compiled with `fullgraph=True` for maximum optimization.
- The default backend is `inductor`. You can change it with `--compile.backend <backend>`.


## TODO
- [ ] More parallelism support (Tensor Parallelism, Pipeline Parallelism, etc)
- [ ] Implement the num_flops_per_token calculation in get_nparams_and_flops() function
