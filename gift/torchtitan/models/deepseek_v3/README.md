## Download Tokenizer

```bash
# DeepSeek 671B tokenizer (automatically downloads tokenizer.json and tokenizer_config.json)
python scripts/download_hf_assets.py --repo_id deepseek-ai/DeepSeek-V3.1-Base --assets tokenizer
```

```bash
# DeepSeek 16B tokenizer:
python scripts/download_hf_assets.py --repo_id deepseek-ai/deepseek-moe-16b-base --assets tokenizer
```

> **Note:** We are reusing the tokenizer from deepseek-ai/deepseek-moe-16b-base to help users test and run the 16B model. This is not the official tokenizer for the DeepSeek-V3-16B model. The DeepSeek-V3 model has a different architecture from the deepseek-moe models (different attention implementation, MoE router implementation, etc.), making it not feasible to load deepseek-moe-16b model weights into DeepSeek-V3-16B.


## Training

```bash
# Quick debug run with the small model on the bundled corpus
./run_train.sh configs/experiments/deepseek_v3_debug.yaml

# FlexAttention / HybridEP dispatcher variants
./run_train.sh configs/experiments/deepseek_v3_debug_flex_attn.yaml
./run_train.sh configs/experiments/deepseek_v3_debug_hybridep.yaml
```

The full sizes are model presets: `configs/presets/models/deepseek_v3/{16b,236b,671b}.yaml`
(16B is adapted from the older https://huggingface.co/deepseek-ai/deepseek-moe-16b-base).
To train one, write an experiment that imports the preset and adds the training
sections, save it as `configs/experiments/deepseek_v3_16b.yaml` (this example
is not a bundled experiment), pin it (see [docs/config.md](/docs/config.md#golden-snapshots)) and launch it:

```yaml
# configs/experiments/deepseek_v3_16b.yaml
config_version: 1
imports:
  - ../presets/models/deepseek_v3/16b.yaml
hf_assets_path: ./assets/hf/deepseek-moe-16b-base
loss: {type: chunked_cross_entropy}
dataloader: {type: hf_text, dataset: c4}
training: {local_batch_size: 4, seq_len: 4096, steps: 1000}
parallelism: {expert_parallel_degree: 8}
```

```bash
NGPU=8 ./run_train.sh configs/experiments/deepseek_v3_16b.yaml
```


## HuggingFace -> DCP Checkpoint Conversion

We implemented StateDictAdapter to perform HuggingFace safetensor to DCP format conversion. Currently, we only support conversion from HF checkpoints to DCP checkpoints offline (using CPU plain tensor).

Run the offline conversion script; `--config` takes the experiment YAML the run trains with (or, as here, a bare model preset):
```bash
python scripts/checkpoint_conversion/convert_from_hf.py <hf_checkpoints_dir> <dcp_output_dir> --config configs/presets/models/deepseek_v3/671b.yaml
```
