**The Qwen3 model is still under development.**


## Available features
#### Dense Model
- Qwen3 dense model:
    - supports FSDP/HSDP, TP, CP, DDP.
    - Supports AC, torch.compile.
- Qwen3 MoE model:
    - Supports FSDP/HSDP, TP, CP, DDP, EP.
    - Supports AC, torch.compile.
    - MoE models use Token Choice routing, which is using auxiluary-loss-free load balancing algorithm.


## Training

```bash
./run_train.sh configs/experiments/qwen3_debug.yaml        # dense debug model
./run_train.sh configs/experiments/qwen3_moe_debug.yaml    # MoE debug model
```

Arch presets exist for every released size (`configs/presets/models/qwen3/`: 0.6B,
1.7B, 4B, 8B, 14B, 32B, 30B-A3B, 235B-A22B); the tested runs are the `qwen3_*` and
`sft_qwen3_8b_math.yaml` experiments under `configs/experiments/`. Compose a run for
another size by importing its preset (see [docs/config.md](/docs/config.md)).

## Download Qwen3 tokenizer
```python scripts/download_hf_assets.py --repo_id <hf_repo_name> --assets tokenizer```

eg, for Qwen3 0.6B model, the HF repo name is `Qwen/Qwen3-0.6B`. For 1.7B model, the HF repo name is `Qwen/Qwen3-1.7B`.


## To be added
- Testing
    - Learning rate verifying: verify learning rate and schedule with real training jobs (eg, 3k stps), or find official references.
    - The model should be tested against established performance benchmarks
