#### Download Llama 4 tokenizer
```bash
# Llama 4 tokenizer.model
python scripts/download_hf_assets.py --assets tokenizer --repo_id meta-llama/Llama-4-Scout-17B-16E --hf_token=...
```

#### Training
```bash
./run_train.sh configs/experiments/llama4_debug.yaml       # debug model on the bundled corpus; CI adds EP
./run_train.sh configs/experiments/llama4_debug_fp8.yaml   # fp8 grouped-GEMM experts over DeepEP
```

Model presets for the released sizes: `configs/presets/models/llama4/{17bx16e,17bx128e}.yaml`;
import one from your own experiment (see [docs/config.md](/docs/config.md)).
