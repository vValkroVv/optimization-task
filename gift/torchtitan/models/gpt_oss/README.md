# gpt-oss Model in torchtitan

## Quick Start
```bash
./run_train.sh configs/experiments/gpt_oss_debug.yaml
```

Model presets for the released sizes: `configs/presets/models/gpt_oss/{20b,120b}.yaml`;
import one from your own experiment (see [docs/config.md](/docs/config.md)).

## Supported Features
- FSDP/HSDP, TP, EP, ETP
- Grouped matrix multiplication for efficient computation


## TODO
1. More parallelism support: CP, PP
