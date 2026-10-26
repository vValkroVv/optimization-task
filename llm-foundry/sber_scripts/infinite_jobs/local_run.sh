export PYTHONPATH=$(pwd)/:$(pwd)/contrib/composer:$(pwd)/contrib/streaming:$(pwd)/contrib/gigafsdp:$(pwd)/contrib/metric-sampler

python -c 'import streaming; streaming.base.util.clean_stale_shared_memory()'
pkill -9 -f python

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=$PYTHONPATH NNODES=1 NODE_RANK=0 torchrun --nproc_per_node 8 scripts/train/train.py  $(pwd)/sber_configs/default_gigar_config.yaml use_clearml=false
