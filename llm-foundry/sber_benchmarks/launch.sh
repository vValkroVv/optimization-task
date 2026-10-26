ppath=/home/jovyan/dvmazur/llm-foundry-mp-torch
export PYTHONPATH=$ppath/:$ppath/contrib/composer:$ppath/contrib/streaming:$ppath/contrib/gigafsdp
nproc_per_node=2
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=$PYTHONPATH NNODES=1 NODE_RANK=0 LOCAL_RANK=0 torchrun --nproc_per_node $nproc_per_node main.py
