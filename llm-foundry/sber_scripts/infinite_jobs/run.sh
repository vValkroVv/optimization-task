# user variables
export CODE_PATH="$(dirname "$(realpath "$0")")/../../"

if [ $# -ne 0 ]; then
   MODALITY=$1
   DEFAULT_CFG=$2
   EDITABLE_CFG=$3
   YAML_CFG=`python sber_scripts/sft/merger/config_merger.py -d $DEFAULT_CFG -e $EDITABLE_CFG --modality $MODALITY | tail -n 1`
   export YAML_CFG="$MODALITY/merged_configs/`basename $YAML_CFG`"
else
   export YAML_CFG=     #29B.yaml
fi

export NNODES=          #2
export NGPUS_PER_NODE=8

# script start
export PPORT=6001
export OPAL_PREFIX=/opt/hpcx/ompi

# call clean script
bash "$(dirname "$(realpath "$0")")/clean.sh"

HOSTLIST=/tmp/hostlist
cat /etc/volcano/mpimaster.host > $HOSTLIST
echo "">> $HOSTLIST
cat /etc/volcano/mpiworker.host >> $HOSTLIST
echo "">> $HOSTLIST

JOB_LOGS_DIR="inf-job-logs/$(date +%Y%m%d-%H%M%S)"

mpirun  -N 1 -np $NNODES \
--hostfile $HOSTLIST \
-bind-to none \
-x NNODES \
-x MASTER_ADDR=`cat $HOSTLIST|grep mpimaster-0 | tr -d "\n"` \
-x PPORT \
-x NGPUS_PER_NODE \
-x GIGAFSDP_GRAD_NAN_ORIGIN \
-x NCCL_DEBUG=INFO \
-x TORCH_CPP_LOG_LEVEL=ERROR \
-x PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
-x PYTHONPATH=$CODE_PATH:$CODE_PATH/contrib/composer:$CODE_PATH/contrib/streaming:$CODE_PATH/contrib/gigafsdp:$CODE_PATH/contrib/flash-linear-attention:$CODE_PATH/contrib/metric-sampler \
-x NCCL_IB_HCA=^mlx5_bond  \
-x PYTHONUNBUFFERED=1 \
-x LD_LIBRARY_PATH \
-x OPAL_PREFIX \
-x OMP_NUM_THREADS=1 \
-x PATH \
-mca pml ucx \
-mca plm_rsh_args "-p 2222" \
-output-filename "$JOB_LOGS_DIR:nojobid" \
-merge-stderr-to-stdout \
-tag-output \
/bin/bash -c "export NODE_RANK=\$(hostname | python3 -c \"hostname = input(); rank = 0 if hostname.endswith('mpimaster-0') else int(hostname.split('-')[-1]) + 1; print(rank, end='') \") && python -m torch.distributed.launch --nproc_per_node \$NGPUS_PER_NODE --nnodes \$NNODES --node_rank \$NODE_RANK --master_addr \$MASTER_ADDR --master_port 6001 --use_env $CODE_PATH/scripts/train/train.py  $CODE_PATH/sber_configs/$YAML_CFG use_clearml=false"
