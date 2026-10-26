# user variables
export NNODES=${NNODES:-1}      #2

export CODE_PATH="$(dirname "$(realpath "$0")")/../../"

echo "CODE_PATH=$CODE_PATH"
export PYTHONPATH=$CODE_PATH:$CODE_PATH/contrib/composer:$CODE_PATH/contrib/streaming:$CODE_PATH/contrib/gigafsdp 

# script start
export OPAL_PREFIX=/opt/hpcx/ompi
HOSTLIST=/tmp/hostlist
cat /etc/volcano/mpimaster.host > $HOSTLIST
echo "">> $HOSTLIST
cat /etc/volcano/mpiworker.host >> $HOSTLIST
echo "">> $HOSTLIST

mpirun --hostfile $HOSTLIST \
-N 1 -np $NNODES \
-bind-to none \
-x OPAL_PREFIX \
-x PYTHONPATH \
-mca plm_rsh_args "-p 2222" \
/bin/bash -c "
if [ -f /home/user/conda/bin/activate ]; then
  source /home/user/conda/bin/activate ;
fi ;
python -c 'import streaming; streaming.base.util.clean_stale_shared_memory()' || true ;
pkill -9 -f '[s]cripts/train/train.py' || true ;
pkill -9 -f '[t]orch.distributed.launch' || true ;
pkill -9 -f '[t]orchrun' || true
"
