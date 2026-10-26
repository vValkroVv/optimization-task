#!/bin/bash
#SBATCH --job-name=metric-sampler-collector-watch
#SBATCH --partition=transfer
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --requeue
#SBATCH --dependency=singleton
set -u

RESTART_SECONDS=60

# This loop never ends by itself, so the successor queued for its failure only starts if the job is killed;
# scancel --name removes both.
sbatch --dependency=afternotok:"$SLURM_JOB_ID" "$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/^ *Command=//p')"

until python3 -m metric_sampler.island.watch \
    --raw-root "$METRIC_SAMPLER_RAW_ROOT" \
    --job-name metric-sampler-collector \
    --sbatch "$METRIC_SAMPLER_COLLECTOR_SBATCH" \
    --dumps
do
    echo "$(date -u +%FT%TZ) watcher exited with $?, restarting in $RESTART_SECONDS s" >&2
    sleep "$RESTART_SECONDS"
done
