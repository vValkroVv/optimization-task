#!/bin/bash
#SBATCH --job-name=metric-sampler-collector
#SBATCH --partition=transfer
#SBATCH --requeue
#SBATCH --dependency=singleton
set -u

COLLECTED_RETENTION_HOURS=24
LOOP_SECONDS=60
IDLE_EXIT_MINUTES=30
RESTART_SECONDS=60

# The watcher submits this job when windows wait, and again after it failed, so the job queues no successor:
# Slurm keeps a job whose dependency can never be satisfied pending, and a pending successor would stop the
# watcher from starting the collector after this job ends idle.
# Change METRIC_SAMPLER_COLLECTOR_SHARDS only after each old shard ran once with --loop-seconds 0, so that no
# batch journal is pending.
collect_shard() {
    local shard_index=$1
    # A failed cycle ends the process; collection is idempotent, so the loop restarts it.
    until python3 -m metric_sampler.island.cli \
        --raw-root "$METRIC_SAMPLER_RAW_ROOT" \
        --rclone-conf "$METRIC_SAMPLER_RCLONE_CONF" \
        --remote-section "$METRIC_SAMPLER_S3_SECTION" \
        --endpoint "$METRIC_SAMPLER_S3_ENDPOINT" \
        --bucket "$METRIC_SAMPLER_BUCKET" \
        --prefix metric-sampler/v3 \
        --dump-remote-section "$METRIC_SAMPLER_DUMP_S3_SECTION" \
        --dump-bucket "$METRIC_SAMPLER_DUMP_BUCKET" \
        --dump-prefix "$METRIC_SAMPLER_DUMP_PREFIX" \
        --chdb-config "$METRIC_SAMPLER_CHDB_CONFIG" \
        --collected-retention-hours "$COLLECTED_RETENTION_HOURS" \
        --scratch-dir /tmp/metric-sampler-collector \
        --loop-seconds "$LOOP_SECONDS" \
        --idle-exit-minutes "$IDLE_EXIT_MINUTES" \
        --shard-index "$shard_index" \
        --shard-count "$METRIC_SAMPLER_COLLECTOR_SHARDS"
    do
        echo "$(date -u +%FT%TZ) collector shard $shard_index exited with $?, restarting in $RESTART_SECONDS s" >&2
        sleep "$RESTART_SECONDS"
    done
}

for ((shard_index = 0; shard_index < METRIC_SAMPLER_COLLECTOR_SHARDS; shard_index++)); do
    collect_shard "$shard_index" &
done
wait
