# ClickHouse cloud deploy

Applies `metric_sampler/cloud/schema.sql` and `serving/*.sql`, the `ingest` sidecar, the two readonly
users and the ingest user, and the summary loops behind the Grafana dashboards.
Security must land before real data does: do steps 1-3 before pointing the collector at this bucket.

## 1. Named collection

1. Read the S3 access key and secret from the host's credential file (not from this repo).
2. Copy `config.d/named_collections.xml.template` to
   `/etc/clickhouse-server/config.d/named_collections.xml` and substitute `__S3_BUCKET_URL__` (the
   bucket's https address), `__S3_ACCESS_KEY__` and `__S3_SECRET_KEY__` in place.
3. `chmod 600` the file; it now holds a live secret. Never commit the filled-in file.

`schema.sql` and `ingest.sql` reference the collection as `obs_ms`, so they never see the key or the secret.

## 2. Users

1. Pick passwords for `grafana_reader`, `human_reader` and `ingest_writer`.
2. `echo -n '<password>' | sha256sum` for each, and substitute the three
   `__*_PASSWORD_SHA256__` placeholders in `users.d/readers.xml.template`. Never commit the filled-in file.
3. Copy the filled-in file to `/etc/clickhouse-server/users.d/metric_sampler_users.xml`. The two readers are
   readonly and accept connections from any address, since the dashboards and the CLI are reached from outside the
   host; `ingest_writer` is the ingest loop's user and accepts only the host and its containers.
4. Copy `config.d/metric_sampler_concurrency.xml` to `/etc/clickhouse-server/config.d/`: with the
   default 2 query threads per core the ingest and the summary loops held every slot and dashboard
   queries ran on one thread. It takes effect on `SYSTEM RELOAD CONFIG`.
5. Copy `users.d/metric_sampler_default_cap.xml` too: it caps the memory of all of the default user's
   queries, which are the summary loops of step 6, so a catch-up cannot starve ingest and Grafana.
6. Copy `config.d/metric_sampler_system_logs.xml` too: the system logs keep days instead of forever and
   the server log stops at `information`. It takes effect on restart, which renames the existing log
   tables whose settings changed to `<name>_0`; drop those once nothing in them is needed.
7. Copy `config.d/metric_sampler_memory.xml` too: the server keeps to about a third of the host's memory
   and merges to a quarter of that. The server limit takes effect on `SYSTEM RELOAD CONFIG`, the merge
   limit on restart.

## 3. Restart and verify

```
systemctl restart clickhouse-server   # or the container equivalent
clickhouse-client --user grafana_reader --password '<password>' --query "SELECT 1"
```

Confirm `grafana_reader` cannot write: `INSERT INTO metric_sampler.processed_objects ...`
must fail with a readonly error.

## 4. Schema

```
clickhouse-client --multiquery < ../../metric_sampler/cloud/schema.sql
cat ../../metric_sampler/cloud/serving/*.sql | clickhouse-client --multiquery
```

Safe to re-run: it drops the retired v1 objects (`raw`, `runs`, `loss_per_step`, `expert_load`)
and recreates the v3 database with `CREATE TABLE IF NOT EXISTS` / `CREATE OR REPLACE VIEW`.
The skip indexes `step_minmax` and `metric_set` cover only parts written after they were added;
for older data run `ALTER TABLE metric_sampler.<table> MATERIALIZE INDEX <index>` once per raw table.

## 5. Ingest sidecar

`docker-compose.yml` is a fragment, not a standalone stack: merge the `ingest` service into the compose file
that runs the ClickHouse server and keep its image tag equal to the server's version. Set
`METRIC_SAMPLER_CLICKHOUSE_HOST` to the compose service name of the server, then:

```
docker compose up -d ingest
docker compose logs -f ingest
```

The loop runs `metric_sampler/cloud/ingest.sql` every 30s. Check
`SELECT * FROM metric_sampler.pipeline_freshness` after the first batch lands to
confirm data is flowing.

Health summaries are refreshed by the health loop of the summaries service (step 6 starts it), one 10-step
window at a time, after all three raw families have matching processed-object counts.
New batches take priority; completed older batches are backfilled automatically without
reloading their raw data. Keep one health worker: refresh reads the canonical raw
`FINAL` result and publishes its generation only after the complete snapshot is stored.
Retries and overlapping batches replace complete per-step values, never add totals.
The refresh runs apart from the ingest pass, which it would slow on a large run.

Apply `schema.sql` before the updated `ingest.sql`: it marks family objects processed from
`ingested_objects`, which the insert triggers of `schema.sql` fill, and without them every new
object would be ingested again on each pass. Before switching the Health
dashboard, check `health_pending_windows` and the `health_max_step` /
`health_refreshed_at` columns in `pipeline_freshness`; historical backfill can lag.
`health_per_step` remains the direct raw audit view. No raw observations are removed.

## 6. Summaries

The dashboards read the serving layer's summaries once they cover a run, and the raw tables until then.
The package `metric_sampler.cloud.summaries` keeps them filled. It imports only the Python standard library (3.10 or
newer), so the host needs nothing but `python3`, `nsenter` and `curl`, and the service has to run as a user that can
read the ClickHouse server process's environment. Copy `metric_sampler/cloud/summaries/` with the two empty
`__init__.py` files above it (`metric_sampler/__init__.py`, `metric_sampler/cloud/__init__.py`) to a directory on the
ClickHouse host and create `logs/` there. Run the service from that directory, which is also where its logs go:

```
setsid -f python3 -m metric_sampler.cloud.summaries run >> logs/service.log 2>&1 < /dev/null
```

`run` starts every loop in one process, one thread per loop, and writes one line per loop it starts to
`logs/service.log`. A loop that raises is logged there in one line and started again after a minute; SIGTERM and
SIGINT stop the service. Each loop logs to its own file in `logs/`: `<summary>.log` for every summary loop,
`hist_latest.log`, `top_cards.log`, `health.log`, `io_guard.log` and `cold.log`. Nothing restarts the service after a
reboot.

Each summary loop (one per entry of `catalog.py`) refills one summary table chunk by chunk behind the ingest and
records the last final step in `summary_progress`, never beyond the run's `complete_steps` frontier; a new summary is
filled from its progress, which starts at nothing. A chunk beyond the frontier is recomputed only after rows were
ingested. The last computation of a run is kept in `/tmp/summaries-<summary>.<exp>.<run>.computed`; delete that file
to force a recompute of the run's chunk (after emptying the summary table, say). At most 8 chunk inserts of all the
loops run at once, each holding one of the locks `/tmp/summaries-slot-<n>.lock`. Both the state files and the slot
locks are the ones the loops of the earlier shell deployment used, so a service started in its place continues where
they stopped.
The latest-histogram loop keeps the newest complete step's activation histograms; the top cards loop keeps
node-rank's "Top 20 cards" rows (largest |robust z| per card) for every run its summaries cover.
A summary derived from another one (`after` in `catalog.py`) computes only the steps that one has finished and is
filled and recomputed after it: the activation statistics per rank, per microbatch and of all ranks sum the finished
rows of `activation_rank_mb`.
The IO guard stops all merges while the data disk falls behind (over 40 GB of dirty pages or over 50% of the last
minute with every task waiting for IO) and starts them once it has caught up, after 30 minutes at the latest; it
starts merges when it starts, so stop merges by hand only while it is not running.
The service reaches ClickHouse over HTTP from inside the server process's network namespace rather than through
`docker exec`, which hangs host-wide while the data disk is saturated. The password goes from the server process's
environment through a pipe into curl's config, so it is never printed, put on a command line or written to a file.

Two commands share the loops' state and run apart from the service, from the same directory:

```
python3 -m metric_sampler.cloud.summaries recompute <summary> <exp> <run> <from> <to> [--chunk-steps 10] [--slots 8]
python3 -m metric_sampler.cloud.summaries move-cold <exp> <run> <table>... [--parallel 6] [--min-age-hours 0] \
    [--min-bytes 0] [--max-io-full 50]
```

`recompute` recomputes the steps `<from>`..`<to>` of one summary for one run in place, for steps that were summarized
before all their data arrived: it takes its chunks through the same slots, computes them exactly as the summary's loop
does, and leaves `summary_progress` as it is, so the dashboards keep their routes while the rows are replaced. Recompute
a summary that is derived from another one after that one. `move-cold` is described in step 8. Both take `--timeout`
(before the command) for the seconds a statement may take, 1900 by default.

## 7. Grafana

Provision `deploy/grafana/provisioning` and `deploy/grafana/dashboards`. The datasource takes the ClickHouse address
from `METRIC_SAMPLER_CLICKHOUSE_HOST` in Grafana's environment and sets `queryTimeout` to 240 s, so Grafana's own
data proxy timeout must be longer: `GF_DATAPROXY_TIMEOUT=300` (or `[dataproxy] timeout = 300` in grafana.ini). So that
a dashboard keeps the experiment and run picked last when one comes back to it, enable
`GF_FEATURE_TOGGLES_PRESERVEDASHBOARDSTATEWHENNAVIGATING=true` (or
`preserveDashboardStateWhenNavigating = true` under `[feature_toggles]`); the dashboards' link bar
carries the selection from one dashboard to the next. Prefer the environment variables: a grafana.ini inside the
container is lost when the container is recreated.

## 8. Cold tier

Finished runs move to the S3 bucket (`metric-sampler/clickhouse-cold/`) and stay queryable through a
300 GiB local cache, so the local disk only has to hold the runs in progress. Parts on the cold volume are not
merged: a merge there would download and upload every part it reads.

1. Fill `config.d/metric_sampler_storage.xml.template` as in step 1 into
   `/etc/clickhouse-server/config.d/metric_sampler_storage.xml`, `chmod 600`, and restart:
   `SYSTEM RELOAD CONFIG` adds a new policy but cannot rename a volume of a policy that already exists.
2. Once per raw table (`router_vectors`, `expert_counts`, `scalars`) and per summary partitioned by run
   (`SUMMARY_TABLES` of `metric_sampler/cloud/summaries/cold_tier.py`):
   `ALTER TABLE metric_sampler.<table> MODIFY SETTING storage_policy = 'tiered'`.
3. From the directory of the summaries service, for a run that no longer receives data:

```
setsid -f python3 -m metric_sampler.cloud.summaries move-cold <exp> <run> scalars expert_counts router_vectors \
    >> logs/move_cold.log 2>&1 < /dev/null
```

It moves 6 parts at a time (`--parallel`), pauses while more than 50% of the last 10 seconds had every task waiting for
IO (`--max-io-full`), reads every part it moved back from the cold volume (which checks each compressed block against
its checksum) and logs the run's rows per disk before and after. Only parts written at least `--min-age-hours` ago and
of at least `--min-bytes` move, every part by default. A `move failed` or `check failed` line names the part; a rerun
moves whatever is still on the local disk.

4. The service of step 6 runs the cold loop, which does the same every hour for every run whose raw
   parts have been on the local disk for 24 hours (`HOT_HOURS`). A run still being written moves only parts
   merged to at least 1 GiB, since parts on the cold volume are not merged again; a run idle for 24 hours moves
   all of them, and after a week (`SUMMARY_HOT_HOURS`) also its rows of the summaries partitioned by run, which the
   dashboards read.
5. A run whose raw rows the data disk cannot write fast enough (every merge writes the rows again) can have its parts
   written to the cold volume directly: switch the raw tables to the `tiered_raw` policy
   (`ALTER TABLE ... MODIFY SETTING storage_policy = 'tiered_raw',
   max_avg_part_size_for_too_many_parts = 268435456`). Inserts of 4 GiB blocks then go to the cold volume and the
   summary loops' first read of them fills the local cache, which makes a first read slower. Parts on the cold volume
   are not merged, so their size stays below the 1 GiB above which ClickHouse would not count them for its
   too-many-parts checks; hence the lower threshold.
