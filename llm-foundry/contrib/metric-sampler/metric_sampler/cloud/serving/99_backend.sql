-- The experiment and run variables of the dashboards open on the pair each viewer chose last: every dashboard's
-- hidden `remember` variable runs a query whose log_comment names the viewer, the experiment and the run, and this
-- view reads those comments back. The dashboard user may not read the query log itself, hence SQL SECURITY DEFINER.
CREATE OR REPLACE VIEW metric_sampler.viewer_selection DEFINER = CURRENT_USER SQL SECURITY DEFINER AS
SELECT splitByChar(' ', log_comment)[2] AS viewer, splitByChar(' ', log_comment)[3] AS exp, splitByChar(' ', log_comment)[4] AS run, event_time
FROM system.query_log
WHERE event_date >= today() - 30 AND type = 'QueryFinish' AND startsWith(log_comment, 'ms-selection ');

-- The Backend dashboard reads the server's disk, CPU, part and error figures through the views below; the dashboard
-- user may not read the system tables, hence SQL SECURITY DEFINER.
CREATE OR REPLACE VIEW metric_sampler.backend_disks DEFINER = CURRENT_USER SQL SECURITY DEFINER AS
SELECT
    (SELECT value FROM system.asynchronous_metrics WHERE metric = 'FilesystemMainPathAvailableBytes') AS disk_free,
    (SELECT sum(bytes_on_disk) FROM system.parts WHERE active AND (disk_name = 'default')) AS local_bytes,
    (SELECT sum(bytes_on_disk) FROM system.parts WHERE active AND (disk_name = 'obs_cold_cached')) AS cold_bytes,
    (SELECT count() FROM system.merges) AS merges;

CREATE OR REPLACE VIEW metric_sampler.backend_host_metrics DEFINER = CURRENT_USER SQL SECURITY DEFINER AS
SELECT event_time, metric, value
FROM system.asynchronous_metric_log
WHERE metric IN ('BlockWriteBytes_vdc', 'BlockReadBytes_vdc', 'BlockActiveTime_vdc', 'OSIOWaitTimeNormalized', 'OSUserTimeNormalized', 'OSSystemTimeNormalized', 'MemoryResident', 'OSMemoryAvailable');

CREATE OR REPLACE VIEW metric_sampler.backend_part_writes DEFINER = CURRENT_USER SQL SECURITY DEFINER AS
SELECT event_time, event_type, size_in_bytes
FROM system.part_log
WHERE event_type IN ('NewPart', 'MergeParts');

CREATE OR REPLACE VIEW metric_sampler.backend_table_sizes DEFINER = CURRENT_USER SQL SECURITY DEFINER AS
SELECT database, table, disk_name AS disk, count() AS parts, sum(rows) AS rows, sum(bytes_on_disk) AS bytes
FROM system.parts
WHERE active
GROUP BY database, table, disk;

CREATE OR REPLACE VIEW metric_sampler.backend_errors DEFINER = CURRENT_USER SQL SECURITY DEFINER AS
SELECT user, replaceRegexpAll(substring(exception, 1, 160), '[[:space:]]+', ' ') AS error, count() AS times, max(event_time) AS last
FROM system.query_log
WHERE (event_time >= (now() - toIntervalHour(6))) AND (exception != '') AND (user IN ('ingest_writer', 'default'))
GROUP BY user, error;
