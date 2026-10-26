from __future__ import annotations

import argparse
import faulthandler
import sys
import time
from pathlib import Path
from typing import IO

from metric_sampler.contract.raw_layout import RawRoot
from metric_sampler.island.collector import Collector, Session, Shard
from metric_sampler.island.s3_target import S3Credentials, S3Target

_SECONDS_PER_MINUTE = 60
_SECONDS_PER_HOUR = 60 * _SECONDS_PER_MINUTE
# A read from the S3 endpoint was seen waiting with no timeout on a connection the endpoint had closed; a cycle
# still running after this long is stuck, and the batch journal makes a restart safe.
_CYCLE_TIMEOUT_S = 30 * _SECONDS_PER_MINUTE

# Parquet buffers stay bounded while a shared collector catches up on a large backlog, with enough threads to encode
# faster than the S3 endpoint takes the uploads. Attempt metadata is uploaded as a SQL literal, and a large model's
# monitor_parameters.json is megabytes.
_CHDB_SETTINGS = {
    "max_threads": 4,
    "max_parsing_threads": 4,
    "max_memory_usage": 12 * 1024**3,
    "max_query_size": 64 * 1024**2,
    "input_format_parquet_max_block_size": 8192,
    "output_format_parquet_row_group_size": 1024**2,
    "output_format_parquet_row_group_size_bytes": 64 * 1024**2,
    "output_format_parquet_parallel_encoding": 1,
}
SESSION_SETTINGS = ", ".join(f"{name}={value}" for name, value in _CHDB_SETTINGS.items())


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch complete node aggregator windows from NFS into S3 as Parquet."
    )
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument(
        "--rclone-conf", type=Path, default=Path("~/.config/rclone/rclone.conf").expanduser()
    )
    parser.add_argument("--remote-section", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", default="metric-sampler/v3")
    parser.add_argument("--dump-remote-section", default=None)
    parser.add_argument("--dump-bucket", default=None)
    parser.add_argument("--dump-prefix", default=None)
    parser.add_argument("--chdb-config", type=Path, default=None)
    parser.add_argument("--loop-seconds", type=int, default=0)
    # A collector started on demand by watch.py exits once the raw root, every shard's share of it, has held no
    # complete window for this many minutes. 0 loops forever.
    parser.add_argument("--idle-exit-minutes", type=float, default=0)
    parser.add_argument("--collected-retention-hours", type=float, default=24)
    # The merged expert_counts and router_vectors of a batch go to a local file as large as the batch before they
    # are uploaded.
    parser.add_argument(
        "--scratch-dir", type=Path, default=None,
        help="directory of the merged files, on local disk (default: the system temporary directory, TMPDIR)",
    )
    # Several collectors can share one raw root, each uploading its nodes' windows in parallel.
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args(argv)
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must be at least 0 and below --shard-count")
    if args.dump_remote_section and args.dump_bucket and not args.dump_prefix:
        parser.error("--dump-prefix is required with --dump-remote-section and --dump-bucket")
    return args


def _build_session(chdb_config: Path | None) -> Session:  # pragma: no cover
    import chdb.session

    if chdb_config is None:
        return chdb.session.Session(":memory:")
    return chdb.session.Session(f":memory:?config-file={chdb_config}")


def _flock_exclusive_nonblocking(file: IO[str]) -> None:  # pragma: no cover
    # fcntl exists on POSIX only; the call is isolated so the lock logic can be tested without it.
    import fcntl

    fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _acquire_collector_lock(lock_path: Path) -> IO[str] | None:
    # Exclusive, non-blocking lock so a second collector for the same raw_root and shard fails fast
    # instead of racing this one; caller must keep the returned file open to hold the lock.
    lock_file = open(lock_path, "w")  # noqa: SIM115
    try:
        _flock_exclusive_nonblocking(lock_file)
    except OSError:
        lock_file.close()
        return None
    return lock_file


def _build_targets(args: argparse.Namespace) -> tuple[S3Target, S3Target | None]:
    credentials = S3Credentials.from_rclone_conf(args.rclone_conf, args.remote_section)
    target = S3Target(endpoint=args.endpoint, bucket=args.bucket, prefix=args.prefix, credentials=credentials)
    if not (args.dump_remote_section and args.dump_bucket):
        return target, None
    dump_credentials = S3Credentials.from_rclone_conf(args.rclone_conf, args.dump_remote_section)
    dump_target = S3Target(
        endpoint=args.endpoint, bucket=args.dump_bucket, prefix=args.dump_prefix, credentials=dump_credentials
    )
    return target, dump_target


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    raw_root = RawRoot(args.raw_root)
    shard = Shard(args.shard_index, args.shard_count)
    lock_path = raw_root.path / shard.lock_name
    lock_file = _acquire_collector_lock(lock_path)
    if lock_file is None:
        print(f"another collector already holds the lock at {lock_path}", file=sys.stderr)
        sys.exit(1)

    target, dump_target = _build_targets(args)
    session = _build_session(args.chdb_config)
    session.query(f"SET {SESSION_SETTINGS}")
    collector = Collector(
        session,
        target,
        collected_retention_s=args.collected_retention_hours * _SECONDS_PER_HOUR,
        dump_target=dump_target,
        shard=shard,
        scratch_dir=args.scratch_dir,
    )

    idle_exit_seconds = args.idle_exit_minutes * _SECONDS_PER_MINUTE
    last_busy_at = time.monotonic()
    while True:
        # On timeout every thread's stack goes to stderr and the process exits for its caller to restart.
        faulthandler.dump_traceback_later(_CYCLE_TIMEOUT_S, exit=True)
        try:
            receipts = collector.collect_cycle(raw_root)
        finally:
            faulthandler.cancel_dump_traceback_later()
        for receipt in receipts:
            print(f"{receipt.family} {receipt.key} {receipt.rows}")
        if args.loop_seconds <= 0:
            return
        if receipts or raw_root.has_uncollected(dumps=dump_target is not None):
            last_busy_at = time.monotonic()
        elif idle_exit_seconds and time.monotonic() - last_busy_at >= idle_exit_seconds:
            print(f"nothing left to collect for {args.idle_exit_minutes:g} minutes, exiting", file=sys.stderr)
            return
        if not receipts:
            time.sleep(args.loop_seconds)


if __name__ == "__main__":  # pragma: no cover
    main()
