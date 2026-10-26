from __future__ import annotations

import argparse
import signal
from collections.abc import Sequence
from pathlib import Path
from types import FrameType

from metric_sampler.cloud.summaries.catalog import DEFAULT_CHUNK_STEPS, SUMMARIES_BY_NAME
from metric_sampler.cloud.summaries.clickhouse import TIMEOUT_SECONDS, ClickHouse
from metric_sampler.cloud.summaries.clock import Clock
from metric_sampler.cloud.summaries.cold_mover import MAX_IO_FULL, PARALLEL, ColdMover
from metric_sampler.cloud.summaries.log import Log
from metric_sampler.cloud.summaries.pressure import IoPressure
from metric_sampler.cloud.summaries.recompute import Recompute
from metric_sampler.cloud.summaries.service import Service
from metric_sampler.cloud.summaries.slots import SLOTS, InsertSlots

LOG_DIRECTORY = Path("logs")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m metric_sampler.cloud.summaries",
        description="The summary loops that run on the ClickHouse server, and the tools that share their state.",
    )
    parser.add_argument(
        "--timeout", type=int, default=TIMEOUT_SECONDS, help="seconds a statement may take (default: %(default)s)"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("run", help="run every loop in this process, logging to logs/<loop>.log")
    recompute = commands.add_parser("recompute", help="recompute a step range of one summary of one run in place")
    recompute.add_argument("summary", choices=sorted(SUMMARIES_BY_NAME), metavar="summary")
    recompute.add_argument("exp")
    recompute.add_argument("run")
    recompute.add_argument("first_step", type=int, metavar="from")
    recompute.add_argument("last_step", type=int, metavar="to")
    recompute.add_argument("--chunk-steps", type=int, default=DEFAULT_CHUNK_STEPS)
    recompute.add_argument("--slots", type=int, default=SLOTS)
    move_cold = commands.add_parser("move-cold", help="move the parts of one run of the tables to the cold volume")
    move_cold.add_argument("exp")
    move_cold.add_argument("run")
    move_cold.add_argument("tables", nargs="+", metavar="table")
    move_cold.add_argument("--parallel", type=int, default=PARALLEL)
    move_cold.add_argument("--min-age-hours", type=int, default=0)
    move_cold.add_argument("--min-bytes", type=int, default=0)
    move_cold.add_argument("--max-io-full", type=int, default=MAX_IO_FULL)
    return parser


def _run(client: ClickHouse, clock: Clock) -> int:
    def stop(signal_number: int, frame: FrameType | None) -> None:
        clock.stopped.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    Service(client, clock, Log(clock), LOG_DIRECTORY).run()
    return 0


def _recompute(arguments: argparse.Namespace, client: ClickHouse, clock: Clock) -> int:
    slots = InsertSlots(client, clock, slots=arguments.slots)
    recompute = Recompute(slots, Log(clock), arguments.chunk_steps)
    summary = SUMMARIES_BY_NAME[arguments.summary]
    return 0 if recompute.run(summary, arguments.exp, arguments.run, arguments.first_step, arguments.last_step) else 1


def _move_cold(arguments: argparse.Namespace, client: ClickHouse, clock: Clock) -> int:
    mover = ColdMover(client, clock, Log(clock), IoPressure(), arguments.parallel, arguments.max_io_full)
    mover.move_run(arguments.exp, arguments.run, arguments.tables, arguments.min_age_hours, arguments.min_bytes)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    client = ClickHouse(arguments.timeout)
    clock = Clock()
    if arguments.command == "run":
        return _run(client, clock)
    if arguments.command == "recompute":
        return _recompute(arguments, client, clock)
    return _move_cold(arguments, client, clock)
