from __future__ import annotations

import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property, partial
from pathlib import Path
from typing import Callable, Literal

from ..config_manager import ConfigManager
from ..config_model import AutometricsConfig, JudgeEndpoint, NodeRunnerConfigModel
from ..convertation_manager import ConvertationManager
from ..inference.engine import EngineLoad, EngineUnit, InferenceEngine
from ..inference.inference_manager import InferenceManager
from ..logger import Logger
from ..orkestration_manager import FastApiOrkestrationManager
from ..trace import EventsStream
from .evaluators import (
    AaLcrEvaluator,
    AccLenEvaluator,
    AppWorldEvaluator,
    BfclEvaluator,
    CheckpointStatsEvaluator,
    ClawEvalEvaluator,
    DreddEvaluator,
    ForgeEvaluator,
    GigaMetricEvaluator,
    GigaMetricJudgedEvaluator,
    HarborRliEvaluator,
    LmEvalEvaluator,
    LmmsEvalEvaluator,
    MeraEvaluator,
    NewObmEvaluator,
    NewObmJudgedEvaluator,
    Nl2RepoEvaluator,
    ObmEvaluator,
    ObmJudgedEvaluator,
    PierRliEvaluator,
    PlanbenchEvaluator,
    SafetyUnifiedEvaluator,
    SkillsBenchEvaluator,
    SweBenchEvaluator,
    Tau2Evaluator,
    Tau3Evaluator,
    TerminalBenchEvaluator,
    TokenNllEvaluator,
)
from .evaluators.interface import EVALUATED, MetricsEvaluator, Stage
from .tensorboard_saver import TensorBoardSaver
from ..job_dispatcher import JD

EVALUATORS: dict[str, type[MetricsEvaluator]] = {
    "lm_eval": LmEvalEvaluator,
    "lmms_eval": LmmsEvalEvaluator,
    "obm": ObmEvaluator,
    "new_obm": NewObmEvaluator,
    "mera": MeraEvaluator,
    "bfcl": BfclEvaluator,
    "gigametric": GigaMetricEvaluator,
    "planbench": PlanbenchEvaluator,
    "swe_bench": SweBenchEvaluator,
    "nl2repo": Nl2RepoEvaluator,
    "terminal_bench": TerminalBenchEvaluator,
    "harbor_rli": HarborRliEvaluator,
    "pier_rli": PierRliEvaluator,
    "acc_len": AccLenEvaluator,
    "skillsbench": SkillsBenchEvaluator,
    "appworld": AppWorldEvaluator,
    "token_nll": TokenNllEvaluator,
    "checkpoint_stats": CheckpointStatsEvaluator,
    "tau2": Tau2Evaluator,
    "tau3": Tau3Evaluator,
    "obm_judged": ObmJudgedEvaluator,
    "new_obm_judged": NewObmJudgedEvaluator,
    "gigametric_judged": GigaMetricJudgedEvaluator,
    "forge": ForgeEvaluator,
    "safety_unified": SafetyUnifiedEvaluator,
    "aa_lcr": AaLcrEvaluator,
    "dredd": DreddEvaluator,
    "claweval": ClawEvalEvaluator,
}


@dataclass(eq=False)
class Task:
    evaluator: str
    stage: Stage
    after: Task | None
    state: Literal["pending", "running", "done", "failed"] = "pending"
    on_end: Callable[[], None] | None = None
    latency: bool = False

    @property
    def name(self) -> str:
        if self.stage.judge is None:
            return self.evaluator
        return f"{self.evaluator}@{self.stage.judge}"

    @property
    def ended(self) -> bool:
        return self.state in ("done", "failed")

    @property
    def scores_previous_stage(self) -> bool:
        return self.after is not None and self.stage.phase == "scoring"

    @property
    def previous_ended(self) -> bool:
        return self.after is None or self.after.ended

    @property
    def admissible(self) -> bool:
        if self.state != "pending":
            return False
        if self.scores_previous_stage:
            return self.after.state == "done"
        return self.previous_ended


@dataclass(eq=False)
class Drain:
    target: EngineSlot
    unit: EngineUnit
    donors: dict[str, list[EngineUnit]]
    deadline: float
    empty_probes: int = 0


@dataclass(eq=False)
class PendingOp:
    engine_key: str
    unit: EngineUnit
    index: int | None
    succeeded: bool | None = None


class EngineOpLog:
    POLL_SECONDS = 5

    def __init__(
        self,
        orkestration_manager: FastApiOrkestrationManager,
        engines: dict[str, InferenceEngine],
    ) -> None:
        self._om = orkestration_manager
        self._engines = engines

    def start(
        self,
        action: Literal["up", "down"],
        units_by_engine: dict[str, list[EngineUnit]],
    ) -> list[PendingOp]:
        pending: list[PendingOp] = []
        ops_on_this_node: list[tuple[int, dict]] = []
        for engine_key, units in units_by_engine.items():
            for unit in units:
                unit_has_no_nodes = not unit.ranks
                if unit_has_no_nodes:
                    succeeded = self._apply(action, engine_key)
                    pending.append(PendingOp(engine_key, unit, None, succeeded))
                    continue
                op = {"action": action, "engine": engine_key, "ranks": list(unit.ranks)}
                index = self._om.publish_op(op)
                pending.append(PendingOp(engine_key, unit, index))
                if self._om.global_node_rank in unit.ranks:
                    ops_on_this_node.append((index, op))
        for index, op in ops_on_this_node:
            self._execute(index, op)
        return pending

    def finished(self, op: PendingOp) -> bool:
        if op.succeeded is not None:
            return True
        if not self._om.barrier_released(self._stage(op.index)):
            return False
        some_rank_failed = self._om.barrier_released(self._failed_stage(op.index))
        op.succeeded = not some_rank_failed
        return True

    def run(
        self,
        action: Literal["up", "down"],
        units_by_engine: dict[str, list[EngineUnit]],
    ) -> list[PendingOp]:
        ops = self.start(action, units_by_engine)
        for op in ops:
            while not self.finished(op):
                time.sleep(self.POLL_SECONDS)
        return ops

    def finish(self) -> None:
        self._om.publish_op({"action": "finish"})

    def follow(self) -> None:
        index = 0
        while True:
            op = self._om.fetch_op(index)
            if op["action"] == "finish":
                return
            if self._om.global_node_rank in op["ranks"]:
                self._execute(index, op)
            index += 1

    def _execute(self, index: int, op: dict) -> None:
        try:
            if not self._apply(op["action"], op["engine"]):
                self._om.wait_barrier(self._failed_stage(index), expected=1)
        finally:
            self._om.wait_barrier(self._stage(index), expected=len(op["ranks"]))

    def _apply(self, action: str, engine_key: str) -> bool:
        try:
            engine = self._engines[engine_key]
            if action == "up":
                engine.start()
            else:
                engine.stop()
            return True
        except Exception:
            Logger.error(
                f"[scheduler] {action} {engine_key} failed:\n{traceback.format_exc()}"
            )
            return False

    @staticmethod
    def _stage(index: int) -> str:
        return f"ENGINE_OP_{index}"

    @classmethod
    def _failed_stage(cls, index: int) -> str:
        return f"{cls._stage(index)}_FAIL"


@dataclass(eq=False)
class EngineSlot:
    COOLDOWN_SECONDS = 10 * 60
    OVERLOAD_IN_FLIGHT_SHARE = 1.25
    OVERLOAD_TOKEN_USAGE = 0.92
    IDLE_LOAD_SHARE = 0.5
    OVERLOAD_PROBES = 3
    UNDERLOAD_PROBES = 6

    key: str
    engine: InferenceEngine
    units: list[EngineUnit]
    up: list[EngineUnit] = field(default_factory=list)
    cooldown_until: float = 0.0
    overloaded_probes: int = 0
    idle_probes: int = 0

    def observe(self, load: EngineLoad) -> None:
        capacity_per_unit = load.capacity / len(self.up)
        capacity_without_one_unit = capacity_per_unit * (len(self.up) - 1)
        in_flight = load.running + load.waiting
        too_many_requests = in_flight > self.OVERLOAD_IN_FLIGHT_SHARE * load.capacity
        kv_cache_full = load.token_usage > self.OVERLOAD_TOKEN_USAGE
        overloaded = too_many_requests or kv_cache_full
        idle = (
            load.waiting == 0
            and load.running <= self.IDLE_LOAD_SHARE * capacity_without_one_unit
        )
        if overloaded:
            self.overloaded_probes += 1
            self.idle_probes = 0
        elif idle:
            self.idle_probes += 1
            self.overloaded_probes = 0
        else:
            self.overloaded_probes = 0
            self.idle_probes = 0

    def changed(self) -> None:
        self.cooldown_until = time.monotonic() + self.COOLDOWN_SECONDS
        self.overloaded_probes = 0
        self.idle_probes = 0

    @property
    def cooling_down(self) -> bool:
        return time.monotonic() < self.cooldown_until

    @property
    def overloaded(self) -> bool:
        return self.overloaded_probes >= self.OVERLOAD_PROBES

    @property
    def underloaded(self) -> bool:
        return self.idle_probes >= self.UNDERLOAD_PROBES


class DynamicScheduler:
    PROBE_SECONDS = 30
    DRAIN_TIMEOUT_SECONDS = 30 * 60
    EMPTY_PROBES_TO_DRAIN = 2

    def __init__(
        self,
        op_log: EngineOpLog,
        engines: dict[str, InferenceEngine],
        tasks: list[Task],
    ) -> None:
        self._op_log = op_log
        self._slots: dict[str, EngineSlot] = {}
        for key, engine in engines.items():
            is_judge = key != EVALUATED
            self._slots[key] = EngineSlot(key, engine, engine.units(from_tail=is_judge))
        self._quarantined: set[int] = set()
        self._starting: list[PendingOp] = []
        self._drains: list[Drain] = []
        self._tasks = tasks
        self._task_threads: list[threading.Thread] = []

    def run(self) -> None:
        units_by_engine = {
            key: [list(unit.ranks) for unit in slot.units]
            for key, slot in self._slots.items()
        }
        task_names = [task.name for task in self._tasks]
        Logger.info(f"[scheduler] units: {units_by_engine}, tasks: {task_names}")
        try:
            while True:
                self._attach_started_units()
                self._advance_drains()
                self._probe()
                self._release_idle_engines()
                self._skip_tasks_that_cannot_run()
                self._grow_onto_free_units()
                self._attach_started_units()
                self._admit_tasks()
                self._take_units_of_other_engines()
                all_tasks_done = all(task.ended for task in self._tasks)
                all_engines_stopped = all(not slot.up for slot in self._slots.values())
                nothing_in_flight = not self._starting and not self._drains
                if all_tasks_done and all_engines_stopped and nothing_in_flight:
                    break
                time.sleep(self.PROBE_SECONDS)
        finally:
            for thread in self._task_threads:
                thread.join()
            self._op_log.finish()

    def _probe(self) -> None:
        for slot in self._slots.values():
            if not slot.up:
                continue
            load = slot.engine.load(slot.up)
            if load is None:
                continue
            slot.observe(load)
            if slot.overloaded or slot.underloaded:
                direction = "overloaded" if slot.overloaded else "underloaded"
                Logger.info(
                    f"[scheduler] {slot.key}: running={load.running} "
                    f"waiting={load.waiting} units={len(slot.up)} {direction}"
                )

    def _has_unfinished_tasks(self, key: str) -> bool:
        return any(
            not task.ended and key in task.stage.engine_keys for task in self._tasks
        )

    def _needed_now(self, key: str) -> bool:
        return any(
            key in task.stage.engine_keys
            and (task.state == "running" or task.admissible)
            for task in self._tasks
        )

    def _pinned(self, slot: EngineSlot) -> bool:
        return slot.key == EVALUATED and any(
            task.stage.pins_workers and task.state == "running" for task in self._tasks
        )

    def _release_idle_engines(self) -> None:
        idle: dict[str, list[EngineUnit]] = {}
        for slot in self._slots.values():
            if self._units_locked(slot):
                continue
            if slot.up and not self._has_unfinished_tasks(slot.key):
                idle[slot.key] = list(slot.up)
        if idle:
            self._release(idle)

    def _placeable(self, unit: EngineUnit) -> bool:
        return self._quarantined.isdisjoint(unit.ranks)

    def _anchors(self, key: str) -> list[EngineUnit]:
        return [
            unit
            for unit in self._slots[key].units
            if unit.standalone and self._placeable(unit)
        ]

    def _can_place_together(self, keys: tuple[str, ...]) -> bool:
        if len(keys) == 1:
            return bool(self._anchors(keys[0]))
        first_key, second_key = keys
        for first in self._anchors(first_key):
            for second in self._anchors(second_key):
                if set(first.ranks).isdisjoint(second.ranks):
                    return True
        return False

    def _skip_tasks_that_cannot_run(self) -> None:
        for task in self._tasks:
            if task.state != "pending" or not task.previous_ended:
                continue
            nothing_to_score = (
                task.scores_previous_stage and task.after.state == "failed"
            )
            if nothing_to_score:
                Logger.error(f"[scheduler] skip {task.name}: {task.after.name} failed")
                self._end(task, "failed")
            elif not self._can_place_together(task.stage.engine_keys):
                Logger.error(
                    f"[scheduler] skip {task.name}: {list(task.stage.engine_keys)} "
                    "can no longer be placed together"
                )
                self._end(task, "failed")

    def _claim_order(self, slot: EngineSlot) -> tuple[bool, int]:
        already_running = bool(slot.up) or slot.key == EVALUATED
        overload_depth = slot.overloaded_probes if slot.overloaded else 0
        return already_running, -overload_depth

    def _starting_units(self, slot: EngineSlot) -> list[EngineUnit]:
        return [op.unit for op in self._starting if op.engine_key == slot.key]

    def _units_locked(self, slot: EngineSlot) -> bool:
        return (
            self._pinned(slot)
            or bool(self._starting_units(slot))
            or self._in_drain(slot)
        )

    def _draining_units(self, slot: EngineSlot) -> list[EngineUnit]:
        units = []
        for drain in self._drains:
            units += drain.donors.get(slot.key, [])
        return units

    def _in_drain(self, slot: EngineSlot) -> bool:
        return any(
            drain.target is slot or slot.key in drain.donors for drain in self._drains
        )

    def _held_units(self, slot: EngineSlot) -> list[EngineUnit]:
        return slot.up + self._starting_units(slot) + self._draining_units(slot)

    def _wants_units(self, slot: EngineSlot) -> bool:
        if self._units_locked(slot):
            return False
        if not self._has_unfinished_tasks(slot.key):
            return False
        return bool(slot.up) or self._needed_now(slot.key)

    def _missing(self, slot: EngineSlot) -> list[EngineUnit]:
        missing = [
            unit for unit in slot.units if unit not in slot.up and self._placeable(unit)
        ]
        if slot.key != EVALUATED and self._has_unfinished_tasks(EVALUATED):
            evaluated_anchors = self._anchors(EVALUATED)
            if evaluated_anchors:
                anchor_ranks = set(evaluated_anchors[0].ranks)
                missing = [
                    unit for unit in missing if anchor_ranks.isdisjoint(unit.ranks)
                ]
        return missing

    def _free_units(
        self, slot: EngineSlot, claimed_ranks: set[int]
    ) -> list[EngineUnit]:
        free = []
        for unit in self._missing(slot):
            if self._free(unit) and claimed_ranks.isdisjoint(unit.ranks):
                free.append(unit)
        if slot.key == EVALUATED:
            return free
        return self._prefer_evaluated_capacity(free, claimed_ranks)

    def _prefer_evaluated_capacity(
        self, candidates: list[EngineUnit], claimed_ranks: set[int]
    ) -> list[EngineUnit]:
        evaluated = self._slots[EVALUATED]
        available = evaluated.up or [
            unit
            for unit in evaluated.units
            if self._placeable(unit) and self._free(unit)
        ]
        available = [unit for unit in available if claimed_ranks.isdisjoint(unit.ranks)]
        return sorted(
            candidates, key=lambda candidate: candidate.allocation_cost(available)
        )

    def _grow_onto_free_units(self) -> None:
        claimed: dict[str, list[EngineUnit]] = {}
        claimed_ranks: set[int] = set()
        starting_judges: list[str] = []
        for slot in sorted(self._slots.values(), key=self._claim_order):
            if not self._wants_units(slot):
                continue
            free = self._free_units(slot, claimed_ranks)
            standalone_free = [unit for unit in free if unit.standalone]
            if slot.up:
                units = free
            elif not standalone_free:
                continue
            elif slot.key == EVALUATED:
                units = free
            else:
                units = standalone_free[:1]
                starting_judges.append(slot.key)
            if not units:
                continue
            claimed[slot.key] = units
            for unit in units:
                claimed_ranks.update(unit.ranks)
        for key in starting_judges:
            for unit in self._free_units(self._slots[key], claimed_ranks):
                if unit not in claimed[key]:
                    claimed[key].append(unit)
                    claimed_ranks.update(unit.ranks)
        if claimed:
            self._grow(claimed)

    def _judge_holds_nodes(self) -> bool:
        for slot in self._slots.values():
            if slot.key == EVALUATED:
                continue
            if any(unit.ranks for unit in self._held_units(slot)):
                return True
        return False

    def _take_units_of_other_engines(self) -> None:
        for slot in sorted(self._slots.values(), key=self._claim_order):
            if not self._wants_units(slot) or slot.cooling_down:
                continue
            if slot.up and not slot.overloaded:
                continue
            candidates = [
                unit for unit in self._missing(slot) if slot.up or unit.standalone
            ]
            for unit in self._prefer_evaluated_capacity(candidates, set()):
                donors = self._donors(unit, slot)
                if donors is None:
                    continue
                self._move(slot, unit, donors)
                break

    def _admit_tasks(self) -> None:
        evaluated = self._slots[EVALUATED]
        for task in sorted(self._tasks, key=lambda task: not task.latency):
            if not task.admissible:
                continue
            if not all(self._slots[key].up for key in task.stage.engine_keys):
                continue
            paced_by = task.stage.judge or EVALUATED
            if self._slots[paced_by].overloaded and not (
                task.latency and paced_by == EVALUATED
            ):
                continue
            if task.stage.pins_workers:
                placeable_units = [
                    unit for unit in evaluated.units if self._placeable(unit)
                ]
                evaluated_fully_up = all(
                    unit in evaluated.up for unit in placeable_units
                )
                if self._pinned(evaluated) or not evaluated_fully_up:
                    continue
            task.state = "running"
            thread = threading.Thread(target=self._run_task, args=(task,), daemon=True)
            thread.start()
            self._task_threads.append(thread)

    def _owners(self, rank: int) -> list[EngineSlot]:
        owners = []
        for slot in self._slots.values():
            if any(rank in unit.ranks for unit in self._held_units(slot)):
                owners.append(slot)
        return owners

    def _free(self, unit: EngineUnit) -> bool:
        for rank in unit.ranks:
            if self._owners(rank):
                return False
        return True

    def _donors(
        self, unit: EngineUnit, receiver: EngineSlot
    ) -> dict[str, list[EngineUnit]] | None:
        starting_judge = not receiver.up and receiver.key != EVALUATED
        first_judge = starting_judge and not self._judge_holds_nodes()
        donors: dict[str, list[EngineUnit]] = {}
        for rank in unit.ranks:
            for donor in self._owners(rank):
                given = donors.setdefault(donor.key, [])
                for up_unit in donor.up:
                    if rank in up_unit.ranks and up_unit not in given:
                        given.append(up_unit)
        for key, given in donors.items():
            donor = self._slots[key]
            if self._units_locked(donor):
                return None
            idle_judge = (
                starting_judge and key != EVALUATED and not self._needed_now(key)
            )
            if not (first_judge or idle_judge) and (
                donor.cooling_down or not donor.underloaded
            ):
                return None
            gives_last_unit = not any(
                up_unit.standalone and up_unit not in given for up_unit in donor.up
            )
            if gives_last_unit:
                if self._has_unfinished_tasks(key) and not idle_judge:
                    return None
                donors[key] = list(donor.up)
        return donors

    def _move(
        self,
        slot: EngineSlot,
        unit: EngineUnit,
        donors: dict[str, list[EngineUnit]],
    ) -> None:
        given_by_donor = {
            key: [list(given_unit.ranks) for given_unit in given]
            for key, given in donors.items()
        }
        Logger.info(
            f"[scheduler] move {list(unit.ranks)} from {given_by_donor} to {slot.key}"
        )
        for key, units in donors.items():
            donor = self._slots[key]
            donor.engine.detach(units)
            for given_unit in units:
                donor.up.remove(given_unit)
        deadline = time.monotonic() + self.DRAIN_TIMEOUT_SECONDS
        self._drains.append(Drain(slot, unit, donors, deadline))

    def _advance_drains(self) -> None:
        for drain in list(self._drains):
            if self._drain_is_empty(drain):
                drain.empty_probes += 1
            else:
                drain.empty_probes = 0
            if drain.empty_probes == self.EMPTY_PROBES_TO_DRAIN:
                self._drains.remove(drain)
                self._finish_move(drain)
            elif time.monotonic() >= drain.deadline:
                self._drains.remove(drain)
                self._put_back(drain)

    def _drain_is_empty(self, drain: Drain) -> bool:
        for key, units in drain.donors.items():
            load = self._slots[key].engine.load(units)
            if load is None:
                return False
            if load.running + load.waiting + load.kv_in_transfer > 0:
                return False
        return True

    def _finish_move(self, drain: Drain) -> None:
        self._stop(drain.donors)
        for key in drain.donors:
            donor = self._slots[key]
            donor.changed()
            if not donor.up:
                donor.cooldown_until = 0.0
        target = drain.target
        target_can_grow = (
            self._placeable(drain.unit)
            and self._free(drain.unit)
            and self._has_unfinished_tasks(target.key)
        )
        if target_can_grow:
            self._grow({target.key: [drain.unit]})
        else:
            target.changed()

    def _put_back(self, drain: Drain) -> None:
        ranks = {
            key: [list(u.ranks) for u in units] for key, units in drain.donors.items()
        }
        Logger.warning(
            f"[scheduler] {ranks} did not drain in {self.DRAIN_TIMEOUT_SECONDS}s, "
            "putting them back"
        )
        for key, units in drain.donors.items():
            donor = self._slots[key]
            donor.engine.attach(units)
            donor.up += units
            donor.changed()
        drain.target.changed()

    def _grow(self, units_by_engine: dict[str, list[EngineUnit]]) -> None:
        for key, units in units_by_engine.items():
            unit_ranks = [list(unit.ranks) for unit in units]
            Logger.info(f"[scheduler] grow {key} by {unit_ranks}")
        self._starting += self._op_log.start("up", units_by_engine)

    def _attach_started_units(self) -> None:
        finished = [op for op in self._starting if self._op_log.finished(op)]
        for slot in self._slots.values():
            ready = [op for op in finished if op.engine_key == slot.key]
            if not ready:
                continue
            still_starting = [
                op
                for op in self._starting
                if op.engine_key == slot.key and op not in finished
            ]
            if slot.key == EVALUATED and still_starting:
                continue
            if any(op.unit.standalone for op in still_starting):
                ready = [op for op in ready if op.unit.standalone]
            for op in ready:
                self._starting.remove(op)
            self._attach(slot, ready)

    def _attach(self, slot: EngineSlot, ready: list[PendingOp]) -> None:
        failed = [op.unit for op in ready if not op.succeeded]
        started = [op.unit for op in ready if op.succeeded]
        without_anchor: list[EngineUnit] = []
        standalone_first = sorted(started, key=lambda unit: not unit.standalone)
        for unit in standalone_first:
            slot_can_serve = any(up_unit.standalone for up_unit in slot.up)
            if not unit.standalone and not slot_can_serve:
                without_anchor.append(unit)
                continue
            try:
                slot.engine.attach([unit])
            except Exception:
                Logger.error(
                    f"[scheduler] {slot.key} router did not take {unit.worker_urls}:\n"
                    f"{traceback.format_exc()}"
                )
                slot.engine.detach([unit])
                failed.append(unit)
                continue
            slot.up.append(unit)
            Logger.info(f"[scheduler] {slot.key} unit {list(unit.ranks)} is up")
        if failed or without_anchor:
            self._stop({slot.key: failed + without_anchor})
        for unit in failed:
            self._quarantined.update(unit.ranks)
            Logger.error(
                f"[scheduler] {slot.key} unit {list(unit.ranks)} failed to start, "
                "its nodes are quarantined"
            )
        slot_failed_to_start = not slot.up and not self._starting_units(slot)
        if slot_failed_to_start:
            for task in self._tasks:
                waits_here = task.state == "pending" and task.previous_ended
                if waits_here and slot.key in task.stage.engine_keys:
                    Logger.error(f"[scheduler] skip {task.name}: [{slot.key!r}] failed")
                    self._end(task, "failed")
        slot.changed()

    def _release(self, units_by_engine: dict[str, list[EngineUnit]]) -> None:
        for key, units in units_by_engine.items():
            ranks = [list(unit.ranks) for unit in units]
            Logger.info(f"[scheduler] release {key} {ranks}")
            self._slots[key].engine.detach(units)
        self._stop(units_by_engine)
        for key, units in units_by_engine.items():
            slot = self._slots[key]
            for unit in units:
                slot.up.remove(unit)
            slot.changed()

    def _stop(self, units_by_engine: dict[str, list[EngineUnit]]) -> None:
        for op in self._op_log.run("down", units_by_engine):
            if not op.succeeded:
                self._quarantined.update(op.unit.ranks)
                Logger.error(
                    f"[scheduler] {op.engine_key} unit {list(op.unit.ranks)} failed "
                    "to stop, its nodes are quarantined"
                )

    def _run_task(self, task: Task) -> None:
        try:
            with EventsStream.shooter(
                task.evaluator,
                phase=task.stage.phase,
                secondary_model=task.stage.judge,
            ):
                task.stage.run()
        except Exception:
            Logger.error(f"[scheduler] {task.name} failed:\n{traceback.format_exc()}")
            self._end(task, "failed")
            return
        self._end(task, "done")

    @staticmethod
    def _end(task: Task, state: Literal["done", "failed"]) -> None:
        task.state = state
        if task.on_end is not None:
            task.on_end()


class AutoMetricsManager:
    def __init__(self):
        self.tensorboard_path: Path | None = None
        self.checkpoint_path: Path | None = None
        self.run_name: str | None = None
        self.artifacts_path: Path | None = None
        self.hf_checkpoint_path: Path | None = None
        self.model_info = None
        self.job_id: str | None = None
        self.pam_config_path: Path | None = None
        self.run_uuid: str = str(uuid.uuid4())
        self.run_config: NodeRunnerConfigModel | None = None
        self.orkestration_manager: FastApiOrkestrationManager | None = None
        self.engines_by_key: dict[str, InferenceEngine] = {}
        self.tasks: list[Task] = []
        self._save_lock = threading.Lock()

    def setup(
        self,
        *,
        config_manager: ConfigManager,
        run_config: NodeRunnerConfigModel,
        orkestration_manager: FastApiOrkestrationManager,
        convertation_manager: ConvertationManager,
    ) -> None:
        autometrics_config = run_config.autometrics

        self.tensorboard_path = (
            Path(run_config.tensorboard_path)
            if run_config.tensorboard_path is not None
            else None
        )
        self.checkpoint_path = config_manager.checkpoint_path
        self.run_name = run_config.run_name
        self.artifacts_path = config_manager.artifacts_path
        self.hf_checkpoint_path = convertation_manager.hf_checkpoint_path
        self.model_info = autometrics_config.model_info
        self.job_id = config_manager.job_id
        self.pam_config_path = config_manager.config_path
        self.run_config = run_config
        self.orkestration_manager = orkestration_manager

        self._setup_engines()
        if orkestration_manager.is_master:
            self._archive_previous_metrics(autometrics_config)
            self._build_tasks()

    def run_pipeline(self) -> None:
        op_log = EngineOpLog(self.orkestration_manager, self.engines_by_key)
        if not self.orkestration_manager.is_master:
            op_log.follow()
            return
        send_at_end = self.run_config.autometrics.globals.send_metrics_at_end
        with JD.metrics_sent_at_end(send_at_end):
            DynamicScheduler(op_log, self.engines_by_key, self.tasks).run()

    def _setup_engines(self) -> None:
        judges: list[str] = []
        for _, config in self.run_config.autometrics.evaluators:
            for judge in config.secondary_model_names:
                if judge == EVALUATED:
                    raise ValueError(
                        f"judge name {EVALUATED!r} is reserved for the evaluated model"
                    )
                if judge not in judges:
                    judges.append(judge)
        self.engines_by_key = {EVALUATED: InferenceManager.setup_inference_engine()}
        for judge_index, judge in enumerate(judges):
            self.engines_by_key[judge] = InferenceManager.setup_preset_inference_engine(
                model_name=judge, judge_index=judge_index
            )

    def _build_tasks(self) -> None:
        evaluated = self.engines_by_key[EVALUATED]
        judge_endpoints: dict[str, JudgeEndpoint] = {}
        for key, engine in self.engines_by_key.items():
            if key != EVALUATED:
                judge_endpoints[key] = JudgeEndpoint(
                    engine.base_url, engine.inference_instances
                )
        for name, config in self.run_config.autometrics.evaluators:
            config.judge_endpoints = judge_endpoints
            evaluator = EVALUATORS[name]()
            base_url = evaluated.base_url
            if (
                evaluator.LATENCY
                and evaluated.obm_proxy_manager is not None
                and (evaluated.worker_urls or self.run_config.is_replay_mode)
            ):
                base_url += "/latency"
            config.apply_defaults(
                inference_instances=evaluated.inference_instances,
                base_url=base_url,
                direct_base_url=evaluated.direct_base_url,
                worker_urls=evaluated.worker_urls,
                inference_config=self.run_config.autometrics.inference,
                max_seq_len=self.run_config.max_seq_len,
                hf_checkpoint_path=self.hf_checkpoint_path,
            )
            evaluator.setup(
                config,
                self.artifacts_path,
                self.hf_checkpoint_path,
                self.model_info,
            )
            Logger.info(f"Evaluator {name} registered successfully")
            tasks: list[Task] = []
            previous_task = None
            for stage in evaluator.stages():
                assert EVALUATED in stage.engine_keys or previous_task is not None
                task = Task(name, stage, previous_task, latency=evaluator.LATENCY)
                tasks.append(task)
                previous_task = task
            if tasks:
                tasks[-1].on_end = partial(
                    self._save_if_measured, name, evaluator, tasks
                )
            self.tasks += tasks

    def _save_if_measured(
        self, evaluator_name: str, evaluator: MetricsEvaluator, tasks: list[Task]
    ) -> None:
        some_stage_done = any(task.state == "done" for task in tasks)
        scoring_failed = any(
            task.scores_previous_stage and task.state == "failed" for task in tasks
        )
        if not some_stage_done or scoring_failed:
            return
        try:
            self.save_evaluator_metrics(evaluator_name, evaluator)
        except Exception:
            Logger.error(
                f"{evaluator_name}: saving metrics failed:\n{traceback.format_exc()}"
            )

    def _archive_previous_metrics(self, autometrics_config: AutometricsConfig) -> None:
        if not autometrics_config.globals.full_metrics_recalculate:
            return
        metrics_dir = self.artifacts_path / "metrics"
        if not metrics_dir.exists():
            return
        archived = metrics_dir.with_name(
            f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        metrics_dir.rename(archived)
        Logger.info(f"full_metrics_recalculate: moved {metrics_dir} to {archived}")

    @cached_property
    def tensorboard_saver(self) -> TensorBoardSaver:
        return TensorBoardSaver(
            self.checkpoint_path, self.tensorboard_path, self.run_name
        )

    def save_evaluator_metrics(
        self, evaluator_name: str, evaluator: MetricsEvaluator
    ) -> None:
        with self._save_lock:
            metrics = evaluator.get_metrics()
            if not metrics:
                return

            upload = None
            # An external API run has no checkpoint, so there is no jd_data.json
            if not self.run_config.autometrics.inference.uses_external_api:
                artifacts_path = JD.checkpoint_artifacts_path(self.hf_checkpoint_path)
                JD.ensure_autometrics_record(
                    artifacts_path=artifacts_path,
                    run_uuid=self.run_uuid,
                    checkpoint_path=self.checkpoint_path,
                    hf_checkpoint_path=self.hf_checkpoint_path,
                    run_name=self.run_name,
                    job_id=self.job_id,
                    pam_config_path=self.pam_config_path,
                )
                measurement_metrics = evaluator.measurement_metrics(metrics)
                if measurement_metrics:
                    JD.add_evaluator_metrics(
                        artifacts_path=artifacts_path,
                        run_uuid=self.run_uuid,
                        evaluator_name=evaluator_name,
                        metrics=measurement_metrics,
                        metrics_files_mtime_ns=evaluator.measured_files_mtime_ns(),
                    )
                else:
                    Logger.warning(
                        f"{evaluator_name} produced no new metric files; "
                        f"{len(metrics)} metric(s) not recorded to jd_data"
                    )
                if evaluator.measurement_files is not None:
                    upload = JD.publish_metrics_cache_per_measurement(
                        make_builder=evaluator.make_cache_scores_builder,
                        measurement=evaluator.measurement_files,
                        metrics_dir=evaluator.metrics_dir,
                        hf_checkpoint_path=evaluator.hf_checkpoint_path,
                        shooter_name=evaluator.METRICS_DIR_NAME,
                        run_uuid=self.run_uuid,
                    )

            self.tensorboard_saver.save_metrics_to_tensorboard(metrics)
            evaluator.after_run(metrics, upload)

    def cleanup(self) -> None:
        for engine in self.engines_by_key.values():
            try:
                engine.close()
            except Exception:
                Logger.error(
                    f"Cleanup: failed to stop {engine.log_profile} inference engine"
                )
