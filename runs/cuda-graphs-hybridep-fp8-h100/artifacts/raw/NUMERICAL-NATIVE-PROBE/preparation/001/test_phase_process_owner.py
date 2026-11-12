#!/usr/bin/env python3
"""CPU-only process-tree checks for the two-phase probe owner."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from preflight_topology import is_exact_nv18, parse_topology_matrix


OWNER = Path(__file__).with_name("phase_process_owner.py").resolve()
PYTHON = sys.executable


def start_ticks(pid: int) -> int:
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    return int(raw[raw.rfind(")") + 2 :].split()[19])


def wait_for(path: Path, timeout: float = 5.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {path}")


def wait_absent(pid: int, original_ticks: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
            fields = raw[raw.rfind(")") + 2 :].split()
            if int(fields[19]) != original_ticks:
                return
            # A zombie is still present until its parent reaps it.
            time.sleep(0.02)
            continue
        except FileNotFoundError:
            return
        time.sleep(0.02)
    raise AssertionError(f"PID {pid} remained present after owner release")


def phase_args(root: Path, command: list[str], *, deadline: float) -> list[str]:
    gift = root / "gift"
    phase = root / "phase"
    gift.mkdir(parents=True, exist_ok=True)
    phase.mkdir(parents=True, exist_ok=True)
    entrypoint = phase / "compiler_entry.py"
    entrypoint.write_text("# isolated CPU fixture\n", encoding="utf-8")
    pid = os.getpid()
    return [
        PYTHON,
        str(OWNER),
        "--phase",
        "fixture",
        "--parent-pid",
        str(pid),
        "--parent-start-ticks",
        str(start_ticks(pid)),
        "--entrypoint",
        str(entrypoint),
        "--cwd",
        str(gift),
        "--ledger",
        str(root / "owned-processes.jsonl"),
        "--status",
        str(root / "owner-status.json"),
        "--owner-record",
        str(root / "owner.json"),
        "--log",
        str(root / "launcher.log"),
        "--deadline-seconds",
        str(deadline),
        "--term-grace",
        "0.4",
        "--kill-grace",
        "1.0",
        "--",
        *command,
    ]


class PhaseOwnerProcessTests(unittest.TestCase):
    def assert_released(self, root: Path, expected_reason: str, expected_exit: int) -> dict:
        status = json.loads((root / "owner-status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["reason"], expected_reason)
        self.assertEqual(status["exit_code"], expected_exit)
        self.assertTrue(status["all_recorded_identities_absent"], status)
        self.assertEqual(status["remaining_identities"], [])
        verify = subprocess.run(
            [PYTHON, str(OWNER), "verify", str(root / "owned-processes.jsonl")],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(verify.returncode, 0, verify.stdout + verify.stderr)
        return status

    def test_fast_orphan_without_entrypoint_is_adopted_between_scans(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child_pid_file = root / "child.pid"
            worker = (
                "import os,sys,time; raw=open(f'/proc/{os.getpid()}/stat').read(); "
                "ticks=raw[raw.rfind(')')+2:].split()[19]; "
                "open(sys.argv[1],'w').write(f'{os.getpid()}:{ticks}'); time.sleep(2)"
            )
            parent = (
                "import subprocess,sys,time,os; time.sleep(.08); "
                "subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]],"
                "start_new_session=True); os._exit(0)"
            )
            command = [PYTHON, "-c", parent, worker, str(child_pid_file)]
            process = subprocess.Popen(phase_args(root, command, deadline=3.0))
            self.assertEqual(process.wait(timeout=6), 70)
            child_pid_text, child_ticks_text = wait_for(child_pid_file).split(":")
            child_pid, child_ticks = int(child_pid_text), int(child_ticks_text)
            status = self.assert_released(root, "orphaned_descendants_after_torchrun_exit", 70)
            self.assertGreaterEqual(status["record_count"], 2)
            ledger = [
                json.loads(line)
                for line in (root / "owned-processes.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            child_record = next(item for item in ledger if item["pid"] == child_pid)
            self.assertEqual(child_record["proof"], "subreaper_adopted_descendant")
            wait_absent(child_pid, child_ticks)

    def test_initial_registration_failure_enters_adopted_child_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ready_file = root / "root.ready"
            child_pid_file = root / "child.pid"
            worker = (
                "import os,sys,time; raw=open(f'/proc/{os.getpid()}/stat').read(); "
                "ticks=raw[raw.rfind(')')+2:].split()[19]; "
                "open(sys.argv[1],'w').write(f'{os.getpid()}:{ticks}'); time.sleep(2)"
            )
            parent = (
                "import os,signal,subprocess,sys,time; from pathlib import Path; "
                "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                "Path(sys.argv[1]).write_text('ready'); time.sleep(.08); "
                "subprocess.Popen([sys.executable,'-c',sys.argv[2],sys.argv[3]],"
                "start_new_session=True); os._exit(0)"
            )
            command = [PYTHON, "-c", parent, str(ready_file), worker, str(child_pid_file)]
            args = phase_args(root, command, deadline=3.0)
            wrapper = root / "registration_failure_owner.py"
            wrapper.write_text(
                "import importlib.util,os,sys,time\n"
                f"owner_path={str(OWNER)!r}\n"
                "from pathlib import Path\n"
                "spec=importlib.util.spec_from_file_location('phase_process_owner',owner_path)\n"
                "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)\n"
                "original=module.PhaseOwner._register; first=True\n"
                "def fail_initial(self,pid,parent,ticks,proof):\n"
                "    global first\n"
                "    if proof == 'phase_owner_direct_child' and first:\n"
                "        first=False; deadline=time.monotonic()+3\n"
                "        ready=Path(os.environ['PHASE_OWNER_TEST_READY_FILE'])\n"
                "        while not ready.exists() and time.monotonic()<deadline: time.sleep(.005)\n"
                "        if not ready.exists(): raise RuntimeError('task child did not reach ready point')\n"
                "        return False\n"
                "    return original(self,pid,parent,ticks,proof)\n"
                "module.PhaseOwner._register=fail_initial\n"
                "sys.argv=[owner_path,*sys.argv[2:]]\n"
                "raise SystemExit(module.main())\n",
                encoding="utf-8",
            )
            args[1:2] = [str(wrapper), str(OWNER)]
            env = os.environ.copy()
            env["PHASE_OWNER_TEST_READY_FILE"] = str(ready_file)
            process = subprocess.Popen(args, env=env)
            self.assertEqual(process.wait(timeout=6), 70)
            child_pid_text, child_ticks_text = wait_for(child_pid_file).split(":")
            child_pid, child_ticks = int(child_pid_text), int(child_ticks_text)
            status = self.assert_released(root, "initial_registration_failure", 70)
            self.assertGreaterEqual(status["record_count"], 1)
            ledger = [
                json.loads(line)
                for line in (root / "owned-processes.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            child_record = next(item for item in ledger if item["pid"] == child_pid)
            self.assertEqual(child_record["proof"], "subreaper_adopted_descendant")
            wait_absent(child_pid, child_ticks)

    def test_orphaned_new_session_child_is_recovered_after_parent_exit_and_exec(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child_pid_file = root / "child.pid"
            worker = (
                "import os,signal,sys,time; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "raw=open(f'/proc/{os.getpid()}/stat').read(); ticks=raw[raw.rfind(')')+2:].split()[19]; "
                "open(sys.argv[2],'w').write(f'{os.getpid()}:{ticks}'); "
                "time.sleep(.35); "
                "os.execv(sys.executable,[sys.executable,'-c',"
                "\"import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)\"])"
            )
            parent = (
                "import subprocess,sys,os; "
                "subprocess.Popen([sys.executable,'-c',sys.argv[2],sys.argv[1],sys.argv[3]],"
                "start_new_session=True); os._exit(0)"
            )
            command = [PYTHON, "-c", parent, str(root / "phase/compiler_entry.py"), worker, str(child_pid_file)]
            process = subprocess.Popen(phase_args(root, command, deadline=10.0))
            self.assertEqual(process.wait(timeout=8), 70)
            child_pid_text, child_ticks_text = wait_for(child_pid_file).split(":")
            child_pid, child_ticks = int(child_pid_text), int(child_ticks_text)
            status = self.assert_released(root, "orphaned_descendants_after_torchrun_exit", 70)
            self.assertGreaterEqual(status["record_count"], 2)
            wait_absent(child_pid, child_ticks)

    def test_deadline_kills_exact_recorded_process_tree_without_extension(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child_pid_file = root / "child.pid"
            worker = (
                "import os,signal,sys,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                "raw=open(f'/proc/{os.getpid()}/stat').read(); ticks=raw[raw.rfind(')')+2:].split()[19]; "
                "open(sys.argv[2],'w').write(f'{os.getpid()}:{ticks}'); time.sleep(30)"
            )
            parent = (
                "import subprocess,sys,time; "
                "subprocess.Popen([sys.executable,'-c',sys.argv[2],sys.argv[1],sys.argv[3]],"
                "start_new_session=True); time.sleep(30)"
            )
            command = [PYTHON, "-c", parent, str(root / "phase/compiler_entry.py"), worker, str(child_pid_file)]
            began = time.monotonic()
            process = subprocess.Popen(phase_args(root, command, deadline=0.7))
            child_pid_text, child_ticks_text = wait_for(child_pid_file).split(":")
            child_pid, child_ticks = int(child_pid_text), int(child_ticks_text)
            self.assertEqual(process.wait(timeout=6), 124)
            self.assertLess(time.monotonic() - began, 6)
            self.assert_released(root, "deadline", 124)
            wait_absent(child_pid, child_ticks)

    def test_controller_term_is_detected_after_early_controller_exit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "gift").mkdir()
            phase = root / "phase"
            phase.mkdir()
            entry = phase / "compiler_entry.py"
            entry.write_text("# isolated CPU fixture\n", encoding="utf-8")
            child_pid_file = root / "child.pid"
            worker = (
                "import os,signal,sys,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                "raw=open(f'/proc/{os.getpid()}/stat').read(); ticks=raw[raw.rfind(')')+2:].split()[19]; "
                "open(sys.argv[2],'w').write(f'{os.getpid()}:{ticks}'); time.sleep(30)"
            )
            parent = (
                "import subprocess,sys,time; "
                "subprocess.Popen([sys.executable,'-c',sys.argv[2],sys.argv[1],sys.argv[3]],"
                "start_new_session=True); time.sleep(30)"
            )
            command = [PYTHON, "-c", parent, str(entry), worker, str(child_pid_file)]
            controller_script = (
                "import os,subprocess,sys,time; "
                "from pathlib import Path; "
                "raw=Path(f'/proc/{os.getpid()}/stat').read_text(); "
                "ticks=int(raw[raw.rfind(')')+2:].split()[19]); "
                "cmd=[sys.executable,sys.argv[1],'--phase','fixture','--parent-pid',str(os.getpid()),"
                "'--parent-start-ticks',str(ticks),'--entrypoint',sys.argv[2],'--cwd',sys.argv[3],"
                "'--ledger',sys.argv[4],'--status',sys.argv[5],'--owner-record',sys.argv[6],"
                "'--log',sys.argv[7],'--deadline-seconds','20','--term-grace','0.4','--kill-grace','1',"
                "'--',*sys.argv[9:]]; "
                "log=open(sys.argv[7]+'.owner','ab',buffering=0); p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT); "
                "open(sys.argv[8],'w').write(str(p.pid)); time.sleep(30)"
            )
            controller_command = [
                PYTHON,
                "-c",
                controller_script,
                str(OWNER),
                str(entry),
                str(root / "gift"),
                str(root / "owned-processes.jsonl"),
                str(root / "owner-status.json"),
                str(root / "owner.json"),
                str(root / "launcher.log"),
                str(root / "owner.pid"),
                *command,
            ]
            controller = subprocess.Popen(controller_command)
            child_pid_text, child_ticks_text = wait_for(child_pid_file).split(":")
            child_pid, child_ticks = int(child_pid_text), int(child_ticks_text)
            wait_for(root / "owner.pid")
            owner_record = json.loads((root / "owner.json").read_text(encoding="utf-8"))
            rss_deadline = time.monotonic() + 3
            rss_values = (0, 0)
            while time.monotonic() < rss_deadline:
                sample = subprocess.run(
                    [PYTHON, str(OWNER), "rss", str(root / "owned-processes.jsonl")],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                rss_values = tuple(int(value) for value in sample.stdout.split())
                if rss_values[1] >= 2:
                    break
                time.sleep(0.05)
            self.assertGreater(rss_values[0], 0)
            self.assertGreaterEqual(rss_values[1], 2)
            ledger = [
                json.loads(line)
                for line in (root / "owned-processes.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(
                any(item["pid"] == child_pid and item["sid"] != owner_record["session"] for item in ledger),
                "separate-session worker was missing from the owned RSS ledger",
            )
            os.kill(controller.pid, signal.SIGTERM)
            controller.wait(timeout=3)
            deadline = time.monotonic() + 6
            while not (root / "owner-status.json").exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue((root / "owner-status.json").exists(), "orphan owner did not publish status")
            self.assert_released(root, "controller_lost", 75)
            wait_absent(child_pid, child_ticks)


class PreflightTopologyTests(unittest.TestCase):
    def test_actual_preflight_validator_accepts_nv18_and_refuses_bad_or_missing_links(self) -> None:
        fixture = Path(__file__).resolve().parents[3] / "NATIVE-SMOKE/001/prelaunch.txt"
        text = fixture.read_text(encoding="utf-8")
        matrix = parse_topology_matrix(text)
        self.assertTrue(is_exact_nv18(matrix))

        altered = {label: values.copy() for label, values in matrix.items()}
        altered["GPU0"][1] = "NODE"
        self.assertFalse(is_exact_nv18(altered))

        missing = {label: values.copy() for label, values in matrix.items()}
        missing["GPU7"].pop(6)
        self.assertFalse(is_exact_nv18(missing))


if __name__ == "__main__":
    unittest.main(verbosity=2)
