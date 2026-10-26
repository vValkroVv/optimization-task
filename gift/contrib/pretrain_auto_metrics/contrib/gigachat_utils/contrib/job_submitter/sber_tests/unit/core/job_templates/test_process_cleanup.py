"""Cleanup behavior against fake proc files; no real signals or process scans."""

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
TEMPLATE = (
    Path(__file__).resolve().parents[4]
    / "job_submitter/core/job_templates/process_cleanup.sh.j2"
)


def stat(pid, state="S", start=10, pgid=100, ppid=100):
    fields = ["0"] * 40
    for index, value in {0: state, 1: ppid, 2: pgid, 3: pgid, 19: start}.items():
        fields[index] = str(value)
    return f"{pid} (worker) {' '.join(fields)}\n"


@pytest.fixture
def harness(tmp_path):
    proc = tmp_path / "proc"
    for pid in (100, 101, 102, 900):
        directory = proc / str(pid)
        directory.mkdir(parents=True)
        (directory / "stat").write_text(stat(pid, pgid=900 if pid == 900 else 100))
        (directory / "cgroup").write_text("0::/\n")
        (directory / "environ").write_bytes(
            b"SLURM_JOB_ID=99\0"
            if pid == 900
            else b"SLURM_JOB_ID=23\0SLURM_PROCID=0\0SLURM_STEP_ID=1\0"
        )
    for pid in (100, 101, 900):
        task = proc / str(pid) / "task" / str(pid)
        task.mkdir(parents=True)
        os.link(proc / str(pid) / "stat", task / "stat")
    task = proc / "101/task/102"
    task.mkdir()
    os.link(proc / "102/stat", task / "stat")
    (proc / "102/wchan").write_text("driver_wait\n")
    (proc / "102/stack").write_text("[<0>] driver_wait\n")
    (tmp_path / "zombie.stat").write_text(stat(101, state="Z"))
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "ps").write_text(
        "#!/bin/sh\nprintf '100 1 100 100\\n101 100 100 100\\n900 1 900 900\\n'\n"
    )
    (bindir / "ps").chmod(0o700)
    source = (
        TEMPLATE.read_text()
        .replace("\\$", "$")
        .split("trap 'handle_shutdown_signal TERM'")[0]
    )
    source = source.replace("{% raw %}", "")
    # Nonleader /proc/TID paths exist, but aren't listed by real /proc readdir.
    source = source.replace("/proc/[0-9]*", '"${FAKE_CANDIDATES[@]}"')
    source = source.replace("/proc/", f"{proc}/")
    (tmp_path / "functions.sh").write_text(source)
    prelude = r"""
source "$CASE/functions.sh"
CLEANUP_LOG="$CASE/cleanup.log"
CLEANUP_HOST=fixture
LOGDIR="$CASE"
SLURM_JOB_ID=23 SLURM_PROCID=0 SLURM_STEP_ID=1
TRAIN_PID=100 CLEANUP_TRAIN_START=10 CLEANUP_CGROUP=
PROCESS_SUPERVISOR_GRACE_S=0 PROCESS_SHUTDOWN_GRACE_S=2 PROCESS_SHUTDOWN_KILL_WAIT_S=2
FAKE_CANDIDATES=("$CASE/proc/100" "$CASE/proc/101" "$CASE/proc/900")
kill() {
    [[ "$1" == -s ]] || return 99
    local sig=$2 target=${!#}
    case "$target" in 100|101|102|-100) ;; *) echo "Unexpected target: $target" >&2; return 99 ;; esac
    printf '%s %s\n' "$sig" "$target" >> "$CASE/signals"
    if [[ "$sig" == KILL || "${MODE:-}" == term ]]; then
        rm -rf "$CASE/proc/100"
        if [[ "${MODE:-}" == dstate ]]; then
            cp "$CASE/zombie.stat" "$CASE/proc/101/stat"
            rm -f "$CASE/proc/101/environ" "$CASE/proc/102/environ"
        else
            rm -rf "$CASE/proc/101" "$CASE/proc/102"
        fi
    fi
}
sleep() { SECONDS=$(( SECONDS + $1 )); }
timeout() {
    shift 2
    if [[ "${PS_FAIL:-0}" == 1 ]]; then return 137; fi
    case "$1" in bash|ps|head) "$@" ;; *) return 99 ;; esac
}
SECONDS=0
"""

    def run(body, **env):
        # No subprocess timeout: neither this runner nor its stubs send signals.
        result = subprocess.run(
            ["bash", "-c", prelude + "\n" + body],
            env={
                **os.environ,
                "CASE": str(tmp_path),
                "PATH": f"{bindir}:/usr/bin:/bin",
                **env,
            },
            text=True,
            capture_output=True,
        )
        assert result.returncode == 0, result.stderr + result.stdout
        return result.stdout, (tmp_path / "cleanup.log").read_text() if (
            tmp_path / "cleanup.log"
        ).exists() else ""

    return run, proc, tmp_path


def test_term_completion(harness):
    run, _, _ = harness
    _, log = run('cleanup\n[[ "${CLEANUP_FAILED:-0}" == 0 ]]', MODE="term")
    assert "phase=term-exit " in log
    assert "phase=final_all snapshot_finished=1" in log


def test_kill_completion(harness):
    run, _, _ = harness
    _, log = run('cleanup\n[[ "${CLEANUP_FAILED:-0}" == 0 ]]')
    assert "phase=kill-exit " in log


def test_d_thread_survives_leader_and_environment_loss(harness):
    run, proc, _ = harness
    (proc / "102/stat").write_text(stat(102, state="D"))
    _, log = run('cleanup\n[[ "$CLEANUP_FAILED" == 1 ]]', MODE="dstate")
    assert "phase=kill-timeout alive_tasks=1" in log
    assert "pid=101 tid=102" in log
    assert "state=D" in log
    assert "driver_wait" in log


def test_reused_pid_is_not_signalled(harness):
    run, proc, case = harness
    (proc / "100/stat").write_text(stat(100, start=99))
    run(
        'CLEANUP_OWNER_START[100]=10; CLEANUP_TASK_START[100]=10; CLEANUP_TASK_OWNER[100]=100\nsignal_cleanup_targets KILL 10\nprobe_cleanup_targets 10\n[[ "$CLEANUP_ALIVE" == 0 ]]'
    )
    assert not (case / "signals").exists()


def test_known_unreadable_task_is_not_gone(harness):
    run, proc, _ = harness
    (proc / "100/stat").unlink()
    run(
        'CLEANUP_TASK_START[100]=10; CLEANUP_TASK_OWNER[100]=100\nprobe_cleanup_targets 10\n[[ "$CLEANUP_UNREADABLE" == 1 && "${#CLEANUP_TASK_START[@]}" == 1 ]]'
    )


def test_failed_ps_preserves_fallback(harness):
    run, _, _ = harness
    _, log = run(
        'build_cleanup_candidates 2\n[[ "${#CLEANUP_SCAN_PDIRS[@]}" == 4 ]]',
        PS_FAIL="1",
    )
    assert "pipeline_exit_code=137" in log


def test_incomplete_discovery_is_not_application_failure(harness):
    run, _, _ = harness
    _, log = run(
        'CLEANUP_TRAIN_START=\ndiscover_cleanup_targets() { CLEANUP_DISCOVERY_COMPLETE=0; }\ncleanup\n[[ "${CLEANUP_FAILED:-0}" == 0 ]]'
    )
    assert "phase=cleanup-unverified " in log


def test_expired_probe_keeps_unchecked_identity(harness):
    run, _, _ = harness
    run(
        'CLEANUP_TASK_START[100]=10; CLEANUP_TASK_OWNER[100]=100\nprobe_cleanup_targets 0\n[[ "$CLEANUP_PROBE_COMPLETE" == 0 && "${#CLEANUP_TASK_START[@]}" == 1 ]]'
    )


def test_diagnostics_do_not_source_bash_env(harness):
    run, _, case = harness
    (case / "startup.sh").write_text(
        'printf "startup-called\\n" >> "$CASE/startup-called"\n'
    )
    _, log = run("""export BASH_ENV="$CASE/startup.sh"
build_cleanup_candidates 2
build_cleanup_candidates 2
log_all_processes
[[ "$BASH_ENV" == "$CASE/startup.sh" ]]
""")
    assert not (case / "startup-called").exists()
    assert log.count("pipeline_exit_code=0") == 2
    assert "ps_exit_code=0 scan_complete=1" in log


def test_kill_reaches_owner_when_leader_became_zombie_before_kill(harness):
    run, proc, case = harness
    (proc / "102/stat").write_text(stat(102, state="D"))
    _, log = run(
        """discover_cleanup_targets 10
cp "$CASE/zombie.stat" "$CASE/proc/101/stat"
rm -f "$CASE/proc/101/environ" "$CASE/proc/102/environ"
probe_cleanup_targets 10
signal_cleanup_targets KILL 10
signal_cleanup_targets KILL 10
""",
        MODE="dstate",
    )
    assert (case / "signals").read_text().splitlines().count("KILL 101") == 1
    assert "pid=101 tid=102 start_ticks=10 signal=KILL result=sent" in log


@pytest.mark.parametrize("changed_pid", [101, 102])
def test_zombie_owner_or_thread_identity_change_prevents_kill(harness, changed_pid):
    run, proc, case = harness
    (proc / "102/stat").write_text(stat(102, state="D"))
    (case / "reused.stat").write_text(stat(changed_pid, start=99))
    _, log = run(f"""CLEANUP_OWNER_START[101]=10
CLEANUP_TASK_START[102]=10 CLEANUP_TASK_OWNER[102]=101
cp "$CASE/zombie.stat" "$CASE/proc/101/stat"
cp "$CASE/reused.stat" "$CASE/proc/{changed_pid}/stat"
signal_cleanup_targets KILL 10
""")
    assert not (case / "signals").exists()
    assert "result=skipped reason=" in log
    assert "identity-changed" in log


@pytest.mark.parametrize(
    "stack_kind,expected",
    [
        ("empty", "stack=empty"),
        ("missing", "stack=error exit_code=1"),
        ("read_error", "stack=error exit_code=1"),
        ("timeout", "stack=timeout exit_code=137"),
    ],
)
def test_stack_diagnostic_reports_failure_instead_of_empty(
    harness, stack_kind, expected
):
    run, proc, _ = harness
    (proc / "102/stat").write_text(stat(102, state="D"))
    if stack_kind == "empty":
        (proc / "102/stack").write_text("")
    elif stack_kind == "missing":
        (proc / "102/stack").unlink()
    body = """CLEANUP_TASK_START[102]=10 CLEANUP_TASK_OWNER[102]=101
CLEANUP_D_TIDS=(102) CLEANUP_PROBE_COMPLETE=1
"""
    if stack_kind == "read_error":
        body += "head() { echo 'head: error reading stack: Permission denied' >&2; return 1; }\n"
    elif stack_kind == "timeout":
        body += "timeout() { return 137; }\n"
    _, log = run(body + "log_d_state final_d 10")
    assert expected in log
    if stack_kind == "read_error":
        assert "Permission denied" in log
    elif stack_kind == "missing":
        assert "No such file" in log


@pytest.mark.parametrize("trigger", ["TERM", "INT", "application-exit"])
def test_cleanup_records_trigger_and_repeated_signals(harness, trigger):
    run, _, _ = harness
    command = (
        "cleanup"
        if trigger == "application-exit"
        else f"handle_shutdown_signal {trigger}"
    )
    _, log = run(command + "\nhandle_shutdown_signal TERM", MODE="term")
    assert f"phase=start trigger={trigger} training_pid=100" in log
    assert log.count("phase=start ") == 1
    assert "phase=signal-received signal=TERM cleanup_started=1" in log


def test_training_group_signal_is_sent_once_per_level(harness):
    run, _, case = harness
    # Supervisor and TERM stages both run; only the first may signal the group.
    _, log = run("PROCESS_SUPERVISOR_GRACE_S=4\nhandle_shutdown_signal TERM")
    signals = (case / "signals").read_text().splitlines()
    assert signals.count("TERM -100") == 1
    assert signals.count("KILL -100") == 1
    assert "training_group=already-sent" in log


def test_group_leader_is_not_signalled_individually(harness):
    run, _, case = harness
    run("PROCESS_SUPERVISOR_GRACE_S=4\nhandle_shutdown_signal TERM")
    signals = (case / "signals").read_text().splitlines()
    assert "TERM -100" in signals
    assert "TERM 100" not in signals  # already covered by the group signal
    assert "TERM 101" in signals  # other targets are still signalled


def test_group_leader_is_signalled_when_the_group_signal_did_not_go_out(harness):
    run, _, case = harness
    run(
        "CLEANUP_OWNER_START[100]=10; CLEANUP_TASK_START[100]=10; CLEANUP_TASK_OWNER[100]=100\n"
        "signal_cleanup_targets TERM 10"
    )
    assert (case / "signals").read_text().splitlines() == ["TERM 100"]
