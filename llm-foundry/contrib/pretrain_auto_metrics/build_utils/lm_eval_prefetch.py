import argparse
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import batched


TAIL_LINES = 100

# Задачи вроде MRCR отказываются строить запросы без чат-шаблона, которого у
# модели dummy нет. Для прогрева достаточно сконструировать задачу: датасет
# скачивается в конструкторе.
DOWNLOAD_ONLY_CODE = """
import json, sys

from lm_eval.tasks import TaskManager, get_task_dict

tasks = sys.argv[1].split(",")
metadata = json.loads(sys.argv[2]) if sys.argv[2] else None
get_task_dict(tasks, TaskManager(metadata=metadata))
"""


def report(label: str, attempt: str, p: subprocess.CompletedProcess) -> None:
    if p.returncode == 0:
        print(f"[prefetch ok] {attempt} {label}", flush=True)
        return
    tail = (p.stdout or "").splitlines()[-TAIL_LINES:]
    print(
        f"[prefetch FAILED rc={p.returncode}] {attempt} {label}\n"
        + "\n".join(tail)
        + f"\n[prefetch FAILED rc={p.returncode}] {attempt} {label} (end of output)",
        flush=True,
    )


def run_lm_eval(
    pybin: str,
    tasks: list[str],
    include_path: str | None,
    metadata: str | None,
    attempt: str,
    download_only: bool = False,
) -> subprocess.CompletedProcess:
    if download_only:
        return run_download_only(pybin, tasks, metadata, attempt)
    cmd = [
        pybin,
        "-m",
        "lm_eval",
        "--model",
        "dummy",
        "--tasks",
        ",".join(tasks),
        "--num_fewshot",
        "0",
        "--limit",
        "1",
        "--trust_remote_code",
        "--confirm_run_unsafe_code",
    ]
    if include_path:
        cmd.extend(["--include_path", include_path])
    if metadata:
        cmd.extend(["--metadata", metadata])
    label = ",".join(tasks)
    print(f"[prefetch run] {attempt} {label}", flush=True)
    p = subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    report(label, attempt, p)
    return p


def run_download_only(
    pybin: str,
    tasks: list[str],
    metadata: str | None,
    attempt: str,
) -> subprocess.CompletedProcess:
    label = ",".join(tasks)
    print(f"[prefetch run] {attempt} {label} (download only)", flush=True)
    p = subprocess.run(
        [pybin, "-c", DOWNLOAD_ONLY_CODE, label, metadata or ""],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    report(label, attempt, p)
    return p


def run_chunk(
    pybin: str,
    task_chunk: list[str],
    retries: int,
    include_path: str | None = None,
    metadata: str | None = None,
    download_only: bool = False,
) -> None:
    last = None
    for i in range(retries + 1):
        p = run_lm_eval(
            pybin,
            task_chunk,
            include_path,
            metadata,
            f"try {i + 1}/{retries + 1}",
            download_only,
        )
        if p.returncode == 0:
            return
        last = p.returncode
        time.sleep(2)

    if len(task_chunk) == 1:
        raise RuntimeError(f"task {task_chunk[0]} failed rc={last}")

    # re-run each task alone to name the one that breaks the chunk
    failures = [
        task
        for task in task_chunk
        if run_lm_eval(
            pybin, [task], include_path, metadata, "solo", download_only
        ).returncode
        != 0
    ]
    if not failures:
        raise RuntimeError(
            f"chunk failed rc={last} but every task passed alone: {task_chunk}"
        )
    raise RuntimeError(f"tasks failed: {failures}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python-bin", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--chunk-size", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--include-path", default=None)
    ap.add_argument("--metadata", default=None)
    ap.add_argument("--download-only", action="store_true")
    args = ap.parse_args()

    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be >= 1")
    if args.jobs < 1:
        raise ValueError("--jobs must be >= 1")
    if args.retries < 0:
        raise ValueError("--retries must be >= 0")

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    task_chunks = [list(chunk) for chunk in batched(tasks, args.chunk_size)]
    if not task_chunks:
        return

    with ThreadPoolExecutor(max_workers=min(args.jobs, len(task_chunks))) as ex:
        futs = {
            ex.submit(
                run_chunk,
                args.python_bin,
                task_chunk,
                args.retries,
                args.include_path,
                args.metadata,
                args.download_only,
            ): task_chunk
            for task_chunk in task_chunks
        }
        failures = []
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:
                failures.append(f"{','.join(futs[f])}: {e}")

    if failures:
        raise SystemExit("prefetch failed:\n" + "\n".join(f"  {x}" for x in failures))


if __name__ == "__main__":
    main()
