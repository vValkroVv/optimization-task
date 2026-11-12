# Reproduction and command record

Existing run command lines are `../../attempts/006/command.txt` through `010/command.txt`.
Frozen launch scripts, accepted owner/controller, per-attempt releases, launch PID/start-tick
receipts, exact argv, eight UUID availability and native resolved configuration are retained
beside each attempt. No new allocation, install, remote command, source edit or launcher edit.

From repository root, existing evidence is extracted with:

```bash
python3 runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/PERFORMANCE-ASSESSMENT/analysis/001/extract.py 006
python3 runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/PERFORMANCE-ASSESSMENT/analysis/001/memory_setup.py 006
```

Repeat extraction with each completed attempt ID. `extract.py` checks source/launcher freezes,
config, logs, counts, token validity, compiler options, native rank-zero metrics and memory;
it does not initialize CUDA. `memory_setup.py` joins existing NVML CSV through recorded UUIDs
and actual UTC timing brackets. It reports sample gaps and boundary exclusions.

The root-authorized successors were released one at a time using `release_next.py ID`.
This script checks the previous attempt, unchanged source/launch bytes and exact eight idle
UUIDs before writing a fresh release and detached controller receipt. **This launches a GPU
run; do not rerun it for offline reproduction.** The immutable controller is
`../../preparation/002/controller.py`, run with `/opt/torchtitan-venv/bin/python3.12`,
`EXPECTED_SOURCE_MANIFEST_SHA256=b2e9144af50e8673e0b8646a4f03f629bc240e6fc3ed1aaccfe6814491fb7d20`,
exclusive controller log and `start_new_session=True`.

Only one bounded local monitor per active attempt checks `controller-complete.json` every10s,
with1900s deadline. Accepted process owner has1800s deadline and owns exact-identity cleanup.
No diagnostic retry or further variant is permitted after a CUDA/actual validity failure.
