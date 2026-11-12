# Preparation 002 transfer manifest

Source: `NUMERICAL-NATIVE-PROBE/preparation/001`. Target: `NUMERICAL-NATIVE-PROBE/preparation/002`.
Eleven task-local release artifacts, the release template, and the README were copied. The twelfth release artifact, the canonical adapter, remains at its existing LOSS-ONLY-PERF path. No runtime logs, preflight receipts, config-check results, attempt output, or checkpoint were copied.

## Mechanical changes

- Preparation scope and artifact paths move from `preparation/001` to `preparation/002`; the release template remains unapproved and awaits the next root-provided source pin.
- Native attempt paths and preflight absence checks move from `NUMERICAL-NATIVE-PROBE/attempts/001` to `.../attempts/002`. Canonical `LOSS-ONLY-PERF/attempts/001` and offline cache `NATIVE-SMOKE/001/cache` paths stay unchanged.
- `launch.sh` defines the task-owned `attempts/` parent and creates it only after both config checks and the post-check no-checkpoint checks. It rechecks that child `attempts/002` is absent, then claims it with a single `mkdir`.
- README records the prior attempt-001 outcome and points to the immutable earlier missing-parent failure evidence. The Source 009 complex64 rejection is a source compatibility issue owned elsewhere, not a launcher failure.
- Production workload, config overlays, compiler adapters, process owner, and topology/software helpers are carried byte-for-byte; prior CPU/config/software evidence is reused.

## The 12 release artifacts

| Artifact path in release | SHA-256 from 001 | SHA-256 for 002 |
|---|---|---|
| `runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/LOSS-ONLY-PERF/attempts/001/compiler_entry.py` | `ffde3588d53012c7155fb5b6961a06be96095b44b8bc55c97a9808b2446ba065` | `ffde3588d53012c7155fb5b6961a06be96095b44b8bc55c97a9808b2446ba065` |
| `runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/NUMERICAL-NATIVE-PROBE/preparation/002/compiler_entry_create_s0.py` | `ffde3588d53012c7155fb5b6961a06be96095b44b8bc55c97a9808b2446ba065` | `ffde3588d53012c7155fb5b6961a06be96095b44b8bc55c97a9808b2446ba065` |
| `runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/NUMERICAL-NATIVE-PROBE/preparation/002/compiler_entry_observe.py` | `ffde3588d53012c7155fb5b6961a06be96095b44b8bc55c97a9808b2446ba065` | `ffde3588d53012c7155fb5b6961a06be96095b44b8bc55c97a9808b2446ba065` |
| `runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/NUMERICAL-NATIVE-PROBE/preparation/002/config/create_s0.yaml` | `450b8ffc8d50a6a8864aadc7fa61e0c7bbc4259b7bb7b71bb0cbe4cb3fbe4116` | `450b8ffc8d50a6a8864aadc7fa61e0c7bbc4259b7bb7b71bb0cbe4cb3fbe4116` |
| `runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/NUMERICAL-NATIVE-PROBE/preparation/002/config/observe.yaml` | `515c1d0524edb34a4736d655b52363c93363b778c9dcc6dc7ea1641c3395a7cf` | `515c1d0524edb34a4736d655b52363c93363b778c9dcc6dc7ea1641c3395a7cf` |
| `runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/NUMERICAL-NATIVE-PROBE/preparation/002/launch.sh` | `a337efada518eb0eff2b534e358d13922d2e83fd5910dab155dd740b57510006` | `0d35664f696bf6c4741bca13cac247520b9f52e1874b244112ab221431fb8043` |
| `runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/NUMERICAL-NATIVE-PROBE/preparation/002/phase_process_owner.py` | `04d47459b9f24e7568f4b14d224673c862eb9a0b7f624650ad35328796d874ba` | `04d47459b9f24e7568f4b14d224673c862eb9a0b7f624650ad35328796d874ba` |
| `runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/NUMERICAL-NATIVE-PROBE/preparation/002/preflight.sh` | `ba61cb44f2089296d1768d09425587f229798784942d9505460ef4d8fe0a1ded` | `80fdc3bbccb3a973e2548186e76a5dbce6fcc09bbe40809cbe7557958f8af174` |
| `runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/NUMERICAL-NATIVE-PROBE/preparation/002/preflight_software.py` | `6d11f14494a1f7b5b0167dad1a25caf3c3c896bdf3e8685d858e601ecb920e38` | `6d11f14494a1f7b5b0167dad1a25caf3c3c896bdf3e8685d858e601ecb920e38` |
| `runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/NUMERICAL-NATIVE-PROBE/preparation/002/preflight_topology.py` | `e20153ee96c5665d79fa95fef6b2e2b1e8e545b592bbce7cb1110c491e5ce330` | `e20153ee96c5665d79fa95fef6b2e2b1e8e545b592bbce7cb1110c491e5ce330` |
| `runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/NUMERICAL-NATIVE-PROBE/preparation/002/test_phase_process_owner.py` | `30cdb5e9b660a6de5787bf3e7d97d8ead6174c6b5d58d577c25b0911fd0d25ee` | `30cdb5e9b660a6de5787bf3e7d97d8ead6174c6b5d58d577c25b0911fd0d25ee` |
| `runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/NUMERICAL-NATIVE-PROBE/preparation/002/verify_release.py` | `a14715d4b014a64ea908a4ea451877f5f55832c6a456008b81257d303ffd6eeb` | `c6afa025dae432e01f51ad31524a6e4a94fafdbe99ba3cd285a9231d2a7a1e51` |

## Handoff documents

| File | SHA-256 from 001 | SHA-256 for 002 |
|---|---|---|
| `README.md` | `a6583911593891189824659e06efbc3b6aca5d320c599fce49ea5db2e0cb86af` | `e938e92e397a34b1cf51f3fe2ca77d67d8e7b962651e3937a964f73e327a4e03` |
| `root-release.template.json` | `d8a610a868f96b4db76e25be62bd8911963192fdd748065b183988e52af5cf1d` | `5173d87817d460074d22bb26e1cb79b83f63f9e10b89bdb5960ae40cc99e1cbc` |

The canonical loss-only adapter is a release dependency outside this preparation and remains at its existing path. Both copied compiler entries are byte-identical to it at SHA-256 `ffde3588d53012c7155fb5b6961a06be96095b44b8bc55c97a9808b2446ba065`.
