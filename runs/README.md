# Run evidence

- [Compile comparison](compile-baseline/report.md): the selected baseline with
  and without compile, rank-0 trace accounting, and custom-kernel candidates.
- [GPU capacity selector](capacity-selector/report.md): the selected production
  stack, isolated EP8 checks, native 10+10 comparison, and matched Perfetto traces.
- [Current crop smoke](100b-crop8-ep8-smoke/report.md): maintained launcher,
  source/config identities, all-rank execution result and limits.
- [CuTe stats-only SAC follow-up](fusion-followup/report.md): the selected
  production replay, the one-shot eight-rank measurement, and experimental
  SwiGLU row16 status.
- [Historical CUDA Graph evidence](cuda-graphs-evidence/README.md): retained
  numerical summaries and trace assessment; full history has a Git archive locator.

For new experiments use [the planning template](../planning/README.md) and
[validation protocol](../planning/validation.md). Keep one report and a compact
machine result per experiment. Store large raw data outside Git with hashes and
explicit evidence locations. Do not create a document chain for every command.
