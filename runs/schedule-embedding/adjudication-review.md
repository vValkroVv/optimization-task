# Offline timing adjudication review

## Verdict

Bounded PASS for the offline recovery route and the completed A attempt.
No material finding remains from this review. This verdict does not validate
unfinished variants or permit the four diagnostics to bypass their seven-timing
gate. Original runner status remains FAIL; only the new record is
`ADJUDICATED_PASS`.

Reviewed implementation SHA-256:
`fe26edcd1ca65591894454f93bff4c195c394f7185a90f58a50c78c4efd24fec`
([adjudicate.py](adjudicate.py)). Reviewed source/configuration/harness freeze:
`3b3244ae1e759111b8240af929715074b751c946881202a9ff95e9f3b9918ffd`.

## Finite acceptance and inspection

The scope was one review of the original post-native runner and campaign
acceptance checks, the single nullable-field exception, original-evidence
preservation, actual A reconstruction, and targeted negative contracts. No
training-source, frozen harness, raw attempt, compiler cache, or GPU work was
changed by this review.

- The original runner requires native exit 0, no timeout, clean/stopped owned
  process group, unchanged source/configuration, complete parser output, valid
  startup receipts, and resolved configuration/path/input checks. The helper
  reconstructs these checks from retained evidence. It calls the unchanged
  base `_verify_resolved_config` for its full original configuration, callback,
  tokenizer, data, and output-path checks.
- The campaign adds exact recipe controls, source/bridge hash guards, cache
  producer/path, all-rank cadence and stage selection, original-capacity
  bootstrap receipts, exact timing/token window, and memory-counter coverage.
  The helper retains these acceptance checks. It uses the actual frozen
  `parse_results.summarize_run(attempt)` on the original attempt.
- Only absent `parallelism.gfsdp_row_owner_embedding_optimization_stages`
  with expected `None` is waived. Its typed nullable default is established
  from the actual config class; explicit child `null`, native typed decode,
  and all eight module-stage receipts are required. The pinned serializer
  `gift/torchtitan/config/save.py:get_typed_document` explicitly strips `None`;
  `gift/torchtitan/config/configs.py:ParallelismConfig` defines this field with
  default `None`. Missing non-null controls and wrong explicit selectors still
  fail. SE requires three stages and SE-fused requires four; neither uses the
  nullable exception.
- The helper records the exact original guard failure, wrapper/native statuses,
  raw receipt hashes, parser-summary hash, and freeze in a new output directory.
  It refuses existing output directories. It does not patch the parser, copy
  normalized raw receipts, fabricate original completion, or change raw status.
- This is offline CPU/file work. It introduces no production computation,
  device allocation, synchronization, copy, communication, or autograd change.
  Existing source and normal stage-selection execution remain frozen.

## Independently verified actual A

Original attempt:
[raw A](artifacts/raw/schedule-embedding-a-timing-140-003/).
New result:
[A recovery](artifacts/raw/adjudication-140-003/A-recovery-003/adjudication.json).

- All 43 listed original receipt hashes agree with the actual raw files.
- Original wrapper FAIL/exit 1 and exact nullable-field failure remain present;
  native child exit is 0, timeout false, orphan false, process group stopped.
- Before/final source snapshots and configuration hash dictionaries agree
  exactly. Recomputing the current combined freeze matched the pinned freeze.
- All eight raw ranks completed 140 optimizer updates with 140 cadence records;
  all eight actual embedding-stage receipts contain zero modules for A.
- The frozen parser independently returned `complete`, `errors: []`, updates
  41–140 after 40 warmups, 209,715,200 global physical tokens and 209,664,000
  supervised tokens. The global envelope is 1401.49943926 seconds. Parsed
  output agrees with the stored new summary, including all-rank memory data.
- Calling the helper's full `_validate` read-only on actual A agreed with its
  stored result after ordinary JSON integer-key normalization.

New summary SHA-256:
`25d28a7cbeda3e009b4a71aa9af077927f5a8d08e849c16c483a99b1d02f43e7`.
New adjudication SHA-256:
`985633e081e35c57b2c1251e36f7ee5031acfd36897800d5a14bc4d89ba6dcb3`.
Original failure SHA-256:
`60c65c050baec46dcaffc8a00645b1e5bf0328c046e44a8b4feea46a3c5f75bf`.

## Executed checks and limits

`CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /opt/torchtitan-venv/bin/python3.12 -m pytest -q -p no:cacheprovider runs/schedule-embedding/test_adjudicate.py`
passed: **8 tests**. They cover old strict-guard rejection, narrow nullable
acceptance, wrong/missing required controls, incomplete nullable evidence,
missing rank stages, incomplete window, source/configuration mismatch, input/
cache mismatch, and wrong original-capacity overlay.

Two review-harness issues were resolved without repository changes: the first
standalone parser import needed its existing helper directory on `sys.path`;
an initial in-memory/stored-result comparison needed JSON normalization for
integer rank keys. Neither was a failed acceptance check in the adjudicator.

The review uses completed A evidence plus targeted CPU contract checks; it
adds no new GPU, NCCL, numerical, convergence, or performance validation. Each
remaining timing must receive its own complete adjudication. All seven valid
adjudications are required before the separately authorized update-21 profiles;
the original campaign's raw-PASS gate cannot accept recovered FAIL attempts
without an explicit reviewed integration decision.
