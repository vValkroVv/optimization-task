# C1 initial-gate diagnosis

## Scope and result

Acceptance: identify a concrete first-gate defect using small records and active
source; prove it through the actual comparator; specify a minimal fix, regression
and provenance-safe reference reuse decision. No source edit, GPU operation,
relaunch, installation or publication was performed. No reference tensor payload
was read or rehashed.

The live mesh coordinate is a tuple; the saved coordinate is a JSON list.
The exact comparator rejects this representational difference before any update.
This defect necessarily rejects all 243 initial parameters on each rank, even
when every value and hash agrees. It does not prove that the failed B run's
unpersisted parameter hashes agreed with A.

## Evidence

- `numerics.py:_mesh_metadata` stores `mesh.get_coordinate()` directly at line
  1663. The installed runtime's
  `/opt/torchtitan-venv/lib/python3.12/site-packages/torch/distributed/device_mesh.py:1278`
  returns `tuple[int, ...] | None`, returning `_coordinate_on_dim` unchanged.
- `numerics_payload.py:_same_json` first rejects differing Python types.
  `verify_initial_gates` compares loaded reference metadata against live metadata.
  JSON publication has converted the reference tuple into a list.
- All eight actual A manifests have 243 initial parameters and 243 list-valued
  mesh coordinates each. Empty local parameter counts by rank are
  `[210, 212, 182, 194, 197, 217, 100, 207]`. Empty shards also fail because
  coordinate representation differs; their empty SHA-256 cannot establish
  initialization noise.
- B persisted only identity/count failure receipts, not live records or hashes.
  The exact count is 243 on every rank. Other initial phases did not appear in
  that mismatch count. Their RNG/optimizer/scheduler/counter gates must remain
  exact; no global RNG exemption is justified.
- A's rank-0 initial inventory has 523 records: 243 parameters, 243 optimizer
  state-key records, 15 buffers, 14 router records, four RNG records, two counters,
  one optimizer group and one scheduler record.

Inspected adapter SHA-256:
`numerics.py=6c3842353626262cec6ab3623c83475595397c083e473b8b89463a3c467a24bf`;
`numerics_payload.py=d6afbab28d04204f15cf19264f3f024cbbbb75174e9610fd99d3328da085578b`.
Installed `device_mesh.py` SHA-256:
`673bca57b78b606d5d4a30e7e0b1cc8e77fdca673a76a87ac42d0e2c808111e4`.

## CPU behavior proof and regression contract

An inline Python check imported the actual `numerics`, `numerics_payload` and
installed `DeviceMesh`. It called actual `DeviceMesh.get_coordinate` on a
minimal object holding `(0,)`, then actual `_mesh_metadata` with an equivalent
mesh/placement adapter. The fixture was the actual rank-0 empty `A_log` record.
Only its mesh field differed; replacing the coordinate tuple with a list made
the entire mesh equal. Actual `verify_initial_gates` rejected the tuple record
with the native failure message, and accepted the coordinate-normalized record.
Changing SHA-256, dtype or ownership independently still failed. Exit code 0;
no GPU or process group was created. This isolates metadata representation; it
does not reproduce the entire GPU capture or recover B's absent hashes.

Minimal fix: normalize the coordinate at its producer to
`None if coordinate is None else list(coordinate)` inside `_mesh_metadata`.
Keep `_same_json` strict. This also fixes selected-tensor metadata validation,
which uses the same producer and would otherwise encounter the same defect.
Do not limit the fix to the initial gate or normalize arbitrary metadata types.

Permanent regression should call the actual producer with tuple coordinates,
JSON round-trip its record, and call actual initial comparison: old producer
fails, fixed producer passes. Cover `None` and multiaxis coordinates, selected
tensor metadata validation, and independent changed hash/dtype/shard/owner
rejections. The narrow inline proof above has not added these permanent tests
or executed a fixed source implementation.

## Runtime cost and immutable A reuse

The proposed change allocates a short Python coordinate list per metadata
observation. It adds no tensor operation, D2H transfer, device synchronization,
collective, compilation change or training arithmetic. Existing diagnostic
tensor copies remain unchanged. No speed claim or new GPU benchmark is needed
to assess this metadata-only fix; actual native gates still need execution.

A remains a valid original export under its original adapter bytes and schema.
Serialized coordinates from old and fixed producers are byte-equivalent JSON;
payload byte order, hashes, coverage and model mathematics are unaffected.
Re-export is therefore not required by the data format itself.

However, reuse is not silently authorized by matching training configuration.
`run.py:source_snapshot` restricts tracked/untracked manifests and Git status
to `gift/` (lines 170–173 and 198); its `head` separately records task HEAD
(line 200). `run.py:_config_hashes` separately records adapter hashes. Thus a
helper-only fix preserves GIFT content hashes while changing the helper hash
and, after commit, task HEAD.
`parse_results.py:_native_reference_summary` verifies helper hashes before/after
each run; it does not establish cross-revision adapter compatibility. The report
requires matching source/runtime/config/data before comparison. Fixed B will
have a different adapter hash, and its task HEAD will differ after commit;
unchanged GIFT content identity is independently verifiable.

An honest reuse route needs a reviewed, explicit compatible-adapter decision in
the existing campaign record: preserve A's original revision and hashes; record
fixed B's actual revision/hashes; identify the sole coordinate normalization
delta; prove serialized metadata/schema and captured mathematics unchanged;
verify GIFT/config/runtime/data identities independently. Keep A's manifest and
payloads immutable, retaining the original failed B receipt. This is a documented
adapter compatibility exception, not a claim of identical helper hashes. If the
campaign requires identical task HEAD and adapter hashes with no such decision,
re-export A under the fixed adapter instead.

Either route requires a fresh B to pass every exact initial hash/state gate and
every prefix-input gate before consuming payloads. The failed B cannot be
retroactively passed; actual A/A metrics, C2 and monitor50 remain open. Preserve
the first-capture format/provenance history and all native preset settings.
