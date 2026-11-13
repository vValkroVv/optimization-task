from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numerics_payload
from numerics_payload import NativeNumericalPayload


_COVERAGE = {
    "1": {
        "input_batches": 32,
        "microbatches": 32,
        "summary_calls": 1,
        "clip_calls": 1,
        "optimizer_calls": 1,
        "batch_end_calls": 1,
    }
}


def _tensor_record(tensor: torch.Tensor) -> dict[str, object]:
    contiguous = tensor.detach().to(device="cpu").contiguous()
    raw = contiguous.reshape(-1).view(torch.uint8).numpy().tobytes()
    floating = tensor.is_floating_point() or tensor.is_complex()
    work = tensor.to(torch.complex128 if tensor.is_complex() else torch.float64)
    return {
        "update": 1,
        "phase": "test_tensor",
        "name": "model.test.weight",
        "microbatch": None,
        "rank": 0,
        "present": True,
        "dtype": str(tensor.dtype),
        "source_device_type": "cpu",
        "global_shape": list(tensor.shape),
        "local_shape": list(tensor.shape),
        "local_stride": list(contiguous.stride()),
        "local_numel": tensor.numel(),
        "mesh": None,
        "placements": [],
        "owner": {"part_index": 0, "canonical_fqn": "test.weight"},
        "aliases": [],
        "sha256": hashlib.sha256(raw).hexdigest(),
        "is_finite": bool(torch.isfinite(work).all().item()) if floating else None,
    }


def _prefix_records() -> list[dict[str, object]]:
    records = []
    for microbatch in range(32):
        for name in ("input.tokens", "labels"):
            tensor = torch.tensor([microbatch], dtype=torch.int64)
            record = _tensor_record(tensor)
            record.update(
                update=1,
                phase="dataloader_input",
                name=name,
                microbatch=microbatch,
                owner=None,
            )
            records.append(record)
    return records


def _export_reference(
    tmp_path: Path,
    tensor: torch.Tensor,
    *,
    prefix: list[dict[str, object]] | None = None,
) -> tuple[Path, dict[str, object], list[dict[str, object]]]:
    root = tmp_path / "reference"
    root.mkdir(parents=True)
    payload = NativeNumericalPayload(
        mode="export",
        root=root,
        rank=0,
        capture_update=1,
        run_id="payload-test",
    )
    record = _tensor_record(tensor)
    inputs = _prefix_records() if prefix is None else prefix
    payload.verify_initial_gates([], {})
    payload.verify_prefix_inputs(inputs)
    payload.observe_tensor(record, tensor.detach().cpu().contiguous())
    payload.finish(
        [*inputs, record],
        coverage=_COVERAGE,
        training_metadata={},
        monitor_attestation=None,
    )
    return root, record, inputs


def _compare(
    tmp_path: Path,
    root: Path,
    record: dict[str, object],
    tensor: torch.Tensor,
    inputs: list[dict[str, object]],
) -> tuple[NativeNumericalPayload, dict[str, object]]:
    payload = NativeNumericalPayload(
        mode="compare",
        root=root,
        rank=0,
        capture_update=1,
        run_id="payload-compare-test",
        comparison_root=tmp_path / "comparison",
    )
    payload.verify_initial_gates([], {})
    payload.verify_prefix_inputs(inputs)
    payload.validate_tensor_metadata(record)
    payload.observe_tensor(record, tensor.detach().cpu().contiguous())
    result = payload.finish(
        [*inputs, record],
        coverage=_COVERAGE,
        training_metadata={},
        monitor_attestation=None,
    )
    return payload, result


def _comparison_metrics(result: dict[str, object]) -> dict[str, object]:
    manifest = json.loads(Path(result["manifest"]).read_text())
    row = next(item for item in manifest["observations"] if item["key"]["phase"] == "test_tensor")
    return json.loads((Path(result["manifest"]).parent / row["result"]).read_text())["metrics"]


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2]
)
def test_original_low_precision_dtype_roundtrips_without_cast(
    dtype: torch.dtype, tmp_path: Path
) -> None:
    reference = torch.tensor([0.0, 1.5, -2.0], dtype=dtype)
    root, _, inputs = _export_reference(tmp_path, reference)
    _, result = _compare(
        tmp_path,
        root,
        _tensor_record(reference),
        reference,
        inputs,
    )
    metrics = _comparison_metrics(result)
    assert metrics["exact_element_match_count"] == 3
    assert metrics["exact_hash_match"] is True
    assert metrics["max_abs_error"] == 0.0


def test_streamed_metrics_cover_exact_nonzero_zero_empty_and_nonfinite(tmp_path: Path) -> None:
    reference = torch.tensor([1.0, 2.0, -4.0], dtype=torch.float32)
    current = torch.tensor([1.0, 2.25, -4.0], dtype=torch.float32)
    root, _, inputs = _export_reference(tmp_path / "nonzero", reference)
    current_record = _tensor_record(current)
    _, result = _compare(tmp_path / "nonzero", root, current_record, current, inputs)
    metrics = _comparison_metrics(result)
    assert metrics["count"] == 3
    assert metrics["exact_element_match_count"] == 2
    assert metrics["exact_hash_match"] is False
    assert metrics["max_abs_error"] == pytest.approx(0.25)
    assert metrics["max_abs_error_logical_flat_index"] == 1
    assert metrics["rms_error"] == pytest.approx((0.0625 / 3) ** 0.5)
    assert metrics["relative_l2_status"] == "defined"

    zero_root, _, zero_inputs = _export_reference(
        tmp_path / "zero", torch.zeros(2, dtype=torch.float32)
    )
    nonzero = torch.tensor([0.001, 0.0], dtype=torch.float32)
    _, zero_result = _compare(
        tmp_path / "zero", zero_root, _tensor_record(nonzero), nonzero, zero_inputs
    )
    zero_metrics = _comparison_metrics(zero_result)
    assert zero_metrics["zero_reference_mismatch_count"] == 1
    assert zero_metrics["zero_reference_max_abs_error"] == pytest.approx(0.001)
    assert zero_metrics["relative_l2"] is None
    assert zero_metrics["relative_l2_status"] == "zero_reference_nonzero_error"

    empty = torch.empty(0, dtype=torch.float32)
    empty_root, _, empty_inputs = _export_reference(tmp_path / "empty", empty)
    _, empty_result = _compare(
        tmp_path / "empty", empty_root, _tensor_record(empty), empty, empty_inputs
    )
    empty_metrics = _comparison_metrics(empty_result)
    assert empty_metrics["count"] == 0
    assert empty_metrics["exact_element_match"] is True
    assert empty_metrics["relative_l2_status"] == "empty_tensor"

    nonfinite_reference = torch.tensor([float("nan"), float("inf"), 1.0])
    nonfinite_current = torch.tensor([float("nan"), float("-inf"), 2.0])
    nonfinite_root, _, nonfinite_inputs = _export_reference(
        tmp_path / "nonfinite", nonfinite_reference
    )
    _, nonfinite_result = _compare(
        tmp_path / "nonfinite",
        nonfinite_root,
        _tensor_record(nonfinite_current),
        nonfinite_current,
        nonfinite_inputs,
    )
    nonfinite_metrics = _comparison_metrics(nonfinite_result)
    assert nonfinite_metrics["nonfinite_reference_count"] == 2
    assert nonfinite_metrics["nonfinite_current_count"] == 2
    assert nonfinite_metrics["nonfinite_mismatch_count"] == 0
    assert nonfinite_metrics["finite_pair_count"] == 1


def test_integer_tensor_and_none_and_float_scalar_comparisons_keep_types(
    tmp_path: Path,
) -> None:
    integer = torch.tensor([1, 2**31 + 3], dtype=torch.int64)
    root, record, inputs = _export_reference(tmp_path / "integer", integer)
    comparison = NativeNumericalPayload(
        mode="compare",
        root=root,
        rank=0,
        capture_update=1,
        run_id="integer-compare",
        comparison_root=root.parent / "integer-comparison",
    )
    comparison.verify_initial_gates([], {})
    comparison.verify_prefix_inputs(inputs)
    comparison.validate_tensor_metadata(record)
    comparison.observe_tensor(record, integer)
    result = comparison.finish(
        [*inputs, record],
        coverage=_COVERAGE,
        training_metadata={},
        monitor_attestation=None,
    )
    metrics = _comparison_metrics(result)
    assert metrics["exact_element_match_count"] == 2
    assert metrics["exact_hash_match"] is True
    assert numerics_payload._numeric_tree(None, None) == {}
    assert numerics_payload._numeric_tree(7, 7) == {}
    assert numerics_payload._numeric_tree(7, 7.0)["kind"] == "type_mismatch"
    assert numerics_payload._numeric_tree(7.0, 7.0)["kind"] == "float"


def test_streamed_comparison_crosses_one_megabyte_block_boundary(tmp_path: Path) -> None:
    count = numerics_payload.BLOCK_BYTES // torch.tensor([], dtype=torch.float32).element_size() + 3
    reference = torch.zeros(count, dtype=torch.float32)
    current = reference.clone()
    current[-1] = 0.125
    root, _, inputs = _export_reference(tmp_path, reference)
    _, result = _compare(tmp_path, root, _tensor_record(current), current, inputs)
    metrics = _comparison_metrics(result)
    assert metrics["count"] == count
    assert metrics["exact_element_match_count"] == count - 1
    assert metrics["max_abs_error"] == pytest.approx(0.125)
    assert metrics["max_abs_error_logical_flat_index"] == count - 1


@pytest.mark.parametrize("field", ["owner", "local_shape", "dtype"])
def test_compare_rejects_wrong_owner_shard_or_dtype_before_values(
    tmp_path: Path, field: str
) -> None:
    reference = torch.tensor([1.0, 2.0], dtype=torch.float32)
    root, _, inputs = _export_reference(tmp_path, reference)
    current_record = _tensor_record(reference)
    if field == "owner":
        current_record["owner"] = {"part_index": 1, "canonical_fqn": "other.weight"}
    elif field == "local_shape":
        current_record["local_shape"] = [1, 2]
    else:
        current_record["dtype"] = "torch.float64"
    payload = NativeNumericalPayload(
        mode="compare",
        root=root,
        rank=0,
        capture_update=1,
        run_id="mismatch-test",
        comparison_root=tmp_path / "comparison",
    )
    payload.verify_initial_gates([], {})
    payload.verify_prefix_inputs(inputs)
    with pytest.raises(ValueError, match="metadata differs"):
        payload.validate_tensor_metadata(current_record)
    assert next((root / "rank-00" / "tensors").glob("*.bin")).is_file()


def test_compare_fails_closed_for_missing_duplicate_truncated_corrupt_and_partial_files(
    tmp_path: Path,
) -> None:
    reference = torch.tensor([1.0, 2.0], dtype=torch.float32)
    root, _, _ = _export_reference(tmp_path / "missing", reference)
    payload_path = next((root / "rank-00" / "tensors").glob("*.bin"))
    payload_path.unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        NativeNumericalPayload(
            mode="compare",
            root=root,
            rank=0,
            capture_update=1,
            run_id="missing-test",
            comparison_root=tmp_path / "missing-comparison",
        )

    duplicate_root, _, _ = _export_reference(tmp_path / "duplicate", reference)
    manifest_path = duplicate_root / "rank-00" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["observations"].append(manifest["observations"][-1])
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="duplicate identity"):
        NativeNumericalPayload(
            mode="compare",
            root=duplicate_root,
            rank=0,
            capture_update=1,
            run_id="duplicate-test",
            comparison_root=tmp_path / "duplicate-comparison",
        )

    truncated_root, _, _ = _export_reference(tmp_path / "truncated", reference)
    truncated_path = next((truncated_root / "rank-00" / "tensors").glob("*.bin"))
    truncated_path.write_bytes(truncated_path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="size differs"):
        NativeNumericalPayload(
            mode="compare",
            root=truncated_root,
            rank=0,
            capture_update=1,
            run_id="truncated-test",
            comparison_root=tmp_path / "truncated-comparison",
        )

    partial_root, _, _ = _export_reference(tmp_path / "partial", reference)
    (partial_root / "rank-00" / "unexpected.tmp").write_bytes(b"partial")
    with pytest.raises(ValueError, match="partial or unexpected"):
        NativeNumericalPayload(
            mode="compare",
            root=partial_root,
            rank=0,
            capture_update=1,
            run_id="partial-test",
            comparison_root=tmp_path / "partial-comparison",
        )

    corrupt_root, corrupt_record, corrupt_inputs = _export_reference(
        tmp_path / "corrupt", reference
    )
    corrupt_path = next((corrupt_root / "rank-00" / "tensors").glob("*.bin"))
    corrupt_bytes = bytearray(corrupt_path.read_bytes())
    corrupt_bytes[0] ^= 1
    corrupt_path.write_bytes(corrupt_bytes)
    comparer = NativeNumericalPayload(
        mode="compare",
        root=corrupt_root,
        rank=0,
        capture_update=1,
        run_id="corrupt-test",
        comparison_root=tmp_path / "corrupt-comparison",
    )
    comparer.verify_initial_gates([], {})
    comparer.verify_prefix_inputs(corrupt_inputs)
    with pytest.raises(ValueError, match="reference tensor comparison failed"):
        comparer.observe_tensor(corrupt_record, reference)
    assert corrupt_path.is_file()
    assert (tmp_path / "corrupt-comparison" / "rank-00" / "failure.json").is_file()


def test_prefix_hash_mismatch_blocks_comparison_and_preserves_payload(tmp_path: Path) -> None:
    reference = torch.tensor([1.0], dtype=torch.float32)
    root, _, inputs = _export_reference(tmp_path, reference)
    current_inputs = [dict(item) for item in inputs]
    current_inputs[0]["sha256"] = "f" * 64
    comparer = NativeNumericalPayload(
        mode="compare",
        root=root,
        rank=0,
        capture_update=1,
        run_id="prefix-mismatch",
        comparison_root=tmp_path / "comparison",
    )
    comparer.verify_initial_gates([], {})
    with pytest.raises(ValueError, match="all-prefix CPU input hash"):
        comparer.verify_prefix_inputs(current_inputs)
    assert next((root / "rank-00" / "tensors").glob("*.bin")).is_file()


def test_failed_atomic_metric_write_keeps_the_reference_tensor(tmp_path: Path, monkeypatch):
    reference = torch.tensor([0.5, 1.5], dtype=torch.float32)
    root, record, inputs = _export_reference(tmp_path, reference)
    comparer = NativeNumericalPayload(
        mode="compare",
        root=root,
        rank=0,
        capture_update=1,
        run_id="write-failure",
        comparison_root=tmp_path / "comparison",
    )
    comparer.verify_initial_gates([], {})
    comparer.verify_prefix_inputs(inputs)
    payload_path = next((root / "rank-00" / "tensors").glob("*.bin"))
    real_write = numerics_payload._write_new_file

    def fail_observation_write(path, chunks):
        if path.parent.name == "observations":
            raise OSError("sentinel evidence write failure")
        return real_write(path, chunks)

    monkeypatch.setattr(numerics_payload, "_write_new_file", fail_observation_write)
    with pytest.raises(ValueError, match="durable comparison evidence"):
        comparer.observe_tensor(record, reference)
    assert payload_path.is_file()
    assert (tmp_path / "comparison" / "rank-00" / "failure.json").is_file()
