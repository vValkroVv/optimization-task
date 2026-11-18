"""GPU0-sized numerical checks for the optional row-owner CUDA kernels."""

import os

import pytest
import torch

from torchtitan.distributed.gfsdp import _row_owner_embedding_kernels as kernels
from torchtitan.distributed.gfsdp.row_owner_embedding import (
    _compact_native_is_safe,
    _native_source_coalesce,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("ROW_OWNER_CUDA_KERNEL_TESTS") != "1"
    or not torch.cuda.is_available(),
    reason="set ROW_OWNER_CUDA_KERNEL_TESTS=1 with CUDA visible to run kernel checks",
)


def _signed_words(words, bits):
    sign = 1 << (bits - 1)
    modulus = 1 << bits
    return [word if word < sign else word - modulus for word in words]


def _float32_from_bits(words, device):
    return torch.tensor(
        _signed_words(words, 32), dtype=torch.int32, device=device
    ).view(torch.float32)


def _bfloat16_from_bits(words, device):
    return torch.tensor(
        _signed_words(words, 16), dtype=torch.int16, device=device
    ).view(torch.bfloat16)


def _captured_ids(num_ids, num_unique, device, seed):
    if num_ids == 0:
        return torch.empty(0, dtype=torch.int64, device=device)
    counts = torch.ones(num_unique, dtype=torch.int64)
    if num_unique >= 6:
        counts[:6] = torch.tensor([9, 10, 11, 19, 20, 21])
    counts[-1] += num_ids - counts.sum().item()
    sorted_ids = torch.repeat_interleave(torch.arange(num_unique), counts)
    order = torch.randperm(num_ids, generator=torch.Generator().manual_seed(seed))
    return sorted_ids[order].to(device=device, dtype=torch.int64)


def _native_reference(grad, ids, unique, num_embeddings):
    dense = torch.ops.aten.embedding_dense_backward.default(
        grad, ids, num_embeddings, -1, False
    )
    result = dense.index_select(0, unique)
    del dense
    return result


def _fallback_routing_metadata(ids, num_embeddings, rows_per_owner, world_size):
    valid = ((ids >= 0) & (ids < num_embeddings)).all().to(torch.int64)
    safe_ids = ids.clamp(0, num_embeddings - 1)
    unique = torch.unique(safe_ids, sorted=True)
    owners = torch.div(unique, rows_per_owner, rounding_mode="floor")
    metadata = torch.zeros(world_size + 1, dtype=torch.int64, device=ids.device)
    metadata[:world_size].scatter_add_(0, owners, torch.ones_like(owners))
    metadata[world_size] = valid
    return safe_ids, metadata


def test_compact_native_gradient_matches_full_unique_at_dispatch_boundaries():
    device = torch.device("cuda", torch.cuda.current_device())
    num_embeddings, width, num_sms = (
        128256,
        4096,
        torch.cuda.get_device_properties(device).multi_processor_count,
    )
    cases = (
        (0, 0),
        (3072, 1800),
        (3073, 1800),
        (3073, 2),
        (8192, 1800),
        (8192, 2),
    )
    assert _compact_native_is_safe(3072, 1, width, num_sms)
    assert _compact_native_is_safe(8192, 1800, width, num_sms)
    assert not _compact_native_is_safe(8192, 2, width, num_sms)

    for index, (num_ids, num_unique) in enumerate(cases):
        ids = _captured_ids(num_ids, num_unique, device, seed=800 + index)
        unique, inverse = torch.unique(ids, sorted=True, return_inverse=True)
        grad_values = (
            torch.arange(num_ids * width, dtype=torch.int32, device=device) % 257 - 128
        ).to(torch.float32) / 128
        grad = grad_values.to(torch.bfloat16).view(num_ids, width)
        expected = _native_reference(grad, ids, unique, num_embeddings)

        compact = _compact_native_is_safe(num_ids, unique.numel(), width, num_sms)
        if compact:
            actual = _native_source_coalesce(
                grad,
                inverse,
                None,
                num_embeddings,
                compact_rows=unique.numel(),
            )
        else:
            actual = _native_source_coalesce(
                grad,
                ids,
                unique,
                num_embeddings,
                output=torch.empty_like(expected),
            )
        assert torch.equal(actual, expected), (num_ids, num_unique, compact)
        del actual, expected, grad, grad_values, inverse, unique, ids


def test_owner_lookup_conversion_matches_native_copy_bits():
    device = torch.device("cuda", torch.cuda.current_device())
    high_words = (
        0x0000,
        0x0001,
        0x007F,
        0x0080,
        0x3F80,
        0xBF80,
        0x7F7F,
        0xFF7F,
        0x7F80,
        0xFF80,
        0x7F81,
        0x7FC1,
        0xFF81,
        0xFFC1,
    )
    values = [
        (upper << 16) | lower
        for upper in high_words
        for lower in (0x0000, 0x7FFF, 0x8000, 0x8001, 0xFFFF)
    ]
    weight = _float32_from_bits(values, device).view(-1, 1).contiguous()
    local_rows = torch.arange(weight.size(0), dtype=torch.int64, device=device)
    actual = torch.empty(weight.shape, dtype=torch.bfloat16, device=device)
    expected = torch.empty_like(actual)
    expected.copy_(weight)

    kernels.owner_lookup_bf16(weight, local_rows, actual)
    torch.cuda.synchronize(device)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("width", (16, 4096))
def test_owner_add_matches_serial_native_index_add_bits(width):
    device = torch.device("cuda", torch.cuda.current_device())
    num_rows = 5
    rows = torch.arange(num_rows, dtype=torch.int64, device=device)
    initial_words = [
        0x00000001,
        0x80000001,
        0x00000000,
        0x80000000,
        0x00800000,
        0x80800000,
        0x3F800000,
        0xBF800000,
        0x7F800000,
        0xFF800000,
        0x7FC12345,
        0xFFC12345,
        0x4F000000,
        0xCF000000,
        0x00010000,
        0x80010000,
    ]
    source_words = (
        [
            0x3F80,
            0xBF80,
            0x0001,
            0x8001,
            0x0000,
            0x8000,
            0x7F80,
            0xFF80,
            0x7FC1,
            0xFFC1,
            0x7F81,
            0xFF81,
            0x5F00,
            0xDF00,
            0x0001,
            0x8001,
        ],
        [
            0xBF80,
            0x3F80,
            0x8001,
            0x0001,
            0x8000,
            0x0000,
            0xFF80,
            0x7F80,
            0xFFC1,
            0x7FC1,
            0xFF81,
            0x7F81,
            0xDF00,
            0x5F00,
            0x8001,
            0x0001,
        ],
        [
            0x3F80,
            0x3F80,
            0x0001,
            0x0001,
            0x8000,
            0x8000,
            0x7F80,
            0x7F80,
            0x7FC1,
            0x7FC1,
            0x7F81,
            0x7F81,
            0x5F00,
            0x5F00,
            0x0001,
            0x0001,
        ],
    )
    actual = _float32_from_bits(initial_words * (num_rows * width // 16), device).view(
        num_rows, width
    )
    expected = actual.clone()

    for words in source_words:
        source = _bfloat16_from_bits(words * (num_rows * width // 16), device).view(
            num_rows, width
        )
        kernels.owner_add_bf16_(actual, rows, source)
        expected.index_add_(0, rows, source.float())
    torch.cuda.synchronize(device)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


def test_routing_kernels_clamp_signed_limits_and_count_empty_intervals():
    device = torch.device("cuda", torch.cuda.current_device())
    num_embeddings, rows_per_owner, world_size = 16, 2, 8

    def check_against_fallback(ids, vocabulary, owner_rows):
        safe_ids = torch.empty_like(ids)
        metadata = torch.empty(world_size + 1, dtype=torch.int64, device=device)
        kernels.validate_and_clamp_ids(
            ids, safe_ids, metadata[world_size], vocabulary
        )
        unique = torch.unique(safe_ids, sorted=True)
        kernels.build_routing_metadata(
            unique,
            metadata[world_size],
            metadata,
            owner_rows,
            world_size,
        )
        expected_safe, expected_metadata = _fallback_routing_metadata(
            ids, vocabulary, owner_rows, world_size
        )
        torch.cuda.synchronize(device)
        assert torch.equal(safe_ids, expected_safe)
        assert torch.equal(metadata, expected_metadata)
        return safe_ids, metadata

    ids = torch.tensor(
        [0, 15, -1, 16, torch.iinfo(torch.int64).min, torch.iinfo(torch.int64).max],
        dtype=torch.int64,
        device=device,
    )
    safe_ids, metadata = check_against_fallback(
        ids, num_embeddings, rows_per_owner
    )
    assert metadata.tolist() == [1, 0, 0, 0, 0, 0, 0, 1, 0]
    assert safe_ids.tolist() == [0, 15, 0, 15, 0, 15]

    empty = torch.empty(0, dtype=torch.int64, device=device)
    _, metadata = check_against_fallback(empty, num_embeddings, rows_per_owner)
    assert metadata.tolist() == [0] * world_size + [1]

    max_count = 8192
    max_ids = torch.arange(max_count, dtype=torch.int64, device=device)
    max_safe, max_metadata = check_against_fallback(
        max_ids, max_count, max_count // world_size
    )
    assert torch.equal(max_safe, max_ids)
    assert max_metadata.tolist() == [1024] * world_size + [1]

    skewed = torch.cat(
        (
            torch.zeros(4096, dtype=torch.int64, device=device),
            torch.ones(3000, dtype=torch.int64, device=device),
            torch.full((800,), 1024, dtype=torch.int64, device=device),
            torch.full((296,), 2048, dtype=torch.int64, device=device),
        )
    )
    _, skewed_metadata = check_against_fallback(skewed, 8192, 1024)
    assert skewed_metadata.tolist() == [2, 1, 1, 0, 0, 0, 0, 0, 1]


def test_compact_dispatch_guard_matches_native_grid_threshold():
    assert _compact_native_is_safe(3072, 1, 4096, 132)
    assert not _compact_native_is_safe(3073, 131, 4096, 132)
    assert _compact_native_is_safe(3073, 132, 4096, 132)
