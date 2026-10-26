"""Fused selection against independent fp32 math: membership, ties and masking."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)
if torch.cuda.get_device_capability() != (9, 0):
    pytest.skip("DSA kernels require Hopper", allow_module_level=True)
from llmfoundry.models.layers import dsa_operator as OP

pytestmark = [pytest.mark.unit, pytest.mark.requires_gpu]

DEV = "cuda"
HI, DI, K = 64, 128, 128
SCALE = DI**-0.5


@pytest.fixture(autouse=True)
def exact_fp32():
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = old


def make_inputs(sq, sk, off=None, shuffle=False, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)

    def r(*s):
        return (torch.randn(*s, device=DEV, generator=g) * 0.5).bfloat16()

    iq, ik, w = r(1, sq, HI, DI), r(1, sk, 1, DI), r(1, sq, HI)
    pos = (
        torch.randperm(sk, device=DEV, generator=g)[:sq]
        if shuffle
        else torch.arange(off or 0, (off or 0) + sq, device=DEV)
    )
    return iq, ik, w, pos


def dense_scores(iq, ik, w, scale=SCALE):
    # Independent torch GEMM/reduction, with the same bf16 scaled-query rounding
    # as the production kernel. No implementation scorer or selector is reused.
    q = (iq[0] * scale).float()
    return (
        torch.einsum("shd,kd->shk", q, ik[0, :, 0].float()).relu()
        * w[0].float().unsqueeze(-1)
    ).sum(1)


def check_structural(ids, lengths, pos, cu, k=K):
    assert ids.dtype == lengths.dtype == torch.int32
    assert ids.shape == (1, pos.numel(), k)
    keep = torch.arange(k, device=DEV)[None] < lengths[:, None]
    assert torch.equal(ids[0] >= 0, keep), "valid keys must precede -1 padding"
    assert (ids[0][~keep] == -1).all()
    assert ((ids[0] <= pos[:, None]) | ~keep).all(), "future key selected"
    starts = torch.zeros_like(pos)
    if cu is not None:
        doc = torch.bucketize(pos, cu[1:-1].to(pos.dtype), right=True)
        starts = cu[doc].long()
        assert ((ids[0] >= starts[:, None]) | ~keep).all(), "cross-document key"
    assert torch.equal(lengths.long(), (pos - starts + 1).clamp_max(k))
    for row, count in zip(ids[0], lengths.tolist()):
        assert row[:count].unique().numel() == count, "duplicate key"


def check_scores(ids, lengths, scores, pos, cu):
    keys = torch.arange(scores.shape[-1], device=DEV)
    for row, count in enumerate(lengths.tolist()):
        valid = keys <= pos[row]
        if cu is not None:
            doc = torch.bucketize(pos[row], cu[1:-1].to(pos.dtype), right=True)
            valid &= keys >= cu[doc]
        masked = scores[row].masked_fill(~valid, -torch.inf)
        values, ref_ids = masked.topk(count)
        got_ids = ids[0, row, :count].long()
        # Floating-point GEMM/reduction order can exchange keys only at the cutoff.
        # The bound is relative to the score magnitude, not a broad recall allowance.
        eps = 2e-5 * scores[row].abs().max().clamp_min(1.0)
        assert (masked[got_ids] >= values[-1] - eps).all()
        omitted = ref_ids[~torch.isin(ref_ids, got_ids)]
        assert (masked[omitted] <= values[-1] + eps).all()


CASES = [
    ("late", 64, 4096, 4032, None, False),
    ("first", 64, 4096, 0, None, False),
    ("ragged", 63, 3800, 3737, None, False),
    ("varlen_late", 64, 4096, 4032, [0, 1500, 2600, 4096], False),
    ("varlen_mid", 64, 4096, 1800, [0, 1500, 2600, 4096], False),
    ("boundary", 64, 4096, 2568, [0, 1500, 2600, 4096], False),
    ("short_doc", 64, 4096, 4032, [0, 2000, 3900, 4096], False),
    ("shuffled", 64, 4096, None, None, True),
    ("shuffled_varlen", 64, 4096, None, [0, 1500, 2600, 4096], True),
    ("singleton", 1, 1, 0, None, False),
]


@pytest.mark.parametrize("name,sq,sk,off,cu,shuffle", CASES, ids=[c[0] for c in CASES])
def test_selection_against_independent_scores(name, sq, sk, off, cu, shuffle):
    iq, ik, w, pos = make_inputs(sq, sk, off, shuffle)
    cu = torch.tensor(cu, device=DEV, dtype=torch.int32) if cu else None
    ids, lengths = OP.select_topk(iq, ik, w, SCALE, K, pos, cu)
    check_structural(ids, lengths, pos, cu)
    check_scores(ids, lengths, dense_scores(iq, ik, w), pos, cu)


@pytest.mark.parametrize("k", [128, 256, 512, 1024, 2048, 4096])
def test_supported_topk_sizes(k):
    iq, ik, w, pos = make_inputs(4, 8192, 8188)
    ids, lengths = OP.select_topk(iq, ik, w, SCALE, k, pos)
    check_structural(ids, lengths, pos, None, k)
    check_scores(ids, lengths, dense_scores(iq, ik, w), pos, None)


@pytest.mark.parametrize("weight", [0.0, -1.0])
def test_ties_and_negative_scores(weight):
    iq, ik, w, pos = make_inputs(8, 512, 504)
    iq.fill_(1)
    ik.fill_(1)
    w.fill_(weight)
    ids, lengths = OP.select_topk(iq, ik, w, SCALE, K, pos)
    check_structural(ids, lengths, pos, None)
    check_scores(ids, lengths, dense_scores(iq, ik, w), pos, None)


def test_rank_selection_matches_split_rows():
    iq, ik, w, pos = make_inputs(16, 8192, shuffle=True)
    cu = torch.tensor([0, 2048, 3072, 8192], device=DEV, dtype=torch.int32)
    all_ids, all_lengths = OP.select_topk(iq, ik, w, SCALE, K, pos, cu)
    for a, b in [(0, 5), (5, 11), (11, 16)]:
        ids, lengths = OP.select_topk(iq[:, a:b], ik, w[:, a:b], SCALE, K, pos[a:b], cu)
        assert torch.equal(lengths, all_lengths[a:b])
        check_scores(
            ids, lengths, dense_scores(iq[:, a:b], ik, w[:, a:b]), pos[a:b], cu
        )
        check_scores(
            all_ids[:, a:b],
            lengths,
            dense_scores(iq[:, a:b], ik, w[:, a:b]),
            pos[a:b],
            cu,
        )


@pytest.mark.parametrize(
    "sq,sk,off,cu,shuffle",
    [
        (7, 32769, 32762, None, False),
        (7, 32769, 16382, [0, 16385, 32769], False),
        (13, 16389, None, [0, 8193, 16389], True),
        (7, 1025, 1018, [0, 1024, 1025], False),
    ],
    ids=["long_tail", "document_boundary", "shuffled_documents", "short_and_singleton"],
)
def test_large_topk_masking_and_merges(sq, sk, off, cu, shuffle):
    iq, ik, w, pos = make_inputs(sq, sk, off, shuffle, seed=19)
    cu = torch.tensor(cu, device=DEV, dtype=torch.int32) if cu else None
    ids, lengths = OP.select_topk(iq, ik, w, SCALE, 4096, pos, cu)
    check_structural(ids, lengths, pos, cu, 4096)
    check_scores(ids, lengths, dense_scores(iq, ik, w), pos, cu)


@pytest.mark.parametrize("k", [128, 2048, 4096])
def test_long_context_selection(k):
    iq, ik, w, pos = make_inputs(3, 131072, 131069, seed=23)
    ids, lengths = OP.select_topk(iq, ik, w, SCALE, k, pos)
    check_structural(ids, lengths, pos, None, k)
    check_scores(ids, lengths, dense_scores(iq, ik, w), pos, None)
