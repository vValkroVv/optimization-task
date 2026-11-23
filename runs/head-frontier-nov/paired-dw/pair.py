"""Hopper paired dW trial. Two independent BF16-input FP32 dot products share one final FP32 store."""
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def paired_dw_kernel(G0, X0, G1, X1, OUT, V: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                     ADD: tl.constexpr, INTERLEAVE: tl.constexpr):
    pid = tl.program_id(0)
    mtiles, ntiles = tl.cdiv(V, BM), tl.cdiv(H, BN)
    first = (pid // (8 * ntiles)) * 8
    size = tl.minimum(mtiles - first, 8)
    within = pid % (8 * ntiles)
    m = (first + within % size) * BM
    n = (within // size) * BN
    a = tl.full((BM, BN), 0, tl.float32)
    b = tl.full((BM, BN), 0, tl.float32)
    for block in range(K // BK):
        k = block * BK
        a = tl.dot(tl.trans(G0.load([k, m])), X0.load([k, n]), a)
        if INTERLEAVE:
            b = tl.dot(tl.trans(G1.load([k, m])), X1.load([k, n]), b)
    if not INTERLEAVE:
        for block in range(K // BK):
            k = block * BK
            b = tl.dot(tl.trans(G1.load([k, m])), X1.load([k, n]), b)
    if ADD:
        a = OUT.load([m, n]) + a
    OUT.store([m, n], a + b)


# Tile sizes, pipeline stages, and whether the two K loops are interleaved.
CONFIGS = [(128, 128, 64, 3, False), (64, 128, 64, 3, False),
           (128, 128, 64, 2, True), (64, 128, 64, 2, True),
           (128, 256, 64, 3, False), (128, 128, 64, 4, False),
           (128, 128, 64, 5, False), (128, 128, 32, 5, False),
           (128, 64, 64, 5, False), (64, 256, 64, 4, False),
           (128, 128, 64, 4, False), (128, 128, 64, 5, False),
           (128, 64, 128, 3, False)]


def launch(g0, x0, g1, x1, output, add, config):
    bm, bn, bk, stages, interleave = CONFIGS[config]
    k, vocab = g0.shape
    width = x0.size(1)
    assert k % bk == 0 and width % bn == 0
    desc = TensorDescriptor.from_tensor
    return paired_dw_kernel[(triton.cdiv(vocab, bm) * triton.cdiv(width, bn),)](
        desc(g0, [bk, bm]), desc(x0, [bk, bn]), desc(g1, [bk, bm]), desc(x1, [bk, bn]),
        desc(output, [bm, bn]), vocab, width, k, bm, bn, bk, add, interleave,
        num_warps=8 if bm == 128 and config not in (10, 11) else 4, num_stages=stages, enable_fp_fusion=False)


class PairWriter:
    def __init__(self, config):
        self.config = config
        self.pending = None
        self.stores = 0

    def __call__(self, grad, hidden, output, add):
        if self.pending is None:
            self.pending = (grad, hidden, add)
            return
        first_grad, first_hidden, first_add = self.pending
        assert add
        launch(first_grad, first_hidden, grad, hidden, output, first_add, self.config)
        self.pending = None
        self.stores += 1
