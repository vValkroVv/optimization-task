"""Compile an eight-CTA ordered DSM reduction for SM90a without launching it.

Set CUTE_DSL_KEEP_IR=1 and CUTE_DSL_DUMP_DIR to preserve compiler artifacts.
"""

import argparse
from pathlib import Path

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cutlass_dsl import dsl_user_op


@dsl_user_op
def _read_peer_f32(ptr, peer, *, loc=None, ip=None):
    # Keep the mapped address in the cluster window through the load.
    return cutlass.Float32(
        llvm.inline_asm(
            cutlass.Float32.mlir_type,
            [
                ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
                cutlass.Int32(peer).ir_value(loc=loc, ip=ip),
            ],
            """{
                .reg .u32 remote_addr;
                mapa.shared::cluster.u32 remote_addr, $1, $2;
                ld.shared::cluster.f32 $0, [remote_addr];
            }""",
            "=f,r,r,~{memory}",
            has_side_effects=True,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.kernel
def _norm_cluster_dsm(src: cute.Tensor, dst: cute.Tensor):
    tid, _, _ = cute.arch.thread_idx()
    rank = cute.arch.block_idx_in_cluster()
    smem = cutlass.utils.SmemAllocator()
    partial = smem.allocate_tensor(cutlass.Float32, 32, byte_alignment=16)
    reduced = smem.allocate_tensor(cutlass.Float32, 32, byte_alignment=16)

    for tile in cutlass.range(2, unroll=1):
        if tid < 32:
            partial[tid] = src[tile, rank, tid]
        cute.arch.cluster_arrive(aligned=True)
        cute.arch.cluster_wait()

        if rank == 0 and tid < 32:
            total = cutlass.Float32(0)
            for peer in cutlass.range_constexpr(8):
                total += _read_peer_f32(partial.iterator + tid, peer)
            reduced[tid] = total
        cute.arch.cluster_arrive(aligned=True)
        cute.arch.cluster_wait()

        if tid < 32:
            dst[tile, rank, tid] = _read_peer_f32(reduced.iterator + tid, 0)
        # Every peer must finish its reads before the next tile or any CTA exits.
        cute.arch.cluster_arrive(aligned=True)
        cute.arch.cluster_wait()


@cute.jit
def launch(src: cute.Tensor, dst: cute.Tensor, stream: cuda.CUstream):
    _norm_cluster_dsm(src, dst).launch(
        grid=(8, 1, 1), block=(384, 1, 1), cluster=(8, 1, 1), stream=stream
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="artifact path prefix"
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shape = (2, 8, 32)
    src, dst = (
        make_fake_compact_tensor(
            cutlass.Float32, shape, stride_order=(2, 1, 0), assumed_align=16
        )
        for _ in range(2)
    )
    compiled = cute.compile(
        launch,
        src,
        dst,
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi --gpu-arch sm_90a --generate-line-info --keep-ptx --keep-cubin",
    )
    ptx = compiled.__ptx__
    ir = compiled.__mlir__
    cubin = compiled.__cubin__
    if not ptx or not isinstance(cubin, bytes) or not cubin:
        raise RuntimeError("CuTe compilation must produce PTX and CUBIN")
    if not ir:
        raise RuntimeError("Repeat with CUTE_DSL_KEEP_IR=1 to preserve MLIR")
    args.output.with_suffix(".ptx").write_text(ptx)
    args.output.with_suffix(".mlir").write_text(str(ir))
    args.output.with_suffix(".cubin").write_bytes(cubin)

    expected = (
        "mapa.shared::cluster.u32",
        "ld.shared::cluster.f32",
        "barrier.cluster.arrive",
        "barrier.cluster.wait",
    )
    missing = [instruction for instruction in expected if instruction not in ptx]
    if missing:
        raise RuntimeError(f"Missing instructions from PTX: {', '.join(missing)}")


if __name__ == "__main__":
    main()
