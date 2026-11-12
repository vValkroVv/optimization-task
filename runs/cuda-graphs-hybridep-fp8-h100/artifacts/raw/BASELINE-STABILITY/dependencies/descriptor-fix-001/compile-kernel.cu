
#include <deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(&sm90_fp8_gemm_1d1d_impl<
        1536, 1024, 0,
        4,
        128, 152, 128,
        128, 128,
        5,
        128, 256,
        2, false,
        116,
        GemmType::KGroupedContiguous, true, cutlass::bfloat16_t
    >);
};
