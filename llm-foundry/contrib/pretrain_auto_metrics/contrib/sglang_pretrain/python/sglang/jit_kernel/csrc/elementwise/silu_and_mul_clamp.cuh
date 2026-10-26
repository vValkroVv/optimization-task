#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <cstdint>
#include <type_traits>

namespace {

template <bool kApplySwigluLimit, bool kPrecise = true, typename DType2>
SGL_DEVICE fp32x2_t silu_and_mul(DType2 gate, DType2 up, float limit) {
  using namespace device;
  // refer to as implementation. TL;DR: must clamp in bf16
  // https://github.com/deepseek-ai/DeepGEMM/blob/7f2a703ed51ac1f7af07f5e1453b2d3267d37d50/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L984-L997
  if constexpr (kApplySwigluLimit) {
    static_assert(std::is_same_v<DType2, bf16x2_t>);
    gate = __hmin2(gate, {limit, limit});
    up = __hmax2(up, {-limit, -limit});
    up = __hmin2(up, {limit, limit});
  }
  const auto [g0, g1] = cast<fp32x2_t>(gate);
  const auto [u0, u1] = cast<fp32x2_t>(up);
  const auto silu0 = g0 / (1.0f + __expf(-g0));
  const auto silu1 = g1 / (1.0f + __expf(-g1));
  const float val0 = silu0 * u0;
  const float val1 = silu1 * u1;
  if constexpr (kPrecise) {  // I don't know if we should enable this?
    return {val0, val1};
  } else {
    return cast<fp32x2_t>(cast<bf16x2_t>(fp32x2_t{val0, val1}));
  }
}

struct SiluAndMulClampParams {
  const void* __restrict__ input;
  void* __restrict__ output;
  float swiglu_limit;
  uint32_t out_dim_vecs;  // = out_dim / kVecSize; lets the kernel stride-loop
                          // over vecs when out_dim exceeds the single-pass cap.
};

template <typename DType, bool kUsePDL>
__global__ __launch_bounds__(1024, 2) void  // maximize occupancy
    silu_mul_clamp_kernel(const SiluAndMulClampParams __grid_constant__ params) {
  using namespace device;
  static_assert(sizeof(DType) == 2, "only fp16/bf16 supported");
  using DType2 = packed_t<DType>;
  constexpr auto kVecSize = 16 / sizeof(DType);
  static_assert(kVecSize % 2 == 0 && kVecSize > 0);
  using Vec = AlignedVector<DType2, kVecSize / 2>;
  const auto bid = blockIdx.x;
  const float limit = params.swiglu_limit;
  const uint32_t out_vecs = params.out_dim_vecs;
  const uint32_t tid = threadIdx.x;
  const uint32_t bdim = blockDim.x;

  // One block per row (token). Within the block, threads stride over vecs:
  // when out_vecs <= bdim, this matches the original 1-vec-per-thread layout;
  // when out_vecs > bdim, each thread handles ceil(out_vecs / bdim) vecs.
  const Vec* in_row = static_cast<const Vec*>(params.input) + bid * 2 * out_vecs;
  Vec* out_row = static_cast<Vec*>(params.output) + bid * out_vecs;

  PDLWaitPrimary<kUsePDL>();
  for (uint32_t v = tid; v < out_vecs; v += bdim) {
    const Vec gate = in_row[v];
    const Vec up = in_row[out_vecs + v];
    Vec out;
#pragma unroll
    for (uint32_t i = 0; i < kVecSize / 2; ++i) {
      out[i] = cast<DType2>(silu_and_mul<true>(cast<bf16x2_t>(gate[i]), cast<bf16x2_t>(up[i]), limit));
    }
    out_row[v] = out;
  }
  PDLTriggerSecondary<kUsePDL>();
}

template <typename DType, bool kUsePDL>
struct SiluAndMulClampKernel {
  static constexpr auto kernel = silu_mul_clamp_kernel<DType, kUsePDL>;

  static void run(const tvm::ffi::TensorView input, const tvm::ffi::TensorView output, const double swiglu_limit) {
    using namespace host;

    auto device = SymbolicDevice{};
    auto M = SymbolicSize{"num_tokens"};
    auto D = SymbolicSize{"gate_up_dim"};  // 2 * out_dim
    auto H = SymbolicSize{"out_dim"};
    device.set_options<kDLCUDA>();

    TensorMatcher({M, D})  // input  (gate || up)
        .with_dtype<DType>()
        .with_device(device)
        .verify(input);
    TensorMatcher({M, H})  // output
        .with_dtype<DType>()
        .with_device(device)
        .verify(output);
    RuntimeCheck(D.unwrap() == 2 * H.unwrap(), "input last dim must be 2 * output last dim");

    constexpr uint32_t kVecSize = 16 / sizeof(DType);
    const auto out_dim = static_cast<uint32_t>(H.unwrap());
    const auto num_tokens = static_cast<uint32_t>(M.unwrap());
    RuntimeCheck(out_dim % kVecSize == 0, "out_dim must be divisible by vector size");
    const auto out_vecs = out_dim / kVecSize;
    // Cap the block size at the __launch_bounds__ limit; the kernel's
    // stride-loop covers any extra vecs when out_vecs > 1024.
    const uint32_t num_threads = out_vecs <= 1024 ? out_vecs : 1024;

    const auto params = SiluAndMulClampParams{
        .input = input.data_ptr(),
        .output = output.data_ptr(),
        .swiglu_limit = static_cast<float>(swiglu_limit),
        .out_dim_vecs = out_vecs,
    };
    LaunchKernel(num_tokens, num_threads, device.unwrap())  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace
