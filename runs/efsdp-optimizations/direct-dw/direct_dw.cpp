#include <c10/cuda/CUDAGuard.h>
#include <pybind11/stl.h>

#include <limits>

#include "csrc/jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp"

namespace {

using namespace deep_gemm;
using Operand = std::pair<torch::Tensor, torch::Tensor>;

c10::DeviceIndex configured_device = -1;

void configure(const std::string& library_root, const std::string& cuda_home,
               const std::string& cache_root, int num_sms) {
    TORCH_CHECK(configured_device == -1, "The direct dW extension is already configured");
    TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->major == 9, "Direct dW requires Hopper");
    Compiler::prepare_init(library_root, cuda_home);
    KernelRuntime::prepare_init(cuda_home);
    compiler->cache_dir_path = cache_root;
    compiler->use_shared_cache = false;
    device_runtime->set_num_sms(num_sms);
    configured_device = c10::cuda::current_device();
}

void check_operand(const Operand& operand, const torch::Device& device, int mn, int64_t sum_k) {
    const auto& [data, scale] = operand;
    TORCH_CHECK(data.device() == device and scale.device() == device, "Operands must share the output device");
    TORCH_CHECK(data.scalar_type() == torch::kFloat8_e4m3fn and data.is_contiguous()
        and data.numel() == sum_k * mn, "Expected contiguous, expert-concatenated E4M3FN data");
    TORCH_CHECK(scale.scalar_type() == torch::kFloat and scale.dim() == 2
        and scale.size(0) == mn and scale.size(1) == sum_k / 128,
        "Expected FP32 scales shaped [M or N, total K / 128]");
    if (sum_k != 0) {
        TORCH_CHECK(scale.stride(0) == 1 and scale.stride(1) == mn,
            "Expected TMA-ready MN-major scales");
        TORCH_CHECK(reinterpret_cast<uintptr_t>(data.data_ptr()) % 16 == 0
            and reinterpret_cast<uintptr_t>(scale.data_ptr()) % 16 == 0,
            "Operand addresses must be 16-byte aligned");
    }
}

void k_grouped_fp8_gemm_nt_contiguous_rounded_f32(
    const Operand& a, const Operand& b, const torch::Tensor& d,
    const std::vector<int>& ks, const torch::Tensor& ks_tensor,
    const std::string& compiled_dims) {
    TORCH_CHECK(configured_device != -1, "Configure the direct dW extension first");
    TORCH_CHECK(d.is_cuda() and d.get_device() == configured_device and d.scalar_type() == torch::kFloat and d.dim() == 3
        and d.is_contiguous() and d.size(0) == 16, "Expected contiguous FP32 output for 16 experts");
    const auto [groups, m, n] = get_shape<3>(d);
    TORCH_CHECK((m == 3072 and n == 4096) or (m == 4096 and n == 1536),
        "Direct dW supports the EP8 w13 and w2 shapes");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(d.data_ptr()) % 16 == 0,
        "Output address must be 16-byte aligned");
    TORCH_CHECK(ks.size() == static_cast<size_t>(groups) and ks_tensor.device() == d.device()
        and ks_tensor.scalar_type() == torch::kInt and ks_tensor.dim() == 1
        and ks_tensor.numel() == groups and ks_tensor.is_contiguous(),
        "Expected 16 K sizes on the host and a contiguous CUDA int32 mirror");
    TORCH_CHECK(compiled_dims == "mn", "Direct dW requires compiled_dims=mn");
    int64_t sum_k = 0;
    for (const auto k : ks) {
        TORCH_CHECK(k >= 0 and k % 128 == 0, "K sizes must be nonnegative multiples of 128");
        sum_k += k;
    }
    TORCH_CHECK(sum_k <= std::numeric_limits<int>::max(), "Total K exceeds the pinned launcher range");
    check_operand(a, d.device(), m, sum_k);
    check_operand(b, d.device(), n, sum_k);

    const c10::cuda::CUDAGuard device_guard(d.device());
    const auto stream = at::cuda::getCurrentCUDAStream();
    if (sum_k != 0) {
        const auto num_sms = device_runtime->get_num_sms();
        const auto tensor_map_buffer = torch::empty({num_sms * 4 * static_cast<int>(sizeof(CUtensorMap))},
            a.first.options().dtype(torch::kByte));
        sm90_k_grouped_fp8_gemm_1d1d(a.first, a.second, b.first, b.second, std::nullopt, d,
            m, n, ks, ks_tensor, tensor_map_buffer, cute::UMMA::Major::K, cute::UMMA::Major::K,
            compiled_dims, true);
    }

    // The K-group scheduler skips empty experts. Existing slot contents must be cleared.
    for (int begin = 0; begin < groups;) {
        if (ks[begin] != 0) {
            ++begin;
            continue;
        }
        int end = begin + 1;
        while (end < groups and ks[end] == 0)
            ++end;
        const size_t group_elements = static_cast<size_t>(m) * n;
        DG_CUDA_RUNTIME_CHECK(cudaMemsetAsync(
            d.data_ptr<float>() + begin * group_elements, 0,
            (end - begin) * group_elements * sizeof(float), stream));
        begin = end;
    }
}

constexpr int probe_width = 64;

class SpecialValueProbe final : public LaunchRuntime<SpecialValueProbe> {
public:
    struct Args {
        const float* input;
        CUtensorMap bf16_zero_output;
        CUtensorMap fp32_output;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args&) {
        return R"(
#include <deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh>

extern "C" __global__ void direct_dw_special_value_probe(
    const float* input,
    const __grid_constant__ cute::TmaDescriptor bf16_zero_output,
    const __grid_constant__ cute::TmaDescriptor fp32_output) {
    __shared__ __align__(128) uint32_t bf16_shared[32];
    __shared__ __align__(128) float2 fp32_shared[32];
    const int lane = threadIdx.x;
    float x = input[2 * lane];
    float y = input[2 * lane + 1];
    bf16_shared[lane] = deep_gemm::cast_into_bf16_and_pack(x, y);
    fp32_shared[lane] = deep_gemm::round_like_bf16_tma_add_zero(x, y);
    cute::tma_store_fence();
    __syncthreads();
    if (lane == 0) {
        cute::SM90_TMA_REDUCE_ADD_2D::copy(&bf16_zero_output, bf16_shared, 0, 0);
        cute::SM90_TMA_STORE_2D::copy(&fp32_output, fp32_shared, 0, 0);
        cute::tma_store_arrive();
        cute::tma_store_wait<0>();
    }
}
)";
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.input, args.bf16_zero_output, args.fp32_output));
    }
};

void probe_special_values(const torch::Tensor& input, const torch::Tensor& bf16_zero_output,
                          const torch::Tensor& fp32_output) {
    TORCH_CHECK(configured_device != -1, "Configure the direct dW extension first");
    TORCH_CHECK(input.is_cuda() and input.get_device() == configured_device and input.scalar_type() == torch::kFloat,
        "Probe input must be CUDA FP32");
    TORCH_CHECK(bf16_zero_output.device() == input.device() and bf16_zero_output.scalar_type() == torch::kBFloat16,
        "Probe BF16 output must be on the input device");
    TORCH_CHECK(fp32_output.device() == input.device() and fp32_output.scalar_type() == torch::kFloat,
        "Probe FP32 output must be on the input device");
    for (const auto& tensor : {input, bf16_zero_output, fp32_output}) {
        TORCH_CHECK(tensor.is_contiguous() and tensor.numel() == probe_width,
            "Probe tensors must contain 64 contiguous values");
    }
    const c10::cuda::CUDAGuard device_guard(input.device());
    const SpecialValueProbe::Args args = {
        .input = input.data_ptr<float>(),
        .bf16_zero_output = make_tma_2d_desc(bf16_zero_output, probe_width, 1, probe_width, 1, probe_width, 0),
        .fp32_output = make_tma_2d_desc(fp32_output, probe_width, 1, probe_width, 1, probe_width, 0),
        .launch_args = LaunchArgs(1, 32),
    };
    const auto code = SpecialValueProbe::generate(args);
    const auto runtime = compiler->build("direct_dw_special_value_probe", code);
    SpecialValueProbe::launch(runtime, args);
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("configure", &configure);
    module.def("k_grouped_fp8_gemm_nt_contiguous_rounded_f32",
        &k_grouped_fp8_gemm_nt_contiguous_rounded_f32, pybind11::arg("a"), pybind11::arg("b"),
        pybind11::arg("d"), pybind11::arg("ks"), pybind11::arg("ks_tensor"),
        pybind11::arg("compiled_dims") = "mn");
    module.def("probe_special_values", &probe_special_values);
}
