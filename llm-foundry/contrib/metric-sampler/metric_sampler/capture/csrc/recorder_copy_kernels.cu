#include "cuda_host_functions.h"
#include "kernel_common.cuh"

namespace {

// Integer counts copied into recorder words in one launch, each converted to int32 as copy_ converts it.
constexpr int kMaxCountCopies = 8;
constexpr int kMaxCopyBlocks = 64;
struct CountCopies {
    int count;
    const void* sources[kMaxCountCopies];
    bool wide[kMaxCountCopies];  // an int64 source, otherwise int32
    int64_t numel[kMaxCountCopies];
    int64_t offsets[kMaxCountCopies];
};

__global__ void copy_counts_kernel(CountCopies copies, int32_t* __restrict__ words) {
    const int entry = static_cast<int>(blockIdx.y);
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < copies.numel[entry];
         i += stride) {
        words[copies.offsets[entry] + i] =
            copies.wide[entry] ? static_cast<int32_t>(static_cast<const int64_t*>(copies.sources[entry])[i])
                               : static_cast<const int32_t*>(copies.sources[entry])[i];
    }
}

}  // namespace

void copy_counts_into(std::vector<torch::Tensor> sources, torch::Tensor words, std::vector<int64_t> offsets) {
    TORCH_CHECK(!sources.empty() && sources.size() <= static_cast<size_t>(kMaxCountCopies) &&
                    offsets.size() == sources.size(),
                "copy_counts_into takes one to ", kMaxCountCopies, " sources, each with an offset");
    TORCH_CHECK(words.is_cuda() && words.scalar_type() == at::kInt && words.is_contiguous(),
                "words must be contiguous int32 CUDA words");
    CountCopies copies{static_cast<int>(sources.size()), {}, {}, {}, {}};
    int64_t longest = 1;
    for (size_t i = 0; i < sources.size(); ++i) {
        const torch::Tensor& source = sources[i];
        TORCH_CHECK(source.device() == words.device() && source.is_contiguous() &&
                        (source.scalar_type() == at::kLong || source.scalar_type() == at::kInt),
                    "counts must be contiguous int64 or int32 tensors on the words' device");
        TORCH_CHECK(offsets[i] >= 0 && offsets[i] + source.numel() <= words.numel(),
                    "a count slot is outside the words");
        copies.sources[i] = source.data_ptr();
        copies.wide[i] = source.scalar_type() == at::kLong;
        copies.numel[i] = source.numel();
        copies.offsets[i] = offsets[i];
        longest = std::max(longest, source.numel());
    }
    const at::cuda::OptionalCUDAGuard device_guard(words.device());
    const dim3 grid(
        static_cast<unsigned int>(std::min<int64_t>(ceil_div(longest, kBlockThreads), kMaxCopyBlocks)),
        copies.count);
    copy_counts_kernel<<<grid, kBlockThreads, 0, at::cuda::getCurrentCUDAStream()>>>(copies,
                                                                                     words.data_ptr<int32_t>());
}

namespace {

// The [amax, mean row norm] of each monitored activation, from its float64 [amax, row norm sum] and its host
// [observations, rows] counts, as the callback's torch operations compute them: NaN rows for an unobserved one,
// the amax kept as NaN with them, the sum divided in float64, both rounded to float32.
__global__ void monitor_values_kernel(
    const double* __restrict__ accumulator, const int64_t* __restrict__ counts, float* __restrict__ words,
    const int64_t* __restrict__ targets, int64_t entries) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= entries) {
        return;
    }
    const double rows = counts[2 * i] > 0 ? static_cast<double>(counts[2 * i + 1] > 1 ? counts[2 * i + 1] : 1)
                                          : __longlong_as_double(kQuietNanBits64);
    words[targets[2 * i]] = static_cast<float>(isnan(rows) ? rows : accumulator[2 * i]);
    words[targets[2 * i + 1]] = static_cast<float>(accumulator[2 * i + 1] / rows);
}

}  // namespace

void monitor_values_into(torch::Tensor accumulator, torch::Tensor counts, torch::Tensor words, torch::Tensor targets) {
    TORCH_CHECK(accumulator.is_cuda() && accumulator.scalar_type() == at::kDouble && accumulator.is_contiguous() &&
                    accumulator.dim() == 2 && accumulator.size(1) == kRowNormStats,
                "accumulator must be a contiguous float64 [entries, 2] CUDA tensor");
    TORCH_CHECK(counts.device().is_cpu() && counts.scalar_type() == at::kLong && counts.is_contiguous() &&
                    counts.sizes() == accumulator.sizes(),
                "counts must be contiguous host int64 counts shaped like the accumulator");
    TORCH_CHECK(words.device() == accumulator.device() && words.scalar_type() == at::kFloat && words.is_contiguous(),
                "words must be the recorder's contiguous float32 words on the accumulator's device");
    TORCH_CHECK(targets.device() == accumulator.device() && targets.scalar_type() == at::kLong &&
                    targets.is_contiguous() && targets.numel() == accumulator.numel(),
                "targets must hold the word of each value on the accumulator's device");
    const at::cuda::OptionalCUDAGuard device_guard(accumulator.device());
    const torch::Tensor device_counts = counts.pin_memory().to(accumulator.options().dtype(at::kLong), true);
    const int64_t entries = accumulator.size(0);
    if (entries == 0) {
        return;
    }
    monitor_values_kernel<<<static_cast<unsigned int>(ceil_div(entries, kBlockThreads)), kBlockThreads, 0,
                            at::cuda::getCurrentCUDAStream()>>>(
        accumulator.data_ptr<double>(), device_counts.data_ptr<int64_t>(), words.data_ptr<float>(),
        targets.data_ptr<int64_t>(), entries);
}
