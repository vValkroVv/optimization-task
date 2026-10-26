#include "cuda_host_functions.h"
#include "kernel_common.cuh"

namespace {

// The documents and the heads' float64 norm sums of a linear attention layer's state statistics, rounded to
// float32 into its slot as fill_ and copy_ round them; the norm maxima are written by amax.
__global__ void linear_state_finish_kernel(
    const double* __restrict__ sums, double documents, int64_t heads, float* __restrict__ slot) {
    const int64_t head = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (head == 0) {
        slot[0] = static_cast<float>(documents);
    }
    if (head < heads) {
        slot[1 + head] = static_cast<float>(sums[head]);
    }
}

}  // namespace

void linear_state_stats(torch::Tensor state, torch::Tensor slot, bool merge) {
    TORCH_CHECK(state.is_cuda() && state.dim() == 4,
                "state must be [documents, value heads, key dim, value dim] on CUDA");
    const int64_t documents = state.size(0), heads = state.size(1);
    TORCH_CHECK(slot.device() == state.device() && slot.scalar_type() == at::kFloat && slot.dim() == 1 &&
                    slot.numel() == 1 + 2 * heads,
                "slot must be the float32 [documents, norm sums, norm maxima] of the state's heads");
    const at::cuda::OptionalCUDAGuard device_guard(state.device());
    at::NoGradGuard no_grad;
    const torch::Tensor norms = at::linalg_vector_norm(state, 2, {-2, -1}, false, at::kFloat);
    const torch::Tensor sums = norms.sum(0, false, at::kDouble);
    if (!merge) {
        // The maxima go straight into the slot.
        torch::Tensor highest = slot.narrow(0, 1 + heads, heads);
        at::amax_out(highest, norms, {0});
        linear_state_finish_kernel<<<static_cast<unsigned int>(ceil_div(heads, kBlockThreads)),
                                     kBlockThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
            sums.data_ptr<double>(), static_cast<double>(documents), heads, slot.data_ptr<float>());
        return;
    }
    const torch::Tensor peaks = norms.amax(0);
    torch::Tensor earlier = slot.to(at::kDouble);
    earlier.narrow(0, 0, 1).add_(static_cast<double>(documents));
    earlier.narrow(0, 1, heads).add_(sums);
    torch::Tensor highest = earlier.narrow(0, 1 + heads, heads);
    at::maximum_out(highest, highest, peaks.to(at::kDouble));
    slot.copy_(earlier);
}
