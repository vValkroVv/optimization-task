#pragma once

#include <torch/extension.h>

#include <cstdint>

inline bool truthy(const py::object& owner, const char* attribute) {
    const int value = PyObject_IsTrue(owner.attr(attribute).ptr());
    if (value < 0) {
        throw py::error_already_set();
    }
    return value == 1;
}

// Whether `sampler` captures the forward in progress: a training forward with grad that is not a checkpoint
// recomputation, as its _capture_forward_active and _grad_forward and torch's graph task tell.
inline bool forward_capture_active(const py::object& sampler, const py::object& graph_task_id) {
    return truthy(sampler, "_capture_forward_active") && truthy(sampler, "_grad_forward") &&
           graph_task_id().cast<int64_t>() == -1;
}

inline void check_recorded_flags(const torch::Tensor& recorded) {
    TORCH_CHECK(recorded.device().is_cpu() && recorded.scalar_type() == torch::kBool && recorded.is_contiguous() &&
                    recorded.dim() == 1,
                "recorded must be the recorder's contiguous host flags");
}

inline void check_recorded_slot(const torch::Tensor& recorded, int64_t slot) {
    TORCH_CHECK(slot >= 0 && slot < recorded.numel(), "slot ", slot, " is outside the recorded flags");
}

inline void check_step_words(const torch::Tensor& words) {
    TORCH_CHECK(words.is_cuda() && words.scalar_type() == torch::kFloat && words.is_contiguous(),
                "words must be the recorder's float32 word buffer of the step");
}
