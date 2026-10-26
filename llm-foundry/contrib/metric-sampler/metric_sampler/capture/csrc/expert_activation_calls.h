#pragma once

#include "observer_helpers.h"

#include <torch/extension.h>

#include <cstdint>
#include <map>
#include <memory>
#include <utility>
#include <vector>

// ExpertBalanceStats.observe_experts from C++ for every MoE layer of a model, its expert modules' activation
// observers being the ExpertActivationCall callables this makes. A call is taken only while `sampler` captures the
// forward, in `sampler`'s microbatch, and checked as observe_experts checks it. A call the extension takes -- a
// bfloat16, float16 or float32 CUDA intermediate of at most 256 experts -- goes to segment_sumsq_into, its
// population's counts kept here and its slot resolved once through resolve(layer, microbatch) -- (offset, slot
// index) of expert_activation_sumsq -- and marked in the host flags `recorded`; any other call goes to
// fallback(layer, microbatch, intermediate, counts), observe_experts itself, whose populations are
// python_populations, keyed by (layer, microbatch). A population is taken by one side only: a call of the other
// side for it raises. `recorder` gives the step's word buffer; flush ends the microbatch's populations and returns
// their (layer, microbatch, counts).
class ExpertActivationCalls {
public:
    ExpertActivationCalls(py::object sampler, py::object recorder, py::object resolve, py::object fallback,
                          py::dict python_populations, int64_t local_experts, int64_t expert_offset,
                          torch::Tensor recorded, py::object capacity_error, py::object graph_task_id);

    bool active() const { return forward_capture_active(sampler_, graph_task_id_); }

    py::list flush();

    bool has_population(int64_t layer, int64_t microbatch) const;

    void observe(int64_t layer, const py::object& intermediate, const py::object& counts);

private:
    // The segments segment_sumsq_into takes: one for each local expert.
    static constexpr Py_ssize_t kMaxLocalExperts = 256;
    // A float64 count is two int32 words of the recorder's buffer.
    static constexpr int64_t kWordsPerDouble = 2;

    struct Slot {
        int64_t offset;
        int64_t index;
    };

    struct Population {
        std::vector<int64_t> counts;
        int64_t tokens;
    };

    // The extension path of add_segment_sums_of_squares (stats/helpers/reductions.py).
    static bool takes(const py::object& intermediate, const py::object& counts);

    const Slot& slot_of(int64_t layer, int64_t microbatch);

    py::object sampler_;
    py::object recorder_;
    py::object resolve_;
    py::object fallback_;
    py::dict python_populations_;
    int64_t local_experts_;
    int64_t expert_offset_;
    torch::Tensor recorded_;
    py::object capacity_error_;
    py::object graph_task_id_;
    std::map<std::pair<int64_t, int64_t>, Slot> slots_;
    std::map<std::pair<int64_t, int64_t>, Population> populations_;
};

// One MoE layer's expert activation observer.
class ExpertActivationCall {
public:
    ExpertActivationCall(std::shared_ptr<ExpertActivationCalls> calls, int64_t layer)
        : calls_(std::move(calls)), layer_(layer) {}

    void operator()(const py::object& intermediate, const py::object& counts);

private:
    std::shared_ptr<ExpertActivationCalls> calls_;
    int64_t layer_;
};
