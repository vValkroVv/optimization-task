#pragma once

#include <torch/extension.h>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// The monitored activations of one microbatch observed from C++, as the Python observers observe them. A call of
// a monitored module hands over its positional inputs and its output with two catalog slots per position, the
// forward observation's and the gradient's (kSkipped for none, kUnlisted for a position the catalog lacks, which
// a tensor must not be at). A forward tensor's [observations, rows] are added to a host int64 [slots, 2] tensor
// and the tensor is queued; the queue is reduced in batches of up to `batch` tensors, and a slot seen again
// flushes it first, so that each slot's observations keep their order. The gradient of a tensor several modules
// saw is observed by one C++ hook under each of their slots with the reductions the Python observer makes
// (row_norm_accumulate into the first slot, accumulate_row_norm_stats into the others) and counted likewise; a
// floating scalar is one row, its magnitude both statistics. A tensor or gradient the extension does not reduce
// goes to `forward_fallback(value, slots)` or `gradient_fallback(slots, gradient)`; a forward tensor of
// `skipped_type` is not observed. The hooks hold the GIL, which the scratch maps rely on and which orders them
// with the observations.
class MonitorObservations {
public:
    static constexpr int64_t kSkipped = -1;
    static constexpr int64_t kUnlisted = -2;
    // The accumulator holds [amax, sum of row norms] and the host counts [observations, rows] of each slot.
    static constexpr int64_t kValuesPerSlot = 2;
    // A monitored position has two catalog slots: its forward observation's, then its gradient's.
    static constexpr size_t kSlotsPerPosition = 2;

    MonitorObservations(torch::Tensor accumulator, torch::Tensor counts, int64_t batch, py::object forward_fallback,
                        py::object gradient_fallback, py::object skipped_type);

    void observe_module(py::handle inputs, py::handle output, const std::vector<int64_t>& slots);

    // Reduces the queued tensors; returns -1, or the first slot of a queued tensor modified in place since it was
    // observed, in which case the queue is dropped without a reduction.
    int64_t flush();

private:
    struct State {
        State(torch::Tensor accumulator, torch::Tensor counts, py::object fallback)
            : accumulator(std::move(accumulator)), counts(std::move(counts)), fallback(std::move(fallback)) {}

        ~State();

        void check_slot(int64_t slot) const;

        void count(int64_t slot, const torch::Tensor& value);

        void observe(int64_t watch, const torch::Tensor& gradient);

        torch::Tensor accumulator;
        torch::Tensor counts;
        py::object fallback;
        std::vector<std::vector<int64_t>> slots;
    };

    struct Queued {
        torch::Tensor value;
        int64_t version;
        std::vector<int64_t> slots;
    };

    struct Watched {
        c10::weak_intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl> value;
        int64_t watch;
    };

    void observe_position(py::handle object, const std::vector<int64_t>& slots, size_t position, const char* kind);

    bool skipped(py::handle object) const;

    static void require_listed(int64_t slot, const char* kind, size_t position);

    void observe_forward(const torch::Tensor& value, int64_t slot);

    void watch(const torch::Tensor& value, int64_t slot);

    int64_t batch_;
    py::object forward_fallback_;
    py::object skipped_type_;
    std::shared_ptr<State> state_;
    std::vector<Queued> queue_;
    std::unordered_map<const c10::TensorImpl*, size_t> queued_index_;
    std::unordered_set<int64_t> queued_slots_;
    std::unordered_map<const c10::TensorImpl*, Watched> watched_;
};

// A monitored module's forward hook from C++, for MetricSampler `sampler`: it skips a module out of training, a
// forward the sampler does not capture and a backward recomputation, then hands the call to the microbatch's
// MonitorObservations with the call's catalog slots, which sampler._monitor_slots gives once per number of inputs
// and outputs. Before the microbatch has its observations, and without them, the call goes to
// sampler._observe_monitor_module(name, ...).
class MonitorModuleHook {
public:
    MonitorModuleHook(py::object sampler, std::string name, py::object graph_task_id)
        : sampler_(std::move(sampler)), name_(std::move(name)), graph_task_id_(std::move(graph_task_id)) {}

    void operator()(const py::object& module, const py::object& inputs, const py::object& output);

private:
    py::object sampler_;
    std::string name_;
    py::object graph_task_id_;
    std::map<std::pair<size_t, size_t>, std::vector<int64_t>> slots_;
};
