#include "expert_activation_calls.h"
#include "cuda_host_functions.h"
#include "observer_helpers.h"

#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

ExpertActivationCalls::ExpertActivationCalls(py::object sampler, py::object recorder, py::object resolve,
                                             py::object fallback, py::dict python_populations, int64_t local_experts,
                                             int64_t expert_offset, torch::Tensor recorded, py::object capacity_error,
                                             py::object graph_task_id)
    : sampler_(std::move(sampler)),
      recorder_(std::move(recorder)),
      resolve_(std::move(resolve)),
      fallback_(std::move(fallback)),
      python_populations_(std::move(python_populations)),
      local_experts_(local_experts),
      expert_offset_(expert_offset),
      recorded_(std::move(recorded)),
      capacity_error_(std::move(capacity_error)),
      graph_task_id_(std::move(graph_task_id)) {
    check_recorded_flags(recorded_);
}

py::list ExpertActivationCalls::flush() {
    py::list counts;
    for (const auto& [key, population] : populations_) {
        counts.append(py::make_tuple(key.first, key.second, py::cast(population.counts)));
    }
    populations_.clear();
    return counts;
}

bool ExpertActivationCalls::has_population(int64_t layer, int64_t microbatch) const {
    return populations_.count(std::make_pair(layer, microbatch)) > 0;
}

void ExpertActivationCalls::observe(int64_t layer, const py::object& intermediate, const py::object& counts) {
    const int64_t microbatch = sampler_.attr("microbatch").cast<int64_t>();
    const auto key = std::make_pair(layer, microbatch);
    if (!takes(intermediate, counts)) {
        if (populations_.count(key) > 0) {
            throw py::value_error("expert layer " + std::to_string(layer) + " microbatch " +
                                  std::to_string(microbatch) + " has a call the extension took");
        }
        fallback_(layer, microbatch, intermediate, counts);
        return;
    }
    if (python_populations_.contains(py::make_tuple(layer, microbatch))) {
        throw py::value_error("expert layer " + std::to_string(layer) + " microbatch " +
                              std::to_string(microbatch) + " has a call the extension did not take");
    }
    // observe_experts's checks, in its order.
    const py::list counts_list = py::reinterpret_borrow<py::list>(counts);
    std::vector<int64_t> values;
    bool too_large = false;
    bool exact = static_cast<int64_t>(counts_list.size()) == local_experts_;
    for (size_t i = 0; exact && i < counts_list.size(); ++i) {
        PyObject* count = counts_list[i].ptr();
        exact = PyLong_CheckExact(count);
        if (!exact) {
            break;
        }
        int overflow = 0;
        const long long value = PyLong_AsLongLongAndOverflow(count, &overflow);
        if (value == -1 && PyErr_Occurred()) {
            throw py::error_already_set();
        }
        exact = overflow > 0 || (overflow == 0 && value >= 0);
        // A count beyond int64 exceeds the Int32 token counts as surely as one beyond int32.
        too_large = too_large || overflow > 0 || value > std::numeric_limits<int32_t>::max();
        values.push_back(overflow > 0 ? 0 : value);
    }
    if (!exact) {
        throw py::value_error("expert token counts must be exact nonnegative integers for every local expert");
    }
    int64_t tokens = 0;
    for (const int64_t value : values) {
        tokens += too_large ? 0 : value;
    }
    auto found = populations_.find(key);
    Population* previous = found == populations_.end() ? nullptr : &found->second;
    const int64_t total_tokens = tokens + (previous == nullptr ? 0 : previous->tokens);
    if (too_large || total_tokens > std::numeric_limits<int32_t>::max()) {
        PyErr_SetString(capacity_error_.ptr(), "expert population exceeds Int32 token counts");
        throw py::error_already_set();
    }
    const torch::Tensor matrix = intermediate.cast<torch::Tensor>();
    if (matrix.dim() != 2 || matrix.size(0) != tokens) {
        throw py::value_error("expert capture requires the complete unpadded two-dimensional intermediate");
    }
    const Slot& slot = slot_of(layer, microbatch);
    // The step's buffer starts zeroed: only the owned experts' float64 words are written.
    segment_sumsq_into(matrix, values, recorder_.attr("buffer").cast<torch::Tensor>(),
                       slot.offset + kWordsPerDouble * expert_offset_, previous != nullptr);
    if (previous == nullptr) {
        recorded_.data_ptr<bool>()[slot.index] = true;
        populations_.emplace(key, Population{std::move(values), tokens});
        return;
    }
    for (size_t i = 0; i < values.size(); ++i) {
        previous->counts[i] += values[i];
    }
    previous->tokens = total_tokens;
}

bool ExpertActivationCalls::takes(const py::object& intermediate, const py::object& counts) {
    if (!THPVariable_Check(intermediate.ptr()) || !PyList_CheckExact(counts.ptr()) ||
        PyList_GET_SIZE(counts.ptr()) > kMaxLocalExperts) {
        return false;
    }
    const torch::Tensor matrix = intermediate.cast<torch::Tensor>();
    const auto dtype = matrix.scalar_type();
    return matrix.is_cuda() && (dtype == at::kBFloat16 || dtype == at::kHalf || dtype == at::kFloat);
}

const ExpertActivationCalls::Slot& ExpertActivationCalls::slot_of(int64_t layer, int64_t microbatch) {
    const auto key = std::make_pair(layer, microbatch);
    auto found = slots_.find(key);
    if (found == slots_.end()) {
        const auto resolved = resolve_(layer, microbatch).cast<py::tuple>();
        Slot slot{resolved[0].cast<int64_t>(), resolved[1].cast<int64_t>()};
        check_recorded_slot(recorded_, slot.index);
        found = slots_.emplace(key, slot).first;
    }
    return found->second;
}

void ExpertActivationCall::operator()(const py::object& intermediate, const py::object& counts) {
    if (calls_->active()) {
        calls_->observe(layer_, intermediate, counts);
    }
}
