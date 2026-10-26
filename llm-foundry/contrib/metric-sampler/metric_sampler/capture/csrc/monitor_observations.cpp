#include "monitor_observations.h"
#include "cuda_host_functions.h"
#include "observer_helpers.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

// How the extension takes an observed tensor: as rows (bfloat16, float16 or float32 with a last axis), as one row
// (a floating scalar), or not at all (another dtype or device, a tensor subclass), in which case Python takes it.
enum class Reduction { kRows, kScalar, kPython };

Reduction reduction_of(const torch::Tensor& value, const torch::Tensor& accumulator) {
    if (!value.is_cuda() || value.device() != accumulator.device() || value.key_set().has(c10::DispatchKey::Python)) {
        return Reduction::kPython;
    }
    if (value.dim() == 0) {
        return value.is_floating_point() ? Reduction::kScalar : Reduction::kPython;
    }
    const auto dtype = value.scalar_type();
    return dtype == at::kBFloat16 || dtype == at::kHalf || dtype == at::kFloat ? Reduction::kRows : Reduction::kPython;
}

}  // namespace

MonitorObservations::MonitorObservations(torch::Tensor accumulator, torch::Tensor counts, int64_t batch,
                                         py::object forward_fallback, py::object gradient_fallback,
                                         py::object skipped_type)
    : batch_(batch), forward_fallback_(std::move(forward_fallback)), skipped_type_(std::move(skipped_type)) {
    TORCH_CHECK(accumulator.is_cuda() && accumulator.scalar_type() == torch::kFloat64 &&
                    accumulator.is_contiguous() && accumulator.dim() == 2 &&
                    accumulator.size(1) == kValuesPerSlot,
                "accumulator must be a contiguous float64 [slots, 2] CUDA tensor");
    TORCH_CHECK(counts.device().is_cpu() && counts.scalar_type() == torch::kInt64 && counts.is_contiguous() &&
                    counts.sizes() == accumulator.sizes(),
                "counts must be a contiguous host int64 tensor shaped like the accumulator");
    TORCH_CHECK(batch >= 1, "a batch holds at least one tensor");
    state_ = std::make_shared<State>(std::move(accumulator), std::move(counts), std::move(gradient_fallback));
}

void MonitorObservations::observe_module(py::handle inputs, py::handle output, const std::vector<int64_t>& slots) {
    size_t position = 0;
    for (py::handle value : inputs) {
        observe_position(value, slots, position++, "input");
    }
    if (THPVariable_Check(output.ptr())) {
        observe_position(output, slots, position, "output");
        return;
    }
    for (py::handle value : output) {
        observe_position(value, slots, position++, "output");
    }
}

int64_t MonitorObservations::flush() {
    std::vector<Queued> queued;
    queued.swap(queue_);
    queued_index_.clear();
    queued_slots_.clear();
    for (const Queued& entry : queued) {
        if (static_cast<int64_t>(entry.value._version()) != entry.version) {
            return entry.slots.front();
        }
    }
    std::vector<torch::Tensor> values;
    std::vector<int64_t> slot_counts, slots;
    for (const Queued& entry : queued) {
        const torch::Tensor& value = entry.value;
        const Reduction reduction = reduction_of(value, state_->accumulator);
        if (reduction == Reduction::kScalar) {
            scalar_accumulate_slots(value, state_->accumulator, entry.slots);
            continue;
        }
        if (reduction == Reduction::kPython) {
            forward_fallback_(value, entry.slots);
            continue;
        }
        values.push_back(value);
        slot_counts.push_back(static_cast<int64_t>(entry.slots.size()));
        slots.insert(slots.end(), entry.slots.begin(), entry.slots.end());
    }
    if (!values.empty()) {
        row_norm_accumulate_batch(values, state_->accumulator, slot_counts, slots);
    }
    return -1;
}

MonitorObservations::State::~State() {
    if (Py_IsInitialized()) {
        py::gil_scoped_acquire gil;
        fallback = py::object();
    } else {
        fallback.release();
    }
}

void MonitorObservations::State::check_slot(int64_t slot) const {
    TORCH_CHECK(slot >= 0 && slot < accumulator.size(0), "slot ", slot, " is outside the accumulator");
}

void MonitorObservations::State::count(int64_t slot, const torch::Tensor& value) {
    int64_t* words = counts.data_ptr<int64_t>();
    words[kValuesPerSlot * slot] += 1;
    words[kValuesPerSlot * slot + 1] += value.dim() == 0 ? 1 : value.numel() / value.size(-1);
}

void MonitorObservations::State::observe(int64_t watch, const torch::Tensor& gradient) {
    py::gil_scoped_acquire gil;
    if (!gradient.defined() || gradient.numel() == 0) {
        return;
    }
    const std::vector<int64_t>& watched = slots[watch];
    const Reduction reduction = reduction_of(gradient, accumulator);
    if (reduction == Reduction::kPython) {
        fallback(watched, gradient);
        return;
    }
    if (reduction == Reduction::kScalar) {
        scalar_accumulate_slots(gradient, accumulator, watched);
    } else {
        row_norm_accumulate_slots(gradient, accumulator, watched);
    }
    for (const int64_t slot : watched) {
        count(slot, gradient);
    }
}

void MonitorObservations::observe_position(py::handle object, const std::vector<int64_t>& slots, size_t position,
                                           const char* kind) {
    if (!THPVariable_Check(object.ptr())) {
        return;
    }
    const torch::Tensor& value = THPVariable_Unpack(object.ptr());
    if (!value.is_floating_point()) {
        return;
    }
    TORCH_CHECK(kSlotsPerPosition * position + 1 < slots.size(), "the slots do not cover monitored ", kind, " ",
                position);
    const int64_t forward = slots[kSlotsPerPosition * position], gradient = slots[kSlotsPerPosition * position + 1];
    if (forward != kSkipped && value.numel() > 0 && !skipped(object)) {
        require_listed(forward, kind, position);
        observe_forward(value, forward);
    }
    if (gradient != kSkipped && value.requires_grad()) {
        require_listed(gradient, kind, position);
        watch(value, gradient);
    }
}

bool MonitorObservations::skipped(py::handle object) const {
    if (skipped_type_.is_none()) {
        return false;
    }
    const int instance = PyObject_IsInstance(object.ptr(), skipped_type_.ptr());
    if (instance < 0) {
        throw py::error_already_set();
    }
    return instance == 1;
}

void MonitorObservations::require_listed(int64_t slot, const char* kind, size_t position) {
    if (slot == kUnlisted) {
        throw py::key_error(std::string("monitored ") + kind + " " + std::to_string(position) +
                            " has no activation catalog entry");
    }
}

void MonitorObservations::observe_forward(const torch::Tensor& value, int64_t slot) {
    state_->check_slot(slot);
    state_->count(slot, value);
    const c10::TensorImpl* impl = value.unsafeGetTensorImpl();
    auto found = queued_index_.find(impl);
    if (queued_slots_.count(slot) != 0 ||
        (found == queued_index_.end() && static_cast<int64_t>(queue_.size()) == batch_)) {
        const int64_t modified = flush();
        TORCH_CHECK(modified < 0, "monitored activation slot ", modified,
                    " was modified in place before its row norms were taken");
        found = queued_index_.end();
    }
    if (found == queued_index_.end()) {
        queued_index_.emplace(impl, queue_.size());
        queue_.push_back(Queued{value, static_cast<int64_t>(value._version()), {slot}});
    } else {
        queue_[found->second].slots.push_back(slot);
    }
    queued_slots_.insert(slot);
}

void MonitorObservations::watch(const torch::Tensor& value, int64_t slot) {
    state_->check_slot(slot);
    const c10::TensorImpl* impl = value.unsafeGetTensorImpl();
    auto found = watched_.find(impl);
    if (found != watched_.end() && found->second.value.lock().get() == impl) {
        state_->slots[found->second.watch].push_back(slot);
        return;
    }
    const int64_t index = static_cast<int64_t>(state_->slots.size());
    state_->slots.push_back({slot});
    value.register_hook(
        [state = state_, index](const torch::Tensor& gradient) { state->observe(index, gradient); });
    watched_.insert_or_assign(
        impl, Watched{c10::weak_intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl>(value.getIntrusivePtr()),
                      index});
}

void MonitorModuleHook::operator()(const py::object& module, const py::object& inputs, const py::object& output) {
    if (!truthy(module, "training") || !forward_capture_active(sampler_, graph_task_id_)) {
        return;
    }
    const py::object observations = sampler_.attr("_monitor_observations");
    if (observations.is_none()) {
        sampler_.attr("_observe_monitor_module")(name_, module, inputs, output);
        return;
    }
    const bool single = THPVariable_Check(output.ptr());
    const auto key = std::make_pair(py::len(inputs), single ? size_t{1} : py::len(output));
    auto found = slots_.find(key);
    if (found == slots_.end()) {
        const py::object outputs = single ? py::object(py::make_tuple(output)) : output;
        found = slots_
                    .emplace(key,
                             sampler_.attr("_monitor_slots")(name_, inputs, outputs).cast<std::vector<int64_t>>())
                    .first;
    }
    observations.cast<MonitorObservations&>().observe_module(inputs, output, found->second);
}
