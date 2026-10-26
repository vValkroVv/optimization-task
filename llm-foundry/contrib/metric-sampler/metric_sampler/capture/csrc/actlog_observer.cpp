#include "actlog_observer.h"
#include "cuda_host_functions.h"
#include "observer_helpers.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

ActlogObserver::ActlogObserver(py::object resolve_name, py::object resolve_point, py::object fallback,
                               torch::Tensor recorded, py::object capacity_error, py::object graph_task_id)
    : resolve_name_(std::move(resolve_name)),
      resolve_point_(std::move(resolve_point)),
      fallback_(std::move(fallback)),
      recorded_(std::move(recorded)),
      capacity_error_(std::move(capacity_error)),
      graph_task_id_(std::move(graph_task_id)) {
    check_recorded_flags(recorded_);
}

void ActlogObserver::begin_step(torch::Tensor words) {
    check_step_words(words);
    words_ = std::move(words);
    ++step_;
    microbatch_ = 0;
}

void ActlogObserver::flush() {
    claimed_.clear();
    claimed_microbatch_ = -1;
}

void ActlogObserver::observe(const torch::Tensor& tensor, const std::string& name, const std::string& suffix,
                             bool /*layer_depends*/, int64_t layer_index) {
    const Name& resolved = name_of(name, suffix);
    if (resolved.point < 0 || graph_task_id_().cast<int64_t>() != -1) {
        return;
    }
    const int64_t layer = resolved.layer_from_index ? layer_index : resolved.layer;
    Point& point = point_of(resolved.point, layer);
    const auto dtype = tensor.scalar_type();
    if (point.python || !tensor.is_cuda() || tensor.numel() == 0 ||
        (dtype != at::kBFloat16 && dtype != at::kHalf && dtype != at::kFloat)) {
        fallback_(points_[resolved.point], layer, tensor);
        return;
    }
    if (point.step != step_) {
        point.step = step_;
        point.elements = 0;
    }
    const int64_t elements = point.elements + tensor.numel();
    if (elements > std::numeric_limits<int32_t>::max()) {
        raise_capacity("activation " + points_[resolved.point] + " layer " + std::to_string(layer) +
                       " microbatch " + std::to_string(microbatch_) + " exceeds int32 counts");
    }
    point.elements = elements;
    if (claimed_microbatch_ >= 0 && claimed_microbatch_ != microbatch_) {
        raise_capacity("quantile populations must be flushed before the next microbatch");
    }
    if (!claimed_.insert({resolved.point, layer}).second) {
        throw py::value_error(point.metric + " layer " + std::to_string(layer) +
                              " was observed twice in microbatch " + std::to_string(microbatch_) +
                              "; exact quantiles of a repeated population are not supported");
    }
    claimed_microbatch_ = microbatch_;
    bool* recorded = recorded_.data_ptr<bool>();
    for (const int64_t slot : point.slots) {
        recorded[slot] = true;
    }
    const int64_t* arguments = point.arguments;
    observe_with_quantiles(tensor, words_, arguments[kMomentsOffset], arguments[kCountsOffset],
                           arguments[kAmaxOffset], arguments[kPowerSumsOffset], arguments[kBinsStart],
                           arguments[kBinsCount], arguments[kQuantilesOffset],
                           arguments[kQuantileCountsOffset], true);
}

const ActlogObserver::Name& ActlogObserver::name_of(const std::string& name, const std::string& suffix) {
    const std::string key = name + '\x1f' + suffix;
    auto found = names_.find(key);
    if (found != names_.end()) {
        return found->second;
    }
    Name resolved{-1, false, 0};
    const py::object result = resolve_name_(name, suffix);
    if (!result.is_none()) {
        const auto pair = result.cast<py::tuple>();
        const std::string point = pair[0].cast<std::string>();
        auto id = point_ids_.find(point);
        if (id == point_ids_.end()) {
            id = point_ids_.emplace(point, static_cast<int>(points_.size())).first;
            points_.push_back(point);
        }
        resolved.point = id->second;
        resolved.layer_from_index = pair[1].is_none();
        resolved.layer = resolved.layer_from_index ? 0 : pair[1].cast<int64_t>();
    }
    return names_.emplace(key, resolved).first->second;
}

ActlogObserver::Point& ActlogObserver::point_of(int point, int64_t layer) {
    const auto key = std::make_tuple(point, layer, microbatch_);
    auto found = states_.find(key);
    if (found != states_.end()) {
        return found->second;
    }
    Point state;
    const py::object result = resolve_point_(points_[point], layer);
    if (!result.is_none()) {
        const auto resolved = result.cast<py::tuple>();
        state.python = false;
        state.metric = resolved[0].cast<std::string>();
        const auto arguments = resolved[1].cast<std::vector<int64_t>>();
        TORCH_CHECK(arguments.size() == kArguments, "a point resolves to eight observe_with_quantiles arguments");
        std::copy(arguments.begin(), arguments.end(), state.arguments);
        state.slots = resolved[2].cast<std::vector<int64_t>>();
        for (const int64_t slot : state.slots) {
            check_recorded_slot(recorded_, slot);
        }
    }
    return states_.emplace(key, std::move(state)).first->second;
}

void ActlogObserver::raise_capacity(const std::string& message) const {
    PyErr_SetString(capacity_error_.ptr(), message.c_str());
    throw py::error_already_set();
}
