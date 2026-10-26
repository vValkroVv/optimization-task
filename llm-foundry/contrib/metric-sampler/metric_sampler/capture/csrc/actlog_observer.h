#pragma once

#include <torch/extension.h>

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

// The actlog observer of a training forward with exact quantiles, as ActivationStats.observe with
// training_forward observes a call. A call (tensor, name, suffix, layer_depends, layer_index) resolves its name and
// suffix once through `resolve_name` (a point and its fixed layer, None for the layer_index, or None for a call that
// is not recorded) and its point, layer and microbatch once through `resolve_point` (the quantile metric, the
// observe_with_quantiles offset and bin arguments and the slots that call writes, or None for a point Python
// takes). It skips a backward recomputation, counts the population's elements per step, claims its quantile
// population for the microbatch, marks its slots in the host `recorded` flags and makes the one statistics and
// selection launch. A tensor the extension does not take (another dtype or device, an empty one) goes to
// `fallback(point, layer, tensor)`. begin_step, begin_microbatch and flush follow ActivationStats.
class ActlogObserver {
public:
    ActlogObserver(py::object resolve_name, py::object resolve_point, py::object fallback, torch::Tensor recorded,
                   py::object capacity_error, py::object graph_task_id);

    void begin_step(torch::Tensor words);

    void begin_microbatch(int64_t microbatch) { microbatch_ = microbatch; }

    void flush();

    void observe(const torch::Tensor& tensor, const std::string& name, const std::string& suffix,
                 bool /*layer_depends*/, int64_t layer_index);

private:
    // The offset and bin arguments of observe_with_quantiles that a point resolves to, in its order.
    enum Argument {
        kMomentsOffset,
        kCountsOffset,
        kAmaxOffset,
        kPowerSumsOffset,
        kBinsStart,
        kBinsCount,
        kQuantilesOffset,
        kQuantileCountsOffset,
        kArguments
    };

    struct Name {
        int point;  // -1: not recorded
        bool layer_from_index;
        int64_t layer;
    };

    struct Point {
        bool python = true;
        std::string metric;
        int64_t arguments[kArguments] = {};
        std::vector<int64_t> slots;
        int64_t step = -1;
        int64_t elements = 0;
    };

    const Name& name_of(const std::string& name, const std::string& suffix);

    Point& point_of(int point, int64_t layer);

    [[noreturn]] void raise_capacity(const std::string& message) const;

    py::object resolve_name_;
    py::object resolve_point_;
    py::object fallback_;
    torch::Tensor recorded_;
    py::object capacity_error_;
    py::object graph_task_id_;
    torch::Tensor words_;
    int64_t step_ = 0;
    int64_t microbatch_ = 0;
    std::unordered_map<std::string, Name> names_;
    std::unordered_map<std::string, int> point_ids_;
    std::vector<std::string> points_;
    std::map<std::tuple<int, int64_t, int64_t>, Point> states_;
    std::set<std::pair<int, int64_t>> claimed_;
    int64_t claimed_microbatch_ = -1;
};
