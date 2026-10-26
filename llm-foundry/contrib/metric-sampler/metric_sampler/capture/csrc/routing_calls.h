#pragma once

#include "triton_launch.h"

#include <torch/extension.h>

#include <cstdint>
#include <map>
#include <memory>
#include <utility>

// MetricSampler._observe_routing from C++ for every routed layer of a model, its gates' routing observers being
// the RoutingCall callables this makes. CUDA top-k indices of a layer go, as TokenLossStats.observe_routing sends
// them, to the routing kernel of `kernels` (token_loss_kernel) through `launcher` (triton_launch), into the top-k
// buffer and the token-count, stream-count and entropy slots of the target token_loss.begin_routing(layer, tokens,
// device) gives, marked in the
// recorder's host flags `recorded`; with `quantiles` (the sampler's ExactQuantileStats, or None) float32 scores of
// a unit last stride then go to entropy_quantiles_into, claimed in `quantiles` and into the slots
// entropy_slots(layer, microbatch) gives -- (quantiles offset, quantiles slot index, counts offset, counts slot
// index) -- in `sampler`'s microbatch, and to copy_quantiles(layer, scores) when the population is too large for
// it; any other scores go to record_quantiles(layer, scores). Other top-k indices go to fallback(layer, topk_indices,
// scores), _observe_routing itself. `recorder` gives the step's word buffer and `experts` is the model's expert
// count.
class RoutingCalls {
public:
    RoutingCalls(py::object sampler, py::object token_loss, py::object quantiles, py::object recorder,
                 py::object entropy_slots, py::object copy_quantiles, py::object record_quantiles, py::object fallback,
                 py::object launcher, py::object kernels, int64_t experts, torch::Tensor recorded);

    void observe(int64_t layer, const torch::Tensor& topk_indices, const torch::Tensor& scores);

private:
    struct EntropySlots {
        int64_t quantiles_offset;
        int64_t quantiles_index;
        int64_t counts_offset;
        int64_t counts_index;
    };

    // The launch values Triton rounds up are 32-bit.
    static constexpr int kLaunchValueBits = 32;

    // triton.next_power_of_2.
    static int64_t next_power_of_2(int64_t value);

    void mark(int64_t slot);

    const EntropySlots& entropy_slots_of(int64_t layer, int64_t microbatch);

    py::object sampler_;
    py::object token_loss_;
    py::object quantiles_;
    py::object recorder_;
    py::object resolve_entropy_;
    py::object copy_quantiles_;
    py::object record_quantiles_;
    py::object fallback_;
    TritonLauncher launcher_;
    py::object kernel_;
    int64_t tile_rows_;
    int64_t histogram_tile_rows_;
    int64_t histogram_min_tokens_;
    int64_t bins_;
    double bins_start_;
    double bins_inverse_width_;
    int64_t stride_;
    int64_t experts_;
    torch::Tensor recorded_;
    std::map<std::pair<int64_t, int64_t>, EntropySlots> entropy_slots_;
};

// One routed layer's routing observer.
class RoutingCall {
public:
    RoutingCall(std::shared_ptr<RoutingCalls> calls, int64_t layer) : calls_(std::move(calls)), layer_(layer) {}

    void operator()(const torch::Tensor& topk_indices, const torch::Tensor& scores);

private:
    std::shared_ptr<RoutingCalls> calls_;
    int64_t layer_;
};
