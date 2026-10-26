#pragma once

#include <torch/extension.h>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

// MetricSampler._observe_router_margin from C++ for every routed layer of a model, the forward hooks of its gates'
// linear layers being the RouterMarginCall callables this makes. A call is taken only from a module in training,
// while `sampler` captures the forward, and once per layer and microbatch: `layers` holds the layers observed in
// `sampler`'s microbatch. Non-empty CUDA logits are checked as _record_router_margin checks them and replayed
// through the CUDA graph graph(scoring_func, top_k, near_count, logits) returns for their shape, its eight statistics
// copied into the slot resolve(layer, microbatch) gives -- (offset, slot index) of router_margin -- and marked in
// the host flags `recorded`; any other logits go to the layer's `record`, _record_router_margin itself.
class RouterMarginCalls {
public:
    RouterMarginCalls(py::object sampler, py::object recorder, py::object resolve, py::object graph, py::set layers,
                      torch::Tensor recorded, py::object graph_task_id);

    bool active(const py::object& module) const;

    // A router's scoring function, top-k, near-margin count and expert count.
    struct Router {
        std::string scoring_func;
        int64_t top_k;
        int64_t near_margin_count;
        int64_t experts;
    };

    void observe(int64_t layer, const Router& router, const py::object& record, const torch::Tensor& output);

private:
    struct Slot {
        int64_t offset;
        int64_t width;
        int64_t index;
    };

    // MetricSampler._router_margin_graphs's key: the logits' shape, dtype and device, and the router.
    using GraphKey = std::tuple<int64_t, int64_t, int, int, std::string, int64_t, int64_t>;

    const py::object& graph_of(const Router& router, const torch::Tensor& logits);

    const Slot& slot_of(int64_t layer, int64_t microbatch);

    py::object sampler_;
    py::object recorder_;
    py::object resolve_;
    py::object graph_;
    py::set layers_;
    torch::Tensor recorded_;
    py::object graph_task_id_;
    std::map<GraphKey, py::object> graphs_;
    std::map<std::pair<int64_t, int64_t>, Slot> slots_;
};

// One routed layer's router margin hook.
class RouterMarginCall {
public:
    RouterMarginCall(std::shared_ptr<RouterMarginCalls> calls, int64_t layer, RouterMarginCalls::Router router,
                     py::object record)
        : calls_(std::move(calls)), layer_(layer), router_(std::move(router)), record_(std::move(record)) {}

    void operator()(const py::object& module, const py::object& inputs, const torch::Tensor& output);

private:
    std::shared_ptr<RouterMarginCalls> calls_;
    int64_t layer_;
    RouterMarginCalls::Router router_;
    py::object record_;
};
