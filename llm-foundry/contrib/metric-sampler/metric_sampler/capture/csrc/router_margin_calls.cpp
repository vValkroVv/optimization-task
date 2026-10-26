#include "router_margin_calls.h"
#include "observer_helpers.h"

#include <cstdint>
#include <string>
#include <utility>

RouterMarginCalls::RouterMarginCalls(py::object sampler, py::object recorder, py::object resolve, py::object graph,
                                     py::set layers, torch::Tensor recorded, py::object graph_task_id)
    : sampler_(std::move(sampler)),
      recorder_(std::move(recorder)),
      resolve_(std::move(resolve)),
      graph_(std::move(graph)),
      layers_(std::move(layers)),
      recorded_(std::move(recorded)),
      graph_task_id_(std::move(graph_task_id)) {
    check_recorded_flags(recorded_);
}

bool RouterMarginCalls::active(const py::object& module) const {
    return truthy(module, "training") && forward_capture_active(sampler_, graph_task_id_);
}

void RouterMarginCalls::observe(int64_t layer, const Router& router, const py::object& record,
                                const torch::Tensor& output) {
    const int64_t microbatch = sampler_.attr("microbatch").cast<int64_t>();
    // Computed now, so no layer's logits are kept until the end of the forward.
    const py::int_ key(layer);
    if (layers_.contains(key)) {
        throw py::value_error("router margin of layer " + std::to_string(layer) +
                              " was observed twice in microbatch " + std::to_string(microbatch));
    }
    layers_.add(key);
    const torch::Tensor logits = output.detach().reshape({-1, router.experts});
    if (logits.size(0) == 0 || !logits.is_cuda()) {
        record(output);
        return;
    }
    if (router.scoring_func != "softmax" && router.scoring_func != "sigmoid") {
        throw py::value_error("rank-local router margins require softmax or sigmoid");
    }
    if (router.experts <= router.top_k) {
        throw py::value_error("router margin has no unselected experts");
    }
    const py::object& graphed = graph_of(router, logits);
    const Slot& slot = slot_of(layer, microbatch);
    recorded_.data_ptr<bool>()[slot.index] = true;
    const torch::Tensor stats = graphed(logits).cast<torch::Tensor>();
    recorder_.attr("buffer")
        .cast<torch::Tensor>()
        .view(at::kFloat)
        .narrow(0, slot.offset, slot.width)
        .copy_(stats);
}

const py::object& RouterMarginCalls::graph_of(const Router& router, const torch::Tensor& logits) {
    const GraphKey key{logits.size(0), logits.size(1), static_cast<int>(logits.scalar_type()),
                       logits.get_device(), router.scoring_func, router.top_k, router.near_margin_count};
    auto found = graphs_.find(key);
    if (found == graphs_.end()) {
        found = graphs_
                    .emplace(key, graph_(router.scoring_func, router.top_k, router.near_margin_count, logits))
                    .first;
    }
    return found->second;
}

const RouterMarginCalls::Slot& RouterMarginCalls::slot_of(int64_t layer, int64_t microbatch) {
    const auto key = std::make_pair(layer, microbatch);
    auto found = slots_.find(key);
    if (found == slots_.end()) {
        const auto resolved = resolve_(layer, microbatch).cast<py::tuple>();
        Slot slot{resolved[0].cast<int64_t>(), resolved[1].cast<int64_t>(), resolved[2].cast<int64_t>()};
        check_recorded_slot(recorded_, slot.index);
        found = slots_.emplace(key, slot).first;
    }
    return found->second;
}

void RouterMarginCall::operator()(const py::object& module, const py::object& inputs, const torch::Tensor& output) {
    if (calls_->active(module)) {
        calls_->observe(layer_, router_, record_, output);
    }
}
