#pragma once

#include "observer_helpers.h"

#include <torch/extension.h>

#include <cstdint>
#include <map>
#include <memory>
#include <utility>
#include <vector>

// ExpertBalanceStats.observe_router from C++ for every routed layer of a model, its gates' balance observers being
// the RouterBalanceCall callables this makes. A call is taken only while `sampler` captures the forward, in
// `sampler`'s microbatch, and checked as observe_router checks it. A call the extension takes -- float32 scores and
// biased scores on CUDA with a unit last stride -- goes to router_accumulate into its population of the
// microbatch, whose sums live here, the recorder's slots resolved once through resolve(layer, microbatch) --
// (offsets, widths, slot indices) of scores_mean, router_balance, router_balance_counts and router_bias -- and
// marked in the host flags `recorded`; any other call goes to fallback(layer, microbatch, operands...),
// observe_router itself, whose populations are python_populations, keyed by (layer, microbatch). A population is
// taken by one side only: a call of the other side for it raises. `recorder` gives the step's word buffer,
// nonbinary_flag(device) the flag of an integer mask other than 0 or 1; flush ends the microbatch's populations.
class RouterBalanceCalls {
public:
    RouterBalanceCalls(py::object sampler, py::object recorder, py::object resolve, py::object fallback,
                       py::dict python_populations, int64_t experts, torch::Tensor recorded, py::object nonbinary_flag,
                       py::object capacity_error, py::object graph_task_id);

    bool active() const { return forward_capture_active(sampler_, graph_task_id_); }

    void flush() { populations_.clear(); }

    bool has_population(int64_t layer, int64_t microbatch) const;

    void observe(int64_t layer, const torch::Tensor& scores, const torch::Tensor& biased_scores,
                 const torch::Tensor& normalized_topk, const torch::Tensor& raw_topk, const torch::Tensor& biased_topk,
                 const c10::optional<torch::Tensor>& valid, const c10::optional<torch::Tensor>& bias,
                 const torch::Tensor& z_loss, const c10::optional<torch::Tensor>& seq_aux_loss, double alpha);

private:
    // The recorder slots of a router population, in the order resolve(layer, microbatch) gives them.
    enum SlotKind { kScoresMean, kBalance, kBalanceCounts, kBias, kSlotKinds };
    // The float64 sums of a router population: its top-k sums, entropy and auxiliary losses.
    static constexpr int64_t kBalanceSums = 7;

    struct Slots {
        std::vector<int64_t> offsets;
        std::vector<int64_t> widths;
        std::vector<int64_t> slots;
    };

    struct Buffers {
        torch::Tensor values;
        torch::Tensor valid_tokens;
        torch::Tensor score_sums;
    };

    struct Population {
        int64_t tokens = 0;
        int64_t calls = 0;
    };

    // ExpertBalanceStats._nonbinary_flag_on: made on the first integer mask, which is rare.
    const torch::Tensor& nonbinary_on(const torch::Device& device);

    const Slots& slots_of(int64_t layer, int64_t microbatch);

    // ExpertBalanceStats._router_buffers: the population sums of a layer and microbatch, reused every step.
    Buffers& buffers_of(const std::pair<int64_t, int64_t>& key, const torch::Tensor& like);

    py::object sampler_;
    py::object recorder_;
    py::object resolve_;
    py::object fallback_;
    py::dict python_populations_;
    int64_t experts_;
    torch::Tensor recorded_;
    py::object nonbinary_flag_;
    torch::Tensor nonbinary_;
    py::object capacity_error_;
    py::object graph_task_id_;
    std::map<std::pair<int64_t, int64_t>, Slots> slots_;
    std::map<std::pair<int64_t, int64_t>, Buffers> buffers_;
    std::map<std::pair<int64_t, int64_t>, Population> populations_;
};

// One routed layer's balance observer.
class RouterBalanceCall {
public:
    RouterBalanceCall(std::shared_ptr<RouterBalanceCalls> calls, int64_t layer)
        : calls_(std::move(calls)), layer_(layer) {}

    void operator()(const torch::Tensor& scores, const torch::Tensor& biased_scores,
                    const torch::Tensor& normalized_topk, const torch::Tensor& raw_topk,
                    const torch::Tensor& biased_topk, const c10::optional<torch::Tensor>& valid,
                    const c10::optional<torch::Tensor>& bias, const torch::Tensor& z_loss,
                    const c10::optional<torch::Tensor>& seq_aux_loss, double alpha);

private:
    std::shared_ptr<RouterBalanceCalls> calls_;
    int64_t layer_;
};
