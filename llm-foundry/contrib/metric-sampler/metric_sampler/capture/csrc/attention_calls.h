#pragma once

#include "observer_helpers.h"
#include "triton_launch.h"

#include <torch/extension.h>

#include <cstdint>
#include <map>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

// The calls of a model's attention layers as AttentionStats observes them on CUDA -- observe_softmax,
// observe_softmax_gate, observe_linear and observe_linear_state -- from C++, each layer's observer a callable this
// makes (SoftmaxAttentionCall, SoftmaxGateCall, LinearAttentionCall). A call is taken only while `sampler` captures
// the forward, in `sampler`'s microbatch. The slots of a call's point are resolved once per layer and microbatch
// through resolve(kind, layer, microbatch) -- (offsets, widths, slot indices) -- and marked in the recorder's host
// flags `recorded`; begin_step starts a step that writes into the float32 `words`. The kernels of `kernels`
// (attention_kernel) are launched through `launcher` (triton_launch); a call without rows or heads goes to the
// kernels' launch functions, which reject it.
class AttentionCalls {
public:
    AttentionCalls(py::object sampler, py::object resolve, py::object launcher, py::object kernels,
                   torch::Tensor recorded, py::object graph_task_id);

    void begin_step(torch::Tensor words);

    bool active() const { return forward_capture_active(sampler_, graph_task_id_); }

    // observe_softmax and launch_softmax_stats_kernel.
    void softmax(int64_t layer, int64_t nope_dim, const torch::Tensor& query, const torch::Tensor& key,
                 const torch::Tensor& value, const torch::Tensor& lse);

    // observe_softmax_gate and launch_gate_stats_kernel.
    void gate(int64_t layer, const torch::Tensor& logits);

    // observe_linear and launch_linear_stats_kernel, then observe_linear_state for a final state.
    void linear(int64_t layer, const torch::Tensor& beta, const torch::Tensor& g, const torch::Tensor& value,
                const torch::Tensor& output, const torch::Tensor& a_log, const torch::Tensor& dt_bias,
                const c10::optional<torch::Tensor>& final_state);

private:
    enum class Kind { kSoftmax, kGate, kLinear, kLinearState };

    struct Point {
        std::vector<int64_t> offsets;
        std::vector<int64_t> widths;
        std::vector<int64_t> slots;
        int64_t step = -1;
        int64_t rows = 0;
    };

    struct Grid {
        int64_t blocks;
        int64_t rows_per_program;
        int64_t blocks_pad;
    };

    static int64_t cdiv(int64_t numerator, int64_t denominator);

    // Python's // of the non-negative sizes it divides, which raises on a zero divisor.
    static int64_t floor_div(int64_t numerator, int64_t denominator);

    static int64_t next_power_of_2(int64_t value);

    static torch::Tensor rows_of(const torch::Tensor& tensor, int64_t heads);

    // attention_kernel._grid.
    Grid grid_of(int64_t rows, int64_t heads) const;

    Point& point_of(Kind kind, int64_t layer);

    // AttentionStats._claim: the rows the step has put in the point's slots before these, which it marks recorded.
    int64_t claim(Point& point, int64_t rows);

    // attention_kernel._scratch: the records and counters of the launches on the device's current stream.
    const torch::Tensor& scratch(const torch::Tensor& like);

    py::object sampler_;
    py::object resolve_;
    TritonLauncher launcher_;
    py::object kernels_;
    torch::Tensor recorded_;
    py::object graph_task_id_;
    py::object softmax_kernel_;
    py::object gate_kernel_;
    py::object linear_kernel_;
    int64_t tile_rows_;
    int64_t num_warps_;
    int64_t target_programs_;
    int64_t max_blocks_;
    int64_t max_programs_;
    int64_t counter_words_;
    torch::Tensor words_;
    int64_t step_ = 0;
    std::map<std::tuple<int, int64_t, int64_t>, Point> points_;
    std::map<std::pair<int64_t, uintptr_t>, torch::Tensor> scratches_;
};

// One MLA layer's attention observer: its call's statistics as AttentionCalls.softmax takes them, then its row
// statistics through `rows`, which the call's arguments are passed to as they came.
class SoftmaxAttentionCall {
public:
    SoftmaxAttentionCall(std::shared_ptr<AttentionCalls> calls, int64_t layer, int64_t nope_dim, py::object rows)
        : calls_(std::move(calls)), layer_(layer), nope_dim_(nope_dim), rows_(std::move(rows)) {}

    void operator()(const py::object& query, const py::object& key, const py::object& value, const py::object& lse,
                    const py::object& softmax_scale, const py::object& cu_seqlens, const py::object& max_seqlen);

private:
    std::shared_ptr<AttentionCalls> calls_;
    int64_t layer_;
    int64_t nope_dim_;
    py::object rows_;
};

// One gated MLA layer's gate observer.
class SoftmaxGateCall {
public:
    SoftmaxGateCall(std::shared_ptr<AttentionCalls> calls, int64_t layer) : calls_(std::move(calls)), layer_(layer) {}

    void operator()(const torch::Tensor& gate);

private:
    std::shared_ptr<AttentionCalls> calls_;
    int64_t layer_;
};

// One GDN layer's attention observer.
class LinearAttentionCall {
public:
    LinearAttentionCall(std::shared_ptr<AttentionCalls> calls, int64_t layer)
        : calls_(std::move(calls)), layer_(layer) {}

    void operator()(const torch::Tensor& beta, const torch::Tensor& g, const torch::Tensor& value,
                    const torch::Tensor& output, const torch::Tensor& a_log, const torch::Tensor& dt_bias,
                    const c10::optional<torch::Tensor>& final_state);

private:
    std::shared_ptr<AttentionCalls> calls_;
    int64_t layer_;
};
