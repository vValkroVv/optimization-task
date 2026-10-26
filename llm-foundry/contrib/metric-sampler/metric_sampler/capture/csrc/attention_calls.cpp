#include "attention_calls.h"
#include "cuda_host_functions.h"
#include "observer_helpers.h"
#include "triton_launch.h"

#include <ATen/cuda/CUDAContext.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

AttentionCalls::AttentionCalls(py::object sampler, py::object resolve, py::object launcher, py::object kernels,
                               torch::Tensor recorded, py::object graph_task_id)
    : sampler_(std::move(sampler)),
      resolve_(std::move(resolve)),
      launcher_(std::move(launcher)),
      kernels_(std::move(kernels)),
      recorded_(std::move(recorded)),
      graph_task_id_(std::move(graph_task_id)),
      softmax_kernel_(kernels_.attr("_softmax_stats_kernel")),
      gate_kernel_(kernels_.attr("_gate_stats_kernel")),
      linear_kernel_(kernels_.attr("_linear_stats_kernel")),
      tile_rows_(kernels_.attr("_TILE_ROWS").cast<int64_t>()),
      num_warps_(kernels_.attr("_NUM_WARPS").cast<int64_t>()),
      target_programs_(kernels_.attr("_TARGET_PROGRAMS").cast<int64_t>()),
      max_blocks_(kernels_.attr("_MAX_BLOCKS").cast<int64_t>()),
      max_programs_(kernels_.attr("_MAX_PROGRAMS").cast<int64_t>()),
      counter_words_(kernels_.attr("_COUNTER_WORDS").cast<int64_t>()) {
    check_recorded_flags(recorded_);
}

void AttentionCalls::begin_step(torch::Tensor words) {
    check_step_words(words);
    words_ = std::move(words);
    ++step_;
}

void AttentionCalls::softmax(int64_t layer, int64_t nope_dim, const torch::Tensor& query, const torch::Tensor& key,
                             const torch::Tensor& value, const torch::Tensor& lse) {
    const int64_t heads = lse.size(0);
    const int64_t rows = floor_div(query.numel(), heads * query.size(-1));
    if (lse.numel() != heads * rows) {
        throw py::value_error("the log-partition must hold one value per query row and head");
    }
    Point& point = point_of(Kind::kSoftmax, layer);
    const int64_t rows_before = claim(point, rows);
    const torch::Tensor query_rows = rows_of(query, heads);
    const torch::Tensor key_rows = rows_of(key, heads);
    const torch::Tensor value_rows = rows_of(value, heads);
    const torch::Tensor lse_rows = lse.dim() == 2 ? lse.unsqueeze(1) : lse;
    const int64_t rope_dim = query_rows.size(-1) - nope_dim;
    const int64_t launch_rows = query_rows.size(0);
    const int64_t lse_offset = point.offsets[0], qkv_offset = point.offsets[1];
    if (launch_rows == 0 || heads == 0) {
        kernels_.attr("launch_softmax_stats_kernel")(query, key, value, lse, words_, lse_offset, qkv_offset,
                                                     nope_dim, rows_before);
        return;
    }
    const Grid grid = grid_of(launch_rows, heads);
    LaunchArguments arguments;
    arguments.tensor(query_rows).tensor(key_rows).tensor(value_rows).tensor(lse_rows).tensor(words_)
        .tensor(scratch(query_rows))
        .integer(query_rows.stride(0)).integer(query_rows.stride(1))
        .integer(key_rows.stride(0)).integer(key_rows.stride(1))
        .integer(value_rows.stride(0)).integer(value_rows.stride(1))
        .integer(lse_rows.stride(0)).integer(lse_rows.stride(1)).integer(lse_rows.stride(2))
        .integer(lse_rows.size(2))
        .integer(launch_rows).integer(rows_before).integer(grid.rows_per_program).integer(grid.blocks)
        .integer(lse_offset).integer(qkv_offset);
    launcher_.launch(softmax_kernel_, grid.blocks, heads, arguments, {
        {"NOPE", nope_dim}, {"ROPE", rope_dim}, {"VDIM", value_rows.size(-1)},
        {"NOPE_PAD", next_power_of_2(nope_dim)}, {"ROPE_PAD", next_power_of_2(rope_dim)},
        {"V_PAD", next_power_of_2(value_rows.size(-1))}, {"TILE_ROWS", tile_rows_},
        {"BLOCKS_PAD", grid.blocks_pad},
        {"COUNTER_WORDS", counter_words_}, {"num_warps", num_warps_},
    });
}

void AttentionCalls::gate(int64_t layer, const torch::Tensor& logits) {
    const int64_t heads = logits.size(-2);
    Point& point = point_of(Kind::kGate, layer);
    const int64_t rows_before = claim(point, floor_div(logits.numel(), heads * logits.size(-1)));
    const torch::Tensor gate = rows_of(logits, heads);
    const int64_t rows = gate.size(0);
    const int64_t gate_offset = point.offsets[0];
    if (rows == 0 || heads == 0) {
        kernels_.attr("launch_gate_stats_kernel")(logits, words_, gate_offset, rows_before);
        return;
    }
    const Grid grid = grid_of(rows, heads);
    LaunchArguments arguments;
    arguments.tensor(gate).tensor(words_).tensor(scratch(gate)).integer(gate.stride(0)).integer(gate.stride(1))
        .integer(rows).integer(rows_before).integer(grid.rows_per_program).integer(grid.blocks)
        .integer(gate_offset);
    launcher_.launch(gate_kernel_, grid.blocks, heads, arguments, {
        {"DIM", gate.size(-1)}, {"DIM_PAD", next_power_of_2(gate.size(-1))}, {"TILE_ROWS", tile_rows_},
        {"BLOCKS_PAD", grid.blocks_pad}, {"COUNTER_WORDS", counter_words_}, {"num_warps", num_warps_},
    });
}

void AttentionCalls::linear(int64_t layer, const torch::Tensor& beta, const torch::Tensor& g,
                            const torch::Tensor& value, const torch::Tensor& output, const torch::Tensor& a_log,
                            const torch::Tensor& dt_bias, const c10::optional<torch::Tensor>& final_state) {
    const int64_t heads = beta.size(-1);
    Point& point = point_of(Kind::kLinear, layer);
    const int64_t rows_before = claim(point, floor_div(beta.numel(), heads));
    const int64_t gate_offset = point.offsets[0], norm_offset = point.offsets[1];
    const int64_t decay_offset = point.offsets.size() > 2 ? point.offsets[2] : -1;
    const torch::Tensor beta_rows = beta.reshape({-1, heads});
    const torch::Tensor gates = g.reshape({-1, heads});
    const torch::Tensor value_rows = rows_of(value, heads);
    const torch::Tensor output_rows = rows_of(output, heads);
    const int64_t rows = beta_rows.size(0);
    if (rows == 0 || heads == 0) {
        kernels_.attr("launch_linear_stats_kernel")(beta, g, value, output, a_log, dt_bias, words_, gate_offset,
                                                    norm_offset, decay_offset, rows_before);
    } else {
        const Grid grid = grid_of(rows, heads);
        LaunchArguments arguments;
        arguments.tensor(beta_rows).tensor(gates).tensor(value_rows).tensor(output_rows).tensor(a_log)
            .tensor(dt_bias).tensor(words_).tensor(scratch(beta_rows))
            .integer(beta_rows.stride(0)).integer(beta_rows.stride(1))
            .integer(gates.stride(0)).integer(gates.stride(1))
            .integer(value_rows.stride(0)).integer(value_rows.stride(1))
            .integer(output_rows.stride(0)).integer(output_rows.stride(1))
            .integer(rows).integer(rows_before).integer(grid.rows_per_program).integer(grid.blocks)
            .integer(gate_offset).integer(norm_offset).integer(decay_offset);
        launcher_.launch(linear_kernel_, grid.blocks, heads, arguments, {
            {"DIM", value_rows.size(-1)}, {"DIM_PAD", next_power_of_2(value_rows.size(-1))},
            {"TILE_ROWS", tile_rows_},
            {"BLOCKS_PAD", grid.blocks_pad}, {"COUNTER_WORDS", counter_words_}, {"num_warps", num_warps_},
        });
    }
    if (final_state.has_value()) {
        Point& state = point_of(Kind::kLinearState, layer);
        const int64_t documents_before = claim(state, final_state->size(0));
        linear_state_stats(*final_state, words_.narrow(0, state.offsets[0], state.widths[0]), documents_before > 0);
    }
}

int64_t AttentionCalls::cdiv(int64_t numerator, int64_t denominator) {
    return (numerator + denominator - 1) / denominator;
}

int64_t AttentionCalls::floor_div(int64_t numerator, int64_t denominator) {
    if (denominator == 0) {
        PyErr_SetString(PyExc_ZeroDivisionError, "integer division or modulo by zero");
        throw py::error_already_set();
    }
    return numerator / denominator;
}

int64_t AttentionCalls::next_power_of_2(int64_t value) {
    int64_t power = 1;
    while (power < value) {
        power <<= 1;
    }
    return power;
}

torch::Tensor AttentionCalls::rows_of(const torch::Tensor& tensor, int64_t heads) {
    const torch::Tensor rows = tensor.dim() != 3 ? tensor.reshape({-1, heads, tensor.size(-1)}) : tensor;
    return rows.stride(-1) == 1 ? rows : rows.contiguous();
}

AttentionCalls::Grid AttentionCalls::grid_of(int64_t rows, int64_t heads) const {
    if (heads > max_programs_) {
        throw py::value_error("attention statistics take at most " + std::to_string(max_programs_) +
                              " heads, got " + std::to_string(heads));
    }
    const int64_t first = std::max<int64_t>(
        1, std::min({max_blocks_, std::min(target_programs_, max_programs_) / heads, cdiv(rows, tile_rows_)}));
    const int64_t rows_per_program = cdiv(cdiv(rows, first), tile_rows_) * tile_rows_;
    const int64_t blocks = cdiv(rows, rows_per_program);
    return Grid{blocks, rows_per_program, next_power_of_2(blocks)};
}

AttentionCalls::Point& AttentionCalls::point_of(Kind kind, int64_t layer) {
    const int64_t microbatch = sampler_.attr("microbatch").cast<int64_t>();
    const auto key = std::make_tuple(static_cast<int>(kind), layer, microbatch);
    auto found = points_.find(key);
    if (found != points_.end()) {
        return found->second;
    }
    static const char* const kinds[] = {"softmax", "gate", "linear", "linear_state"};
    const auto resolved = resolve_(kinds[static_cast<int>(kind)], layer, microbatch).cast<py::tuple>();
    Point point;
    point.offsets = resolved[0].cast<std::vector<int64_t>>();
    point.widths = resolved[1].cast<std::vector<int64_t>>();
    point.slots = resolved[2].cast<std::vector<int64_t>>();
    for (const int64_t slot : point.slots) {
        check_recorded_slot(recorded_, slot);
    }
    return points_.emplace(key, std::move(point)).first->second;
}

int64_t AttentionCalls::claim(Point& point, int64_t rows) {
    if (point.step != step_) {
        point.step = step_;
        point.rows = 0;
    }
    const int64_t before = point.rows;
    point.rows += rows;
    bool* recorded = recorded_.data_ptr<bool>();
    for (const int64_t slot : point.slots) {
        recorded[slot] = true;
    }
    return before;
}

const torch::Tensor& AttentionCalls::scratch(const torch::Tensor& like) {
    const auto key =
        std::make_pair(static_cast<int64_t>(like.get_device()),
                       reinterpret_cast<uintptr_t>(at::cuda::getCurrentCUDAStream(like.get_device()).stream()));
    auto found = scratches_.find(key);
    if (found == scratches_.end()) {
        found = scratches_
                    .emplace(key, kernels_.attr("_scratch")(py::cast(like.device())).cast<torch::Tensor>())
                    .first;
    }
    return found->second;
}

static const torch::Tensor& tensor_of(const py::object& value) {
    if (!THPVariable_Check(value.ptr())) {
        throw py::type_error("an attention observer takes tensors");
    }
    return THPVariable_Unpack(value.ptr());
}

void SoftmaxAttentionCall::operator()(const py::object& query, const py::object& key, const py::object& value,
                                      const py::object& lse, const py::object& softmax_scale,
                                      const py::object& cu_seqlens, const py::object& max_seqlen) {
    if (!calls_->active()) {
        return;
    }
    calls_->softmax(layer_, nope_dim_, tensor_of(query), tensor_of(key), tensor_of(value), tensor_of(lse));
    rows_(query, key, value, lse, softmax_scale, cu_seqlens, max_seqlen);
}

void SoftmaxGateCall::operator()(const torch::Tensor& gate) {
    if (calls_->active()) {
        calls_->gate(layer_, gate);
    }
}

void LinearAttentionCall::operator()(const torch::Tensor& beta, const torch::Tensor& g, const torch::Tensor& value,
                                     const torch::Tensor& output, const torch::Tensor& a_log,
                                     const torch::Tensor& dt_bias, const c10::optional<torch::Tensor>& final_state) {
    if (calls_->active()) {
        calls_->linear(layer_, beta, g, value, output, a_log, dt_bias, final_state);
    }
}
