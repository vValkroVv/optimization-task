#include "routing_calls.h"
#include "cuda_host_functions.h"
#include "observer_helpers.h"
#include "triton_launch.h"

#include <cstdint>
#include <utility>

RoutingCalls::RoutingCalls(py::object sampler, py::object token_loss, py::object quantiles, py::object recorder,
                           py::object entropy_slots, py::object copy_quantiles, py::object record_quantiles,
                           py::object fallback, py::object launcher, py::object kernels, int64_t experts,
                           torch::Tensor recorded)
    : sampler_(std::move(sampler)),
      token_loss_(std::move(token_loss)),
      quantiles_(std::move(quantiles)),
      recorder_(std::move(recorder)),
      resolve_entropy_(std::move(entropy_slots)),
      copy_quantiles_(std::move(copy_quantiles)),
      record_quantiles_(std::move(record_quantiles)),
      fallback_(std::move(fallback)),
      launcher_(std::move(launcher)),
      kernel_(kernels.attr("_routing_kernel")),
      tile_rows_(kernels.attr("_ROUTING_TILE_ROWS").cast<int64_t>()),
      histogram_tile_rows_(experts <= kernels.attr("_ROUTING_HISTOGRAM_MAX_EXPERTS").cast<int64_t>()
                               ? kernels.attr("_ROUTING_HISTOGRAM_TILE_ROWS").cast<int64_t>()
                               : tile_rows_),
      histogram_min_tokens_(kernels.attr("_ROUTING_HISTOGRAM_MIN_TOKENS").cast<int64_t>()),
      bins_(kernels.attr("ROUTER_ENTROPY_BINS").attr("count").cast<int64_t>()),
      bins_start_(kernels.attr("ROUTER_ENTROPY_BINS").attr("start").cast<double>()),
      bins_inverse_width_(1.0 / kernels.attr("ROUTER_ENTROPY_BINS").attr("width").cast<double>()),
      stride_(kernels.attr("ENTROPY_TOKEN_STRIDE").cast<int64_t>()),
      experts_(experts),
      recorded_(std::move(recorded)) {
    check_recorded_flags(recorded_);
}

void RoutingCalls::observe(int64_t layer, const torch::Tensor& topk_indices, const torch::Tensor& scores) {
    if (!topk_indices.is_cuda() || topk_indices.dim() != 2) {
        fallback_(layer, topk_indices, scores);
        return;
    }
    // TokenLossStats.observe_routing and launch_routing_kernel.
    const int64_t tokens = topk_indices.size(0);
    const int64_t top_k = topk_indices.size(1);
    const py::object target = token_loss_.attr("begin_routing")(layer, tokens, topk_indices.device());
    const torch::Tensor detached = scores.detach();
    const torch::Tensor topk_buffer = target.attr("topk_buffer").cast<torch::Tensor>();
    const py::object counts = target.attr("counts");
    const py::object stream_counts = target.attr("stream_counts");
    const py::object entropy_histogram = target.attr("entropy_histogram");
    mark(counts.attr("slot_index").cast<int64_t>());
    mark(stream_counts.attr("slot_index").cast<int64_t>());
    mark(entropy_histogram.attr("slot_index").cast<int64_t>());
    const torch::Tensor words = recorder_.attr("buffer").cast<torch::Tensor>();
    LaunchArguments arguments;
    arguments.tensor(topk_indices)
        .integer(topk_indices.stride(0))
        .tensor(detached)
        .integer(detached.stride(0))
        .tensor(topk_buffer)
        .integer(topk_buffer.stride(0))
        .integer(topk_buffer.stride(1))
        .integer(target.attr("position").cast<int64_t>())
        .tensor(words)
        .integer(entropy_histogram.attr("offset").cast<int64_t>())
        .integer(counts.attr("offset").cast<int64_t>())
        .integer(stream_counts.attr("offset").cast<int64_t>())
        .tensor(target.attr("stream_ids").is_none() ? topk_indices
                                                     : target.attr("stream_ids").cast<torch::Tensor>())
        .integer(target.attr("tokens_per_sample").cast<int64_t>())
        .integer(tokens)
        .integer(target.attr("streams").cast<int64_t>())
        .integer(experts_)
        .integer(top_k)
        .real(bins_start_)
        .real(bins_inverse_width_);
    const int64_t tile_rows = tokens >= histogram_min_tokens_ ? histogram_tile_rows_ : tile_rows_;
    launcher_.launch(kernel_, (tokens + tile_rows - 1) / tile_rows, 1, arguments,
                     {{"BINS", bins_},
                      {"STRIDE", stride_},
                      {"TILE_ROWS", tile_rows},
                      {"BLOCK_E", next_power_of_2(experts_)},
                      {"BLOCK_K", next_power_of_2(top_k)}});
    if (quantiles_.is_none()) {
        return;
    }
    if (!detached.is_cuda() || detached.scalar_type() != at::kFloat || detached.dim() != 2 ||
        detached.numel() == 0 || detached.stride(-1) != 1) {
        record_quantiles_(layer, detached);
        return;
    }
    // One extension call reads the scores in place and writes the quantiles into their slots.
    const int64_t microbatch = sampler_.attr("microbatch").cast<int64_t>();
    const EntropySlots& slots = entropy_slots_of(layer, microbatch);
    quantiles_.attr("claim")("router/entropy_quantiles", layer, microbatch);
    mark(slots.quantiles_index);
    mark(slots.counts_index);
    if (!entropy_quantiles_into(detached, words, slots.quantiles_offset, slots.counts_offset)) {
        copy_quantiles_(layer, detached);
    }
}

int64_t RoutingCalls::next_power_of_2(int64_t value) {
    value -= 1;
    for (int shift = 1; shift <= kLaunchValueBits; shift *= 2) {
        value |= value >> shift;
    }
    return value + 1;
}

void RoutingCalls::mark(int64_t slot) {
    check_recorded_slot(recorded_, slot);
    recorded_.data_ptr<bool>()[slot] = true;
}

const RoutingCalls::EntropySlots& RoutingCalls::entropy_slots_of(int64_t layer, int64_t microbatch) {
    const auto key = std::make_pair(layer, microbatch);
    auto found = entropy_slots_.find(key);
    if (found == entropy_slots_.end()) {
        const auto resolved = resolve_entropy_(layer, microbatch).cast<py::tuple>();
        found = entropy_slots_
                    .emplace(key, EntropySlots{resolved[0].cast<int64_t>(), resolved[1].cast<int64_t>(),
                                               resolved[2].cast<int64_t>(), resolved[3].cast<int64_t>()})
                    .first;
    }
    return found->second;
}

void RoutingCall::operator()(const torch::Tensor& topk_indices, const torch::Tensor& scores) {
    calls_->observe(layer_, topk_indices, scores);
}
