#include "router_balance_calls.h"
#include "cuda_host_functions.h"
#include "observer_helpers.h"

#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

RouterBalanceCalls::RouterBalanceCalls(py::object sampler, py::object recorder, py::object resolve, py::object fallback,
                                       py::dict python_populations, int64_t experts, torch::Tensor recorded,
                                       py::object nonbinary_flag, py::object capacity_error, py::object graph_task_id)
    : sampler_(std::move(sampler)),
      recorder_(std::move(recorder)),
      resolve_(std::move(resolve)),
      fallback_(std::move(fallback)),
      python_populations_(std::move(python_populations)),
      experts_(experts),
      recorded_(std::move(recorded)),
      nonbinary_flag_(std::move(nonbinary_flag)),
      capacity_error_(std::move(capacity_error)),
      graph_task_id_(std::move(graph_task_id)) {
    check_recorded_flags(recorded_);
}

bool RouterBalanceCalls::has_population(int64_t layer, int64_t microbatch) const {
    return populations_.count(std::make_pair(layer, microbatch)) > 0;
}

void RouterBalanceCalls::observe(int64_t layer, const torch::Tensor& scores, const torch::Tensor& biased_scores,
                                 const torch::Tensor& normalized_topk, const torch::Tensor& raw_topk,
                                 const torch::Tensor& biased_topk, const c10::optional<torch::Tensor>& valid,
                                 const c10::optional<torch::Tensor>& bias, const torch::Tensor& z_loss,
                                 const c10::optional<torch::Tensor>& seq_aux_loss, double alpha) {
    const int64_t microbatch = sampler_.attr("microbatch").cast<int64_t>();
    const bool native = scores.is_cuda() && scores.scalar_type() == at::kFloat &&
                        biased_scores.scalar_type() == at::kFloat && scores.dim() > 0 && scores.stride(-1) == 1 &&
                        scores.numel() > 0;
    const auto key = std::make_pair(layer, microbatch);
    if (!native) {
        if (populations_.count(key) > 0) {
            throw py::value_error("router layer " + std::to_string(layer) + " microbatch " +
                                  std::to_string(microbatch) + " has a call the extension took");
        }
        fallback_(layer, microbatch, scores, biased_scores, normalized_topk, raw_topk, biased_topk, valid, bias,
                  z_loss, seq_aux_loss, alpha);
        return;
    }
    if (python_populations_.contains(py::make_tuple(layer, microbatch))) {
        throw py::value_error("router layer " + std::to_string(layer) + " microbatch " +
                              std::to_string(microbatch) + " has a call the extension did not take");
    }
    // observe_router's checks, in its order.
    if (scores.dim() < 2 || scores.size(-1) != experts_ || !scores.is_floating_point()) {
        throw py::value_error("router scores must contain every expert for every source token");
    }
    if (biased_scores.sizes() != scores.sizes() || biased_scores.device() != scores.device() ||
        !biased_scores.is_floating_point()) {
        throw py::value_error("biased router scores must match the complete source scores");
    }
    const int64_t tokens = scores.numel() / experts_;
    auto found = populations_.find(key);
    Population* previous = found == populations_.end() ? nullptr : &found->second;
    const int64_t total_tokens = tokens + (previous == nullptr ? 0 : previous->tokens);
    const int64_t total_calls = previous == nullptr ? 1 : previous->calls + 1;
    if (total_tokens > std::numeric_limits<int32_t>::max() || total_calls > std::numeric_limits<int32_t>::max()) {
        PyErr_SetString(capacity_error_.ptr(), "router population exceeds Int32 token or call counts");
        throw py::error_already_set();
    }
    const torch::Device device = scores.device();
    if (valid.has_value()) {
        const auto dtype = valid->scalar_type();
        if ((dtype != at::kBool && dtype != at::kByte && dtype != at::kChar && dtype != at::kShort &&
             dtype != at::kInt && dtype != at::kLong) ||
            valid->device() != device) {
            throw py::value_error("router validity mask must be Boolean or binary integer on the source device");
        }
        if (valid->numel() != tokens) {
            throw py::value_error("router validity mask must cover every source token");
        }
    }
    for (const torch::Tensor* topk : {&normalized_topk, &raw_topk, &biased_topk}) {
        if (topk->dim() < 2 || topk->size(-1) == 0 || topk->numel() != tokens * topk->size(-1) ||
            topk->device() != device || !topk->is_floating_point()) {
            throw py::value_error("router top-k values must cover every source token on its device");
        }
    }
    if (bias.has_value() &&
        (bias->dim() != 1 || bias->size(0) != experts_ || bias->device() != device || !bias->is_floating_point())) {
        throw py::value_error("router bias must contain every expert on the source device");
    }
    const torch::Tensor words = recorder_.attr("buffer").cast<torch::Tensor>();
    const Slots& slots = slots_of(layer, microbatch);
    if (previous != nullptr) {
        const torch::Tensor current = bias.has_value()
                                          ? bias->detach().to(at::kFloat)
                                          : torch::zeros({experts_}, scores.options().dtype(at::kFloat));
        if (!at::equal(words.view(at::kFloat).narrow(0, slots.offsets[kBias], slots.widths[kBias]), current)) {
            throw py::value_error(
                "router bias changed within one microbatch; a single snapshot would lose information");
        }
    }
    Buffers& buffers = buffers_of(key, scores);
    bool* recorded = recorded_.data_ptr<bool>();
    for (const int64_t slot : slots.slots) {
        recorded[slot] = true;
    }
    const bool integer_mask = valid.has_value() && valid->scalar_type() != at::kBool;
    router_accumulate(scores, biased_scores, normalized_topk, raw_topk, biased_topk, valid, z_loss, seq_aux_loss,
                      alpha, buffers.values, buffers.valid_tokens, buffers.score_sums, previous == nullptr, words,
                      slots.offsets[kScoresMean], slots.offsets[kBalance], slots.offsets[kBalanceCounts],
                      slots.offsets[kBias], total_tokens, total_calls, bias,
                      integer_mask ? c10::optional<torch::Tensor>(nonbinary_on(device)) : c10::nullopt);
    Population& population =
        previous == nullptr ? populations_.emplace(key, Population{}).first->second : *previous;
    population.tokens = total_tokens;
    population.calls = total_calls;
}

const torch::Tensor& RouterBalanceCalls::nonbinary_on(const torch::Device& device) {
    if (!nonbinary_.defined()) {
        nonbinary_ = nonbinary_flag_(device).cast<torch::Tensor>();
    }
    return nonbinary_;
}

const RouterBalanceCalls::Slots& RouterBalanceCalls::slots_of(int64_t layer, int64_t microbatch) {
    const auto key = std::make_pair(layer, microbatch);
    auto found = slots_.find(key);
    if (found == slots_.end()) {
        const auto resolved = resolve_(layer, microbatch).cast<py::tuple>();
        Slots slots{resolved[0].cast<std::vector<int64_t>>(), resolved[1].cast<std::vector<int64_t>>(),
                    resolved[2].cast<std::vector<int64_t>>()};
        TORCH_CHECK(slots.offsets.size() == kSlotKinds && slots.slots.size() == kSlotKinds &&
                        slots.widths.size() == kSlotKinds,
                    "a router resolves to four slots");
        for (const int64_t slot : slots.slots) {
            check_recorded_slot(recorded_, slot);
        }
        found = slots_.emplace(key, std::move(slots)).first;
    }
    return found->second;
}

RouterBalanceCalls::Buffers& RouterBalanceCalls::buffers_of(const std::pair<int64_t, int64_t>& key,
                                                            const torch::Tensor& like) {
    auto found = buffers_.find(key);
    if (found == buffers_.end()) {
        const auto options = like.options();
        found = buffers_.emplace(key, Buffers{torch::empty({kBalanceSums}, options.dtype(at::kDouble)),
                                              torch::empty({}, options.dtype(at::kLong)),
                                              torch::empty({experts_}, options.dtype(at::kDouble))})
                    .first;
    }
    return found->second;
}

void RouterBalanceCall::operator()(const torch::Tensor& scores, const torch::Tensor& biased_scores,
                                   const torch::Tensor& normalized_topk, const torch::Tensor& raw_topk,
                                   const torch::Tensor& biased_topk, const c10::optional<torch::Tensor>& valid,
                                   const c10::optional<torch::Tensor>& bias, const torch::Tensor& z_loss,
                                   const c10::optional<torch::Tensor>& seq_aux_loss, double alpha) {
    if (calls_->active()) {
        calls_->observe(layer_, scores, biased_scores, normalized_topk, raw_topk, biased_topk, valid, bias, z_loss,
                        seq_aux_loss, alpha);
    }
}
