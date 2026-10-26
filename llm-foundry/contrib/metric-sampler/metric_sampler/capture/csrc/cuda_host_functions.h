#pragma once

#include <torch/extension.h>

#include <cstdint>
#include <tuple>
#include <vector>

torch::Tensor observe(
    torch::Tensor tensor,
    torch::Tensor buffer,
    int64_t moments_offset,
    int64_t counts_offset,
    int64_t amax_offset,
    int64_t power_sums_offset,
    int64_t bins_start,
    int64_t bins_count,
    bool quantile_first_pass,
    bool key_histogram);

void exact_quantiles(
    torch::Tensor population, c10::optional<torch::Tensor> mask, torch::Tensor quantiles, torch::Tensor counts,
    c10::optional<torch::Tensor> counted);

void observe_with_quantiles(
    torch::Tensor tensor,
    torch::Tensor buffer,
    int64_t moments_offset,
    int64_t counts_offset,
    int64_t amax_offset,
    int64_t power_sums_offset,
    int64_t bins_start,
    int64_t bins_count,
    int64_t quantiles_offset,
    int64_t quantile_counts_offset,
    bool key_histogram);

c10::optional<std::tuple<torch::Tensor, torch::Tensor>> exact_quantiles_float64(torch::Tensor values);

torch::Tensor order_statistics(torch::Tensor values, std::vector<int64_t> ranks);

torch::Tensor row_norm_stats(torch::Tensor tensor);

c10::optional<torch::Tensor> row_norm_accumulate(
    torch::Tensor tensor, torch::Tensor accumulator, int64_t slot, bool keep_stats);

void accumulate_row_norm_stats(torch::Tensor stats, torch::Tensor accumulator, int64_t slot);

// row_norm_accumulate of `tensor` into each of `slots` in order, in one launch for up to eight of them.
void row_norm_accumulate_slots(
    const torch::Tensor& tensor, const torch::Tensor& accumulator, const std::vector<int64_t>& slots);

// A floating scalar into each of `slots` as amax_and_row_norm_sum takes it: one row, its magnitude both statistics.
void scalar_accumulate_slots(
    const torch::Tensor& tensor, const torch::Tensor& accumulator, const std::vector<int64_t>& slots);

void linear_state_stats(torch::Tensor state, torch::Tensor slot, bool merge);

void row_norm_accumulate_batch(
    std::vector<torch::Tensor> tensors, torch::Tensor accumulator, std::vector<int64_t> slot_counts,
    std::vector<int64_t> slots);

torch::Tensor weight_norms(std::vector<torch::Tensor> weights);

bool weight_norms_into(
    std::vector<torch::Tensor> weights, std::vector<int64_t> experts, torch::Tensor buffer,
    std::vector<int64_t> offsets, std::vector<int64_t> launch_numels);

void segment_sumsq_into(
    torch::Tensor matrix, std::vector<int64_t> counts, torch::Tensor words, int64_t offset, bool accumulate);

torch::Tensor column_sums(torch::Tensor matrix);

torch::Tensor row_top_values(torch::Tensor matrix, int64_t k);

torch::Tensor row_entropy(torch::Tensor matrix);

bool entropy_quantiles_into(torch::Tensor matrix, torch::Tensor words, int64_t quantiles_offset, int64_t counts_offset);

void copy_counts_into(std::vector<torch::Tensor> sources, torch::Tensor words, std::vector<int64_t> offsets);

void monitor_values_into(torch::Tensor accumulator, torch::Tensor counts, torch::Tensor words, torch::Tensor targets);

torch::Tensor row_margins(torch::Tensor matrix, torch::Tensor boundary);

bool router_margin_into(
    torch::Tensor logits, int64_t top_k, int64_t near_margin_count, torch::Tensor buffer, int64_t offset);

torch::Tensor entropy_sum(torch::Tensor values);

void router_accumulate(
    torch::Tensor scores, torch::Tensor biased_scores, torch::Tensor normalized_topk, torch::Tensor raw_topk,
    torch::Tensor biased_topk, c10::optional<torch::Tensor> mask, torch::Tensor z_loss,
    c10::optional<torch::Tensor> seq_aux_loss, double alpha, torch::Tensor values, torch::Tensor valid_tokens,
    torch::Tensor score_sums, bool first, c10::optional<torch::Tensor> words, int64_t mean_offset,
    int64_t balance_offset, int64_t counts_offset, int64_t bias_offset, int64_t tokens_total, int64_t calls_total,
    c10::optional<torch::Tensor> expert_bias, c10::optional<torch::Tensor> nonbinary);
