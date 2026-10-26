#include "actlog_observer.h"
#include "attention_calls.h"
#include "cuda_host_functions.h"
#include "expert_activation_calls.h"
#include "materialized_weight_observer.h"
#include "monitor_observations.h"
#include "router_balance_calls.h"
#include "router_margin_calls.h"
#include "routing_calls.h"
#include "triton_launch.h"

#include <torch/extension.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<MonitorModuleHook>(m, "MonitorModuleHook")
        .def(py::init<py::object, std::string, py::object>(),
             "A monitored module's forward hook for MetricSampler `sampler` under the module's `name`; graph_task_id "
             "is torch's current graph task",
             py::arg("sampler"), py::arg("name"), py::arg("graph_task_id"))
        .def("__call__", &MonitorModuleHook::operator(), "Observes one call of the module", py::arg("module"),
             py::arg("inputs"), py::arg("output"));
    m.def("triton_arguments", &triton_arguments,
          "(specialization, values) of a Triton kernel's runtime arguments as triton_launch._variant makes them: per "
          "tensor its dtype and 16-byte alignment, per integer or float whether it is divisible by 16 or equal to 1; "
          "a tensor's address and any other argument as it is",
          py::arg("args"));
    py::class_<TritonKernel>(m, "TritonKernel")
        .def(py::init<uint64_t, int, int, const std::vector<std::string>&>(),
             "The launch of the compiled Triton kernel `function` with num_warps warps and `shared` bytes of shared "
             "memory; kinds is the Triton type of each runtime argument, \"\" for a constant the launch leaves out",
             py::arg("function"), py::arg("num_warps"), py::arg("shared"), py::arg("kinds"))
        .def("__call__", &TritonKernel::operator(),
             "Launches the kernel on the current stream with one value per runtime argument, a tensor as its address",
             py::arg("grid_x"), py::arg("grid_y"), py::arg("grid_z"), py::arg("values"));
    py::class_<ActlogObserver>(m, "ActlogObserver")
        .def(py::init<py::object, py::object, py::object, torch::Tensor, py::object, py::object>(),
             "The actlog observer of a training forward with exact quantiles: resolve_name(name, suffix) gives (point, "
             "layer or None) or None, resolve_point(point, layer) gives (quantile metric, the eight "
             "observe_with_quantiles offset and bin arguments, slots) or None for a point fallback takes, "
             "fallback(point, layer, tensor) observes in Python, `recorded` is the recorder's host flags, "
             "capacity_error the error of a population too large and graph_task_id torch's current graph task",
             py::arg("resolve_name"), py::arg("resolve_point"), py::arg("fallback"), py::arg("recorded"),
             py::arg("capacity_error"), py::arg("graph_task_id"))
        .def("__call__", &ActlogObserver::observe, "Observes one actlog call", py::arg("tensor"), py::arg("name"),
             py::arg("suffix"), py::arg("layer_depends"), py::arg("layer_index"))
        .def("begin_step", &ActlogObserver::begin_step, "Starts a step that writes into the float32 `words`",
             py::arg("words"))
        .def("begin_microbatch", &ActlogObserver::begin_microbatch, "Starts a microbatch", py::arg("microbatch"))
        .def("flush", &ActlogObserver::flush, "Ends the microbatch's quantile populations");
    py::class_<AttentionCalls, std::shared_ptr<AttentionCalls>>(m, "AttentionCalls")
        .def(py::init<py::object, py::object, py::object, py::object, torch::Tensor, py::object>(),
             "The attention calls of a model as AttentionStats observes them on CUDA, from C++: `sampler` gates them "
             "and gives their microbatch, resolve(kind, layer, microbatch) gives a point's (offsets, widths, slot "
             "indices), `launcher` is triton_launch and `kernels` attention_kernel, `recorded` the recorder's host "
             "flags and graph_task_id torch's current graph task",
             py::arg("sampler"), py::arg("resolve"), py::arg("launcher"), py::arg("kernels"), py::arg("recorded"),
             py::arg("graph_task_id"))
        .def("begin_step", &AttentionCalls::begin_step, "Starts a step that writes into the float32 `words`",
             py::arg("words"))
        .def(
            "softmax",
            [](const std::shared_ptr<AttentionCalls>& self, int64_t layer, int64_t nope_dim, py::object rows) {
                return SoftmaxAttentionCall(self, layer, nope_dim, std::move(rows));
            },
            "The attention observer of MLA layer `layer`, whose calls' rows go to `rows`", py::arg("layer"),
            py::arg("nope_dim"), py::arg("rows"))
        .def(
            "gate",
            [](const std::shared_ptr<AttentionCalls>& self, int64_t layer) { return SoftmaxGateCall(self, layer); },
            "The gate observer of gated MLA layer `layer`", py::arg("layer"))
        .def(
            "linear",
            [](const std::shared_ptr<AttentionCalls>& self, int64_t layer) { return LinearAttentionCall(self, layer); },
            "The attention observer of GDN layer `layer`", py::arg("layer"));
    py::class_<SoftmaxAttentionCall>(m, "SoftmaxAttentionCall")
        .def("__call__", &SoftmaxAttentionCall::operator(), "Observes one MLA call", py::arg("query"), py::arg("key"),
             py::arg("value"), py::arg("lse"), py::arg("softmax_scale"), py::arg("cu_seqlens") = py::none(),
             py::arg("max_seqlen") = py::none());
    py::class_<SoftmaxGateCall>(m, "SoftmaxGateCall")
        .def("__call__", &SoftmaxGateCall::operator(), "Observes one MLA call's gate logits", py::arg("gate"));
    py::class_<LinearAttentionCall>(m, "LinearAttentionCall")
        .def("__call__", &LinearAttentionCall::operator(), "Observes one GDN call", py::arg("beta"),
             py::arg("log_decay"), py::arg("value"), py::arg("output"), py::arg("a_log"), py::arg("dt_bias"),
             py::arg("final_state"));
    py::class_<RouterBalanceCalls, std::shared_ptr<RouterBalanceCalls>>(m, "RouterBalanceCalls")
        .def(py::init<py::object, py::object, py::object, py::object, py::dict, int64_t, torch::Tensor, py::object,
                      py::object, py::object>(),
             "The routers' balance observations of ExpertBalanceStats.observe_router from C++: `sampler` gates "
             "them and gives their microbatch, `recorder` the step's words, resolve(layer, microbatch) a router's "
             "(offsets, widths, slot indices), fallback(layer, microbatch, operands...) observes a call the "
             "extension does not take into python_populations, keyed by (layer, microbatch), `recorded` is the "
             "recorder's host flags, nonbinary_flag(device) the flag of an integer mask other than 0 or 1, "
             "capacity_error the error of a population too large and graph_task_id torch's current graph task",
             py::arg("sampler"), py::arg("recorder"), py::arg("resolve"), py::arg("fallback"),
             py::arg("python_populations"), py::arg("experts"), py::arg("recorded"), py::arg("nonbinary_flag"),
             py::arg("capacity_error"), py::arg("graph_task_id"))
        .def("flush", &RouterBalanceCalls::flush, "Ends the microbatch's populations")
        .def("has_population", &RouterBalanceCalls::has_population,
             "Whether the extension holds the population of a router layer and microbatch", py::arg("layer"),
             py::arg("microbatch"))
        .def(
            "router",
            [](const std::shared_ptr<RouterBalanceCalls>& self, int64_t layer) {
                return RouterBalanceCall(self, layer);
            },
            "The balance observer of routed layer `layer`", py::arg("layer"));
    py::class_<RouterBalanceCall>(m, "RouterBalanceCall")
        .def("__call__", &RouterBalanceCall::operator(), "Observes one router call", py::arg("scores"),
             py::arg("biased_scores"), py::arg("normalized_topk"), py::arg("raw_topk"), py::arg("biased_topk"),
             py::arg("valid"), py::arg("bias"), py::arg("z_loss"), py::arg("seq_aux_loss"), py::arg("alpha"));
    py::class_<ExpertActivationCalls, std::shared_ptr<ExpertActivationCalls>>(m, "ExpertActivationCalls")
        .def(py::init<py::object, py::object, py::object, py::object, py::dict, int64_t, int64_t, torch::Tensor,
                      py::object, py::object>(),
             "The MoE layers' activation observations of ExpertBalanceStats.observe_experts from C++: `sampler` "
             "gates them and gives their microbatch, `recorder` the step's words, resolve(layer, microbatch) an "
             "expert module's (offset, slot index), fallback(layer, microbatch, intermediate, counts) observes a "
             "call the extension does not take into python_populations, keyed by (layer, microbatch), "
             "`local_experts` and `expert_offset` are the rank's experts, `recorded` is the recorder's host flags, "
             "capacity_error the error of a population too large and graph_task_id torch's current graph task",
             py::arg("sampler"), py::arg("recorder"), py::arg("resolve"), py::arg("fallback"),
             py::arg("python_populations"), py::arg("local_experts"), py::arg("expert_offset"), py::arg("recorded"),
             py::arg("capacity_error"), py::arg("graph_task_id"))
        .def("flush", &ExpertActivationCalls::flush,
             "Ends the microbatch's populations and returns their (layer, microbatch, counts)")
        .def("has_population", &ExpertActivationCalls::has_population,
             "Whether the extension holds the population of an expert layer and microbatch", py::arg("layer"),
             py::arg("microbatch"))
        .def(
            "experts",
            [](const std::shared_ptr<ExpertActivationCalls>& self, int64_t layer) {
                return ExpertActivationCall(self, layer);
            },
            "The activation observer of the expert module of MoE layer `layer`", py::arg("layer"));
    py::class_<ExpertActivationCall>(m, "ExpertActivationCall")
        .def("__call__", &ExpertActivationCall::operator(), "Observes one expert module call",
             py::arg("intermediate"), py::arg("counts"));
    py::class_<RouterMarginCalls, std::shared_ptr<RouterMarginCalls>>(m, "RouterMarginCalls")
        .def(py::init<py::object, py::object, py::object, py::object, py::set, torch::Tensor, py::object>(),
             "The router margins of MetricSampler._observe_router_margin from C++: `sampler` gates them and gives "
             "their microbatch, `recorder` the step's words, resolve(layer, microbatch) a router's (offset, width, "
             "slot index), graph(scoring_func, top_k, near_count, logits) the CUDA graph of logits of that shape, "
             "`layers` the layers observed in the microbatch, `recorded` is the recorder's host flags and "
             "graph_task_id torch's current graph task",
             py::arg("sampler"), py::arg("recorder"), py::arg("resolve"), py::arg("graph"), py::arg("layers"),
             py::arg("recorded"), py::arg("graph_task_id"))
        .def(
            "router",
            [](const std::shared_ptr<RouterMarginCalls>& self, int64_t layer, std::string scoring_func, int64_t top_k,
               int64_t near_margin_count, int64_t experts, py::object record) {
                return RouterMarginCall(
                    self, layer, RouterMarginCalls::Router{std::move(scoring_func), top_k, near_margin_count, experts},
                    std::move(record));
            },
            "The forward hook of the gate's linear layer of routed layer `layer`: its router's scoring function, "
            "top-k, near-margin count and experts, and record(logits), the Python recording of other logits",
            py::arg("layer"), py::arg("scoring_func"), py::arg("top_k"), py::arg("near_count"), py::arg("experts"),
            py::arg("record"));
    py::class_<RouterMarginCall>(m, "RouterMarginCall")
        .def("__call__", &RouterMarginCall::operator(), "Observes one router gate call", py::arg("module"),
             py::arg("inputs"), py::arg("output"));
    py::class_<RoutingCalls, std::shared_ptr<RoutingCalls>>(m, "RoutingCalls")
        .def(py::init<py::object, py::object, py::object, py::object, py::object, py::object, py::object, py::object,
                      py::object, py::object, int64_t, torch::Tensor>(),
             "The routing observations of MetricSampler._observe_routing from C++: `sampler` gives their microbatch, "
             "token_loss.begin_routing(layer, tokens, device) the top-k target, `quantiles` is the ExactQuantileStats "
             "or None, `recorder` gives the step's words, entropy_slots(layer, microbatch) the entropy quantile "
             "slots, copy_quantiles(layer, scores) and record_quantiles(layer, scores) the Python quantiles of "
             "scores the extension does not select, fallback(layer, topk_indices, scores) observes a call the "
             "extension does not take, `launcher` is triton_launch, `kernels` token_loss_kernel, `experts` the "
             "model's expert count and `recorded` the recorder's host flags",
             py::arg("sampler"), py::arg("token_loss"), py::arg("quantiles"), py::arg("recorder"),
             py::arg("entropy_slots"), py::arg("copy_quantiles"), py::arg("record_quantiles"), py::arg("fallback"),
             py::arg("launcher"), py::arg("kernels"), py::arg("experts"), py::arg("recorded"))
        .def(
            "router",
            [](const std::shared_ptr<RoutingCalls>& self, int64_t layer) { return RoutingCall(self, layer); },
            "The routing observer of routed layer `layer`", py::arg("layer"));
    py::class_<RoutingCall>(m, "RoutingCall")
        .def("__call__", &RoutingCall::operator(), "Observes one routing call", py::arg("topk_indices"),
             py::arg("scores"));
    py::class_<MaterializedWeightObserver>(m, "MaterializedWeightObserver")
        .def(py::init<py::object, py::set, py::dict, py::dict, py::object, py::object>(),
             "The weight monitor's observer of MaterializedWeights.observe from C++: `sampler` gates its calls, "
             "`modules_seen`, `units` and `pending` are the MaterializedWeights' own, record_modules(modules) records "
             "the weights of modules outside a SuperTensorModule and graph_task_id is torch's current graph task",
             py::arg("sampler"), py::arg("modules_seen"), py::arg("units"), py::arg("pending"),
             py::arg("record_modules"), py::arg("graph_task_id"))
        .def("__call__", &MaterializedWeightObserver::operator(), "Observes one module forward", py::arg("module"));
    py::class_<MonitorObservations>(m, "MonitorObservations")
        .def(py::init<torch::Tensor, torch::Tensor, int64_t, py::object, py::object, py::object>(),
             "The C++ observers of one microbatch's monitored activations, which write into `accumulator` and add "
             "to host int64 `counts`, reducing forward tensors in batches of `batch`; a tensor the extension does not "
             "reduce goes to forward_fallback(value, slots) or gradient_fallback(slots, gradient), and a forward "
             "tensor of skipped_type is not observed",
             py::arg("accumulator"), py::arg("counts"), py::arg("batch"), py::arg("forward_fallback"),
             py::arg("gradient_fallback"), py::arg("skipped_type"))
        .def("observe_module", &MonitorObservations::observe_module,
             "Observes one call of a monitored module: its positional inputs, then its output (a tensor or a sequence "
             "of them), slots[2 * position] being the forward observation's slot and slots[2 * position + 1] the "
             "gradient's, -1 for none and -2 for a position without a catalog entry",
             py::arg("inputs"), py::arg("output"), py::arg("slots"))
        .def("flush", &MonitorObservations::flush,
             "Reduces the queued forward tensors; -1, or the first slot of one modified in place since it was "
             "observed, the queue then being dropped unreduced");
    m.def(
        "observe",
        &observe,
        "Accumulate the activation moments/power sums/histograms/amax of a tensor's last-axis rows into the "
        "recorder's word buffer, on the tensor's current CUDA stream; with quantile_first_pass, also returns the "
        "tensor's exact-selection scratch with its counts and first pass counted (otherwise empty): the first "
        "digit's histogram, or for a bf16/fp16 tensor, unless key_histogram is false, the counts of its 65536 raw "
        "values (int64, a word for each pair of values 2w and 2w + 1, the count of 2w in its low half, then 512 "
        "words of the int32 total of each group of 32 words), which leave nothing to refine and from which its "
        "statistics are taken",
        py::arg("tensor"),
        py::arg("buffer"),
        py::arg("moments_offset"),
        py::arg("counts_offset"),
        py::arg("amax_offset"),
        py::arg("power_sums_offset"),
        py::arg("bins_start"),
        py::arg("bins_count"),
        py::arg("quantile_first_pass"),
        py::arg("key_histogram") = true);
    m.def(
        "exact_quantiles",
        &exact_quantiles,
        "Exact q0..q100 (into nine float32 quantiles) and [finite, NaN, +Inf, -Inf, masked] (into five int32 "
        "counts) of all elements of a bf16/fp16/fp32 population by radix selection, on the current CUDA stream; "
        "mask is None or a boolean mask shaped like it; counted is None or observe's scratch for the same unmasked "
        "population",
        py::arg("population"),
        py::arg("mask"),
        py::arg("quantiles"),
        py::arg("counts"),
        py::arg("counted"));
    m.def(
        "observe_with_quantiles",
        &observe_with_quantiles,
        "observe with the first selection pass, then the tensor's exact quantiles and counts selected from it, "
        "written as nine float32 and five int32 words at the given offsets of the recorder's word buffer; "
        "key_histogram false counts a bf16/fp16 tensor's first digit only and refines it, instead of counting "
        "its raw values",
        py::arg("tensor"),
        py::arg("buffer"),
        py::arg("moments_offset"),
        py::arg("counts_offset"),
        py::arg("amax_offset"),
        py::arg("power_sums_offset"),
        py::arg("bins_start"),
        py::arg("bins_count"),
        py::arg("quantiles_offset"),
        py::arg("quantile_counts_offset"),
        py::arg("key_histogram") = true);
    m.def(
        "exact_quantiles_float64",
        &exact_quantiles_float64,
        "exact_quantiles of an unmasked contiguous 1D float64 CUDA population, float64 q0..q100 and int32 [finite, "
        "NaN, +Inf, -Inf, masked], selected in one block; None for a population too large for one block",
        py::arg("values"));
    m.def(
        "order_statistics",
        &order_statistics,
        "Values at the given ranks (one to seven) of a 1D bf16/fp16/fp32 population in torch.sort order, "
        "NaN after +inf, by radix selection on the current CUDA stream; float32",
        py::arg("values"),
        py::arg("ranks"));
    m.def(
        "row_norm_stats",
        &row_norm_stats,
        "float64 [amax over |tensor|, sum of the L2 norms of the tensor's last-axis rows]; each row reduced in fp32, "
        "NaN propagating",
        py::arg("tensor"));
    m.def(
        "row_norm_accumulate",
        &row_norm_accumulate,
        "row_norm_stats of a tensor's last-axis rows, also taken into slot `slot` of a float64 [slots, 2] "
        "accumulator (NaN-propagating maximum of the amax, sum of the row-norm sums); returns the statistics "
        "if keep_stats, otherwise None",
        py::arg("tensor"),
        py::arg("accumulator"),
        py::arg("slot"),
        py::arg("keep_stats"));
    m.def(
        "accumulate_row_norm_stats",
        &accumulate_row_norm_stats,
        "Takes statistics row_norm_accumulate returned into another accumulator slot",
        py::arg("stats"),
        py::arg("accumulator"),
        py::arg("slot"));
    m.def(
        "row_norm_accumulate_batch",
        &row_norm_accumulate_batch,
        "row_norm_accumulate of every tensor of a list in one launch (up to 32 per launch): tensor i is reduced "
        "as it is alone and taken into slot_counts[i] slots of the float64 [slots, 2] accumulator, listed in "
        "order in `slots`, the first of them as the tensor's own statistics and the others as they would be "
        "by accumulate_row_norm_stats; no slot may be listed twice",
        py::arg("tensors"),
        py::arg("accumulator"),
        py::arg("slot_counts"),
        py::arg("slots"));
    m.def(
        "weight_norms",
        &weight_norms,
        "float32 [amax, L2 norm rounded to the weight's dtype, 1] of each contiguous bf16/fp16/fp32 weight, "
        "one row per weight",
        py::arg("weights"));
    m.def(
        "weight_norms_into",
        &weight_norms_into,
        "weight_norms of each weight written into a float32 buffer, weight i's row at offsets[i]; launch_numels[i] "
        "is the element count of the largest weight of the launch weight i is reduced like in, which sets its "
        "block count. Tensor j is one weight, or with experts[j] > 0 a stack of that many, its rows as "
        "[experts[j], -1]. False, with nothing launched, when a weight is not a contiguous bfloat16, float16 or "
        "float32 CUDA tensor",
        py::arg("weights"),
        py::arg("experts"),
        py::arg("buffer"),
        py::arg("offsets"),
        py::arg("launch_numels"));
    m.def(
        "segment_sumsq_into",
        &segment_sumsq_into,
        "float64 sum of squares of the rows of each consecutive segment (row counts given on the host, at most "
        "256) of a [rows, H] tensor, stored as two int32 words per segment in words from offset, or added to the "
        "float64 values there when accumulate",
        py::arg("matrix"),
        py::arg("counts"),
        py::arg("words"),
        py::arg("offset"),
        py::arg("accumulate"));
    m.def(
        "column_sums",
        &column_sums,
        "float64 sum of each column of a float32 [rows, n] tensor with a contiguous last axis",
        py::arg("matrix"));
    m.def(
        "row_top_values",
        &row_top_values,
        "The k largest values of each row of a float32 [rows, n <= 1024] tensor, descending with NaN "
        "largest, as torch.topk(k).values",
        py::arg("matrix"),
        py::arg("k"));
    m.def(
        "row_entropy",
        &row_entropy,
        "float64 entropy of each row of a float32 [rows, n] tensor normalized to sum to one, divided by log(n)",
        py::arg("matrix"));
    m.def(
        "copy_counts_into",
        &copy_counts_into,
        "Each int64 or int32 count tensor of `sources` copied into int32 `words` from its offset, converted as copy_ "
        "converts it, in one launch for up to eight of them",
        py::arg("sources"),
        py::arg("words"),
        py::arg("offsets"));
    m.def(
        "monitor_values_into",
        &monitor_values_into,
        "The [amax, mean row norm] of each monitored activation written as float32 into `words` at `targets`, two "
        "words per entry, from its float64 [amax, row norm sum] `accumulator` and the host int64 [observations, rows] "
        "`counts`: NaN for an entry without observations",
        py::arg("accumulator"),
        py::arg("counts"),
        py::arg("words"),
        py::arg("targets"));
    m.def(
        "entropy_quantiles_into",
        &entropy_quantiles_into,
        "exact_quantiles of row_entropy(matrix) written into int32 `words`: the float64 quantiles as word pairs from "
        "quantiles_offset, the counts from counts_offset; False without writing for a population too large for one "
        "block",
        py::arg("matrix"),
        py::arg("words"),
        py::arg("quantiles_offset"),
        py::arg("counts_offset"));
    m.def(
        "row_margins",
        &row_margins,
        "boundary[:, None] - matrix of a float32 [rows, n] tensor and one float32 boundary per row, a NaN of matrix "
        "counting as +inf",
        py::arg("matrix"),
        py::arg("boundary"));
    m.def(
        "router_margin_into",
        &router_margin_into,
        "The eight margin statistics of a sigmoid router's [rows, experts] logits written into the float32 buffer "
        "from offset: for the scores, then the logits as float32, the median gap after the top_k-th largest value, "
        "the median of the near_count margins after it and the q25 and q10 margins of the unselected experts, bit "
        "for bit as torch's sigmoid, median and lerp over row_top_values, row_margins and order_statistics give "
        "them. False, with nothing launched, beyond 1024 experts or int32 population counts",
        py::arg("logits"),
        py::arg("top_k"),
        py::arg("near_count"),
        py::arg("buffer"),
        py::arg("offset"));
    m.def(
        "linear_state_stats",
        &linear_state_stats,
        "AttentionStats.observe_linear_state of a CUDA state, issued from C++: the same operations writing "
        "[documents, float64 norm sums, norm maxima] into the float32 slot, or with merge folding them into "
        "what the slot holds",
        py::arg("state"),
        py::arg("slot"),
        py::arg("merge"));
    m.def(
        "entropy_sum",
        &entropy_sum,
        "float64 sum of -(x * log(x + 1e-12)) over a contiguous float32 CUDA tensor, each term in float64",
        py::arg("values"));
    m.def(
        "router_accumulate",
        &router_accumulate,
        "Adds one router call of float32 CUDA scores to its population in two launches: values (seven float64) "
        "+= expert_balance.router_sums' values, valid_tokens (one int64) += the valid token count and score_sums "
        "(float64 per expert) += the score column sums; the entropy and column sums are entropy_sum's and "
        "column_sums' bit for bit, the top-k sums are reduced in a fixed order of their own. mask None counts "
        "every token valid; a boolean or integer mask counts a token valid where it is not zero, and an integer "
        "one sets the int32 flag nonbinary where it is neither zero nor one; a seq_aux_loss of None adds NaN; a "
        "loss on the host is read by value. first starts the population from zero instead of adding to it. With "
        "words (the recorder's int32 buffer), the same "
        "launch writes the population's recorder slots at the given word offsets: router/scores_mean (score_sums "
        "times the float64 reciprocal of tokens_total, as float32; NaN without tokens), monitor/router_balance_local "
        "(values as 14 words), monitor/router_balance_counts_local (tokens_total, valid_tokens, calls_total as "
        "int32) and, on the first call, monitor/router_bias_local (expert_bias, or zeros without one)",
        py::arg("scores"),
        py::arg("biased_scores"),
        py::arg("normalized_topk"),
        py::arg("raw_topk"),
        py::arg("biased_topk"),
        py::arg("mask"),
        py::arg("z_loss"),
        py::arg("seq_aux_loss"),
        py::arg("alpha"),
        py::arg("values"),
        py::arg("valid_tokens"),
        py::arg("score_sums"),
        py::arg("first") = false,
        py::arg("words") = py::none(),
        py::arg("mean_offset") = 0,
        py::arg("balance_offset") = 0,
        py::arg("counts_offset") = 0,
        py::arg("bias_offset") = 0,
        py::arg("tokens_total") = 0,
        py::arg("calls_total") = 1,
        py::arg("expert_bias") = py::none(),
        py::arg("nonbinary") = py::none());
}
