#pragma once

#include <torch/extension.h>

#include <utility>

// MetricSampler._observe_materialized_weights and MaterializedWeights.observe from C++: the weight monitor's call
// for a module, taken only for a module in training while `sampler` captures the forward. A module in
// `modules_seen` is skipped; one inside a SuperTensorModule, which `units` maps it to, joins that module's entry of
// `pending`; any other goes to record_modules((module,)).
class MaterializedWeightObserver {
public:
    MaterializedWeightObserver(py::object sampler, py::set modules_seen, py::dict units, py::dict pending,
                               py::object record_modules, py::object graph_task_id)
        : sampler_(std::move(sampler)),
          modules_seen_(std::move(modules_seen)),
          units_(std::move(units)),
          pending_(std::move(pending)),
          record_modules_(std::move(record_modules)),
          graph_task_id_(std::move(graph_task_id)) {}

    void operator()(const py::object& module);

private:
    py::object sampler_;
    py::set modules_seen_;
    py::dict units_;
    py::dict pending_;
    py::object record_modules_;
    py::object graph_task_id_;
};
