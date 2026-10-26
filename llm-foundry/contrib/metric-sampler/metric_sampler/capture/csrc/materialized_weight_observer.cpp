#include "materialized_weight_observer.h"
#include "observer_helpers.h"

void MaterializedWeightObserver::operator()(const py::object& module) {
    if (!truthy(module, "training") || !forward_capture_active(sampler_, graph_task_id_)) {
        return;
    }
    if (modules_seen_.contains(module)) {
        return;
    }
    PyObject* unit = PyDict_GetItemWithError(units_.ptr(), module.ptr());
    if (unit == nullptr) {
        if (PyErr_Occurred()) {
            throw py::error_already_set();
        }
        record_modules_(py::make_tuple(module));
        return;
    }
    const py::dict empty;
    PyObject* modules = PyDict_SetDefault(pending_.ptr(), unit, empty.ptr());
    if (modules == nullptr || PyDict_SetItem(modules, module.ptr(), Py_None) < 0) {
        throw py::error_already_set();
    }
}
