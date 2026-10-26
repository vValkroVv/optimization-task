#include "triton_launch.h"

#include <ATen/cuda/CUDAContext.h>
#include <cudaTypedefs.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace {

// The JIT of Triton specializes a launch on whether each pointer is 16-byte aligned and whether each integer or
// float is divisible by 16, equal to 1, or neither.
constexpr int kSpecializationDivisor = 16;

constexpr int kThreadsPerWarp = 32;

bool pointer_specialization(uintptr_t address) { return address % kSpecializationDivisor == 0; }

int integer_specialization(long long value) {
    return value % kSpecializationDivisor == 0 ? 0 : (value == 1 ? 1 : 2);
}

int real_specialization(double value) {
    return std::fmod(value, static_cast<double>(kSpecializationDivisor)) == 0 ? 0 : (value == 1 ? 1 : 2);
}

}  // namespace

py::tuple triton_arguments(const py::tuple& args) {
    py::list specialization;
    py::list values;
    for (const py::handle arg : args) {
        if (THPVariable_Check(arg.ptr())) {
            const auto address = reinterpret_cast<uintptr_t>(THPVariable_Unpack(arg.ptr()).data_ptr());
            specialization.append(static_cast<int>(THPVariable_Unpack(arg.ptr()).scalar_type()));
            specialization.append(pointer_specialization(address));
            values.append(py::int_(static_cast<unsigned long long>(address)));
            continue;
        }
        if (PyFloat_Check(arg.ptr())) {
            const double value = PyFloat_AsDouble(arg.ptr());
            specialization.append(real_specialization(value));
        } else {
            const long long value = arg.cast<long long>();
            specialization.append(integer_specialization(value));
        }
        values.append(arg);
    }
    return py::make_tuple(py::tuple(specialization), values);
}

// The driver's cuCtxGetCurrent and cuLaunchKernel, taken through the runtime once: the extension links no libcuda.
struct DriverLaunch {
    PFN_cuCtxGetCurrent current_context;
    PFN_cuLaunchKernel launch_kernel;
};

static const DriverLaunch& driver_launch() {
    static const DriverLaunch driver = [] {
        void* current_context = nullptr;
        void* launch_kernel = nullptr;
        cudaDriverEntryPointQueryResult found;
        C10_CUDA_CHECK(cudaGetDriverEntryPoint("cuCtxGetCurrent", &current_context, cudaEnableDefault, &found));
        TORCH_CHECK(found == cudaDriverEntryPointSuccess, "the CUDA driver has no cuCtxGetCurrent");
        C10_CUDA_CHECK(cudaGetDriverEntryPoint("cuLaunchKernel", &launch_kernel, cudaEnableDefault, &found));
        TORCH_CHECK(found == cudaDriverEntryPointSuccess, "the CUDA driver has no cuLaunchKernel");
        return DriverLaunch{reinterpret_cast<PFN_cuCtxGetCurrent>(current_context),
                            reinterpret_cast<PFN_cuLaunchKernel>(launch_kernel)};
    }();
    return driver;
}

TritonKernel::TritonKernel(uint64_t function, int num_warps, int shared, const std::vector<std::string>& kinds)
    : function_(reinterpret_cast<CUfunction>(function)), num_warps_(num_warps), shared_(shared) {
    for (const std::string& kind : kinds) {
        if (kind.empty()) {
            kinds_.push_back(Kind::kConstant);
        } else if (kind[0] == '*') {
            kinds_.push_back(Kind::kPointer);
        } else if (kind == "i1" || kind == "i32") {
            kinds_.push_back(Kind::kInt32);
        } else if (kind == "i64") {
            kinds_.push_back(Kind::kInt64);
        } else if (kind == "u64") {
            kinds_.push_back(Kind::kUInt64);
        } else if (kind == "fp32") {
            kinds_.push_back(Kind::kFloat32);
        } else {
            TORCH_CHECK_VALUE(false, "TritonKernel does not take arguments of kind ", kind);
        }
    }
}

void TritonKernel::operator()(int grid_x, int grid_y, int grid_z, const py::list& values) const {
    TORCH_CHECK_VALUE(values.size() == kinds_.size(), "TritonKernel expected ", kinds_.size(), " values, got ",
                      values.size());
    if (static_cast<int64_t>(grid_x) * grid_y * grid_z == 0) {
        return;
    }
    // Eight bytes per argument, each holding its value as the kernel's parameter reads it.
    std::vector<uint64_t> storage(kinds_.size());
    std::vector<void*> parameters;
    for (size_t i = 0; i < kinds_.size(); ++i) {
        const py::handle value = values[i];
        void* slot = &storage[i];
        switch (kinds_[i]) {
            case Kind::kConstant:
                continue;
            case Kind::kPointer:
                storage[i] = PyLong_AsUnsignedLongLong(value.ptr());
                if (PyErr_Occurred()) {
                    throw py::error_already_set();
                }
                break;
            case Kind::kInt32:
                *static_cast<int32_t*>(slot) = value.cast<int32_t>();
                break;
            case Kind::kInt64:
                *static_cast<int64_t*>(slot) = value.cast<int64_t>();
                break;
            case Kind::kUInt64:
                *static_cast<uint64_t*>(slot) = value.cast<uint64_t>();
                break;
            case Kind::kFloat32:
                *static_cast<float*>(slot) = value.cast<float>();
                break;
        }
        parameters.push_back(slot);
    }
    const DriverLaunch& driver = driver_launch();
    // A thread that has not used the device yet has no context to launch in: the primary one becomes current.
    CUcontext context = nullptr;
    TORCH_CHECK(driver.current_context(&context) == CUDA_SUCCESS, "cuCtxGetCurrent failed");
    if (context == nullptr) {
        C10_CUDA_CHECK(cudaFree(nullptr));
    }
    const CUresult launched =
        driver.launch_kernel(function_, grid_x, grid_y, grid_z, kThreadsPerWarp * num_warps_, 1, 1, shared_,
                             at::cuda::getCurrentCUDAStream(), parameters.data(), nullptr);
    TORCH_CHECK(launched == CUDA_SUCCESS, "cuLaunchKernel failed with CUresult ", static_cast<int>(launched));
}

LaunchArguments& LaunchArguments::tensor(const torch::Tensor& value) {
    const auto address = reinterpret_cast<uintptr_t>(value.data_ptr());
    specialization_.append(static_cast<int>(value.scalar_type()));
    specialization_.append(pointer_specialization(address));
    values_.append(py::int_(static_cast<unsigned long long>(address)));
    arguments_.push_back(Argument{value, 0, 0.0, false});
    return *this;
}

LaunchArguments& LaunchArguments::integer(int64_t value) {
    specialization_.append(integer_specialization(value));
    values_.append(py::int_(value));
    arguments_.push_back(Argument{torch::Tensor(), value, 0.0, false});
    return *this;
}

LaunchArguments& LaunchArguments::real(double value) {
    specialization_.append(real_specialization(value));
    values_.append(py::float_(value));
    arguments_.push_back(Argument{torch::Tensor(), 0, value, true});
    return *this;
}

py::tuple LaunchArguments::python() const {
    py::tuple arguments(arguments_.size());
    for (size_t i = 0; i < arguments_.size(); ++i) {
        const Argument& argument = arguments_[i];
        arguments[i] = argument.tensor.defined() ? py::cast(argument.tensor)
                       : argument.is_real          ? py::cast(argument.real)
                                                   : py::cast(argument.value);
    }
    return arguments;
}

void TritonLauncher::launch(const py::object& kernel, int64_t grid_x, int64_t grid_y, const LaunchArguments& arguments,
                            const std::vector<std::pair<const char*, int64_t>>& constexprs) const {
    if (module_.attr("_DIRECT_LAUNCH").cast<bool>()) {
        const int64_t device = c10::cuda::current_device();
        py::list key;
        key.append(kernel);
        key.append(device);
        for (const auto& entry : constexprs) {
            key.append(entry.second);
        }
        for (const py::handle item : arguments.specialization()) {
            key.append(item);
        }
        const py::object launched = compiled_.attr("get")(py::tuple(key));
        if (!launched.is_none()) {
            const py::object native = launched.cast<py::tuple>()[2];
            if (!native.is_none() && hooks_.attr("launch_enter_hook").is_none() &&
                hooks_.attr("launch_exit_hook").is_none()) {
                native.cast<const TritonKernel&>()(static_cast<int>(grid_x), static_cast<int>(grid_y), 1,
                                                   arguments.values());
            } else {
                module_.attr("_run")(launched, device, py::make_tuple(grid_x, grid_y, 1), arguments.values());
            }
            return;
        }
    }
    py::dict keywords;
    for (const auto& entry : constexprs) {
        keywords[entry.first] = entry.second;
    }
    module_.attr("launch")(kernel, py::make_tuple(grid_x, grid_y, 1), *arguments.python(), **keywords);
}
