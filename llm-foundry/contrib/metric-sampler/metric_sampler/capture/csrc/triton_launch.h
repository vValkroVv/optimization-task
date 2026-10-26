#pragma once

#include <torch/extension.h>
#include <cuda.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

// The launch values of a Triton kernel's runtime arguments and what its JIT specializes on, as
// triton_launch._variant makes them: a tensor's address, with its dtype and its 16-byte alignment; an integer or a
// float as it is, with its class: divisible by 16, equal to 1, or neither.
py::tuple triton_arguments(const py::tuple& args);

// A compiled Triton kernel launched as the launcher Triton 3.2 generates for it: `kinds` holds the type of each
// runtime argument of the kernel's signature, in order, as Triton names it, or "" for one of its constants, which
// the launch leaves out; a pointer is passed as its device address and any other argument as the C type of its
// kind. The kernel runs on the current stream with 32 * num_warps threads per program and `shared` bytes of shared
// memory, and a grid without programs launches nothing.
class TritonKernel {
public:
    TritonKernel(uint64_t function, int num_warps, int shared, const std::vector<std::string>& kinds);

    void operator()(int grid_x, int grid_y, int grid_z, const py::list& values) const;

private:
    enum class Kind { kConstant, kPointer, kInt32, kInt64, kUInt64, kFloat32 };

    CUfunction function_;
    int num_warps_;
    int shared_;
    std::vector<Kind> kinds_;
};

// The runtime arguments of a Triton launch made from C++, as triton_launch.launch takes them: each one's launch
// value and what the JIT specializes on are those triton_arguments gives it, and the tensors and integers
// themselves are kept for a launch through Python.
class LaunchArguments {
public:
    LaunchArguments& tensor(const torch::Tensor& value);

    LaunchArguments& integer(int64_t value);

    LaunchArguments& real(double value);

    const py::list& specialization() const { return specialization_; }
    const py::list& values() const { return values_; }

    py::tuple python() const;

private:
    struct Argument {
        torch::Tensor tensor;
        int64_t value;
        double real;
        bool is_real;
    };

    py::list specialization_;
    py::list values_;
    std::vector<Argument> arguments_;
};

// The launches triton_launch.launch makes, from C++ (`module` is triton_launch): the compiled kernel of the
// arguments' variant is launched natively when triton_launch holds it with its native launch and no launch hook is
// set, through triton_launch._run when it holds the compiled kernel only, and through triton_launch.launch
// otherwise -- a variant's first launch, which compiles it, or every launch without the direct launch.
class TritonLauncher {
public:
    explicit TritonLauncher(py::object module)
        : module_(std::move(module)), compiled_(module_.attr("_compiled")), hooks_(module_.attr("CompiledKernel")) {}

    void launch(const py::object& kernel, int64_t grid_x, int64_t grid_y, const LaunchArguments& arguments,
                const std::vector<std::pair<const char*, int64_t>>& constexprs) const;

private:
    py::object module_;
    py::object compiled_;
    py::object hooks_;
};
