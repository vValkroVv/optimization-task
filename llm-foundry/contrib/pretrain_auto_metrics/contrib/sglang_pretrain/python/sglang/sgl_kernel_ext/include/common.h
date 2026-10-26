#pragma once

#include <ATen/ATen.h>
#include <ATen/ops/from_blob.h>
#include <c10/core/Device.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <functional>

namespace sgl_kernel_ext {

/// Copy a **POD / trivially-copyable** ``value`` into a 1-D ``kByte`` tensor on ``device``:
/// non-owning CPU ``from_blob`` over ``sizeof(PodT)`` bytes, ``clone()``, then ``.to(device)``.
/// Caller is responsible for only using ``PodT`` layouts that are safe to ``memcpy`` (no pointers
/// unless they remain valid after the copy to the target device).
template <typename PodT>
[[nodiscard]] at::Tensor pack_pod_struct_to_device(const PodT& value, c10::Device device) {
  const int64_t nbytes = static_cast<int64_t>(sizeof(PodT));
  return at::from_blob(const_cast<void*>(reinterpret_cast<const void*>(&value)), {nbytes},
                       at::TensorOptions().dtype(at::kByte).device(at::kCPU))
      .clone()
      .to(device);
}

/// True if the current device supports Programmatic Dependent Launch (sm_90+,
/// Hopper/Blackwell). Queried once and cached for the process. Defined in
/// common.cpp (host-only: it uses the CUDA runtime, no device code).
bool pdl_supported();

/// Launch ``kernel`` with the programmatic stream serialization (PDL) attribute
/// when the device supports it, so it can overlap its prologue with the previous
/// kernel's tail; otherwise an ordinary launch.
template <typename KernelFn, typename... Args>
void launch_maybe_pdl(KernelFn kernel, dim3 grid, dim3 block, std::size_t smem, cudaStream_t stream,
                      Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attr = {};
  if (pdl_supported()) {
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
  }
  cudaLaunchKernelEx(&cfg, kernel, args...);
}

}  // namespace sgl_kernel_ext
