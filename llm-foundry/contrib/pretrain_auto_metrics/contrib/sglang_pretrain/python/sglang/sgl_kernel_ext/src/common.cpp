#include "common.h"

#include <cuda_runtime.h>

namespace sgl_kernel_ext {

bool pdl_supported() {
  // Compute capability does not change during a run, so query once and cache.
  static const bool supported = [] {
    int dev = 0, cc_major = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, dev);
    return cc_major >= 9;  // PDL is Hopper+ (sm_90)
  }();
  return supported;
}

}  // namespace sgl_kernel_ext
