#include <cuda_runtime.h>
#include <cstdio>

__global__ void opt004_trap_kernel() {
  asm volatile("trap;");
}

int main() {
  cudaError_t status = cudaSetDevice(0);
  if (status != cudaSuccess) {
    std::fprintf(stderr, "setup failed: %s\n", cudaGetErrorString(status));
    return 2;
  }
  std::puts("OPT004 isolated exception-capture smoke: launching one trap thread");
  std::fflush(stdout);
  opt004_trap_kernel<<<1, 1>>>();
  status = cudaDeviceSynchronize();
  std::fprintf(stderr, "synchronize returned %d: %s\n", int(status), cudaGetErrorString(status));
  return status == cudaSuccess ? 3 : 86;
}
