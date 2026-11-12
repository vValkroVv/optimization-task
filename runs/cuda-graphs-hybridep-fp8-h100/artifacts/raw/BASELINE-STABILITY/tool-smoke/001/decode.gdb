set pagination off
target cudacore /workspace-SR008.nfs2/vkropotin/optimization-task/runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/BASELINE-STABILITY/tool-smoke/001/core.682990.1794479390.nvcudmp
info cuda kernels
info cuda threads
bt
x/8i $pc
disassemble /r opt004_trap_kernel
quit
