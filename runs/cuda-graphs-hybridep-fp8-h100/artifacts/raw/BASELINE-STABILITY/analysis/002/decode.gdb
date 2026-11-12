set pagination off
target cudacore /workspace-SR008.nfs2/vkropotin/optimization-task/runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/BASELINE-STABILITY/attempts/002/cores/core.74585.1794497460.nvcudmp
info cuda kernels
cuda device sm warp lane block thread
info cuda lanes
p/x $pc
info symbol $pc
bt
set cuda disassemble_from elf_image
x/4i $pc-32
x/4i $pc
quit
