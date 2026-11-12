set pagination off
target cudacore /workspace-SR008.nfs2/vkropotin/optimization-task/runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/BASELINE-STABILITY/attempts/002/cores/core.74585.1794497460.nvcudmp
set cuda disassemble_from elf_image
info symbol 0x7fd81aff1cc0
x/16i 0x7fd81aff1c80
python
import gdb, re, shutil
s=gdb.execute("info symbol 0x7fd81aff1cc0",to_string=True)
p=re.search(r" of (.+)$", s.strip()).group(1)
shutil.copy2(p, '/workspace-SR008.nfs2/vkropotin/optimization-task/runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/BASELINE-STABILITY/analysis/002/fault-module.elf')
print("Preserved fault module", p)
end
quit
