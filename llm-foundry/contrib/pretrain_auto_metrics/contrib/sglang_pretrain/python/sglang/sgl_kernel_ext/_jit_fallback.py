from pathlib import Path
import warnings
from torch.utils.cpp_extension import load, _get_build_directory


warnings.warn(
    "[sgl_kernel_ext] Ahead-of-time extension not found, falling back to JIT build. "
    "This may take a while and requires a compiler/NVCC.",
    UserWarning
)

ext_root = Path(__file__).parent
include_dirs=[str(ext_root/"include")]
src_dir = ext_root/"src"
sources = [str(f) for f in src_dir.glob("*.cpp")] + [str(f) for f in src_dir.glob("*.cu")]
verbose = True
# The same 'name' and 'build_directory' are used in the sglang/cleartorchextensionslock.py file. 
# Don't forget to update it.
name = "sgl_kernel_ext_jit"
build_directory = _get_build_directory(name, verbose)

# It will compile for archs provided by TORCH_CUDA_ARCH_LIST or for all visible cards if variable is not set
# TODO: think about common place for flags used here and in setup precompilation
_C = load(
    name=name,
    sources=sources,
    extra_include_paths=include_dirs,
    extra_cflags=["-O3", "-std=c++20"],
    extra_cuda_cflags=["-O3", "-std=c++20"],
    verbose=True,
    build_directory=build_directory,
)
