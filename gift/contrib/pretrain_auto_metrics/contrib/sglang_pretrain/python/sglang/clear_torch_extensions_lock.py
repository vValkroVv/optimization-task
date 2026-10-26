import os
import logging

from torch.utils.cpp_extension import _get_build_directory
from torch.utils.file_baton import FileBaton

# Sometimes a file lock is not removed by PyTorch. For example, if we kill a program during JIT compilation.
# In that case, during the next launch, PyTorch will enter an infinite loop waiting for the lock to be removed. 
# Therefore, we need to manually remove the lock if it already exists in order to avoid this problem.
# NOTE: we have to make sure we are looking for the same lock file as PyTorch. Newer PyTorch versions
# might use a different name for the lock file, or store it in a different directory. This works for torch==2.8.0.

if __name__ == '__main__':
    name = "sgl_kernel_ext_jit"
    build_directory = _get_build_directory(name, True) # make sure it's the same as in sglang/sgl_kernel_ext/_jit_fallback.py
    baton = FileBaton(os.path.join(build_directory, 'lock'))
    try:
        baton.release()
    except FileNotFoundError:
        logging.info(f"Lock file doesn't exist: {baton.lock_file_path}.")
    except Exception as e:
        logging.warning(f"Cannot delete lock file: {baton.lock_file_path}. Error: {e}")
