"""``composer.utils.dist`` for CPU tests of the sequence-parallel helpers.

The helpers under test (``llmfoundry.models.parallel.utils``, ``…sequence.mappings``
and the conformer encoder) import a handful of names from ``composer.utils.dist``
(the conformer's module-level ``get_global_rank`` included).
On a host with composer the real module is returned; otherwise a shim with the
same API is registered in ``sys.modules`` — the group getters read the same
module globals composer uses, so a test pins the SP group with
``composer_dist._SEQUENCE_PARALLEL_GROUP = group`` in both environments.
"""

import math
import os
import sys
import types

import torch.distributed as dist


def install_composer_dist_shim() -> types.ModuleType:
    """Return the real ``composer.utils.dist`` when importable, else install and return the shim."""
    try:
        import composer.utils.dist as composer_dist

        return composer_dist
    except ImportError:
        pass
    shim = types.ModuleType("composer.utils.dist")
    shim._SEQUENCE_PARALLEL_GROUP = None
    shim._TENSOR_SEQUENCE_PARALLEL_GROUP = None

    def is_trivial_process_group(group):
        return group is None or group.size() == 1 or math.isnan(group.size())

    def _group_size(group):
        return None if group is None else group.size()

    def _group_rank(group):
        return None if group is None else group.rank()

    def all_reduce(tensor, reduce_operation="SUM", group=None, async_op=True):
        if not dist.is_initialized():
            return
        op = getattr(dist.ReduceOp, reduce_operation.upper())
        work = dist.all_reduce(tensor, op=op, group=group, async_op=async_op)
        if async_op:
            work.wait()

    def get_global_rank():
        if dist.is_initialized():
            return dist.get_rank()
        return int(os.environ.get("RANK", 0))

    shim.is_trivial_process_group = is_trivial_process_group
    shim.all_reduce = all_reduce
    shim.get_global_rank = get_global_rank
    shim.get_sp_group = lambda: shim._SEQUENCE_PARALLEL_GROUP
    shim.get_tp_sp_group = lambda: shim._TENSOR_SEQUENCE_PARALLEL_GROUP
    shim.get_sp_group_size = lambda: _group_size(shim._SEQUENCE_PARALLEL_GROUP)
    shim.get_sp_group_rank = lambda: _group_rank(shim._SEQUENCE_PARALLEL_GROUP)

    composer = types.ModuleType("composer")
    composer.__path__ = []
    composer_utils = types.ModuleType("composer.utils")
    composer_utils.__path__ = []
    composer_utils.dist = shim
    composer.utils = composer_utils
    sys.modules["composer"] = composer
    sys.modules["composer.utils"] = composer_utils
    sys.modules["composer.utils.dist"] = shim
    return shim
