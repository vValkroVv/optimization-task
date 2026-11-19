"""Specialize a compiled CuTe norm call when its activation output is dead."""

import operator

from torch._inductor.custom_graph_pass import CustomGraphPass, get_hash_for_files
from torch._subclasses.fake_tensor import FakeTensor

from .op import _forward, _forward_residual, _forward_stats


class GatedNormStatsOnlyPass(CustomGraphPass):
    def uuid(self):
        return get_hash_for_files((__file__,))

    def __call__(self, graph):
        for target, num_args in (
            (_forward._opoverload, 6),
            (_forward_residual._opoverload, 7),
        ):
            for node in graph.find_nodes(op="call_function", target=target):
                if len(node.args) != num_args or node.kwargs or not node.users:
                    continue
                outputs = node.meta.get("val")
                if not isinstance(outputs, tuple) or len(outputs) != 7:
                    continue
                if any(type(value) is not FakeTensor for value in outputs):
                    continue
                users = tuple(node.users)
                # Whole-tuple consumers and live activation consumers retain the full op.
                if any(
                    user.op != "call_function"
                    or user.target is not operator.getitem
                    or len(user.args) != 2
                    or user.args[0] is not node
                    or type(user.args[1]) is not int
                    or not 0 <= user.args[1] < len(outputs)
                    or (user.args[1] == 0 and user.users)
                    for user in users
                ):
                    continue
                with graph.inserting_before(node):
                    stats = graph.call_function(
                        _forward_stats._opoverload, node.args[:5]
                    )
                stats.meta = {**node.meta, "val": outputs[1:]}
                if "eager_input_vals" in stats.meta:
                    args, kwargs = stats.meta["eager_input_vals"]
                    # Inductor uses these values to preserve input layout constraints.
                    stats.meta["eager_input_vals"] = (
                        args[:5],
                        {
                            name: value
                            for name, value in kwargs.items()
                            if name not in ("gating_weight", "residual")
                        },
                    )
                for user in users:
                    index = user.args[1]
                    if index == 0:
                        graph.erase_node(user)
                    else:
                        user.args = (stats, index - 1)
                graph.erase_node(node)
