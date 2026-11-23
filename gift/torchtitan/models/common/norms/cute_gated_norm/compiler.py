"""Specialize CuTe norm calls after AOT Autograd partitions the graph."""

import operator

import torch
from torch._inductor.custom_graph_pass import CustomGraphPass, get_hash_for_files
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx import Node
from torch.fx.experimental.symbolic_shapes import statically_known_true

from torchtitan.tools.logging import logger

from .op import _backward, _backward_skip, _forward, _forward_residual, _forward_stats


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


class GatedNormSkipGradientPass(CustomGraphPass):
    """Fuse an already available BF16 skip into the single consumer of dx."""

    def uuid(self):
        return ("cute_gated_norm_bwd_skip", get_hash_for_files((__file__,)))

    def __call__(self, graph):
        candidates = graph.find_nodes(op="call_function", target=_backward._opoverload)
        if not candidates:
            return
        position = {node: index for index, node in enumerate(graph.nodes)}
        fused_sites = 0
        for node in candidates:
            if len(node.args) != 12 or node.kwargs:
                continue
            users = tuple(node.users)
            if any(
                user.op != "call_function"
                or user.target is not operator.getitem
                or len(user.args) != 2
                or user.args[0] is not node
                or type(user.args[1]) is not int
                or not 0 <= user.args[1] < 4
                for user in users
            ):
                continue
            input_grads = [user for user in users if user.args[1] == 0]
            if len(input_grads) != 1 or len(input_grads[0].users) != 1:
                continue
            dx = input_grads[0]
            (add,) = dx.users
            if (
                add.op != "call_function"
                or add.target is not torch.ops.aten.add.Tensor
                or len(add.args) != 2
                or add.kwargs
            ):
                continue
            skip_first = add.args[1] is dx
            skip = add.args[0] if skip_first else add.args[1]
            # The backward call also returns parameter gradients. Keep it at
            # its original position with all of its inputs already available.
            if not isinstance(skip, Node) or position[skip] >= position[node]:
                continue
            outputs = node.meta.get("val")
            skip_value = skip.meta.get("val")
            add_value = add.meta.get("val")
            if (
                not isinstance(outputs, tuple)
                or len(outputs) != 4
                or any(type(value) is not FakeTensor for value in outputs)
                or type(skip_value) is not FakeTensor
                or type(add_value) is not FakeTensor
            ):
                continue
            dx_value = outputs[0]
            if (
                dx_value.device.type != "cuda"
                or dx_value.dtype != torch.bfloat16
                or any(
                    value.device != dx_value.device
                    or value.dtype != dx_value.dtype
                    or value.ndim != dx_value.ndim
                    or not value.is_contiguous()
                    or not all(
                        statically_known_true(got == expected)
                        for got, expected in zip(value.shape, dx_value.shape)
                    )
                    for value in (skip_value, add_value)
                )
            ):
                continue
            with graph.inserting_before(node):
                fused = graph.call_function(
                    _backward_skip._opoverload, (*node.args, skip, skip_first)
                )
            fused.meta = {**node.meta, "val": (add_value, *outputs[1:])}
            if "eager_input_vals" in fused.meta:
                args, kwargs = fused.meta["eager_input_vals"]
                fused.meta["eager_input_vals"] = (
                    (*args, skip_value, skip_first),
                    kwargs,
                )
            node.replace_all_uses_with(fused)
            add.replace_all_uses_with(dx)
            graph.erase_node(add)
            graph.erase_node(node)
            fused_sites += 1
        if fused_sites:
            logger.info(
                "CuTe norm skip-gradient fusion applied to %d site(s)", fused_sites
            )
