"""Specialize CuTe norm calls after AOT Autograd partitions the graph."""

import operator
from pathlib import Path

import torch
from torch._inductor.custom_graph_pass import CustomGraphPass, get_hash_for_files
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx import Node
from torch.fx.experimental.symbolic_shapes import statically_known_true

from torchtitan.tools.logging import logger

from .op import _backward, _backward_skip, _forward, _forward_residual, _forward_stats
from .pair import PAIR_SHAPE, forward_pair, supports_pair


class GatedNormStatsOnlyPass(CustomGraphPass):
    def uuid(self):
        return get_hash_for_files((__file__,))

    def __call__(self, graph):
        rewritten_sites = 0
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
                rewritten_sites += 1
        if rewritten_sites:
            logger.info("CuTe norm statistics-only SAC applied to %d site(s)", rewritten_sites)


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


_LOCAL_PREPARATION = {
    operator.getitem,
    torch.ops.aten.alias.default,
    torch.ops.aten.detach.default,
    torch.ops.aten.view.default,
    torch.ops.aten._unsafe_view.default,
    torch.ops.prims.convert_element_type.default,
}


def _is_local_preparation(node, device):
    if node.op != "call_function":
        return False
    if node.target in _LOCAL_PREPARATION:
        return True
    value = node.meta.get("val")
    if type(value) is not FakeTensor or value.device != device:
        return False
    if node.target == torch.ops.aten._to_copy.default:
        source = node.args[0]
        original = source.meta.get("val") if isinstance(source, Node) else None
        return (
            type(original) is FakeTensor
            and original.device == device
            and not node.kwargs.get("non_blocking", False)
        )
    # The zero-centered norm computes 1 + weight.float() on this device.
    return (
        node.target in (torch.ops.aten.add.Tensor, torch.ops.aten.add.Scalar)
        and tuple(value.shape) == (PAIR_SHAPE[-1],)
        and value.dtype == torch.float32
        and all(
            not isinstance(arg, Node)
            or (
                type(arg.meta.get("val")) is FakeTensor
                and arg.meta["val"].device == device
            )
            for arg in node.args
        )
    )


def _tuple_items(node):
    items = tuple(node.users)
    values = node.meta.get("val")
    if not isinstance(values, tuple) or any(
        item.op != "call_function"
        or item.target is not operator.getitem
        or len(item.args) != 2
        or item.args[0] is not node
        or item.kwargs
        or type(item.args[1]) is not int
        or not 0 <= item.args[1] < len(values)
        for item in items
    ):
        return None
    return items


class GatedNormPairPass(CustomGraphPass):
    """Keep the output indices chosen by AOT and fuse only an eligible edge."""

    def uuid(self):
        source = Path(__file__)
        sources = (source, source.with_name("pair.py"), source.with_name("forward.py"))
        return ("cute_gated_norm_pair", get_hash_for_files(tuple(map(str, sources))))

    def __call__(self, graph):
        fused_sites = 0
        for consumer_op in (_forward, _forward_stats):
            consumers = graph.find_nodes(
                op="call_function", target=consumer_op._opoverload
            )
            for second in consumers:
                stats_only = consumer_op is _forward_stats
                if len(second.args) != (5 if stats_only else 6) or second.kwargs:
                    continue
                x = second.args[0]
                value = x.meta.get("val") if isinstance(x, Node) else None
                if (
                    type(value) is not FakeTensor
                    or value.device.type != "cuda"
                    or value.dtype != torch.bfloat16
                    or any(type(size) is not int for size in value.shape)
                    or tuple(value.shape) != PAIR_SHAPE
                    or not value.is_contiguous()
                ):
                    continue
                if (
                    x.op != "call_function"
                    or x.target is not operator.getitem
                    or len(x.args) != 2
                ):
                    continue
                first, index = x.args
                if (
                    type(index) is not int
                    or index != 0
                    or not isinstance(first, Node)
                    or first.op != "call_function"
                    or first.target != _forward_residual._opoverload
                    or len(first.args) != 7
                    or first.kwargs
                ):
                    continue
                if not supports_pair(value.device):
                    continue
                first_items = _tuple_items(first)
                second_items = _tuple_items(second)
                if not first_items or not second_items:
                    continue

                order = {node: i for i, node in enumerate(graph.nodes)}
                # Insert before the second norm, once both sets of weights are ready.
                # The first norm's other consumers must still follow the new call.
                if any(
                    user is not second and order[user] < order[second]
                    for item in first_items
                    for user in item.users
                ):
                    continue
                # Unknown calls can have stream effects even with functional schemas.
                if any(
                    not _is_local_preparation(node, value.device)
                    for node in order
                    if order[first] < order[node] < order[second]
                ):
                    continue
                second_args = (*second.args[1:], 0.0) if stats_only else second.args[1:]
                args = (*first.args, *second_args, stats_only)
                with graph.inserting_before(second):
                    merged = graph.call_function(forward_pair._opoverload, args)
                    merged.meta["val"] = [*first.meta["val"], *second.meta["val"]]
                    merged.meta["eager_input_vals"] = (
                        tuple(
                            value.meta["val"] if isinstance(value, Node) else value
                            for value in args
                        ),
                        {},
                    )
                    for items, offset in ((first_items, 0), (second_items, 7)):
                        for item in items:
                            replacement = graph.call_function(
                                operator.getitem, (merged, item.args[1] + offset)
                            )
                            replacement.meta = dict(item.meta)
                            item.replace_all_uses_with(replacement)
                for item in second_items:
                    graph.erase_node(item)
                graph.erase_node(second)
                for item in first_items:
                    graph.erase_node(item)
                graph.erase_node(first)
                fused_sites += 1
        if fused_sites:
            graph.lint()
            logger.info("CuTe norm pair fusion applied to %d site(s)", fused_sites)
