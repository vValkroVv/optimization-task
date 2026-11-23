"""Route pinned linear CE candidates through the same native GFSDP component."""

from types import MethodType

import torch


def build(kind, backend):
    if kind == "fla":
        from torchtitan.experiments.four_fusion.linear_ce import native_linear_ce
        return torch.compile(native_linear_ce, backend=backend, fullgraph=True, dynamic=True)

    if kind.startswith("q"):
        from quack.linear_cross_entropy import scaled_exp_linear_cross_entropy
        tile = int(kind[1:])

        def terms(h, w, target, denominator, weights):
            count = len(weights)
            outputs = [
                (scaled_exp_linear_cross_entropy(part.flatten(0, 1), w, y.flatten(),
                    chunk_size=tile, reduction="sum", ignore_index=-100) / denominator) * scale
                for part, y, scale in zip(h.chunk(count), target.chunk(count), weights, strict=True)
            ]
            return torch.stack(outputs)

        return torch.compile(terms, backend=backend, fullgraph=True)

    if kind == "liger":
        from liger_kernel.ops.cutedsl.ops.fused_scaled_cross_entropy_sm90 import (
            LigerFusedScaledCrossEntropySM90Function,
        )

        def per_token(h, weight, labels):
            raw = LigerFusedScaledCrossEntropySM90Function.apply(
                h, weight, labels, 1.0, -100, 1, False, None, torch.float32
            )
            return raw[0] if isinstance(raw, tuple) else raw

    elif kind == "cce":
        from cut_cross_entropy import linear_cross_entropy

        def per_token(h, weight, labels):
            return linear_cross_entropy(
                h, weight, labels, ignore_index=-100, reduction="none", filter_eps=None,
                accum_e_fp32=True, accum_c_fp32=True, filter_e_grad=False, filter_c_grad=False,
                impl="cce",
            )
    else:
        raise ValueError(kind)

    def terms(h, w, target, denominator, weights):
        nll = per_token(h.flatten(0, 1), w, target.flatten())
        return torch.stack([
            (part.sum() / denominator) * scale
            for part, scale in zip(nll.chunk(len(weights)), weights, strict=True)
        ])

    return torch.compile(terms, backend=backend)


def install(kind, backend, loss, head):
    call = build(kind, backend)
    forward = head.forward
    native = loss._chunked_loss_and_grad
    native_loss = loss._chunk_loss
    state = {}
    route = {"calls": 0}

    def chunked(self, hidden, labels, denominator=None, **kwargs):
        assert kwargs.get("argmax_sink") is None and not kwargs.get("shape_probe")
        state["chunks"] = iter(t.contiguous() for t in labels.chunk(self.num_chunks, dim=1))
        state["denominator"] = denominator
        state["head_weights"] = [1.0, *self.loss_weights]
        try:
            return native(hidden, labels, denominator, **kwargs)
        finally:
            assert next(state["chunks"], None) is None
            state.clear()

    def head_forward(self, hidden, **kwargs):
        if "chunks" not in state:
            return forward(hidden, **kwargs)
        assert not kwargs
        labels = next(state["chunks"])
        route["calls"] += 1
        return call(hidden, self.weight, labels, state["denominator"], state["head_weights"])

    def chunk_loss(self, terms, labels, denominator, *, num_heads=1):
        if "chunks" not in state:
            return native_loss(terms, labels, denominator, num_heads=num_heads)
        assert num_heads == 2 and terms.shape == (num_heads,)
        self.add_component("loss_ce", terms[0].detach())
        self.add_component("loss_mtp", terms[1].detach() / self.loss_weights[0])
        return terms[0] + terms[1]

    loss._chunked_loss_and_grad = MethodType(chunked, loss)
    loss._chunk_loss = MethodType(chunk_loss, loss)
    head.forward = MethodType(head_forward, head)
    return route
