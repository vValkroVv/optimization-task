"""Cache measured cuBLASLt descriptors and retain ATen for uncalibrated layouts."""
import json
from pathlib import Path

import torch
from tuning.proj.lt_ops import prepare


class LinearPolicy:
    def __init__(self, manifest):
        self.manifest = Path(manifest)
        self.shapes = set()
        self.cache = {}
        self.calls = {"candidate": 0, "fallback": 0}
        for key in json.loads(self.manifest.read_text()):
            _, _, descriptor = json.loads(key)
            m, n, k = descriptor[:3]
            output_row_major = descriptor[10]
            if descriptor[11] == 1:
                self.shapes.add((m, k, n) if output_row_major else (n, k, m))

    def supports_shape(self, lhs, rhs):
        return (lhs.size(0), lhs.size(1), rhs.size(1)) in self.shapes

    def run(self, lhs, rhs, output_shape=None):
        shape = (lhs.size(0), rhs.size(1))
        result = lhs.new_empty(shape if output_shape is None else output_shape)
        # The consuming CE may mutate the projection. Autograd requires the
        # custom Function to return the owner, even though cuBLASLt takes a 2D view.
        out = result if output_shape is None else result.view(shape)
        pointers = lhs.data_ptr() | rhs.data_ptr() | out.data_ptr()
        alignment = min(256, pointers & -pointers)
        key = (lhs.device, lhs.dtype, rhs.dtype, lhs.shape, lhs.stride(), rhs.shape, rhs.stride(),
               alignment, torch.cuda.current_stream(lhs.device).cuda_stream)
        if key not in self.cache:
            self.cache[key] = prepare(lhs, rhs, out, choices=self.manifest)
        match = self.cache[key]
        if match is None:
            self.calls["fallback"] += 1
            torch.mm(lhs, rhs, out=out)
            return result
        self.calls["candidate"] += 1
        plan, index = match
        plan.run(index, lhs, rhs, out)
        return result
