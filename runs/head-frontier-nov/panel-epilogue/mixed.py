"""Probe cuBLASLt's FP32 residual and directly rounded BF16 dHidden output on SM90."""
import ctypes as c
import json
import os
import sys
from pathlib import Path

import torch
from frontier_dw_lib import BF16, FP32, COMPUTE_FP32, Int, Ptr, U64, Heuristic, LtWriter, Layout, Algo, MAX_WORKSPACE, REDUCTION_MASK


class DHidden(LtWriter):
    def __init__(self, rows, cols, width, out_dtype):
        self.out_type = out_dtype
        super().__init__(rows, cols, width)

    def new_layout(self, row_major, rows, cols, width):
        op = Ptr()
        self.check(self.desc_create, c.byref(op), COMPUTE_FP32, FP32)
        dims = ((rows, cols, cols), (cols, width, width), (rows, width, width)) if row_major else (
            (width, cols, width), (cols, rows, cols), (width, rows, width))
        a = self.matrix(row_major, BF16, *dims[0])
        b = self.matrix(row_major, BF16, *dims[1])
        inp = self.matrix(row_major, FP32, *dims[2])
        out = self.matrix(row_major, self.out_type, *dims[2])
        layout = Layout(op, a, b, inp, row_major)
        layout.out = out
        return layout

    def find(self, order, cap):
        lay=self.layouts[order]
        pref=Ptr()
        self.check(self.pref_create,c.byref(pref))
        try:
            self.attribute(self.pref_attr,pref,MAX_WORKSPACE,U64(cap))
            self.attribute(self.pref_attr,pref,REDUCTION_MASK,c.c_uint(2))
            n=16
            results, returned=(Heuristic*n)(),Int()
            status=self.heuristic(self.handle,lay.operation,lay.a,lay.b,lay.cd,lay.out,
                                  pref,n,results,c.byref(returned))
            return status,[self.describe(order,r) for r in results[:returned.value] if r.state==0]
        finally:
            self.check(self.pref_destroy,pref)

    def run(self, cfg, g, w, acc, out, scratch):
        lay=self.layouts[cfg["order"]]
        a,b=(g,w) if lay.row_major else (w,g)
        algo=Algo.from_buffer_copy(bytes.fromhex(cfg["algo"]))
        self.check(self.matmul,self.handle,lay.operation,c.byref(self.alpha),a.data_ptr(),lay.a,
                   b.data_ptr(),lay.b,c.byref(self.beta[1]),acc.data_ptr(),lay.cd,out.data_ptr(),
                   lay.out,c.byref(algo),scratch.data_ptr() if scratch is not None else None,
                   cfg["workspace"],torch.cuda.current_stream().cuda_stream)

    def close(self):
        for layout in self.layouts.values():
            self.check(self.layout_destroy,layout.out)
        super().close()


if __name__=="__main__":
    chunks=int(sys.argv[1])
    cases=[]
    for dtype in (BF16,FP32):
        for cols in (32768,65536):
            engine=DHidden(16384//chunks,cols,4096,dtype)
            for order in ("col","row"):
                rc,configs=engine.find(order,32*2**20)
                cases.append(dict(chunks=chunks,columns=cols,output_type=dtype,order=order,
                                  status=rc,cuBLASLt=engine.version,algorithms=configs))
            engine.close()
    Path(f"probe-c{chunks}.json").write_text(json.dumps(cases,indent=2)+"\n")
    print([(r['columns'],r['output_type'],r['order'],r['status'],len(r['algorithms'])) for r in cases])
