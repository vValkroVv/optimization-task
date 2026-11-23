"""Compare a panel's native FP32 addmm + cast with a single Hopper GEMM residual epilogue."""
import hashlib
import inspect
import json
from pathlib import Path
import sys
import textwrap

import torch

import panel_replay as panels
import replay

chunks, suffix = int(sys.argv[1]), sys.argv[2]
split = 32768 if "small" in suffix else 65536
mixed = suffix.startswith("quack")
output = Path(f"c{chunks}-{suffix}.json").resolve()
digest = output.with_suffix(".digest.json")
save = torch.save
def save_digest(tensors,destination,*args,**kwargs):
    if destination != digest:
        return save(tensors,destination,*args,**kwargs)
    digest.write_text(json.dumps({k:hashlib.sha256(t.reshape(-1).contiguous().view(torch.uint8).numpy()).hexdigest()
                                  for k,t in tensors.items()}, indent=2)+"\n")
torch.save=save_digest

if mixed:
    from quack.epilogue.library import residual_epi
    def finish(grad, panel, acc):
        result = torch.empty_like(acc, dtype=torch.bfloat16)
        residual_epi.gemm(grad,panel.T,result,epi_args={"res":acc},
                          tile_M=128,tile_N=128,tile_K=64,cluster_M=1,cluster_N=2)
        return result
    old = inspect.getsource(panels.PanelLoss.backward)
    source = textwrap.dedent(old).replace("@staticmethod\n","")
    anchor="torch.addmm(dx, grad, panel, out_dtype=torch.float32, out=dx)"
    assert source.count(anchor)==1
    source=source.replace(anchor,"dx = finish(grad, panel, dx)")
    namespace = dict(vars(panels),finish=finish)
    exec(compile(source,__file__,"exec"),namespace)
    panels.PanelLoss.backward=staticmethod(namespace["backward"])
sys.argv[1:] = ["panel",str(chunks),str(output),"--split",str(split),"--dump",str(digest)]
panels.main()
record=json.loads(output.read_text())
record["panel"]["dhidden_epilogue"] = "quack residual+BF16 output" if mixed else "native FP32 addmm+cast"
if mixed:
    record["panel"]["implemented_dhidden"] = "FP32 mm + Quack FP32 accumulate/residual and BF16 store"
output.write_text(json.dumps(record,indent=2)+"\n")
