import json, torch
from torch._inductor import config
def normal(v):
 if v is None or isinstance(v,(bool,int,float,str)): return v
 if isinstance(v,dict): return {str(k):normal(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)): return [normal(x) for x in v]
 if isinstance(v,(set,frozenset)): return sorted((normal(x) for x in v),key=lambda x:json.dumps(x,sort_keys=True))
 if callable(v): return f"{v.__module__}.{v.__qualname__}"
 return str(v)
print(json.dumps({'torch':str(torch.__version__),'config':normal(config.get_config_copy()),'cuda_initialized':torch.cuda.is_initialized()},sort_keys=True))
