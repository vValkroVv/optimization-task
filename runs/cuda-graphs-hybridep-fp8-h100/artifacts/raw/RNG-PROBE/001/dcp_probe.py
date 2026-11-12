"""CPU probe 001b: rank-local Stateful callback objects through native DCP/Gloo.
Question: do unique callback keys preserve all ranks, and do shared keys lose rank identity?
Decision: use qualified keys only if actual two-rank save/load demonstrates both facts.
"""
import json, pickle, random, sys
from datetime import timedelta
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
ROOT=Path(__file__).resolve().parents[6]
sys.path.insert(0,str(ROOT/'gift'))
from torchtitan.callbacks.base import CallbackHandler, Observer
OUT=Path(__file__).resolve().parent
class RNGState(Observer):
    def __init__(self,rank,qualified):
        self.key=f'continuation_rng_rank_{rank}' if qualified else 'continuation_rng'
        self.rank=rank;self.pending=None;self.loads=0
    def stateful_objects(self):return {self.key:self}
    def state_dict(self):
        return {'rank':self.rank,'cpu':torch.get_rng_state().clone(),'python_numpy':pickle.dumps((random.getstate(),np.random.get_state()))}
    def load_state_dict(self,value):self.pending=value;self.loads+=1

def worker(rank):
    dist.init_process_group('gloo',rank=rank,world_size=2,init_method=f'file://{OUT}/gloo-init-v3',timeout=timedelta(seconds=60))
    results={}
    for qualified in (False,True):
        random.seed(400+rank);np.random.seed(400+rank);torch.manual_seed(400+rank)
        cb=RNGState(rank,qualified);handler=CallbackHandler([cb]); states=handler.stateful_objects()
        expected=cb.state_dict(); path=OUT/('qualified-v3' if qualified else 'shared-v3')
        dcp.save(states,checkpoint_id=path)
        random.seed(900+rank);np.random.seed(900+rank);torch.manual_seed(900+rank)
        before=torch.get_rng_state().clone(); dcp.load(states,checkpoint_id=path)
        assert cb.loads==1 and torch.equal(before,torch.get_rng_state()),'load must stage without restoring'
        value=cb.pending
        exact=(value['rank']==rank and torch.equal(value['cpu'],expected['cpu']) and value['python_numpy']==expected['python_numpy'])
        results['qualified' if qualified else 'shared']=dict(saved_rank=rank,loaded_rank=value['rank'],exact_rank_payload=exact,pending_load_calls=cb.loads,cpu_equal=torch.equal(value['cpu'],expected['cpu']),python_numpy_equal=value['python_numpy']==expected['python_numpy'])
        if qualified:assert exact
        all_results=[None,None];dist.all_gather_object(all_results,exact)
        if not qualified:assert not all(all_results),all_results
    (OUT/f'dcp_rank{rank}.json').write_text(json.dumps(results,indent=2))
    dist.destroy_process_group()
if __name__=='__main__':
    mp.spawn(worker,nprocs=2,join=True)
    print(json.dumps([json.loads((OUT/f'dcp_rank{r}.json').read_text()) for r in range(2)],indent=2))
