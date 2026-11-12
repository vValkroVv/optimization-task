"""CPU probe 001a: actual target loader continuation and RNG restore boundary."""
import copy, json, os, platform, random, sys
from pathlib import Path
import numpy as np
import torch
import datasets, torchdata
ROOT=Path(__file__).resolve().parents[6]
sys.path.insert(0,str(ROOT/'gift'))
from torchtitan.datasets.hf.text_datasets import HuggingFaceTextDataLoader
from torchtitan.config.schema.datasets.hf.text_datasets import HuggingFaceTextDataLoaderConfig
from torchtitan.components.tokenizer import HuggingFaceTokenizer

def make():
    tok=HuggingFaceTokenizer(tokenizer_path=str(ROOT/'gift/tests/assets/tokenizer'))
    config=HuggingFaceTextDataLoaderConfig(dataset='c4_test',dataset_path=str(ROOT/'gift/tests/assets/c4_test'),num_workers=0,pin_memory=False,infinite=True)
    return HuggingFaceTextDataLoader(config,dp_world_size=8,dp_rank=0,tokenizer=tok,seq_len=4096,local_batch_size=8)
def clone_batch(x): return ({k:v.clone() for k,v in x[0].items()},x[1].clone())
def equal(a,b): return torch.equal(a[1],b[1]) and all(torch.equal(a[0][k],b[0][k]) for k in a[0])
def state():return (random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
def restore(s):random.setstate(s[0]);np.random.set_state(s[1]);torch.set_rng_state(s[2])
random.seed(713);np.random.seed(713);torch.manual_seed(713)
loader=make(); before=state(); it=iter(loader); after_iter=state()
for _ in range(4): next(it)
saved_loader=copy.deepcopy(loader.state_dict()); saved_rng=state()
expected=[clone_batch(next(it)) for _ in range(4)]; expected_rng=state()
resumed=make();resumed.load_state_dict(copy.deepcopy(saved_loader));restore(saved_rng)
pre_iter=state();rit=iter(resumed);post_iter=state();actual=[clone_batch(next(rit)) for _ in range(4)];actual_rng=state()
# Candidate boundary: eagerly create actual iterator, then restore saved RNG before data use.
fixed=make();fixed.load_state_dict(copy.deepcopy(saved_loader));restore(saved_rng);fit=iter(fixed);restore(saved_rng)
fixed_actual=[clone_batch(next(fit)) for _ in range(4)];fixed_rng=state()
result=dict(python=sys.executable,python_version=platform.python_version(),torch=torch.__version__,torch_git=torch.version.git_version,torchdata=torchdata.__version__,datasets=datasets.__version__,fresh_iter_changes_cpu_rng=not torch.equal(before[2],after_iter[2]),resumed_iter_changes_cpu_rng=not torch.equal(pre_iter[2],post_iter[2]),early_restore_inputs_equal=all(map(lambda x:equal(*x),zip(expected,actual))),early_restore_cpu_rng_equal=torch.equal(expected_rng[2],actual_rng[2]),after_iter_restore_inputs_equal=all(map(lambda x:equal(*x),zip(expected,fixed_actual))),after_iter_restore_cpu_rng_equal=torch.equal(expected_rng[2],fixed_rng[2]),after_iter_restore_python_equal=expected_rng[0]==fixed_rng[0],after_iter_restore_numpy_equal=all(np.array_equal(a,b) for a,b in zip(expected_rng[1],fixed_rng[1])),shape=list(expected[0][1].shape))
callback_loader=make();callback_loader.load_state_dict(copy.deepcopy(saved_loader));restore(saved_rng);callback_iter=iter(callback_loader)
first=clone_batch(next(callback_iter));restore(saved_rng)
callback_actual=[first]+[clone_batch(next(callback_iter)) for _ in range(3)]
result['first_after_dataloader_inputs_equal']=all(equal(a,b) for a,b in zip(expected,callback_actual))
result['first_after_dataloader_cpu_rng_equal']=torch.equal(expected_rng[2],state()[2])
assert result['first_after_dataloader_inputs_equal'] and result['first_after_dataloader_cpu_rng_equal']
print(json.dumps(result,indent=2));assert all(result[k] for k in ['early_restore_inputs_equal','after_iter_restore_inputs_equal','after_iter_restore_cpu_rng_equal','after_iter_restore_python_equal','after_iter_restore_numpy_equal'])
