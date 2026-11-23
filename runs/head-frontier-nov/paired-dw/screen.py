"""Graph replay and captured FP64 validation of paired stores at the native shapes."""
import argparse
import json
import os
from pathlib import Path
import statistics
import traceback

import torch
from dw_check import FIXTURE, get_operands
from frontier_dw_lib import PlannedWriter
from frontier_stack import PAIRS
from pair import CONFIGS, launch


def timing(fn):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for _ in range(3):
        graph.replay()
    samples = []
    for _ in range(8):
        start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        graph.replay()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop))
    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument('chunks', type=int)
    p.add_argument('output', type=Path)
    p.add_argument('--first', type=int, default=0)
    args = p.parse_args()
    c = args.chunks
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    rec = torch.load(FIXTURE, map_location='cpu', mmap=True, weights_only=True)['ce']
    inputs = get_operands(rec, c)
    v, h = 128256, 4096
    output = torch.empty((v, h), device='cuda', dtype=torch.float32)
    native = PlannedWriter(json.loads((Path(os.environ['REPLAY_DIR']) / 'frontier-dw' / f'tune-{c}.json').read_text()), PAIRS[c])
    def baseline():
        for i, (g, x) in enumerate(inputs):
            native(g, x, output, i != 0)
    # Spread FP64 oracle rows over the vocabulary, including actual target gradients and both boundaries.
    labels = rec['labels'].reshape(-1)
    ids = torch.unique(torch.cat([torch.linspace(0,v-1,512).long(), labels[labels.ge(0)][::16].long()])).cuda()
    oracle = torch.zeros((len(ids),h),device='cuda',dtype=torch.float64)
    for g, x in inputs:
        oracle.addmm_(g.index_select(1,ids).double().T,x.double())
    def check():
        sample = output[ids].double()
        delta = sample-oracle
        return {'rel_l2':(delta.norm()/oracle.norm()).item(),'max_abs':delta.abs().max().item(),
                'bf16_disagreements_to_oracle':torch.count_nonzero(sample.bfloat16()!=oracle.bfloat16()).item()}
    result={'chunks':c,'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'rows_fp64':len(ids),'width':h,
            'fixture':str(FIXTURE),'torch':torch.__version__,'samples':[]}
    def append(label, fn):
        try:
            samples=timing(fn)
            row={'label':label,'graph_dw_chain_ms':samples,'median_ms':statistics.median(samples),**check()}
        except Exception:
            row={'label':label,'error':traceback.format_exc()}
        result['samples'].append(row)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(label, row.get('median_ms'),row.get('rel_l2'),row.get('error','')[-400:],flush=True)
    append('native-first',baseline)
    for idx in range(args.first, len(CONFIGS)):
        def run():
            for i in range(0,c,2):
                launch(*inputs[i],*inputs[i+1],output,i!=0,idx)
        append(idx,run)
    append('native-last',baseline)
    native.close()


if __name__=='__main__':
    main()
