"""Isolated native compiled MTPLoss vs pinned Quack, complete fwd/dX/dW."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'gift'))
from torchtitan.components.loss import MTPLoss
from torchtitan.config import CompileConfig
p = argparse.ArgumentParser()
p.add_argument('--variant', choices=['native', 'q4096', 'q8192', 'oracle'], required=True)
p.add_argument('--mode', choices=['timing', 'profile', 'numeric'], default='timing')
p.add_argument('--output', required=True, type=Path)
p.add_argument('--warmup', type=int, default=10)
p.add_argument('--samples', type=int, default=30)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
torch.cuda.set_device(0)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(2048)
fixture_path = ROOT / 'runs/quack-lmhead/artifacts/input/prepared.pt'
fixture = torch.load(fixture_path, map_location='cpu', weights_only=False)
x = fixture['hidden'].cuda().requires_grad_()
w = torch.nn.Parameter(fixture['weight'].cuda())
y = fixture['labels'].cuda()
positions = fixture['positions'].cuda()
den = torch.tensor(2096640., device='cuda')
y_mtp = MTPLoss._shift_labels(y, 1, positions)
dw = torch.empty_like(w, dtype=torch.float32)
# Emulates GFSDP per-chunk post-backward FP32 collection, without its collectives.
count = [0]
def collect(param):
    if count[0] == 0:
        dw.copy_(param.grad)
    else:
        dw.add_(param.grad)
    param.grad = None
    count[0] += 1
handle = w.register_post_accumulate_grad_hook(collect)
head = torch.nn.Linear(4096, 128256, bias=False, device='meta', dtype=torch.bfloat16)
head.weight = w
loss_fn = MTPLoss(MTPLoss.Config(num_chunks=2, loss_weights=[.3]), compile_config=CompileConfig(enable=True, components=['loss'], backend='inductor'))
loss_fn.set_lm_head(head)
if a.variant.startswith('q'):
    from quack.linear_cross_entropy import scaled_exp_lce_supported, scaled_exp_linear_cross_entropy
    chunk = int(a.variant[1:])
    assert scaled_exp_lce_supported(x[0].flatten(0,1), w, chunk, 'sum')
    def ce(h, weight, labels):
        return scaled_exp_linear_cross_entropy(h, weight, labels, chunk_size=chunk, reduction='sum', ignore_index=-100)
    # Custom-op route remains opaque to Inductor; same compile requirement as native CE.
    ce = torch.compile(ce, backend='inductor', fullgraph=True)

def call():
    x.grad = None
    w.grad = None
    count[0] = 0
    if a.variant == 'native':
        loss_fn.reset_components()
        loss = loss_fn(x, torch.stack([y, positions], dim=1), den)
        loss.backward()
    elif a.variant.startswith('q'):
        loss_fn.reset_components()
        loss = torch.zeros((), device='cuda')
        # Immediate backwards bound each head's precomputed weight gradient lifetime.
        for k, (labels, scale) in enumerate(((y, 1.), (y_mtp, .3))):
            term = ce(x[k].flatten(0, 1), w, labels.flatten()) / den * scale
            term.backward()
            loss_fn.add_component('loss_ce' if k == 0 else 'loss_mtp', term.detach() / scale)
            loss = loss + term.detach()
    else:
        raise RuntimeError('oracle uses separate path')
    assert count[0] == 2, count
    return loss

def memory():
    return {'allocated': torch.cuda.memory_allocated(), 'reserved': torch.cuda.memory_reserved()}

def metrics(ref, actual):
    sums = torch.zeros(5, device='cuda', dtype=torch.float64)
    maxabs = torch.zeros((), device='cuda', dtype=torch.float32)
    nonfinite = 0
    for r, v in zip(ref.flatten().split(1048576), actual.flatten().split(1048576)):
        r, v = r.float(), v.float()
        d = v-r
        sums += torch.stack([d.double().square().sum(), r.double().square().sum(), v.double().square().sum(), (r.double()*v.double()).sum(), torch.zeros((), device='cuda', dtype=torch.float64)])
        maxabs = torch.maximum(maxabs, d.abs().max())
        nonfinite += int((~torch.isfinite(v)).sum())
    ds, rs, vs, dot, _ = sums.tolist()
    return {'relative_l2': (ds / rs)**.5 if rs else None, 'max_abs': float(maxabs), 'cosine': dot/(rs*vs)**.5 if rs*vs else None, 'nonfinite': nonfinite}

receipt = {'variant': a.variant, 'mode': a.mode, 'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(), 'shape': list(x.shape), 'weight_shape': list(w.shape), 'denominator': float(den), 'valid_main': int((y!=-100).sum()), 'valid_mtp': int((y_mtp!=-100).sum()), 'compile_loss': True, 'gradient_contract': 'BF16 per-native-chunk or Quack-head gradients copied/added into FP32 local buffer; no communication or optimizer', 'timer_scope': 'gradient clearing + complete component forward/backward + synchronize; CUDA events include host submission gaps, not device-compute-only time'}
receipt['source_head'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
receipt['harness_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
receipt['input_receipt'] = json.loads((fixture_path.parent/'receipt.json').read_text())['prepared_sha256']
receipt['active_source_sha256'] = {name: hashlib.sha256((ROOT/'gift'/name).read_bytes()).hexdigest() for name in ['torchtitan/components/loss.py','torchtitan/experiments/four_fusion/stable_ce.py','torchtitan/distributed/gfsdp/_param_group.py']}
if a.variant == 'oracle':
    # True FP32 leaves and FP32 GEMMs; stream 128-token slices to bound logits.
    handle.remove()
    del head, loss_fn
    w.requires_grad_(False)
    wf = w.detach().float().requires_grad_()
    xf = x.detach().float().requires_grad_()
    del dw
    total = torch.zeros((), device='cuda')
    for k, (labels, scale) in enumerate(((y, 1.), (y_mtp, .3))):
        for start in range(0, 8192, 128):
            h = xf[k].reshape(-1,4096)[start:start+128]
            targets = labels.flatten()[start:start+128]
            logits = torch.nn.functional.linear(h, wf)
            term = torch.nn.functional.cross_entropy(logits, targets, reduction='sum', ignore_index=-100) / den * scale
            term.backward()
            total += term.detach()
    torch.cuda.synchronize()
    for variant in ['native','q4096','q8192']:
        path = a.output.parent / ('numeric-'+variant) / 'gradients.pt'
        if not path.exists():
            continue
        other = torch.load(path, map_location='cpu', weights_only=True)
        receipt[variant] = {'loss': float(other['loss']), 'loss_abs_error': abs(float(other['loss']) - float(total)), 'dx': metrics(xf.grad, other['dx'].cuda()), 'dw': metrics(wf.grad, other['dw'].cuda())}
    receipt['oracle_loss'] = float(total)
else:
    for _ in range(a.warmup):
        loss = call()
    torch.cuda.synchronize()
    if a.mode == 'numeric':
        loss = call()
        torch.cuda.synchronize()
        torch.save({'loss': loss.cpu(), 'dx': x.grad.cpu(), 'dw': dw.cpu()}, a.output/'gradients.pt')
        receipt['loss'] = float(loss)
        saved_y, saved_mtp = y.clone(), y_mtp.clone()
        y.fill_(-100)
        y_mtp.fill_(-100)
        ignored_loss = call()
        torch.cuda.synchronize()
        receipt['all_ignored_case'] = {'loss': float(ignored_loss), 'dx_nonzero': int(torch.count_nonzero(x.grad)), 'dw_nonzero': int(torch.count_nonzero(dw)), 'finite': bool(torch.isfinite(x.grad).all() & torch.isfinite(dw).all())}
        y.copy_(saved_y)
        y_mtp.copy_(saved_mtp)
        assert receipt['all_ignored_case'] == {'loss': 0., 'dx_nonzero': 0, 'dw_nonzero': 0, 'finite': True}
        assert bool((y[:, -1] == -100).all()) and bool((y_mtp[:, -2:] == -100).all())
        receipt['target_tail_and_shared_denominator_check'] = True
    elif a.mode == 'profile':
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
            with torch.profiler.record_function('lmhead_full_fwd_dx_dw'):
                call()
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(a.output/'trace.json'))
        receipt['activation'] = [e.key for e in prof.key_averages() if 'lce_scaled_exp' in e.key or 'stable_ce' in e.key]
    else:
        x.grad = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        receipt['memory_start'] = memory()
        torch.cuda.reset_peak_memory_stats()
        wall, events = [], []
        for _ in range(a.samples):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            start.record()
            loss = call()
            end.record()
            torch.cuda.synchronize()
            wall.append((time.perf_counter()-t0)*1000)
            events.append(start.elapsed_time(end))
        receipt.update(wall_ms=wall, cuda_event_host_submission_inclusive_ms=events, median_wall_ms=statistics.median(wall), median_cuda_event_ms=statistics.median(events), memory_end=memory(), memory_peak={'allocated': torch.cuda.max_memory_allocated(), 'reserved': torch.cuda.max_memory_reserved()})
(a.output/'result.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt), flush=True)
