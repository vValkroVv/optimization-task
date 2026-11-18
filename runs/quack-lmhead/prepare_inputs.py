"""Replay native norm/MTP once on captured real inputs; no optimizer or training."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'gift'))
spec = importlib.util.spec_from_file_location('ffn_components', ROOT / 'runs/ffn-row-owner/ffn_components.py')
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)
cfg = f.resolved_config('mtp')
selected = copy.deepcopy(cfg.model_spec.model)
selected.layers = [selected.placed_layers()[0]]
# Build only one non-MoE trunk block on meta, then prune before allocation.
with torch.device('meta'):
    model = selected.build()
model.layers.clear()
model.tok_embeddings = None
model.to_empty(device='cuda')
torch.manual_seed(2048)
with torch.no_grad():
    model.init_weights(buffer_device=torch.device('cuda'))
model.bfloat16().train()
fixture = ROOT / 'runs/production-data-abcd/artifacts/raw/proddata-component-capture-001/native-inputs'
records, receipt = f.fixture_module().load_fixture(fixture, 0)
row = records['mtp']
restore = f.fixture_module().restore
with torch.no_grad():
    hidden = row['hidden'].cuda()
    embedding = row['embedding'].cuda()
    stacked = model._forward_mtp(hidden, embedding, restore(row['attention_masks'], 'cuda'), restore(row['positions'], 'cuda'), restore(row['token_mask'], 'cuda'))
ids = records['input_ids']
labels = torch.cat([ids[:, 1:], torch.full_like(ids[:, :1], -100)], dim=1)
out = ROOT / 'runs/quack-lmhead/artifacts/input'
out.mkdir(parents=True, exist_ok=True)
path = out / 'prepared.pt'
torch.save({'hidden': stacked.cpu(), 'weight': model.lm_head.weight.detach().cpu(), 'labels': labels, 'positions': row['positions']}, path)
manifest = {'fixture': receipt, 'prepared_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'shape': list(stacked.shape), 'hidden_dtype': str(stacked.dtype), 'weight_dtype': str(model.lm_head.weight.dtype), 'valid_main_per_microbatch': int((labels != -100).sum()), 'global_denominator': int((labels != -100).sum()) * 32 * 8, 'state': 'canonical modules initialized with seed 2048, not original checkpoint weights; MTP and norm replay excluded from timed head component', 'resolved_config': cfg.to_dict()}
(out / 'receipt.json').write_text(json.dumps(manifest, indent=2, default=str))
print(json.dumps({k: v for k, v in manifest.items() if k not in ('fixture', 'resolved_config')}), flush=True)
