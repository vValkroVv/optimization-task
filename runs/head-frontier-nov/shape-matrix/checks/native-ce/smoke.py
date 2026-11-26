import json
import sys
from pathlib import Path

import torch

from torchtitan.ops import exact_folded_ce

run = Path(sys.argv[1])
dynamic_artifact = run / "native-ce/centered-folded-v1.so"
dynamic_sha256 = (run / "native-ce/sha256.txt").read_text().splitlines()[-1].split()[0]
original_artifact = run.parent / "fused-ce/centered.so"

load_original = exact_folded_ce._load_launcher
used_pointers = []


def load_and_record(path, sha256):
    launch = load_original(path, sha256)

    def record(logits_pointer, *args):
        used_pointers.append(logits_pointer)
        return launch(logits_pointer, *args)

    return record


exact_folded_ce._load_launcher = load_and_record
candidate = exact_folded_ce.ExactFoldedCrossEntropy(
    str(dynamic_artifact), library_sha256=dynamic_sha256
)
legacy = exact_folded_ce.ExactFoldedCrossEntropy(str(original_artifact))


def execute(loss, source, labels, weights, count, unaligned=False):
    if unaligned:
        storage = source.new_empty(source.numel() + 1)
        input_logits = storage[1:].view_as(source)
        input_logits.copy_(source)
    else:
        input_logits = source.clone()
    input_logits.requires_grad_(True)
    ptr = input_logits.data_ptr()
    raw = loss(input_logits, labels, weights, count)
    total = sum(value / count * weight for value, weight in zip(raw, weights, strict=True))
    total.backward()
    torch.cuda.synchronize()
    assert used_pointers[-1] == ptr, (used_pointers[-1], ptr)
    assert torch.equal(input_logits, input_logits.grad)
    return torch.stack(raw).detach(), input_logits.grad


def fp32_oracle(source, labels, weights, count, grads):
    heads = len(weights)
    flat = source.view(heads, -1, source.size(-1))
    targets = labels.view(heads, -1)
    observed = grads.view_as(flat)
    losses = []
    matched = 0
    max_error = 0.0
    for i, weight in enumerate(weights):
        scale = torch.tensor(weight, device=source.device) / count
        head_losses = []
        for block, y, actual in zip(flat[i].split(64), targets[i].split(64), observed[i].split(64), strict=True):
            shifted = block.float()
            shifted -= shifted.amax(dim=-1, keepdim=True)
            logdenom = shifted.exp().sum(dim=-1, keepdim=True).log()
            valid = y != -100
            target = y.clamp_min(0).unsqueeze(-1)
            centered = shifted - logdenom
            head_losses.append((-centered.gather(-1, target)).flatten().masked_fill(~valid, 0.0))
            dz = centered.exp()
            dz.scatter_add_(-1, target, -torch.ones_like(logdenom))
            dz *= scale
            dz.masked_fill_(~valid.unsqueeze(-1), 0.0)
            expected = dz.to(torch.bfloat16)
            matched += (actual == expected).sum().item()
            max_error = max(max_error, (actual.float() - expected.float()).abs().max().item())
            torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.0)
        losses.append(torch.cat(head_losses).sum())
    return torch.stack(losses), matched, max_error


torch.manual_seed(63)
print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(), "candidate": dynamic_sha256, "legacy": legacy.library_sha256}), flush=True)
for name, batch, sequence, vocab, weights, count, unaligned in [
    ("100B-C2", 2, 2048, 128256, [1.0, 0.3], 2097152.0, False),
    ("100B-C8", 2, 512, 128256, [1.0, 0.3], 2097152.0, False),
    ("700B-C8", 3, 512, 128256, [1.0, 0.1], 3012345.0, False),
    ("odd-vocab-short-chunk-MTP10", 2, 3, 4099, [1.0, 0.3, 0.0, *[0.1] * 8], 222.0, True),
    ("aligned-small-vocab", 2, 5, 16392, [1.0, 0.2], 37.0, False),
    ("unaligned-base-aligned-vocab", 2, 1, 24, [1.0, 0.2], 37.0, True),
]:
    heads = len(weights)
    source = torch.randn((heads * batch, sequence, vocab), dtype=torch.bfloat16, device="cuda")
    labels = torch.randint(vocab, source.shape[:-1], device="cuda")
    labels[:, 0] = -100
    labels[0, -1] = vocab - 1
    if heads > 2:
        labels[batch * 5:batch * 6] = -100
    actual_loss, actual_grad = execute(candidate, source, labels, weights, count, unaligned)
    if vocab == 128256:
        baseline_loss, baseline_grad = execute(legacy, source, labels, weights, count)
        assert torch.equal(actual_loss, baseline_loss)
        assert torch.equal(actual_grad, baseline_grad)
    expected_loss, matched, max_error = fp32_oracle(source, labels, weights, count, actual_grad)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-6, atol=1e-5)
    print(json.dumps({
        "case": name, "shape": list(source.shape), "heads": heads,
        "legacy_bitwise": True if vocab == 128256 else None,
        "oracle_loss_max_abs": (actual_loss - expected_loss).abs().max().item(),
        "oracle_dz_bitwise_fraction": matched / source.numel(),
        "oracle_dz_max_abs": max_error,
        "overwritten_allocation": True,
    }), flush=True)
    del source, actual_grad
    if vocab == 128256:
        del baseline_grad
