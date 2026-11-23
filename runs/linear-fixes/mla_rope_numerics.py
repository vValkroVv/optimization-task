"""Localize MLA RoPE rounding with fixed projections and cotangents."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch
import triton

from mla_component import ROOT, SAC_SAVE, build_mla, load_inputs, load_module
from torchtitan.config import ActivationCheckpointConfig
from torchtitan.distributed.activation_checkpoint import _apply_op_sac
from torchtitan.models.common.attention.softmax import mla_rope_pack


def compile_fn(fn):
    return torch.compile(fn, backend="inductor", fullgraph=True, dynamic=False)


def projections(model, x):
    batch, seqlen, _ = x.shape
    if model.q_lora_rank:
        q = model.wq_b(model.alpha_q * model.q_norm(model.wq_a(x)))
    else:
        q = model.alpha_q * model.wq(x)
    latent, kr = model.wkv_a(x).split(
        (model.kv_lora_rank, model.qk_rope_head_dim), dim=-1
    )
    kn, v = model.wkv_b(model.alpha_kv * model.kv_norm(latent)).view(
        batch, seqlen, -1, model.qk_nope_head_dim + model.v_head_dim
    ).split((model.qk_nope_head_dim, model.v_head_dim), dim=-1)
    return q.view(batch, seqlen, -1, model.qk_head_dim), kn, kr, v


def stats(actual, reference):
    assert actual.shape == reference.shape and actual.dtype == reference.dtype
    bit_dtype = {torch.bfloat16: torch.int16, torch.float32: torch.int32}[actual.dtype]
    abs_sum = ref_sum = maximum = 0.0
    mismatches = bits = nonfinite_a = nonfinite_b = 0
    for a, b in zip(actual.reshape(-1).split(1 << 20), reference.reshape(-1).split(1 << 20)):
        af, bf = a.double(), b.double()
        nonfinite_a += (~af.isfinite()).sum().item()
        nonfinite_b += (~bf.isfinite()).sum().item()
        bits += (a.contiguous().view(bit_dtype) != b.contiguous().view(bit_dtype)).sum().item()
        mismatches += (af != bf).sum().item()
        delta = af - bf
        abs_sum += delta.square().sum().item()
        ref_sum += bf.square().sum().item()
        maximum = max(maximum, delta.abs().max().item())
    finite = nonfinite_a == nonfinite_b == 0
    return {
        "shape": list(actual.shape),
        "elements": actual.numel(),
        "finite": finite,
        "nonfinite_actual": nonfinite_a,
        "nonfinite_reference": nonfinite_b,
        "mismatches": mismatches,
        "bit_mismatches": bits,
        "max_abs": maximum if finite else None,
        "rmse": (abs_sum / actual.numel()) ** 0.5 if finite else None,
        "relative_l2": (abs_sum / max(ref_sum, 1e-300)) ** 0.5 if finite else None,
    }


def compare(actual, reference):
    return {name: stats(value, reference[name]) for name, value in actual.items()}


def rotate_ref(value, c, s, formula):
    """Compute explicitly rounded FP32 operations, or complex products in FP64."""
    re, im = value.unflatten(-1, (-1, 2)).unbind(-1)
    if formula == "fp64":
        re, im, c, s = (t.double() for t in (re, im, c, s))
        yr, yi = re * c - im * s, re * s + im * c
    else:
        re, im = re.float(), im.float()
        if formula == "split_fp32":
            yr, yi = re * c - im * s, re * s + im * c
        elif formula == "fma_re":
            # A binary64 product and sum emulate one FP32 FMA for these operands.
            yr = (re.double() * c.double() - (im * s).double()).float()
            yi = (re.double() * s.double() + (im * c).double()).float()
        elif formula == "fma_im":
            yr = ((re * c).double() - im.double() * s.double()).float()
            yi = ((re * s).double() + im.double() * c.double()).float()
        else:
            raise ValueError(formula)
    return torch.stack((yr, yi), dim=-1).flatten(-2)


def sum_ref(value, formula):
    if formula == "fp64_before_rotate":
        return value.sum(dim=2, dtype=torch.float64)
    if formula == "fp64_then_fp32":
        return value.sum(dim=2, dtype=torch.float64).float()
    if formula == "torch_fp32":
        return value.sum(dim=2, dtype=torch.float32)
    # These two trees differ only by head order. Apply one FP32 add at a time.
    value = value.float()
    while value.shape[2] > 1:
        if formula == "adjacent_fp32":
            value = value[:, :, ::2] + value[:, :, 1::2]
        elif formula == "halves_fp32":
            first, second = value.chunk(2, dim=2)
            value = first + second
        else:
            raise ValueError(formula)
    return value[:, :, 0]


def candidate_key_fp32(gq, gk, cache, positions, nope, rope):
    """Reuse the exact unpack source with an FP32 shared-key destination."""
    batch, seqlen, heads, _ = gq.shape
    rows = mla_rope_pack._ROWS
    query_tiles = triton.cdiv(seqlen * heads, rows)
    dq = torch.empty_like(gq)
    dkr = torch.empty((batch, seqlen, rope), dtype=torch.float32, device=gq.device)
    psb, pss, shared = mla_rope_pack._position_strides(positions)
    cache = torch.view_as_real(cache)
    mla_rope_pack._mla_rope_unpack[(batch * (query_tiles + seqlen),)](
        gq, gk, cache, positions, dq, dkr,
        seqlen, query_tiles, cache.shape[0], psb, pss, *cache.stride()[:2],
        *gq.stride(), *gk.stride(),
        HEADS=heads, NOPE=nope, ROPE=rope, ROWS=rows,
        PLAIN_TILE=min(128, triton.next_power_of_2(max(1, nope))),
        PAIR_TILE=min(32, triton.next_power_of_2(rope // 2)),
        HEAD_TILE=min(128, triton.next_power_of_2(heads)),
        HAS_POSITIONS=positions is not None, ALLOW_NEGATIVE=shared,
        num_warps=4, enable_fp_fusion=False, debug=True,
    )
    return dkr


def tap_head_sums(module_path, gq, gk, cache, positions, nope, old, new):
    """Replay Inductor's original sum and the candidate with an identity rotation."""
    batch, seqlen, heads, width = gk.shape
    rope = width - nope
    module = load_module("mla_rope_generated_backward", module_path)
    baseline_sum_kernel = module.triton_per_fused__to_copy_slice_sum_view_view_as_complex_0
    old_sum = torch.empty((batch, seqlen, rope), dtype=torch.float32, device=gk.device)
    baseline_sum_kernel.run(
        gk, old_sum, old_sum.numel(), heads, stream=torch.cuda.current_stream().cuda_stream
    )
    new_sum = candidate_key_fp32(gq, gk, torch.ones_like(cache), positions, nope, rope)
    new_rot = candidate_key_fp32(gq, gk, cache, positions, nope, rope)
    c, s = torch.view_as_real(cache[positions]).unbind(-1)
    old_calc = rotate_ref(old_sum, c, -s, "fma_re")
    new_calc = rotate_ref(new_sum, c, -s, "fma_re")
    observed_diff = old["dk_rot"] != new["dk_rot"]
    predicted_diff = old_calc.to(gk.dtype) != new_calc.to(gk.dtype)
    sum_diff = old_sum != new_sum
    pair_diff = sum_diff.unflatten(-1, (-1, 2)).any(-1, keepdim=True)
    same_sum_diff = observed_diff & (~pair_diff.expand(*pair_diff.shape[:-1], 2).flatten(-2))
    return {
        "baseline_generated_module": str(module_path),
        "baseline_generated_module_sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
        "head_sum_candidate_vs_baseline": stats(new_sum, old_sum),
        "candidate_fp32_rotation_vs_fma_from_tapped_sum": stats(new_rot, new_calc),
        "candidate_output_vs_fp32_tap_rounded": stats(new["dk_rot"], new_rot.to(gk.dtype)),
        "candidate_output_vs_fma_from_tapped_sum": stats(new["dk_rot"], new_calc.to(gk.dtype)),
        "baseline_output_vs_fma_from_tapped_sum": stats(old["dk_rot"], old_calc.to(gk.dtype)),
        "different_outputs_with_same_sum_pair": same_sum_diff.sum().item(),
        "difference_mask_mismatches": (observed_diff != predicted_diff).sum().item(),
        "disagreements": [
            {
                "index": (b, t, d),
                "baseline_sum": old_sum[b, t, d // 2 * 2 : d // 2 * 2 + 2].tolist(),
                "candidate_sum": new_sum[b, t, d // 2 * 2 : d // 2 * 2 + 2].tolist(),
                "baseline_fp32_rotated": old_calc[b, t, d].item(),
                "candidate_fp32_rotated": new_calc[b, t, d].item(),
                "baseline_bf16": old["dk_rot"][b, t, d].item(),
                "candidate_bf16": new["dk_rot"][b, t, d].item(),
            }
            for b, t, d in torch.nonzero(observed_diff)[:16].tolist()
        ],
    }


def oracle(q, kr, gq, gk, c, s, old, new, nope):
    measures = {}
    formulas = ("fp64", "split_fp32", "fma_re", "fma_im")
    for label, value, cos, sin in (
        ("q_rot", q[..., nope:], c, s),
        ("k_rot", kr, c.squeeze(2), s.squeeze(2)),
        ("dq_rot", gq[..., nope:], c, -s),
    ):
        outputs = {}
        expected = {
            "baseline": old[label][:, :, 0] if label == "k_rot" else old[label],
            "candidate": new[label][:, :, 0] if label == "k_rot" else new[label],
        }
        for formula in formulas:
            outputs[formula] = rotate_ref(value, cos, sin, formula).to(value.dtype)
        measures[label] = {
            formula: compare(expected, dict.fromkeys(expected, result))
            for formula, result in outputs.items()
        }
        measures[label]["formula_vs_fp64"] = compare(outputs, dict.fromkeys(outputs, outputs["fp64"]))

    heads = gk[..., nope:]
    sums = {
        mode: sum_ref(heads, mode)
        for mode in ("fp64_before_rotate", "fp64_then_fp32", "torch_fp32", "adjacent_fp32", "halves_fp32")
    }
    ref_sum = sums["fp64_then_fp32"]
    measures["head_sum_vs_fp64_rounded_to_fp32"] = {
        mode: stats(summed, ref_sum)
        for mode, summed in sums.items()
        if mode != "fp64_before_rotate"
    }
    expected = {"baseline": old["dk_rot"], "candidate": new["dk_rot"]}
    exact = rotate_ref(sums["fp64_before_rotate"], c.squeeze(2), -s.squeeze(2), "fp64").to(q.dtype)
    measures["dk_rot"] = {}
    for mode, summed in sums.items():
        for formula in formulas if mode != "fp64_before_rotate" else ("fp64",):
            for early_cast in (False, True) if formula == "fma_re" else (False,):
                stage = "early_bf16" if early_cast else "late_bf16"
                value = summed.to(q.dtype) if early_cast else summed
                result = rotate_ref(value, c.squeeze(2), -s.squeeze(2), formula).to(q.dtype)
                measures["dk_rot"][f"{mode}/{formula}/{stage}"] = {
                    **compare(expected, dict.fromkeys(expected, result)),
                    "vs_fp64": stats(result, exact),
                }
    indices = torch.nonzero(new["dk_rot"] != old["dk_rot"])[:12].tolist()
    measures["dk_rot_sample"] = [
        {
            "index": (batch, token, component),
            "baseline": old["dk_rot"][batch, token, component].item(),
            "candidate": new["dk_rot"][batch, token, component].item(),
            "fp64": exact[batch, token, component].item(),
            "sum_pairs": {
                mode: summed[batch, token, component // 2 * 2 : component // 2 * 2 + 2].tolist()
                for mode, summed in sums.items()
            },
        }
        for batch, token, component in indices
    ]
    return measures


def kernel_names(fn):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sorted({event.name for event in prof.events() if event.device_type == torch.autograd.DeviceType.CUDA})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seqlen", required=True, type=int, choices=(4096, 8192))
    parser.add_argument("--input-fixtures", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=19321)
    parser.add_argument("--baseline-backward-module", type=Path)
    args = parser.parse_args()
    torch._inductor.config.triton.cudagraphs = False
    torch.set_num_threads(1)
    heads = 32 if args.seqlen == 4096 else 64
    (hidden, cache, _, positions), incoming, fixtures, digest = load_inputs(args.input_fixtures, args.seqlen)
    print(f"MLA RoPE B2 S{args.seqlen} H{heads}: compile projections", flush=True)
    model = build_mla(heads, args.seed, "baseline", None)
    assert not model._mla_q_latent_z_loss_coef and not model._mla_kv_latent_z_loss_coef
    assert model.gate is not None
    with torch.no_grad():
        q, kn, kr, v = compile_fn(lambda x: projections(model, x))(hidden)
    nope = kn.shape[-1]

    def baseline(q, kn, kr, cache, positions):
        q = model._rotate_query(q, cache, positions)
        kr = model._rotate_shared_key(kr, cache, positions)
        return q, torch.cat((kn, kr.expand(-1, -1, heads, -1)), dim=-1)

    def candidate(q, kn, kr, cache, positions):
        return mla_rope_pack.MLARoPEPack.apply(q, kn, kr, cache, positions)

    baseline, candidate = compile_fn(baseline), compile_fn(candidate)

    class Tail(torch.nn.Module):
        def forward(self, q, k, v, hidden):
            output = model.inner_attention(
                q, k, v, attention_masks=None, scale=model.softmax_scale
            ).contiguous()
            output = output.view(hidden.shape[0], hidden.shape[1], -1)
            return model.wo(model.gate(hidden, output))

    tail = compile_fn(_apply_op_sac(
        Tail(),
        ActivationCheckpointConfig(selective_save=[SAC_SAVE]),
        save=frozenset({SAC_SAVE}),
    ))

    def tail_grad(q, k):
        q, k, values = (t.detach().requires_grad_() for t in (q, k, v))
        out = tail(q, k, values, hidden.detach())
        dq, dk, dv = torch.autograd.grad(out, (q, k, values), incoming)
        return {"output": out.detach(), "dq": dq, "dk": dk, "dv": dv}

    print("Replay the SAC attention tail at identical packed inputs", flush=True)
    with torch.no_grad():
        pq, pk = baseline(q, kn, kr, cache, positions)
        aq, ak = candidate(q, kn, kr, cache, positions)
    incoming_base = tail_grad(pq, pk)
    tail_control = compare(tail_grad(pq, pk), incoming_base)
    tail_diff = compare(tail_grad(aq, ak), incoming_base)
    gq, gk = incoming_base["dq"], incoming_base["dk"]

    def replay(fn, d_q=gq, d_k=gk):
        inputs = tuple(value.detach().requires_grad_() for value in (q, kn, kr))
        packed_q, packed_k = fn(*inputs, cache, positions)
        dq, dk, dkr = torch.autograd.grad((packed_q, packed_k), inputs, (d_q, d_k))
        return {
            "q_plain": packed_q.detach()[..., :nope], "q_rot": packed_q.detach()[..., nope:],
            "k_plain": packed_k.detach()[..., :nope], "k_rot": packed_k.detach()[..., nope:],
            "dq_plain": dq[..., :nope], "dq_rot": dq[..., nope:], "dk_plain": dk, "dk_rot": dkr,
        }

    print("Replay identical cotangents through each packing boundary", flush=True)
    reference = replay(baseline)
    actual = replay(candidate)
    fixed_control = compare(replay(baseline), reference)
    fixed_diff = compare(actual, reference)
    fixed_candidate_control = compare(replay(candidate), actual)
    markers = {
        "baseline": kernel_names(lambda: replay(baseline)),
        "candidate": kernel_names(lambda: replay(candidate)),
        "sac_tail": kernel_names(lambda: tail_grad(pq, pk)),
    }
    assert not any("_mla_rope_" in name for name in markers["baseline"])
    assert all(any(marker in name for name in markers["candidate"])
               for marker in ("_mla_rope_pack", "_mla_rope_unpack"))
    assert any("cudnn" in name.lower() or "flash" in name.lower()
               for name in markers["sac_tail"])

    print("Compare explicit arithmetic and FP64 oracles", flush=True)
    c, s = torch.view_as_real(cache[positions]).unbind(-1)
    arithmetic = oracle(q, kr, gq, gk, c.unsqueeze(2), s.unsqueeze(2), reference, actual, nope)
    if args.baseline_backward_module:
        print("Tap actual FP32 head sums and reconstruct both BF16 outputs", flush=True)
        arithmetic["actual_head_sums"] = tap_head_sums(
            args.baseline_backward_module, gq, gk, cache, positions, nope, reference, actual,
        )

    report = {
        "command": sys.argv,
        "gift_revision": subprocess.check_output(
            ["git", "-C", str(ROOT / "gift"), "rev-parse", "HEAD"], text=True
        ).strip(),
        "kernel_sha256": digest(Path(mla_rope_pack.__file__)),
        "diagnostic_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "triton": triton.__version__,
        "device": torch.cuda.get_device_name(),
        "shape": {"batch": 2, "sequence": args.seqlen, "heads": heads, "nope": nope, "rope": kr.shape[-1]},
        "seed": args.seed, "dtype": str(q.dtype), "fixtures": fixtures,
        "compile": "Inductor fullgraph, static, CUDA graphs disabled",
        "tail": f"GIFT MLA attention, gate and FP8 output projection with GIFT op SAC save:{SAC_SAVE}",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "strides": {n: list(t.stride()) for n, t in dict(q=q, k_plain=kn, k_shared=kr, v=v, packed_q=pq, packed_k=pk, grad_q=gq, grad_k=gk).items()},
        "kernels": markers,
        "tail_aa_fixed_packed": tail_control,
        "tail_ab": tail_diff,
        "fixed_baseline_aa": fixed_control,
        "fixed_candidate_bb": fixed_candidate_control,
        "fixed_candidate_vs_baseline": fixed_diff,
        "arithmetic": arithmetic,
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
