"""Joint MoE spectrum + gate↔up alignment over an HF safetensors export.

For each ``(layer, projection)`` group of MoE experts, accumulate
``G = Σ_e W_eᵀ W_e`` streamingly (never materialising the stacked
matrix), then take top-K eigendecomposition. From the same pass,
compute per-expert energy along each top direction and the
participation ratio.

HF exports give us individual ``mlp.experts.<i>.<proj>.weight``
tensors, so we just group by ``(layer, proj)`` and iterate — no
packed-tensor slicing logic needed.

Derived per layer: a ``gate ↔ up`` alignment score
``σ_g_1 · σ_u_1 · |⟨v_g_1, v_u_1⟩|``. High score predicts activation
blow-up at ``down_proj`` input without ever running the network on
activations.

Output JSON shape::

    {
        "meta": {<run metadata>},
        "joint": {
            "L<layer>.<proj>": {
                "n_experts": int, "M": int, "N": int, "k": int,
                "lambda": [top-k eigenvalues],
                "sigma":  [√λ — top-k singular values],
                "joint_stable_rank": float,
                "v": [[...N...] × k],
                "proj_norm_per_expert": [[...k...] × n_experts],
                "participation_ratio": [k]
            },
            ...
        },
        "alignment": {
            "L<layer>": {sigma_gate_1, sigma_up_1, cos_v_gate_up_top1,
                         cos_v_gate_up_top2, explosion_score,
                         joint_sr_gate, joint_sr_up},
            ...
        }
    }
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from tqdm.auto import tqdm

from weights._timing import StageTimer
from weights.analyze import classify
from weights.hf_reader import SafetensorsReader


# ---------------------------------------------------------------------------
# Group enumeration — gather expert FQNs per (layer, proj).
# ---------------------------------------------------------------------------
def _enumerate_groups(names: Iterable[str]) -> dict[tuple[int, str], list[str]]:
    """Return ``{(layer, proj): [fqn, ...]}`` for every MoE-expert group.

    Only ``role == "moe_expert"`` tensors are aggregated. Shared experts
    and dense MLP are NOT mixed in — they aren't part of the "joint
    expert subspace" question.
    """
    groups: dict[tuple[int, str], list[str]] = defaultdict(list)
    for name in names:
        c = classify(name)
        if c["role"] != "moe_expert":
            continue
        if c["layer"] is None or c["proj"] is None:
            continue
        groups[(c["layer"], c["proj"])].append(name)
    # Sort each group's FQNs by expert index for deterministic iteration.
    for k in groups:
        groups[k].sort(key=lambda n: _expert_idx(n) or -1)
    return dict(groups)


_EXPERT_IDX_RE = re.compile(r"\.experts\.(\d+)\.")


def _expert_idx(name: str) -> int | None:
    m = _EXPERT_IDX_RE.search(name)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Per-(layer, proj) joint compute
# ---------------------------------------------------------------------------
def compute_layer_joint(
    expert_tensors: list[torch.Tensor],
    *,
    device: str,
    k: int,
    gpu_gram_cap_gb: float = 30.0,
    max_gram_gb: float = 200.0,
    cpu_fallback: bool = False,
    timer: StageTimer | None = None,
) -> dict[str, Any] | None:
    """Streaming joint Gram + top-K eigendecomposition over per-expert tensors.

    Each tensor must be 2-D ``[M, N]`` where ``N`` is the last axis (the
    shared output dimension across experts; the Gram is ``[N, N]``).

    Returns ``None`` if the Gram exceeds the configured cap, or if the
    input list is empty.
    """
    if not expert_tensors:
        return None
    first = expert_tensors[0]
    if first.ndim != 2:
        raise ValueError(f"expert tensor must be 2-D, got shape {list(first.shape)}")
    M0, N0 = first.shape
    gram_gb = N0 * N0 * 4 / 1e9
    actual_device = device
    if actual_device.startswith("cuda") and gram_gb > gpu_gram_cap_gb:
        if not cpu_fallback:
            print(
                f"  [joint] SKIP: N={N0}, Gram = {gram_gb:.1f} GB > "
                f"{gpu_gram_cap_gb:.0f} GB GPU cap. Pass --cpu-fallback "
                f"to try CPU (slow).",
                flush=True,
            )
            return None
        print(
            f"  [joint] N={N0}, Gram = {gram_gb:.1f} GB > GPU cap; falling back to CPU",
            flush=True,
        )
        actual_device = "cpu"
    if gram_gb > max_gram_gb:
        print(
            f"  [joint] SKIP: N={N0}, Gram = {gram_gb:.1f} GB > "
            f"{max_gram_gb:.0f} GB hard cap.",
            flush=True,
        )
        return None

    t = timer or StageTimer()
    is_cuda = actual_device.startswith("cuda")

    # Pass 1 — G = Σ Wᵀ W.
    G = torch.zeros(N0, N0, dtype=torch.float32, device=actual_device)
    count = 0
    M = M0
    N = N0
    with t.stage("joint_pass1_gram", sync_cuda=is_cuda):
        for W in expert_tensors:
            if W.ndim != 2 or W.shape[1] != N0:
                raise ValueError(
                    f"inconsistent expert shape; expected (M, {N0}), got "
                    f"{list(W.shape)}"
                )
            Wf = W.to(device=actual_device, dtype=torch.float32, non_blocking=True)
            G.addmm_(Wf.t(), Wf)
            count += 1
            del Wf

    with t.stage("joint_eigh", sync_cuda=is_cuda):
        G = 0.5 * (G + G.t())  # symmetrise against accumulation drift
        eigvals, eigvecs = torch.linalg.eigh(G)
        k_eff = min(k, N)
        top_lambda = eigvals[-k_eff:].flip(0).clamp_min(0.0)
        top_v = eigvecs[:, -k_eff:].flip(1).contiguous()  # [N, k]
        sum_lambda = float(eigvals.clamp_min(0).sum())
        sigma1_sq = float(top_lambda[0]) if k_eff >= 1 else 0.0
        joint_stable_rank = (sum_lambda / sigma1_sq) if sigma1_sq > 0 else 0.0
        del eigvals, eigvecs, G

    # Pass 2 — per-expert projection energies along each top direction.
    proj_norms: list[list[float]] = []
    with t.stage("joint_pass2_proj", sync_cuda=is_cuda):
        for W in expert_tensors:
            Wf = W.to(device=actual_device, dtype=torch.float32, non_blocking=True)
            Wv = Wf @ top_v
            proj_norms.append([float(v) for v in torch.linalg.vector_norm(Wv, dim=0)])
            del Wf, Wv

    # Participation ratio: PR_k = (Σ p_e)² / Σ p_e²,  p_e = ||W_e v_k||²
    pn = torch.tensor(proj_norms, dtype=torch.float32, device=actual_device).pow(2)
    pr = ((pn.sum(0) ** 2) / (pn.pow(2).sum(0).clamp_min(1e-30))).cpu().tolist()

    return {
        "n_experts": count,
        "M": M,
        "N": N,
        "k": k_eff,
        "lambda": top_lambda.cpu().tolist(),
        "sigma": top_lambda.sqrt().cpu().tolist(),
        "joint_stable_rank": joint_stable_rank,
        "v": top_v.t().cpu().tolist(),
        "proj_norm_per_expert": proj_norms,
        "participation_ratio": pr,
    }


# ---------------------------------------------------------------------------
# Gate ↔ up alignment score
# ---------------------------------------------------------------------------
def compute_alignment(joint: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """For every layer with both ``L{L}.gate`` and ``L{L}.up`` joint entries,
    compute ``score = σ_g_1 · σ_u_1 · |⟨v_g_1, v_u_1⟩|`` plus a top-2
    cosine for context.
    """
    by_layer: dict[int, dict[str, dict[str, Any]]] = {}
    for key, v in joint.items():
        m = re.match(r"L(\d+)\.(\w+)", key)
        if not m:
            continue
        by_layer.setdefault(int(m.group(1)), {})[m.group(2)] = v

    out: dict[str, dict[str, Any]] = {}
    for L, projs in sorted(by_layer.items()):
        gate = projs.get("gate")
        up = projs.get("up")
        if gate is None or up is None or not gate.get("v") or not up.get("v"):
            continue
        sg = float(gate["sigma"][0])
        su = float(up["sigma"][0])
        v_g = gate["v"][0]
        v_u = up["v"][0]
        ng = math.sqrt(sum(a * a for a in v_g)) or 1.0
        nu = math.sqrt(sum(a * a for a in v_u)) or 1.0
        cos = abs(sum(a * b for a, b in zip(v_g, v_u))) / (ng * nu)
        cos_2: float | None = None
        if len(gate["v"]) > 1 and len(up["v"]) > 1:
            v_g_2, v_u_2 = gate["v"][1], up["v"][1]
            ng2 = math.sqrt(sum(a * a for a in v_g_2)) or 1.0
            nu2 = math.sqrt(sum(a * a for a in v_u_2)) or 1.0
            cos_2 = abs(sum(a * b for a, b in zip(v_g_2, v_u_2))) / (ng2 * nu2)
        out[f"L{L}"] = {
            "sigma_gate_1": sg,
            "sigma_up_1": su,
            "cos_v_gate_up_top1": cos,
            "cos_v_gate_up_top2": cos_2,
            "explosion_score": sg * su * cos,
            "joint_sr_gate": float(gate.get("joint_stable_rank") or 0),
            "joint_sr_up": float(up.get("joint_stable_rank") or 0),
        }
    return out


def print_alignment_table(
    align: dict[str, dict[str, Any]],
    *,
    top_n: int = 12,
    warn: float = 1e3,
    danger: float = 1e4,
) -> None:
    if not align:
        print("[align] no aligned (gate, up) pairs found")
        return
    rows = sorted(align.items(), key=lambda kv: -kv[1]["explosion_score"])
    print("\n=== gate ↔ up rank-1 alignment ===")
    print(
        f"{'rank':>4} {'layer':>6} {'σ_g':>8} {'σ_u':>8} {'|cos₁|':>7} "
        f"{'|cos₂|':>7} {'score':>10}  {'sr_g':>7} {'sr_u':>7}"
    )
    for i, (L, d) in enumerate(rows[:top_n]):
        c2 = d.get("cos_v_gate_up_top2")
        c2s = f"{c2:>7.3f}" if c2 is not None else "      -"
        flag = (
            "  *** DANGER"
            if d["explosion_score"] >= danger
            else ("  ** warn" if d["explosion_score"] >= warn else "")
        )
        print(
            f"{i + 1:>4} {L:>6} {d['sigma_gate_1']:>8.1f} "
            f"{d['sigma_up_1']:>8.1f} {d['cos_v_gate_up_top1']:>7.3f} "
            f"{c2s} {d['explosion_score']:>10.1f}  "
            f"{d['joint_sr_gate']:>7.1f} {d['joint_sr_up']:>7.1f}{flag}"
        )
    n_warn = sum(1 for _, d in rows if d["explosion_score"] >= warn)
    n_danger = sum(1 for _, d in rows if d["explosion_score"] >= danger)
    print(
        f"  {len(rows)} layers; {n_warn} ≥ warn ({warn:.0g}), "
        f"{n_danger} ≥ danger ({danger:.0g})."
    )


# ---------------------------------------------------------------------------
# Config + entry point
# ---------------------------------------------------------------------------
@dataclass
class Config:
    device: str | None = None
    k: int = 4
    # If the Gram matrix exceeds this on a CUDA device, skip that
    # (layer, proj) group by default. Pass cpu_fallback=True for CPU.
    gpu_gram_cap_gb: float = 30.0
    # Hard cap even with cpu_fallback=True.
    max_gram_gb: float = 200.0
    cpu_fallback: bool = False


def _device(cfg: Config) -> str:
    if cfg.device:
        return cfg.device
    return "cuda" if torch.cuda.is_available() else "cpu"


def run_hf(
    hf_dir: Path | str,
    out_path: Path | str,
    *,
    config: Config | None = None,
) -> Path:
    """Build joint Gram + alignment for every (layer, proj) MoE group in an HF export."""
    cfg = config or Config()
    device = _device(cfg)
    out_path = Path(out_path)

    reader = SafetensorsReader(hf_dir)
    groups = _enumerate_groups(reader.names)
    if not groups:
        raise RuntimeError(
            f"no MoE expert tensors found in {hf_dir} (looked for role=moe_expert)."
        )

    print(
        f"[joint] {len(groups)} (layer, proj) groups; device={device}; "
        f"k={cfg.k}; gpu_gram_cap_gb={cfg.gpu_gram_cap_gb}",
        flush=True,
    )

    timer = StageTimer()
    joint_out: dict[str, dict[str, Any]] = {}
    skipped: list[str] = []
    t_start = time.time()

    for (layer, proj), fqns in tqdm(sorted(groups.items()), desc="joint"):
        # Load all per-expert tensors for this group. They're small enough
        # (e.g. 1280×1920 bf16 ≈ 5 MB; 64 experts → ~300 MB) to hold in
        # RAM together for the streaming Gram pass.
        tensors_by_name = reader.load_many(fqns)
        expert_tensors = [tensors_by_name[fqn] for fqn in fqns]
        out = compute_layer_joint(
            expert_tensors,
            device=device,
            k=cfg.k,
            gpu_gram_cap_gb=cfg.gpu_gram_cap_gb,
            max_gram_gb=cfg.max_gram_gb,
            cpu_fallback=cfg.cpu_fallback,
            timer=timer,
        )
        key = f"L{layer}.{proj}"
        if out is None:
            skipped.append(key)
            joint_out[key] = {"error": "skipped (Gram cap exceeded)"}
            continue
        joint_out[key] = out
        for fqn in fqns:
            del tensors_by_name[fqn]
        del tensors_by_name, expert_tensors
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    alignment = compute_alignment(joint_out)
    elapsed = time.time() - t_start
    timer.print_summary()
    print_alignment_table(alignment)

    blob = {
        "meta": {
            "n_groups": len(joint_out),
            "n_skipped": len(skipped),
            "skipped": skipped,
            "elapsed_sec": elapsed,
            "device": device,
            "k": cfg.k,
            "gpu_gram_cap_gb": cfg.gpu_gram_cap_gb,
            "max_gram_gb": cfg.max_gram_gb,
            "cpu_fallback": cfg.cpu_fallback,
            "hf_dir": str(reader.path),
            "timing": timer.totals(),
        },
        "joint": joint_out,
        "alignment": alignment,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(blob))
    print(
        f"\nWrote {out_path}  "
        f"({len(joint_out)} groups, {len(skipped)} skipped, "
        f"{len(alignment)} aligned layers, {elapsed:.1f}s)"
    )
    return out_path
