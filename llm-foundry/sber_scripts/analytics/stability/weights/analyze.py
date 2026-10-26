"""Per-tensor analysis of a HF safetensors export.

For every tensor in the export (or a filtered subset), compute:

* **basic**: counts (NaN/Inf/zero/pos/neg), moments (mean/std/skew/kurt),
  Frobenius norm, top-K |x| means, ``log10|x|`` histogram.
* **rowcol** (for 2-D tensors): per-row & per-column L2 stats, with
  ``max/median`` as the rank-1-collapse detector.
* **spectral** (for 2-D tensors with finite size): top-K randomised SVD —
  σ samples, spectral norm, nuclear-top-K, stable rank, effective rank.

I/O strategy
------------
Safetensors are tab-indexed and mmap-able: ``f.get_tensor(name)`` is
effectively free. We just iterate every (or filtered) tensor, group by
shard so each shard is opened once, and run the stats on the device
of choice.

Output JSON shape::

    {
        "meta": {<run metadata>},
        "tensors": {
            "<fqn>": {
                "name": str, "shape": [...], "dtype": str,
                "role": str, "proj": str|None, "layer": int|None,
                "expert": int|None, "block": "main"|"mtp",
                "base": {<basic stats>},
                "rowcol": {"row": {...}, "col": {...}},   # if 2-D
                "spectral": {<svd stats>}                  # if 2-D & SPECTRAL_ROLE
            },
            ...
        }
    }

Per-expert handling: HF exports unpack MoE experts into individual
``mlp.experts.<i>.<proj>.weight`` tensors, so each gets its own record
with ``role="moe_expert"`` and ``expert=<i>``. No extra ``per_expert``
sub-block is needed.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from tqdm.auto import tqdm

from weights._timing import StageTimer
from weights.hf_reader import SafetensorsReader


# ---------------------------------------------------------------------------
# Tensor stat constants
# ---------------------------------------------------------------------------
HIST_LOG_MIN = -8.0
HIST_LOG_MAX = 4.0
HIST_BINS = 80
TOP_K_TAIL = 128
SPECTRAL_NUMEL_CAP = 20 * 1024**3 // 4  # 20 GB at fp32


# ---------------------------------------------------------------------------
# Classification — name → role/layer/proj/expert/block
# ---------------------------------------------------------------------------
_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
_EXPERT_RE = re.compile(r"\.experts\.(\d+)\.")

# (regex, role, proj). Priority order — first match wins. ``proj`` may be
# a literal or ``None``; literal "_capture" means use regex group 1.
_CLASS_PATTERNS: list[tuple[str, str, str | None]] = [
    # ---- top-level -------------------------------------------------------
    (r"^model\.embed_tokens\.weight$", "embedding", "embed_tokens"),
    (r"(^|\.)lm_head\.weight$", "lm_head", "lm_head"),
    (r"^model\.norm\.weight$", "final_norm", "weight"),
    (r"^model\.norm\.gate_up_projection\.weight$", "final_norm", "gate_up"),
    (r"^model\.norm\.gate_down_projection\.weight$", "final_norm", "gate_down"),
    # ---- MTP per-layer extras --------------------------------------------
    (r"\.eh_proj\.weight$", "mtp", "eh_proj"),
    (r"\.shared_head\.head\.weight$", "mtp", "shared_head"),
    (r"\.shared_head\.norm\.weight$", "mtp", "shared_head_norm"),
    (
        r"\.shared_head\.norm\.gate_up_projection\.weight$",
        "mtp",
        "shared_head_norm_gate_up",
    ),
    (
        r"\.shared_head\.norm\.gate_down_projection\.weight$",
        "mtp",
        "shared_head_norm_gate_down",
    ),
    (r"\.enorm\.weight$", "mtp", "enorm"),
    (r"\.enorm\.gate_up_projection\.weight$", "mtp", "enorm_gate_up"),
    (r"\.enorm\.gate_down_projection\.weight$", "mtp", "enorm_gate_down"),
    (r"\.hnorm\.weight$", "mtp", "hnorm"),
    (r"\.hnorm\.gate_up_projection\.weight$", "mtp", "hnorm_gate_up"),
    (r"\.hnorm\.gate_down_projection\.weight$", "mtp", "hnorm_gate_down"),
    (r"\.layers\.\d+\.embed_tokens\.weight$", "mtp", "embed_tokens"),
    # ---- MLA (DeepSeek-style) attention ---------------------------------
    (r"\.self_attn\.q_a_proj\.weight$", "mla", "q_a"),
    (r"\.self_attn\.q_b_proj\.weight$", "mla", "q_b"),
    (r"\.self_attn\.kv_a_proj(_with_mqa)?\.weight$", "mla", "kv_a"),
    (r"\.self_attn\.kv_b_proj\.weight$", "mla", "kv_b"),
    (r"\.self_attn\.q_a_layernorm\.weight$", "layernorm", "mla_q_a_weight"),
    (
        r"\.self_attn\.q_a_layernorm\.gate_up_projection\.weight$",
        "layernorm",
        "mla_q_a_gate_up",
    ),
    (
        r"\.self_attn\.q_a_layernorm\.gate_down_projection\.weight$",
        "layernorm",
        "mla_q_a_gate_down",
    ),
    (r"\.self_attn\.kv_a_layernorm\.weight$", "layernorm", "mla_kv_a_weight"),
    (
        r"\.self_attn\.kv_a_layernorm\.gate_up_projection\.weight$",
        "layernorm",
        "mla_kv_a_gate_up",
    ),
    (
        r"\.self_attn\.kv_a_layernorm\.gate_down_projection\.weight$",
        "layernorm",
        "mla_kv_a_gate_down",
    ),
    # ---- Linear-attention / Mamba / DeltaNet -----------------------------
    (r"\.self_attn\.qkv_proj\.weight$", "linear_attn", "qkv"),
    (r"\.self_attn\.qkv_conv1d\.weight$", "linear_attn", "qkv_conv1d"),
    (r"\.self_attn\.conv1d\.weight$", "linear_attn", "conv1d"),
    (r"\.self_attn\.gate_proj\.weight$", "linear_attn", "gate"),
    (r"\.self_attn\.z_proj\.weight$", "linear_attn", "z"),
    (r"\.self_attn\.in_proj_qkvz\.weight$", "linear_attn", "in_qkvz"),
    (r"\.self_attn\.in_proj_ba\.weight$", "linear_attn", "in_ba"),
    (r"\.self_attn\.ba_proj\.weight$", "linear_attn", "ba"),
    (r"\.self_attn\.o_proj\.weight$", "linear_attn", "o"),
    (r"\.self_attn\.out_proj\.weight$", "linear_attn", "out"),
    (r"\.self_attn\.A_log$", "ssm_param", "A_log"),
    (r"\.self_attn\.dt_bias$", "ssm_param", "dt_bias"),
    (r"\.self_attn\.norm\.weight$", "ssm_param", "norm"),
    (r"\.self_attn\.o_norm\.weight$", "ssm_param", "o_norm"),
    # ---- MoE ------------------------------------------------------------
    (r"\.mlp\.gate\.weight$", "moe_router", "weight"),
    (r"\.mlp\.gate\.e_score_correction_bias$", "moe_router", "bias"),
    (r"\.mlp\.experts\.\d+\.gate_proj\.weight$", "moe_expert", "gate"),
    (r"\.mlp\.experts\.\d+\.up_proj\.weight$", "moe_expert", "up"),
    (r"\.mlp\.experts\.\d+\.down_proj\.weight$", "moe_expert", "down"),
    (r"\.mlp\.shared_experts\.gate_proj\.weight$", "moe_shared_expert", "gate"),
    (r"\.mlp\.shared_experts\.up_proj\.weight$", "moe_shared_expert", "up"),
    (r"\.mlp\.shared_experts\.down_proj\.weight$", "moe_shared_expert", "down"),
    # ---- Dense MLP (no MoE; layers that don't have .experts. or .shared_experts.) ----
    (r"\.mlp\.gate_proj\.weight$", "dense_mlp", "gate"),
    (r"\.mlp\.up_proj\.weight$", "dense_mlp", "up"),
    (r"\.mlp\.down_proj\.weight$", "dense_mlp", "down"),
    # ---- Layer-level GatedRMSNorms --------------------------------------
    (
        r"\.(input_layernorm|post_attention_layernorm|post_self_attn_layernorm|post_feedforward_layernorm)\.weight$",
        "layernorm",
        "_capture_weight",
    ),
    (
        r"\.(input_layernorm|post_attention_layernorm|post_self_attn_layernorm|post_feedforward_layernorm)\.gate_up_projection\.weight$",
        "layernorm",
        "_capture_gate_up",
    ),
    (
        r"\.(input_layernorm|post_attention_layernorm|post_self_attn_layernorm|post_feedforward_layernorm)\.gate_down_projection\.weight$",
        "layernorm",
        "_capture_gate_down",
    ),
]
_CLASS_COMPILED = [(re.compile(p), r, pr) for p, r, pr in _CLASS_PATTERNS]


# Roles for which top-K SVD is meaningful (and worth the cost).
SPECTRAL_ROLES: frozenset[str] = frozenset(
    {
        "embedding",
        "lm_head",
        "mla",
        "linear_attn",
        "moe_expert",
        "moe_shared_expert",
        "moe_router",
        "dense_mlp",
        "mtp",  # eh_proj / shared_head — matrix-valued
    }
)


# A tensor is in the "mtp" block when it sits in an MTP-specific
# submodule. ``model.layers.<N>.<one of these>`` → block="mtp", otherwise
# "main". We can't rely on a `mtp_block.mtp_layers.N` path here because
# this checkpoint family fuses MTP into the layer index instead.
_MTP_MARKERS = re.compile(
    r"\.(eh_proj|enorm|hnorm|shared_head)\."
    r"|\.layers\.\d+\.embed_tokens\.weight$"
)


def classify(name: str) -> dict[str, Any]:
    """Return ``{role, layer, expert, proj, block}`` for one tensor FQN.

    ``block`` is ``"mtp"`` for MTP-specific tensors (per-layer
    ``eh_proj``/``enorm``/``hnorm``/``embed_tokens``/``shared_head``)
    and ``"main"`` otherwise.
    """
    role = "unknown"
    proj: str | None = None
    for rx, r, pr in _CLASS_COMPILED:
        m = rx.search(name)
        if not m:
            continue
        role = r
        if pr is None:
            proj = None
        elif pr.startswith("_capture"):
            # `_capture_<suffix>` → use group(1) + optional fixed suffix
            captured = m.group(1) if m.groups() else None
            tail = pr[len("_capture") :].lstrip("_")
            proj = (
                f"{captured}_{tail}"
                if (captured and tail)
                else (captured or tail or None)
            )
        else:
            proj = pr
        break

    m_layer = _LAYER_RE.search(name)
    m_expert = _EXPERT_RE.search(name)
    block = "mtp" if _MTP_MARKERS.search(name) else "main"
    return {
        "role": role,
        "layer": int(m_layer.group(1)) if m_layer else None,
        "expert": int(m_expert.group(1)) if m_expert else None,
        "proj": proj,
        "block": block,
    }


# ---------------------------------------------------------------------------
# Per-tensor stat functions.
# ---------------------------------------------------------------------------
def basic_stats(x: torch.Tensor) -> dict[str, Any]:
    """One tensor → dict of basic stats. ``x`` is 1-D (flattened)."""
    x = x.reshape(-1)
    n_nan = int(torch.isnan(x).sum())
    n_inf = int(torch.isinf(x).sum())
    n_zero = int((x == 0).sum())
    n_pos = int((x > 0).sum())
    n_neg = int((x < 0).sum())
    absx = x.abs()
    mean = float(x.mean())
    centered = x - mean
    m2 = float(centered.pow(2).mean().clamp_min(0))
    m3 = float(centered.pow(3).mean())
    m4 = float(centered.pow(4).mean())
    std = m2**0.5
    skew = m3 / (m2**1.5 + 1e-30)
    kurt = m4 / (m2**2 + 1e-30) - 3.0
    k_tail = min(TOP_K_TAIL, x.numel())
    top = torch.topk(absx, k_tail).values
    safe = absx.clamp_min(1e-38)
    log10_abs = torch.log10(safe)
    width = (HIST_LOG_MAX - HIST_LOG_MIN) / HIST_BINS
    idx = ((log10_abs - HIST_LOG_MIN) / width).long().clamp_(0, HIST_BINS - 1)
    hist = torch.zeros(HIST_BINS, dtype=torch.int64, device=x.device)
    hist.scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.int64))
    return {
        "n_nan": n_nan,
        "n_inf": n_inf,
        "n_zero": n_zero,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "mean": mean,
        "std": std,
        "fro": float(x.pow(2).sum().sqrt()),
        "mean_abs": float(absx.mean()),
        "max_abs": float(absx.amax()),
        "min_signed": float(x.amin()),
        "max_signed": float(x.amax()),
        "skew": skew,
        "kurt_excess": kurt,
        "top16_abs_mean": float(top[: min(16, k_tail)].mean()),
        "top64_abs_mean": float(top[: min(64, k_tail)].mean()),
        "top128_abs_mean": float(top[:k_tail].mean()),
        "abs_hist_counts": hist.cpu().tolist(),
    }


def rowcol_stats(W: torch.Tensor) -> dict[str, dict[str, float]]:
    """Per-row & per-column L2 stats. ``W`` is 2-D."""
    row = W.pow(2).sum(dim=1).sqrt()
    col = W.pow(2).sum(dim=0).sqrt()

    def s(v: torch.Tensor) -> dict[str, float]:
        med = float(v.median())
        return {
            "min": float(v.amin()),
            "max": float(v.amax()),
            "median": med,
            "mean": float(v.mean()),
            "std": float(v.std(unbiased=False)),
            "max_over_median": float(v.amax()) / (med + 1e-30),
        }

    return {"row": s(row), "col": s(col)}


def spectral_stats(W: torch.Tensor, k: int, niter: int) -> dict[str, Any]:
    """Top-K randomised SVD of 2-D ``W``."""
    q = min(k, *W.shape)
    try:
        _, S, _ = torch.svd_lowrank(W, q=q, niter=niter)
    except Exception as e:  # noqa: BLE001
        return {"error": f"svd_lowrank_failed: {e!r}"}
    fro2 = float(W.pow(2).sum())
    sig1 = float(S[0])
    p = S / (S.sum().clamp_min(1e-30))
    eff_rank = float(torch.exp(-(p * (p + 1e-30).log()).sum()))
    stable_rank = fro2 / (sig1**2 + 1e-30)
    idxs = [0, min(1, q - 1), max(0, q // 4), max(0, q // 2), q - 1]
    return {
        "k": q,
        "sigma_sample_idx": idxs,
        "sigma_samples": [float(S[i]) for i in idxs],
        "spectral_norm": sig1,
        "nuclear_topk_approx": float(S.sum()),
        "stable_rank": stable_rank,
        "effective_rank_entropy_topk": eff_rank,
    }


# ---------------------------------------------------------------------------
# Config + entry point
# ---------------------------------------------------------------------------
@dataclass
class Config:
    device: str | None = None
    topk_sigma: int = 16
    svd_niter: int = 2
    # Detailed per-tensor timing prints. Off by default — produces noisy logs.
    debug_timing: bool = False


def _device(cfg: Config) -> str:
    if cfg.device:
        return cfg.device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _analyze_one(
    name: str,
    tensor: torch.Tensor,
    device: str,
    cfg: Config,
    timer: StageTimer,
) -> dict[str, Any]:
    """Compute basic + rowcol + spectral stats for one tensor."""
    meta = classify(name)
    rec: dict[str, Any] = {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        **meta,
    }
    is_cuda = device.startswith("cuda")

    with timer.stage("to_device", sync_cuda=is_cuda):
        t = tensor.to(device=device, non_blocking=True).float()

    with timer.stage("basic", sync_cuda=is_cuda):
        rec["base"] = basic_stats(t)
    if cfg.debug_timing:
        print(f"  [time] {name}  base done", flush=True)

    if t.ndim == 2:
        with timer.stage("rowcol", sync_cuda=is_cuda):
            rec["rowcol"] = rowcol_stats(t)
        if cfg.debug_timing:
            print(f"  [time] {name}  rowcol done", flush=True)

        if (
            meta["role"] in SPECTRAL_ROLES
            and min(t.shape) >= 8
            and t.numel() <= SPECTRAL_NUMEL_CAP
            and cfg.topk_sigma > 0
        ):
            with timer.stage("spectral", sync_cuda=is_cuda):
                rec["spectral"] = spectral_stats(t, cfg.topk_sigma, cfg.svd_niter)
            if cfg.debug_timing:
                print(f"  [time] {name}  spectral done", flush=True)

    del t
    return rec


def run(
    reader: SafetensorsReader,
    out_path: Path | str,
    *,
    config: Config | None = None,
    names: Iterable[str] | None = None,
) -> Path:
    """Analyse every tensor (or a filtered subset) and write a JSON report."""
    cfg = config or Config()
    device = _device(cfg)
    out_path = Path(out_path)
    target_names = list(names) if names is not None else list(reader.names)
    if not target_names:
        raise RuntimeError("no tensors selected for analysis")

    timer = StageTimer()
    print(
        f"[analyze] {len(target_names)} tensor(s) on device={device}; "
        f"safetensors mmap'd from {reader.path}",
        flush=True,
    )

    stats: dict[str, dict[str, Any]] = {}
    n_errors = 0
    pbar = tqdm(total=len(target_names), desc="analyze")

    target_set = set(target_names)
    t_start = time.time()
    for name, tensor in reader.iter_tensors():
        if name not in target_set:
            continue
        try:
            with timer.stage("analyze_one_total"):
                stats[name] = _analyze_one(name, tensor, device, cfg, timer)
        except Exception as e:  # noqa: BLE001
            n_errors += 1
            stats[name] = {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                **classify(name),
                "error": repr(e),
            }
        del tensor
        pbar.update(1)
        if device.startswith("cuda") and pbar.n % 64 == 0:
            torch.cuda.empty_cache()

    pbar.close()
    elapsed = time.time() - t_start
    timer.print_summary()

    blob = {
        "meta": {
            "n_tensors": len(stats),
            "n_errors": n_errors,
            "elapsed_sec": elapsed,
            "device": device,
            "topk_sigma": cfg.topk_sigma,
            "svd_niter": cfg.svd_niter,
            "hf_dir": str(reader.path),
            "timing": timer.totals(),
            "hist": {
                "log10_min": HIST_LOG_MIN,
                "log10_max": HIST_LOG_MAX,
                "n_bins": HIST_BINS,
            },
            "top_k_tail": TOP_K_TAIL,
        },
        "tensors": stats,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(blob))
    print(
        f"\nWrote {out_path}  ({len(stats)} tensors, {n_errors} errors, "
        f"{elapsed:.1f}s total)"
    )
    return out_path


def run_hf(
    hf_dir: Path | str,
    out_path: Path | str,
    *,
    config: Config | None = None,
    include: str | None = None,
    exclude: str | None = None,
    limit: int | None = None,
) -> Path:
    """Convenience: open a :class:`SafetensorsReader` and run :func:`run`.

    ``include`` and ``exclude`` are regexes over tensor FQNs.
    """
    reader = SafetensorsReader(hf_dir)
    names = reader.names
    if include:
        rx = re.compile(include)
        names = [n for n in names if rx.search(n)]
    if exclude:
        rx = re.compile(exclude)
        names = [n for n in names if not rx.search(n)]
    if limit is not None:
        names = names[:limit]
    return run(reader, out_path, config=config, names=names)
