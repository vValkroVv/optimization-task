"""Targeted DeepGEMM JIT warmup at training start.
"""

import inspect
import math
import os
import shutil
import time
import uuid
import warnings

import torch
from composer.utils import dist

from llmfoundry.models.ops.warmup.cache_manager import _deepgemm_cache_manager

_ALIGN = 128
_BLOCK_K = 128
_SMEM_CAPACITY = 232448

_MAX_M_TOKENS = 1 << 21
_MAX_K_TOKENS = 1 << 18

def _rank_cache_dir(local_rank: int) -> str:
    # Same layout as cache_manager._deepgemm_cache_manager.
    return f"/temp/.cache/rank_{local_rank}/deepgemm/"


# ---------------------------------------------------------------------------
# Port of SM90 get_best_config (FP8, K-major A/B, BF16 output)
# ---------------------------------------------------------------------------

def _sm90_config(k_grouped: bool, m: int, n: int, k: int, num_groups: int, num_sms: int) -> tuple:
    """Template-relevant config of an m-grouped contiguous 1D2D GEMM, or of a
    k-grouped contiguous 1D1D GEMM when ``k_grouped`` (then ``k`` is the max k)."""
    cdiv = lambda a, b: (a + b - 1) // b
    align = lambda a, b: cdiv(a, b) * b

    if k_grouped:
        block_ms = [64, 128]  # 256 is illegal for 1D1D
        block_ns = [16] + list(range(24, 153, 16))  # 1D1D: block_n <= 152
    else:
        block_ms = [_ALIGN]  # MGroupedContiguous uses the m alignment
        block_ns = [bn for bn in range(16, 257, 16) if bn <= 128 or bn in (144, 160, 192)]

    # Block sizes: fewest waves, then best last-wave utilization
    best = None
    for bm in block_ms:
        for bn in block_ns:
            num_blocks = cdiv(m, bm) * cdiv(n, bn) * num_groups
            waves = cdiv(num_blocks, num_sms)
            last = num_blocks % num_sms or num_sms
            if best is None or waves < best[2]:
                better = True
            elif waves == best[2]:
                better = last > best[3]
                if last == best[3]:
                    better |= bm == best[0] and bn < best[1]
                    better |= bn == best[1] and bm < best[0]
                    better |= bm != best[0] and bn > best[1] and bn <= n and bm <= m
            else:
                better = False
            if better:
                best = (bm, bn, waves, last)
    bm, bn = best[:2]

    # TMA multicast: disabled for k-grouped with > 4 groups, needs an even SM count
    multicast, on_a = 1, False
    if not (k_grouped and num_groups > 4) and m >= 512 and num_sms % 2 == 0:
        multicast, on_a = 2, bm > bn

    # Largest number of stages that fits into shared memory
    smem_cd = align(bm * bn * 2, 1024)
    smem_per_stage = bm * _BLOCK_K + bn * _BLOCK_K + align(bm * 4, 128)
    if k_grouped:
        smem_per_stage += align(bn * 4, 128)
    smem_extra = smem_cd + align(cdiv(k, _BLOCK_K) * 4 * (1 if _BLOCK_K % bn == 0 else 2), 8)
    if k_grouped:
        smem_extra += 4 * 128  # tensor maps
    max_stages = 4 if _BLOCK_K % bn != 0 and _BLOCK_K // math.gcd(bn, _BLOCK_K) <= 4 else 32
    stages = next(s for s in range(max_stages, 0, -1)
                  if smem_extra + s * (smem_per_stage + 16) <= _SMEM_CAPACITY)

    config = (bm, bn, stages, multicast, on_a)
    if not k_grouped:
        config += (cdiv(k, _BLOCK_K) % stages,)  # num_last_stages is a 1D2D template argument
    return config


def _plan(hidden: int, intermediate: int, num_groups: int, num_sms: int) -> list[tuple]:
    """One representative ``(gemm name, call, config)`` per distinct kernel of
    the MoE grouped GEMMs."""
    h, i2, i = hidden, 2 * intermediate, intermediate
    calls = {}
    # m-grouped fwd and dgrad: (n, k, dgrad); dgrad quantizes W with a transpose
    for name, n, k, dgrad in (("w1_fwd", i2, h, False), ("w1_dgrad", h, i2, True),
                              ("w2_fwd", h, i, False), ("w2_dgrad", i, h, True)):
        for m in range(_ALIGN, _MAX_M_TOKENS + 1, _ALIGN):
            # The m-grouped heuristic is called with num_groups=1 (sm90_fp8_gemm_1d2d.hpp)
            config = _sm90_config(False, m, n, k, 1, num_sms)
            calls.setdefault(("m_grouped", n, k, dgrad, config), (name, ("m_grouped", m, n, k, dgrad), config))
    # k-grouped wgrad of shape (G, m, n), the config depends only on the max k
    for name, m, n in (("w1_wgrad", i2, h), ("w2_wgrad", h, i)):
        for max_k in range(_ALIGN, _MAX_K_TOKENS + 1, _ALIGN):
            config = _sm90_config(True, m, n, max_k, num_groups, num_sms)
            calls.setdefault(("k_grouped", m, n, config), (name, ("k_grouped", m, n, max_k), config))
    return sorted(calls.values())


def _format_config(config: tuple) -> str:
    bm, bn, stages, multicast, on_a = config[:5]
    text = f"block={bm}x{bn} stages={stages}"
    if multicast > 1:
        text += f" multicast=2{'A' if on_a else 'B'}"
    return text


# ---------------------------------------------------------------------------
# DeepGEMM calls, mirroring llmfoundry.models.ops.float8.grouped_gemm
# ---------------------------------------------------------------------------

@torch.no_grad()
def _run(call: tuple, num_groups: int, handle, device: torch.device) -> None:
    from llmfoundry.models.ops.float8.triton_kernels import (
        block_quant_fn,
        block_quant_trans_fn,
        row2col_deep_gemm_fn,
        row_quant_fn,
    )
    from llmfoundry.models.ops.float8.triton_kernels.utils import build_m_indices

    fp8, bf16, g = torch.float8_e4m3fn, torch.bfloat16, num_groups

    if call[0] == "m_grouped":
        _, m, n, k, dgrad = call
        a = row_quant_fn(torch.randn(m, k, dtype=bf16, device=device))
        a = (a[0].view(fp8).contiguous(), a[1])
        if dgrad:
            w = block_quant_trans_fn(torch.randn(g, k, n, dtype=bf16, device=device))
        else:
            w = block_quant_fn(torch.randn(g, n, k, dtype=bf16, device=device))
        w = (w[0].reshape(g, -1, w[0].size(-1)), w[1].reshape(g, -1, w[1].size(-1)))
        out = torch.empty(m, n, dtype=bf16, device=device)
        m_indices = torch.zeros(m, dtype=torch.int32, device=device)
        handle.m_grouped_fp8_gemm_nt_contiguous(a, w, out, m_indices)
    else:
        _, m, n, max_k = call
        ks = [max_k] + [_ALIGN] * (g - 1)
        ks_tensor = torch.tensor(ks, dtype=torch.int32, device=device)
        m_indices = build_m_indices(ks, ks_tensor)
        cols = []
        for dim in (m, n):  # grad_output, then input
            q = row_quant_fn(torch.randn(sum(ks), dim, dtype=bf16, device=device))
            q = row2col_deep_gemm_fn(q[0].view(fp8).contiguous(), m_indices, ks_tensor, q[1])
            cols.append((q[0].view(fp8), q[1].mT))
        wgrad = torch.zeros(g, m, n, dtype=bf16, device=device)
        handle.k_grouped_fp8_gemm_contiguous(a=cols[0], b=cols[1], d=wgrad, ks=ks, ks_tensor=ks_tensor)
    torch.cuda.synchronize(device)


def _list_kernel_dirs(root: str) -> set[str]:
    return {name for name in os.listdir(root) if name.startswith("kernel.")} if os.path.isdir(root) else set()


def _kernel_dirs(local_rank: int) -> set[str]:
    return _list_kernel_dirs(os.path.join(_rank_cache_dir(local_rank), "cache"))


def _shared_kernel_dirs() -> set[str]:
    shared = os.environ.get("DG_JIT_SHARED_CACHE_DIR")
    return _list_kernel_dirs(os.path.join(shared, "cache")) if shared else set()


def _log_summary(records: list[dict], planned: int, header: str, cache_size: int, elapsed: float) -> None:
    """Print the warmup results gathered from all ranks (on global rank 0)."""
    print(f"[DeepGEMM warmup] {header}")
    print(f"[DeepGEMM warmup] {planned} kernel(s) planned per node; node 0:")
    for r in sorted((r for r in records if r["node"] == 0), key=lambda r: (r["name"], r["call"])):
        print(f"[DeepGEMM warmup]   {r['name']:9s} {r['status']:8s} {r['seconds']:6.1f}s  "
              f"local_rank={r['local_rank']}  {_format_config(r['config'])}  call={r['call']}")
    for node in sorted({r["node"] for r in records}):
        node_records = [r for r in records if r["node"] == node]
        counts = {s: sum(r["status"] == s for r in node_records) for s in ("compiled", "shared", "cached")}
        print(f"[DeepGEMM warmup] node {node}: " + " ".join(f"{s}={n}" for s, n in counts.items())
              + f" slowest={max((r['seconds'] for r in node_records), default=0.0):.1f}s")

    # The shared cache is a DeepGEMM fork feature: a build without it silently
    # ignores DG_JIT_SHARED_CACHE_DIR and never publishes compiled kernels.
    unpublished = [r for r in records if r["status"] == "compiled" and not r["published"]]
    if os.environ.get("DG_JIT_SHARED_CACHE_DIR") and unpublished:
        warnings.warn(
            f"[DeepGEMM warmup] {len(unpublished)} compiled kernel(s) did not reach "
            f"DG_JIT_SHARED_CACHE_DIR={os.environ['DG_JIT_SHARED_CACHE_DIR']}: either this DeepGEMM "
            f"build does not support the shared cache (deep_gemm.__commit__={_deep_gemm_commit()}) "
            f"or the directory is not writable"
        )

    on_miss = "compile" if os.environ.get("DG_CACHE_FORCE_SKIP") == "1" else "fail (set DG_CACHE_FORCE_SKIP=1 to compile)"
    print(f"[DeepGEMM warmup] done in {elapsed:.1f}s, rank 0 cache now has {cache_size} kernel(s); "
          f"cache misses during training will {on_miss}")


def _deep_gemm_commit() -> str:
    import deep_gemm
    return getattr(deep_gemm, "__commit__", "unknown")


def _share_node_kernels(local_rank: int, local_world_size: int) -> int:
    """Copy kernels compiled by the other local ranks into this rank's cache."""
    mine = os.path.join(_rank_cache_dir(local_rank), "cache")
    copied = 0
    for r in range(local_world_size):
        src_root = os.path.join(_rank_cache_dir(r), "cache")
        if r == local_rank or not os.path.isdir(src_root):
            continue
        for name in os.listdir(src_root):
            dst = os.path.join(mine, name)
            if not name.startswith("kernel.") or os.path.exists(dst):
                continue
            # DeepGEMM treats a directory with kernel.cu and kernel.cubin as
            # valid, so publish it with a single rename.
            tmp = os.path.join(mine, f".tmp.{name}.{uuid.uuid4().hex}")
            shutil.copytree(os.path.join(src_root, name), tmp)
            os.rename(tmp, dst)
            copied += 1
    return copied


def _model_fields(llm_cfg, keys: tuple) -> dict:
    """Model config fields from the overrides, else the GigaMixConfig defaults."""
    from llmfoundry.models.giga_mix.configuration_giga_mix import GigaMixConfig

    defaults = inspect.signature(GigaMixConfig.__init__).parameters
    fields = {}
    for key in keys:
        value = llm_cfg[key] if key in llm_cfg else getattr(defaults.get(key), "default", None)
        assert value is not None and value is not inspect.Parameter.empty, (
            f"deep_gemm_warmup_on_start needs `{key}` in the model config overrides"
        )
        fields[key] = value
    return fields


def _warmup_node(llm_cfg, ep_size: int, local_rank: int) -> None:
    from llmfoundry.models.ops.float8.grouped_gemm import _get_deep_gemm_handle

    fields = _model_fields(
        llm_cfg, ("hidden_size", "intermediate_size", "num_routed_experts", "float8_deep_gemm_num_sms"),
    )
    hidden, intermediate = fields["hidden_size"], fields["intermediate_size"]
    num_groups = fields["num_routed_experts"] // ep_size

    # Same handle (and set_num_sms) as the model's GroupedGemmFp8Wrapper
    handle = _get_deep_gemm_handle(num_sms=fields["float8_deep_gemm_num_sms"])
    import deep_gemm
    num_sms = deep_gemm.get_num_sms()
    # _sm90_config does not model the per-shape SM count this mode picks
    assert os.environ.get("DG_JIT_MINIMIZE_NUM_SMS", "0") in ("", "0"), (
        "deep_gemm_warmup_on_start does not support DG_JIT_MINIMIZE_NUM_SMS"
    )

    t0 = time.monotonic()
    calls = _plan(hidden, intermediate, num_groups, num_sms)
    local_world_size = dist.get_local_world_size()
    mine = calls[local_rank::local_world_size]

    # Wait for the cache manager's propagation before writing into the caches
    dist.barrier()
    device = torch.device("cuda", torch.cuda.current_device())
    records = []
    warmup_env = {"DG_CACHE_FORCE_SKIP": "1", "DG_SILENT_CACHE_MISS": "1"}
    # After the warmup a missed kernel compiles (and is reported) instead of
    # failing the run, unless DG_CACHE_FORCE_SKIP is set explicitly
    prev = {key: os.environ.get(key) for key in warmup_env}
    prev["DG_CACHE_FORCE_SKIP"] = prev["DG_CACHE_FORCE_SKIP"] or "1"
    os.environ.update(warmup_env)
    try:
        shared_before = _shared_kernel_dirs()
        for name, call, config in mine:
            before = _kernel_dirs(local_rank)
            t_call = time.monotonic()
            _run(call, num_groups, handle, device)
            seconds = time.monotonic() - t_call
            new = _kernel_dirs(local_rank) - before
            if not new:
                status = "cached"
            elif new <= shared_before:
                status = "shared"  # copied from DG_JIT_SHARED_CACHE_DIR
            else:
                status = "compiled"
            records.append(dict(
                name=name, call=call, config=config,
                node=dist.get_node_rank(), local_rank=local_rank,
                status=status, seconds=seconds,
                published=status == "compiled" and new <= _shared_kernel_dirs(),
            ))
    finally:
        for key, value in prev.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    torch.cuda.empty_cache()

    dist.barrier()
    _share_node_kernels(local_rank, local_world_size)
    dist.barrier()

    all_records = [r for rank_records in dist.all_gather_object(records) for r in rank_records]
    if dist.get_global_rank() == 0:
        _log_summary(
            all_records, len(calls),
            header=f"hidden={hidden} intermediate={intermediate} groups={num_groups} num_sms={num_sms}",
            cache_size=len(_kernel_dirs(local_rank)), elapsed=time.monotonic() - t0,
        )


def _prepare_shared_cache() -> None:
    shared = os.environ.get("DG_JIT_SHARED_CACHE_DIR")
    if not shared:
        return
    for path in (shared, os.path.join(shared, "cache")):
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
            try:
                os.chmod(path, 0o777)
            except OSError as e:
                print(f"Failed to set mode 777 on {path}: {e}")


def setup_deep_gemm_cache(local_rank: int, llm_cfg, ep_size: int, warmup: bool = False) -> None:
    """Set up the node-local DeepGEMM cache and, if ``warmup``, fill it with the
    MoE grouped-GEMM kernels of this model."""
    _prepare_shared_cache()
    if not warmup or os.environ.get("PROPAGATE_DG_CACHE"):
        _deepgemm_cache_manager(local_rank)
    else:
        os.makedirs(os.path.join(_rank_cache_dir(local_rank), "cache"), exist_ok=True)
        os.environ["DG_JIT_CACHE_DIR"] = _rank_cache_dir(local_rank)
    if warmup:
        _warmup_node(llm_cfg, ep_size, local_rank)
