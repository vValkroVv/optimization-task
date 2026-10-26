#!/usr/bin/env python3
from __future__ import annotations
import argparse, gc, logging, re, os, hashlib, copy, shutil
from pathlib import Path
from typing import Any, Iterable, Tuple, List, Set, Dict, Optional
import torch

log = logging.getLogger("distcp_filter")

# ========================= IO helpers =========================

def find_distcp_files(base: Path) -> Iterable[Path]:
    return sorted(base.rglob("*.distcp"), key=lambda p: _nums_from_name(p.name))

_NUM_RE = re.compile(r"\d+")
_EP_RANK_RE = re.compile(r"ep_rank(\d+)", re.IGNORECASE)
_LAYER_RE = re.compile(r"\.layers\.(\d+)\.", re.IGNORECASE)

def get_ep_rank_from_path(p: Path) -> Optional[int]:
    for part in p.parts:
        m = _EP_RANK_RE.search(part)
        if m:
            return int(m.group(1))
    return None

def _nums_from_name(name: str) -> Tuple[int, ...]:
    nums = tuple(int(x) for x in _NUM_RE.findall(name))
    return nums if nums else (10**12,)

def safe_torch_load(path: Path, map_location: str = "cpu") -> Any:
    return torch.load(path, map_location=map_location, mmap=False, weights_only=False)

def safe_torch_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)

# ========================= Transforms =========================

def remove_fsdp_optimizer(obj: Any) -> Any:
    if isinstance(obj, dict):
        state = obj.get("state")
        if isinstance(state, dict) and "optimizers" in state:
            del state["optimizers"]
        if "optimizers" in obj:
            del obj["optimizers"]
    return obj

def _is_floating_tensor(t: torch.Tensor) -> bool:
    return t.is_floating_point() or t.dtype in (
        torch.float64, torch.float32, torch.float16, torch.bfloat16
    )

def cast_tensors_bf16(obj: Any) -> Any:
    if torch.is_tensor(obj):
        if _is_floating_tensor(obj) and obj.dtype != torch.bfloat16:
            return obj.to(torch.bfloat16)
        return obj
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            obj[k] = cast_tensors_bf16(v)
        return obj
    if isinstance(obj, (list, tuple)):
        t = [cast_tensors_bf16(v) for v in obj]
        return type(obj)(t) if isinstance(obj, tuple) else t
    return obj

# ---------- meta helpers ----------

def approx_bytes(x: Any) -> int:
    import io
    buf = io.BytesIO()
    try:
        torch.save(x, buf)
        return buf.tell()
    except Exception:
        return -1

def approx_total_bytes(x: Any) -> int:
    import io
    buf = io.BytesIO()
    try:
        torch.save(x, buf)
        return buf.tell()
    except Exception:
        return -1

def count_meta_keys_recursive(obj: Any, prefix: str = "") -> int:
    if prefix.startswith("state.integrations") or prefix.startswith("state.metadata"):
        return 1
    cnt = 0
    if isinstance(obj, dict):
        for k, v in obj.items():
            kp = f"{prefix}.{k}" if prefix else str(k)
            cnt += count_meta_keys_recursive(v, kp)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            kp = f"{prefix}[{i}]"
            cnt += count_meta_keys_recursive(v, kp)
    return cnt

def count_flat_meta_keys(obj: Any, prefix: str = "") -> int:
    cnt = 0
    if isinstance(obj, dict):
        for k, v in obj.items():
            kp = f"{prefix}.{k}" if prefix else str(k)
            if kp.startswith("state.integrations") or kp.startswith("state.metadata"):
                cnt += 1
            cnt += count_flat_meta_keys(v, kp)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            kp = f"{prefix}[{i}]"
            cnt += count_flat_meta_keys(v, kp)
    return cnt

def extract_meta_subtree(obj: Any, prefix: str = "") -> Dict[str, Any] | None:
    if prefix.startswith("state.integrations") or prefix.startswith("state.metadata"):
        return obj
    if isinstance(obj, dict):
        acc: Dict[str, Any] = {}
        for k, v in obj.items():
            kp = f"{prefix}.{k}" if prefix else str(k)
            sub = extract_meta_subtree(v, kp)
            if sub is not None:
                acc[k] = sub
        return acc or None
    if isinstance(obj, (list, tuple)):
        acc_list = []
        for i, v in enumerate(obj):
            kp = f"{prefix}[{i}]"
            sub = extract_meta_subtree(v, kp)
            if sub is not None:
                acc_list.append(sub)
        if isinstance(obj, tuple):
            acc_list = tuple(acc_list)
        return acc_list or None
    return None

# ---------- namespace normalization ----------

def _pop_flat_namespace(sd: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    out = {}
    pref = prefix if prefix.endswith('.') else prefix + '.'
    for k in list(sd.keys()):
        if isinstance(k, str) and k.startswith(pref):
            out[k[len(pref):]] = sd.pop(k)
    return out

def normalize_nested_state_model(obj: Any, keep_meta_here: bool) -> Any:
    if not isinstance(obj, dict):
        return obj

    sd = obj
    if 'state' not in sd or not isinstance(sd['state'], dict):
        sd['state'] = {}
    state = sd['state']

    if 'model' not in state or not isinstance(state['model'], dict):
        state['model'] = {}
    model_dict = state['model']

    flat_model = _pop_flat_namespace(sd, 'state.model')
    for name, val in flat_model.items():
        if name not in model_dict:
            model_dict[name] = val

    nested_flat_model = _pop_flat_namespace(state, 'model')
    for name, val in nested_flat_model.items():
        if name not in model_dict:
            model_dict[name] = val

    if not keep_meta_here:
        state.pop('metadata', None)
        state.pop('integrations', None)
        for k in list(sd.keys()):
            if isinstance(k, str) and (
                k.startswith('state.metadata.') or k.startswith('state.integrations.')
            ):
                sd.pop(k, None)

    return sd

# ---------- EMA / NON-EMA pairing filter ----------

def drop_non_ema_weights(obj: Any, ema_regex: re.Pattern, drop_regex: re.Pattern) -> Any:
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            try:
                v = obj[k]
            except KeyError:
                continue
            k_lower = str(k).lower()
            if drop_regex.search(k_lower) and not ema_regex.search(k_lower):
                del obj[k]
                continue
            obj[k] = drop_non_ema_weights(v, ema_regex, drop_regex)
        return obj
    if isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = drop_non_ema_weights(obj[i], ema_regex, drop_regex)
        return obj
    if isinstance(obj, tuple):
        return tuple(drop_non_ema_weights(v, ema_regex, drop_regex) for v in obj)
    return obj

# ---------- точечные префикс-дропы ----------

_TOKENIZER_MERGES_PREFIX = "state.integrations.huggingface.tokenizer.tokenizer.json.content.model.merges"
_TOKENIZER_VOCAB_PREFIX  = "state.integrations.huggingface.tokenizer.tokenizer.json.content.model.vocab"

def _prefix_matches(prefix: str, target: str) -> bool:
    return prefix == target or prefix.startswith(target + ".") or prefix.startswith(target + "[")

def drop_by_prefixes(obj: Any, prefixes: tuple[str, ...], prefix: str = "") -> Any | None:
    for pfx in prefixes:
        if _prefix_matches(prefix, pfx):
            return None
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            kp = f"{prefix}.{k}" if prefix else str(k)
            kept = drop_by_prefixes(v, prefixes, kp)
            if kept is not None:
                out[k] = kept
        return out
    if isinstance(obj, list):
        out_l = []
        for i, v in enumerate(obj):
            kp = f"{prefix}[{i}]"
            kept = drop_by_prefixes(v, prefixes, kp)
            if kept is not None:
                out_l.append(kept)
        return out_l
    if isinstance(obj, tuple):
        out_t = []
        for i, v in enumerate(obj):
            kp = f"{prefix}[{i}]"
            kept = drop_by_prefixes(v, prefixes, kp)
            if kept is not None:
                out_t.append(kept)
        return tuple(out_t)
    return obj

def count_nodes_by_prefixes(obj: Any, prefixes: tuple[str, ...], prefix: str = "") -> int:
    for pfx in prefixes:
        if _prefix_matches(prefix, pfx):
            return 1
    cnt = 0
    if isinstance(obj, dict):
        for k, v in obj.items():
            kp = f"{prefix}.{k}" if prefix else str(k)
            cnt += count_nodes_by_prefixes(v, prefixes, kp)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            kp = f"{prefix}[{i}]"
            cnt += count_nodes_by_prefixes(v, prefixes, kp)
    return cnt

# ========================= Layer-parity split (ema_1 / ema_2) =========================

def _route_by_layer_parity(path_low: str) -> Optional[int]:
    m = _LAYER_RE.search(path_low)
    if m:
        return int(m.group(1)) % 2
    return None

def _split_by_parity(obj: Any,
                     drop_regex: re.Pattern,
                     prefix: str = "") -> tuple[Any | None, Any | None]:
    # dict
    if isinstance(obj, dict):
        out0, out1 = {}, {}
        for k, v in obj.items():
            kp = f"{prefix}.{k}" if prefix else str(k)
            b0, b1 = _split_by_parity(v, drop_regex, kp)
            if b0 is not None:
                out0[k] = b0
            if b1 is not None:
                out1[k] = b1
        if not out0 and not out1:
            return None, None
        return out0, out1

    # list
    if isinstance(obj, list):
        out0_list, out1_list = [], []
        any0 = any1 = False
        for i, v in enumerate(obj):
            kp = f"{prefix}[{i}]"
            b0, b1 = _split_by_parity(v, drop_regex, kp)
            out0_list.append(b0)
            out1_list.append(b1)
            any0 |= (b0 is not None)
            any1 |= (b1 is not None)
        if not any0 and not any1:
            return None, None
        return out0_list if any0 else None, out1_list if any1 else None

    # tuple
    if isinstance(obj, tuple):
        tmp0, tmp1 = [], []
        any0 = any1 = False
        for i, v in enumerate(obj):
            kp = f"{prefix}[{i}]"
            b0, b1 = _split_by_parity(v, drop_regex, kp)
            tmp0.append(b0); any0 |= (b0 is not None)
            tmp1.append(b1); any1 |= (b1 is not None)
        if not any0 and not any1:
            return None, None
        return (tuple(tmp0) if any0 else None,
                tuple(tmp1) if any1 else None)

    # leaf
    path_low = prefix.lower()
    is_modelish = bool(drop_regex.search(path_low))
    if is_modelish:
        route = _route_by_layer_parity(path_low)
        if route is None:
            return obj, obj
        return (obj, None) if route == 0 else (None, obj)
    else:
        return obj, obj

# ========================= Diagnostics =========================

def _iter_key_paths(obj: Any, prefix: str = "") -> Iterable[str]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            kp = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, (dict, list, tuple)):
                yield from _iter_key_paths(v, kp)
            else:
                yield kp
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            kp = f"{prefix}[{i}]"
            if isinstance(v, (dict, list, tuple)):
                yield from _iter_key_paths(v, kp)
            else:
                yield kp
    else:
        if prefix:
            yield prefix

def _normalize_path(path_low: str, ema_regex: re.Pattern) -> str:
    norm = ema_regex.sub("", path_low)
    return re.sub(r"[._-]{2,}", ".", norm)

def _classify_keys_for_print(obj: Any, ema_regex: re.Pattern, drop_regex: re.Pattern) -> Tuple[Set[str], Set[str], Dict[str, str]]:
    ema_paths: Set[str] = set()
    nonema_paths: Set[str] = set()
    norm_map: Dict[str, str] = {}
    for path in _iter_key_paths(obj):
        path_low = path.lower()
        is_ema = bool(ema_regex.search(path_low))
        is_drop_candidate = bool(drop_regex.search(path_low))
        if is_ema:
            ema_paths.add(path)
        elif is_drop_candidate:
            nonema_paths.add(path)
        norm_map[path] = _normalize_path(path_low, ema_regex)
    return ema_paths, nonema_paths, norm_map

def _print_ema_report(src_path: Path, ema_paths: Set[str], nonema_paths: Set[str], norm_map: Dict[str, str], max_items: int) -> None:
    ema_norms = {norm_map[p] for p in ema_paths}
    nonema_norms = {norm_map[p] for p in nonema_paths}
    paired_norms = ema_norms & nonema_norms
    unpaired_ema = [p for p in ema_paths if norm_map[p] not in paired_norms]
    unpaired_nonema = [p for p in nonema_paths if norm_map[p] not in paired_norms]

    log.info(f"[{src_path.name}] EMA/non-EMA key scan:")
    log.info(f"  EMA keys: {len(ema_paths)} | non-EMA candidates: {len(nonema_paths)}")
    log.info(f"  Pairs by normalized path: {len(paired_norms)} | unpaired EMA: {len(unpaired_ema)} | unpaired non-EMA: {len(unpaired_nonema)}")

    if ema_paths:
        sample = sorted(ema_paths)[:max_items]
        log.info("  Sample EMA keys:")
        for s in sample:
            log.info(f"    EMA: {s}")
    if nonema_paths:
        sample = sorted(nonema_paths)[:max_items]
        log.info("  Sample non-EMA keys:")
        for s in sample:
            log.info(f"    NONEMA: {s}")

    if paired_norms:
        shown = 0
        log.info("  Sample EMA↔NONEMA pairs:")
        for ema_p in ema_paths:
            n = norm_map[ema_p]
            mate = next((p for p in nonema_paths if norm_map[p] == n), None)
            if mate:
                log.info(f"    PAIR: [{ema_p}] <-> [{mate}]")
                shown += 1
            if shown >= max(1, max_items // 4):
                break

def _filter_by_pairs(obj: Any,
                     ema_regex: re.Pattern,
                     drop_regex: re.Pattern,
                     paired_norms: Set[str],
                     action: str,
                     prefix: str = "") -> Any:
    """
    action:
      - "keep-ema"    -> удалить non-EMA из пар (оставить EMA)
      - "keep-nonema" -> удалить EMA из пар (оставить non-EMA)
      - "drop-both"   -> удалить и EMA, и non-EMA из пар
      - "none"        -> ничего не делать
    """
    if action == "none" or not paired_norms:
        return obj

    if isinstance(obj, dict):
        for k in list(obj.keys()):
            kp = f"{prefix}.{k}" if prefix else str(k)
            kp_low = kp.lower()

            is_drop_cand = bool(drop_regex.search(kp_low))
            if not is_drop_cand:
                obj[k] = _filter_by_pairs(obj[k], ema_regex, drop_regex, paired_norms, action, kp)
                continue

            is_ema = bool(ema_regex.search(kp_low))
            norm   = _normalize_path(kp_low, ema_regex)

            if norm in paired_norms:
                drop_it = (
                    (action == "keep-ema" and not is_ema) or
                    (action == "keep-nonema" and is_ema) or
                    (action == "drop-both")
                )
                if drop_it:
                    del obj[k]
                    continue

            obj[k] = _filter_by_pairs(obj[k], ema_regex, drop_regex, paired_norms, action, kp)
        return obj

    if isinstance(obj, list):
        for i in range(len(obj)):
            kp = f"{prefix}[{i}]"
            obj[i] = _filter_by_pairs(obj[i], ema_regex, drop_regex, paired_norms, action, kp)
        return obj

    if isinstance(obj, tuple):
        return tuple(_filter_by_pairs(v, ema_regex, drop_regex, paired_norms, action, f"{prefix}[{i}]")
                     for i, v in enumerate(obj))

    return obj

# ========================= Rank helpers =========================

def is_first_file_of_rank(src_root: Path, path: Path) -> bool:
    rank = get_ep_rank_from_path(path)
    if rank is None:
        return False
    parts = list(path.parts)
    rank_idx = None
    for i, part in enumerate(parts):
        if _EP_RANK_RE.search(part):
            rank_idx = i
            break
    if rank_idx is None:
        return False
    rank_dir = Path(*parts[:rank_idx+1])
    files = sorted([p for p in rank_dir.glob("*.distcp")], key=lambda p: p.name)
    if not files:
        return False
    return path.samefile(files[0])

def copy_artifacts(src_root: Path, dst_root: Path, node_index: int) -> None:
    if node_index != 0:
        log.info(f"node-index={node_index}: skip artifacts copy (only node 0 does this)")
        return

    art_src = src_root / "artifacts"
    if not art_src.exists():
        log.info(f"No artifacts dir at {art_src}; nothing to copy")
        return

    art_dst = dst_root / "artifacts"
    log.info(f"Copying artifacts: {art_src} -> {art_dst}")
    shutil.copytree(art_src, art_dst, symlinks=True, dirs_exist_ok=True)
    log.info("Artifacts copied/merged successfully")

def copy_artifacts_to_halves(src_root: Path, dst_root: Path, node_index: int) -> None:
    if node_index != 0:
        log.info(f"node-index={node_index}: skip half-artifacts copy (only node 0 does this)")
        return
    art_src = src_root / "artifacts"
    if not art_src.exists():
        log.info(f"No artifacts dir at {art_src}; nothing to copy to halves")
        return
    for sub in ("ema_1", "ema_2"):
        art_dst = dst_root / sub / "artifacts"
        log.info(f"Copying artifacts to half: {art_src} -> {art_dst}")
        shutil.copytree(art_src, art_dst, symlinks=True, dirs_exist_ok=True)
    log.info("Artifacts copied to both halves")

# ========================= Main =========================

def _log_meta_snapshot(tag: str, obj: Any) -> Dict[str, int]:
    total = approx_total_bytes(obj)
    subtree = extract_meta_subtree(obj)
    meta_bytes = approx_bytes(subtree) if subtree is not None else 0
    meta_trees = count_meta_keys_recursive(obj)
    meta_flat  = count_flat_meta_keys(obj)
    log.info(f"[{tag}] total_bytes={total} | meta_bytes={meta_bytes} | meta_trees={meta_trees} | meta_flat_keys={meta_flat}")
    return {"total": total, "meta": meta_bytes, "trees": meta_trees, "flat": meta_flat}

def process_file(src_path: Path, dst_root: Path, src_root: Path,
                 ema_regex: re.Pattern, drop_regex: re.Pattern,
                 cast_bf16: bool, map_location: str = "cpu", dry_run: bool = False,
                 print_keys: bool = False, print_max: int = 50,
                 pair_action: str = "none", keep_meta_here: bool = False,
                 drop_tokenizer_merges: bool = False,
                 drop_tokenizer_vocab: bool = False,
                 split_into_two: bool = False) -> None:

    rel = src_path.relative_to(src_root)
    dst_path = dst_root / rel
    log.info(f"Processing {src_path} -> {dst_path} | keep_meta_here={keep_meta_here}")

    obj = safe_torch_load(src_path, map_location=map_location)

    snap_before = _log_meta_snapshot("before_any_drop", obj)

    # 0) optional tokenizer drops
    if drop_tokenizer_merges:
        before_nodes = count_nodes_by_prefixes(obj, (_TOKENIZER_MERGES_PREFIX,))
        if before_nodes:
            obj = drop_by_prefixes(obj, (_TOKENIZER_MERGES_PREFIX,))
            log.info(f"[tokenizer.merges] dropped subtree x{before_nodes}")
        else:
            log.info(f"[tokenizer.merges] nothing to drop")
        _log_meta_snapshot("after_drop_tokenizer_merges", obj)

    if drop_tokenizer_vocab:
        before_nodes = count_nodes_by_prefixes(obj, (_TOKENIZER_VOCAB_PREFIX,))
        if before_nodes:
            obj = drop_by_prefixes(obj, (_TOKENIZER_VOCAB_PREFIX,))
            log.info(f"[tokenizer.vocab] dropped subtree x{before_nodes}")
        else:
            log.info(f"[tokenizer.vocab] nothing to drop")
        _log_meta_snapshot("after_drop_tokenizer_vocab", obj)

    # 1) normalize + drop optimizer
    obj = normalize_nested_state_model(obj, keep_meta_here=keep_meta_here)
    obj = remove_fsdp_optimizer(obj)

    # 2) pairing analysis (for pair_action filtering)
    paired_norms: Set[str] = set()
    ema_paths: Set[str] = set()
    nonema_paths: Set[str] = set()
    try:
        ema_paths, nonema_paths, norm_map = _classify_keys_for_print(obj, ema_regex, drop_regex)
        if print_keys:
            _print_ema_report(src_path, ema_paths, nonema_paths, norm_map, max_items=print_max)
        ema_norms = {norm_map[p] for p in ema_paths}
        nonema_norms = {norm_map[p] for p in nonema_paths}
        paired_norms = ema_norms & nonema_norms
    except Exception as e:
        log.warning(f"Pairing analysis failed for {src_path}: {e}")
        paired_norms = set()

    # 3) apply pair_action FIRST (как ты просил)
    if pair_action != "none" and paired_norms:
        obj = _filter_by_pairs(obj, ema_regex, drop_regex, paired_norms, action=pair_action)

    # 4) cast
    if cast_bf16:
        obj = cast_tensors_bf16(obj)

    snap_final = _log_meta_snapshot("final_before_save", obj)
    if snap_before["total"] >= 0 and snap_final["total"] >= 0:
        log.info(
            f"[summary] total_bytes Δ = {snap_final['total'] - snap_before['total']} "
            f"(from {snap_before['total']} to {snap_final['total']}); "
            f"meta_bytes Δ = {snap_final['meta'] - snap_before['meta']} "
            f"(from {snap_before['meta']} to {snap_final['meta']}); "
            f"meta_flat_keys Δ = {snap_final['flat'] - snap_before['flat']} "
            f"(from {snap_before['flat']} to {snap_final['flat']})"
        )

    if dry_run:
        log.info("Dry-run: skipping save")
        del obj; gc.collect()
        return

    # 5) save
    if not split_into_two:
        safe_torch_save(obj, dst_path)
        del obj; gc.collect()
        return

    b0, b1 = _split_by_parity(obj, drop_regex, prefix="")
    del obj; gc.collect()

    dst1 = (dst_root / "ema_1" / rel.parent).resolve()
    dst2 = (dst_root / "ema_2" / rel.parent).resolve()
    dst1.mkdir(parents=True, exist_ok=True)
    dst2.mkdir(parents=True, exist_ok=True)

    if b0 is None: b0 = {}
    if b1 is None: b1 = {}

    safe_torch_save(b0, dst1 / rel.name)
    safe_torch_save(b1, dst2 / rel.name)
    del b0, b1
    gc.collect()

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Filter .distcp shards: apply EMA/non-EMA pairing filter, cast bf16, remove optimizer, optional split by layer parity into two folders"
    )
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--dst", type=Path, required=True)
    p.add_argument("--map-location", default="cpu", choices=["cpu"])
    p.add_argument("--ema-regex", default=r"(^|[._-])ema($|[._-])")
    p.add_argument("--drop-regex", default=r"(^|[._-])(model|module|weights|params)($|[._-])")
    p.add_argument("--cast-bf16", action="store_true")
    p.add_argument("--pair-action", choices=["none", "keep-ema", "keep-nonema", "drop-both"],
                   default="none",
                   help="How to handle EMA/non-EMA pairs matched by normalized path (applied BEFORE split).")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--print-keys", action="store_true")
    p.add_argument("--print-max", type=int, default=50)
    p.add_argument("--loglevel", default="INFO")
    p.add_argument("--node-index", type=int, required=True)
    p.add_argument("--num-nodes", type=int, required=True)
    p.add_argument("--drop-tokenizer-merges", action="store_true",
                   help="Drop HF tokenizer merges array under state.integrations...model.merges")
    p.add_argument("--drop-tokenizer-vocab", action="store_true",
                   help="Drop HF tokenizer vocab map under state.integrations...model.vocab")
    p.add_argument("--split-into-two", action="store_true",
                   help="After filtering, split by LAYER parity: even -> ema_1/, odd -> ema_2/. Non-layer keys are duplicated.")
    return p

def main() -> None:
    args = build_argparser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.loglevel.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    log.info(os.environ)

    src_root: Path = args.src.resolve()
    dst_root: Path = args.dst.resolve()
    if not src_root.exists():
        raise FileNotFoundError(f"Source path does not exist: {src_root}")

    ema_regex = re.compile(args.ema_regex, flags=re.IGNORECASE)
    drop_regex = re.compile(args.drop_regex, flags=re.IGNORECASE)

    files = list(find_distcp_files(src_root))
    if not files:
        log.warning(f"No .distcp files found under {src_root}")
        return

    node_index = args.node_index
    num_nodes  = args.num_nodes
    assert num_nodes >= 1
    assert 0 <= node_index < num_nodes
    log.info(f"node-index={node_index} num-nodes={num_nodes}")

    try:
        if args.split_into_two:
            copy_artifacts_to_halves(src_root, dst_root, node_index)
        else:
            copy_artifacts(src_root, dst_root, node_index)
    except Exception as e:
        log.exception(f"Artifacts copy failed: {e}")

    selected: List[Path] = []
    for f in files:
        ep_rank = get_ep_rank_from_path(f)
        if ep_rank is None:
            if node_index == 0:
                selected.append(f)
        else:
            if (ep_rank % num_nodes) == node_index:
                selected.append(f)

    log.info(f"Found {len(files)} .distcp files | selected for node {node_index}/{num_nodes}: {len(selected)}")

    for i, f in enumerate(selected, 1):
        keep_meta_here = (get_ep_rank_from_path(f) == 0) and is_first_file_of_rank(src_root, f)
        try:
            process_file(
                src_path=f,
                dst_root=dst_root,
                src_root=src_root,
                ema_regex=ema_regex,
                drop_regex=drop_regex,
                cast_bf16=args.cast_bf16,
                map_location=args.map_location,
                dry_run=args.dry_run,
                print_keys=args.print_keys,
                print_max=args.print_max,
                pair_action=args.pair_action,
                keep_meta_here=keep_meta_here,
                drop_tokenizer_merges=args.drop_tokenizer_merges,
                drop_tokenizer_vocab=args.drop_tokenizer_vocab,
                split_into_two=args.split_into_two,
            )
        except Exception as e:
            log.exception(f"Failed to process {f}: {e}")
        finally:
            if i % 4 == 0:
                gc.collect()

if __name__ == "__main__":
    main()
