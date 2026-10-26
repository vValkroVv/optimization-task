import argparse
import json
import hashlib
import logging
import math
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

import torch
from safetensors import safe_open

_LAYERNORMS = (
    r"(input_layernorm|post_attention_layernorm"
    r"|post_self_attn_layernorm|post_feedforward_layernorm)"
)


class CheckpointStatsAnalyzer:
    PROGRESS_EVERY = 100
    TOP_K_TAIL = 128
    SPECTRAL_MIN_DIM = 8

    LAYER_RE = re.compile(r"(?:^|\.)(?:layers|h|blocks|layer)\.(\d+)\.")
    EXPERT_RE = re.compile(r"\.experts\.(\d+)\.")
    MTP_PATH_RE = re.compile(r"(?:^|\.)mtp(?:\.|_|$)")
    SCALE_RE = re.compile(r"(?:_scale_inv|[._]scales?|\.qzeros|_zeros)$")
    MTP_MARKERS = re.compile(
        r"\.(eh_proj|enorm|hnorm|shared_head)\.|\.layers\.\d+\.embed_tokens\.weight$"
    )
    CLASS_PATTERNS: list[tuple[re.Pattern, str, str | None]] = [
        (re.compile(pattern), role, proj)
        for pattern, role, proj in (
            (
                r"(?:^|\.)model\.(?:language_model\.)?embed_tokens\.weight$",
                "embedding",
                "embed_tokens",
            ),
            (r"(^|\.)lm_head\.weight$", "lm_head", "lm_head"),
            (
                r"(?:^|\.)model\.(?:language_model\.)?norm\.weight$",
                "final_norm",
                "weight",
            ),
            (
                r"(?:^|\.)model\.(?:language_model\.)?norm\.gate_up_projection\.weight$",
                "final_norm",
                "gate_up",
            ),
            (
                r"(?:^|\.)model\.(?:language_model\.)?norm\.gate_down_projection\.weight$",
                "final_norm",
                "gate_down",
            ),
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
            (r"\.linear_attn\.(A_log|dt_bias)$", "ssm_param", "_capture"),
            (r"\.linear_attn\.(norm|o_norm)\.weight$", "ssm_param", "_capture"),
            (r"\.linear_attn\.(\w+)\.(weight|bias)$", "linear_attn", "_capture"),
            (r"\.self_attn\.qkv_proj\.weight$", "linear_attn", "qkv"),
            (r"\.self_attn\.qkv_conv1d\.weight$", "linear_attn", "qkv_conv1d"),
            (r"\.self_attn\.conv1d\.weight$", "linear_attn", "conv1d"),
            (r"\.self_attn\.gate_proj\.(weight|bias)$", "attn", "gate"),
            (r"\.self_attn\.z_proj\.weight$", "linear_attn", "z"),
            (r"\.self_attn\.in_proj_qkvz\.weight$", "linear_attn", "in_qkvz"),
            (r"\.self_attn\.in_proj_ba\.weight$", "linear_attn", "in_ba"),
            (r"\.self_attn\.ba_proj\.weight$", "linear_attn", "ba"),
            (r"\.self_attn\.o_proj\.(weight|bias)$", "attn", "o"),
            (r"\.self_attn\.out_proj\.weight$", "linear_attn", "out"),
            (r"\.self_attn\.A_log$", "ssm_param", "A_log"),
            (r"\.self_attn\.dt_bias$", "ssm_param", "dt_bias"),
            (r"\.self_attn\.norm\.weight$", "ssm_param", "norm"),
            (r"\.self_attn\.o_norm\.weight$", "ssm_param", "o_norm"),
            (r"\.mlp\.shared_expert_gate\.weight$", "moe_router", "shared_expert_gate"),
            (r"\.mlp\.gate\.weight$", "moe_router", "weight"),
            (r"\.mlp\.gate\.e_score_correction_bias$", "moe_router", "bias"),
            (r"\.mlp\.experts\.\d+\.gate_proj\.weight$", "moe_expert", "gate"),
            (r"\.mlp\.experts\.\d+\.up_proj\.weight$", "moe_expert", "up"),
            (r"\.mlp\.experts\.\d+\.down_proj\.weight$", "moe_expert", "down"),
            (
                r"\.mlp\.shared_experts?\.gate_proj\.weight$",
                "moe_shared_expert",
                "gate",
            ),
            (r"\.mlp\.shared_experts?\.up_proj\.weight$", "moe_shared_expert", "up"),
            (
                r"\.mlp\.shared_experts?\.down_proj\.weight$",
                "moe_shared_expert",
                "down",
            ),
            (r"\.mlp\.gate_proj\.weight$", "dense_mlp", "gate"),
            (r"\.mlp\.up_proj\.weight$", "dense_mlp", "up"),
            (r"\.mlp\.down_proj\.weight$", "dense_mlp", "down"),
            (r"\.self_attn\.(q|k|v)_proj\.(weight|bias)$", "attn", "_capture"),
            (r"\.(q|k)_norm\.weight$", "layernorm", "_capture_norm"),
            (r"\.indexer\.", "dsa_indexer", None),
            (
                r"\.experts\.(gate_up_proj|gate_proj|up_proj|down_proj)(\.weight)?$",
                "moe_expert",
                "_capture",
            ),
            (r"(?:^|\.)transformer\.(wte|wpe)\.weight$", "embedding", "_capture"),
            (r"\.attn\.(c_attn|c_proj)\.(weight|bias)$", "attn", "_capture"),
            (r"\.mlp\.(c_fc|c_proj)\.(weight|bias)$", "dense_mlp", "_capture"),
            (r"\.(ln_1|ln_2|ln_f)\.(weight|bias)$", "layernorm", "_capture"),
            (rf"\.{_LAYERNORMS}\.weight$", "layernorm", "_capture_weight"),
            (
                rf"\.{_LAYERNORMS}\.gate_up_projection\.weight$",
                "layernorm",
                "_capture_gate_up",
            ),
            (
                rf"\.{_LAYERNORMS}\.gate_down_projection\.weight$",
                "layernorm",
                "_capture_gate_down",
            ),
        )
    ]
    NON_SPECTRAL_ROLES = frozenset({"layernorm", "final_norm", "ssm_param"})

    def __init__(
        self, hf_checkpoint_path: Path, report_path: Path, config: argparse.Namespace
    ) -> None:
        self.hf_checkpoint_path = hf_checkpoint_path
        self.report_path = report_path
        self.autometric_runs_config = config

    def run(self) -> None:
        torch.set_num_threads(self.autometric_runs_config.num_threads)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_report(self._analyze_checkpoint())

    @property
    def partial_path(self) -> Path:
        return self.report_path.with_suffix(".jsonl.partial")

    def _analyze_checkpoint(self) -> dict:
        cfg = self.autometric_runs_config
        shard_index = self._shard_index()
        if not shard_index:
            raise FileNotFoundError(
                f"No *.safetensors tensors selected under {self.hf_checkpoint_path}"
            )
        identity = {
            "version": 3,
            "hf_dir": str(self.hf_checkpoint_path.resolve()),
            "config": {**vars(cfg), "role_overrides": list(cfg.role_overrides.items())},
            "model_config": self.model_config,
            "index": self.full_index,
            "shards": {
                shard: [path.stat().st_size, path.stat().st_mtime_ns]
                for shard in sorted(set(self.full_index.values()))
                for path in [self.hf_checkpoint_path / shard]
            },
        }
        fingerprint = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()
        completed = self._resume(fingerprint)
        logging.info(
            f"Checkpoint stats: analysing {len(shard_index)} tensors of "
            f"{self.hf_checkpoint_path} on {cfg.device}, resuming {len(completed)} records"
        )
        started = time.time()
        with self.partial_path.open("a", encoding="utf-8") as partial:
            for record, tensor in self._iter_tensors(shard_index, completed):
                try:
                    tensor = self._prepare_tensor(record, tensor)
                    record.update(self._tensor_stats(tensor, record["role"]))
                except Exception as error:  # noqa: BLE001
                    record["error"] = repr(error)
                partial.write(
                    json.dumps(self._json_safe(record), allow_nan=False) + "\n"
                )
                partial.flush()
                completed.add(record["name"])
                del tensor
                if len(completed) % self.PROGRESS_EVERY == 0:
                    logging.info(f"Checkpoint stats: {len(completed)} tensors done")
                    if cfg.device.startswith("cuda"):
                        torch.cuda.empty_cache()
        n_errors = 0
        unknown = Counter()
        unsupported = Counter()
        for record in self._partial_records():
            n_errors += "error" in record or "error" in record.get("spectral", {})
            error = record.get("error", "")
            if "unsupported quantization" in error:
                unsupported[error] += 1
            if record["role"] == "unknown":
                unknown[re.sub(r"\d+", "N", record["name"])] += 1
        if unsupported:
            logging.warning(
                f"Checkpoint stats: {sum(unsupported.values())} tensors skipped: "
                f"unsupported quantization {dict(unsupported)}"
            )
        if unknown:
            logging.warning(
                f"Checkpoint stats: unknown tensor patterns: {unknown.most_common(10)}"
            )
        return {
            "n_tensors": len(completed),
            "n_errors": n_errors,
            "n_unknown": sum(unknown.values()),
            "elapsed_sec": time.time() - started,
            "hf_dir": str(self.hf_checkpoint_path),
            "top_k_tail": self.TOP_K_TAIL,
            **vars(cfg),
        }

    def _resume(self, fingerprint: str) -> set[str]:
        header = {"fingerprint": fingerprint}
        completed: set[str] = set()
        if self.partial_path.exists():
            with self.partial_path.open("r+b") as partial:
                first = partial.readline()
                if first.endswith(b"\n") and json.loads(first) == header:
                    while True:
                        offset = partial.tell()
                        line = partial.readline()
                        if not line:
                            return completed
                        if not line.endswith(b"\n"):
                            partial.truncate(offset)
                            return completed
                        completed.add(json.loads(line)["name"])
            logging.info(
                "Checkpoint stats: checkpoint or settings changed; restarting partial report"
            )
        self.partial_path.write_text(json.dumps(header) + "\n", encoding="utf-8")
        return completed

    def _partial_records(self) -> Iterator[dict]:
        with self.partial_path.open(encoding="utf-8") as partial:
            next(partial)
            for line in partial:
                yield json.loads(line)

    def _write_report(self, meta: dict) -> None:
        tmp_path = self.report_path.with_name(self.report_path.name + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as output:
            output.write(
                '{"meta": ' + json.dumps(meta, allow_nan=False) + ', "tensors": {\n'
            )
            separator = ""
            for record in self._partial_records():
                output.write(
                    separator
                    + json.dumps(record["name"])
                    + ": "
                    + json.dumps(record, allow_nan=False)
                )
                separator = ",\n"
            output.write("\n}}\n")
        os.replace(tmp_path, self.report_path)
        self.partial_path.unlink()
        logging.info(
            f"Checkpoint stats: wrote {self.report_path} ({meta['n_tensors']} "
            f"tensors, {meta['n_errors']} errors, {meta['elapsed_sec']:.1f}s)"
        )

    def _shard_index(self) -> dict[str, str]:
        cfg = self.autometric_runs_config
        config_path = self.hf_checkpoint_path / "config.json"
        self.model_config = (
            json.loads(config_path.read_text()) if config_path.exists() else {}
        )
        index_path = self.hf_checkpoint_path / "model.safetensors.index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        else:
            index = {}
            for shard in sorted(self.hf_checkpoint_path.glob("*.safetensors")):
                with safe_open(shard, framework="pt") as f:
                    index.update({name: shard.name for name in f.keys()})
        if not index and any(self.hf_checkpoint_path.glob("pytorch_model*.bin")):
            raise ValueError(
                "checkpoint_stats requires safetensors; convert pytorch_model*.bin to safetensors first"
            )
        self.full_index = index
        self.layer_roles: dict[str, str] = {}
        self.mtp_layers: set[str] = set()
        for name in index:
            layer = self.LAYER_RE.search(name)
            if layer is None:
                continue
            prefix = name[: layer.end()]
            if re.search(r"\.self_attn\.(q_a_proj|kv_a_proj(?:_with_mqa)?)\.", name):
                self.layer_roles[prefix] = "mla"
            elif re.search(r"\.self_attn\.(in_proj_qkvz|A_log)(?:\.|$)", name):
                self.layer_roles[prefix] = "linear_attn"
            if self.MTP_MARKERS.search(name):
                self.mtp_layers.add(prefix)
        include = re.compile(cfg.include) if cfg.include else None
        exclude = re.compile(cfg.exclude) if cfg.exclude else None
        return {
            name: shard
            for name, shard in index.items()
            if not self.SCALE_RE.search(name)
            and (include is None or include.search(name))
            and (exclude is None or not exclude.search(name))
        }

    def _iter_tensors(
        self, shard_index: dict[str, str], completed: set[str]
    ) -> Iterator[tuple[dict, torch.Tensor]]:
        by_shard: dict[str, list[str]] = defaultdict(list)
        for name, shard in shard_index.items():
            by_shard[shard].append(name)
        for shard in sorted(by_shard):
            with safe_open(self.hf_checkpoint_path / shard, framework="pt") as f:
                for name in sorted(by_shard[shard]):
                    classification = self.classify(
                        name,
                        self.layer_roles,
                        self.mtp_layers,
                        self.autometric_runs_config.role_overrides,
                    )
                    shape = f.get_slice(name).get_shape()
                    fused = len(shape) == 3 and ".experts." in name
                    names = (
                        [f"{name}[expert={i}]" for i in range(shape[0])]
                        if fused
                        else [name]
                    )
                    if all(key in completed for key in names):
                        continue
                    tensor = f.get_tensor(name)
                    for i, key in enumerate(names):
                        if key in completed:
                            continue
                        weight = tensor[i] if fused else tensor
                        dtype = str(weight.dtype)
                        record = {
                            "name": key,
                            "source_name": name,
                            "shape": list(weight.shape),
                            "dtype": dtype,
                            **classification,
                        }
                        if fused:
                            record["expert"] = i
                        yield record, weight
                        del weight
                    del tensor

    def _prepare_tensor(self, record: dict, tensor: torch.Tensor) -> torch.Tensor:
        dtype = record["dtype"]
        is_fp8 = dtype.startswith("torch.float8_")
        quant = self.model_config.get("quantization_config", {})
        if not is_fp8:
            if (
                tensor.is_floating_point()
                or "quantization_config" not in self.model_config
            ):
                return tensor
        name = record["source_name"]
        qm, fmt, scale_fmt = (
            quant.get("quant_method"),
            quant.get("fmt"),
            quant.get("scale_fmt"),
        )
        if (
            not is_fp8
            or qm not in (None, "fp8", "finegrained_fp8")
            or fmt not in (None, "e4m3", "e5m2")
            or scale_fmt not in (None, "fp32", "float32")
            or name + "_scale_inv" not in self.full_index
            or not quant.get("weight_block_size")
        ):
            raise ValueError(
                f"unsupported quantization: dtype={dtype}, quant_method={qm}, "
                f"fmt={fmt}, scale_fmt={scale_fmt}; requires block FP8 with <name>_scale_inv"
            )
        expert = record["expert"] if record["name"] != name else None
        return self._dequantize(name, tensor, expert)

    def _dequantize(
        self, name: str, weight: torch.Tensor, expert: int | None
    ) -> torch.Tensor:
        scale_name = name + "_scale_inv"
        block_size = self.model_config.get("quantization_config", {}).get(
            "weight_block_size"
        )
        if (
            weight.ndim != 2
            or not block_size
            or len(block_size) != 2
            or min(block_size) <= 0
        ):
            raise ValueError(
                f"{name}: FP8 requires a 2D weight and quantization_config.weight_block_size"
            )
        with safe_open(
            self.hf_checkpoint_path / self.full_index[scale_name], framework="pt"
        ) as f:
            scale = f.get_tensor(scale_name)
            if expert is not None:
                scale = scale[expert]
            m, n = weight.shape
            bm, bn = block_size
            if list(scale.shape) != [(m + bm - 1) // bm, (n + bn - 1) // bn]:
                raise ValueError(
                    f"{name}: FP8 scale shape does not match weight/block size"
                )
            result = weight.float()
            for row in range(0, m, bm):
                row_scale = scale[row // bm].float().repeat_interleave(bn)[:n]
                result[row : row + bm].mul_(row_scale)
        return result

    def _tensor_stats(self, tensor: torch.Tensor, role: str) -> dict[str, Any]:
        cfg = self.autometric_runs_config
        x = tensor.to(device=cfg.device).float()
        stats: dict[str, Any] = {"base": self.basic_stats(x)}
        if x.ndim != 2:
            return stats
        stats["rowcol"] = self.rowcol_stats(x)
        if (
            role not in self.NON_SPECTRAL_ROLES
            and cfg.topk_sigma > 0
            and min(x.shape) >= self.SPECTRAL_MIN_DIM
            and x.numel() <= cfg.spectral_numel_cap
        ):
            stats["spectral"] = self.spectral_stats(x, cfg.topk_sigma, cfg.svd_niter)
        return stats

    @classmethod
    def classify(
        cls,
        name: str,
        layer_roles: dict[str, str] | None = None,
        mtp_layers: set[str] | None = None,
        role_overrides: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        role, proj = "unknown", None
        for pattern, pattern_role, pattern_proj in cls.CLASS_PATTERNS:
            match = pattern.search(name)
            if not match:
                continue
            role = pattern_role
            if pattern_proj is not None and pattern_proj.startswith("_capture"):
                captured = match.group(1) if match.groups() else None
                tail = pattern_proj[len("_capture") :].lstrip("_")
                proj = "_".join(part for part in (captured, tail) if part) or None
            else:
                proj = pattern_proj
            break
        layer = cls.LAYER_RE.search(name)
        expert = cls.EXPERT_RE.search(name)
        layer_prefix = name[: layer.end()] if layer else ""
        if re.search(r"\.self_attn\.(o_proj|gate_proj)\.(weight|bias)$", name):
            role = (layer_roles or {}).get(layer_prefix, "attn")
        for pattern, override in (role_overrides or {}).items():
            if re.search(pattern, name):
                role = override
                break
        return {
            "role": role,
            "layer": int(layer.group(1)) if layer else None,
            "expert": int(expert.group(1)) if expert else None,
            "proj": proj,
            "block": (
                "mtp"
                if (
                    cls.MTP_MARKERS.search(name)
                    or cls.MTP_PATH_RE.search(name)
                    or layer_prefix in (mtp_layers or set())
                )
                else "main"
            ),
        }

    def basic_stats(self, x: torch.Tensor) -> dict[str, Any]:
        cfg = self.autometric_runs_config
        x = x.reshape(-1)
        absx = x.abs()
        mean = float(x.mean())
        centered = x - mean
        m2 = float(centered.pow(2).mean().clamp_min(0))
        m3 = float(centered.pow(3).mean())
        m4 = float(centered.pow(4).mean())
        k_tail = min(self.TOP_K_TAIL, x.numel())
        top = torch.topk(absx, k_tail).values
        width = (cfg.hist_log_max - cfg.hist_log_min) / cfg.hist_bins
        bins = ((absx.clamp_min(1e-38).log10() - cfg.hist_log_min) / width).long()
        hist = torch.bincount(
            bins.clamp_(0, cfg.hist_bins - 1), minlength=cfg.hist_bins
        )
        return {
            "n_nan": int(torch.isnan(x).sum()),
            "n_inf": int(torch.isinf(x).sum()),
            "n_zero": int((x == 0).sum()),
            "n_pos": int((x > 0).sum()),
            "n_neg": int((x < 0).sum()),
            "mean": mean,
            "std": m2**0.5,
            "fro": float(x.pow(2).sum().sqrt()),
            "mean_abs": float(absx.mean()),
            "max_abs": float(absx.amax()),
            "min_signed": float(x.amin()),
            "max_signed": float(x.amax()),
            "skew": m3 / (m2**1.5 + 1e-30),
            "kurt_excess": m4 / (m2**2 + 1e-30) - 3.0,
            "top16_abs_mean": float(top[:16].mean()),
            "top64_abs_mean": float(top[:64].mean()),
            "top128_abs_mean": float(top.mean()),
            "abs_hist_counts": hist.tolist(),
        }

    @staticmethod
    def rowcol_stats(w: torch.Tensor) -> dict[str, dict[str, float]]:
        def describe(norms: torch.Tensor) -> dict[str, float]:
            median = float(norms.median())
            return {
                "min": float(norms.amin()),
                "max": float(norms.amax()),
                "median": median,
                "mean": float(norms.mean()),
                "std": float(norms.std(unbiased=False)),
                "max_over_median": float(norms.amax()) / (median + 1e-30),
            }

        return {
            "row": describe(w.pow(2).sum(dim=1).sqrt()),
            "col": describe(w.pow(2).sum(dim=0).sqrt()),
        }

    @staticmethod
    def spectral_stats(w: torch.Tensor, k: int, niter: int) -> dict[str, Any]:
        q = min(k, *w.shape)
        try:
            _, sigma, _ = torch.svd_lowrank(w, q=q, niter=niter)
        except Exception as error:  # noqa: BLE001
            return {"error": f"svd_lowrank_failed: {error!r}"}
        sigma1 = float(sigma[0])
        p = sigma / sigma.sum().clamp_min(1e-30)
        sample_idx = [0, min(1, q - 1), q // 4, q // 2, q - 1]
        return {
            "k": q,
            "sigma_sample_idx": sample_idx,
            "sigma_samples": [float(sigma[i]) for i in sample_idx],
            "spectral_norm": sigma1,
            "nuclear_topk_approx": float(sigma.sum()),
            "stable_rank": float(w.pow(2).sum()) / (sigma1**2 + 1e-30),
            "effective_rank_entropy_topk": float(
                torch.exp(-(p * (p + 1e-30).log()).sum())
            ),
        }

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, dict):
            return {key: cls._json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._json_safe(item) for item in value]
        return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute HF checkpoint weight statistics"
    )
    parser.add_argument("--hf-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--role-overrides", type=json.loads, default={})
    parser.add_argument("--topk-sigma", type=int, default=16)
    parser.add_argument("--svd-niter", type=int, default=2)
    parser.add_argument("--spectral-numel-cap", type=int, default=20 * 1024**3 // 4)
    parser.add_argument("--include")
    parser.add_argument("--exclude")
    parser.add_argument("--hist-bins", type=int, default=80)
    parser.add_argument("--hist-log-min", type=float, default=-8.0)
    parser.add_argument("--hist-log-max", type=float, default=4.0)
    args = parser.parse_args()
    hf_dir, output = args.hf_dir, args.output
    del args.hf_dir, args.output
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    CheckpointStatsAnalyzer(hf_dir, output, args).run()


if __name__ == "__main__":
    main()
