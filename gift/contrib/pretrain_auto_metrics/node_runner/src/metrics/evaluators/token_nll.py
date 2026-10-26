import hashlib
import io
import json
import math
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import requests
import zstandard
from transformers import AutoTokenizer

from ...config_model import (
    ModelInfoConfig,
    TokenNllDatasetConfig,
    TokenNllEvaluatorConfig,
)
from ...job_dispatcher import JD
from ...job_dispatcher.cache_scores.scalar import ScalarCacheScoresBuilder
from ...job_dispatcher.measurement import MeasurementUpload
from ...logger import Logger
from .interface import MetricsEvaluator

RETRY_BACKOFF_SECONDS = 5


class TokenNllEvaluator(MetricsEvaluator):
    SHOOTER_NAME = "token_nll"
    SHOOTER_VERSION = "1.0.0"
    ALL_METRICS_FILENAME = "all_metrics.json"

    autometric_runs_config: TokenNllEvaluatorConfig

    def setup(
        self,
        autometric_runs_config: TokenNllEvaluatorConfig,
        artifacts_path: Path,
        hf_checkpoint_path: Path,
        model_info: ModelInfoConfig,
    ) -> None:
        self.autometric_runs_config = autometric_runs_config
        self.artifacts_path = Path(artifacts_path)
        self.hf_checkpoint_path = Path(hf_checkpoint_path)
        self.model_info = model_info
        self._failed = False
        self._metrics: dict[str, float] = {}
        self._loss_samples: dict[
            str, tuple[list[tuple[str, float, int]], int, str]
        ] = {}
        self._create_metrics_dir(self.artifacts_path, self.SHOOTER_NAME)
        self.all_metrics_path = self.metrics_dir / self.ALL_METRICS_FILENAME
        run_dir, _, _, self.step = JD.parse_checkpoint_path(self.hf_checkpoint_path)
        cache_dir = autometric_runs_config.cache_dir
        self.cache_dir = (
            Path(cache_dir) if cache_dir else Path(run_dir) / "token_nll_cache"
        )
        Logger.info(f"TokenNll setup: step={self.step}, cache_dir={self.cache_dir}")

    def run(self) -> None:
        cfg = self.autometric_runs_config
        try:
            tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_dir().as_posix())
            nlls = {}
            for name, dataset in cfg.datasets.items():
                nll = self._run_dataset(name, dataset, tokenizer)
                if nll is None:
                    self._failed = True
                    return
                nlls[name] = nll
            self._collect_val_loss(nlls)
        except Exception as e:
            self._failed = True
            Logger.error(f"token_nll failed: {e}\n{traceback.format_exc()}")

    def _tokenizer_dir(self) -> Path:
        # Loading from hf_checkpoint also parses its config.json, which the image's transformers
        # rejects for yarn-scaled configs; the plain tokenizer dir has no config.json.
        plain = self.hf_checkpoint_path.parent / "tokenizer"
        if (plain / "tokenizer_config.json").exists():
            return plain
        return self.hf_checkpoint_path

    def get_metrics(self) -> dict[str, float]:
        if self._failed or not self._metrics:
            return {}
        self._save_all_metrics()
        return dict(self._metrics)

    def after_run(
        self, metrics: dict[str, float], uploaded: Optional[MeasurementUpload] = None
    ) -> None:
        self._send_metrics_to_jd(metrics)
        tar_path = JD.tar_metrics_dir(
            self.metrics_dir, self.artifacts_path, type(self).__name__
        )
        JD.symlink_metrics_for_transfer(
            tar_path, self.hf_checkpoint_path, type(self).__name__
        )

    def _run_dataset(
        self, name: str, dataset: TokenNllDatasetConfig, tokenizer: AutoTokenizer
    ) -> np.ndarray | None:
        cfg = self.autometric_runs_config
        tokenized = self._tokenize(dataset, tokenizer)
        if tokenized is None:
            return None
        tokens, doc_offsets = tokenized
        fingerprint = self._fingerprint(
            tokens, cfg.window_tokens, self.hf_checkpoint_path.name
        )
        dataset_cache_dir = self.cache_dir / name
        dataset_cache_dir.mkdir(parents=True, exist_ok=True)
        self._save_tokens(dataset_cache_dir, fingerprint, tokens, doc_offsets, dataset)

        nll_path = dataset_cache_dir / f"nll_{fingerprint}_ba{self.step}.npy"
        nll = self._load_nll(nll_path)
        if nll is None:
            nll = self._score(tokens, doc_offsets)
            if nll is None:
                return None
            self._save_atomic(nll_path, nll)
        else:
            Logger.info(f"token_nll: reusing cached NLL {nll_path}")
        mask = ~np.isnan(nll)
        expected_mask = np.ones(tokens.size, dtype=bool)
        expected_mask[doc_offsets] = False
        complete = nll.shape == tokens.shape and np.array_equal(mask, expected_mask)
        identity = hashlib.sha256(
            tokens.tobytes()
            + mask.tobytes()
            + json.dumps([doc_offsets, cfg.window_tokens]).encode()
        ).hexdigest()
        bounds = [*doc_offsets, tokens.size]
        samples = [
            (
                f"{name}:{index}",
                float(np.nansum(nll[start:end], dtype=np.float64)),
                int(mask[start:end].sum()),
            )
            for index, (start, end) in enumerate(zip(bounds, bounds[1:]))
            if mask[start:end].any()
        ]
        if complete:
            self._loss_samples[name] = (samples, int(mask.sum()), identity)
        else:
            Logger.warning(
                f"token_nll CI unavailable for {name}: incomplete token coverage"
            )
        self._collect_memorization(
            name, dataset_cache_dir, fingerprint, tokens, doc_offsets, nll, tokenizer
        )
        return nll

    def _tokenize(
        self, dataset: TokenNllDatasetConfig, tokenizer: AutoTokenizer
    ) -> tuple[np.ndarray, list[int]] | None:
        max_docs = dataset.max_docs or self.autometric_runs_config.max_docs
        pieces: list[np.ndarray] = []
        doc_offsets: list[int] = []
        total = 0
        for text in self._iter_texts(Path(dataset.path), max_docs):
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if tokenizer.bos_token_id is not None:
                ids = [tokenizer.bos_token_id, *ids]
            if len(ids) < 2:
                continue
            doc_offsets.append(total)
            pieces.append(np.asarray(ids, dtype=np.uint32))
            total += len(ids)
        if not pieces:
            Logger.error(f"token_nll: no documents with text in {dataset.path}")
            return None
        return np.concatenate(pieces), doc_offsets

    def _save_tokens(
        self,
        dataset_cache_dir: Path,
        fingerprint: str,
        tokens: np.ndarray,
        doc_offsets: list[int],
        dataset: TokenNllDatasetConfig,
    ) -> None:
        tokens_path = dataset_cache_dir / f"tokens_{fingerprint}.npy"
        if tokens_path.exists():
            return
        self._save_atomic(tokens_path, tokens)
        meta = {
            "path": dataset.path,
            "n_docs": len(doc_offsets),
            "n_tokens": int(tokens.size),
            "window_tokens": self.autometric_runs_config.window_tokens,
            "doc_offsets": doc_offsets,
        }
        (dataset_cache_dir / f"meta_{fingerprint}.json").write_text(
            json.dumps(meta), encoding="utf-8"
        )

    def _score(self, tokens: np.ndarray, doc_offsets: list[int]) -> np.ndarray | None:
        cfg = self.autometric_runs_config
        windows = list(self._iter_windows(tokens.size, doc_offsets, cfg.window_tokens))
        url = f"{cfg.require_base_url()}/generate"
        Logger.info(
            f"token_nll: scoring {tokens.size} tokens in {len(windows)} windows "
            f"with concurrency={cfg.concurrency}"
        )
        nll = np.full(tokens.size, np.nan, dtype=np.float32)
        window_tokens = [tokens[start:end] for start, end in windows]
        with ThreadPoolExecutor(max_workers=cfg.concurrency) as pool:
            logprobs_per_window = pool.map(
                partial(self._request_logprobs, url), window_tokens
            )
            for (start, end), logprobs in zip(windows, logprobs_per_window):
                if logprobs is None:
                    return None
                nll[start + 1 : end] = -np.asarray(logprobs, dtype=np.float32)
        return nll

    def _request_logprobs(self, url: str, window: np.ndarray) -> list[float] | None:
        cfg = self.autometric_runs_config
        payload = {
            "input_ids": [int(token) for token in window],
            "sampling_params": {"max_new_tokens": 1, "temperature": 0},
            "return_logprob": True,
            "logprob_start_len": 0,
        }
        attempt = 0
        while True:
            attempt += 1
            try:
                response = requests.post(url, json=payload, timeout=cfg.request_timeout)
                response.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                if attempt == cfg.request_retries:
                    Logger.error(
                        f"token_nll: request failed after {attempt} attempts: {e}"
                    )
                    return None
                Logger.warning(f"token_nll: request failed ({e}), retry {attempt}")
                time.sleep(RETRY_BACKOFF_SECONDS)
        entries = response.json()["meta_info"]["input_token_logprobs"]
        n_scored = len(window) - 1
        logprobs = [entry[0] for entry in entries[-n_scored:]]
        if len(logprobs) != n_scored or any(value is None for value in logprobs):
            Logger.error(
                f"token_nll: expected {n_scored} input logprobs, got {len(entries)}"
            )
            return None
        return logprobs

    def _collect_val_loss(self, nlls: dict[str, np.ndarray]) -> None:
        total_sum = 0.0
        total_count = 0
        for name, nll in nlls.items():
            scored = nll[~np.isnan(nll)].astype(np.float64)
            mean = float(scored.mean())
            self._metrics[f"val_loss/{name}"] = mean
            self._metrics[f"val_ppl/{name}"] = math.exp(mean)
            total_sum += float(scored.sum())
            total_count += scored.size
        self._metrics["val_loss/all"] = total_sum / total_count

    def _collect_memorization(
        self,
        name: str,
        dataset_cache_dir: Path,
        fingerprint: str,
        tokens: np.ndarray,
        doc_offsets: list[int],
        nll: np.ndarray,
        tokenizer: AutoTokenizer,
    ) -> None:
        cfg = self.autometric_runs_config
        reference = self._find_reference(
            dataset_cache_dir, fingerprint, self.step - cfg.ref_lag_batches
        )
        if reference is None:
            Logger.info(
                f"token_nll: no reference for {name} at step <= "
                f"{self.step - cfg.ref_lag_batches}, skipping memorization metrics"
            )
            return
        ref_step, ref_path = reference
        ref_nll = self._load_nll(ref_path)
        if ref_nll is None:
            Logger.error(
                f"token_nll: reference {ref_path} is unreadable, skipping memorization "
                f"metrics for {name}"
            )
            return
        for key, value in self.memorization_proxy(
            ref_nll, nll, cfg.memorization_threshold
        ).items():
            self._metrics[f"memorization/{name}/{key}"] = value
        self._metrics[f"memorization/{name}/ref_step"] = float(ref_step)
        self._write_spans(name, tokens, doc_offsets, ref_nll, nll, tokenizer)

    def _write_spans(
        self,
        name: str,
        tokens: np.ndarray,
        doc_offsets: list[int],
        ref_nll: np.ndarray,
        cur_nll: np.ndarray,
        tokenizer: AutoTokenizer,
    ) -> None:
        cfg = self.autometric_runs_config
        spans = self.memorized_spans(ref_nll, cur_nll, cfg.memorization_threshold)
        spans = sorted(spans, key=lambda span: -span[2])[: cfg.top_spans]
        offsets = np.asarray(doc_offsets)
        with (self.metrics_dir / f"memorized_spans_{name}.jsonl").open(
            "w", encoding="utf-8"
        ) as out:
            for start, end, contribution in spans:
                doc = int(np.searchsorted(offsets, start, side="right") - 1)
                record = {
                    "doc": doc,
                    "offset": int(start - offsets[doc]),
                    "n_tokens": end - start,
                    "contribution": contribution,
                    "nll_ref_mean": float(np.nanmean(ref_nll[start:end])),
                    "nll_cur_mean": float(np.nanmean(cur_nll[start:end])),
                    "text": tokenizer.decode(tokens[start:end]),
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _save_all_metrics(self) -> None:
        try:
            self.all_metrics_path.write_text(
                json.dumps(self._metrics, ensure_ascii=False, indent=4),
                encoding="utf-8",
            )
        except OSError as error:
            Logger.error(f"Failed to save {self.ALL_METRICS_FILENAME}: {error}")

    def _send_metrics_to_jd(self, processed_metrics: dict[str, float]) -> None:
        if not JD.can_send_metrics(
            metrics=processed_metrics,
            hf_checkpoint_path=self.hf_checkpoint_path,
            metrics_dir=self.metrics_dir,
            model_info=self.model_info,
        ):
            return
        scores = ScalarCacheScoresBuilder(self.metrics_dir, processed_metrics)
        all_samples = []
        total_tokens = 0
        identities = {}
        for name, (samples, token_count, identity) in self._loss_samples.items():
            if token_count == 0:
                continue
            scores.add_ratio_metric(
                f"val_loss/{name}",
                samples,
                identity,
            )
            all_samples.extend(samples)
            total_tokens += token_count
            identities[name] = identity
        if total_tokens and set(identities) == set(
            self.autometric_runs_config.datasets
        ):
            scores.add_ratio_metric(
                "val_loss/all",
                all_samples,
                identities,
            )
        JD.send_evaluator_metrics_with_scores(
            scores=scores,
            hf_checkpoint_path=self.hf_checkpoint_path,
            metrics_dir=self.metrics_dir,
            shooter_name=self.SHOOTER_NAME,
            shooter_version=self.SHOOTER_VERSION,
            model_info=self.model_info,
        )

    @staticmethod
    def memorization_proxy(
        ref_nll: np.ndarray, cur_nll: np.ndarray, threshold: float
    ) -> dict[str, float]:
        valid = ~np.isnan(ref_nll) & ~np.isnan(cur_nll)
        delta = ref_nll[valid].astype(np.float64) - cur_nll[valid].astype(np.float64)
        below = cur_nll[valid] < threshold
        stats = {
            "frac_tokens_below_thr": float(below.mean()),
            "delta_vs_ref": float(delta.mean()),
        }
        if below.any():
            stats["delta_below_thr"] = float(delta[below].mean())
        if (~below).any():
            stats["delta_rest"] = float(delta[~below].mean())
        total = float(delta.sum())
        if total > 0:
            stats["proxy"] = float(delta[below].sum() / total)
        return stats

    @staticmethod
    def memorized_spans(
        ref_nll: np.ndarray, cur_nll: np.ndarray, threshold: float
    ) -> list[tuple[int, int, float]]:
        below = np.nan_to_num(cur_nll, nan=np.inf) < threshold
        delta = np.nan_to_num(ref_nll.astype(np.float64) - cur_nll, nan=0.0)
        edges = np.flatnonzero(
            np.diff(np.concatenate(([0], below.astype(np.int8), [0])))
        )
        starts, ends = edges[0::2], edges[1::2]
        cumulative = np.concatenate(([0.0], np.cumsum(delta)))
        return [
            (int(start), int(end), float(cumulative[end] - cumulative[start]))
            for start, end in zip(starts, ends)
        ]

    @staticmethod
    def _find_reference(
        dataset_cache_dir: Path, fingerprint: str, max_step: int
    ) -> tuple[int, Path] | None:
        candidates = []
        for path in dataset_cache_dir.glob(f"nll_{fingerprint}_ba*.npy"):
            step = int(path.stem.rsplit("_ba", 1)[1])
            if step <= max_step:
                candidates.append((step, path))
        return max(candidates) if candidates else None

    @staticmethod
    def _iter_windows(
        n_tokens: int, doc_offsets: list[int], window_tokens: int
    ) -> Iterator[tuple[int, int]]:
        bounds = [*doc_offsets, n_tokens]
        for doc_start, doc_end in zip(bounds, bounds[1:]):
            start = doc_start
            while start < doc_end - 1:
                end = min(start + window_tokens, doc_end)
                yield start, end
                start = end - 1

    @staticmethod
    def _iter_texts(path: Path, max_docs: int) -> Iterator[str]:
        with (
            path.open("rb") as raw,
            zstandard.ZstdDecompressor().stream_reader(raw) as stream,
        ):
            yielded = 0
            for line in io.TextIOWrapper(stream, encoding="utf-8"):
                if yielded >= max_docs:
                    return
                if not line.strip():
                    continue
                text = json.loads(line).get("text")
                if not text:
                    continue
                yielded += 1
                yield text

    @staticmethod
    def _load_nll(path: Path) -> np.ndarray | None:
        if not path.exists():
            return None
        try:
            return np.load(path)
        except (OSError, ValueError) as e:
            Logger.error(f"token_nll: dropping unreadable cache {path}: {e}")
            path.unlink(missing_ok=True)
            return None

    @staticmethod
    def _save_atomic(path: Path, array: np.ndarray) -> None:
        tmp_path = path.with_name(f".{path.stem}.tmp.npy")
        np.save(tmp_path, array)
        os.replace(tmp_path, path)

    @staticmethod
    def _fingerprint(tokens: np.ndarray, window_tokens: int, model_kind: str) -> str:
        digest = hashlib.sha1(
            tokens.tobytes() + f"{window_tokens}:{model_kind}".encode("utf-8")
        )
        return digest.hexdigest()[:12]
