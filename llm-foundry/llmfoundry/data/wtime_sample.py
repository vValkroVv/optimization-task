"""Read-time planner for "timed units" MDS samples (`sample_format: timed_units`)."""
from __future__ import annotations

import collections
import io
import json
import logging
import math
import typing as tp

import audioaug
import av
import numpy as np

logger = logging.getLogger(__name__)

META_VERSION = 2
# UNCERTAIN: always audio, its text is never shown; FORCED_TEXT: never audio (zero-length span)
UNCERTAIN, PART_START, SENT_END, FORCED_TEXT = 1, 4, 8, 32
KNOWN_FLAGS = UNCERTAIN | PART_START | SENT_END | FORCED_TEXT | 2 | 16  # 2, 16: writer-side marks, ignored
MS_PER_TOKEN = 160.0
PAD_MS = 150
OPUS_SR = 48000
PREROLL = 3840


class WtimeDecodeError(RuntimeError):
    pass


class Item(tp.NamedTuple):
    """bos / eos / tag (start = token id) / audio (start, end in ms) / text (start, end = token span)."""
    kind: str
    start: int = 0
    end: int = 0
    labeled: bool = False


def _b(x: tp.Any) -> bytes:
    return x.tobytes() if isinstance(x, memoryview) else bytes(x)


def parse(sample: tp.Dict[str, tp.Any]) -> tp.Dict[str, tp.Any]:
    al = np.frombuffer(_b(sample["align"]), np.int32).reshape(-1, 4)
    meta = json.loads(_b(sample["meta"]).decode("utf-8"))
    if meta.get("v") != META_VERSION or "parts" not in meta:
        raise ValueError(f"not a meta v{META_VERSION} timed units sample: v={meta.get('v')}")
    if (al[:, 3] & ~KNOWN_FLAGS).any():
        raise ValueError(f"unknown unit flags {int(np.bitwise_or.reduce(al[:, 3]) & ~KNOWN_FLAGS)}: update the reader")
    s = {"tokens": np.frombuffer(_b(sample["tokens"]), np.int32), "tok_end": al[:, 0], "s": al[:, 1],
         "e": al[:, 2], "flags": al[:, 3], "meta": meta, "opus": _b(sample["opus"])}
    if len(s["tok_end"]) and int(s["tok_end"][-1]) != len(s["tokens"]):
        raise ValueError(f"align covers {int(s['tok_end'][-1])} tokens of {len(s['tokens'])}")
    return s


def part_bounds(s):
    starts = list(np.flatnonzero(s["flags"] & PART_START)) + [len(s["flags"])]
    return list(zip(starts[:-1], starts[1:]))


def tok0(s, i):
    return int(s["tok_end"][i - 1]) if i > 0 else 0


def _span(s, a, b, ms=MS_PER_TOKEN):
    st, en = int(s["s"][a]), int(s["e"][b - 1])
    prev_e = int(s["e"][a - 1]) if a > 0 and not s["flags"][a] & PART_START else None
    next_s = int(s["s"][b]) if b < len(s["s"]) and not s["flags"][b] & PART_START else None
    st = max(0, st - (PAD_MS if prev_e is None else max(0, min(PAD_MS, (st - prev_e) // 2))))
    en += PAD_MS if next_s is None else max(0, min(PAD_MS, (next_s - en) // 2))
    if en - st < ms:
        st = max(0, int(st - (ms - (en - st)) // 2))
        en = int(st + ms)
    return st, en


def lm_plan(s, x=0.5, run_sec=(2.0, 10.0), rng=None, max_clip_sec=None, ms=MS_PER_TOKEN):
    """Audio runs of U(run_sec) s alternate with text runs so that a share x of the text becomes audio."""
    if not 0 < x < 1:
        raise ValueError(f"x must be in (0, 1), got {x}")
    rng = rng or np.random.default_rng(0)
    plan = []
    for a, b in part_bounds(s):
        audio = (s["flags"][a:b] & UNCERTAIN) > 0
        forced = (s["flags"][a:b] & FORCED_TEXT) > 0
        ntok = np.diff(np.concatenate([[tok0(s, a)], s["tok_end"][a:b]]))
        converted = kept = k = 0
        mode = "audio" if rng.random() < x else "text"
        while k < b - a:
            if mode == "audio":
                want = rng.uniform(*run_sec) * 1000
                t_start, first = s["s"][a + k], True
                while k < b - a and (s["e"][a + k] - t_start < want or first):
                    if not audio[k] and not forced[k]:
                        audio[k] = True
                        converted += ntok[k]
                    first = False
                    k += 1
                mode = "text"
            else:
                goal = converted * (1 - x) / x
                first = True
                while k < b - a and (kept < goal or first):
                    if not audio[k] and not forced[k]:
                        kept += ntok[k]
                    first = False
                    k += 1
                mode = "audio"
        plan.append(Item("bos"))
        i = 0
        while i < b - a:
            j = i
            while j < b - a and audio[j] == audio[i]:
                j += 1
            if audio[i]:
                c0 = i
                for c in range(i + 1, j + 1):
                    if c == j or (max_clip_sec and s["e"][a + c] - s["s"][a + c0] > max_clip_sec * 1000):
                        plan.append(Item("audio", *_span(s, a + c0, a + c, ms)))
                        c0 = c
            else:
                plan.append(Item("text", tok0(s, a + i), int(s["tok_end"][a + j - 1]), True))
            i = j
        plan.append(Item("eos"))
    return plan


def asr_plan(s, max_chunk_sec=12.0, ms=MS_PER_TOKEN, tag=None):
    """Chunks of <= max_chunk_sec as audio, each followed by its transcript; a tag token after every BOS."""
    plan = []
    for a, b in part_bounds(s):
        plan.append(Item("bos"))
        if tag is not None:
            plan.append(Item("tag", tag))
        i = a
        while i < b:
            if s["flags"][i] & FORCED_TEXT:
                plan.append(Item("text", tok0(s, i), int(s["tok_end"][i]), True))
                i += 1
                continue
            j = i + 1
            while j < b and not s["flags"][j] & FORCED_TEXT and s["e"][j] - s["s"][i] <= max_chunk_sec * 1000:
                j += 1
            plan.append(Item("audio", *_span(s, i, j, ms)))
            plan += [Item("text", tok0(s, k), int(s["tok_end"][k]), not s["flags"][k] & UNCERTAIN) for k in range(i, j)]
            i = j
        plan.append(Item("eos"))
    return plan


def text_plan(s):
    """Text-only fallback: the certain text of every part."""
    plan = []
    for a, b in part_bounds(s):
        plan.append(Item("bos"))
        plan += [Item("text", tok0(s, k), int(s["tok_end"][k]), True) for k in range(a, b) if not s["flags"][k] & UNCERTAIN]
        plan.append(Item("eos"))
    return plan


def audio_tokens(st, en, ms=MS_PER_TOKEN):
    return max(1, math.ceil((en - st) / ms))


def _lengths(plan, audio_slack=0, ms=MS_PER_TOKEN):
    return [audio_tokens(it.start, it.end, ms) + audio_slack if it.kind == "audio"
            else it.end - it.start if it.kind == "text" else 1 for it in plan]


def trim_plan(plan, max_len, audio_slack=1, ms=MS_PER_TOKEN):
    """The plan cut to at most max_len positions; a cut plan gets no EOS, and a clip goes together with the text after it."""
    lens = _lengths(plan, audio_slack, ms)
    if sum(lens) <= max_len:
        return plan
    out, n = [], 0
    for it, need in zip(plan, lens):
        if n + need > max_len:
            if it.kind == "text" and max_len - n > 0 and out[-1].kind != "audio":
                out.append(Item("text", it.start, it.start + (max_len - n), it.labeled))
            break
        out.append(it)
        n += need
    while out and out[-1].kind == "audio":
        out.pop()
    return out


def _starts_sentence(s, it):
    if it.kind == "bos":
        return True
    u = int(np.searchsorted(s["tok_end"], it.start, side="right") if it.kind == "text"
            else np.searchsorted(s["s"], it.start, side="left"))
    return u < len(s["flags"]) and (u == 0 or bool(s["flags"][u] & PART_START) or bool(s["flags"][u - 1] & SENT_END))


def window_plan(s, plan, max_len, rng, audio_slack=1, audio_start=False, ms=MS_PER_TOKEN):
    """A random full window of a plan longer than max_len, starting at a part or sentence start if possible."""
    lens = _lengths(plan, audio_slack, ms)
    if sum(lens) <= max_len:
        return plan
    suffix = np.cumsum(lens[::-1])[::-1]
    head = plan[:2] if len(plan) > 1 and plan[1].kind == "tag" else plan[:1]
    ok, pref = [], []
    for i, it in enumerate(plan):
        if suffix[i] + len(head) * (it.kind != "bos") < max_len:
            break
        if it.kind in ("eos", "tag") or (audio_start and it.kind == "text"):
            continue
        ok.append(i)
        if _starts_sentence(s, it):
            pref.append(i)
    i = int(rng.choice(pref if len(pref) > 1 else ok))
    return trim_plan(plan[i:] if plan[i].kind == "bos" else head + plan[i:], max_len, audio_slack, ms)


def build_sequence(s, plan, audio_token_id, bos_token_id, eos_token_id, ignore_index=-100):
    """-> input_ids, labels, and the audio spans (ms), one placeholder per span."""
    ids, labels, spans = [], [], []
    for it in plan:
        if it.kind == "text":
            t = s["tokens"][it.start:it.end].astype(np.int64)
            ids.append(t)
            labels.append(t if it.labeled else np.full(len(t), ignore_index))
            continue
        ids.append(np.array([it.start if it.kind == "tag" else {"bos": bos_token_id, "eos": eos_token_id, "audio": audio_token_id}[it.kind]]))
        labels.append(np.array([eos_token_id if it.kind == "eos" else ignore_index]))
        if it.kind == "audio":
            spans.append((it.start, it.end))
    return np.concatenate(ids).astype(np.int64), np.concatenate(labels).astype(np.int64), spans


def decode_spans(opus: bytes, spans_ms: tp.Sequence[tp.Tuple[int, int]], target_sr: int = 16000) -> tp.List[np.ndarray]:
    """float32 mono PCM at target_sr of every [start_ms, end_ms) of the stream."""
    if not spans_ms:
        return []
    b48 = [(int(st) * OPUS_SR // 1000, int(en) * OPUS_SR // 1000) for st, en in spans_ms]
    runs: tp.List[tp.List[int]] = []
    for i, (b0, b1) in enumerate(b48):
        lo = max(0, b0 - PREROLL)
        if runs and lo <= runs[-1][1]:
            runs[-1][1], runs[-1][3] = max(runs[-1][1], b1), i
        else:
            runs.append([lo, b1, i, i])
    out: tp.List[tp.Optional[np.ndarray]] = [None] * len(b48)
    try:
        with av.open(io.BytesIO(opus)) as c:
            if not c.streams.audio:
                raise WtimeDecodeError("no audio stream")
            st = c.streams.audio[0]
            origin = st.start_time or 0
            pkts = []
            for p in c.demux(st):
                if p.size and p.pts is not None:
                    pkts.append(p)
                    if p.pts - origin >= runs[-1][1]:
                        break
            starts = np.array([p.pts - origin for p in pkts], np.int64)
            cc = st.codec_context
            for lo, hi, i0, i1 in runs:
                k0 = max(0, int(np.searchsorted(starts, lo, side="right")) - 1)
                k1 = int(np.searchsorted(starts, hi, side="left"))
                cc.flush_buffers()
                pos, bufs = None, []
                for p in pkts[k0:k1]:
                    for f in cc.decode(p):
                        pos = int(f.pts) - origin if pos is None else pos
                        bufs.append(f.to_ndarray().reshape(len(f.layout.channels), -1).mean(0))
                if pos is None:
                    raise WtimeDecodeError(f"no audio decoded for {lo}..{hi} (48 kHz samples)")
                buf = np.concatenate(bufs).astype(np.float32)
                for i in range(i0, i1 + 1):
                    seg = buf[max(0, b48[i][0] - pos):max(0, b48[i][1] - pos)]
                    if seg.size == 0:
                        raise WtimeDecodeError(f"span {spans_ms[i]} ms is outside the decoded audio")
                    out[i] = seg
    except WtimeDecodeError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise WtimeDecodeError(repr(exc)) from exc
    aug = [audioaug.Resample(target_sr)]
    return [audioaug.apply_augmentation(seg, audioaug.AudioInfo(sample_rate=OPUS_SR, channels=1, frames=seg.size), aug,
                                        return_dtype=seg.dtype)[0] for seg in out]


def to_omni_sample(sample, cfg: tp.Dict[str, tp.Any], rng: np.random.Generator, *, guard: "FallbackGuard",
                   max_seq_len: int, audio_token_id: int, sample_rate: int, bos_token_id: int, eos_token_id: int,
                   ms_per_token: float = MS_PER_TOKEN):
    """One stored sample -> (input_ids, labels, waves), one audio_token_id per wave.
    A plan longer than max_seq_len is cut at the end, or a random window of it is taken with random_window.
    With asr_tag_id an ASR sample carries that token (label ignored) after every BOS."""
    if bos_token_id is None or eos_token_id is None:
        raise ValueError("timed_units needs a tokenizer with BOS and EOS ids")
    s = parse(sample)
    if (s["tokens"] == audio_token_id).any():
        raise ValueError(f"doc {s['meta'].get('doc_id')}: text tokens contain the audio placeholder {audio_token_id}")
    slack = int(cfg.get("audio_slack", 1))
    asr = rng.random() < float(cfg.get("task_mix", {}).get("asr", 0.0))
    tag = int(cfg["asr_tag_id"]) if asr and "asr_tag_id" in cfg else None

    def fit(plan, audio_start=False):
        if cfg.get("random_window", False):
            return window_plan(s, plan, max_seq_len, rng, slack, audio_start, ms_per_token)
        return trim_plan(plan, max_seq_len, slack, ms_per_token)

    if asr:
        plan = asr_plan(s, float(cfg.get("asr_max_chunk_sec", 12.0)), ms_per_token, tag)
    else:
        plan = lm_plan(s, x=float(cfg.get("x", 0.5)), run_sec=tuple(cfg.get("run_sec", (2.0, 10.0))), rng=rng,
                       max_clip_sec=cfg.get("max_clip_sec", 30.0), ms=ms_per_token)
    plan = fit(plan, audio_start=asr)
    ids, labels, spans = build_sequence(s, plan, audio_token_id, bos_token_id, eos_token_id)
    try:
        waves = decode_spans(s["opus"], spans, target_sr=sample_rate)
    except WtimeDecodeError as exc:
        guard.record(False, f"doc {s['meta'].get('doc_id')}: {exc} -> text only")
        plan = fit(text_plan(s))
        ids, labels, _ = build_sequence(s, plan, audio_token_id, bos_token_id, eos_token_id)
        return ids, labels, []
    guard.record(True)
    return ids, labels, waves


class FallbackGuard:
    """Counts text-only fallbacks; raises on max_consecutive in a row or max_rate over `window` samples."""

    def __init__(self, window: int = 1000, max_rate: float = 0.01, max_consecutive: int = 20, name: str = "wtime"):
        self.window, self.max_rate, self.max_consecutive, self.name = window, max_rate, max_consecutive, name
        self.recent: tp.Deque[bool] = collections.deque(maxlen=window)
        self.consecutive = self.total = self.failed = 0
        logger.info("[%s] fallback guard: raise at > %d consecutive failures or > %.2f%% over %d samples",
                    name, max_consecutive, 100 * max_rate, window)

    def record(self, ok: bool, detail: str = "") -> None:
        self.total += 1
        self.recent.append(ok)
        if ok:
            self.consecutive = 0
            return
        self.failed += 1
        self.consecutive += 1
        logger.warning("[%s] fallback %d/%d (consecutive %d): %s", self.name, self.failed, self.total,
                       self.consecutive, detail)
        if self.consecutive > self.max_consecutive:
            raise RuntimeError(f"[{self.name}] {self.consecutive} consecutive fallbacks, last: {detail}")
        if len(self.recent) == self.window:
            rate = 1 - sum(self.recent) / self.window
            if rate > self.max_rate:
                raise RuntimeError(f"[{self.name}] fallback rate {rate:.2%} over the last {self.window} samples "
                                   f"> {self.max_rate:.2%}, last: {detail}")
