import json
import logging
import re
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from tokenizers import Tokenizer

from tree_sitter import Parser
from tree_sitter_language_pack import get_parser

from torchtitan.config.schema.datasets.streaming.text.transforms import (
    validate_fim_rates,
)

log = logging.getLogger(__name__)


_ALL_LANGUAGES = ("java", "cpp", "rust", "go", "python")
_WHITELIST_TYPES = {
    "constructor_declaration",
    "method_declaration",
    "class_definition",
    "function_definition",
    "if_statement",
    "enhanced_for_statement",
    "for_statement",
}
_CLOSING_LINE_RE = re.compile(r"\s*[),;:\\]}]*\s*$")


def _is_sentencepiece_style(tokenizer: Any) -> bool:
    """Detect tokenizers that prepend a space marker when encoding mid-text."""
    candidates = (tokenizer, getattr(tokenizer, "tokenizer", None))
    for candidate in candidates:
        if candidate is not None and isinstance(candidate, Tokenizer):
            try:
                spec = json.loads(candidate.to_str())
            except Exception:
                return False
            if (spec.get("model") or {}).get("type") == "Unigram":
                return True
            sections = json.dumps(
                [spec.get(key) for key in ("normalizer", "pre_tokenizer", "decoder")]
            )
            return '"Metaspace"' in sections
    return False


class FIMTransform:
    """Fill-in-middle transform for eos-delimited pretokenized documents.

    Each document is split into prefix/middle/suffix by a configured FIM variant
    (vanilla, inline, multiline, syntax-aware) and reassembled in PSM or SPM
    order with sentinel tokens. Randomness comes only from a per-sample generator
    seeded with ``(seed, epoch, idx)``, so every rank produces a bit-identical
    stream.

    Requires a torchtitan ``HuggingFaceTokenizer``-style interface: ``encode``
    honoring ``add_bos``/``add_eos``/``add_special_tokens`` (so re-encoded pieces
    carry no BOS/EOS), plus ``decode``, ``token_to_id`` and an ``eos_id``.
    """

    def __init__(
        self,
        *,
        tokenizer: Any,
        max_seq_len: int,
        seed: int = 0,
        stream_fim_rates: list[dict[str, float]] | None = None,
        default_fim_rates: dict[str, float] | None = None,
        fim_prefix_token: str = "<|fim_prefix|>",
        fim_middle_token: str = "<|fim_middle|>",
        fim_suffix_token: str = "<|fim_suffix|>",
    ) -> None:
        if tokenizer is None:
            raise ValueError("FIM requires a tokenizer")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if seed < 0:
            raise ValueError(f"seed must be non-negative, got {seed}")

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.seed = seed
        eos_id = getattr(tokenizer, "eos_id", None)
        if eos_id is None:
            raise ValueError("FIM requires tokenizer eos_id")
        self.eos_token_id = int(eos_id)
        bos_id = getattr(tokenizer, "bos_id", None)
        self.bos_token_id = None if bos_id is None else int(bos_id)
        self.stream_fim_rates = list(stream_fim_rates or [])
        self.default_fim_rates = default_fim_rates or {}
        for rates in (self.default_fim_rates, *self.stream_fim_rates):
            validate_fim_rates(rates)
        self._warned_syntax_fallback = False

        self._parsers: dict[str, Parser] = {}
        syntax_aware_enabled = any(
            rates.get("syntax_aware_fim", 0) > 0
            for rates in (self.default_fim_rates, *self.stream_fim_rates)
        )
        if syntax_aware_enabled:
            try:
                self._parsers = {lang: get_parser(lang) for lang in _ALL_LANGUAGES}
            except Exception as exc:
                raise RuntimeError(
                    "syntax_aware_fim is enabled but Tree-sitter grammars "
                    f"failed to load: {exc}"
                ) from exc

        self.fim_prefix_token_id = self._token_to_id(fim_prefix_token)
        self.fim_middle_token_id = self._token_to_id(fim_middle_token)
        self.fim_suffix_token_id = self._token_to_id(fim_suffix_token)
        for name, token, token_id in (
            ("prefix", fim_prefix_token, self.fim_prefix_token_id),
            ("middle", fim_middle_token, self.fim_middle_token_id),
            ("suffix", fim_suffix_token, self.fim_suffix_token_id),
        ):
            if token_id is None:
                log.warning(
                    "FIM %s token %s not found in tokenizer vocabulary", name, token
                )

        self.fim_tokens_exist_in_vocab = all(
            token_id is not None
            for token_id in (
                self.fim_prefix_token_id,
                self.fim_middle_token_id,
                self.fim_suffix_token_id,
            )
        )
        self.char_level_split_method = (
            self._sp_charlevel_split
            if _is_sentencepiece_style(tokenizer)
            else self._bpe_charlevel_split
        )

    def _token_to_id(self, token: str) -> int | None:
        if hasattr(self.tokenizer, "token_to_id"):
            return self.tokenizer.token_to_id(token)
        raise ValueError("FIM requires tokenizer token_to_id support")

    def _encode(self, text: str) -> list[int]:
        token_ids = self.tokenizer.encode(
            text, add_bos=False, add_eos=False, add_special_tokens=False
        )
        return [int(token_id) for token_id in token_ids]

    def _decode(self, tokens: list[int] | np.ndarray) -> str:
        token_ids = [int(token_id) for token_id in tokens]
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def __call__(
        self,
        tokens: torch.Tensor,
        *,
        stream_id: int,
        idx: int | None = None,
        epoch: int = 0,
    ) -> torch.Tensor:
        fim_rates = (
            self.stream_fim_rates[stream_id]
            if self.stream_fim_rates
            else self.default_fim_rates
        )
        if sum(fim_rates.values()) <= 0:
            return tokens
        if idx is None:
            raise ValueError(
                "FIMTransform requires the sample idx to derive its "
                "per-sample RNG; got idx=None"
            )
        if epoch < 0:
            raise ValueError(f"epoch must be non-negative, got {epoch}")
        rng = np.random.default_rng((self.seed, epoch, idx))
        return self.transform(tokens, fim_rates=fim_rates, rng=rng)

    def split_context_into_doc(self, tokens: np.ndarray) -> list[np.ndarray]:
        parts = np.split(tokens, np.where(tokens == self.eos_token_id)[0])
        return [part if idx == 0 else part[1:] for idx, part in enumerate(parts)]

    def _sp_charlevel_split(
        self,
        token_id: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        def sp_encode(text: str) -> list[int]:
            return self._encode("\n" + text)[2:]

        return self._base_charlevel_split(token_id, sp_encode, rng)

    def _bpe_charlevel_split(
        self,
        token_id: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._base_charlevel_split(token_id, self._encode, rng)

    def _base_charlevel_split(
        self,
        token_id: int,
        encode_method: Callable[[str], list[int]],
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        token = self._decode([int(token_id)])
        if not token:
            return (
                np.asarray([], dtype=np.int64),
                np.asarray([int(token_id)], dtype=np.int64),
            )
        idx = int(rng.integers(len(token)))
        left_text, right_text = token[:idx], token[idx:]
        left = encode_method(left_text) if left_text else []
        right = encode_method(right_text) if right_text else []
        return np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)

    def _error_count(self, node: Any) -> int:
        count = 0
        stack = [node]
        while stack:
            current = stack.pop()
            count += current.type == "ERROR"
            stack.extend(current.children)
        return count

    def _get_best_parser(self, code_bytes: bytes) -> tuple[Parser, str]:
        best_score = float("inf")
        best_lang = None
        best_parser = None

        for lang, parser in self._parsers.items():
            tree = parser.parse(code_bytes)
            score = self._error_count(tree.root_node)
            if score < best_score:
                best_score = score
                best_lang = lang
                best_parser = parser
                if score == 0:
                    break

        if best_parser is None or best_lang is None:
            raise RuntimeError("No suitable Tree-Sitter grammar found")
        return best_parser, best_lang

    def _all_descendants(self, node: Any):
        stack = [node]
        while stack:
            current = stack.pop()
            yield current
            stack.extend(reversed(current.children))

    @staticmethod
    def _code_for(node: Any, source_bytes: bytes) -> str:
        return source_bytes[node.start_byte : node.end_byte].decode()

    def _pick_random_multiline_nonleaf_node(
        self,
        tree: Any,
        source_bytes: bytes,
        rng: np.random.Generator,
    ) -> Any | None:
        candidates = []
        for node in self._all_descendants(tree.root_node):
            if node.parent is None or not node.children:
                continue
            snippet = self._code_for(node, source_bytes)
            if len(snippet) < 10:
                continue
            if sum(1 for line in snippet.split("\n") if line.strip()) < 2:
                continue
            if node.type not in _WHITELIST_TYPES:
                continue
            candidates.append(node)
        if not candidates:
            return None
        return candidates[int(rng.integers(len(candidates)))]

    @staticmethod
    def _split_three_random(
        text: str,
        rng: np.random.Generator,
    ) -> tuple[str, str, str]:
        if len(text) < 3:
            return text, "", ""

        a = int(rng.integers(1, len(text) - 1))
        b = int(rng.integers(a + 1, len(text)))
        newline_at = text.find("\n", b)
        if newline_at != -1:
            return text[:a], text[a : newline_at + 1], text[newline_at + 1 :]
        return text[:a], text[a:], ""

    def syntax_aware_split(
        self,
        tokens: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        source_code = self._decode(tokens)
        source_bytes = source_code.encode()

        try:
            parser, _ = self._get_best_parser(source_bytes)
            tree = parser.parse(source_bytes)
            node = self._pick_random_multiline_nonleaf_node(tree, source_bytes, rng)

            if node is None:
                return self.multiline_fim_split(tokens, rng)

            snippet = self._code_for(node, source_bytes)
            part1, part2, _ = self._split_three_random(snippet, rng)
            # node offsets are bytes; downstream slicing is on the decoded string
            start_pos = len(source_bytes[: node.start_byte].decode())
            pos1 = start_pos + len(part1)
            pos2 = pos1 + len(part2)

            return (
                self._encode_np(source_code[:pos1]),  # prefix_tokens
                self._encode_np(source_code[pos1:pos2]),  # middle_tokens
                self._encode_np(source_code[pos2:]),  # suffix_tokens
            )
        except ValueError as exc:
            if not self._warned_syntax_fallback:
                self._warned_syntax_fallback = True
                log.warning(
                    "Syntax-aware splitting failed: %s; "
                    "falling back to random splitting",
                    exc,
                )
            return self.random_split_tokens_legacy(tokens, rng)

    def inline_fim_split(
        self,
        tokens: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        source_code = self._decode(tokens)
        lines = source_code.split("\n")
        if not lines:
            return self.random_split_tokens_legacy(tokens, rng)

        line_indices = [idx for idx, line in enumerate(lines) if line.strip()]
        if not line_indices:
            return self.random_split_tokens_legacy(tokens, rng)
        line_idx = line_indices[int(rng.integers(len(line_indices)))]
        selected_line = lines[line_idx]

        if not selected_line:
            return self.random_split_tokens_legacy(tokens, rng)

        char_pos = int(rng.integers(len(selected_line)))
        middle_length = min(int(rng.integers(121)), len(selected_line) - char_pos)

        prefix_text = (
            "\n".join(lines[:line_idx])
            + ("\n" if line_idx > 0 else "")
            + selected_line[:char_pos]
        )
        middle_text = selected_line[char_pos : char_pos + middle_length]
        suffix_text = (
            selected_line[char_pos + middle_length :]
            + ("\n" if line_idx < len(lines) - 1 else "")
            + "\n".join(lines[line_idx + 1 :])
        )

        return (
            self._encode_np(prefix_text),
            self._encode_np(middle_text),
            self._encode_np(suffix_text),
        )

    def multiline_fim_split(
        self,
        tokens: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        source_code = self._decode(tokens)
        if "\n" not in source_code:
            return self.random_split_tokens_legacy(tokens, rng)

        lines = source_code.splitlines(keepends=True)
        line_idx = int(rng.integers(len(lines)))
        p = 0.1
        num_lines = 1
        while rng.random() > p and (line_idx + num_lines) < len(lines):
            num_lines += 1

        char_at_line = sum(len(line) for line in lines[:line_idx])
        ws_match = re.match(r"\s*", source_code[char_at_line:])
        ws_len = ws_match.end() if ws_match is not None else 0
        prefix_end = (
            char_at_line
            if ws_len == 0
            else int(rng.integers(char_at_line, char_at_line + ws_len + 1))
        )

        middle_end_line_idx = min(line_idx + num_lines + 1, len(lines))
        for idx in range(line_idx, middle_end_line_idx):
            if _CLOSING_LINE_RE.fullmatch(lines[idx].rstrip("\n")):
                middle_end_line_idx = idx
                break

        middle_end_char = sum(len(line) for line in lines[:middle_end_line_idx])
        return (
            self._encode_np(source_code[:prefix_end]),  # prefix_tokens
            self._encode_np(source_code[prefix_end:middle_end_char]),  # middle_tokens
            self._encode_np(source_code[middle_end_char:]),  # suffix_tokens
        )

    def random_split_tokens_legacy(
        self,
        tokens: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        idxs = rng.choice(len(tokens), size=2, replace=False)
        idxs.sort()

        pre_mid_left, pre_mid_right = self.char_level_split_method(
            int(tokens[idxs[0]]), rng
        )
        mid_suf_left, mid_suf_right = self.char_level_split_method(
            int(tokens[idxs[1]]), rng
        )
        prefix = np.concatenate([tokens[: idxs[0]], pre_mid_left])
        middle = np.concatenate(
            [pre_mid_right, tokens[idxs[0] + 1 : idxs[1]], mid_suf_left]
        )
        suffix = np.concatenate([mid_suf_right, tokens[idxs[1] + 1 :]])
        return prefix, middle, suffix

    def random_split_tokens(
        self,
        tokens: np.ndarray,
        fim_variant: str = "vanilla_fim",
        *,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if fim_variant == "vanilla_fim":
            return self.random_split_tokens_legacy(tokens, rng)
        if fim_variant == "inline_fim":
            return self.inline_fim_split(tokens, rng)
        if fim_variant in {"syntax_aware", "syntax_aware_fim"}:
            return self.syntax_aware_split(tokens, rng)
        if fim_variant == "multiline_fim":
            return self.multiline_fim_split(tokens, rng)
        raise ValueError(f"Unknown FIM variant: {fim_variant!r}")

    def _fim_token_ids(self) -> tuple[int, int, int]:
        if (
            self.fim_prefix_token_id is None
            or self.fim_middle_token_id is None
            or self.fim_suffix_token_id is None
        ):
            raise RuntimeError(
                "Attempting FIM transformation, but FIM tokens "
                "(<|fim_prefix|>, etc) do not exist in vocab"
            )
        return (
            self.fim_prefix_token_id,
            self.fim_middle_token_id,
            self.fim_suffix_token_id,
        )

    def concat_psm(
        self,
        prefix: np.ndarray,
        middle: np.ndarray,
        suffix: np.ndarray,
    ) -> np.ndarray:
        prefix_id, middle_id, suffix_id = self._fim_token_ids()
        return np.concatenate(
            [
                [prefix_id],
                prefix,
                [suffix_id],
                suffix,
                [middle_id],
                middle,
                [self.eos_token_id],
            ]
        )

    def concat_spm(
        self,
        prefix: np.ndarray,
        middle: np.ndarray,
        suffix: np.ndarray,
    ) -> np.ndarray:
        prefix_id, middle_id, suffix_id = self._fim_token_ids()
        return np.concatenate(
            [
                [suffix_id],
                suffix,
                [prefix_id],
                prefix,
                [middle_id],
                middle,
                [self.eos_token_id],
            ]
        )

    def concat_causal(self, tokens: np.ndarray) -> np.ndarray:
        return np.concatenate([tokens, [self.eos_token_id]])

    def sample_transform(
        self,
        tokens: np.ndarray,
        fim_rates: dict[str, float],
        rng: np.random.Generator,
    ) -> np.ndarray:
        total_rate = sum(fim_rates.values())

        if total_rate <= 0 or len(tokens) < 30 or rng.random() > total_rate:
            return self.concat_causal(tokens)

        if tokens[-1] == self.eos_token_id:
            tokens = tokens[:-1]

        bos = tokens[:0]
        if self.bos_token_id is not None and tokens[0] == self.bos_token_id:
            bos, tokens = tokens[:1], tokens[1:]

        fim_variants = list(fim_rates.keys())
        fim_probs = [fim_rates[variant] / total_rate for variant in fim_variants]
        chosen_variant = str(rng.choice(fim_variants, p=fim_probs))

        for _ in range(10):
            prefix, middle, suffix = self.random_split_tokens(
                tokens, chosen_variant, rng=rng
            )
            if len(prefix) >= 5 and len(middle) >= 5 and len(suffix) >= 5:
                break

        if rng.random() < 0.5:
            return np.concatenate([bos, self.concat_psm(prefix, middle, suffix)])
        return np.concatenate([bos, self.concat_spm(prefix, middle, suffix)])

    def transform(
        self,
        tokens: torch.Tensor | np.ndarray | list[int],
        *,
        fim_rates: dict[str, float],
        rng: np.random.Generator,
    ) -> torch.Tensor:
        if not self.fim_tokens_exist_in_vocab:
            raise RuntimeError(
                "Attempting FIM transformation, but FIM tokens "
                "(<|fim_prefix|>, etc) do not exist in vocab"
            )

        parts = self.split_context_into_doc(self._as_numpy_tokens(tokens))
        parts = [self.sample_transform(part, fim_rates, rng) for part in parts]
        # no padding here: the collator pads to seq_len and masks pad labels
        tokens_transformed = np.concatenate(parts).astype(np.int64)[: self.max_seq_len]
        return torch.from_numpy(tokens_transformed)

    def _encode_np(self, text: str) -> np.ndarray:
        return np.asarray(self._encode(text), dtype=np.int64)

    @staticmethod
    def _as_numpy_tokens(tokens: torch.Tensor | np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, torch.Tensor):
            return tokens.detach().cpu().numpy().astype(np.int64, copy=False)
        return np.asarray(tokens, dtype=np.int64)
