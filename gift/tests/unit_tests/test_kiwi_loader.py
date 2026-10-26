"""Synthetic-data tests for the kiwi token dataloader.

Vision streams currently emit one image per sample, so the fixture is 1:1
``id → tar → caption``. Scale stays small for CI; ``build_kiwi_fixture``
accepts larger ``n_tars`` / ``n_ids`` if a soak run is needed.
"""

from __future__ import annotations

import io
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from torchtitan.datasets.kiwi.batch import ModelBatchData
from torchtitan.datasets.kiwi.configs import (
    DatasetConfig,
    PackingConfig,
    PreprocessorConfig,
    TextDataStreamConfig,
    VisionDataStreamConfig,
)
from torchtitan.datasets.kiwi.constants import ENCODER_ORDER, DataModality
from torchtitan.datasets.kiwi.magi_ranges import MagiRanges, make_sdpa_mask
from torchtitan.datasets.kiwi.packing import BatchClass
from torchtitan.datasets.kiwi.preprocessing import (
    Preprocessor,
    SampleClass,
    TextPreprocessor,
    VisionPreprocessor,
    DataClass,
)
from torchtitan.datasets.kiwi.kiwi_dataloader import (
    KiwiDataLoader,
    _pad_batch,
    ensure_modalities,
)
from torchtitan.tools.logging import logger

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast


REPO_ROOT = Path(__file__).resolve().parents[2]
TOKENIZER_PATH = REPO_ROOT / "tests" / "assets" / "tokenizer"

# Budget must fit the fattest fixture sample (~950).
TOKEN_BUDGET = 1024
TEXT_TOKENS = 64
TEXT_MARK_BASE = 100_000
PATCH_SIZE = 16
IMAGE_SIZES = ((256, 384), (384, 256), (512, 256), (256, 512))


def write_dummy_tokenizer(save_dir: str, extra_vocab: list[str] | None = None) -> str:
    vocab = {"<unk>": 0, "<pad>": 1, "<s>": 2, "</s>": 3, "<mask>": 4}
    for i, tok in enumerate(extra_vocab or [], start=len(vocab)):
        vocab.setdefault(tok, i)

    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()

    fast = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        cls_token="<s>",
        sep_token="</s>",
        mask_token="<mask>",
    )

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(save_dir)
    return save_dir


@dataclass(frozen=True)
class KiwiFixture:
    root: Path
    semantic_ids: frozenset[int]
    gen_ids: frozenset[int]
    text_ids: frozenset[int]
    semantic_list: Path
    gen_list: Path
    text_list: Path


def _write_tar(path: Path, members: dict[str, object]) -> None:
    with tarfile.open(path, "w") as tar:
        for name, payload in members.items():
            buffer = io.BytesIO()
            torch.save(payload, buffer)
            raw = buffer.getvalue()
            info = tarfile.TarInfo(name=f"{name}.pt")
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))


def build_kiwi_fixture(
    root: Path, *, n_tars: int = 4, n_ids: int = 8, n_text_shards: int = 4
) -> KiwiFixture:
    """Write semantic-image tars, gen-image (latent) tars, and dummy text parquet."""
    semantic_dir = root / "semantic"
    gen_dir = root / "gen"
    text_dir = root / "text"
    semantic_dir.mkdir()
    gen_dir.mkdir()
    text_dir.mkdir()

    semantic_ids: list[int] = []
    gen_ids: list[int] = []
    semantic_parquets: list[str] = []
    gen_parquets: list[str] = []

    for tar_i in range(n_tars):
        sem_members: dict[str, object] = {}
        gen_members: dict[str, object] = {}
        sem_rows: list[dict[str, object]] = []
        gen_rows: list[dict[str, object]] = []
        sem_tar = semantic_dir / f"sem_{tar_i}.tar"
        gen_tar = gen_dir / f"gen_{tar_i}.tar"

        for local_i in range(n_ids):
            sample_id = tar_i * n_ids + local_i + 1
            height, width = IMAGE_SIZES[sample_id % len(IMAGE_SIZES)]
            latent_h, latent_w = height // PATCH_SIZE, width // PATCH_SIZE

            image = torch.full(
                (1, 3, height, width), float(sample_id), dtype=torch.float32
            )
            sem_members[str(sample_id)] = {"image": image}
            sem_rows.append(
                {
                    "id": str(sample_id),
                    "tar": str(sem_tar),
                    "caption": f"semantic caption {sample_id}",
                }
            )
            semantic_ids.append(sample_id)

            latent = torch.zeros(1, latent_h, latent_w, 16, dtype=torch.float32)
            latent[..., 0] = float(sample_id)
            gen_members[str(sample_id)] = latent
            gen_rows.append(
                {
                    "id": str(sample_id),
                    "tar": str(gen_tar),
                    "caption": f"gen caption {sample_id}",
                }
            )
            gen_ids.append(sample_id)

        _write_tar(sem_tar, sem_members)
        _write_tar(gen_tar, gen_members)
        sem_parquet = semantic_dir / f"sem_{tar_i}.parquet"
        gen_parquet = gen_dir / f"gen_{tar_i}.parquet"
        pd.DataFrame(sem_rows).to_parquet(sem_parquet, index=False)
        pd.DataFrame(gen_rows).to_parquet(gen_parquet, index=False)
        semantic_parquets.append(str(sem_parquet))
        gen_parquets.append(str(gen_parquet))

    text_ids: list[int] = []
    text_parquets: list[str] = []
    docs_per_shard = max(1, n_ids // 2)
    for shard_i in range(n_text_shards):
        rows: list[dict[str, object]] = []
        for local_i in range(docs_per_shard):
            doc_id = shard_i * docs_per_shard + local_i + 1
            tokens = np.arange(TEXT_TOKENS, dtype=np.int64)
            tokens[0] = TEXT_MARK_BASE + doc_id
            rows.append({"tokens": tokens.tobytes(), "dtype": 2})
            text_ids.append(doc_id)
        path = text_dir / f"text_{shard_i}.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        text_parquets.append(str(path))

    semantic_list = root / "semantic_shards.txt"
    gen_list = root / "gen_shards.txt"
    text_list = root / "text_shards.txt"
    semantic_list.write_text("\n".join(semantic_parquets) + "\n")
    gen_list.write_text("\n".join(gen_parquets) + "\n")
    text_list.write_text("\n".join(text_parquets) + "\n")

    return KiwiFixture(
        root=root,
        semantic_ids=frozenset(semantic_ids),
        gen_ids=frozenset(gen_ids),
        text_ids=frozenset(text_ids),
        semantic_list=semantic_list,
        gen_list=gen_list,
        text_list=text_list,
    )


def _preprocessor_config() -> PreprocessorConfig:
    return PreprocessorConfig(
        tokenizer_path=str(TOKENIZER_PATH),
        template=["text", "semantic_image", "gen_image"],
        patch_size=PATCH_SIZE,
        patch_size_latent=1,
        max_seq_len=TEXT_TOKENS,
        max_size=256,
    )


def _kiwi_loader_config(fixture: KiwiFixture) -> KiwiDataLoader.Config:
    return KiwiDataLoader.Config(
        datasets=[
            DatasetConfig(
                name="text",
                type_of_content="text",
                proportion=1.0,
                text_streams=[
                    TextDataStreamConfig(
                        name="dummy_text",
                        data_file=str(fixture.text_list),
                        proportion=1.0,
                    )
                ],
                num_reader_threads=1,
                prefetch_size=8,
            ),
            DatasetConfig(
                name="semantic",
                type_of_content="image",
                proportion=1.0,
                vision_streams=[
                    VisionDataStreamConfig(
                        name="semantic_images",
                        data_file=str(fixture.semantic_list),
                        proportion=1.0,
                        visual_field_name="tar",
                        visual_type="semantic_image",
                        text_probs={"caption": 1.0},
                    )
                ],
                num_reader_threads=1,
                prefetch_size=8,
            ),
            DatasetConfig(
                name="gen",
                type_of_content="image",
                proportion=1.0,
                vision_streams=[
                    VisionDataStreamConfig(
                        name="gen_latents",
                        data_file=str(fixture.gen_list),
                        proportion=1.0,
                        visual_field_name="tar",
                        visual_type="gen_image",
                        text_probs={"caption": 1.0},
                    )
                ],
                num_reader_threads=1,
                prefetch_size=8,
            ),
        ],
        preprocessor=_preprocessor_config(),
        token_budget=TOKEN_BUDGET,
        packing=PackingConfig(max_banks=8),
        num_packing_workers=1,
        batch_queue_size=4,
        startup_timeout=60.0,
        worker_log_level="WARNING",
    )


def _denorm_siglip_constant(value: float) -> int:
    mean = 0.5 * 255.0
    std = 0.5 * 255.0
    return int(round(value * std + mean))


def _collect_marks(batch: ModelBatchData) -> tuple[set[int], set[int], set[int]]:
    semantic: set[int] = set()
    gen: set[int] = set()
    text: set[int] = set()
    if batch.semantic_visual is not None:
        tokens = batch.semantic_visual.tokens
        for start, end in zip(
            batch.semantic_visual.cu_seqlens[:-1],
            batch.semantic_visual.cu_seqlens[1:],
            strict=True,
        ):
            if end > start:
                semantic.add(_denorm_siglip_constant(float(tokens[int(start), 0])))
    if batch.gen_visual is not None:
        tokens = batch.gen_visual.tokens
        for start, end in zip(
            batch.gen_visual.cu_seqlens[:-1],
            batch.gen_visual.cu_seqlens[1:],
            strict=True,
        ):
            if end > start:
                gen.add(int(round(float(tokens[int(start), 0]))))
    if batch.text is not None:
        marks = batch.text.tokens[
            (batch.text.tokens >= TEXT_MARK_BASE)
            & (batch.text.tokens < TEXT_MARK_BASE + 10_000)
        ]
        text.update((marks - TEXT_MARK_BASE).tolist())
    return semantic, gen, text


def _document_order(sample: SampleClass) -> list[tuple[int, int]]:
    """``(encoder_idx, local_item_idx)`` in the order items were added."""
    dst = sample.dst
    order: list[tuple[int, int]] = []
    for col in range(dst.shape[1]):
        enc = int(torch.where(~torch.isnan(dst[:, col]))[0][0].item())
        local = int(dst[enc, col].item())
        order.append((enc, local))
    return order


def _global_item_index(
    encoder: int, local: int, sample_idx: int, per_sample_counts: list[list[int]]
) -> int:
    grouped_offset = sum(
        sum(counts[enc] for counts in per_sample_counts) for enc in range(encoder)
    )
    within = sum(per_sample_counts[s][encoder] for s in range(sample_idx)) + local
    return grouped_offset + within


def _item_token_mask(item_lengths: list[int], adj: torch.Tensor) -> np.ndarray:
    offsets = np.cumsum([0, *item_lengths])
    n = int(offsets[-1])
    mask = np.zeros((n, n), dtype=bool)
    adj_np = adj.detach().cpu().numpy()
    for i, (a, b) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        for j, (c, d) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
            if adj_np[i, j] == 2:
                mask[a:b, c:d] = True
            elif adj_np[i, j] == 1:
                mask[a:b, c:d] = np.tril(np.ones((b - a, d - c), dtype=bool))
    return mask


def _assert_magi_ranges_match_adjacency(
    item_lengths: list[int], adj: torch.Tensor, magi: MagiRanges
) -> None:
    n = sum(item_lengths)
    expected = torch.from_numpy(_item_token_mask(item_lengths, adj))
    sdpa = make_sdpa_mask(magi.q_ranges, magi.k_ranges, magi.attn_type_map)
    assert magi.seq_len == n
    assert torch.equal(sdpa[:n, :n], expected)


def _encoder_of_grouped_item(batch: BatchClass, item_index: int) -> int:
    cursor = 0
    for encoder, lengths in enumerate(batch.item_length):
        nxt = cursor + len(lengths)
        if item_index < nxt:
            return encoder
        cursor = nxt
    raise IndexError(item_index)


def test_stream_mixer_yields_from_chosen_queue() -> None:
    import queue as queue_mod
    import random as rng_mod
    import threading

    from torchtitan.datasets.kiwi.kiwi_dataloader import KiwiDataLoader
    from torchtitan.datasets.kiwi.packing import StreamMixer

    stop_event = threading.Event()
    filled: queue_mod.Queue[dict[str, int]] = queue_mod.Queue()
    filled.put({"idx": 7})
    key = ("ds", "st")
    mixer = StreamMixer(
        {
            key: KiwiDataLoader.iter_queue(
                filled, stop_event=stop_event, timeout=1.0, name="ds/st"
            )
        },
        {key: 1.0},
        rng=rng_mod.Random(0),
    )
    raw, got = next(iter(mixer))
    assert got == key
    assert raw == {"idx": 7}
    stop_event.set()


def test_stream_mixer_times_out_on_empty_chosen_queue() -> None:
    import queue as queue_mod
    import random as rng_mod
    import threading

    from torchtitan.datasets.kiwi.kiwi_dataloader import KiwiDataLoader
    from torchtitan.datasets.kiwi.packing import StreamMixer

    stop_event = threading.Event()
    empty: queue_mod.Queue[dict[str, int]] = queue_mod.Queue()
    other: queue_mod.Queue[dict[str, int]] = queue_mod.Queue()
    other.put({"idx": 1})
    mixer = StreamMixer(
        {
            ("empty", "st"): KiwiDataLoader.iter_queue(
                empty, stop_event=stop_event, timeout=0.3, name="empty/st"
            ),
            ("full", "st"): KiwiDataLoader.iter_queue(
                other, stop_event=stop_event, timeout=0.3, name="full/st"
            ),
        },
        {("empty", "st"): 1.0, ("full", "st"): 0.0},
        rng=rng_mod.Random(0),
    )
    with pytest.raises(TimeoutError, match="empty/st"):
        next(iter(mixer))
    stop_event.set()


def test_stream_mixer_stops_when_iterator_ends() -> None:
    import random as rng_mod

    from torchtitan.datasets.kiwi.packing import StreamMixer

    def ended() -> Any:
        return
        yield None

    mixer = StreamMixer(
        {("ds", "st"): ended()},
        {("ds", "st"): 1.0},
        rng=rng_mod.Random(0),
    )
    assert list(mixer) == []


def test_kiwi_router_ready_same_process(tmp_path: Path) -> None:
    """Packing logic without spawn: mixer over DataStreams + router."""
    import random as rng_mod

    from torchtitan.datasets.kiwi.kiwi_dataloader import (
        _build_stream,
        _preprocess_raw,
    )
    from torchtitan.datasets.kiwi.packing import BankRouter, StreamMixer, queue_weights
    from torchtitan.datasets.kiwi.preprocessing import SampleRef

    fixture = build_kiwi_fixture(tmp_path)
    config = _kiwi_loader_config(fixture)
    preprocessor = Preprocessor(config.preprocessor)
    content = {dataset.name: dataset.type_of_content for dataset in config.datasets}
    weights = queue_weights(config.datasets)
    streams: dict[tuple[str, str], Any] = {}
    for dataset_config in config.datasets:
        for stream_cfg in dataset_config.streams:
            stream = _build_stream(stream_cfg, dp_rank=0, dp_world_size=1)
            streams[(dataset_config.name, stream_cfg.name)] = stream.iter_samples()
    try:
        mixer = StreamMixer(streams, weights, rng=rng_mod.Random(0))
        router = BankRouter(
            token_budget=TOKEN_BUDGET,
            config=PackingConfig(max_banks=8),
        )
        deadline = time.monotonic() + 30
        n_added = 0
        for (idx, payload), (dataset_name, stream_name) in mixer:
            assert time.monotonic() < deadline, (
                f"router not ready after {n_added} samples, "
                f"banks={[bank.length for bank in router.banks]}"
            )
            assert payload is not None
            sample = _preprocess_raw(
                preprocessor,
                payload,
                type_of_content=content[dataset_name],
            )
            assert sample is not None
            router.add(
                sample,
                dataset_name,
                SampleRef(
                    idx=idx,
                    source_dataset=dataset_name,
                    source_stream=stream_name,
                ),
            )
            n_added += 1
            if router.ready():
                break
        else:
            raise AssertionError("mixer stopped before router was ready")
        refs, batch = router.pop()
        assert refs
        assert 0 < batch.length <= TOKEN_BUDGET
        logger.info(
            f"same-process router ready after {n_added} samples, length={batch.length}"
        )
    finally:
        for iterator in streams.values():
            iterator.close()


def test_kiwi_loader_covers_fixture_and_token_budget(tmp_path: Path) -> None:
    fixture = build_kiwi_fixture(tmp_path)
    n_iters = 128
    seen_semantic: set[int] = set()
    seen_gen: set[int] = set()
    seen_text: set[int] = set()
    step_times: list[float] = []
    ntokens: list[int] = []
    pad_frac: list[float] = []

    with KiwiDataLoader(
        _kiwi_loader_config(fixture), dp_rank=0, dp_world_size=1
    ) as loader:
        iterator = loader.iter_model_batches()
        for step in range(n_iters):
            started = time.perf_counter()
            batch = next(iterator)
            elapsed = time.perf_counter() - started
            if step >= 64:
                step_times.append(elapsed)

            assert isinstance(batch, ModelBatchData)
            n_tok = batch.num_tokens()
            ntokens.append(n_tok)
            assert n_tok == TOKEN_BUDGET, f"step {step}: {n_tok} != {TOKEN_BUDGET}"

            if batch.text is not None:
                pad_frac.append(float((~batch.text.target_mask).float().mean()))
            sem, gen, text = _collect_marks(batch)
            seen_semantic |= sem
            seen_gen |= gen
            seen_text |= text

    logger.info(
        "kiwi loader step time from iter 64: "
        f"mean={sum(step_times) / len(step_times):.4f}s "
        f"max={max(step_times):.4f}s over {len(step_times)} steps"
    )
    logger.info(
        f"kiwi loader tokens: mean={sum(ntokens) / len(ntokens):.1f} "
        f"pad_frac_text_mean={sum(pad_frac) / max(len(pad_frac), 1):.4f}"
    )

    missing_sem = fixture.semantic_ids - seen_semantic
    missing_gen = fixture.gen_ids - seen_gen
    missing_text = fixture.text_ids - seen_text
    assert not missing_sem, f"unseen semantic ids: {sorted(missing_sem)}"
    assert not missing_gen, f"unseen gen ids: {sorted(missing_gen)}"
    assert not missing_text, f"unseen text ids: {sorted(missing_text)}"


def test_kiwi_loader_terminate_does_not_hang(tmp_path: Path) -> None:
    fixture = build_kiwi_fixture(tmp_path)
    loader = KiwiDataLoader(_kiwi_loader_config(fixture), dp_rank=0, dp_world_size=1)
    next(iter(loader))
    started = time.monotonic()
    loader.terminate()
    elapsed = time.monotonic() - started
    assert elapsed < 5.0, f"terminate hung for {elapsed:.2f}s"
    assert all(not process.is_alive() for process in loader._processes)
    assert not loader._io_process.is_alive()
    loader.terminate()


def test_ensure_modalities_pads_only_missing() -> None:
    preprocessor = Preprocessor(_preprocessor_config())
    image = torch.full((1, 3, 256, 384), 7.0, dtype=torch.float32)
    semantic = preprocessor(
        layout="<semantic_image><text>",
        text=["hello semantic"],
        semantic_image=[{"image": image}],
        add_special_tokens=True,
    )
    batch = BatchClass()
    batch.add(semantic)
    assert batch.semantic_visual
    assert not batch.gen_visual

    ensure_modalities(preprocessor, batch, ["text", "semantic_image", "gen_image"])
    assert batch.semantic_visual
    assert len(batch.gen_visual) == 1
    assert batch.gen_visual[0].is_padding
    assert batch.gen_visual[0].mask == 0
    semantic_len = len(batch.semantic_visual)
    ensure_modalities(preprocessor, batch, ["text", "semantic_image", "gen_image"])
    assert len(batch.semantic_visual) == semantic_len
    assert len(batch.gen_visual) == 1


def test_kiwi_batch_masks_positions_and_padding() -> None:
    preprocessor = Preprocessor(_preprocessor_config())
    image = torch.full((1, 3, 256, 384), 7.0, dtype=torch.float32)
    latent = torch.zeros(1, 16, 24, 16, dtype=torch.float32)
    latent[..., 0] = 3.0

    semantic = preprocessor(
        layout="<semantic_image><text>",
        text=["hello semantic"],
        semantic_image=[{"image": image}],
        add_special_tokens=True,
    )
    semantic.type_of_content = "image"
    gen = preprocessor(
        layout="<text><gen_image>",
        text=["hello gen"],
        gen_image=[latent],
        add_special_tokens=True,
    )
    gen.type_of_content = "image"

    samples = [semantic, gen]
    batch = BatchClass()
    for sample in samples:
        batch.add(sample)

    per_sample_counts = [
        [len(sample.item_length[enc]) for enc in range(len(ENCODER_ORDER))]
        for sample in samples
    ]
    text_idx = ENCODER_ORDER.index("text")
    assert batch.item_length[text_idx]
    assert batch.item_length[ENCODER_ORDER.index("semantic_visual")]
    assert batch.item_length[ENCODER_ORDER.index("gen_visual")]

    grouped_lengths = batch.get_ref_item_length()
    grouped_masks = batch.get_ref_mask_types()
    for item_index, mask in enumerate(grouped_masks):
        encoder = _encoder_of_grouped_item(batch, item_index)
        if encoder == text_idx:
            assert mask == 1
        else:
            assert mask == 2

    adj = batch.get_adjacency_matrix()
    assert adj.shape[0] == len(grouped_lengths)

    sample_item_sets: list[set[int]] = []
    document_perm: list[int] = []
    for sample_idx, sample in enumerate(samples):
        indices: list[int] = []
        for encoder, local in _document_order(sample):
            indices.append(
                _global_item_index(encoder, local, sample_idx, per_sample_counts)
            )
        sample_item_sets.append(set(indices))
        document_perm.extend(indices)

    for i, items_i in enumerate(sample_item_sets):
        for j, items_j in enumerate(sample_item_sets):
            if i == j:
                continue
            for a in items_i:
                for b in items_j:
                    assert adj[a, b] == 0, (
                        f"samples {i} and {j} share adj[{a},{b}]={adj[a, b]}"
                    )

    adj_doc = adj[document_perm][:, document_perm]
    cursor = 0
    for sample in samples:
        n_items = sample.dst.shape[1]
        block = adj_doc[cursor : cursor + n_items, cursor : cursor + n_items]
        doc_masks = [
            int(sample.mask_type[encoder][local])
            for encoder, local in _document_order(sample)
        ]
        assert torch.equal(block.diag(), torch.tensor(doc_masks, dtype=block.dtype))
        tril = torch.tril(torch.ones_like(block))
        assert torch.all(block[tril == 0] == 0)
        for row in range(n_items):
            for col in range(row):
                assert block[row, col] == 2
        cursor += n_items

    token_mask = _item_token_mask(grouped_lengths, adj)
    sample_token_sets: list[set[int]] = []
    offsets = np.cumsum([0, *grouped_lengths])
    for items in sample_item_sets:
        tokens: set[int] = set()
        for item in items:
            tokens.update(range(int(offsets[item]), int(offsets[item + 1])))
        sample_token_sets.append(tokens)
    for i, toks_i in enumerate(sample_token_sets):
        for j, toks_j in enumerate(sample_token_sets):
            if i == j:
                continue
            keys = sorted(toks_j)
            if not keys:
                continue
            for query in toks_i:
                if token_mask[query, keys].any():
                    pytest.fail(f"sample {i} attends to sample {j} at query {query}")

    batch.postprocessing()
    assert batch.magi_ranges is not None
    _assert_magi_ranges_match_adjacency(grouped_lengths, adj, batch.magi_ranges)
    model_input = batch.to_model_input()
    _assert_magi_ranges_match_adjacency(grouped_lengths, adj, model_input.magi_ranges)
    assert model_input.semantic_visual is not None
    assert model_input.gen_visual is not None
    assert model_input.text is not None

    for modality in (
        model_input.text,
        model_input.semantic_visual,
        model_input.gen_visual,
    ):
        cu = modality.cu_seqlens
        assert torch.all(cu[1:] >= cu[:-1])
        assert int(cu[-1]) == modality.tokens.shape[0]
        for start, end in zip(cu[:-1].tolist(), cu[1:].tolist(), strict=True):
            positions = modality.position_ids[int(start) : int(end)]
            assert int(positions.min()) == 0

    # ``index`` is document order across modalities, not a per-encoder prefix.
    # Sample 2 is therefore shifted by the full length of sample 1.
    assert min(int(item.index[0]) for item in gen.text) == len(semantic)
    for sample in samples:
        ordered: list[DataClass] = [
            item for encoder in ENCODER_ORDER for item in getattr(sample, encoder)
        ]
        ordered.sort(key=lambda span: int(span.index[0]))
        expected = int(ordered[0].index[0])
        for span in ordered:
            assert torch.equal(
                span.index,
                torch.arange(len(span), dtype=span.index.dtype) + span.index[0],
            )
            assert int(span.index[0]) == expected
            expected += len(span)
        assert expected - int(ordered[0].index[0]) == len(sample)

    pad_budget = batch.length + 128
    pad_start_items = len(batch.item_length[text_idx])
    _pad_batch(preprocessor, batch, pad_budget, [])
    assert len(batch) == pad_budget
    pad_item = batch.text[-1]
    assert pad_item.is_padding
    assert pad_item.mask == 0
    assert not bool(pad_item.target_mask.any())
    assert len(batch.item_length[text_idx]) == pad_start_items + 1

    batch.postprocessing()
    assert batch.magi_ranges is not None
    padded_adj = batch.attention_adjacency
    assert padded_adj is not None
    _assert_magi_ranges_match_adjacency(
        batch.get_ref_item_length(), padded_adj, batch.magi_ranges
    )
    pad_index = len(batch.item_length[text_idx]) - 1
    assert int(padded_adj[pad_index, pad_index].item()) == 0
    off_diag = padded_adj[pad_index].clone()
    off_diag[pad_index] = 0
    assert torch.all(off_diag == 0)


def test_build_kiwi_fixture_layout(tmp_path: Path) -> None:
    fixture = build_kiwi_fixture(tmp_path, n_tars=2, n_ids=2, n_text_shards=2)
    assert len(fixture.semantic_ids) == 4
    assert fixture.semantic_ids == fixture.gen_ids
    assert len(fixture.text_ids) == 2
    assert fixture.semantic_list.is_file()
    markup = pd.read_parquet(fixture.semantic_list.read_text().splitlines()[0])
    assert set(markup.columns) == {"id", "tar", "caption"}


@pytest.fixture(scope="module")
def text_preprocessor(tmp_path_factory: pytest.TempPathFactory) -> TextPreprocessor:
    return TextPreprocessor(
        write_dummy_tokenizer(str(tmp_path_factory.mktemp("tokenizer")))
    )


@pytest.fixture(scope="module")
def vision_preprocessor() -> VisionPreprocessor:
    return VisionPreprocessor(patch_size=1, num_channels=1)


def _pixels(frames: int, h: int, w: int, first: int) -> torch.Tensor:
    return torch.arange(first, first + frames * h * w, dtype=torch.float32).reshape(
        frames, h, w, 1
    )


def _sample(*blocks: tuple[DataModality, DataClass]) -> SampleClass:
    sample = SampleClass()
    for modality, data in blocks:
        sample.add(data, modality)
    return sample


def _batch(*samples: SampleClass) -> BatchClass:
    batch = BatchClass()
    for sample in samples:
        batch.add(sample)
    batch.postprocessing()
    assert batch.magi_ranges is not None
    adjacency = batch.attention_adjacency
    assert adjacency is not None
    _assert_magi_ranges_match_adjacency(
        batch.get_ref_item_length(), adjacency, batch.magi_ranges
    )
    return batch


def _in_modality_order(batch: BatchClass) -> list[DataClass]:
    return [x for modality in ENCODER_ORDER for x in getattr(batch, modality)]


def _in_attention_order(batch: BatchClass) -> list[DataClass]:
    return sorted(_in_modality_order(batch), key=lambda x: int(x.index[0]))


def _adjacency_in_attention_order(batch: BatchClass) -> torch.Tensor:
    adjacency = batch.attention_adjacency
    assert adjacency is not None
    order = torch.tensor([int(x.index[0]) for x in _in_modality_order(batch)]).argsort()
    return adjacency[order][:, order]


def test_consecutive_text_merges_into_one_item(
    text_preprocessor: TextPreprocessor,
) -> None:
    batch = _batch(
        _sample(
            ("text", text_preprocessor(torch.tensor([10, 11, 12]))),
            ("text", text_preprocessor(torch.tensor([13, 14]))),
        )
    )
    attention_order = _in_attention_order(batch)

    assert [x.tokens.tolist() for x in attention_order] == [[10, 11, 12, 13, 14]]
    assert [x.index.tolist() for x in attention_order] == [[0, 1, 2, 3, 4]]
    assert [x.span_ids.tolist() for x in attention_order] == [[0, 0, 0, 1, 1]]
    assert [x.position.tolist() for x in attention_order] == [
        [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]],
    ]

    assert batch.get_ref_item_length() == [5]
    assert batch.attention_adjacency is not None
    assert batch.attention_adjacency.tolist() == [[1]]
    assert _adjacency_in_attention_order(batch).tolist() == [[1]]


def test_text_and_images_in_one_sample(
    text_preprocessor: TextPreprocessor, vision_preprocessor: VisionPreprocessor
) -> None:
    batch = _batch(
        _sample(
            ("text", text_preprocessor(torch.tensor([10, 11]))),
            ("gen_image", vision_preprocessor(_pixels(1, 2, 3, first=100))),
            ("text", text_preprocessor(torch.tensor([12, 13]))),
            ("gen_image", vision_preprocessor(_pixels(1, 2, 3, first=200))),
            ("text", text_preprocessor(torch.tensor([14]))),
        )
    )
    attention_order = _in_attention_order(batch)

    assert [x.tokens.flatten().tolist() for x in attention_order] == [
        [10, 11],
        [100, 101, 102, 103, 104, 105],
        [12, 13],
        [200, 201, 202, 203, 204, 205],
        [14],
    ]
    assert [x.index.tolist() for x in attention_order] == [
        [0, 1],
        [2, 3, 4, 5, 6, 7],
        [8, 9],
        [10, 11, 12, 13, 14, 15],
        [16],
    ]
    assert [x.span_ids.tolist() for x in attention_order] == [
        [0, 0],
        [1, 1, 1, 1, 1, 1],
        [2, 2],
        [3, 3, 3, 3, 3, 3],
        [4],
    ]
    assert [x.position.tolist() for x in attention_order] == [
        [[0, 0, 0], [1, 1, 1]],
        [[2, 2, 2], [2, 2, 3], [2, 2, 4], [2, 3, 2], [2, 3, 3], [2, 3, 4]],
        [[5, 5, 5], [6, 6, 6]],
        [[7, 7, 7], [7, 7, 8], [7, 7, 9], [7, 8, 7], [7, 8, 8], [7, 8, 9]],
        [[10, 10, 10]],
    ]

    assert batch.get_ref_item_length() == [2, 2, 1, 6, 6]
    assert batch.attention_adjacency is not None
    assert batch.attention_adjacency.tolist() == [
        [1, 0, 0, 0, 0],
        [2, 1, 0, 2, 0],
        [2, 2, 1, 2, 2],
        [2, 0, 0, 2, 0],
        [2, 2, 0, 2, 2],
    ]
    assert _adjacency_in_attention_order(batch).tolist() == [
        [1, 0, 0, 0, 0],
        [2, 2, 0, 0, 0],
        [2, 2, 1, 0, 0],
        [2, 2, 2, 2, 0],
        [2, 2, 2, 2, 1],
    ]


def test_samples_in_a_batch_are_isolated(
    text_preprocessor: TextPreprocessor, vision_preprocessor: VisionPreprocessor
) -> None:
    samples = []
    for name, first in [("a", 10), ("b", 20)]:
        sample = SampleClass()
        sample.add(
            text_preprocessor(torch.tensor([first, first + 1])),
            "text",
            properties={f"{name}0"},
        )
        sample.add(
            vision_preprocessor(_pixels(1, 2, 3, first=first * 10)),
            "gen_image",
            properties={f"{name}1"},
        )
        sample.add(
            text_preprocessor(torch.tensor([first + 2])),
            "text",
            properties={f"{name}2"},
        )
        samples.append(sample)
    batch = _batch(*samples)
    attention_order = _in_attention_order(batch)

    assert batch.span_properties == [{"a0"}, {"a1"}, {"a2"}, {"b0"}, {"b1"}, {"b2"}]
    assert [x.tokens.flatten().tolist() for x in attention_order] == [
        [10, 11],
        [100, 101, 102, 103, 104, 105],
        [12],
        [20, 21],
        [200, 201, 202, 203, 204, 205],
        [22],
    ]
    assert [x.index.tolist() for x in attention_order] == [
        [0, 1],
        [2, 3, 4, 5, 6, 7],
        [8],
        [9, 10],
        [11, 12, 13, 14, 15, 16],
        [17],
    ]
    assert [x.span_ids.tolist() for x in attention_order] == [
        [0, 0],
        [1, 1, 1, 1, 1, 1],
        [2],
        [3, 3],
        [4, 4, 4, 4, 4, 4],
        [5],
    ]
    assert [x.position.tolist() for x in attention_order] == [
        [[0, 0, 0], [1, 1, 1]],
        [[2, 2, 2], [2, 2, 3], [2, 2, 4], [2, 3, 2], [2, 3, 3], [2, 3, 4]],
        [[5, 5, 5]],
        [[0, 0, 0], [1, 1, 1]],
        [[2, 2, 2], [2, 2, 3], [2, 2, 4], [2, 3, 2], [2, 3, 3], [2, 3, 4]],
        [[5, 5, 5]],
    ]

    assert batch.get_ref_item_length() == [2, 1, 2, 1, 6, 6]
    assert batch.attention_adjacency is not None
    assert batch.attention_adjacency.tolist() == [
        [1, 0, 0, 0, 0, 0],
        [2, 1, 0, 0, 2, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 2, 1, 0, 2],
        [2, 0, 0, 0, 2, 0],
        [0, 0, 2, 0, 0, 2],
    ]
    assert _adjacency_in_attention_order(batch).tolist() == [
        [1, 0, 0, 0, 0, 0],
        [2, 2, 0, 0, 0, 0],
        [2, 2, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 2, 2, 0],
        [0, 0, 0, 2, 2, 1],
    ]


def test_video_time_axis_sets_next_text_start(
    text_preprocessor: TextPreprocessor, vision_preprocessor: VisionPreprocessor
) -> None:
    batch = _batch(
        _sample(
            ("text", text_preprocessor(torch.tensor([10, 11]))),
            ("gen_video", vision_preprocessor(_pixels(3, 1, 2, first=100))),
            ("text", text_preprocessor(torch.tensor([12]))),
        )
    )
    attention_order = _in_attention_order(batch)

    assert [x.index.tolist() for x in attention_order] == [
        [0, 1],
        [2, 3, 4, 5, 6, 7],
        [8],
    ]
    assert [x.position.tolist() for x in attention_order] == [
        [[0, 0, 0], [1, 1, 1]],
        [[2, 2, 2], [2, 2, 3], [3, 2, 2], [3, 2, 3], [4, 2, 2], [4, 2, 3]],
        [[5, 5, 5]],
    ]


def test_padding_neither_attends_nor_is_attended(
    text_preprocessor: TextPreprocessor,
) -> None:
    batch = _batch(
        _sample(("text", text_preprocessor(torch.tensor([10, 11])))),
        _sample(("text", text_preprocessor.get_padding(3))),
    )

    assert batch.get_ref_item_length() == [2, 3]
    assert batch.attention_adjacency is not None
    assert batch.attention_adjacency.tolist() == [[1, 0], [0, 0]]
    assert _adjacency_in_attention_order(batch).tolist() == [[1, 0], [0, 0]]


def test_model_input_positions_are_global_and_block_local(
    text_preprocessor: TextPreprocessor, vision_preprocessor: VisionPreprocessor
) -> None:
    out = _batch(
        _sample(
            ("text", text_preprocessor(torch.tensor([10, 11]))),
            ("gen_image", vision_preprocessor(_pixels(1, 2, 3, first=100))),
            ("text", text_preprocessor(torch.tensor([12]))),
        )
    ).to_model_input()

    assert out.magi_ranges is not None
    assert out.text is not None
    assert out.gen_visual is not None
    assert out.magi_ranges.seq_len == out.position_ids.shape[0]
    assert out.text.cu_seqlens.tolist() == [0, 2, 3]
    assert out.text.position_ids.tolist() == [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
    assert out.gen_visual.cu_seqlens.tolist() == [0, 6]
    assert out.gen_visual.position_ids.tolist() == [
        [0, 0, 0],
        [0, 0, 1],
        [0, 0, 2],
        [0, 1, 0],
        [0, 1, 1],
        [0, 1, 2],
    ]
    assert out.position_ids.tolist() == [
        [0, 0, 0],
        [1, 1, 1],
        [5, 5, 5],
        [2, 2, 2],
        [2, 2, 3],
        [2, 2, 4],
        [2, 3, 2],
        [2, 3, 3],
        [2, 3, 4],
    ]
