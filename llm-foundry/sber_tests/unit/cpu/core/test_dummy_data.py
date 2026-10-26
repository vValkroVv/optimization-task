import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf as om
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[4]


class TokenizerStub:
    def __init__(
        self,
        bos_token_id=None,
        eos_token_id=None,
        pad_token_id=0,
        vocab_size=128256,
        tokenizer_size=None,
    ):
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size
        self.tokenizer_size = vocab_size if tokenizer_size is None else tokenizer_size

    def __len__(self):
        return self.tokenizer_size

    @property
    def all_special_ids(self):
        return [
            token_id
            for token_id in (self.bos_token_id, self.eos_token_id, self.pad_token_id)
            if token_id is not None
        ]

    def pad(self, encoded_inputs, return_tensors=None, pad_to_multiple_of=None):
        del return_tensors, pad_to_multiple_of
        max_len = max(sample.size(0) for sample in encoded_inputs)
        padded = [
            torch.nn.functional.pad(
                sample,
                (0, max_len - sample.size(0)),
                value=self.pad_token_id,
            )
            for sample in encoded_inputs
        ]
        return {"input_ids": torch.stack(padded)}


def _load_module(module_name: str, file_path: Path):
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module {module_name} from {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _load_dummy_data_module():
    return _load_module(
        "llmfoundry.data.dummy_data_for_test",
        REPO_ROOT / "llmfoundry/data/dummy_data.py",
    )


def _load_text_data_module():
    for contrib_path in (
        REPO_ROOT / "contrib/composer",
        REPO_ROOT / "contrib/streaming",
        REPO_ROOT / "contrib/gigafsdp",
    ):
        contrib_path_str = str(contrib_path)
        if contrib_path_str not in sys.path:
            sys.path.insert(0, contrib_path_str)

    llmfoundry_pkg = sys.modules.setdefault("llmfoundry", types.ModuleType("llmfoundry"))
    llmfoundry_pkg.__path__ = [str(REPO_ROOT / "llmfoundry")]

    data_pkg = sys.modules.setdefault("llmfoundry.data", types.ModuleType("llmfoundry.data"))
    data_pkg.__path__ = [str(REPO_ROOT / "llmfoundry/data")]

    _load_module("llmfoundry.data.data", REPO_ROOT / "llmfoundry/data/data.py")
    _load_module("llmfoundry.data.fim", REPO_ROOT / "llmfoundry/data/fim.py")
    _load_module("llmfoundry.data.dummy_data", REPO_ROOT / "llmfoundry/data/dummy_data.py")
    return _load_module("llmfoundry.data.text_data", REPO_ROOT / "llmfoundry/data/text_data.py")


class TestDummyData:
    @pytest.mark.unit
    @pytest.mark.fast
    def test_dummy_dataset_uses_fallback_special_tokens(self):
        dummy_data = _load_dummy_data_module()
        dataset = dummy_data.DummyDataset(
            max_seq_len=8,
            dataset_size=2,
            tokenizer=TokenizerStub(),
        )

        sample = next(iter(dataset))

        assert sample.shape == torch.Size([8])
        assert sample[0].item() == 1
        assert sample[-1].item() == 2

    @pytest.mark.unit
    @pytest.mark.fast
    def test_dummy_dataset_supports_high_special_token_ids(self):
        dummy_data = _load_dummy_data_module()
        tokenizer = TokenizerStub(
            bos_token_id=30,
            eos_token_id=31,
            vocab_size=32,
        )
        dataset = dummy_data.DummyDataset(
            max_seq_len=8,
            dataset_size=2,
            tokenizer=tokenizer,
        )

        sample = next(iter(dataset))

        assert sample[0].item() == 30
        assert sample[-1].item() == 31
        assert 30 not in sample[1:-1].tolist()
        assert 31 not in sample[1:-1].tolist()

    @pytest.mark.unit
    @pytest.mark.fast
    def test_dummy_dataset_excludes_pad_and_special_token_ids(self):
        dummy_data = _load_dummy_data_module()
        tokenizer = TokenizerStub(
            bos_token_id=28,
            eos_token_id=29,
            pad_token_id=30,
            vocab_size=32,
        )
        dataset = dummy_data.DummyDataset(
            max_seq_len=16,
            dataset_size=4,
            tokenizer=tokenizer,
        )

        sample = next(iter(dataset))

        assert 28 not in sample[1:-1].tolist()
        assert 29 not in sample[1:-1].tolist()
        assert 30 not in sample[1:-1].tolist()

    @pytest.mark.unit
    @pytest.mark.fast
    def test_dummy_dataset_uses_tokenizer_length_for_special_tokens(self):
        dummy_data = _load_dummy_data_module()
        tokenizer = TokenizerStub(
            bos_token_id=34,
            eos_token_id=35,
            vocab_size=32,
            tokenizer_size=36,
        )
        dataset = dummy_data.DummyDataset(
            max_seq_len=8,
            dataset_size=2,
            tokenizer=tokenizer,
        )

        sample = next(iter(dataset))

        assert sample[0].item() == 34
        assert sample[-1].item() == 35
        assert sample.max().item() < 36

    @pytest.mark.unit
    @pytest.mark.fast
    def test_dummy_dataset_shards_iteration_across_workers(self):
        dummy_data = _load_dummy_data_module()
        dataset = dummy_data.DummyDataset(
            max_seq_len=8,
            dataset_size=5,
            tokenizer=TokenizerStub(),
        )

        loader = DataLoader(dataset, batch_size=None, num_workers=2)
        samples = list(loader)

        assert len(samples) == 5
        assert all(sample.shape == torch.Size([8]) for sample in samples)

    @pytest.mark.unit
    @pytest.mark.fast
    def test_dummy_dataset_shards_iteration_across_distributed_ranks(
        self,
        monkeypatch,
    ):
        dummy_data = _load_dummy_data_module()
        recorded_seeds = []

        def fake_default_rng(seed):
            recorded_seeds.append(seed)
            return SimpleNamespace(
                integers=lambda low, high, size: np.zeros(size, dtype=np.int64)
            )

        monkeypatch.setattr(dummy_data.np.random, "default_rng", fake_default_rng)
        monkeypatch.setenv("WORLD_SIZE", "2")

        monkeypatch.setenv("RANK", "0")
        rank0_dataset = dummy_data.DummyDataset(
            max_seq_len=8,
            dataset_size=5,
            tokenizer=TokenizerStub(),
        )
        rank0_samples = list(rank0_dataset)

        monkeypatch.setenv("RANK", "1")
        rank1_dataset = dummy_data.DummyDataset(
            max_seq_len=8,
            dataset_size=5,
            tokenizer=TokenizerStub(),
        )
        rank1_samples = list(rank1_dataset)

        assert len(rank0_samples) == 3
        assert len(rank1_samples) == 2
        assert recorded_seeds == [42, 43]

    @pytest.mark.unit
    @pytest.mark.fast
    def test_dummy_text_dataloader_builds_without_special_tokens(self):
        text_data = _load_text_data_module()
        cfg = om.create(
            {
                "name": "text",
                "dataset": {
                    "use_dummy": True,
                    "max_seq_len": 8,
                    "dataset_size": 4,
                },
                "drop_last": False,
                "num_workers": 0,
                "pin_memory": False,
                "prefetch_factor": None,
                "persistent_workers": False,
                "enable_data_profiler": False,
            }
        )

        tokenizer = TokenizerStub(
            bos_token_id=34,
            eos_token_id=35,
            vocab_size=32,
            tokenizer_size=36,
        )
        loader = text_data.build_text_dataloader(cfg, tokenizer, device_batch_size=2)
        batch = next(iter(loader))

        assert batch["input_ids"].shape == torch.Size([2, 8])
        assert batch["labels"].shape == torch.Size([2, 8])
        assert batch["input_ids"].dtype == torch.long

    @pytest.mark.unit
    @pytest.mark.fast
    def test_dummy_text_dataloader_handles_zero_workers_defaults(self):
        text_data = _load_text_data_module()
        cfg = om.create(
            {
                "name": "text",
                "dataset": {
                    "use_dummy": True,
                    "max_seq_len": 8,
                    "dataset_size": 4,
                },
                "drop_last": False,
                "num_workers": 0,
                "pin_memory": False,
                "enable_data_profiler": False,
            }
        )

        tokenizer = TokenizerStub(
            bos_token_id=34,
            eos_token_id=35,
            vocab_size=32,
            tokenizer_size=36,
        )
        loader = text_data.build_text_dataloader(cfg, tokenizer, device_batch_size=2)
        batch = next(iter(loader))

        assert batch["input_ids"].shape == torch.Size([2, 8])

    @pytest.mark.unit
    @pytest.mark.fast
    def test_dummy_text_dataloader_builds_without_tokenizer(self):
        text_data = _load_text_data_module()
        cfg = om.create(
            {
                "name": "text",
                "dataset": {
                    "use_dummy": True,
                    "max_seq_len": 8,
                    "dataset_size": 4,
                },
                "drop_last": False,
                "num_workers": 0,
                "pin_memory": False,
                "enable_data_profiler": False,
            }
        )

        loader = text_data.build_text_dataloader(cfg, None, device_batch_size=2)
        batch = next(iter(loader))

        assert batch["input_ids"].shape == torch.Size([2, 8])
        assert batch["labels"].shape == torch.Size([2, 8])
        assert batch["input_ids"].dtype == torch.long
