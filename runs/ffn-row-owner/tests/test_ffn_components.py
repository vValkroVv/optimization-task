"""CPU-only native crop factory/CLI contracts; no GPU or fake training evidence."""

import importlib.util
import os
from pathlib import Path

import pytest
import torch

PATH = Path(__file__).resolve().parents[1] / "ffn_components.py"
spec = importlib.util.spec_from_file_location("ffn_components", PATH)
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


@pytest.mark.parametrize("component", ["layer0", "mtp"])
def test_native_meta_crop_contract(component):
    assert not torch.cuda.is_available(), "run with CUDA_VISIBLE_DEVICES=''"
    previous_world = os.environ.get("WORLD_SIZE")
    cfg = harness.resolved_config(component, True)
    assert os.environ.get("WORLD_SIZE") == previous_world
    original_layers = cfg.model_spec.model.placed_layers()
    model = harness.meta_component(cfg, component)
    result = harness.contract(cfg, model, component)
    assert result["shape"] == [2, 4096, 4096]
    assert result["segments"] == [
        "layers.0" if component == "layer0" else "mtp.layers.0"
    ]
    assert all(name.endswith("FP8Linear") for name in result["ffn_types"].values())
    assert all(
        p.device.type == "meta" and p.dtype == torch.float32 for p in model.parameters()
    )
    assert len(original_layers) == len(cfg.model_spec.model.placed_layers()) == 8
    assert cfg.parallelism.gfsdp_split_layer0_ffn == (component == "layer0")
    assert cfg.parallelism.gfsdp_split_mtp_ffn == (component == "mtp")
    assert (
        cfg.parallelism.expert_parallel_degree
        == cfg.parallelism.data_parallel_shard_degree
        == 8
    )
    assert model.tok_embeddings is model.norm is model.lm_head is None
    if component == "mtp":
        assert not model.layers
        assert len(model.mtp.layers) == 1


def test_error_report_preserves_element_evidence():
    expected = {"grad": torch.tensor([0.0, 2.0, 4.0])}
    same = harness.errors(expected, expected)["grad"]
    changed = harness.errors(expected, {"grad": torch.tensor([0.0, 2.0, 5.0])})["grad"]
    assert same["unequal"] == same["max_abs"] == same["nonfinite"] == 0
    assert changed["unequal"] == changed["max_abs"] == 1
    assert changed["relative_l2"] == pytest.approx(1 / 20**0.5)
    with pytest.raises(RuntimeError, match="coverage"):
        harness.errors(expected, {})


@pytest.mark.parametrize(
    "flags",
    [
        ["--mode", "diagnostic", "--variant", "control"],
        ["--trace-microbatches", "4"],
        ["--warmup", "0"],
        ["--accumulation", "2"],
    ],
)
def test_cli_rejects_invalid_or_unbounded_capture(flags):
    with pytest.raises(SystemExit):
        harness.arguments(["--component", "mtp", "--output", "unused", *flags])


def test_compiler_cache_cli_reuses_existing_rank_dirs_without_fallback(
    monkeypatch, tmp_path
):
    allowed_root = tmp_path / "authorized-ffn-runs"
    allowed_root.mkdir()
    monkeypatch.setattr(harness, "COMPILER_CACHE_AREA", allowed_root)
    cache_root = allowed_root / "producer" / "cache"
    for _, suffix in harness.COMPILER_CACHE_DIRS:
        (cache_root / "rank3" / suffix).mkdir(parents=True)

    output = tmp_path / "reuse-output"
    args = harness.arguments(
        [
            "--component",
            "layer0",
            "--mode",
            "benchmark",
            "--output",
            str(output),
            "--compiler-cache-root",
            str(cache_root),
        ]
    )
    selected = harness.configure_compiler_caches(
        args.output, rank=3, compiler_cache_root=args.compiler_cache_root
    )
    assert selected["policy"] == "reuse_existing"
    assert selected["root"] == str(cache_root.resolve())
    assert not (output / "cache").exists()
    for env, suffix in harness.COMPILER_CACHE_DIRS:
        expected = str((cache_root / "rank3" / suffix).resolve())
        assert selected["directories"][env] == expected
        assert os.environ[env] == expected

    with pytest.raises(FileNotFoundError, match="compiler cache directory is missing"):
        harness.configure_compiler_caches(
            output, rank=0, compiler_cache_root=cache_root
        )
    assert not (output / "cache").exists()

    outside_root = tmp_path / "outside-cache"
    outside_root.mkdir()
    with pytest.raises(ValueError, match="inside this task's FFN artifact area"):
        harness.configure_compiler_caches(
            output, rank=3, compiler_cache_root=outside_root
        )

    fresh_output = tmp_path / "fresh-output"
    fresh = harness.configure_compiler_caches(fresh_output, rank=2)
    assert fresh["policy"] == "fresh_per_output"
    for env, suffix in harness.COMPILER_CACHE_DIRS:
        expected = str((fresh_output / "cache" / "rank2" / suffix).resolve())
        assert fresh["directories"][env] == expected
        assert os.environ[env] == expected


def test_captured_microbatch_passes_exact_masks_positions_embedding_and_upstream():
    hidden = torch.randn(2, 4, 3).requires_grad_()
    embedding = torch.randn(2, 4, 3).requires_grad_()
    upstream = torch.randn_like(hidden)
    positions = torch.tensor([[0, 1, 0, 1], [0, 0, 1, 2]])
    masks = {"captured": torch.tensor([True, False])}
    token_mask = positions != 0

    class Model:
        def __call__(self, h, e, **kwargs):
            assert h is hidden and e is embedding
            assert kwargs["positions"] is positions
            assert kwargs["attention_masks"] is masks
            assert kwargs["token_mask"] is token_mask
            return h * 2 + e * 3

    result = harness.microbatch(Model(), "mtp", (hidden, embedding, upstream, positions, masks, token_mask))
    assert torch.equal(result, hidden * 2 + embedding * 3)
    assert torch.equal(hidden.grad, upstream * 2)
    assert torch.equal(embedding.grad, upstream * 3)


def test_fixture_cli_is_explicit_and_numeric_campaign_is_rejected():
    args = harness.arguments(["--component", "layer0", "--mode", "benchmark", "--output", "unused", "--input-fixtures", "captured"])
    assert args.input_fixtures == Path("captured")
    assert harness.arguments(["--component", "mtp", "--output", "unused"]).input_fixtures is None
    with pytest.raises(SystemExit):
        harness.arguments(["--component", "mtp", "--output", "unused", "--input-fixtures", "captured", "--mode", "check"])
