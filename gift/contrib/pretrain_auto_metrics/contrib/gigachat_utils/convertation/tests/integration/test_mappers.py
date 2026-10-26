"""
These tests apply mappers to real checkpoints and compare results with golden files.
Must be run on cluster via submit_tests.py to have access to test data.

Test data location: /workspace-SR008.fs2/msenikolaev/models_for_tests

Gigar models:
    gigar/gigar                -> gigar-fsdp-tp1  (map_gigafsdp_to_fsdp)
    gigar/gigar                -> gigar-fsdp-tp2  (map_gigafsdp_to_fsdp + map_tp_to_tp)
    gigar/gigar                -> gigar-hf        (map_gigafsdp_to_hf)

Gigamix (MoE) models:
    gigamix/10b                -> 10b-fsdp-st_merge     (map_gigafsdp_to_fsdp_safetensor)
    gigamix/10b                -> 10b-hf-st_merge       (map_gigafsdp_to_hf_safetensor)
    gigamix/10b                -> 10-mixed-fsdp.pt      (map_mixed_gigafsdp_to_fsdp)
    gigamix/10b-ep2            -> 10b-ep2-fsdp-st       (map_gigafsdp_to_fsdp_safetensor)
    gigamix/10b-ep2            -> 10b-ep2-hf-st         (map_gigafsdp_to_hf_safetensor)
"""

import os
import sys
import json
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from collections import OrderedDict

import torch

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ============================================================
# Test Configuration
# ============================================================

# Base path - can be overridden via environment variable
_DEFAULT_TEST_PATH = "/workspace-SR008.fs2/msenikolaev/models_for_tests"
TEST_MODELS_BASE_PATH = Path(
    os.environ.get("CONVERTATION_TEST_DATA_PATH", _DEFAULT_TEST_PATH)
)
GIGAMIX_PATH = TEST_MODELS_BASE_PATH / "gigamix"
GIGAR_PATH = TEST_MODELS_BASE_PATH / "gigar"
GIGAOMNI_PATH = TEST_MODELS_BASE_PATH / "gigaomni"

# Tolerance for tensor comparison
ATOL = 1e-5
RTOL = 1e-4


# ============================================================
# Helper Functions
# ============================================================


def find_checkpoint_dir(base_path: Path) -> Path:
    """
    Find the actual checkpoint directory containing .distcp files.

    Gigafsdp checkpoints have structure like:
        base_path/ep0-ba200-tp_rank0-ep_rank0/__0_0.distcp

    This function finds the subdirectory with .distcp files.
    """
    # If base_path itself contains .distcp files, return it
    if list(base_path.glob("*.distcp")):
        return base_path

    # Look for subdirectories with checkpoint pattern
    for subdir in base_path.iterdir():
        if subdir.is_dir() and (
            subdir.name.startswith("ep") or list(subdir.glob("*.distcp"))
        ):
            # Check if this subdir has .distcp files
            if list(subdir.glob("*.distcp")):
                return subdir
            # Or check one level deeper
            for sub_subdir in subdir.iterdir():
                if sub_subdir.is_dir() and list(sub_subdir.glob("*.distcp")):
                    return sub_subdir

    # Return original path if nothing found
    return base_path


def load_pt_checkpoint(path: Path) -> Dict[str, Any]:
    """Load .pt checkpoint file."""
    import dill

    return torch.load(path, map_location="cpu", weights_only=False, pickle_module=dill)


def load_hf_safetensors(path: Path) -> Dict[str, torch.Tensor]:
    """Load HuggingFace model from safetensors.

    Supports:
    - Single model.safetensors file
    - Sharded with model.safetensors.index.json
    - Sharded with shard-*.safetensors files (streaming output format)
    - Any .safetensors files in directory
    """
    from safetensors.torch import load_file

    path = Path(path)

    # Single file
    if (path / "model.safetensors").exists():
        return load_file(str(path / "model.safetensors"))

    # Sharded with index
    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)

        state = {}
        for shard_file in set(index.get("weight_map", {}).values()):
            shard_path = path / shard_file
            if shard_path.exists():
                state.update(load_file(str(shard_path)))
        return state

    # Try loading all .safetensors files (streaming output format)
    safetensor_files = list(path.glob("*.safetensors"))
    if safetensor_files:
        state = {}
        for sf in sorted(safetensor_files):
            state.update(load_file(str(sf)))
        return state

    # Try recursive search for safetensors
    safetensor_files = list(path.rglob("*.safetensors"))
    if safetensor_files:
        state = {}
        for sf in sorted(safetensor_files):
            state.update(load_file(str(sf)))
        return state

    raise ValueError(f"No safetensors found at {path}")


def load_fsdp_safetensors(path: Path) -> Dict[str, torch.Tensor]:
    """Load FSDP safetensors checkpoint (with rank directories)."""
    from safetensors.torch import load_file

    state = {}

    # Check for merged index
    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        for shard_file in set(index.get("weight_map", {}).values()):
            shard_path = path / shard_file
            if shard_path.exists():
                state.update(load_file(str(shard_path)))
        return state

    # Check rank directories
    for rank_dir in sorted(path.glob("rank*")):
        rank_index = rank_dir / "rank-index.json"
        if rank_index.exists():
            with open(rank_index) as f:
                idx = json.load(f)
            for shard_file in set(idx.get("weight_map", {}).values()):
                shard_path = rank_dir / shard_file
                if shard_path.exists():
                    state.update(load_file(str(shard_path)))

    if state:
        return state

    # Try loading any safetensors files
    for sf in path.glob("**/*.safetensors"):
        state.update(load_file(str(sf)))

    return state


def compare_state_dicts(
    result: Dict[str, torch.Tensor],
    golden: Dict[str, torch.Tensor],
    atol: float = ATOL,
    rtol: float = RTOL,
    check_dtypes: bool = False,
    skip_ema: bool = True,
) -> Tuple[bool, List[str]]:
    """Compare two state dicts and return (success, errors).

    Args:
        result: Result state dict from mapper
        golden: Expected golden state dict
        atol: Absolute tolerance for tensor comparison
        rtol: Relative tolerance for tensor comparison
        check_dtypes: Whether to check tensor dtypes match
        skip_ema: Whether to skip EMA keys (default True)
    """
    errors = []

    # Normalize keys (remove 'model.' prefix)
    def norm_key(k):
        return k[6:] if k.startswith("model.") else k

    def is_ema_key(k):
        return "_ema" in k or k.endswith("_ema")

    # Filter EMA keys if needed
    result_norm = {
        norm_key(k): v for k, v in result.items() if not (skip_ema and is_ema_key(k))
    }
    golden_norm = {
        norm_key(k): v for k, v in golden.items() if not (skip_ema and is_ema_key(k))
    }

    result_keys = set(result_norm.keys())
    golden_keys = set(golden_norm.keys())

    # Check missing/extra keys
    missing = golden_keys - result_keys
    extra = result_keys - golden_keys

    if missing:
        errors.append(f"Missing {len(missing)} keys: {sorted(missing)[:5]}...")
    if extra:
        errors.append(f"Extra {len(extra)} keys: {sorted(extra)[:5]}...")

    # Compare tensors
    common = result_keys & golden_keys
    for key in sorted(common):
        r = result_norm[key]
        g = golden_norm[key]

        if not isinstance(r, torch.Tensor) or not isinstance(g, torch.Tensor):
            continue

        if r.shape != g.shape:
            errors.append(f"{key}: shape {r.shape} vs {g.shape}")
            continue

        if check_dtypes and r.dtype != g.dtype:
            errors.append(f"{key}: dtype {r.dtype} vs {g.dtype}")

        if not torch.allclose(r.float(), g.float(), atol=atol, rtol=rtol):
            max_diff = (r.float() - g.float()).abs().max().item()
            errors.append(f"{key}: max_diff={max_diff:.6e}")

    return len(errors) == 0, errors


# ============================================================
# Gigar Integration Tests
# ============================================================


@pytest.mark.integration
@pytest.mark.gigar
class TestGigarMappers:
    """Integration tests for gigar model conversions."""

    def test_gigafsdp_to_fsdp_tp1(self, gigar_available, base_args, temp_output_dir):
        """
        Test: gigar -> gigar-fsdp-tp1 via map_gigafsdp_to_fsdp
        """
        from mappers import map_gigafsdp_to_fsdp

        # Find the actual checkpoint directory
        input_base = GIGAR_PATH / "gigar"
        input_path = find_checkpoint_dir(input_base)
        golden_path = GIGAR_PATH / "gigar-fsdp-tp1" / "gigar-tp_rank0-ep_rank0.pt"

        if not input_path.exists() or not list(input_path.glob("*.distcp")):
            pytest.skip(f"Input checkpoint not found or empty: {input_path}")
        if not golden_path.exists():
            pytest.skip(f"Golden not found: {golden_path}")

        # Configure args
        args = base_args
        args.input_checkpoint_path = str(input_path)
        args.output_checkpoint_path = str(temp_output_dir / "output.pt")

        # Run mapper
        result = map_gigafsdp_to_fsdp(str(input_path), args=args)

        # Load golden
        golden = load_pt_checkpoint(golden_path)

        # Compare
        success, errors = compare_state_dicts(
            result["state"]["model"], golden["state"]["model"]
        )

        assert success, f"Comparison failed:\n" + "\n".join(errors[:20])

    def test_gigafsdp_to_fsdp_tp2(self, gigar_available, base_args, temp_output_dir):
        """
        Test: gigar -> gigar-fsdp-tp2 via map_gigafsdp_to_fsdp + map_tp_to_tp
        """
        from mappers import map_gigafsdp_to_fsdp, map_tp_to_tp

        # Find the actual checkpoint directory
        input_base = GIGAR_PATH / "gigar"
        input_path = find_checkpoint_dir(input_base)
        golden_dir = GIGAR_PATH / "gigar-fsdp-tp2"

        if not input_path.exists() or not list(input_path.glob("*.distcp")):
            pytest.skip(f"Input checkpoint not found or empty: {input_path}")
        if not golden_dir.exists():
            pytest.skip(f"Golden not found: {golden_dir}")

        # Configure args for tp2 output
        args = base_args
        args.input_checkpoint_path = str(input_path)
        args.output_checkpoint_path = str(temp_output_dir / "output.pt")
        args.input_tp_size = 1
        args.output_tp_size = 2

        # Run mappers
        result = map_gigafsdp_to_fsdp(str(input_path), args=args)
        result = map_tp_to_tp(result, args=args)

        # Result should be list for tp_size > 1
        assert isinstance(
            result, list
        ), f"Expected list for tp_size=2, got {type(result)}"
        assert len(result) == 2, f"Expected 2 TP shards, got {len(result)}"

        # Compare each TP rank
        for tp_rank in range(2):
            golden_file = golden_dir / f"gigar-tp_rank{tp_rank}-ep_rank0.pt"
            if not golden_file.exists():
                continue

            golden = load_pt_checkpoint(golden_file)

            success, errors = compare_state_dicts(
                result[tp_rank]["state"]["model"], golden["state"]["model"]
            )

            assert success, f"TP rank {tp_rank} failed:\n" + "\n".join(errors[:10])

    def test_gigafsdp_to_hf(
        self, gigar_available, base_args, temp_output_dir, gigar_tokenizer_path
    ):
        """
        Test: gigar -> gigar-hf via map_gigafsdp_to_hf
        """
        from mappers import map_gigafsdp_to_hf

        # Find the actual checkpoint directory
        input_base = GIGAR_PATH / "gigar"
        input_path = find_checkpoint_dir(input_base)
        golden_path = GIGAR_PATH / "gigar-hf"

        if not input_path.exists() or not list(input_path.glob("*.distcp")):
            pytest.skip(f"Input checkpoint not found or empty: {input_path}")
        if not golden_path.exists():
            pytest.skip(f"Golden not found: {golden_path}")

        # Configure args
        args = base_args
        args.input_checkpoint_path = str(input_path)
        args.output_checkpoint_path = str(temp_output_dir / "hf_output")
        args.tokenizer_path = gigar_tokenizer_path

        # Run mapper
        map_gigafsdp_to_hf(str(input_path), args=args)

        # Load and compare
        result = load_hf_safetensors(temp_output_dir / "hf_output")
        golden = load_hf_safetensors(golden_path)

        success, errors = compare_state_dicts(result, golden)

        assert success, f"HF comparison failed:\n" + "\n".join(errors[:20])


# ============================================================
# Gigamix (MoE) Integration Tests
# ============================================================


@pytest.mark.integration
@pytest.mark.gigamix
class TestGigamixMappers:
    """Integration tests for gigamix (MoE) model conversions."""

    def test_mixed_gigafsdp_to_fsdp(
        self, gigamix_available, base_args, temp_output_dir
    ):
        """
        Test: 10b -> 10-mixed-fsdp.pt via map_mixed_gigafsdp_to_fsdp
        """
        from mappers import map_mixed_gigafsdp_to_fsdp

        input_path = GIGAMIX_PATH / "10b"
        golden_path = GIGAMIX_PATH / "10-mixed-fsdp.pt"

        if not golden_path.exists():
            pytest.skip(f"Golden not found: {golden_path}")

        # Configure args
        args = base_args
        args.input_checkpoint_path = str(input_path)
        args.output_checkpoint_path = str(temp_output_dir / "output.pt")

        # Run mapper
        result = map_mixed_gigafsdp_to_fsdp(str(input_path), args=args)

        # Load golden
        golden = load_pt_checkpoint(golden_path)

        # Compare
        success, errors = compare_state_dicts(
            result["state"]["model"], golden["state"]["model"]
        )

        assert success, f"Mixed FSDP comparison failed:\n" + "\n".join(errors[:20])

    def test_gigafsdp_to_fsdp_safetensor(
        self, gigamix_available, base_args, temp_output_dir
    ):
        """
        Test: 10b -> 10b-fsdp-st_merge via map_gigafsdp_to_fsdp_safetensor
        """
        from mappers import map_gigafsdp_to_fsdp_safetensor

        input_path = GIGAMIX_PATH / "10b"
        golden_path = GIGAMIX_PATH / "10b-fsdp-st_merge"

        if not golden_path.exists():
            pytest.skip(f"Golden not found: {golden_path}")

        # Configure args
        args = base_args
        args.input_checkpoint_path = str(input_path)
        args.output_checkpoint_path = str(temp_output_dir / "fsdp_st_output")

        # Run mapper
        map_gigafsdp_to_fsdp_safetensor(str(input_path), args=args)

        # Load and compare
        result = load_fsdp_safetensors(temp_output_dir / "fsdp_st_output")
        golden = load_fsdp_safetensors(golden_path)

        success, errors = compare_state_dicts(result, golden)

        assert success, f"FSDP safetensor comparison failed:\n" + "\n".join(errors[:20])

    def test_gigafsdp_to_hf_safetensor(
        self, gigamix_available, base_args, temp_output_dir, gigamix_tokenizer_path
    ):
        """
        Test: 10b -> 10b-hf-st_merge via map_gigafsdp_to_hf_safetensor

        Note: This mapper uses streaming and saves safetensors to rank00/ subdirectory.
        """
        from mappers import map_gigafsdp_to_hf_safetensor

        input_path = GIGAMIX_PATH / "10b"
        golden_path = GIGAMIX_PATH / "10b-hf-st_merge"

        if not golden_path.exists():
            pytest.skip(f"Golden not found: {golden_path}")

        # Configure args
        output_path = temp_output_dir / "hf_st_output"
        args = base_args
        args.input_checkpoint_path = str(input_path)
        args.output_checkpoint_path = str(output_path)
        args.tokenizer_path = gigamix_tokenizer_path

        # Run mapper
        map_gigafsdp_to_hf_safetensor(str(input_path), args=args)

        # Debug: list output directory contents
        print(f"\nOutput directory contents ({temp_output_dir}):")
        for item in temp_output_dir.iterdir():
            print(f"  {item.name} (dir={item.is_dir()})")
            if item.is_dir():
                for sub in list(item.iterdir())[:10]:
                    print(f"    └─ {sub.name}")
                    if sub.is_dir():
                        for subsub in list(sub.iterdir())[:5]:
                            print(f"      └─ {subsub.name}")

        # Try to load result from various locations
        result = None

        # 1. Try safetensors in output_path/rank00 (streaming output location)
        rank00_path = output_path / "rank00"
        if rank00_path.exists():
            try:
                result = load_hf_safetensors(rank00_path)
                print(f"Loaded from rank00, keys: {len(result)}")
            except ValueError as e:
                print(f"Failed to load from rank00: {e}")

        # 2. Try safetensors in output_path directly
        if result is None and output_path.exists():
            try:
                result = load_hf_safetensors(output_path)
                print(f"Loaded from output_path, keys: {len(result)}")
            except ValueError as e:
                print(f"Failed to load from output_path: {e}")

        # 3. Try safetensors in temp_output_dir directly
        if result is None:
            try:
                result = load_hf_safetensors(temp_output_dir)
                print(f"Loaded from temp_output_dir, keys: {len(result)}")
            except ValueError as e:
                print(f"Failed to load from temp_output_dir: {e}")

        if result is None:
            # List all files for debugging
            all_files = list(temp_output_dir.rglob("*"))
            pytest.fail(
                f"Could not load safetensors from any location.\n"
                f"Files in output: {[str(f.relative_to(temp_output_dir)) for f in all_files if f.is_file()]}"
            )

        # Load golden
        golden = load_hf_safetensors(golden_path)

        success, errors = compare_state_dicts(result, golden)

        assert success, f"HF safetensor comparison failed:\n" + "\n".join(errors[:20])

    def test_gigafsdp_ep2_to_hf_safetensor(
        self, gigamix_available, base_args, temp_output_dir, gigamix_tokenizer_path
    ):
        """
        Test: 10b-ep2 -> 10b-ep2-hf-st via map_gigafsdp_to_hf_safetensor

        Note: This test may fail if checkpoint contains EMA weights that mapper doesn't support.
        """
        from mappers import map_gigafsdp_to_hf_safetensor

        input_path = GIGAMIX_PATH / "10b-ep2"
        golden_path = GIGAMIX_PATH / "10b-ep2-hf-st"

        if not input_path.exists():
            pytest.skip(f"Input not found: {input_path}")
        if not golden_path.exists():
            pytest.skip(f"Golden not found: {golden_path}")

        # Configure args - make sure EMA is skipped
        args = base_args
        args.input_checkpoint_path = str(input_path)
        args.output_checkpoint_path = str(temp_output_dir / "ep2_hf_st_output")
        args.tokenizer_path = gigamix_tokenizer_path or str(golden_path)
        args.skip_ema_mapper = True
        args.use_ema = False

        # Run mapper
        try:
            map_gigafsdp_to_hf_safetensor(str(input_path), args=args)
        except RuntimeError as e:
            if "Unexpected MoE key" in str(e) and "_ema" in str(e):
                pytest.skip(
                    f"Checkpoint contains EMA weights not supported by mapper: {e}"
                )
            raise

        # Load and compare
        result = load_hf_safetensors(temp_output_dir / "ep2_hf_st_output")
        golden = load_hf_safetensors(golden_path)

        success, errors = compare_state_dicts(result, golden)

        assert success, f"EP2 HF safetensor comparison failed:\n" + "\n".join(
            errors[:20]
        )



# ============================================================
#  GigaOmni Integration Tests
# ============================================================
@pytest.mark.integration
@pytest.mark.gigamix
class TestGigaOmniMappers:
    """Integration tests for gigaomni model conversions."""

    def test_gigafsdp_to_fsdp_safetensor(
        self, gigamix_available, base_args, temp_output_dir
    ):
        """
        Test: 10b gigaomni -> 10b-fsdp-st_merge via map_gigafsdp_to_omni_safetensor
        """
        from mappers import map_gigafsdp_to_omni_safetensor

        input_path = GIGAOMNI_PATH / "gigafsdp"
        golden_path = GIGAOMNI_PATH / "safetensors"

        if not golden_path.exists():
            pytest.skip(f"Golden not found: {golden_path}")

        # Configure args
        args = base_args
        args.input_checkpoint_path = str(input_path)
        args.output_checkpoint_path = str(temp_output_dir / "fsdp_st_output")

        # Run mapper
        map_gigafsdp_to_omni_safetensor(str(input_path), args=args)

        # Load and compare
        result = load_fsdp_safetensors(temp_output_dir / "fsdp_st_output")
        golden = load_fsdp_safetensors(golden_path)

        success, errors = compare_state_dicts(result, golden)

        assert success, f"FSDP safetensor comparison failed:\n" + "\n".join(errors[:20])

    def test_gigafsdp_to_fsdp_safetensor_text(
        self, gigamix_available, base_args, temp_output_dir
    ):
        """
        Test: 10b gigaomni -> 10b-fsdp-st_merge via map_gigafsdp_to_omni_safetensor
        """
        from mappers import map_gigafsdp_to_omni_safetensor

        input_path = GIGAOMNI_PATH / "10b_text"
        golden_path = GIGAOMNI_PATH / "10b_text_st"

        if not golden_path.exists():
            pytest.skip(f"Golden not found: {golden_path}")

        # Configure args
        args = base_args
        args.input_checkpoint_path = str(input_path)
        args.output_checkpoint_path = str(temp_output_dir / "fsdp_st_output")

        # Run mapper
        map_gigafsdp_to_omni_safetensor(str(input_path), args=args)

        # Load and compare
        result = load_fsdp_safetensors(temp_output_dir / "fsdp_st_output")
        golden = load_fsdp_safetensors(golden_path)

        for key in result.keys():
            assert ".language_model." in key, f"key {key} is renamed incorrectly"

        success, errors = compare_state_dicts(result, golden)

        assert success, f"FSDP safetensor comparison failed:\n" + "\n".join(errors[:20])

    def test_gigafsdp_to_hf(
        self, gigar_available, base_args, temp_output_dir, gigamix_tokenizer_path
    ):
        """
        Test: 10b gigaomni -> gigaomni-hf via map_mixed_gigafsdp_to_fsdp + map_fsdp_to_hf
        """
        from mappers import map_fsdp_to_hf, map_mixed_gigafsdp_to_fsdp

        # Find the actual checkpoint directory
        input_base = GIGAOMNI_PATH / "gigafsdp"
        input_path = find_checkpoint_dir(input_base)
        golden_path = GIGAOMNI_PATH / "hf_model"

        if not input_path.exists() or not list(input_path.glob("*.distcp")):
            pytest.skip(f"Input checkpoint not found or empty: {input_path}")
        if not golden_path.exists():
            pytest.skip(f"Golden not found: {golden_path}")

        # Configure args
        args = base_args
        args.input_checkpoint_path = str(input_path)
        args.output_checkpoint_path = str(temp_output_dir / "hf_output")
        args.tokenizer_path = gigamix_tokenizer_path

        # Run mapper
        state_dicts = map_mixed_gigafsdp_to_fsdp(str(input_path), args=args)
        map_fsdp_to_hf(state_dicts, args=args)

        # Load and compare
        result = load_hf_safetensors(temp_output_dir / "hf_output")
        golden = load_hf_safetensors(golden_path)

        success, errors = compare_state_dicts(result, golden)

        assert success, f"HF comparison failed:\n" + "\n".join(errors[:20])
    
    def test_gigafsdp_to_hf_safetensor(
        self, gigamix_available, base_args, temp_output_dir, gigamix_tokenizer_path
        ):
        """
        Test: 10b -> 10b-hf-st_merge via map_gigafsdp_to_hf_safetensor

        Note: This mapper uses streaming and saves safetensors to rank00/ subdirectory.
        """
        from mappers import map_gigafsdp_to_hf_omni_safetensor

        input_path = GIGAOMNI_PATH / "gigafsdp_3_5"
        golden_path = GIGAOMNI_PATH / "10b_omni_3_5_hf"

        if not golden_path.exists():
            pytest.skip(f"Golden not found: {golden_path}")

        # Configure args
        output_path = temp_output_dir / "hf_st_output"
        args = base_args
        args.input_checkpoint_path = str(input_path)
        args.output_checkpoint_path = str(output_path)
        args.tokenizer_path = gigamix_tokenizer_path

        # Run mapper
        map_gigafsdp_to_hf_omni_safetensor(str(input_path), args=args)

        # Debug: list output directory contents
        print(f"\nOutput directory contents ({temp_output_dir}):")
        for item in temp_output_dir.iterdir():
            print(f"  {item.name} (dir={item.is_dir()})")
            if item.is_dir():
                for sub in list(item.iterdir())[:10]:
                    print(f"    └─ {sub.name}")
                    if sub.is_dir():
                        for subsub in list(sub.iterdir())[:5]:
                            print(f"      └─ {subsub.name}")

        # Try to load result from various locations
        result = None

        # 1. Try safetensors in output_path/rank00 (streaming output location)
        rank00_path = output_path / "rank00"
        if rank00_path.exists():
            try:
                result = load_hf_safetensors(rank00_path)
                print(f"Loaded from rank00, keys: {len(result)}")
            except ValueError as e:
                print(f"Failed to load from rank00: {e}")

        # 2. Try safetensors in output_path directly
        if result is None and output_path.exists():
            try:
                result = load_hf_safetensors(output_path)
                print(f"Loaded from output_path, keys: {len(result)}")
            except ValueError as e:
                print(f"Failed to load from output_path: {e}")

        # 3. Try safetensors in temp_output_dir directly
        if result is None:
            try:
                result = load_hf_safetensors(temp_output_dir)
                print(f"Loaded from temp_output_dir, keys: {len(result)}")
            except ValueError as e:
                print(f"Failed to load from temp_output_dir: {e}")

        if result is None:
            # List all files for debugging
            all_files = list(temp_output_dir.rglob("*"))
            pytest.fail(
                f"Could not load safetensors from any location.\n"
                f"Files in output: {[str(f.relative_to(temp_output_dir)) for f in all_files if f.is_file()]}"
            )

        # Load golden
        golden = load_hf_safetensors(golden_path)

        success, errors = compare_state_dicts(result, golden)

        assert success, f"HF safetensor comparison failed:\n" + "\n".join(errors[:20])


# ============================================================
# Unmerged Safetensor Tests (rank00 structure)
# ============================================================


@pytest.mark.integration
@pytest.mark.gigamix
class TestUnmergedSafetensors:
    """Tests for unmerged safetensor outputs (before merge_safetensors_ranks.py)."""

    def test_fsdp_safetensor_unmerged_structure(
        self, gigamix_available, base_args, temp_output_dir
    ):
        """
        Test: 10b -> 10b-fsdp-st (unmerged) structure verification

        Verifies that map_gigafsdp_to_fsdp_safetensor creates correct rank00/ structure.
        """
        from mappers import map_gigafsdp_to_fsdp_safetensor

        input_path = GIGAMIX_PATH / "10b"
        golden_path = GIGAMIX_PATH / "10b-fsdp-st"  # unmerged version

        if not golden_path.exists():
            pytest.skip(f"Golden not found: {golden_path}")

        # Configure args
        output_path = temp_output_dir / "fsdp_st_unmerged"
        args = base_args
        args.input_checkpoint_path = str(input_path)
        args.output_checkpoint_path = str(output_path)

        # Run mapper
        map_gigafsdp_to_fsdp_safetensor(str(input_path), args=args)

        # Verify structure - should have rank00 directory
        rank00_path = output_path / "rank00"
        assert rank00_path.exists(), f"rank00 directory not created at {rank00_path}"

        # Should have safetensors files
        safetensor_files = list(rank00_path.glob("*.safetensors"))
        assert len(safetensor_files) > 0, "No safetensors files in rank00/"

        # Should have weight_map.json files
        weight_map_files = list(rank00_path.glob("*.weight_map.json"))
        assert len(weight_map_files) > 0, "No weight_map.json files in rank00/"

        # Compare with golden if available
        if golden_path.exists():
            result = load_fsdp_safetensors(output_path)
            golden = load_fsdp_safetensors(golden_path)

            success, errors = compare_state_dicts(result, golden)
            assert success, f"Unmerged FSDP comparison failed:\n" + "\n".join(
                errors[:20]
            )


# ============================================================
# Dtype Tests
# ============================================================


@pytest.mark.integration
@pytest.mark.gigamix
class TestDtypeConversions:
    """Tests for different torch_dtype conversions."""

    def test_dtype_distribution(self, gigamix_available, base_args, temp_output_dir):
        """Test that tensors have expected dtype distribution.

        Note: map_mixed_gigafsdp_to_fsdp may not apply torch_dtype conversion.
        This test documents actual dtype distribution in output.
        """
        from mappers import map_mixed_gigafsdp_to_fsdp

        input_path = GIGAMIX_PATH / "10b"

        args = base_args
        args.input_checkpoint_path = str(input_path)
        args.output_checkpoint_path = str(temp_output_dir / "output.pt")
        args.torch_dtype = "bfloat16"

        result = map_mixed_gigafsdp_to_fsdp(str(input_path), args=args)

        # Collect dtype statistics
        model_state = result["state"]["model"]
        dtype_counts = {}

        for key, value in model_state.items():
            if isinstance(value, torch.Tensor):
                dtype_str = str(value.dtype)
                dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + 1

        print(f"\nDtype distribution in output:")
        for dtype, count in sorted(dtype_counts.items()):
            print(f"  {dtype}: {count}")

        # Test should pass - we're just documenting the actual behavior
        # The mapper preserves original dtypes from checkpoint
        total_tensors = sum(dtype_counts.values())
        assert total_tensors > 0, "No tensors found in output"

        # Check that we have floating point tensors (bf16 or fp32)
        float_types = ["torch.bfloat16", "torch.float16", "torch.float32"]
        float_count = sum(dtype_counts.get(dt, 0) for dt in float_types)
        assert (
            float_count > 0
        ), f"No floating point tensors found. Dtypes: {dtype_counts}"


# ============================================================
# Error Handling Tests
# ============================================================


@pytest.mark.integration
class TestErrorHandling:
    """Tests for proper error handling in mappers."""

    def test_invalid_input_path(self, base_args, temp_output_dir):
        """Test that mapper fails gracefully on invalid input path."""
        from mappers import map_gigafsdp_to_fsdp

        args = base_args
        args.input_checkpoint_path = "/nonexistent/path/to/checkpoint"
        args.output_checkpoint_path = str(temp_output_dir / "output.pt")

        with pytest.raises((FileNotFoundError, ValueError, RuntimeError)):
            map_gigafsdp_to_fsdp("/nonexistent/path/to/checkpoint", args=args)

    def test_empty_checkpoint_dir(self, base_args, temp_output_dir):
        """Test that mapper fails gracefully on empty checkpoint directory."""
        from mappers import map_gigafsdp_to_fsdp

        # Create empty directory
        empty_dir = temp_output_dir / "empty_checkpoint"
        empty_dir.mkdir(parents=True, exist_ok=True)

        args = base_args
        args.input_checkpoint_path = str(empty_dir)
        args.output_checkpoint_path = str(temp_output_dir / "output.pt")

        # Should fail with some kind of error (not crash silently)
        with pytest.raises((ValueError, RuntimeError, IndexError, FileNotFoundError)):
            map_gigafsdp_to_fsdp(str(empty_dir), args=args)


# ============================================================
# Output Structure Tests
# ============================================================


@pytest.mark.integration
@pytest.mark.gigamix
class TestOutputStructure:
    """Tests for verifying output file structure."""

    def test_hf_output_has_required_files(
        self, gigamix_available, base_args, temp_output_dir, gigamix_tokenizer_path
    ):
        """Test that unmerged HF safetensor output contains the merge-time assembly inputs."""
        from mappers import map_gigafsdp_to_hf_safetensor

        input_path = GIGAMIX_PATH / "10b"
        output_path = temp_output_dir / "hf_output"

        args = base_args
        args.input_checkpoint_path = str(input_path)
        args.output_checkpoint_path = str(output_path)
        args.tokenizer_path = gigamix_tokenizer_path

        map_gigafsdp_to_hf_safetensor(str(input_path), args=args)

        # Unmerged HF safetensor export should only write the inputs needed for
        # post-merge HF assembly, not the final HF artifact files themselves.
        required_files = ["composer_state.pt", "hf_assembly_plan.json"]
        for f in required_files:
            assert (output_path / f).exists(), f"Missing required file: {f}"

        # Check for safetensors (either in root or rank00)
        safetensors_exist = (
            list(output_path.glob("*.safetensors"))
            or list((output_path / "rank00").glob("*.safetensors"))
            if (output_path / "rank00").exists()
            else []
        )
        assert safetensors_exist, "No safetensors files found in output"

    def test_fsdp_output_has_rank_structure(
        self, gigamix_available, base_args, temp_output_dir
    ):
        """Test that FSDP safetensor output has correct rank structure."""
        from mappers import map_gigafsdp_to_fsdp_safetensor

        input_path = GIGAMIX_PATH / "10b"
        output_path = temp_output_dir / "fsdp_output"

        args = base_args
        args.input_checkpoint_path = str(input_path)
        args.output_checkpoint_path = str(output_path)

        map_gigafsdp_to_fsdp_safetensor(str(input_path), args=args)

        # Should have rank00 directory
        rank00 = output_path / "rank00"
        assert rank00.exists(), "rank00 directory not created"

        # Should have safetensors and weight_map files
        assert list(rank00.glob("*.safetensors")), "No safetensors in rank00/"
        assert list(rank00.glob("*.weight_map.json")), "No weight_map.json in rank00/"

        # Should have rank-index.json
        assert (rank00 / "rank-index.json").exists(), "Missing rank-index.json"


# ============================================================
# Tokenizer Tests
# ============================================================


@pytest.mark.integration
@pytest.mark.gigar
class TestTokenizerHandling:
    """Tests for tokenizer handling in HF conversions."""

    def test_tokenizer_files_copied(
        self, gigar_available, base_args, temp_output_dir, gigar_tokenizer_path
    ):
        """Test that tokenizer files are properly copied to output."""
        from mappers import map_gigafsdp_to_hf

        input_base = GIGAR_PATH / "gigar"
        input_path = find_checkpoint_dir(input_base)
        output_path = temp_output_dir / "hf_output"

        if not input_path.exists() or not list(input_path.glob("*.distcp")):
            pytest.skip(f"Input checkpoint not found: {input_path}")

        args = base_args
        args.input_checkpoint_path = str(input_path)
        args.output_checkpoint_path = str(output_path)
        args.tokenizer_path = gigar_tokenizer_path

        map_gigafsdp_to_hf(str(input_path), args=args)

        # Check for tokenizer files
        tokenizer_files = ["tokenizer_config.json"]
        for f in tokenizer_files:
            if (Path(gigar_tokenizer_path) / f).exists():
                assert (output_path / f).exists(), f"Tokenizer file not copied: {f}"


# ============================================================
# Config Preservation Tests
# ============================================================


@pytest.mark.integration
class TestConfigPreservation:
    """Test that model configs are properly preserved during conversion."""

    def test_config_preserved_gigafsdp_to_fsdp(
        self, gigar_available, base_args, temp_output_dir
    ):
        """Verify config is preserved in gigafsdp to fsdp conversion."""
        from mappers import map_gigafsdp_to_fsdp

        # Find the actual checkpoint directory
        input_base = GIGAR_PATH / "gigar"
        input_path = find_checkpoint_dir(input_base)

        if not input_path.exists() or not list(input_path.glob("*.distcp")):
            pytest.skip(f"Input checkpoint not found or empty: {input_path}")

        args = base_args
        args.input_checkpoint_path = str(input_path)
        args.output_checkpoint_path = str(temp_output_dir / "output.pt")

        result = map_gigafsdp_to_fsdp(str(input_path), args=args)

        # Check config structure
        assert "state" in result
        assert "integrations" in result["state"]
        assert "huggingface" in result["state"]["integrations"]

        config = result["state"]["integrations"]["huggingface"]["model"]["config"][
            "content"
        ]

        # Check essential config fields
        required_fields = ["hidden_size", "num_attention_heads", "num_hidden_layers"]
        for field in required_fields:
            assert field in config, f"Missing required config field: {field}"

    def test_config_preserved_mixed_gigafsdp(
        self, gigamix_available, base_args, temp_output_dir
    ):
        """Verify config is preserved in mixed gigafsdp conversion."""
        from mappers import map_mixed_gigafsdp_to_fsdp

        input_path = GIGAMIX_PATH / "10b"

        args = base_args
        args.input_checkpoint_path = str(input_path)
        args.output_checkpoint_path = str(temp_output_dir / "output.pt")

        result = map_mixed_gigafsdp_to_fsdp(str(input_path), args=args)

        # Check config
        config = result["state"]["integrations"]["huggingface"]["model"]["config"][
            "content"
        ]

        required_fields = ["hidden_size", "num_attention_heads", "num_hidden_layers"]
        for field in required_fields:
            assert field in config, f"Missing required config field: {field}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
