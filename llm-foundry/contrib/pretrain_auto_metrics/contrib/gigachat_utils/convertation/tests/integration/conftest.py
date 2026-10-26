import pytest
import sys
import os
import tempfile
import shutil
from pathlib import Path
from argparse import Namespace

_DEFAULT_TEST_PATH = "/workspace-SR008.fs2/msenikolaev/models_for_tests"
TEST_MODELS_BASE_PATH = Path(
    os.environ.get("CONVERTATION_TEST_DATA_PATH", _DEFAULT_TEST_PATH)
)
GIGAMIX_PATH = TEST_MODELS_BASE_PATH / "gigamix"
GIGAR_PATH = TEST_MODELS_BASE_PATH / "gigar"
GOLDENS_PATH = TEST_MODELS_BASE_PATH / "goldens"


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "golden: mark test as using golden files")
    config.addinivalue_line("markers", "slow: mark test as slow-running")
    config.addinivalue_line("markers", "gigar: mark test for gigar models")
    config.addinivalue_line("markers", "gigamix: mark test for gigamix (MoE) models")

    # Debug: print paths at startup
    print(f"\n{'=' * 60}")
    print(f"INTEGRATION TESTS DEBUG INFO")
    print(f"{'=' * 60}")
    print(f"TEST_MODELS_BASE_PATH: {TEST_MODELS_BASE_PATH}")
    print(f"  exists: {TEST_MODELS_BASE_PATH.exists()}")
    print(f"GIGAR_PATH: {GIGAR_PATH}")
    print(f"  exists: {GIGAR_PATH.exists()}")
    print(f"GIGAMIX_PATH: {GIGAMIX_PATH}")
    print(f"  exists: {GIGAMIX_PATH.exists()}")

    # List contents if paths exist
    if GIGAR_PATH.exists():
        print(f"\nContents of {GIGAR_PATH}:")
        for item in sorted(GIGAR_PATH.iterdir()):
            print(f"  {item.name} (dir={item.is_dir()})")
            if item.is_dir():
                # Show first level subdirs
                subdirs = list(item.iterdir())[:5]
                for sub in subdirs:
                    print(f"    └─ {sub.name}")
                if len(list(item.iterdir())) > 5:
                    print(f"    └─ ... and {len(list(item.iterdir())) - 5} more")

    if GIGAMIX_PATH.exists():
        print(f"\nContents of {GIGAMIX_PATH}:")
        for item in sorted(GIGAMIX_PATH.iterdir()):
            print(f"  {item.name} (dir={item.is_dir()})")
            if item.is_dir():
                subdirs = list(item.iterdir())[:5]
                for sub in subdirs:
                    print(f"    └─ {sub.name}")
                if len(list(item.iterdir())) > 5:
                    print(f"    └─ ... and {len(list(item.iterdir())) - 5} more")

    print(f"{'=' * 60}\n")


def pytest_collection_modifyitems(config, items):
    """Add markers based on test names."""
    for item in items:
        # Auto-add markers based on test class/function names
        if "Gigar" in item.nodeid:
            item.add_marker(pytest.mark.gigar)
        if "Gigamix" in item.nodeid:
            item.add_marker(pytest.mark.gigamix)
        if "golden" in item.nodeid.lower():
            item.add_marker(pytest.mark.golden)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture(scope="session")
def test_models_base_path():
    """Return the base path for test models."""
    return TEST_MODELS_BASE_PATH


@pytest.fixture(scope="session")
def gigamix_path():
    """Return path to gigamix test models."""
    return GIGAMIX_PATH


@pytest.fixture(scope="session")
def gigar_path():
    """Return path to gigar test models."""
    return GIGAR_PATH


@pytest.fixture(scope="session")
def goldens_path():
    """Return path to golden files."""
    return GOLDENS_PATH


@pytest.fixture(scope="module")
def test_models_available():
    """Check if test models are available. Skip all tests in module if not."""
    if not TEST_MODELS_BASE_PATH.exists():
        # List what we tried
        msg = (
            f"Test models not found at {TEST_MODELS_BASE_PATH}. "
            f"Integration tests must be run on cluster with access to test data. "
            f"Current working dir: {os.getcwd()}"
        )
        pytest.skip(msg)
    return True


@pytest.fixture(scope="module")
def gigar_available(test_models_available):
    """Check if gigar test models are available."""
    if not GIGAR_PATH.exists():
        pytest.skip(f"Gigar test models not found at {GIGAR_PATH}")

    # Check for required subdirectories
    required = ["gigar", "gigar-fsdp-tp1", "gigar-hf"]
    missing = [r for r in required if not (GIGAR_PATH / r).exists()]
    if missing:
        available = (
            [p.name for p in GIGAR_PATH.iterdir()] if GIGAR_PATH.exists() else []
        )
        pytest.skip(f"Missing gigar test data: {missing}. " f"Available: {available}")

    return True


@pytest.fixture(scope="module")
def gigamix_available(test_models_available):
    """Check if gigamix test models are available."""
    if not GIGAMIX_PATH.exists():
        pytest.skip(f"Gigamix test models not found at {GIGAMIX_PATH}")

    # Check for required files/directories
    required = ["10b"]
    missing = [r for r in required if not (GIGAMIX_PATH / r).exists()]
    if missing:
        available = (
            [p.name for p in GIGAMIX_PATH.iterdir()] if GIGAMIX_PATH.exists() else []
        )
        pytest.skip(f"Missing gigamix test data: {missing}. " f"Available: {available}")

    return True


@pytest.fixture(scope="module")
def goldens_available(test_models_available):
    """Check if golden files are available."""
    if not GOLDENS_PATH.exists():
        pytest.skip(
            f"Golden files not found at {GOLDENS_PATH}. "
            "Run generate_golden_files.py to create them."
        )
    return True


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory that is cleaned up after test."""
    tmpdir = tempfile.mkdtemp(prefix="integration_test_")
    yield Path(tmpdir)
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def base_args():
    """Create base args object with all necessary attributes for mappers."""
    args = Namespace()

    # Basic conversion args
    args.skip_optimizer = True
    args.skip_ema_mapper = True
    args.use_ema = False
    args.prepare_ema_weight = False
    args.print_final_model_structure = False

    # TP/EP args
    args.input_tp_size = 1
    args.output_tp_size = 1
    args.input_ep_size = 1
    args.output_ep_size = 1
    args.input_tp_embedding_type = "EmbeddingParallelEmbedding"
    args.output_tp_embedding_type = "EmbeddingParallelEmbedding"

    # Dtype and tokenizer
    args.torch_dtype = "bfloat16"
    args.tokenizer_path = None

    # Safetensor streaming args
    args.chunk_layers = 2
    args.global_rank = 0
    args.world_size = 1

    # Other args
    args.embeddings_expand_size = None
    args.model_type = None
    args.hf_llm_onnx = False
    args.save_full_model = False

    return args


@pytest.fixture
def gigar_tokenizer_path(gigar_available):
    """Return path to gigar tokenizer."""
    tok_path = GIGAR_PATH / "gigar-hf"
    if tok_path.exists():
        return str(tok_path)
    return None


@pytest.fixture
def gigamix_tokenizer_path(gigamix_available):
    """Return path to gigamix tokenizer."""
    candidates = [
        GIGAMIX_PATH / "10b-hf-st_merge",
        GIGAMIX_PATH / "10b-ep2-hf-st",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None
