import json
import os
import sys
from pathlib import Path

import pytest

# Make `import sglang` resolve to the in-repo copy under python/, which carries
# the gigachat3.5 runtime support (e.g. the Gigachat35ForCausalLMNextN special
# cases). A stale system-installed sglang would otherwise raise
# "PP is not compatible with MTP models." when EAGLE speculative decoding runs.
_SGLANG_PYTHON = Path(__file__).resolve().parents[2] / "python"
if str(_SGLANG_PYTHON) not in sys.path:
    sys.path.insert(0, str(_SGLANG_PYTHON))

# gigachat3.5 HF checkpoints used by the integration tests. Each test runs once
# per checkpoint (single MTP head + multi-MTP). Override the base directory with
# the GIGACHAT35_TEST_CHECKPOINT_DIR env var (submit_tests.py sets it on cluster).
_DEFAULT_GIGACHAT35_CHECKPOINT_DIR = (
    "/workspace-SR008.nfs2/models/models_for_tests/sglang_pretrain/"
    "3b_3_5_arch_5000steps_4ba_ep8_tp1_sp1"
)
GIGACHAT35_CHECKPOINT_DIR = Path(
    os.environ.get(
        "GIGACHAT35_TEST_CHECKPOINT_DIR", _DEFAULT_GIGACHAT35_CHECKPOINT_DIR
    )
)
# {test id: subdirectory name}
GIGACHAT35_CHECKPOINTS = {
    "original": "original",
    "multi_mtp": "multi_mtp",
    "dense": "dense",
}


@pytest.fixture(scope="session", params=list(GIGACHAT35_CHECKPOINTS), ids=list(GIGACHAT35_CHECKPOINTS))
def gigachat35_checkpoint(request):
    """Path to a gigachat3.5 HF checkpoint; skips if it is absent.

    Parametrized so every test that uses it runs once per checkpoint.
    """
    checkpoint = GIGACHAT35_CHECKPOINT_DIR / GIGACHAT35_CHECKPOINTS[request.param]
    if not checkpoint.exists():
        pytest.skip(
            f"gigachat3.5 checkpoint not found at {checkpoint}. "
            "Integration tests must run on a cluster node with access to the "
            "checkpoints (or set GIGACHAT35_TEST_CHECKPOINT_DIR)."
        )
    if not (checkpoint / "config.json").exists():
        pytest.skip(f"{checkpoint} has no config.json — not an HF checkpoint")
    return checkpoint


@pytest.fixture(scope="session")
def gigachat35_config(gigachat35_checkpoint):
    """Parsed config.json of the gigachat3.5 checkpoint."""
    return json.loads((gigachat35_checkpoint / "config.json").read_text())


@pytest.fixture(scope="session")
def gigachat35_num_mtp_layers(gigachat35_config):
    """Number of built-in MTP heads (num_nextn_predict_layers) in the checkpoint."""
    return int(gigachat35_config.get("num_nextn_predict_layers", 0))


# GigaVision HF checkpoint used by the integration tests. Override the path with
# GIGAVISION_TEST_CHECKPOINT_DIR (submit_tests.py sets it on cluster).
_DEFAULT_GIGAVISION_CHECKPOINT_DIR = (
    "/home/jovyan/models/10b_vision_test_no_delete"
)
GIGAVISION_CHECKPOINT_DIR = Path(
    os.environ.get(
        "GIGAVISION_TEST_CHECKPOINT_DIR", _DEFAULT_GIGAVISION_CHECKPOINT_DIR
    )
)


@pytest.fixture(scope="session")
def gigavision_checkpoint():
    """Path to a GigaVision HF checkpoint; skips if it is absent."""
    checkpoint = GIGAVISION_CHECKPOINT_DIR
    if not checkpoint.exists():
        pytest.skip(
            f"GigaVision checkpoint not found at {checkpoint}. "
            "Integration tests must run on a cluster node with access to the "
            "checkpoint, or set GIGAVISION_TEST_CHECKPOINT_DIR."
        )
    if not (checkpoint / "config.json").exists():
        pytest.skip(f"{checkpoint} has no config.json — not an HF checkpoint")
    if not (checkpoint / "processors/GigaVisionProcessor").is_dir():
        pytest.skip(
            f"{checkpoint} has no processors/GigaVisionProcessor directory"
        )
    return checkpoint


_DEFAULT_GIGAOMNI_CHECKPOINT_DIR = (
    "/workspace-SR008.nfs2/models/models_for_tests/sglang_pretrain/"
    "260825143000__10b_3_1_chatml_v4_1__3__va_aa_ve_ae_llm__continuous_pretrain_v2_1__from_050426140000/"
)

_GIGAOMNI_IMAGE_PATH = (
    Path(__file__).parent / "fixtures" / "gigaomni" / "test_image_1.jpg"
)

_GIGAOMNI_AUDIO_PATH = (
    Path(__file__).parent / "fixtures" / "gigaomni" / "test_audio_1.wav"
)

@pytest.fixture(scope="session")
def gigaomni_checkpoint():
    checkpoint = Path(
        os.environ.get("GIGAOMNI_TEST_CHECKPOINT_DIR", _DEFAULT_GIGAOMNI_CHECKPOINT_DIR)
    )
    if not checkpoint.exists():
        pytest.skip(f"GigaOmni checkpoint not found at {checkpoint}")
    if not (checkpoint / "config.json").exists():
        pytest.skip(f"{checkpoint} has no config.json")
    if not (checkpoint / "modalities").is_dir():
        pytest.skip(f"{checkpoint} has no modalities/ directory")
    return checkpoint


@pytest.fixture(scope="session")
def gigaomni_image_path():
    assert _GIGAOMNI_IMAGE_PATH.exists(), f"missing fixture: {_GIGAOMNI_IMAGE_PATH}"
    return _GIGAOMNI_IMAGE_PATH


@pytest.fixture(scope="session")
def gigaomni_audio_path():
    assert _GIGAOMNI_AUDIO_PATH.exists(), f"missing fixture: {_GIGAOMNI_AUDIO_PATH}"
    return _GIGAOMNI_AUDIO_PATH


@pytest.fixture(scope="session")
def gigaomni_tokenizer(gigaomni_checkpoint):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(gigaomni_checkpoint))
