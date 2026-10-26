"""Tests for checkpoint conversion utilities."""

import json
from pathlib import Path
from typing import Any

import node_runner.src.convertation_manager as convertation_manager_module
import pytest
from node_runner.src.convertation_manager import ConvertationUtils

pytestmark = pytest.mark.unit

# ============================================================================
# Constants for test configuration
# ============================================================================

CHECKPOINT_DIR_NAME = "checkpoint"
CONFIG_FILENAME = "config.json"
MODEL_TYPE_FIELD = "model_type"
MODEL_SAFETENSORS_FILENAME = "model.safetensors"
MODEL_BIN_FILENAME = "pytorch_model.bin"
TOKENIZER_JSON_FILENAME = "tokenizer.json"
TOKENIZER_CONFIG_FILENAME = "tokenizer_config.json"
TEST_MODEL_TYPE = "test-model"
TEST_DATA_DIR = "test_data"
HF_CHECKPOINTS_DIR = "hf_chekps"
KIMI_K2_MODEL = "kimi_k2"
LLM_F_MODEL = "llm_f"
TEST_WEIGHTS_CONTENT = "weights"
EMPTY_JSON = "{}"
INVALID_JSON = "{invalid json"
MISSING_TOKENIZER_ERROR = "tokenizer files are missing"

# Common model types for parameterized tests
COMMON_MODEL_TYPES = ["gpt2", "bert", "roberta", "t5", "llama", "mistral", "gemma"]

# ============================================================================
# Helper functions
# ============================================================================


def write_json(path: Path, data: dict) -> None:
    """Helper function to write JSON data to a file."""
    path.write_text(json.dumps(data))


def create_minimal_checkpoint(
    checkpoint_dir: Path,
    model_type: str = TEST_MODEL_TYPE,
    weight_filename: str = MODEL_SAFETENSORS_FILENAME,
    include_tokenizer_config: bool = True,
) -> None:
    """Helper to create a minimal valid checkpoint for testing."""
    write_json(checkpoint_dir / CONFIG_FILENAME, {MODEL_TYPE_FIELD: model_type})
    (checkpoint_dir / weight_filename).write_text(TEST_WEIGHTS_CONTENT)
    (checkpoint_dir / TOKENIZER_JSON_FILENAME).write_text(EMPTY_JSON)

    if include_tokenizer_config:
        (checkpoint_dir / TOKENIZER_CONFIG_FILENAME).write_text(EMPTY_JSON)


# ============================================================================
# Pytest fixtures
# ============================================================================


@pytest.fixture
def temp_checkpoint_dir(tmp_path: Path) -> Path:
    """Fixture providing a temporary directory for checkpoint testing."""
    checkpoint_dir = tmp_path / CHECKPOINT_DIR_NAME
    checkpoint_dir.mkdir()
    return checkpoint_dir


@pytest.fixture
def minimal_hf_checkpoint(temp_checkpoint_dir: Path) -> Path:
    """Fixture providing a minimal valid HuggingFace checkpoint."""
    create_minimal_checkpoint(temp_checkpoint_dir)
    return temp_checkpoint_dir


@pytest.fixture
def mock_autotokenizer_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fixture mocking AutoTokenizer.from_pretrained to raise error."""

    def _raise_tokenizer_error(*args: Any, **kwargs: Any) -> None:
        raise OSError(MISSING_TOKENIZER_ERROR)

    monkeypatch.setattr(
        convertation_manager_module.AutoTokenizer,
        "from_pretrained",
        _raise_tokenizer_error,
    )


# ============================================================================
# Test classes
# ============================================================================


class TestIsHFCheckpoint:
    """Tests for HuggingFace checkpoint validation."""

    # ------------------------------------------------------------------------
    # Tests with real checkpoint data
    # ------------------------------------------------------------------------

    def test_kimi_k2_tokenizer(self) -> None:
        """Test validation with real kimi_k2 checkpoint data."""
        hf_checkpoint_path = (
            Path(__file__).resolve().parents[1]
            / TEST_DATA_DIR
            / HF_CHECKPOINTS_DIR
            / KIMI_K2_MODEL
        )
        assert ConvertationUtils.is_hf_checkpoint(hf_checkpoint_path) is True

    def test_llm_f_tokenizer(self) -> None:
        """Test validation with real llm_f checkpoint data."""
        hf_checkpoint_path = (
            Path(__file__).resolve().parents[1]
            / TEST_DATA_DIR
            / HF_CHECKPOINTS_DIR
            / LLM_F_MODEL
        )
        assert ConvertationUtils.is_hf_checkpoint(hf_checkpoint_path) is True

    # ------------------------------------------------------------------------
    # Tests with invalid paths and directories
    # ------------------------------------------------------------------------

    def test_not_directory(self, tmp_path: Path) -> None:
        """Test that non-directory paths are rejected."""
        file_path = tmp_path / "not_a_dir"
        file_path.write_text("just a file")
        assert ConvertationUtils.is_hf_checkpoint(file_path) is False

    def test_missing_config(self, temp_checkpoint_dir: Path) -> None:
        """Test checkpoint without config.json is rejected."""
        checkpoint_dir = temp_checkpoint_dir
        (checkpoint_dir / MODEL_SAFETENSORS_FILENAME).write_text("")
        (checkpoint_dir / TOKENIZER_JSON_FILENAME).write_text(EMPTY_JSON)
        (checkpoint_dir / TOKENIZER_CONFIG_FILENAME).write_text(EMPTY_JSON)
        assert ConvertationUtils.is_hf_checkpoint(checkpoint_dir) is False

    def test_invalid_config_json(self, temp_checkpoint_dir: Path) -> None:
        """Test checkpoint with invalid JSON config is rejected."""
        checkpoint_dir = temp_checkpoint_dir
        (checkpoint_dir / CONFIG_FILENAME).write_text(INVALID_JSON)
        (checkpoint_dir / MODEL_SAFETENSORS_FILENAME).write_text("")
        (checkpoint_dir / TOKENIZER_JSON_FILENAME).write_text(EMPTY_JSON)
        (checkpoint_dir / TOKENIZER_CONFIG_FILENAME).write_text(EMPTY_JSON)
        assert ConvertationUtils.is_hf_checkpoint(checkpoint_dir) is False

    def test_missing_model_type(self, temp_checkpoint_dir: Path) -> None:
        """Test checkpoint without model_type in config is rejected."""
        checkpoint_dir = temp_checkpoint_dir
        write_json(checkpoint_dir / CONFIG_FILENAME, {"some_other_field": "value"})
        (checkpoint_dir / MODEL_SAFETENSORS_FILENAME).write_text("")
        (checkpoint_dir / TOKENIZER_JSON_FILENAME).write_text(EMPTY_JSON)
        (checkpoint_dir / TOKENIZER_CONFIG_FILENAME).write_text(EMPTY_JSON)
        assert ConvertationUtils.is_hf_checkpoint(checkpoint_dir) is False

    def test_no_weights(self, temp_checkpoint_dir: Path) -> None:
        """Test checkpoint without model weights is rejected."""
        checkpoint_dir = temp_checkpoint_dir
        write_json(checkpoint_dir / CONFIG_FILENAME, {MODEL_TYPE_FIELD: "gpt-test"})
        (checkpoint_dir / TOKENIZER_JSON_FILENAME).write_text(EMPTY_JSON)
        (checkpoint_dir / TOKENIZER_CONFIG_FILENAME).write_text(EMPTY_JSON)
        assert ConvertationUtils.is_hf_checkpoint(checkpoint_dir) is False

    def test_no_tokenizer(
        self,
        temp_checkpoint_dir: Path,
        mock_autotokenizer_error: None,
    ) -> None:
        """Tokenizer load failure does not reject checkpoint if layout is valid (config + weights)."""
        checkpoint_dir = temp_checkpoint_dir
        write_json(checkpoint_dir / CONFIG_FILENAME, {MODEL_TYPE_FIELD: "gpt-test"})
        (checkpoint_dir / MODEL_SAFETENSORS_FILENAME).write_text("")
        assert ConvertationUtils.is_hf_checkpoint(checkpoint_dir) is True

    def test_tokenizer_returns_none_keeps_checkpoint_valid(
        self,
        temp_checkpoint_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Tokenizer returning None is tolerated for structurally valid HF checkpoints."""
        checkpoint_dir = temp_checkpoint_dir
        write_json(checkpoint_dir / CONFIG_FILENAME, {MODEL_TYPE_FIELD: "gpt-test"})
        (checkpoint_dir / MODEL_SAFETENSORS_FILENAME).write_text("")
        monkeypatch.setattr(
            convertation_manager_module.AutoTokenizer,
            "from_pretrained",
            lambda *args, **kwargs: None,
        )
        assert ConvertationUtils.is_hf_checkpoint(checkpoint_dir) is True

    # ------------------------------------------------------------------------
    # Positive validation tests
    # ------------------------------------------------------------------------

    def test_valid_minimal_checkpoint(self, minimal_hf_checkpoint: Path) -> None:
        """Test that a minimal valid checkpoint is accepted."""
        assert ConvertationUtils.is_hf_checkpoint(minimal_hf_checkpoint) is True

    # ------------------------------------------------------------------------
    # Tests with invalid paths and directories
    # ------------------------------------------------------------------------

    def test_empty_directory(self, temp_checkpoint_dir: Path) -> None:
        """Test that an empty directory is rejected."""
        checkpoint_dir = temp_checkpoint_dir
        assert ConvertationUtils.is_hf_checkpoint(checkpoint_dir) is False

    def test_directory_does_not_exist(self, tmp_path: Path) -> None:
        """Test that non-existent directory is rejected."""
        non_existent = tmp_path / "non_existent"
        assert ConvertationUtils.is_hf_checkpoint(non_existent) is False

    # ------------------------------------------------------------------------
    # Tests with various file formats and configurations
    # ------------------------------------------------------------------------

    def test_checkpoint_with_different_weight_formats(
        self,
        temp_checkpoint_dir: Path,
    ) -> None:
        """Test checkpoints with various weight file formats."""
        checkpoint_dir = temp_checkpoint_dir

        # Test with .bin weights (alternative format)
        create_minimal_checkpoint(checkpoint_dir, weight_filename=MODEL_BIN_FILENAME)

        # Should still be valid with .bin weights
        assert ConvertationUtils.is_hf_checkpoint(checkpoint_dir) is True

        # Clean up and test with both .safetensors and .bin
        (checkpoint_dir / MODEL_SAFETENSORS_FILENAME).write_text(TEST_WEIGHTS_CONTENT)
        assert ConvertationUtils.is_hf_checkpoint(checkpoint_dir) is True

    @pytest.mark.parametrize("model_type", COMMON_MODEL_TYPES)
    def test_various_model_types(
        self,
        temp_checkpoint_dir: Path,
        model_type: str,
    ) -> None:
        """Test checkpoints with various model types."""
        create_minimal_checkpoint(temp_checkpoint_dir, model_type=model_type)
        assert ConvertationUtils.is_hf_checkpoint(temp_checkpoint_dir) is True

    def test_checkpoint_with_additional_files(
        self,
        temp_checkpoint_dir: Path,
    ) -> None:
        """Test checkpoint with additional files doesn't affect validation."""
        create_minimal_checkpoint(temp_checkpoint_dir)

        # Add extra files that shouldn't affect validation
        (temp_checkpoint_dir / "extra_config.json").write_text(EMPTY_JSON)
        (temp_checkpoint_dir / "README.md").write_text("# Model")
        (temp_checkpoint_dir / "special_tokens_map.json").write_text(EMPTY_JSON)

        assert ConvertationUtils.is_hf_checkpoint(temp_checkpoint_dir) is True

    def test_checkpoint_with_only_required_tokenizer_file(
        self,
        temp_checkpoint_dir: Path,
    ) -> None:
        """Test checkpoint with only tokenizer.json (no tokenizer_config.json)."""
        create_minimal_checkpoint(temp_checkpoint_dir, include_tokenizer_config=False)

        # Should still be valid with just tokenizer.json
        assert ConvertationUtils.is_hf_checkpoint(temp_checkpoint_dir) is True


class TestIsFsdpCheckpoint:
    """Tests for FSDP checkpoint validation."""

    def test_valid_fsdp_checkpoint(self, temp_checkpoint_dir: Path) -> None:
        create_minimal_checkpoint(temp_checkpoint_dir)
        (temp_checkpoint_dir / "composer_state.pt").write_text("state")
        assert ConvertationUtils.is_fsdp_checkpoint(temp_checkpoint_dir) is True

    def test_not_directory(self, tmp_path: Path) -> None:
        file_path = tmp_path / "not_a_dir"
        file_path.write_text("just a file")
        assert ConvertationUtils.is_fsdp_checkpoint(file_path) is False

    def test_directory_does_not_exist(self, tmp_path: Path) -> None:
        assert ConvertationUtils.is_fsdp_checkpoint(tmp_path / "missing") is False

    def test_missing_composer_state(self, temp_checkpoint_dir: Path) -> None:
        create_minimal_checkpoint(temp_checkpoint_dir)
        assert ConvertationUtils.is_fsdp_checkpoint(temp_checkpoint_dir) is False

    def test_missing_config(self, temp_checkpoint_dir: Path) -> None:
        (temp_checkpoint_dir / "composer_state.pt").write_text("state")
        (temp_checkpoint_dir / MODEL_SAFETENSORS_FILENAME).write_text("w")
        assert ConvertationUtils.is_fsdp_checkpoint(temp_checkpoint_dir) is False

    def test_invalid_config_json(self, temp_checkpoint_dir: Path) -> None:
        (temp_checkpoint_dir / "composer_state.pt").write_text("state")
        (temp_checkpoint_dir / CONFIG_FILENAME).write_text(INVALID_JSON)
        (temp_checkpoint_dir / MODEL_SAFETENSORS_FILENAME).write_text("w")
        assert ConvertationUtils.is_fsdp_checkpoint(temp_checkpoint_dir) is False

    def test_missing_model_type(self, temp_checkpoint_dir: Path) -> None:
        (temp_checkpoint_dir / "composer_state.pt").write_text("state")
        write_json(temp_checkpoint_dir / CONFIG_FILENAME, {"other": "value"})
        (temp_checkpoint_dir / MODEL_SAFETENSORS_FILENAME).write_text("w")
        assert ConvertationUtils.is_fsdp_checkpoint(temp_checkpoint_dir) is False

    def test_no_weights(self, temp_checkpoint_dir: Path) -> None:
        (temp_checkpoint_dir / "composer_state.pt").write_text("state")
        write_json(temp_checkpoint_dir / CONFIG_FILENAME, {MODEL_TYPE_FIELD: "gpt"})
        assert ConvertationUtils.is_fsdp_checkpoint(temp_checkpoint_dir) is False
