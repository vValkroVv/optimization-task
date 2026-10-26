"""
Общие фикстуры для всех юнит-тестов.

Этот файл содержит переиспользуемые pytest фикстуры.
Импортируйте их в ваших тестах:

from sber_tests.test_utils.fixtures import minimal_config, temp_artifact_dir
"""

from pathlib import Path
import pytest
from omegaconf import OmegaConf, DictConfig
from unittest.mock import MagicMock


# ============================================================================
# CONFIG FIXTURES
# ============================================================================

@pytest.fixture
def minimal_config() -> DictConfig:
    return OmegaConf.create({
        'model_name': 'test_model',
        'max_duration': '10ba',
        'global_train_batch_size': 4,
        'device_train_microbatch_size': 1,
    })


@pytest.fixture
def sample_config() -> DictConfig:
    """Расширенный тестовый конфиг."""
    return OmegaConf.create({
        'model_name': 'test_model',
        'max_duration': '10ba',
        'save_folder': '/tmp/test/ckpt',
        'global_train_batch_size': 4,
        'device_train_microbatch_size': 1,
        'learning_rate': 1e-4,
    })


# ============================================================================
# DIRECTORY FIXTURES
# ============================================================================

@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    test_dir = tmp_path / "test_run"
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir


@pytest.fixture
def temp_artifact_dir(tmp_path: Path) -> Path:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


@pytest.fixture
def temp_config_dir(tmp_path: Path) -> Path:
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


# ============================================================================
# MOCK FIXTURES
# ============================================================================

@pytest.fixture
def mock_job():
    job = MagicMock()
    job.status.return_value = "SUCCESS"
    job.wait.return_value = None
    job.poll.return_value = 0  # завершился успешно
    return job


@pytest.fixture
def mock_executor():
    executor = MagicMock()
    mock_job = MagicMock()
    mock_job.status.return_value = "SUCCESS"
    executor.submit_job.return_value = mock_job
    return executor


@pytest.fixture
def mock_test_entry(minimal_config, temp_artifact_dir):
    entry = MagicMock()
    entry.test_name = "test_mock"
    entry.n_gpus = 1
    entry.modality = ""
    entry.config = minimal_config
    entry.job = None
    entry.id = "base=test"
    entry.artifact_dir = str(temp_artifact_dir)
    entry.run_path = str(temp_artifact_dir / "run" / "test")
    entry.golden_path = str(temp_artifact_dir / "golden" / "test")
    return entry


# ============================================================================
# COMPONENT FIXTURES
# ============================================================================

@pytest.fixture
def sample_components() -> dict:
    return {
        'base': 'base_gigar.yaml',
        'fsdp_type': 'gigafsdp.yaml',
        'tp_size': 'tp-1.yaml',
        'sp_size': 'sp-1.yaml',
    }


@pytest.fixture
def checkpoint_pattern() -> str:
    return "{base},{fsdp_type},{tp_size}"
