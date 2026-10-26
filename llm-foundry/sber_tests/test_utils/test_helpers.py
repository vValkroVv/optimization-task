"""
Общие helper функции для юнит-тестов.

Этот файл содержит переиспользуемые утилиты для тестирования.
Импортируйте их в ваших тестах:

from sber_tests.test_utils.helpers import create_test_config, assert_config_has_fields
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
from omegaconf import OmegaConf, DictConfig


# ============================================================================
# CONFIG HELPERS
# ============================================================================

def create_test_config(**overrides) -> DictConfig:
    """
    Создает тестовый конфиг с возможностью переопределения полей.

    Args:
        **overrides: Поля для переопределения

    Returns:
        DictConfig с тестовыми данными

    Example:
        >>> config = create_test_config(batch_size=8, lr=0.001)
        >>> assert config.batch_size == 8
    """
    base = {
        'model_name': 'test_model',
        'max_duration': '10ba',
        'global_train_batch_size': 4,
        'device_train_microbatch_size': 1,
    }
    base.update(overrides)
    return OmegaConf.create(base)


# ============================================================================
# ASSERTION HELPERS
# ============================================================================

def assert_config_has_fields(config: DictConfig, fields: List[str]) -> None:
    """
    Проверяет что конфиг содержит все указанные поля.

    Args:
        config: Конфиг для проверки
        fields: Список обязательных полей (поддерживает dot notation)

    Raises:
        AssertionError: Если поле отсутствует

    Example:
        >>> config = OmegaConf.create({'model': 'gpt', 'train': {'lr': 0.001}})
        >>> assert_config_has_fields(config, ['model', 'train.lr'])
    """
    for field in fields:
        parts = field.split('.')
        current = config
        for part in parts:
            assert part in current, f"Field '{field}' not found in config"
            current = current[part]


def assert_matrix_combinations_count(matrix_data: Dict[str, List], expected: int) -> None:
    """
    Проверяет что матрица создает ожидаемое число комбинаций.

    Args:
        matrix_data: Dict с списками значений
        expected: Ожидаемое число комбинаций

    Example:
        >>> matrix = {'a': ['1', '2'], 'b': ['x', 'y']}
        >>> assert_matrix_combinations_count(matrix, 4)  # 2*2=4
    """
    actual = 1
    for values in matrix_data.values():
        actual *= len(values)

    assert actual == expected, \
        f"Expected {expected} combinations, got {actual}"


# ============================================================================
# CONFIG MATRIX HELPERS
# ============================================================================

def create_test_matrix(**fields) -> Dict[str, List[str]]:
    """
    Создает тестовую config matrix.

    Args:
        **fields: Поля матрицы с их значениями

    Example:
        >>> matrix = create_test_matrix(
        ...     fsdp_type=['fsdp.yaml', 'gigafsdp.yaml'],
        ...     tp_size=['tp-1.yaml']
        ... )
    """
    defaults = {
        'base': ['base_gigar.yaml'],
        'fsdp_type': ['gigafsdp.yaml'],
        'tp_size': ['tp-1.yaml'],
        'sp_size': ['sp-1.yaml'],
    }
    defaults.update(fields)
    return defaults


def build_component_dict(**components) -> Dict[str, str]:
    """
    Создает словарь компонентов для TestEntry.

    Example:
        >>> components = build_component_dict(
        ...     base='base.yaml',
        ...     tp_size='tp-2.yaml'
        ... )
    """
    defaults = {
        'base': 'base_gigar.yaml',
        'fsdp_type': 'gigafsdp.yaml',
        'tp_size': 'tp-1.yaml',
    }
    defaults.update(components)
    return defaults


# ============================================================================
# PATH HELPERS
# ============================================================================

def create_fake_checkpoint_files(
    base_dir: Path,
    n_ranks: int = 2,
    batch_num: int = 5
) -> List[Path]:
    """
    Создает фейковые checkpoint файлы для тестирования.

    Args:
        base_dir: Базовая директория
        n_ranks: Количество ranks
        batch_num: Номер батча

    Returns:
        Список созданных файлов

    Example:
        >>> from pathlib import Path
        >>> files = create_fake_checkpoint_files(Path('/tmp/test'), n_ranks=2)
    """
    batch_dir = base_dir / f"ep0_ba{batch_num}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for rank in range(n_ranks):
        ckpt_file = batch_dir / f"ep0-ba{batch_num}-rank{rank}.pt"
        ckpt_file.touch()
        files.append(ckpt_file)

    return files


def create_fake_tensorboard_logs(tb_dir: Path) -> Path:
    """
    Создает фейковый tensorboard event file.

    Example:
        >>> from pathlib import Path
        >>> event_file = create_fake_tensorboard_logs(Path('/tmp/tb'))
    """
    tb_dir.mkdir(parents=True, exist_ok=True)
    run_dir = tb_dir / "run_0"
    run_dir.mkdir(parents=True, exist_ok=True)

    event_file = run_dir / "events.out.tfevents.1234567890.test"
    event_file.touch()

    return event_file


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

def validate_entry_paths(entry) -> None:
    """
    Валидирует что TestEntry имеет корректные пути.

    Args:
        entry: TestEntry объект

    Raises:
        AssertionError: Если пути некорректны
    """
    assert hasattr(entry, 'run_path'), "Entry missing run_path"
    assert hasattr(entry, 'golden_path'), "Entry missing golden_path"
    assert hasattr(entry, 'config_path'), "Entry missing config_path"

    assert len(entry.run_path) > 0, "run_path is empty"
    assert len(entry.golden_path) > 0, "golden_path is empty"
    assert len(entry.config_path) > 0, "config_path is empty"


def validate_matrix_structure(matrix) -> None:
    """
    Валидирует структуру ConfigMatrix.

    Args:
        matrix: ConfigMatrix объект

    Raises:
        AssertionError: Если структура некорректна
    """
    assert hasattr(matrix, 'data'), "Matrix missing 'data' attribute"
    assert isinstance(matrix.data, dict), "Matrix.data should be dict"
    assert len(matrix.data) > 0, "Matrix.data is empty"

    # Проверяем что каждое значение - список
    for key, value in matrix.data.items():
        assert isinstance(value, list), \
            f"Matrix.data['{key}'] should be list, got {type(value)}"
