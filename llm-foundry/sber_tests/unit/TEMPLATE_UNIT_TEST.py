import pytest

# ОБЯЗАТЕЛЬНО: Импортируйте общие фикстуры и helpers
from sber_tests.test_utils.test_fixtures import (
    minimal_config,      # Минимальный конфиг
    sample_config,       # Расширенный конфиг
    temp_dir,            # Временная директория
    temp_artifact_dir,   # Директория для артефактов
    mock_job,            # Mock джобы
    mock_executor,       # Mock executor
)
from sber_tests.test_utils.test_helpers import (
    create_test_config,          # Создание конфига
    assert_config_has_fields,    # Проверка полей конфига
    validate_matrix_structure,   # Валидация матрицы
)

from sber_tests.test_local_core.matrix import ConfigMatrix


# ============================================================================
# ПРИМЕР CPU: Тесты для класса ConfigMatrix
# ============================================================================

class TestConfigMatrix:
    """
    Юнит-тесты для ConfigMatrix класса.

    Тестируем: создание, дефолты, кастомные поля
    НЕ тестируем: запуск джоб с этой матрицей (это интеграция)
    """

    @pytest.mark.unit
    @pytest.mark.fast
    def test_basic_creation(self):
        """Создание с дефолтами."""
        matrix = ConfigMatrix()

        # Используем helper для валидации
        validate_matrix_structure(matrix)

        assert 'base' in matrix.data
        assert matrix.data['base'] == ['base_gigar.yaml']

    @pytest.mark.unit
    def test_custom_fields(self):
        """Кастомные поля."""
        matrix = ConfigMatrix(
            fsdp_type=['fsdp.yaml', 'gigafsdp.yaml'],
            tp_size=['tp-1.yaml', 'tp-2.yaml']
        )

        assert len(matrix.data['fsdp_type']) == 2
        assert len(matrix.data['tp_size']) == 2

    @pytest.mark.unit
    @pytest.mark.parametrize("field,expected", [
        ('base', ['base_gigar.yaml']),
        ('tp_size', ['tp-1.yaml']),
    ])
    def test_defaults(self, field, expected):
        """Дефолтные значения."""
        matrix = ConfigMatrix()
        assert matrix.data[field] == expected


# ============================================================================
# ПРИМЕР CPU: Тесты для класса с конфигами
# ============================================================================

class TestConfigOperations:
    """Тесты работы с конфигами."""

    @pytest.mark.unit
    def test_config_creation(self, minimal_config):
        """Используем fixture из fixtures.py."""
        # Используем helper для проверки
        assert_config_has_fields(
            minimal_config,
            ['model_name', 'max_duration', 'global_train_batch_size']
        )

    @pytest.mark.unit
    def test_create_custom_config(self):
        """Используем helper для создания конфига."""
        config = create_test_config(
            batch_size=16,
            learning_rate=0.001
        )

        assert config.batch_size == 16
        assert config.learning_rate == 0.001

    @pytest.mark.unit
    def test_with_temp_dir(self, temp_dir):
        """Используем fixture temp_dir для файловых операций."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("content")

        assert test_file.exists()


# ============================================================================
# ПРИМЕР CPU: Тесты с моками
# ============================================================================

class TestWithMocks:
    """Тесты использующие моки."""

    @pytest.mark.unit
    def test_with_mock_job(self, mock_job):
        """Используем mock_job fixture."""
        status = mock_job.status()

        assert status == "SUCCESS"
        mock_job.wait.assert_not_called()

    @pytest.mark.unit
    def test_with_mock_executor(self, mock_executor):
        """Используем mock_executor fixture."""
        job = mock_executor.submit_job(
            config_path='/fake/config.yaml',
            processes_per_worker=2
        )

        assert job is not None
        assert job.status() == "SUCCESS"


# ============================================================================
# ПРИМЕР GPU: Тесты требующие GPU
# ============================================================================

class TestGPUOperations:
    """
    Тесты требующие GPU.

    ВАЖНО: Используйте @pytest.mark.gpu для всех GPU тестов
    """

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_cuda_available(self):
        """Проверка доступности CUDA."""
        import torch
        assert torch.cuda.is_available(), "CUDA должна быть доступна для GPU тестов"

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    @pytest.mark.parametrize("device_id", [0])
    def test_gpu_tensor_operations(self, device_id):
        """Базовые операции с тензорами на GPU."""
        import torch

        device = torch.device(f'cuda:{device_id}')
        x = torch.randn(10, 10, device=device)
        y = torch.randn(10, 10, device=device)

        result = torch.matmul(x, y)

        assert result.is_cuda
        assert result.shape == (10, 10)


# ============================================================================
# СТРУКТУРА ЮНИТ-ТЕСТА
# ============================================================================

"""
1. Импорты:
   - pytest
   - Общие фикстуры из sber_tests.test_utils.test_fixtures
   - Общие helpers из sber_tests.test_utils.test_helpers
   - Тестируемый класс

2. Один класс тестов = Один тестируемый класс:
   class TestConfigMatrix  → тестирует ConfigMatrix
   class TestTestEntry     → тестирует TestEntry

3. Каждый тест:
   @pytest.mark.unit          # ОБЯЗАТЕЛЬНО для всех юнит-тестов
   @pytest.mark.requires_gpu  # для GPU тестов
   @pytest.mark.fast          # для быстрых тестов
   @pytest.mark.slow          # для медленных тестов
   def test_что_тестируем(self, fixtures_если_нужны):
       # Arrange
       obj = create_object()

       # Act
       result = obj.method()

       # Assert
       assert result == expected

4. Используйте общие инструменты:
   ✓ Фикстуры из test_utils/test_fixtures.py
   ✓ Helpers из test_utils/test_helpers.py
   ✗ НЕ создавайте свои фикстуры в тесте (добавьте в test_fixtures.py)

5. Маркеры:
   @pytest.mark.unit                - ОБЯЗАТЕЛЬНО для всех юнит-тестов
   @pytest.mark.requires_gpu        - для GPU тестов
   @pytest.mark.fast                - для быстрых тестов (<1s)
   @pytest.mark.slow                - для медленных тестов (>5s)
"""

# ============================================================================
# КАК ЗАПУСТИТЬ
# ============================================================================

"""
# Все юнит-тесты
pytest sber_tests/unit/ -v -m unit

# Только GPU тесты
pytest sber_tests/unit/ -v -m "unit and requires_gpu"

# Только быстрые
pytest sber_tests/unit/ -v -m "unit and fast"

# Конкретный файл
pytest sber_tests/unit/cpu/core/test_config_matrix.py -v

# Параллельно (только CPU тесты)
pytest sber_tests/unit/ -n auto -m "unit and cpu"
"""

if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'unit'])
