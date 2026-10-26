import pytest
from sber_tests.test_local_core.matrix import (
    ConfigMatrix,
    build_checkpoint_pattern,
    test_config
)


class TestConfigMatrix:

    @pytest.mark.unit
    @pytest.mark.fast
    def test_basic_creation(self):
        matrix = ConfigMatrix()

        assert 'base' in matrix.data
        assert matrix.data['base'] == ['base_gigar.yaml']
        assert matrix.data['fsdp_type'] == ['gigafsdp.yaml']
        assert matrix.data['tp_size'] == ['tp-1.yaml']
        assert matrix.data['sp_size'] == ['sp-1.yaml']

    @pytest.mark.unit
    def test_custom_fields(self):
        matrix = ConfigMatrix(
            fsdp_type=['fsdp.yaml', 'gigafsdp.yaml'],
            tp_size=['tp-1.yaml', 'tp-2.yaml']
        )

        assert len(matrix.data['fsdp_type']) == 2
        assert len(matrix.data['tp_size']) == 2
        assert 'fsdp.yaml' in matrix.data['fsdp_type']
        assert 'gigafsdp.yaml' in matrix.data['fsdp_type']

    @pytest.mark.unit
    @pytest.mark.parametrize("field,expected", [
        ('base', ['base_gigar.yaml']),
        ('fsdp_type', ['gigafsdp.yaml']),
        ('tp_size', ['tp-1.yaml']),
        ('sp_size', ['sp-1.yaml']),
    ])
    def test_default_values(self, field, expected):
        matrix = ConfigMatrix()
        assert matrix.data[field] == expected

    @pytest.mark.unit
    def test_override_base(self):
        matrix = ConfigMatrix(base=['custom_base.yaml'])

        assert matrix.data['base'] == ['custom_base.yaml']
        # Остальные дефолты должны остаться
        assert matrix.data['tp_size'] == ['tp-1.yaml']

    @pytest.mark.unit
    def test_multiple_values_per_field(self):
        matrix = ConfigMatrix(
            tp_size=['tp-1.yaml', 'tp-2.yaml', 'tp-4.yaml']
        )

        assert len(matrix.data['tp_size']) == 3
        assert 'tp-2.yaml' in matrix.data['tp_size']

    @pytest.mark.unit
    def test_custom_field_without_default(self):
        matrix = ConfigMatrix(
            icl_type=['icl-default.yaml', 'icl-extended.yaml']
        )

        assert 'icl_type' in matrix.data
        assert len(matrix.data['icl_type']) == 2

    @pytest.mark.unit
    def test_none_values_ignored(self):
        matrix = ConfigMatrix(
            fsdp_type=None,
            tp_size=['tp-2.yaml']
        )

        assert matrix.data['fsdp_type'] == ['gigafsdp.yaml']
        assert matrix.data['tp_size'] == ['tp-2.yaml']


class TestConfigMatrixPresets:

    @pytest.mark.unit
    def test_ring_attention_preset(self):
        matrix = ConfigMatrix.ring_attention()

        assert 'attention_type' in matrix.data
        assert 'ring_attn.yaml' in matrix.data['attention_type']
        assert matrix.data['sp_size'] == ['sp-2.yaml']

    @pytest.mark.unit
    def test_vision_preset(self):
        """Preset для vision."""
        matrix = ConfigMatrix.vision()

        assert 'base' in matrix.data
        assert 'model_gigar.yaml' in matrix.data['base']

        assert 'sft_type' in matrix.data
        assert 'base_train.yaml' in matrix.data['sft_type']

        assert 'icl_type' in matrix.data
        assert 'icl_task.yaml' in matrix.data['icl_type']


    @pytest.mark.unit
    def test_video_preset(self):
        """Preset для video."""
        matrix = ConfigMatrix.video()

        assert 'base' in matrix.data
        assert 'model_gigar.yaml' in matrix.data['base']

        assert 'sft_type' in matrix.data
        assert 'base_train.yaml' in matrix.data['sft_type']

        assert 'icl_type' in matrix.data
        assert 'icl_task.yaml' in matrix.data['icl_type']

    @pytest.mark.unit
    def test_tts_preset(self):
        matrix = ConfigMatrix.tts()

        assert 'base_tts.yaml' in matrix.data['sft_type']
        assert 'base_tts_lora.yaml' in matrix.data['sft_type']

    @pytest.mark.unit
    def test_dpo_preset(self):
        matrix = ConfigMatrix.dpo()

        assert matrix.data['base'] == ['base_dpo.yaml']
        assert len(matrix.data['fsdp_type']) == 2


class TestBuildCheckpointPattern:

    @pytest.mark.unit
    def test_basic_pattern(self):
        matrix = ConfigMatrix()
        pattern = build_checkpoint_pattern(matrix)

        assert '{base}' in pattern
        assert '{fsdp_type}' in pattern
        assert '{tp_size}' in pattern
        assert '{sp_size}' in pattern

    @pytest.mark.unit
    def test_pattern_with_exclude(self):
        matrix = ConfigMatrix(icl_type=['icl-default.yaml'])
        pattern = build_checkpoint_pattern(matrix, exclude={'icl_type'})

        assert '{icl_type}' not in pattern
        assert '{base}' in pattern

    @pytest.mark.unit
    def test_pattern_order(self):
        matrix = ConfigMatrix()
        pattern = build_checkpoint_pattern(matrix)

        parts = pattern.split(',')
        assert parts[0] == '{base}'

    @pytest.mark.unit
    def test_pattern_with_custom_fields(self):
        matrix = ConfigMatrix(
            sft_type=['base_sft.yaml'],
            eval=['eval.yaml']
        )
        pattern = build_checkpoint_pattern(matrix)

        assert '{sft_type}' in pattern
        assert '{eval}' in pattern


class TestTestConfigFunction:

    @pytest.mark.unit
    def test_basic_config(self):
        result = test_config(
            test_name="TestBasic",
            matrix=ConfigMatrix()
        )

        assert result['test_name'] == "TestBasic"
        assert 'override_matrix' in result
        assert 'checkpoint_cache_pattern' in result
        assert result['start_from_checkpoint'] is False

    @pytest.mark.unit
    def test_config_with_gpu_mapping(self):
        gpu_map = {'gigafsdp.yaml': 4, 'tp-2.yaml': 2}
        result = test_config(
            test_name="TestGPU",
            matrix=ConfigMatrix(),
            gpu_mapping=gpu_map
        )

        assert 'gpu_mult_mapping' in result
        assert result['gpu_mult_mapping'] == gpu_map

    @pytest.mark.unit
    def test_config_with_fixed_gpus(self):
        """test_config с фиксированным числом GPU."""
        result = test_config(
            test_name="TestFixedGPU",
            matrix=ConfigMatrix(),
            n_gpus=8
        )

        assert result['n_gpus'] == 8
        assert 'gpu_mult_mapping' not in result

    @pytest.mark.unit
    def test_config_with_exclude(self):
        excludes = [
            ('fsdp.yaml', 'tp-2.yaml'),
            ('gigafsdp.yaml', 'tp-4.yaml'),
        ]
        result = test_config(
            test_name="TestExclude",
            matrix=ConfigMatrix(),
            exclude_subsets=excludes
        )

        assert 'exclude_subsets' in result
        assert len(result['exclude_subsets']) == 2

    @pytest.mark.unit
    def test_config_with_modality(self):
        result = test_config(
            test_name="TestVision",
            matrix=ConfigMatrix.vision(),
            modality="VISION"
        )

        assert result['modality'] == "VISION"

    @pytest.mark.unit
    def test_config_default_gpu_mapping(self):
        result = test_config(
            test_name="TestDefault",
            matrix=ConfigMatrix()
        )

        assert 'gpu_mult_mapping' in result
        assert result['gpu_mult_mapping'] == ConfigMatrix.GPU_DEFAULTS

    @pytest.mark.unit
    def test_config_checkpoint_pattern_auto(self):
        result = test_config(
            test_name="TestAuto",
            matrix=ConfigMatrix(icl_type=['icl-default.yaml'])
        )

        # icl_type должен быть исключен из pattern
        assert '{icl_type}' not in result['checkpoint_cache_pattern']
        assert '{base}' in result['checkpoint_cache_pattern']

    @pytest.mark.unit
    def test_config_custom_checkpoint_pattern(self):
        custom_pattern = "{base},{custom_field}"
        result = test_config(
            test_name="TestCustomPattern",
            matrix=ConfigMatrix(),
            checkpoint_pattern=custom_pattern
        )

        assert result['checkpoint_cache_pattern'] == custom_pattern


class TestConfigMatrixCombinations:

    @pytest.mark.unit
    def test_total_combinations_count(self):
        matrix = ConfigMatrix(
            fsdp_type=['fsdp.yaml', 'gigafsdp.yaml'],  # 2
            tp_size=['tp-1.yaml', 'tp-2.yaml'],  # 2
        )

        total = 1
        for values in matrix.data.values():
            total *= len(values)

        assert total == 4

    @pytest.mark.unit
    def test_large_combination_matrix(self):
        matrix = ConfigMatrix(
            fsdp_type=['fsdp.yaml', 'gigafsdp.yaml'],  # 2
            tp_size=['tp-1.yaml', 'tp-2.yaml', 'tp-4.yaml'],  # 3
            sp_size=['sp-1.yaml', 'sp-2.yaml'],  # 2
        )

        total = 1
        for values in matrix.data.values():
            total *= len(values)

        assert total == 12


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'unit'])
