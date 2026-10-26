import pytest
from sber_tests.test_local_core.modality import (
    MODALITIES,
    get_modality_version
)


class TestModalitiesStructure:

    @pytest.mark.unit
    @pytest.mark.fast
    def test_modalities_is_dict(self):
        assert isinstance(MODALITIES, dict)
        assert len(MODALITIES) > 0

    @pytest.mark.unit
    def test_golden_modality_exists(self):
        assert "GOLDEN" in MODALITIES

    @pytest.mark.unit
    @pytest.mark.parametrize("modality_name", [
        "GOLDEN",
        "SFT",
        "VISION",
        "VIDEO",
        "SPEECH",
        "TTS",
    ])
    def test_expected_modalities_exist(self, modality_name):
        assert modality_name in MODALITIES

    @pytest.mark.unit
    def test_modality_has_version(self):
        for modality_name, modality in MODALITIES.items():
            assert hasattr(modality, 'version'), \
                f"Modality {modality_name} missing 'version'"
            assert isinstance(modality.version, str)
            assert len(modality.version) > 0

    @pytest.mark.unit
    def test_modality_has_allow_update(self):
        for modality_name, modality in MODALITIES.items():
            assert hasattr(modality, 'allow_update'), \
                f"Modality {modality_name} missing 'allow_update'"
            assert isinstance(modality.allow_update, bool)


class TestGetModalityVersion:

    @pytest.mark.unit
    @pytest.mark.fast
    def test_get_golden_version(self):
        version = get_modality_version("GOLDEN")

        assert isinstance(version, str)
        assert len(version) > 0

    @pytest.mark.unit
    @pytest.mark.parametrize("modality", [
        "GOLDEN",
        "SFT",
        "VISION",
        "VIDEO",
        "SPEECH",
        "TTS",
    ])
    def test_get_version_for_all_modalities(self, modality):
        version = get_modality_version(modality)

        assert isinstance(version, str)
        assert len(version) > 0

    @pytest.mark.unit
    def test_version_format(self):
        version = get_modality_version("GOLDEN")

        assert len(version) > 0
        assert ' ' not in version


class TestModalityVersions:

    @pytest.mark.unit
    def test_golden_allow_update_false(self):
        golden = MODALITIES["GOLDEN"]

        assert hasattr(golden, 'allow_update')

    @pytest.mark.unit
    def test_versions_are_unique_per_modality(self):
        versions = {}
        for modality_name, modality in MODALITIES.items():
            versions[modality_name] = modality.version

        assert len(versions) == len(MODALITIES)


class TestModalityValidation:

    @pytest.mark.unit
    def test_modality_names_uppercase(self):
        for modality_name in MODALITIES.keys():
            assert modality_name.isupper(), \
                f"Modality name {modality_name} should be uppercase"

    @pytest.mark.unit
    def test_no_duplicate_modalities(self):
        modality_names = list(MODALITIES.keys())
        unique_names = set(modality_names)

        assert len(modality_names) == len(unique_names)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'unit'])
