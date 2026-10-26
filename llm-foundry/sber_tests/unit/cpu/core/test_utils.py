import pytest
from unittest.mock import patch, MagicMock
import sber_tests.test_utils.git_utils

from sber_tests.test_utils.test_fixtures import temp_dir


# ============================================================================
# TESTS: argfilter
# ============================================================================

class TestArgFilter:

    @pytest.mark.unit
    @pytest.mark.fast
    def test_get_modality_mapping(self):
        from sber_tests.test_utils.argfilter import get_modality_mapping

        mapping = get_modality_mapping()

        assert isinstance(mapping, dict)
        assert 'CORE' in mapping
        assert 'UNIT' in mapping
        assert 'VISION' in mapping
        assert 'DPO' in mapping

        core_tests = mapping['CORE']
        assert 'TestBasicSetups' in core_tests
        assert 'TestEval' in core_tests

    @pytest.mark.unit
    @pytest.mark.fast
    def test_get_test_filter_args_unit(self):
        from sber_tests.test_utils.argfilter import get_test_filter_args

        result = get_test_filter_args("/fake/code/path", modality="UNIT")

        assert "/fake/code/path/sber_tests/unit/" in result

    @pytest.mark.unit
    @pytest.mark.fast
    def test_get_test_filter_args_core(self):
        from sber_tests.test_utils.argfilter import get_test_filter_args

        result = get_test_filter_args("/fake/code/path", modality="CORE")

        assert "TestBasicSetups" in result
        assert "integration/test_local_jobs.py" in result

    @pytest.mark.unit
    @pytest.mark.fast
    def test_get_test_filter_args_custom_classes(self):
        from sber_tests.test_utils.argfilter import get_test_filter_args

        result = get_test_filter_args(
            "/fake/code/path",
            test_classes=["TestVision", "TestDPO"]
        )

        assert result.startswith('-k "')
        assert "TestVision" in result
        assert "TestDPO" in result
        assert " or " in result


# ============================================================================
# TESTS: git_utils
# ============================================================================

class TestGitUtils:

    @pytest.mark.unit
    @pytest.mark.fast
    def test_make_run_name_suffix_import(self):
        try:
            from sber_tests.test_utils.git_utils import make_run_name_suffix
            assert callable(make_run_name_suffix)
        except ImportError as e:
            pytest.skip(f"Git utils not available: {e}")

    @pytest.mark.unit
    @pytest.mark.fast
    @patch('sber_tests.test_utils.git_utils.Repo')
    def test_make_run_name_suffix_with_mock(self, mock_repo_class):
        try:
            from sber_tests.test_utils.git_utils import make_run_name_suffix

            mock_repo = MagicMock()
            mock_repo.head.commit.hexsha = "abc123def456"
            mock_repo.is_dirty.return_value = False

            mock_commit = MagicMock()
            mock_commit.author.name = "TestUser"
            mock_repo.iter_commits.return_value = iter([mock_commit])

            mock_repo_class.return_value = mock_repo

            result = make_run_name_suffix("/fake/path")

            assert isinstance(result, str)
            assert len(result) > 0

        except ImportError as e:
            pytest.skip(f"Git utils not available: {e}")


# ============================================================================
# TESTS: argument_parser
# ============================================================================

class TestArgumentParser:

    @pytest.mark.unit
    @pytest.mark.fast
    def test_create_parser(self):
        from sber_tests.test_utils.argument_parser import create_parser

        parser = create_parser(8)

        assert parser is not None

        args = parser.parse_args(['--modality', 'UNIT', '--n-gpus', '1'])
        assert args.modality == 'UNIT'
        assert args.n_gpus == 1

    @pytest.mark.unit
    @pytest.mark.fast
    def test_parser_choices(self):
        from sber_tests.test_utils.argument_parser import create_parser

        parser = create_parser(8)

        args = parser.parse_args(['--modality', 'VISION'])
        assert args.modality == 'VISION'

        with pytest.raises(SystemExit):
            parser.parse_args(['--modality', 'INVALID'])


# ============================================================================
# TESTS: extract_traceback
# ============================================================================

class TestExtractTraceback:

    @pytest.mark.unit
    @pytest.mark.fast
    def test_extract_stderr_traceback_import(self):
        try:
            from sber_tests.test_utils.extract_traceback import extract_stderr_traceback
            assert callable(extract_stderr_traceback)
        except ImportError as e:
            pytest.skip(f"Extract traceback not available: {e}")

    @pytest.mark.unit
    @pytest.mark.fast
    def test_extract_stderr_traceback_with_fake_file(self):
        try:
            from sber_tests.test_utils.extract_traceback import extract_stderr_traceback

            result = extract_stderr_traceback("/fake/file/path.txt")

            assert result is None or result == ""

        except ImportError as e:
            pytest.skip(f"Extract traceback not available: {e}")


if __name__ == '__main__':
    pytest.main([__file__])
