import pytest

from job_submitter.core.sqlite_s3_import import (
    is_sqlite_s3_query_missing,
    raise_sqlite_s3_query_import_error,
)


pytestmark = pytest.mark.unit


def test_detects_missing_sqlite_s3_query_by_name() -> None:
    exc = ModuleNotFoundError("No module named sqlite_s3_query")
    exc.name = "sqlite_s3_query"
    assert is_sqlite_s3_query_missing(exc) is True


def test_detects_missing_sqlite_s3_query_by_message_fallback() -> None:
    assert is_sqlite_s3_query_missing(ModuleNotFoundError("sqlite_s3_query")) is True


def test_does_not_mask_unrelated_missing_dependency() -> None:
    exc = ModuleNotFoundError("No module named not_sqlite_dep")
    exc.name = "not_sqlite_dep"

    with pytest.raises(ModuleNotFoundError) as exc_info:
        raise_sqlite_s3_query_import_error(exc)

    assert exc_info.value is exc


def test_raises_helpful_message_for_missing_sqlite_s3_query() -> None:
    exc = ModuleNotFoundError("No module named sqlite_s3_query")
    exc.name = "sqlite_s3_query"

    with pytest.raises(ModuleNotFoundError) as exc_info:
        raise_sqlite_s3_query_import_error(exc)

    assert "sqlite_s3_query is not available" in str(exc_info.value)
    assert "contrib/job_submitter/contrib/sqlite_s3_query" in str(exc_info.value)
