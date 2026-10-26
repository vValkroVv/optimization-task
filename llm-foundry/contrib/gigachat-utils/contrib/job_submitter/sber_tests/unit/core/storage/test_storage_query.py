import pytest

from job_submitter.core.storage.query import QueryResult


pytestmark = pytest.mark.unit


def test_query_result_dataclass_fields() -> None:
    result = QueryResult(columns=("id", "name"), rows=[(1, "a")])

    assert result.columns == ("id", "name")
    assert result.rows == [(1, "a")]
