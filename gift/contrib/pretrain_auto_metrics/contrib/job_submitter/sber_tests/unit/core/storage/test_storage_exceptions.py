import pytest

from job_submitter.core.storage.exceptions import (
    DatabaseSchemaError,
    JobSubmitterDBError,
    RemoteQueryError,
    UnsafeQueryError,
)


pytestmark = pytest.mark.unit


def test_db_exceptions_hierarchy() -> None:
    err = DatabaseSchemaError("schema")
    assert isinstance(err, JobSubmitterDBError)

    err = RemoteQueryError("remote")
    assert isinstance(err, JobSubmitterDBError)

    err = UnsafeQueryError("unsafe")
    assert isinstance(err, RemoteQueryError)
    assert isinstance(err, JobSubmitterDBError)
