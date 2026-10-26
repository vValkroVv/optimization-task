import pytest

from job_submitter.core.storage.exceptions import JobSubmitterDBError
from job_submitter.infrastructure.s3.exceptions import (
    ConcurrentWriteError,
    DatabaseSyncError,
)


pytestmark = pytest.mark.unit


def test_sync_exceptions_hierarchy() -> None:
    err = DatabaseSyncError("sync")
    assert isinstance(err, JobSubmitterDBError)

    err = ConcurrentWriteError("conflict")
    assert isinstance(err, DatabaseSyncError)
    assert isinstance(err, JobSubmitterDBError)
