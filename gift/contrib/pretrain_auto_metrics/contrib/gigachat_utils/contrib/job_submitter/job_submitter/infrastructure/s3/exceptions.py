from job_submitter.core.storage.exceptions import JobSubmitterDBError


class DatabaseSyncError(JobSubmitterDBError):
    """Remote <-> local database synchronization failed."""


class ConcurrentWriteError(DatabaseSyncError):
    """Remote database changed between download and upload."""
