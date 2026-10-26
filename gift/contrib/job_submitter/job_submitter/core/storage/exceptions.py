class JobSubmitterDBError(Exception):
    """Base exception for the DB integration layer."""


class DatabaseSchemaError(JobSubmitterDBError):
    """Schema creation or migration failed."""


class RemoteQueryError(JobSubmitterDBError):
    """Remote database query failed."""


class UnsafeQueryError(RemoteQueryError):
    """Unsafe SQL identifier passed to query builder."""
