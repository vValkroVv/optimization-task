from __future__ import annotations

S3_NOT_FOUND_ERROR_CODES = frozenset({"404", "NoSuchKey", "NotFound"})

S3_SERVICE_NAME = "s3"
S3_SIGNATURE_VERSION = "s3v4"
S3_ADDRESSING_STYLE = "virtual"
S3_PAYLOAD_SIGNING_ENABLED = False
S3_CHECKSUM_OPTIONS_MIN_BOTOCORE_VERSION = (1, 36)

LOCAL_DB_FILENAME = "database.sqlite"
TEMP_DIR_PREFIX = "job-submitter-db-"
