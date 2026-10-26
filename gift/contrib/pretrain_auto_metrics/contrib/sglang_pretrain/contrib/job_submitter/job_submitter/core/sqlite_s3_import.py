from typing import NoReturn


SQLITE_S3_QUERY_IMPORT_HINT = (
    "sqlite_s3_query is not available. "
    "Initialize the sqlite_s3_query submodule before using S3-backed tracking. "
    "Standalone job-submitter checkout: "
    "`git submodule update --init --recursive contrib/sqlite_s3_query`. "
    "If job_submitter is vendored as contrib/job_submitter in another repo, "
    "or you import it via sys.path.insert(.../contrib/job_submitter), "
    "also initialize the nested contrib: "
    "`git submodule update --init --recursive "
    "contrib/job_submitter/contrib/sqlite_s3_query` "
    "or run from the parent repo: "
    "`git -C contrib/job_submitter submodule update --init --recursive "
    "contrib/sqlite_s3_query`."
)


def is_sqlite_s3_query_missing(exc: ModuleNotFoundError) -> bool:
    missing_name = getattr(exc, "name", None)
    if missing_name:
        return missing_name == "sqlite_s3_query" or missing_name.startswith(
            "sqlite_s3_query."
        )
    return "sqlite_s3_query" in str(exc)


def raise_sqlite_s3_query_import_error(exc: ModuleNotFoundError) -> NoReturn:
    if not is_sqlite_s3_query_missing(exc):
        raise exc
    raise ModuleNotFoundError(SQLITE_S3_QUERY_IMPORT_HINT) from exc
