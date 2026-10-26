import sys
from pathlib import Path


def _prepend_sys_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


_root = Path(__file__).resolve().parent.parent

if __name__.startswith("contrib.") or __name__.startswith("sber_src.contribs."):
    _prepend_sys_path(_root)

# Support running from a vendored checkout without editable-install, including
# manual `sys.path.insert(.../contrib/job_submitter); import job_submitter`.
_sqlite_s3_query_root = _root / "contrib" / "sqlite_s3_query"
_sqlite_s3_query_pkg = _sqlite_s3_query_root / "sqlite_s3_query"
if _sqlite_s3_query_pkg.is_dir():
    _prepend_sys_path(_sqlite_s3_query_root)
