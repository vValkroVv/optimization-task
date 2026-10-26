import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.mark.unit
def test_package_init_skips_sys_path_insert_when_root_already_present() -> None:
    init_path = Path(__file__).resolve().parents[2] / "job_submitter" / "__init__.py"
    source = init_path.read_text(encoding="utf-8")
    root = str(init_path.resolve().parent.parent)
    sqlite_root_path = Path(root) / "contrib" / "sqlite_s3_query"
    sqlite_pkg_path = sqlite_root_path / "sqlite_s3_query"
    sqlite_root = str(sqlite_root_path)

    old_sys_path = sys.path.copy()
    try:
        if root not in sys.path:
            sys.path.insert(0, root)
        if sqlite_pkg_path.is_dir() and sqlite_root not in sys.path:
            sys.path.insert(0, sqlite_root)
        before = list(sys.path)
        code = compile(source, str(init_path), "exec")
        exec(  # noqa: S102
            code,
            {
                "__name__": "contrib.job_submitter",
                "__file__": str(init_path),
                "__package__": "contrib",
            },
        )
        assert sys.path == before
    finally:
        sys.path[:] = old_sys_path


@pytest.mark.unit
def test_contrib_import_adds_repo_root_to_sys_path() -> None:
    init_file = Path(__file__).resolve().parents[2] / "job_submitter" / "__init__.py"
    module_name = "contrib.job_submitter_test"
    expected_root = str(init_file.resolve().parent.parent)
    expected_root_resolved = str(Path(expected_root).resolve())
    sys.path[:] = [
        p for p in sys.path if str(Path(p).resolve()) != expected_root_resolved
    ]

    spec = importlib.util.spec_from_file_location(module_name, init_file)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    old_sys_path = sys.path.copy()
    try:
        spec.loader.exec_module(module)
        assert expected_root in sys.path
    finally:
        sys.path[:] = old_sys_path


@pytest.mark.unit
def test_contrib_import_adds_sqlite_submodule_to_sys_path_when_present() -> None:
    init_file = Path(__file__).resolve().parents[2] / "job_submitter" / "__init__.py"
    module_name = "contrib.job_submitter_sqlite_submodule_test"
    expected_root = init_file.resolve().parent.parent
    expected_sqlite_root = expected_root / "contrib" / "sqlite_s3_query"
    expected_sqlite_pkg = expected_sqlite_root / "sqlite_s3_query"
    if not expected_sqlite_pkg.is_dir():
        pytest.skip("sqlite_s3_query submodule is not available in this checkout")

    expected_sqlite_root_str = str(expected_sqlite_root)
    expected_sqlite_root_resolved = str(expected_sqlite_root.resolve())
    sys.path[:] = [
        p for p in sys.path if str(Path(p).resolve()) != expected_sqlite_root_resolved
    ]

    spec = importlib.util.spec_from_file_location(module_name, init_file)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    old_sys_path = sys.path.copy()
    try:
        spec.loader.exec_module(module)
        assert expected_sqlite_root_str in sys.path
    finally:
        sys.path[:] = old_sys_path


@pytest.mark.unit
def test_plain_import_adds_sqlite_submodule_for_manual_sys_path_case() -> None:
    init_file = Path(__file__).resolve().parents[2] / "job_submitter" / "__init__.py"
    expected_root = init_file.resolve().parent.parent
    expected_sqlite_root = expected_root / "contrib" / "sqlite_s3_query"
    expected_sqlite_pkg = expected_sqlite_root / "sqlite_s3_query"
    if not expected_sqlite_pkg.is_dir():
        pytest.skip("sqlite_s3_query submodule is not available in this checkout")

    expected_sqlite_root_str = str(expected_sqlite_root)
    expected_sqlite_root_resolved = str(expected_sqlite_root.resolve())
    sys.path[:] = [
        p for p in sys.path if str(Path(p).resolve()) != expected_sqlite_root_resolved
    ]

    source = init_file.read_text(encoding="utf-8")
    old_sys_path = sys.path.copy()
    try:
        exec(  # noqa: S102
            compile(source, str(init_file), "exec"),
            {
                "__name__": "job_submitter",
                "__file__": str(init_file),
                "__package__": "job_submitter",
            },
        )
        assert expected_sqlite_root_str in sys.path
    finally:
        sys.path[:] = old_sys_path


@pytest.mark.unit
def test_contrib_import_skips_sqlite_sys_path_when_submodule_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_file = Path(__file__).resolve().parents[2] / "job_submitter" / "__init__.py"
    sqlite_pkg = (
        init_file.resolve().parent.parent
        / "contrib"
        / "sqlite_s3_query"
        / "sqlite_s3_query"
    )
    original_is_dir = Path.is_dir

    def patched_is_dir(self: Path) -> bool:
        if self.resolve() == sqlite_pkg.resolve():
            return False
        return original_is_dir(self)

    monkeypatch.setattr(Path, "is_dir", patched_is_dir)

    module_name = "contrib.job_submitter_sqlite_absent_test"
    spec = importlib.util.spec_from_file_location(module_name, init_file)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    old_sys_path = sys.path.copy()
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_sys_path
