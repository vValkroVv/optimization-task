from __future__ import annotations

import sys
from io import StringIO

import pytest

from swe_bench_harbor_rli.src import apply_patch_shim


def _run_patch(monkeypatch, patch: str, *arguments: str) -> None:
    monkeypatch.setattr(sys, "argv", ["apply_patch", *arguments])
    monkeypatch.setattr(sys, "stdin", StringIO(patch))
    apply_patch_shim.main()


def test_main_applies_update_add_and_delete(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "existing.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    (tmp_path / "obsolete.txt").write_text("old\n", encoding="utf-8")
    patch = """*** Begin Patch
*** Update File: existing.txt
@@
 alpha
-beta
+gamma
*** Add File: nested/added.txt
+first
+second
*** Delete File: obsolete.txt
*** End Patch
"""

    _run_patch(monkeypatch, patch)

    assert (tmp_path / "existing.txt").read_text(encoding="utf-8") == ("alpha\ngamma\n")
    assert (tmp_path / "nested" / "added.txt").read_text(encoding="utf-8") == (
        "first\nsecond\n"
    )
    assert not (tmp_path / "obsolete.txt").exists()


def test_main_reads_patch_from_argument(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    patch_path = tmp_path / "change.patch"
    patch_path.write_text(
        """*** Begin Patch
*** Add File: created.txt
+created from file
*** End Patch
""",
        encoding="utf-8",
    )

    _run_patch(monkeypatch, "", str(patch_path))

    assert (tmp_path / "created.txt").read_text(encoding="utf-8") == (
        "created from file\n"
    )


@pytest.mark.parametrize(
    "patch, message",
    [
        ("not a patch", "missing '\\*\\*\\* Begin Patch'"),
        (
            """*** Begin Patch
*** Update File: existing.txt
@@
-missing
+replacement
*** End Patch
""",
            "hunk not found",
        ),
    ],
    ids=["missing-envelope", "missing-hunk"],
)
def test_main_rejects_invalid_patch_without_modifying_files(
    monkeypatch,
    tmp_path,
    patch: str,
    message: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "existing.txt"
    target.write_text("original\n", encoding="utf-8")

    with pytest.raises(SystemExit, match=message):
        _run_patch(monkeypatch, patch)

    assert target.read_text(encoding="utf-8") == "original\n"
