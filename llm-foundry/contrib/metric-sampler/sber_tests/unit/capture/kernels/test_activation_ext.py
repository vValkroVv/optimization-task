from __future__ import annotations

import importlib.machinery
import importlib.util
import logging
import re
from pathlib import Path

import pytest

from metric_sampler.capture.kernels import activation_ext

pytestmark = pytest.mark.unit


def _spec_of_python_file(name: str, path: Path) -> importlib.machinery.ModuleSpec | None:
    return importlib.util.spec_from_loader(name, importlib.machinery.SourceFileLoader(name, str(path)))


def test_build_directory_is_keyed_by_the_sources_hash(tmp_path: Path) -> None:
    directory = activation_ext.build_directory(tmp_path)

    assert directory.parent == tmp_path
    assert len(directory.name) == 12
    assert activation_ext.build_directory(tmp_path) == directory


@pytest.fixture
def csrc(monkeypatch, tmp_path: Path) -> Path:
    directory = tmp_path / "csrc"
    directory.mkdir()
    for name in ("first.cpp", "second.cu", "shared.h", "device.cuh", "notes.txt"):
        (directory / name).write_text(f"contents of {name}")
    monkeypatch.setattr(activation_ext, "_CSRC_DIR", directory)
    monkeypatch.setattr(activation_ext, "_SOURCES", ("first.cpp", "second.cu"))
    return directory


@pytest.mark.parametrize("changed", ["first.cpp", "second.cu", "shared.h", "device.cuh"])
def test_build_directory_is_keyed_by_every_source_and_header(csrc: Path, tmp_path: Path, changed: str) -> None:
    directory = activation_ext.build_directory(tmp_path / "build")

    (csrc / changed).write_text("changed")

    assert activation_ext.build_directory(tmp_path / "build") != directory


def test_build_directory_is_not_keyed_by_other_files_of_csrc(csrc: Path, tmp_path: Path) -> None:
    directory = activation_ext.build_directory(tmp_path / "build")

    (csrc / "notes.txt").write_text("changed")

    assert activation_ext.build_directory(tmp_path / "build") == directory


def test_build_directory_is_keyed_by_which_file_holds_a_text(csrc: Path, tmp_path: Path) -> None:
    directory = activation_ext.build_directory(tmp_path / "build")

    (csrc / "first.cpp").write_text("contents of second.cu")
    (csrc / "second.cu").write_text("contents of first.cpp")

    assert activation_ext.build_directory(tmp_path / "build") != directory


def test_build_directory_is_keyed_by_the_build_flags(monkeypatch, tmp_path: Path) -> None:
    directory = activation_ext.build_directory(tmp_path)
    monkeypatch.setattr(activation_ext, "_CXX_FLAGS", ())

    assert activation_ext.build_directory(tmp_path) != directory


def test_build_directory_defaults_to_the_env_var_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("METRIC_SAMPLER_ACTIVATION_EXT_DIR", str(tmp_path / "from-env"))

    assert activation_ext.build_directory().parent == tmp_path / "from-env"


def test_build_directory_defaults_to_the_repo_build_dir_without_the_env_var(monkeypatch) -> None:
    monkeypatch.delenv("METRIC_SAMPLER_ACTIVATION_EXT_DIR", raising=False)

    assert activation_ext.build_directory().parent.parts[-2:] == ("build", "activation_ext")


def test_build_compiles_the_sources_into_the_hashed_directory(monkeypatch, tmp_path: Path) -> None:
    captured = {}
    sentinel = object()
    monkeypatch.setattr("torch.utils.cpp_extension.load", lambda **kwargs: captured.update(kwargs) or sentinel)

    assert activation_ext.build(tmp_path) is sentinel
    assert captured["name"] == "metric_sampler_activation_ext"
    assert captured["build_directory"] == str(activation_ext.build_directory(tmp_path))
    assert Path(captured["build_directory"]).is_dir()
    csrc_files = (Path(activation_ext.__file__).parents[1] / "csrc").iterdir()
    assert sorted(Path(source).name for source in captured["sources"]) == sorted(
        path.name for path in csrc_files if path.suffix in (".cpp", ".cu")
    )
    # The bindings, compiled without an optimization level of the JIT build's own, are optimized too.
    assert captured["extra_cflags"] == ["-O3"]
    assert captured["extra_cuda_cflags"] == ["-O3", "--expt-extended-lambda"]


def test_every_file_a_csrc_file_includes_is_in_csrc() -> None:
    sources = Path(activation_ext.__file__).parents[1] / "csrc"
    includes = [
        (path.name, name)
        for path in sorted(sources.iterdir())
        for name in re.findall(r'^#include "([^"]+)"', path.read_text(), re.MULTILINE)
    ]

    assert includes
    assert [(path, name) for path, name in includes if not (sources / name).is_file()] == []


def test_load_imports_the_prebuilt_module(tmp_path: Path) -> None:
    directory = activation_ext.build_directory(tmp_path)
    directory.mkdir(parents=True)
    # A plain Python file stands in for the compiled .so: importlib loads both the same way.
    (directory / "metric_sampler_activation_ext.so").write_text("def observe():\n    return 'prebuilt'\n")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("importlib.util.spec_from_file_location", _spec_of_python_file)
        module = activation_ext.load(tmp_path)

    assert module.observe() == "prebuilt"


def test_load_without_a_prebuilt_module_warns_once_and_returns_none(tmp_path: Path, caplog) -> None:
    with caplog.at_level(logging.WARNING):
        assert activation_ext.load(tmp_path) is None

    assert [record.message for record in caplog.records] == [
        "activation_ext: no prebuilt extension for these sources, using the Triton kernel"
    ]


def test_load_of_a_broken_module_warns_once_and_returns_none(tmp_path: Path, caplog) -> None:
    directory = activation_ext.build_directory(tmp_path)
    directory.mkdir(parents=True)
    (directory / "metric_sampler_activation_ext.so").write_bytes(b"not a shared object")

    with caplog.at_level(logging.WARNING):
        assert activation_ext.load(tmp_path) is None

    assert len(caplog.records) == 1
    assert caplog.records[0].message.startswith("activation_ext: loading the prebuilt extension failed")


def test_prebuilt_loads_once_per_process(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(activation_ext, "load", lambda: calls.append(1) or "module")
    activation_ext.prebuilt.cache_clear()
    try:
        assert [activation_ext.prebuilt(), activation_ext.prebuilt()] == ["module", "module"]
    finally:
        activation_ext.prebuilt.cache_clear()

    assert calls == [1]
