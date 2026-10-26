from __future__ import annotations

import asyncio
import importlib
import json
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest

_IMPORT_BLOCK = "from swebench.harness.test_spec.test_spec import make_test_spec"
_CALL_BLOCK = "test_spec   = make_test_spec(datum)"
_EXEC_BLOCK = """set +e
export PATH="/root/.local/bin:$PATH"
uv run parser.py | tee -a "$LOG_FILE"
exit_code=$?
set -e"""
_INSTALL_ONLY_BOUNDARY = (
    "# First we reset tracked files touched by the SWE-bench test patch."
)
_TEST_SCRIPT = f"""#!/bin/bash
{_IMPORT_BLOCK}
{_CALL_BLOCK}
{_EXEC_BLOCK}
{_INSTALL_ONLY_BOUNDARY}
"""


@dataclass
class _FakeVerifierResult:
    rewards: dict[str, float]


class _FakeEnvironment:
    verifier_timeout_sec: int | None = None

    @contextmanager
    def use_verifier_command_timeout(self, timeout_sec: int):
        self.verifier_timeout_sec = timeout_sec
        yield


class _FakeVerifier:
    environment = _FakeEnvironment()
    task = type(
        "Task",
        (),
        {
            "config": type(
                "TaskConfig",
                (),
                {
                    "verifier": type(
                        "VerifierConfig",
                        (),
                        {"timeout_sec": 3_600},
                    )()
                },
            )()
        },
    )()

    def _resolve_tests(self) -> tuple[list[Path], Path, Path]:
        return self.source_dirs, self.tests_source_dir, self.host_test_path

    async def verify(self) -> _FakeVerifierResult:
        source_dirs, tests_source_dir, host_test_path = self._resolve_tests()
        self.delegated_source_dirs = source_dirs
        self.delegated_tests_source_dir = tests_source_dir
        self.delegated_test_text = host_test_path.read_text(encoding="utf-8")
        self.delegated_config_text = (tests_source_dir / "config.json").read_text(
            encoding="utf-8"
        )
        return _FakeVerifierResult(rewards={"reward": 1.0})


@pytest.fixture
def verifier_module(monkeypatch):
    harbor_module = types.ModuleType("harbor")
    models_module = types.ModuleType("harbor.models")
    models_verifier_module = types.ModuleType("harbor.models.verifier")
    result_module = types.ModuleType("harbor.models.verifier.result")
    harbor_verifier_module = types.ModuleType("harbor.verifier")
    verifier_module = types.ModuleType("harbor.verifier.verifier")

    result_module.VerifierResult = _FakeVerifierResult
    verifier_module.Verifier = _FakeVerifier

    modules = {
        "harbor": harbor_module,
        "harbor.models": models_module,
        "harbor.models.verifier": models_verifier_module,
        "harbor.models.verifier.result": result_module,
        "harbor.verifier": harbor_verifier_module,
        "harbor.verifier.verifier": verifier_module,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "swe_bench_harbor_rli.src.verifier"
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    yield module
    sys.modules.pop(module_name, None)


def test_public_import_path_exposes_harbor_verifier(verifier_module) -> None:
    assert verifier_module.SWEBenchLocalRLIVerifier.__module__ == (
        "swe_bench_harbor_rli.src.verifier"
    )
    assert issubclass(verifier_module.SWEBenchLocalRLIVerifier, _FakeVerifier)


def test_prepare_test_script_uses_local_spec_and_runtime_archive(
    verifier_module,
) -> None:
    prepared = verifier_module.prepare_test_script(_TEST_SCRIPT)

    assert "make_test_spec" not in prepared
    assert "test_spec = TestSpec(" in prepared
    assert "/usr/local/share/.dev_logs/test_runtime.tar.zst" in prepared
    assert 'if [ ! -r "$runtime_archive" ]' in prepared
    assert 'tar --zstd -xf "$runtime_archive" -C /tmp' in prepared
    assert '"$runtime_dir/bin/python" parser.py' in prepared
    assert "uv run parser.py" not in prepared


@pytest.mark.parametrize(
    "name, block",
    [
        ("import", _IMPORT_BLOCK),
        ("call", _CALL_BLOCK),
        ("execution", _EXEC_BLOCK),
    ],
)
def test_prepare_test_script_rejects_missing_required_block(
    verifier_module,
    name: str,
    block: str,
) -> None:
    with pytest.raises(ValueError, match="found 0"):
        verifier_module.prepare_test_script(_TEST_SCRIPT.replace(block, ""))


@pytest.mark.parametrize(
    "name, block",
    [
        ("import", _IMPORT_BLOCK),
        ("call", _CALL_BLOCK),
        ("execution", _EXEC_BLOCK),
    ],
)
def test_prepare_test_script_rejects_duplicated_required_block(
    verifier_module,
    name: str,
    block: str,
) -> None:
    with pytest.raises(ValueError, match="found 2"):
        verifier_module.prepare_test_script(f"{_TEST_SCRIPT}\n{block}")


@pytest.mark.parametrize(
    "fail_to_pass, pass_to_pass",
    [
        (["test_fail"], ["test_pass"]),
        ('["test_fail"]', '["test_pass"]'),
    ],
    ids=["lists", "json-strings"],
)
def test_generated_test_spec_accepts_list_and_json_values(
    verifier_module,
    fail_to_pass,
    pass_to_pass,
) -> None:
    class TestSpec:
        def __init__(self, **values):
            self.values = values

    namespace = {
        "json": json,
        "datum": {
            "instance_id": "django__django-13343",
            "repo": "django/django",
            "version": "3.2",
            "FAIL_TO_PASS": fail_to_pass,
            "PASS_TO_PASS": pass_to_pass,
        },
        "TestSpec": TestSpec,
        "KEY_INSTANCE_ID": "instance_id",
        "FAIL_TO_PASS": "FAIL_TO_PASS",
        "PASS_TO_PASS": "PASS_TO_PASS",
    }

    prepared = verifier_module.prepare_test_script(_TEST_SCRIPT)
    start = prepared.index("def _test_list")
    end = prepared.index("\nruntime_archive=", start)
    exec(prepared[start:end], namespace)

    values = namespace["test_spec"].values
    assert values["instance_id"] == "django__django-13343"
    assert values["FAIL_TO_PASS"] == ["test_fail"]
    assert values["PASS_TO_PASS"] == ["test_pass"]
    assert values["repo_script_list"] == []
    assert values["env_script_list"] == []
    assert values["eval_script_list"] == []


def test_verify_prepares_copy_and_delegates_to_harbor(
    verifier_module,
    tmp_path: Path,
) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_path = tests_dir / "test.sh"
    config_path = tests_dir / "config.json"
    test_path.write_text(_TEST_SCRIPT, encoding="utf-8")
    config_path.write_text('{"instance_id": "django__django-13343"}', encoding="utf-8")

    verifier = object.__new__(verifier_module.SWEBenchLocalRLIVerifier)
    verifier.source_dirs = [tests_dir]
    verifier.tests_source_dir = tests_dir
    verifier.host_test_path = test_path

    result = asyncio.run(verifier.verify())

    assert result.rewards == {"reward": 1.0}
    assert test_path.read_text(encoding="utf-8") == _TEST_SCRIPT
    assert verifier.delegated_source_dirs == [verifier.delegated_tests_source_dir]
    assert "make_test_spec" not in verifier.delegated_test_text
    assert "test_spec = TestSpec(" in verifier.delegated_test_text
    assert verifier.delegated_config_text == ('{"instance_id": "django__django-13343"}')
    assert verifier.environment.verifier_timeout_sec == 3_600


def test_verify_corrects_astropy_7606_test_id(
    verifier_module,
    tmp_path: Path,
) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_path = tests_dir / "test.sh"
    test_path.write_text(_TEST_SCRIPT, encoding="utf-8")
    old = "astropy/units/tests/test_units.py::test_compose_roundtrip[]"
    new = "astropy/units/tests/test_units.py::test_compose_roundtrip[unit0]"
    (tests_dir / "config.json").write_text(
        json.dumps(
            {
                "instance_id": "astropy__astropy-7606",
                "PASS_TO_PASS": json.dumps([old]),
            }
        ),
        encoding="utf-8",
    )

    verifier = object.__new__(verifier_module.SWEBenchLocalRLIVerifier)
    verifier.source_dirs = [tests_dir]
    verifier.tests_source_dir = tests_dir
    verifier.host_test_path = test_path

    result = asyncio.run(verifier.verify())

    assert result.rewards == {"reward": 1.0}
    delegated_config = json.loads(verifier.delegated_config_text)
    assert json.loads(delegated_config["PASS_TO_PASS"]) == [new]


def test_verify_enables_local_httpbin_for_legacy_requests(
    verifier_module,
    tmp_path: Path,
) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_path = tests_dir / "test.sh"
    test_path.write_text(
        _TEST_SCRIPT
        + "\n#Then we run all the tests in the repository"
        + "\n# And we reset the test files back to the base commit.\n",
        encoding="utf-8",
    )
    (tests_dir / "config.json").write_text(
        json.dumps(
            {
                "instance_id": "psf__requests-2317",
                "repo": "psf/requests",
                "version": "2.4",
            }
        ),
        encoding="utf-8",
    )

    verifier = object.__new__(verifier_module.SWEBenchLocalRLIVerifier)
    verifier.source_dirs = [tests_dir]
    verifier.tests_source_dir = tests_dir
    verifier.host_test_path = test_path

    result = asyncio.run(verifier.verify())

    assert result.rewards == {"reward": 1.0}
    assert "HTTPBIN_URL=http://127.0.0.1:18080/" in verifier.delegated_test_text
    assert "HTTPSBIN_URL=https://127.0.0.1:18080/" in verifier.delegated_test_text
    assert '"$httpbin_runtime_dir/bin/gunicorn"' in verifier.delegated_test_text
