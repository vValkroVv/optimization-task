from __future__ import annotations

import asyncio
import importlib
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest


@dataclass
class _FakeVerifierResult:
    rewards: dict[str, float]


class _FakeEnvironment:
    def __init__(self):
        self.commands = []
        self.verifier_timeout_active = False
        self.verifier_timeout_sec = None

    async def exec(self, *, command, user):
        self.commands.append((command, user))
        return types.SimpleNamespace(return_code=0, stdout="", stderr="")

    @contextmanager
    def use_verifier_command_timeout(self, timeout_sec):
        self.verifier_timeout_active = True
        self.verifier_timeout_sec = timeout_sec
        try:
            yield
        finally:
            self.verifier_timeout_active = False


class _FakeVerifier:
    environment = _FakeEnvironment()

    def _resolve_tests(self) -> tuple[list[Path], Path, Path]:
        return self.source_dirs, self.tests_source_dir, self.host_test_path

    async def verify(self) -> _FakeVerifierResult:
        assert self.environment.verifier_timeout_active
        source_dirs, tests_source_dir, host_test_path = self._resolve_tests()
        self.delegated_source_dirs = source_dirs
        self.delegated_tests_source_dir = tests_source_dir
        self.delegated_test_text = host_test_path.read_text(encoding="utf-8")
        self.delegated_private_test = (
            tests_source_dir / "private_test.tar.gz"
        ).read_bytes()
        self.delegated_python_test = (tests_source_dir / "test_outputs.py").read_text(
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

    module_name = "terminal_bench.src.verifier"
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    yield module
    sys.modules.pop(module_name, None)


def test_verify_prepares_complete_tests_copy_and_delegates(
    verifier_module, tmp_path: Path
) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_path = tests_dir / "test.sh"
    original = """#!/bin/bash
apt-get update
apt-get install -y curl
curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh
source $HOME/.local/bin/env
uvx -p 3.13 -w pytest==8.4.1 pytest /tests/test_outputs.py
"""
    test_path.write_text(original, encoding="utf-8")
    (tests_dir / "private_test.tar.gz").write_bytes(b"private")
    (tests_dir / "test_outputs.py").write_text(
        'open("/tests/private_test.txt")\n',
        encoding="utf-8",
    )

    verifier = object.__new__(verifier_module.TerminalBenchLocalRLIVerifier)
    verifier.task = types.SimpleNamespace(
        name="terminal-bench/example",
        config=types.SimpleNamespace(
            verifier=types.SimpleNamespace(timeout_sec=3600.0)
        ),
    )
    verifier.environment = _FakeEnvironment()
    verifier.source_dirs = [tests_dir]
    verifier.tests_source_dir = tests_dir
    verifier.host_test_path = test_path

    result = asyncio.run(verifier.verify())

    assert result.rewards == {"reward": 1.0}
    assert test_path.read_text(encoding="utf-8") == original
    assert verifier.delegated_source_dirs == [verifier.delegated_tests_source_dir]
    assert "apt-get" not in verifier.delegated_test_text
    assert "astral.sh" not in verifier.delegated_test_text
    assert 'UV_OFFLINE="${UV_OFFLINE:-1}"' in verifier.delegated_test_text
    assert "/tmp/terminal-bench/tests/test_outputs.py" in (verifier.delegated_test_text)
    assert verifier.environment.verifier_timeout_sec == 3600.0
    assert verifier.delegated_private_test == b"private"
    assert (
        'open("/tmp/terminal-bench/tests/private_test.txt")'
        in verifier.delegated_python_test
    )


def test_verify_adapts_mailman_to_unprivileged_smtp_port(
    verifier_module, tmp_path: Path
) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_path = tests_dir / "test.sh"
    test_path.write_text(
        "#!/bin/bash\npytest /tests/test_outputs.py\n",
        encoding="utf-8",
    )
    (tests_dir / "private_test.tar.gz").write_bytes(b"private")
    (tests_dir / "test_outputs.py").write_text(
        'smtplib.SMTP("localhost", 25)\n',
        encoding="utf-8",
    )

    verifier = object.__new__(verifier_module.TerminalBenchLocalRLIVerifier)
    verifier.task = types.SimpleNamespace(
        name="terminal-bench/mailman",
        config=types.SimpleNamespace(
            verifier=types.SimpleNamespace(timeout_sec=1800.0)
        ),
    )
    verifier.environment = _FakeEnvironment()
    verifier.source_dirs = [tests_dir]
    verifier.tests_source_dir = tests_dir
    verifier.host_test_path = test_path

    result = asyncio.run(verifier.verify())

    assert result.rewards == {"reward": 1.0}
    assert 'SMTP("localhost", 2525)' in verifier.delegated_python_test
    command, user = verifier.environment.commands[0]
    assert "smtp_port: 2525" in command
    assert "{4})y([[:space:]])/\\1n\\3/" in command
    assert "service postfix restart" in command
    assert user == "root"
