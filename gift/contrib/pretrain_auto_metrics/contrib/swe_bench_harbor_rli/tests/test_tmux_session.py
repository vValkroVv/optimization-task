from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest


def _load_tmux_session_module():
    harbor_module = types.ModuleType("harbor")
    agents_module = types.ModuleType("harbor.agents")
    terminus_pkg = types.ModuleType("harbor.agents.terminus_2")
    terminus_tmux_module = types.ModuleType("harbor.agents.terminus_2.tmux_session")

    class HarborTmuxSession:
        def __init__(
            self,
            session_name: str,
            environment,
            logging_path: Path,
            local_asciinema_recording_path: Path | None,
            remote_asciinema_recording_path: Path | None,
            pane_width: int = 160,
            pane_height: int = 40,
            extra_env: dict[str, str] | None = None,
            user: str | int | None = None,
        ) -> None:
            self._session_name = session_name
            self.environment = environment
            self._logging_path = logging_path
            self._local_asciinema_recording_path = local_asciinema_recording_path
            self._remote_asciinema_recording_path = remote_asciinema_recording_path
            self._pane_width = int(pane_width)
            self._pane_height = int(pane_height)
            self._extra_env = extra_env or {}
            self._user = user

        @property
        def _tmux_start_session(self) -> str:
            return (
                "tmux new-session -d "
                f"-s {self._session_name} "
                f"-x {self._pane_width} "
                f"-y {self._pane_height} "
                "'bash --login'"
            )

        async def _attempt_tmux_installation(self) -> None:
            return None

        async def start(self) -> None:
            await self._attempt_tmux_installation()
            start_session_result = await self.environment.exec(
                command=self._tmux_start_session,
                user=self._user,
            )
            if start_session_result.return_code != 0:
                raise RuntimeError("failed to start tmux")
            await self.environment.exec(
                command="tmux set-option -g history-limit 10000000",
                user=self._user,
            )

        async def send_keys(self, *args, **kwargs):
            _ = args, kwargs
            return None

        async def capture_pane(self, *args, **kwargs):
            _ = args, kwargs
            return ""

        async def get_incremental_output(self):
            return "Current Terminal Screen:\n"

    terminus_tmux_module.TmuxSession = HarborTmuxSession

    sys.modules["harbor"] = harbor_module
    sys.modules["harbor.agents"] = agents_module
    sys.modules["harbor.agents.terminus_2"] = terminus_pkg
    sys.modules["harbor.agents.terminus_2.tmux_session"] = terminus_tmux_module

    sys.modules.pop("swe_bench_harbor_rli.agents.terminus.tmux_session", None)
    return importlib.import_module(
        "swe_bench_harbor_rli.agents.terminus.tmux_session"
    )


def test_tmux_session_inherits_harbor_interactive_methods() -> None:
    tmux_session_module = _load_tmux_session_module()

    session_cls = tmux_session_module.TmuxSession

    assert hasattr(session_cls, "send_keys")
    assert hasattr(session_cls, "capture_pane")
    assert hasattr(session_cls, "get_incremental_output")


@pytest.mark.asyncio
async def test_start_installs_tmux_via_apt_when_missing() -> None:
    tmux_session_module = _load_tmux_session_module()

    calls: list[str] = []
    results = [
        type("ExecResult", (), {"stdout": "", "stderr": "", "return_code": 127})(),
        type("ExecResult", (), {"stdout": "/usr/bin/apt-get\n", "stderr": "", "return_code": 0})(),
        type("ExecResult", (), {"stdout": "", "stderr": "", "return_code": 0})(),
        type("ExecResult", (), {"stdout": "", "stderr": "", "return_code": 0})(),
        type("ExecResult", (), {"stdout": "", "stderr": "", "return_code": 0})(),
        type("ExecResult", (), {"stdout": "tmux 3.4", "stderr": "", "return_code": 0})(),
        type("ExecResult", (), {"stdout": "", "stderr": "", "return_code": 0})(),
        type("ExecResult", (), {"stdout": "", "stderr": "", "return_code": 0})(),
    ]

    async def fake_exec(self, command: str, user: str | None = None, **kwargs):
        _ = self, user, kwargs
        calls.append(command)
        return results.pop(0)

    env = type(
        "FakeEnvironment",
        (),
        {
            "session_id": "sess-123",
            "default_user": "root",
            "exec": fake_exec,
        },
    )()

    session = tmux_session_module.TmuxSession(
        session_name="terminus_2",
        environment=env,
        logging_path=Path("/tmp/terminus_2.pane"),
        local_asciinema_recording_path=None,
        remote_asciinema_recording_path=None,
        user="root",
    )

    await session.start()

    assert calls[0] == "tmux -V"
    assert calls[1] == "command -v apt-get >/dev/null 2>&1"
    assert calls[2] == "apt-get clean"
    assert calls[3] == (
        "DEBIAN_FRONTEND=noninteractive apt-get update || "
        "DEBIAN_FRONTEND=noninteractive apt-get update"
    )
    assert calls[4] == "DEBIAN_FRONTEND=noninteractive apt-get install -y tmux"
    assert calls[5] == "tmux -V"
    assert "tmux new-session" in calls[6]
    assert calls[7] == "tmux set-option -g history-limit 10000000"


@pytest.mark.asyncio
async def test_start_raises_when_tmux_is_missing_and_apt_is_unavailable() -> None:
    tmux_session_module = _load_tmux_session_module()

    calls: list[str] = []
    results = [
        type("ExecResult", (), {"stdout": "", "stderr": "", "return_code": 127})(),
        type("ExecResult", (), {"stdout": "", "stderr": "", "return_code": 1})(),
    ]

    async def fake_exec(self, command: str, user: str | None = None, **kwargs):
        _ = self, user, kwargs
        calls.append(command)
        return results.pop(0)

    env = type(
        "FakeEnvironment",
        (),
        {
            "session_id": "sess-123",
            "default_user": "root",
            "exec": fake_exec,
        },
    )()

    session = tmux_session_module.TmuxSession(
        session_name="terminus_2",
        environment=env,
        logging_path=Path("/tmp/terminus_2.pane"),
        local_asciinema_recording_path=None,
        remote_asciinema_recording_path=None,
        user="root",
    )

    with pytest.raises(RuntimeError, match="tmux is not installed"):
        await session.start()

    assert calls == ["tmux -V", "command -v apt-get >/dev/null 2>&1"]
