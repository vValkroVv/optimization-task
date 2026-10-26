from __future__ import annotations

from harbor.agents.terminus_2.tmux_session import TmuxSession as HarborTmuxSession


class TmuxSession(HarborTmuxSession):
    async def _attempt_tmux_installation(self) -> None:
        tmux_result = await self.environment.exec(command="tmux -V", user="root")
        if tmux_result.return_code == 0:
            return

        apt_check = await self.environment.exec(
            command="command -v apt-get >/dev/null 2>&1",
            user="root",
        )
        if apt_check.return_code != 0:
            session_id = getattr(self.environment, "session_id", "<unknown>")
            raise RuntimeError(f"tmux is not installed for session {session_id}")

        clean_result = await self.environment.exec(
            command="apt-get clean",
            user="root",
        )
        if clean_result.return_code != 0:
            session_id = getattr(self.environment, "session_id", "<unknown>")
            raise RuntimeError(f"tmux is not installed for session {session_id}")

        update_result = await self.environment.exec(
            command=(
                "DEBIAN_FRONTEND=noninteractive apt-get update || "
                "DEBIAN_FRONTEND=noninteractive apt-get update"
            ),
            user="root",
        )
        if update_result.return_code != 0:
            session_id = getattr(self.environment, "session_id", "<unknown>")
            raise RuntimeError(f"tmux is not installed for session {session_id}")

        install_result = await self.environment.exec(
            command="DEBIAN_FRONTEND=noninteractive apt-get install -y tmux",
            user="root",
        )
        if install_result.return_code != 0:
            session_id = getattr(self.environment, "session_id", "<unknown>")
            raise RuntimeError(f"tmux is not installed for session {session_id}")

        verify_result = await self.environment.exec(command="tmux -V", user="root")
        if verify_result.return_code != 0:
            session_id = getattr(self.environment, "session_id", "<unknown>")
            raise RuntimeError(f"tmux is not installed for session {session_id}")
