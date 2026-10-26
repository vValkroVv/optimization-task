from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

HTTP_URL = "http://127.0.0.1:8123/"
TIMEOUT_SECONDS = 1900
SERVER_COMMAND = re.compile(r"^clickhouse-server --config-file")
PASSWORD_PREFIX = b"CLICKHOUSE_PASSWORD="
PROC_ROOT = Path("/proc")


class ClickHouseError(Exception):
    """A statement that did not run: `output` and `error_output` are what the transport wrote to stdout and stderr."""

    def __init__(self, status: int, output: str, error_output: str) -> None:
        super().__init__(error_output.strip() or f"exit status {status}")
        self.status = status
        self.output = output
        self.error_output = error_output


class ClickHouse:
    """Runs statements as the default user over HTTP from inside the server process's network namespace.

    This replaces docker exec, which hangs while the data disk is saturated. The password goes from the server
    process's environment through a pipe into curl's config, so it is never printed, put on a command line or written
    to a file, not even when the service stops in the middle of a statement.
    """

    def __init__(self, timeout_seconds: int = TIMEOUT_SECONDS, proc_root: Path = PROC_ROOT) -> None:
        self._timeout_seconds = timeout_seconds
        self._proc_root = proc_root

    def execute(self, statement: str) -> str:
        """The response body; raises ClickHouseError on any HTTP error."""
        try:
            server_pid = self._server_pid()
            if server_pid is None:
                raise ClickHouseError(1, "", "no clickhouse-server process")
            config_reader = self._curl_config(self._password(server_pid))
            try:
                completed = subprocess.run(
                    self._command(server_pid, f"/dev/fd/{config_reader}"), input=statement.encode(),
                    capture_output=True, check=False, pass_fds=(config_reader,),
                )
            finally:
                os.close(config_reader)
        except OSError as error:
            raise ClickHouseError(1, "", str(error)) from error
        output = completed.stdout.decode(errors="replace")
        if completed.returncode != 0:
            raise ClickHouseError(completed.returncode, output, completed.stderr.decode(errors="replace"))
        return output

    def value(self, statement: str) -> str:
        """The response body without its trailing newlines, as a shell command substitution gives it."""
        return self.execute(statement).rstrip("\n")

    def rows(self, statement: str) -> list[list[str]]:
        return [line.split("\t") for line in self.execute(statement).splitlines()]

    def _server_pid(self) -> int | None:
        for pid in sorted(int(entry.name) for entry in self._proc_root.iterdir() if entry.name.isdigit()):
            try:
                command_line = (self._proc_root / str(pid) / "cmdline").read_bytes()
            except OSError:
                continue
            if SERVER_COMMAND.search(command_line.rstrip(b"\0").replace(b"\0", b" ").decode(errors="replace")):
                return pid
        return None

    def _password(self, server_pid: int) -> bytes:
        environment = (self._proc_root / str(server_pid) / "environ").read_bytes()
        lines = environment.replace(b"\0", b"\n").split(b"\n")
        passwords = [line.removeprefix(PASSWORD_PREFIX) for line in lines if line.startswith(PASSWORD_PREFIX)]
        return b"\n".join(passwords).rstrip(b"\n")

    def _command(self, server_pid: int, config_path: str) -> list[str]:
        return [
            "nsenter", "-t", str(server_pid), "-n",
            "curl", "-sS", "--fail-with-body", "-m", str(self._timeout_seconds), "-K", config_path,
            "--data-binary", "@-", HTTP_URL,
        ]

    @staticmethod
    def _curl_config(password: bytes) -> int:
        """The read end of a pipe holding curl's config with the password, written in full before curl starts: the
        config is far smaller than a pipe's buffer."""
        reader, writer = os.pipe()
        with os.fdopen(writer, "wb") as config:
            config.write(b'user = "default:' + password + b'"\n')
        return reader
