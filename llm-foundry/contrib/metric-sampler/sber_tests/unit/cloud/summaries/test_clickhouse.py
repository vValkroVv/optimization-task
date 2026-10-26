from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from metric_sampler.cloud.summaries.clickhouse import ClickHouse, ClickHouseError

pytestmark = pytest.mark.unit

SERVER = b"clickhouse-server\0--config-file=/etc/clickhouse-server/config.xml\0"
STATEMENT = "SELECT 'it''s'\n    FROM t WHERE a = 1"
CONFIG_READ_BYTES = 4096


class FakeCurl:
    """Stands in for `subprocess.run` of nsenter and curl: records the call and what curl read from its config."""

    def __init__(
        self, returncode: int = 0, stdout: bytes = b"1\n", stderr: bytes = b"", error: OSError | None = None
    ) -> None:
        self.calls: list[list[str]] = []
        self.inputs: list[bytes] = []
        self.configs: list[bytes] = []
        self.config_descriptors: list[int] = []
        self.passed_descriptors: list[tuple[int, ...]] = []
        self._answer = subprocess.CompletedProcess([], returncode, stdout, stderr)
        self._error = error

    def __call__(self, command, input, capture_output, check, pass_fds):
        self.calls.append(command)
        self.inputs.append(input)
        descriptor = int(command[command.index("-K") + 1].removeprefix("/dev/fd/"))
        self.config_descriptors.append(descriptor)
        self.passed_descriptors.append(pass_fds)
        self.configs.append(os.read(descriptor, CONFIG_READ_BYTES))
        assert capture_output
        assert not check
        if self._error is not None:
            raise self._error
        return self._answer


def make_proc(root: Path, processes: dict[str, dict[str, bytes]]) -> Path:
    for pid, files in processes.items():
        (root / pid).mkdir(parents=True)
        for name, content in files.items():
            (root / pid / name).write_bytes(content)
    return root


@pytest.fixture
def curl(monkeypatch) -> FakeCurl:
    fake = FakeCurl()
    monkeypatch.setattr("metric_sampler.cloud.summaries.clickhouse.subprocess.run", fake)
    return fake


def server_proc(tmp_path: Path, environ: bytes = b"A=1\0CLICKHOUSE_PASSWORD=secret\0B=2\0") -> Path:
    return make_proc(tmp_path / "proc", {"100": {"cmdline": SERVER, "environ": environ}})


def test_a_statement_goes_to_the_server_over_http_from_its_network_namespace_and_the_body_comes_back(
    tmp_path, curl,
) -> None:
    client = ClickHouse(proc_root=server_proc(tmp_path))

    assert client.execute(STATEMENT) == "1\n"

    assert curl.calls == [[
        "nsenter", "-t", "100", "-n",
        "curl", "-sS", "--fail-with-body", "-m", "1900", "-K", f"/dev/fd/{curl.config_descriptors[0]}",
        "--data-binary", "@-", "http://127.0.0.1:8123/",
    ]]
    assert curl.inputs == [STATEMENT.encode()]


def assert_closed(descriptor: int) -> None:
    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_the_password_reaches_curl_only_through_a_pipe_it_inherits_which_is_closed_afterwards(tmp_path, curl) -> None:
    ClickHouse(proc_root=server_proc(tmp_path)).execute(STATEMENT)

    assert curl.configs == [b'user = "default:secret"\n']
    assert curl.passed_descriptors == [(curl.config_descriptors[0],)]
    assert all("secret" not in argument for argument in curl.calls[0])
    assert_closed(curl.config_descriptors[0])


def test_the_timeout_is_the_curl_limit_of_the_whole_request(tmp_path, curl) -> None:
    ClickHouse(timeout_seconds=3000, proc_root=server_proc(tmp_path)).execute(STATEMENT)

    assert curl.calls[0][curl.calls[0].index("-m") + 1] == "3000"


@pytest.mark.parametrize(
    ("environ", "password"),
    [
        pytest.param(b"CLICKHOUSE_PASSWORD=p\"a$b c\0", b'p"a$b c', id="every character is kept as it is"),
        pytest.param(b"A=1\0B=2\0", b"", id="no password variable"),
        pytest.param(b"XCLICKHOUSE_PASSWORD=no\0CLICKHOUSE_PASSWORD=yes\0", b"yes", id="the name has to match in full"),
        pytest.param(b"CLICKHOUSE_PASSWORD=ends\n\0", b"ends", id="trailing newlines go, as a command substitution"),
        pytest.param(b"CLICKHOUSE_PASSWORD=one\0CLICKHOUSE_PASSWORD=two\0", b"one\ntwo", id="every definition counts"),
    ],
)
def test_the_password_is_the_value_of_the_servers_clickhouse_password_variable(
    tmp_path, curl, environ, password,
) -> None:
    ClickHouse(proc_root=server_proc(tmp_path, environ)).execute(STATEMENT)

    assert curl.configs == [b'user = "default:' + password + b'"\n']


def test_the_server_is_the_lowest_process_id_whose_command_line_starts_with_the_server_and_its_config(
    tmp_path, curl,
) -> None:
    proc = make_proc(tmp_path / "proc", {
        "7": {"cmdline": b"python3\0worker.py\0"},
        "10": {"cmdline": b"bash\0-c\0clickhouse-server --config-file=x\0"},
        "30": {"cmdline": SERVER, "environ": b"CLICKHOUSE_PASSWORD=late\0"},
        "20": {"cmdline": SERVER, "environ": b"CLICKHOUSE_PASSWORD=early\0"},
        "5": {},
    })
    (proc / "self").mkdir()
    (proc / "cpuinfo").write_text("model name", encoding="utf-8")

    ClickHouse(proc_root=proc).execute(STATEMENT)

    assert curl.calls[0][:4] == ["nsenter", "-t", "20", "-n"]
    assert curl.configs == [b'user = "default:early"\n']


def test_without_a_server_process_no_statement_is_sent(tmp_path, curl) -> None:
    client = ClickHouse(proc_root=make_proc(tmp_path / "proc", {"7": {"cmdline": b"python3\0"}}))

    with pytest.raises(ClickHouseError, match="no clickhouse-server process") as raised:
        client.execute(STATEMENT)

    error = raised.value
    assert (error.status, error.output, error.error_output) == (1, "", "no clickhouse-server process")
    assert curl.calls == []


def test_a_failed_request_carries_the_status_the_body_and_what_curl_said_and_closes_the_config_pipe(
    tmp_path, monkeypatch,
) -> None:
    curl = FakeCurl(
        returncode=22, stdout=b"Code: 241. Memory limit\n", stderr=b"curl: (22) The requested URL returned error: 500\n"
    )
    monkeypatch.setattr("metric_sampler.cloud.summaries.clickhouse.subprocess.run", curl)

    with pytest.raises(ClickHouseError) as raised:
        ClickHouse(proc_root=server_proc(tmp_path)).execute(STATEMENT)

    assert raised.value.status == 22
    assert raised.value.output == "Code: 241. Memory limit\n"
    assert raised.value.error_output == "curl: (22) The requested URL returned error: 500\n"
    assert str(raised.value) == "curl: (22) The requested URL returned error: 500"
    assert_closed(curl.config_descriptors[0])


def test_a_failure_without_any_error_text_is_described_by_its_exit_status(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "metric_sampler.cloud.summaries.clickhouse.subprocess.run", FakeCurl(returncode=137, stdout=b"")
    )

    with pytest.raises(ClickHouseError, match="exit status 137"):
        ClickHouse(proc_root=server_proc(tmp_path)).execute(STATEMENT)


def test_a_server_process_that_cannot_be_read_is_a_failed_statement(tmp_path, curl) -> None:
    proc = make_proc(tmp_path / "proc", {"100": {"cmdline": SERVER}})

    with pytest.raises(ClickHouseError, match="environ") as raised:
        ClickHouse(proc_root=proc).execute(STATEMENT)

    assert raised.value.status == 1
    assert curl.calls == []


def test_a_missing_nsenter_is_a_failed_statement_and_closes_the_config_pipe(tmp_path, monkeypatch) -> None:
    curl = FakeCurl(error=FileNotFoundError("nsenter"))
    monkeypatch.setattr("metric_sampler.cloud.summaries.clickhouse.subprocess.run", curl)

    with pytest.raises(ClickHouseError, match="nsenter"):
        ClickHouse(proc_root=server_proc(tmp_path)).execute(STATEMENT)

    assert_closed(curl.config_descriptors[0])


@pytest.mark.parametrize(
    ("body", "value", "rows"),
    [
        ("", "", []),
        ("1\n", "1", [["1"]]),
        ("a\tb\nc\td\n\n", "a\tb\nc\td", [["a", "b"], ["c", "d"], [""]]),
        ("e\ta\t120\n", "e\ta\t120", [["e", "a", "120"]]),
    ],
)
def test_a_value_has_no_trailing_newlines_and_rows_are_the_tab_separated_lines_of_the_body(
    tmp_path, monkeypatch, body, value, rows,
) -> None:
    monkeypatch.setattr(
        "metric_sampler.cloud.summaries.clickhouse.subprocess.run", FakeCurl(stdout=body.encode())
    )
    client = ClickHouse(proc_root=server_proc(tmp_path))

    assert client.value(STATEMENT) == value
    assert client.rows(STATEMENT) == rows
