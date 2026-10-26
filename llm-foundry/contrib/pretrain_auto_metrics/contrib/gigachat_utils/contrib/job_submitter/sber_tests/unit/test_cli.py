import runpy

import click
import pytest
import typer

from job_submitter.cli import main


pytestmark = pytest.mark.unit


def test_kill_command_calls_kill_job(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "job_submitter.cli.kill_job", lambda job_id: f'Job "{job_id}" deleted'
    )

    code = main(["kill", "job-123"])
    captured = capsys.readouterr()

    assert code == 0
    assert captured.out.strip() == 'Job "job-123" deleted'


def test_kill_command_returns_non_zero_on_error(monkeypatch, capsys) -> None:
    def _raise(job_id: str) -> str:
        raise RuntimeError("boom")

    monkeypatch.setattr("job_submitter.cli.kill_job", _raise)

    code = main(["kill", "job-123"])
    captured = capsys.readouterr()

    assert code == 1
    assert "boom" in captured.err


def test_main_returns_2_on_usage_error(capsys) -> None:
    code = main(["kill"])
    captured = capsys.readouterr()

    assert code == 2
    assert "Missing" in captured.err


def test_main_returns_2_on_vendored_click_usage_error(monkeypatch, capsys) -> None:
    class _VendoredUsageError(Exception):
        exit_code = 2

        def show(self) -> None:
            import sys

            print("Missing parameter: job_id", file=sys.stderr)

    def _raise(*_args, **_kwargs) -> None:
        raise _VendoredUsageError()

    monkeypatch.setattr("job_submitter.cli.app", _raise)

    code = main(["kill"])
    captured = capsys.readouterr()

    assert code == 2
    assert "Missing parameter: job_id" in captured.err


def test_main_reraises_unhandled_exceptions(monkeypatch) -> None:
    def _raise(*_args, **_kwargs) -> None:
        raise RuntimeError("unexpected")

    monkeypatch.setattr("job_submitter.cli.app", _raise)

    with pytest.raises(RuntimeError, match="unexpected"):
        main(["kill"])


def test_main_returns_code_from_typer_exit(monkeypatch) -> None:
    def _raise(*args, **kwargs):
        raise typer.Exit(code=7)

    monkeypatch.setattr("job_submitter.cli.app", _raise)

    assert main(["kill", "job-123"]) == 7


def test_main_returns_code_from_click_exception(monkeypatch, capsys) -> None:
    def _raise(*args, **kwargs):
        raise click.UsageError("Invalid usage")

    monkeypatch.setattr("job_submitter.cli.app", _raise)

    code = main(["kill", "job-123"])
    captured = capsys.readouterr()

    assert code == 2
    assert "Invalid usage" in captured.err


def test_main_handles_click_abort(monkeypatch, capsys) -> None:
    def _raise(*args, **kwargs):
        raise click.Abort()

    monkeypatch.setattr("job_submitter.cli.app", _raise)
    code = main(["kill", "job-123"])
    captured = capsys.readouterr()

    assert code == 1
    assert "Aborted!" in captured.err


def test_main_reraises_unknown_exception(monkeypatch) -> None:
    def _raise(*args, **kwargs):
        raise RuntimeError("unexpected")

    monkeypatch.setattr("job_submitter.cli.app", _raise)

    with pytest.raises(RuntimeError, match="unexpected"):
        main(["kill"])


def test_main_returns_zero_when_app_returns(monkeypatch) -> None:
    monkeypatch.setattr("job_submitter.cli.app", lambda *args, **kwargs: None)

    assert main(["kill", "job-123"]) == 0


def test_module_main_raises_system_exit_with_main_code(monkeypatch) -> None:
    monkeypatch.setattr("job_submitter.cli.main", lambda: 7)

    with pytest.raises(SystemExit) as exc:
        runpy.run_module("job_submitter", run_name="__main__")

    assert exc.value.code == 7


def test_module_main_does_not_call_main_when_not_executed_as_main(
    monkeypatch,
) -> None:
    def _should_not_be_called() -> int:
        raise AssertionError("cli.main must not be called")

    monkeypatch.setattr("job_submitter.cli.main", _should_not_be_called)
    module_globals = runpy.run_module("job_submitter", run_name="not_main")

    assert module_globals["__name__"] == "not_main"
