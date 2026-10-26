from __future__ import annotations

from terminal_bench.src.rli_policy import (
    CreateSessionFailureKind,
    RetryDecision,
    StepFailureKind,
    TerminalBenchRLIPolicy,
    TerminalBenchRLIPolicyConfig,
    TransferFailureKind,
)


def test_policy_defaults_use_poll_only_warmup() -> None:
    config = TerminalBenchRLIPolicyConfig()

    assert config.warmup_mode == "poll_only"
    assert config.enable_probe_warmup is False
    assert config.probe_warmup_command == "pwd"
    assert config.max_bootstrap_step_retries == 2
    assert config.max_transfer_retries == 2
    assert config.max_verifier_step_retries == 2
    assert config.max_agent_step_retries == 0
    assert config.probe_retry_backoff_sec == 2.0
    assert config.max_create_session_retries == 2
    assert config.create_session_retry_backoff_sec == 2.0
    assert config.create_session_retry_backoff_max_sec == 30.0
    assert config.retryable_create_session_statuses == (408, 429, 500, 502, 503, 504)
    assert config.create_session_jitter_max_sec == 0.0


def test_classify_retryable_create_session_status() -> None:
    policy = TerminalBenchRLIPolicy()

    kind = policy.classify_create_session_failure(
        status_code=504,
        transport_error=False,
    )

    assert kind is CreateSessionFailureKind.RETRYABLE_STATUS


def test_classify_retryable_create_session_transport_error() -> None:
    policy = TerminalBenchRLIPolicy()

    kind = policy.classify_create_session_failure(
        status_code=None,
        transport_error=True,
    )

    assert kind is CreateSessionFailureKind.RETRYABLE_TRANSPORT


def test_classify_non_retryable_create_session_status() -> None:
    policy = TerminalBenchRLIPolicy()

    kind = policy.classify_create_session_failure(
        status_code=400,
        transport_error=False,
    )

    assert kind is CreateSessionFailureKind.NON_RETRYABLE


def test_retry_create_session_transient_failure_within_budget() -> None:
    policy = TerminalBenchRLIPolicy(
        TerminalBenchRLIPolicyConfig(max_create_session_retries=2)
    )

    decision = policy.should_retry_create_session(
        failure_kind=CreateSessionFailureKind.RETRYABLE_STATUS,
        attempt=1,
    )

    assert decision == RetryDecision(
        retry=True, reason="retryable_create_session_status"
    )


def test_stop_retrying_create_session_after_policy_limit() -> None:
    policy = TerminalBenchRLIPolicy(
        TerminalBenchRLIPolicyConfig(max_create_session_retries=2)
    )

    decision = policy.should_retry_create_session(
        failure_kind=CreateSessionFailureKind.RETRYABLE_STATUS,
        attempt=3,
    )

    assert decision == RetryDecision(retry=False, reason="retry_budget_exhausted")


def test_create_session_retry_uses_exponential_backoff_with_cap() -> None:
    policy = TerminalBenchRLIPolicy(
        TerminalBenchRLIPolicyConfig(
            create_session_retry_backoff_sec=2.0,
            create_session_retry_backoff_max_sec=5.0,
        )
    )

    assert policy.create_session_retry_backoff_seconds(attempt=1) == 2.0
    assert policy.create_session_retry_backoff_seconds(attempt=2) == 4.0
    assert policy.create_session_retry_backoff_seconds(attempt=3) == 5.0


def test_policy_does_not_require_probe_step_warmup_by_default() -> None:
    policy = TerminalBenchRLIPolicy()

    assert policy.should_run_probe_warmup() is False


def test_policy_requires_probe_step_warmup_when_enabled() -> None:
    policy = TerminalBenchRLIPolicy(
        TerminalBenchRLIPolicyConfig(enable_probe_warmup=True)
    )

    assert policy.should_run_probe_warmup() is True


def test_policy_exposes_probe_warmup_command() -> None:
    policy = TerminalBenchRLIPolicy(
        TerminalBenchRLIPolicyConfig(
            enable_probe_warmup=True,
            probe_warmup_command="mkdir -p /tmp/terminal-bench/.warmup",
        )
    )

    assert policy.probe_warmup_command() == "mkdir -p /tmp/terminal-bench/.warmup"


def test_classify_early_exec_proxy_failure_as_early_session_instability() -> None:
    policy = TerminalBenchRLIPolicy()

    kind = policy.classify_step_failure(
        phase="bootstrap",
        exit_code=124,
        error="exec-proxy connection lost; partial result preserved",
        incomplete=True,
        step_index=1,
    )

    assert kind is StepFailureKind.EARLY_SESSION_INSTABILITY


def test_retry_bootstrap_step_for_early_session_instability() -> None:
    policy = TerminalBenchRLIPolicy()

    decision = policy.should_retry_step(
        phase="bootstrap",
        failure_kind=StepFailureKind.EARLY_SESSION_INSTABILITY,
        attempt=1,
    )

    assert decision == RetryDecision(retry=True, reason="early_session_instability")


def test_retry_probe_step_for_early_session_instability() -> None:
    policy = TerminalBenchRLIPolicy(
        TerminalBenchRLIPolicyConfig(enable_probe_warmup=True)
    )

    decision = policy.should_retry_step(
        phase="probe",
        failure_kind=StepFailureKind.EARLY_SESSION_INSTABILITY,
        attempt=1,
    )

    assert decision == RetryDecision(retry=True, reason="early_session_instability")


def test_probe_retry_uses_exponential_backoff_with_cap() -> None:
    policy = TerminalBenchRLIPolicy(
        TerminalBenchRLIPolicyConfig(probe_retry_backoff_sec=2.0)
    )

    assert policy.retry_backoff_seconds(phase="probe", attempt=1) == 2.0
    assert policy.retry_backoff_seconds(phase="probe", attempt=2) == 4.0
    assert policy.retry_backoff_seconds(phase="probe", attempt=3) == 8.0
    assert policy.retry_backoff_seconds(phase="probe", attempt=4) == 8.0


def test_stop_retrying_bootstrap_after_policy_limit() -> None:
    policy = TerminalBenchRLIPolicy(
        TerminalBenchRLIPolicyConfig(max_bootstrap_step_retries=2)
    )

    decision = policy.should_retry_step(
        phase="bootstrap",
        failure_kind=StepFailureKind.EARLY_SESSION_INSTABILITY,
        attempt=3,
    )

    assert decision == RetryDecision(
        retry=False,
        reason="retry_budget_exhausted",
    )


def test_do_not_retry_normal_agent_command_failure() -> None:
    policy = TerminalBenchRLIPolicy()

    kind = policy.classify_step_failure(
        phase="agent",
        exit_code=1,
        error="root@rli:/app#",
        incomplete=False,
        step_index=25,
    )
    decision = policy.should_retry_step(
        phase="agent",
        failure_kind=kind,
        attempt=1,
    )

    assert kind is StepFailureKind.NORMAL_COMMAND_FAILURE
    assert decision == RetryDecision(retry=False, reason="normal_command_failure")


def test_retry_verifier_step_once_for_broken_exec_proxy() -> None:
    policy = TerminalBenchRLIPolicy(
        TerminalBenchRLIPolicyConfig(max_verifier_step_retries=1)
    )

    kind = policy.classify_step_failure(
        phase="verifier",
        exit_code=-1,
        error="exec-proxy connection lost; partial result preserved",
        incomplete=True,
        step_index=0,
    )
    first = policy.should_retry_step(
        phase="verifier",
        failure_kind=kind,
        attempt=1,
    )
    second = policy.should_retry_step(
        phase="verifier",
        failure_kind=kind,
        attempt=2,
    )

    assert kind is StepFailureKind.BROKEN_STEP
    assert first == RetryDecision(retry=True, reason="broken_verifier_step")
    assert second == RetryDecision(retry=False, reason="retry_budget_exhausted")


def test_classify_upload_failure_as_opaque_transfer_failure() -> None:
    policy = TerminalBenchRLIPolicy()

    kind = policy.classify_upload_failure(
        return_code=1,
        stdout="",
        stderr="",
    )

    assert kind is TransferFailureKind.OPAQUE_TRANSFER_FAILURE


def test_retry_opaque_upload_failure_within_budget() -> None:
    policy = TerminalBenchRLIPolicy()

    decision = policy.should_retry_transfer(
        phase="upload",
        failure_kind=TransferFailureKind.OPAQUE_TRANSFER_FAILURE,
        attempt=1,
    )

    assert decision == RetryDecision(retry=True, reason="opaque_transfer_failure")


def test_classify_missing_remote_verifier_dir_separately() -> None:
    policy = TerminalBenchRLIPolicy()

    kind = policy.classify_download_failure(
        source_path="/logs/verifier",
        return_code=2,
        stdout=(
            "tar: /tmp/terminal-bench/logs/verifier: Cannot open: "
            "No such file or directory"
        ),
        stderr="root@rli:/app#",
    )

    assert kind is TransferFailureKind.REMOTE_OUTPUT_MISSING


def test_do_not_retry_missing_remote_verifier_dir() -> None:
    policy = TerminalBenchRLIPolicy()

    decision = policy.should_retry_transfer(
        phase="download",
        failure_kind=TransferFailureKind.REMOTE_OUTPUT_MISSING,
        attempt=1,
    )

    assert decision == RetryDecision(retry=False, reason="remote_output_missing")
