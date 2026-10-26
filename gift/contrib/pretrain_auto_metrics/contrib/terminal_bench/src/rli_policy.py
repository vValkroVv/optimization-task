from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class StepFailureKind(str, Enum):
    """Classification for step-level failures in terminal-bench RLI runs."""

    EARLY_SESSION_INSTABILITY = "early_session_instability"
    BROKEN_STEP = "broken_step"
    NORMAL_COMMAND_FAILURE = "normal_command_failure"


class CreateSessionFailureKind(str, Enum):
    """Classification for create-session failures."""

    RETRYABLE_STATUS = "retryable_status"
    RETRYABLE_TRANSPORT = "retryable_transport"
    RETRYABLE_RLI_ERROR = "retryable_rli_error"
    NON_RETRYABLE = "non_retryable"


class TransferFailureKind(str, Enum):
    """Classification for upload/download support failures."""

    OPAQUE_TRANSFER_FAILURE = "opaque_transfer_failure"
    REMOTE_OUTPUT_MISSING = "remote_output_missing"


@dataclass(frozen=True)
class RetryDecision:
    """Decision returned by the policy for a failed operation."""

    retry: bool
    reason: str


@dataclass(frozen=True)
class TerminalBenchRLIPolicyConfig:
    """Configuration for terminal-bench-specific RLI runtime policy."""

    warmup_mode: str = "poll_only"
    enable_probe_warmup: bool = False
    probe_warmup_command: str = "pwd"
    max_bootstrap_step_retries: int = 2
    max_transfer_retries: int = 2
    max_verifier_step_retries: int = 2
    max_agent_step_retries: int = 0
    probe_retry_backoff_sec: float = 2.0
    max_create_session_retries: int = 2
    create_session_retry_backoff_sec: float = 2.0
    create_session_retry_backoff_max_sec: float = 30.0
    retryable_create_session_statuses: tuple[int, ...] = (
        408,
        429,
        500,
        502,
        503,
        504,
    )
    create_session_jitter_max_sec: float = 0.0


class TerminalBenchRLIPolicy:
    """Decision logic for terminal-bench RLI session handling."""

    def __init__(self, config: TerminalBenchRLIPolicyConfig | None = None) -> None:
        self.config = config or TerminalBenchRLIPolicyConfig()

    def should_run_probe_warmup(self) -> bool:
        """Return whether policy requires an explicit probe-step warmup."""

        return self.config.enable_probe_warmup

    def probe_warmup_command(self) -> str:
        """Return the command used for startup probe-step warmup."""

        return self.config.probe_warmup_command

    def classify_create_session_failure(
        self,
        *,
        status_code: int | None,
        transport_error: bool,
        retryable: bool | None = None,
        retry_strategy: str | None = None,
        error_code: str | None = None,
        error_category: str | None = None,
    ) -> CreateSessionFailureKind:
        """Classify a failed ``POST /v1/sessions`` operation."""

        _ = error_category
        if transport_error:
            return CreateSessionFailureKind.RETRYABLE_TRANSPORT

        strategy = (retry_strategy or "").strip().lower()
        if strategy in {
            "wait_and_retry_step",
            "create_new_session",
            "increase_client_timeout",
        }:
            return CreateSessionFailureKind.RETRYABLE_RLI_ERROR
        if strategy in {
            "fix_request",
            "not_recommended",
            "split_command_or_increase_timeout",
        }:
            return CreateSessionFailureKind.NON_RETRYABLE

        if retryable is True:
            return CreateSessionFailureKind.RETRYABLE_RLI_ERROR

        code = (error_code or "").strip().upper()
        if code in {
            "ARI_CONTROLLER_UNAVAILABLE",
            "BAD_REQUEST_FLOOD",
            "DATABASE_ERROR",
            "INTERNAL",
            "NETWORK_PARTITION",
            "POD_NEVER_REACHED",
            "QUOTA_EXCEEDED",
            "RATE_LIMITED",
            "RESOURCE_EXHAUSTED",
            "S3_STORAGE_ERROR",
            "TIMEOUT",
            "UNAVAILABLE",
        }:
            return CreateSessionFailureKind.RETRYABLE_RLI_ERROR

        if status_code in self.config.retryable_create_session_statuses:
            return CreateSessionFailureKind.RETRYABLE_STATUS
        return CreateSessionFailureKind.NON_RETRYABLE

    def should_retry_create_session(
        self,
        *,
        failure_kind: CreateSessionFailureKind,
        attempt: int,
    ) -> RetryDecision:
        """Return retry decision for a failed create-session operation."""

        if failure_kind is CreateSessionFailureKind.NON_RETRYABLE:
            return RetryDecision(
                retry=False,
                reason="non_retryable_create_session_failure",
            )

        if attempt > self.config.max_create_session_retries:
            return RetryDecision(retry=False, reason="retry_budget_exhausted")

        if failure_kind is CreateSessionFailureKind.RETRYABLE_STATUS:
            return RetryDecision(retry=True, reason="retryable_create_session_status")
        if failure_kind is CreateSessionFailureKind.RETRYABLE_TRANSPORT:
            return RetryDecision(
                retry=True,
                reason="retryable_create_session_transport",
            )
        if failure_kind is CreateSessionFailureKind.RETRYABLE_RLI_ERROR:
            return RetryDecision(
                retry=True, reason="retryable_create_session_rli_error"
            )

        return RetryDecision(
            retry=False,
            reason="non_retryable_create_session_failure",
        )

    def create_session_retry_backoff_seconds(self, *, attempt: int) -> float:
        """Return capped exponential backoff for create-session retries."""

        base = max(0.0, float(self.config.create_session_retry_backoff_sec))
        if base <= 0:
            return 0.0
        cap = max(base, float(self.config.create_session_retry_backoff_max_sec))
        return min(base * (2 ** (attempt - 1)), cap)

    def retry_backoff_seconds(self, *, phase: str, attempt: int) -> float:
        """Return backoff for retryable failures in a given phase."""

        if phase == "probe":
            base = max(0.0, float(self.config.probe_retry_backoff_sec))
            if base <= 0:
                return 0.0
            return min(base * (2 ** (attempt - 1)), 8.0)
        return 0.0

    def classify_step_failure(
        self,
        *,
        phase: str,
        exit_code: int,
        error: str | None,
        incomplete: bool,
        step_index: int,
    ) -> StepFailureKind:
        """Classify a failed RLI step for retry decisions."""

        if self._is_broken_exec_proxy_step(
            exit_code=exit_code,
            error=error,
            incomplete=incomplete,
        ):
            if phase in {"bootstrap", "probe"} and step_index <= 1:
                return StepFailureKind.EARLY_SESSION_INSTABILITY
            return StepFailureKind.BROKEN_STEP
        return StepFailureKind.NORMAL_COMMAND_FAILURE

    def should_retry_step(
        self,
        *,
        phase: str,
        failure_kind: StepFailureKind,
        attempt: int,
    ) -> RetryDecision:
        """Return retry decision for a failed step."""

        if failure_kind is StepFailureKind.NORMAL_COMMAND_FAILURE:
            return RetryDecision(retry=False, reason="normal_command_failure")

        max_retries = self._max_step_retries_for_phase(phase)
        if attempt > max_retries:
            return RetryDecision(retry=False, reason="retry_budget_exhausted")

        if failure_kind is StepFailureKind.EARLY_SESSION_INSTABILITY:
            return RetryDecision(retry=True, reason="early_session_instability")

        if failure_kind is StepFailureKind.BROKEN_STEP and phase == "verifier":
            return RetryDecision(retry=True, reason="broken_verifier_step")

        if failure_kind is StepFailureKind.BROKEN_STEP and phase == "bootstrap":
            return RetryDecision(retry=True, reason="broken_bootstrap_step")

        if failure_kind is StepFailureKind.BROKEN_STEP and phase == "agent":
            if self.config.max_agent_step_retries > 0:
                return RetryDecision(retry=True, reason="broken_agent_step")
            return RetryDecision(retry=False, reason="broken_agent_step")

        return RetryDecision(retry=False, reason="normal_command_failure")

    def classify_upload_failure(
        self,
        *,
        return_code: int,
        stdout: str,
        stderr: str,
    ) -> TransferFailureKind:
        """Classify upload-support failure."""

        _ = stdout, stderr
        if return_code in (-1, 1):
            return TransferFailureKind.OPAQUE_TRANSFER_FAILURE
        return TransferFailureKind.OPAQUE_TRANSFER_FAILURE

    def classify_download_failure(
        self,
        *,
        source_path: str,
        return_code: int,
        stdout: str,
        stderr: str,
    ) -> TransferFailureKind:
        """Classify download-support failure."""

        _ = stderr
        if (
            return_code == 2
            and source_path == "/logs/verifier"
            and "Cannot open: No such file or directory" in stdout
        ):
            return TransferFailureKind.REMOTE_OUTPUT_MISSING
        return TransferFailureKind.OPAQUE_TRANSFER_FAILURE

    def should_retry_transfer(
        self,
        *,
        phase: str,
        failure_kind: TransferFailureKind,
        attempt: int,
    ) -> RetryDecision:
        """Return retry decision for upload/download support failures."""

        _ = phase
        if failure_kind is TransferFailureKind.REMOTE_OUTPUT_MISSING:
            return RetryDecision(retry=False, reason="remote_output_missing")

        if attempt > self.config.max_transfer_retries:
            return RetryDecision(retry=False, reason="retry_budget_exhausted")

        return RetryDecision(retry=True, reason="opaque_transfer_failure")

    def _max_step_retries_for_phase(self, phase: str) -> int:
        if phase == "probe":
            return self.config.max_bootstrap_step_retries
        if phase == "bootstrap":
            return self.config.max_bootstrap_step_retries
        if phase == "verifier":
            return self.config.max_verifier_step_retries
        if phase == "agent":
            return self.config.max_agent_step_retries
        return 0

    @staticmethod
    def _is_broken_exec_proxy_step(
        *,
        exit_code: int,
        error: str | None,
        incomplete: bool,
    ) -> bool:
        if incomplete:
            return True
        if exit_code in (-1, 124):
            return True
        if error and "exec-proxy connection lost" in error:
            return True
        return False
