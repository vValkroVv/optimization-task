"""Public package surface for the minimal GigaRLI sessions client."""

from .client import RLIClient, RLIClientError
from .client_session import RLIClientSession
from .schemas.common import (
    ErrorDetails,
    ErrorResponse,
    RequestLimits,
    ResponseMeta,
    SuccessResponse,
)
from .schemas.env import (
    CloseSessionRequest,
    CreateSessionRequest,
    CreateSessionResult,
    GetSessionResult,
    SessionAttribution,
    SessionEnv,
    StepAction,
    StepActionArgs,
    StepEvent,
    StepObservation,
    StepRequest,
    StepResult,
    StepStreamMeta,
)

__all__ = [
    "CloseSessionRequest",
    "CreateSessionRequest",
    "CreateSessionResult",
    "ErrorDetails",
    "ErrorResponse",
    "GetSessionResult",
    "RLIClientSession",
    "RequestLimits",
    "RLIClient",
    "RLIClientError",
    "ResponseMeta",
    "SessionAttribution",
    "SessionEnv",
    "StepAction",
    "StepActionArgs",
    "StepEvent",
    "StepObservation",
    "StepRequest",
    "StepResult",
    "StepStreamMeta",
    "SuccessResponse",
]
