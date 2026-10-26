from __future__ import annotations

from dataclasses import dataclass


def client_status_code_for_sglang(status_code: int) -> int:
    if status_code == 503:
        return 429
    return status_code


class SGLangRetryableRequestError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class SGLangRetryableStatusError(Exception):
    status_code: int
    response_text: str


@dataclass(frozen=True, slots=True)
class SGLangNonRetryableStatusError(Exception):
    status_code: int
    response_text: str
