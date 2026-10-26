from __future__ import annotations

import os

from .errors import OutputTokenBudgetExceededError


MAX_TOTAL_OUTPUT_TOKENS_ENV = "PAM_MAX_TOTAL_OUTPUT_TOKENS"

_consumed_output_tokens = 0


def account_output_tokens(output_tokens: int | None) -> None:
    if output_tokens is None:
        return

    raw_limit = os.getenv(MAX_TOTAL_OUTPUT_TOKENS_ENV)
    if raw_limit is None:
        return

    limit = int(raw_limit)
    global _consumed_output_tokens
    _consumed_output_tokens += int(output_tokens)

    if _consumed_output_tokens >= limit:
        raise OutputTokenBudgetExceededError(
            f"Output-token budget reached: {_consumed_output_tokens}/{limit}"
        )
