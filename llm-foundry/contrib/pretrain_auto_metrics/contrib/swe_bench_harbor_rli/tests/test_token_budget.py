import pytest

from swe_bench_harbor_rli.src import token_budget
from swe_bench_harbor_rli.src.errors import OutputTokenBudgetExceededError


def test_account_output_tokens_raises_when_accumulated_budget_reached(
    monkeypatch,
) -> None:
    monkeypatch.setenv(token_budget.MAX_TOTAL_OUTPUT_TOKENS_ENV, "100")
    monkeypatch.setattr(token_budget, "_consumed_output_tokens", 0)

    token_budget.account_output_tokens(40)

    with pytest.raises(OutputTokenBudgetExceededError, match="100/100"):
        token_budget.account_output_tokens(60)
