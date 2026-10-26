PYTEST ?= pytest
CORE_COV_TARGETS = \
	--cov=src.utils.io \
	--cov=src.utils.judgment_summary \
	--cov=src.models \
	--cov=dredd

.PHONY: test test-unit test-integration test-cov

test:
	$(PYTEST) -m "(unit or integration) and not e2e"

test-unit:
	$(PYTEST) -m "unit"

test-integration:
	$(PYTEST) -m "integration"

test-cov:
	$(PYTEST) -m "(unit or integration) and not e2e" $(CORE_COV_TARGETS) --cov-report=term-missing --cov-fail-under=60
