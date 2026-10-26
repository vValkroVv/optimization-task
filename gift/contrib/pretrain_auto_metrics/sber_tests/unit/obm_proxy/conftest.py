import logging

import pytest


@pytest.fixture(autouse=True)
def _propagate_obm_proxy_logger():
    # obm_proxy.proxy.main runs configure_logging() on import, which sets
    # propagate=False on the shared "obm_proxy" logger and breaks caplog
    # for every test collected after it.
    logging.getLogger("obm_proxy").propagate = True
    yield
