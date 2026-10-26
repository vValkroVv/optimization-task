import pytest

from job_submitter.infrastructure.mlspace import MlSpaceApiError
from job_submitter.infrastructure.mlspace.auth import ServiceAuth
from job_submitter.infrastructure.mlspace.config import MlSpaceConfig
from sber_tests.test_support.mlspace import FakeHttp, FakeResponse, set_mlspace_env

pytestmark = pytest.mark.unit


@pytest.fixture
def config(monkeypatch):
    set_mlspace_env(monkeypatch)
    return MlSpaceConfig.from_env()


def test_headers_carry_token_workspace_and_region(config):
    auth = ServiceAuth(config, FakeHttp())

    assert auth.headers("SR008") == {
        "Authorization": "Bearer tok1",
        "x-api-key": "key",
        "x-workspace-id": "ws",
        "x-region": "SR008",
    }


def test_token_is_reused_until_reset(config):
    http = FakeHttp()
    auth = ServiceAuth(config, http)

    auth.headers("SR008")
    auth.headers("SR008")
    auth.reset()
    headers = auth.headers("SR008")

    assert http.logins == 2
    assert headers["Authorization"] == "Bearer tok2"


def test_token_is_refreshed_before_expiry(config):
    http = FakeHttp()
    http.login_response = FakeResponse(
        200, {"token": {"access_token": "short", "expires_in": 30}}
    )
    auth = ServiceAuth(config, http)

    auth.headers("SR008")
    auth.headers("SR008")

    assert http.logins == 2


def test_login_accepts_unwrapped_token_payload(config):
    http = FakeHttp()
    http.login_response = FakeResponse(200, {"access_token": "flat"})

    assert ServiceAuth(config, http).headers("SR008")["Authorization"] == "Bearer flat"


def test_login_failure_is_reported_with_status_and_body(config):
    http = FakeHttp()
    http.login_response = FakeResponse(500, text="boom")

    with pytest.raises(MlSpaceApiError, match="service_auth failed: HTTP 500 boom"):
        ServiceAuth(config, http).headers("SR008")
