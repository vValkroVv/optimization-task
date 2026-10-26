import pytest

from job_submitter.infrastructure.mlspace import MlSpaceConfigError
from job_submitter.infrastructure.mlspace.config import DEFAULT_API_URL, MlSpaceConfig
from sber_tests.test_support.mlspace import set_mlspace_env

pytestmark = pytest.mark.unit


def test_config_reads_stripped_credentials_and_default_url(monkeypatch):
    set_mlspace_env(monkeypatch)
    monkeypatch.setenv("MLS_CLIENT_ID", " id ")
    monkeypatch.delenv("MLS_API_URL")

    config = MlSpaceConfig.from_env()

    assert config == MlSpaceConfig(
        base_url=DEFAULT_API_URL,
        client_id="id",
        client_secret="secret",
        api_key="key",
        workspace_id="ws",
    )


def test_config_strips_trailing_slash_from_url(monkeypatch):
    set_mlspace_env(monkeypatch)
    monkeypatch.setenv("MLS_API_URL", "https://api.test/public/v2/")

    assert MlSpaceConfig.from_env().base_url == "https://api.test/public/v2"


def test_config_reports_every_missing_credential(monkeypatch):
    set_mlspace_env(monkeypatch)
    monkeypatch.delenv("MLS_CLIENT_ID")
    monkeypatch.setenv("MLS_API_KEY", "  ")

    with pytest.raises(MlSpaceConfigError, match="MLS_CLIENT_ID, MLS_API_KEY"):
        MlSpaceConfig.from_env()
