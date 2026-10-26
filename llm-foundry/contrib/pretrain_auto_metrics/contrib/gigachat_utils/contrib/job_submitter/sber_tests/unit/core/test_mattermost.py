import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from job_submitter.core.job_templates.models import User
from job_submitter.core.mattermost import MattermostClient
from job_submitter.asserts.base import BaseJobAssertor
from sber_tests.factories import DummyJob

_MOCK_GET = "job_submitter.core.mattermost.client.requests.Session.get"


def _mock_client(
    status: int = 200,
    json: dict | None = None,
    exc: Exception | None = None,
) -> MagicMock:
    """Build a mock MattermostClient."""
    client = MagicMock(spec=MattermostClient)
    if exc:
        client.get_user.side_effect = exc
    elif status == 200:
        client.get_user.return_value = json or {"id": "x", "username": "u"}
    else:
        resp = requests.Response()
        resp.status_code = status
        client.get_user.side_effect = requests.HTTPError(response=resp)
    return client


# ===================================================================
# User — format validation (no client needed)
# ===================================================================


@pytest.mark.unit
class TestUserFormat:
    def test_empty_login(self) -> None:
        user = User.verify("  @  ")
        assert not user.verified and "Empty" in user.error

    def test_invalid_chars(self) -> None:
        user = User.verify("юзер")
        assert not user.verified and "invalid" in user.error.lower()

    def test_no_client_skips(self) -> None:
        user = User.verify("alice", client=None)
        assert user.skipped and not user.verified

    def test_display_name_full(self) -> None:
        user = User(username="jdoe", first_name="John", last_name="Doe", verified=True)
        assert user.display_name == "John Doe"

    def test_display_name_fallback(self) -> None:
        user = User(username="jdoe", verified=True)
        assert user.display_name == "jdoe"

    def test_is_valid_format(self) -> None:
        assert User(username="alice.bob-1", verified=True).is_valid_format
        assert not User(username="", verified=False).is_valid_format

    def test_sanitize(self) -> None:
        assert User._sanitize("  @Alice.Smith  ") == "alice.smith"


# ===================================================================
# User.verify — with injected mock client
# ===================================================================


@pytest.mark.unit
class TestUserVerify:
    def test_verified(self) -> None:
        client = _mock_client(
            200,
            {"id": "abc", "username": "gdedov", "first_name": "G", "last_name": "D"},
        )
        user = User.verify("gdedov", client=client)
        assert user.verified and user.mm_id == "abc" and user.display_name == "G D"

    def test_not_found(self) -> None:
        client = _mock_client(404)
        user = User.verify("ghost", client=client)
        assert not user.verified and "not found" in user.error.lower()

    def test_server_error(self) -> None:
        client = _mock_client(502)
        user = User.verify("gdedov", client=client)
        assert not user.verified and "error" in user.error.lower()

    def test_network_error(self) -> None:
        client = _mock_client(exc=requests.ConnectionError("fail"))
        user = User.verify("gdedov", client=client)
        assert not user.verified and "unavailable" in user.error.lower()

    def test_sanitizes_at_prefix(self) -> None:
        client = _mock_client(200, {"id": "x", "username": "alice"})
        User.verify("  @Alice  ", client=client)
        client.get_user.assert_called_once_with("alice")


# ===================================================================
# MattermostClient (HTTP layer)
# ===================================================================


@pytest.mark.unit
class TestMattermostClient:
    def _client(self) -> MattermostClient:
        return MattermostClient("https://mm.test", "tok")

    @patch(_MOCK_GET)
    def test_200(self, mock_get: MagicMock) -> None:
        mock_get.return_value = MagicMock(
            status_code=200, json=lambda: {"id": "a", "username": "alice"}
        )
        data = self._client().get_user("alice")
        assert data["username"] == "alice"

    @patch(_MOCK_GET)
    def test_404_raises(self, mock_get: MagicMock) -> None:
        resp = requests.Response()
        resp.status_code = 404
        mock_get.return_value = resp
        with pytest.raises(requests.HTTPError):
            self._client().get_user("ghost")

    @patch(_MOCK_GET)
    def test_5xx_retries(self, mock_get: MagicMock) -> None:
        resp = requests.Response()
        resp.status_code = 502
        resp._content = b"Bad Gateway"
        mock_get.return_value = resp
        with pytest.raises(requests.HTTPError):
            self._client().get_user("alice")
        assert mock_get.call_count == 3

    @patch(_MOCK_GET)
    def test_5xx_recovers(self, mock_get: MagicMock) -> None:
        bad = requests.Response()
        bad.status_code = 502
        bad._content = b"Bad Gateway"
        good = MagicMock(status_code=200, json=lambda: {"id": "x", "username": "alice"})
        mock_get.side_effect = [bad, good]
        data = self._client().get_user("alice")
        assert data["username"] == "alice"
        assert mock_get.call_count == 2

    def test_is_retryable_false_for_generic_exception(self) -> None:
        from job_submitter.core.mattermost.client import _is_retryable

        assert _is_retryable(ValueError("nope")) is False

    def test_resolve_token_reads_nfs_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("MM_TOKEN", raising=False)
        token_file = tmp_path / ".mm_token"
        token_file.write_text("nfs-token\n")
        monkeypatch.setattr(
            "job_submitter.core.mattermost.client._TOKEN_FILE_PATHS",
            (str(token_file),),
        )
        assert MattermostClient._resolve_token() == "nfs-token"

    @patch(_MOCK_GET)
    def test_network_error_retries(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = requests.ConnectionError("fail")
        with pytest.raises(requests.ConnectionError):
            self._client().get_user("alice")
        assert mock_get.call_count == 3

    @patch(_MOCK_GET)
    def test_caches(self, mock_get: MagicMock) -> None:
        mock_get.return_value = MagicMock(
            status_code=200, json=lambda: {"id": "x", "username": "alice"}
        )
        c = self._client()
        c.get_user("alice")
        c.get_user("alice")
        assert mock_get.call_count == 1

    def test_from_env_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MM_TOKEN", raising=False)
        monkeypatch.setattr(
            "job_submitter.core.mattermost.client._TOKEN_FILE_PATHS", ()
        )
        assert MattermostClient.from_env() is None

    def test_from_env_with_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MM_TOKEN", "tok")
        assert MattermostClient.from_env() is not None

    def test_resolve_token_from_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("MM_TOKEN", raising=False)
        token_file = tmp_path / ".mm_token"
        token_file.write_text("file-token\n")
        monkeypatch.setattr(
            "job_submitter.core.mattermost.client._TOKEN_FILE_PATHS", (str(token_file),)
        )
        assert MattermostClient._resolve_token() == "file-token"

    def test_resolve_token_skips_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MM_TOKEN", raising=False)
        monkeypatch.setattr(
            "job_submitter.core.mattermost.client._TOKEN_FILE_PATHS", ("/no/such/file",)
        )
        assert MattermostClient._resolve_token() == ""

    def test_resolve_token_skips_empty_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("MM_TOKEN", raising=False)
        empty = tmp_path / ".mm_token"
        empty.write_text("  \n")
        monkeypatch.setattr(
            "job_submitter.core.mattermost.client._TOKEN_FILE_PATHS", (str(empty),)
        )
        assert MattermostClient._resolve_token() == ""

    def test_resolve_token_handles_os_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.delenv("MM_TOKEN", raising=False)
        token_file = tmp_path / ".mm_token"
        token_file.write_text("secret")  # exists, is_file() → True
        monkeypatch.setattr(
            "job_submitter.core.mattermost.client._TOKEN_FILE_PATHS",
            (str(token_file),),
        )
        monkeypatch.setattr(
            Path,
            "read_text",
            lambda *a, **kw: (_ for _ in ()).throw(OSError("disk error")),
        )
        assert MattermostClient._resolve_token() == ""


# ===================================================================
# Assertor integration
# ===================================================================


@pytest.mark.unit
class TestAssertUserLogin:
    def _verified(self, name: str = "tester") -> User:
        return User(username=name, mm_id="u1", first_name="T", verified=True)

    def _not_found(self, name: str = "ghost") -> User:
        return User(username=name, error=f"User '{name}' not found in Mattermost")

    def _net_err(self) -> User:
        return User(username="tester", error="Mattermost unavailable: timeout")

    def _skipped(self) -> User:
        return User(username="tester", skipped=True, error="No MM client")

    def test_skips_service_login(self, cloud_template) -> None:
        job = DummyJob(
            user_login="autotest", exp_name="e", region="SR008", submission_type="cloud"
        )
        BaseJobAssertor().assert_job(job, cloud_template)  # no mock needed, no MM call

    @patch("job_submitter.asserts.base.User.verify")
    def test_passes_on_verified(self, mock_v, cloud_template) -> None:
        mock_v.return_value = self._verified()
        job = DummyJob(
            user_login="tester", exp_name="e", region="SR008", submission_type="cloud"
        )
        BaseJobAssertor().assert_job(job, cloud_template)

    @patch("job_submitter.asserts.base.User.verify")
    def test_fails_on_not_found(self, mock_v, cloud_template) -> None:
        mock_v.return_value = self._not_found()
        job = DummyJob(
            user_login="ghost", exp_name="e", region="SR008", submission_type="cloud"
        )
        with pytest.raises(AssertionError, match="not found"):
            BaseJobAssertor().assert_job(job, cloud_template)

    @patch("job_submitter.asserts.base.User.verify")
    def test_warns_on_network_error(self, mock_v, cloud_template) -> None:
        mock_v.return_value = self._net_err()
        job = DummyJob(
            user_login="tester", exp_name="e", region="SR008", submission_type="cloud"
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            BaseJobAssertor().assert_job(job, cloud_template)
            assert any("unavailable" in str(x.message).lower() for x in w)

    @patch("job_submitter.asserts.base.User.verify")
    def test_warns_when_no_token(self, mock_v, cloud_template) -> None:
        mock_v.return_value = self._skipped()
        job = DummyJob(
            user_login="tester", exp_name="e", region="SR008", submission_type="cloud"
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            BaseJobAssertor().assert_job(job, cloud_template)
            assert any("MM_TOKEN" in str(x.message) for x in w)

    @patch("job_submitter.asserts.base.User.verify")
    def test_fails_on_invalid_chars(self, mock_v, cloud_template) -> None:
        mock_v.return_value = User(
            username="юзер", error="Username 'юзер' has invalid characters."
        )
        job = DummyJob(
            user_login="юзер", exp_name="e", region="SR008", submission_type="cloud"
        )
        with pytest.raises(AssertionError, match="invalid"):
            BaseJobAssertor().assert_job(job, cloud_template)
