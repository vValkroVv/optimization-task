from __future__ import annotations

import ssl

from swe_bench_harbor_rli.swe_uv_image_poc.scripts import registry


def test_new_and_existing_registration_methods() -> None:
    assert registry.registration_method(None) == "POST"
    assert registry.registration_method({"image": "old"}) == "PUT"


def test_environment_name_and_payload() -> None:
    assert registry.environment_name("django__django-13343") == "swe-bench-django-13343"
    payload = registry.registration_payload(
        "swe-bench-django-13343", "image:1.0.1", digest="sha256:test"
    )
    assert payload["version"] == "1.0.1"
    assert payload["image_digest"] == "sha256:test"
    assert payload["api_version"] == "v2"
    assert "digest" not in payload


def test_ssl_verification_matches_gigagym_registration() -> None:
    context = registry.ssl_context()

    assert context.check_hostname is False
    assert context.verify_mode == ssl.CERT_NONE


def test_get_existing_unwraps_result_envelope(monkeypatch) -> None:
    monkeypatch.setattr(
        registry,
        "request_json",
        lambda method, path, payload=None, **kwargs: {
            "result": {"image": "image:1.0.1"}
        },
    )

    assert registry.get_existing("swe-bench-django-13343") == {"image": "image:1.0.1"}


def test_matching_registration_is_not_updated(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []
    state: dict[str, object] = {}

    def fake_request(method, path, payload=None, **kwargs):
        calls.append((method, path))
        if method == "GET":
            return state or None
        state.update(payload or {})
        state["digest"] = "sha256:test"
        state["deprecated"] = False
        return state

    monkeypatch.setattr(registry, "request_json", fake_request)
    registry.register("swe-bench-django-13343", "image:1.0.1", digest="sha256:test")
    registry.register("swe-bench-django-13343", "image:1.0.1", digest="sha256:test")

    assert [method for method, _ in calls] == [
        "GET",
        "POST",
        "GET",
        "GET",
    ]


def test_old_image_is_updated_with_put(monkeypatch) -> None:
    calls: list[str] = []
    state = {
        "name": "swe-bench-django-13343",
        "version": "1.0.1",
        "type": "ENV",
        "image": "image:1.0.1-uv-poc",
        "backend_type": "SWE",
        "max_concurrency": 10,
        "api_version": "v2",
        "deprecated": False,
    }

    def fake_request(method, path, payload=None, **kwargs):
        calls.append(method)
        if method == "GET":
            return state
        state.update(payload or {})
        return state

    monkeypatch.setattr(registry, "request_json", fake_request)

    registry.register("swe-bench-django-13343", "image:1.0.1")

    assert calls == ["GET", "PUT", "GET"]
    assert state["image"] == "image:1.0.1"


def test_matching_image_with_wrong_configuration_is_updated(monkeypatch) -> None:
    calls: list[str] = []
    state = {
        "name": "swe-bench-django-13343",
        "version": "1.0.1",
        "type": "ENV",
        "image": "image:1.0.1",
        "backend_type": "SWE",
        "max_concurrency": 1,
        "api_version": "v2",
        "deprecated": False,
    }

    def fake_request(method, path, payload=None, **kwargs):
        calls.append(method)
        if method == "GET":
            return state
        state.update(payload or {})
        state["deprecated"] = False
        return state

    monkeypatch.setattr(registry, "request_json", fake_request)

    registry.register("swe-bench-django-13343", "image:1.0.1")

    assert calls == ["GET", "PUT", "GET"]
    assert state["max_concurrency"] == 10
