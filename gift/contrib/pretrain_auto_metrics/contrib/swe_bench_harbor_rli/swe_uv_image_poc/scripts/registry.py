from __future__ import annotations

import argparse
import json
import ssl
import urllib.parse
import urllib.request
from urllib.error import HTTPError
from typing import Any

GATEWAY_URL = "https://rli.sberdevices.ru"
ENV_VERSION = "1.0.1"
HARNESS_ENV_VERSION = "1.0.2"
SUPPORTED_ENV_VERSIONS = (ENV_VERSION, HARNESS_ENV_VERSION)
BACKEND_TYPE = "SWE"
MAX_CONCURRENCY = 10
API_VERSION = "v2"
RESOURCE_PROFILE: dict[str, object] = {
    "cpu": {"requests": 1000, "limits": 1000},
    "memory": {"requests": 2048, "limits": 4096},
    "ephemeral": {"requests": 10240, "limits": 10240},
}
HARNESS_RESOURCE_PROFILE: dict[str, object] = {
    "cpu": {"requests": 1000, "limits": 2000},
    "memory": {"requests": 4096, "limits": 8192},
    "ephemeral": {"requests": 10240, "limits": 20480},
}


def resource_profile(version: str) -> dict[str, object]:
    if version == ENV_VERSION:
        return RESOURCE_PROFILE
    if version == HARNESS_ENV_VERSION:
        return HARNESS_RESOURCE_PROFILE
    raise ValueError(f"unsupported environment version: {version}")


def ssl_context() -> ssl.SSLContext:
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context


def request_json(
    method: str,
    path: str,
    payload: dict[str, object] | None = None,
    *,
    gateway_url: str = GATEWAY_URL,
) -> dict[str, Any] | None:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        gateway_url.rstrip("/") + path,
        data=data,
        method=method,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=ssl_context()),
    )
    with opener.open(request, timeout=120) as response:
        body = response.read().decode("utf-8", errors="replace")
        return json.loads(body) if body.strip() else None


def environment_name(task_id: str) -> str:
    short_id = task_id.split("__", 1)[1] if "__" in task_id else task_id
    return f"swe-bench-{short_id}"


def registration_payload(
    name: str,
    image: str,
    *,
    digest: str | None = None,
    version: str = ENV_VERSION,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": name,
        "version": version,
        "type": "ENV",
        "image": image,
        "backend_type": BACKEND_TYPE,
        "max_concurrency": MAX_CONCURRENCY,
        "api_version": API_VERSION,
        "resource_profile": resource_profile(version),
    }
    if digest:
        payload["image_digest"] = digest
    return payload


def registration_method(existing: dict[str, Any] | None) -> str:
    return "PUT" if existing is not None else "POST"


def _registry_path(name: str, *, version: str = ENV_VERSION) -> str:
    return f"/v1/registry/ENV/{urllib.parse.quote(name, safe='')}/{version}"


def _unwrap_response(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if isinstance(payload, dict) and isinstance(payload.get("result"), dict):
        return payload["result"]
    return payload


def get_existing(
    name: str,
    *,
    version: str = ENV_VERSION,
) -> dict[str, Any] | None:
    try:
        return _unwrap_response(
            request_json("GET", _registry_path(name, version=version))
        )
    except HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def _find_digest(payload: object) -> str | None:
    if isinstance(payload, dict):
        for key in ("digest", "image_digest", "manifest_digest"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
        image = payload.get("image")
        if isinstance(image, str) and "@" in image:
            return image.rsplit("@", 1)[1]
        for value in payload.values():
            digest = _find_digest(value)
            if digest:
                return digest
    elif isinstance(payload, list):
        for value in payload:
            digest = _find_digest(value)
            if digest:
                return digest
    return None


def is_sha256_digest(digest: str | None) -> bool:
    if digest is None or len(digest) != 71 or not digest.startswith("sha256:"):
        return False
    return all(character in "0123456789abcdef" for character in digest[7:])


def registration_matches(
    existing: dict[str, Any] | None,
    image: str,
    *,
    digest: str | None = None,
    version: str = ENV_VERSION,
) -> bool:
    if not isinstance(existing, dict) or existing.get("image") != image:
        return False
    expected = {
        "version": version,
        "type": "ENV",
        "backend_type": BACKEND_TYPE,
        "max_concurrency": MAX_CONCURRENCY,
        "api_version": API_VERSION,
        "resource_profile": resource_profile(version),
        "deprecated": False,
    }
    return all(existing.get(key) == value for key, value in expected.items()) and (
        digest is None or _find_digest(existing) == digest
    )


def register(
    name: str,
    image: str,
    *,
    digest: str | None = None,
    version: str = ENV_VERSION,
) -> dict[str, Any] | None:
    if version == HARNESS_ENV_VERSION and not is_sha256_digest(digest):
        raise ValueError(f"sha256 image digest is required for version {version}")
    existing = get_existing(name, version=version)
    if registration_matches(existing, image, digest=digest, version=version):
        return existing
    payload = registration_payload(name, image, digest=digest, version=version)
    method = registration_method(existing)
    path = "/v1/registry" if method == "POST" else _registry_path(name, version=version)
    response = request_json(method, path, payload)
    verified = get_existing(name, version=version)
    if not isinstance(verified, dict):
        raise RuntimeError(f"registry entry missing for {name}: {verified!r}")
    expected = {
        "name": name,
        "version": version,
        "type": "ENV",
        "image": image,
        "backend_type": BACKEND_TYPE,
        "max_concurrency": MAX_CONCURRENCY,
        "api_version": API_VERSION,
        "resource_profile": resource_profile(version),
        "deprecated": False,
    }
    mismatches = {
        key: (value, verified.get(key))
        for key, value in expected.items()
        if verified.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"registry mismatch for {name}: {mismatches!r}")
    if digest and _find_digest(verified) != digest:
        raise RuntimeError(f"registry digest mismatch for {name}: {verified!r}")
    return response


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("task_id")
    parser.add_argument("image")
    parser.add_argument("--digest")
    parser.add_argument(
        "--version",
        choices=SUPPORTED_ENV_VERSIONS,
        default=ENV_VERSION,
    )
    args = parser.parse_args()
    response = register(
        environment_name(args.task_id),
        args.image,
        digest=args.digest,
        version=args.version,
    )
    print(json.dumps(response, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
