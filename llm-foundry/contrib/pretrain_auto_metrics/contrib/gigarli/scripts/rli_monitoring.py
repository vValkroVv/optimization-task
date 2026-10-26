#!/usr/bin/env python3
"""Collect RLI platform monitoring snapshots.

This script is intentionally dependency-free so it can run from debug shells
without installing project extras. It is a manual diagnostic helper, not part
of benchmark execution.
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_BASE_URL = "https://rli.sberdevices.ru"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "rli_monitoring_snapshots"
DEFAULT_TIMEOUT_SECONDS = 20.0


@dataclass(frozen=True)
class RequestSpec:
    name: str
    path: str
    query: dict[str, str | int] | None = None


class RLIHttpClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float,
        insecure: bool,
        use_proxy: bool,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.context = self._build_ssl_context(insecure)
        self.opener = self._build_opener(use_proxy)

    @staticmethod
    def _build_ssl_context(insecure: bool) -> ssl.SSLContext | None:
        if not insecure:
            return None
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx

    def _build_opener(self, use_proxy: bool) -> urllib.request.OpenerDirector:
        handlers = []
        if not use_proxy:
            handlers.append(urllib.request.ProxyHandler({}))
        if self.context is not None:
            handlers.append(urllib.request.HTTPSHandler(context=self.context))
        return urllib.request.build_opener(*handlers)

    def get(
        self,
        path: str,
        query: dict[str, str | int] | None = None,
    ) -> dict[str, Any]:
        url = self._url(path, query)
        req = urllib.request.Request(url, method="GET")
        try:
            with self.opener.open(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                payload = json.loads(raw) if raw else None
                return {
                    "ok": True,
                    "url": url,
                    "status_code": resp.status,
                    "payload": payload,
                }
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            return {
                "ok": False,
                "url": url,
                "status_code": exc.code,
                "error": body,
                "payload": _json_or_none(body),
            }
        except Exception as exc:
            return {
                "ok": False,
                "url": url,
                "status_code": None,
                "error": repr(exc),
                "payload": None,
            }

    def _url(self, path: str, query: dict[str, str | int] | None = None) -> str:
        if not path.startswith("/"):
            path = f"/{path}"
        url = f"{self.base_url}{path}"
        if query:
            url = f"{url}?{urllib.parse.urlencode(query)}"
        return url


def _json_or_none(raw: str) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _utc_ts() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _safe_ts_for_path(ts: str) -> str:
    return ts.replace(":", "-").replace(".", "-").replace("Z", "Z")


def _result(response: dict[str, Any]) -> Any:
    payload = response.get("payload")
    if isinstance(payload, dict):
        return payload.get("result")
    return None


def _error_code(response: dict[str, Any]) -> str | None:
    payload = response.get("payload")
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            code = error.get("code")
            return str(code) if code is not None else None
    return None


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _collect_snapshot(
    client: RLIHttpClient,
    *,
    window: str,
    errors_recent_limit: int,
    include_recent_errors: bool,
) -> dict[str, Any]:
    started = time.monotonic()
    ts = _utc_ts()

    requests = [
        RequestSpec("dashboard", "/v1/system/dashboard", {"window": window}),
    ]
    if include_recent_errors:
        requests.append(
            RequestSpec(
                "errors_recent",
                "/v1/errors/recent",
                {"limit": errors_recent_limit},
            )
        )

    responses: dict[str, Any] = {}
    for spec in requests:
        responses[spec.name] = client.get(spec.path, spec.query)

    elapsed_ms = int((time.monotonic() - started) * 1000)
    return {
        "collected_at": ts,
        "base_url": client.base_url,
        "elapsed_ms": elapsed_ms,
        "responses": responses,
        "summary": _build_summary(responses),
    }


def _build_summary(responses: dict[str, Any]) -> dict[str, Any]:
    dashboard = _result(responses.get("dashboard", {}))
    recent_errors = _as_list(_result(responses.get("errors_recent", {})))

    failed_requests = {
        name: {
            "status_code": response.get("status_code"),
            "error_code": _error_code(response),
            "error": response.get("error"),
            "url": response.get("url"),
        }
        for name, response in responses.items()
        if not response.get("ok")
    }

    return {
        "dashboard": _pick_dict(
            dashboard,
            [
                "active_sessions",
                "sessions_by_status",
                "avg_step_latency_ms",
                "error_rate_percent_1m",
            ],
        ),
        "recent_errors_count": len(recent_errors),
        "recent_errors": [
            _compact_error(error)
            for error in recent_errors
            if isinstance(error, dict)
        ],
        "failed_requests": failed_requests,
    }


def _compact_error(error: dict[str, Any]) -> dict[str, Any]:
    return _pick_dict(
        error,
        [
            "timestamp",
            "http_status",
            "code",
            "category",
            "endpoint",
            "request_id",
            "id",
            "session_id",
            "message",
        ],
    )


def _pick_dict(value: Any, keys: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {key: value.get(key) for key in keys if key in value}


def _write_snapshot(output_dir: Path, snapshot: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"rli_monitoring_{_safe_ts_for_path(snapshot['collected_at'])}.json"
    path = output_dir / filename
    body = json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n"
    path.write_text(body, encoding="utf-8")

    latest_path = output_dir / "latest.json"
    latest_path.write_text(body, encoding="utf-8")
    return path


def _print_summary(snapshot: dict[str, Any], output_path: Path | None) -> None:
    summary = snapshot["summary"]
    print(f"[{snapshot['collected_at']}] RLI monitoring snapshot")
    print(f"base_url: {snapshot['base_url']}")

    responses = snapshot.get("responses", {})
    ok_count = sum(1 for response in responses.values() if response.get("ok"))
    print(f"elapsed_ms: {snapshot['elapsed_ms']}")
    print(f"requests: {ok_count}/{len(responses)} ok")
    if output_path:
        print(f"written: {output_path}")

    dashboard = summary.get("dashboard", {})
    if dashboard:
        print("dashboard:")
        print(f"  active_sessions: {dashboard.get('active_sessions')}")
        print(
            "  sessions_by_status: "
            f"{json.dumps(dashboard.get('sessions_by_status'), ensure_ascii=False)}"
        )
        print(f"  avg_step_latency_ms: {dashboard.get('avg_step_latency_ms')}")
        print(f"  error_rate_percent_1m: {dashboard.get('error_rate_percent_1m')}")

    print(f"recent_errors: {summary.get('recent_errors_count', 0)}")
    for error in summary.get("recent_errors", [])[:5]:
        print(
            "  "
            f"{error.get('timestamp')} "
            f"status={error.get('http_status')} code={error.get('code')} "
            f"category={error.get('category')} endpoint={error.get('endpoint')} "
            f"request_id={error.get('request_id')}"
        )
        message = error.get("message")
        if message:
            print(f"    {message}")

    failed = summary.get("failed_requests") or {}
    if failed:
        print("failed_requests:")
        for name, item in failed.items():
            print(
                f"  {name}: status={item.get('status_code')} "
                f"code={item.get('error_code')} url={item.get('url')}"
            )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.getenv("RLI_API_BASE", DEFAULT_BASE_URL),
        help=f"RLI gateway base URL. Default: env RLI_API_BASE or {DEFAULT_BASE_URL}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for JSON snapshots. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument("--window", default="5m", help="Dashboard time window.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="HTTP timeout in seconds.",
    )

    verify_tls_default = os.getenv("RLI_VERIFY_TLS", "").lower() in {
        "1",
        "true",
        "yes",
    }
    parser.add_argument(
        "--insecure",
        action="store_true",
        default=not verify_tls_default,
        help="Disable TLS certificate verification. Enabled by default for this RLI debug script.",
    )
    parser.add_argument(
        "--verify-tls",
        action="store_false",
        dest="insecure",
        help="Enable TLS certificate verification.",
    )
    parser.add_argument(
        "--use-proxy",
        action="store_true",
        default=os.getenv("RLI_USE_PROXY", "").lower() in {"1", "true", "yes"},
        help="Use HTTP_PROXY/HTTPS_PROXY from the environment. Disabled by default.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of snapshots to collect. Use 0 for forever.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=60.0,
        help="Seconds between snapshots.",
    )
    parser.add_argument(
        "--errors-recent-limit",
        type=int,
        default=20,
        help="Recent errors limit.",
    )
    parser.add_argument(
        "--no-recent-errors",
        action="store_true",
        help="Do not call /v1/errors/recent.",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Write JSON snapshots to --output-dir.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print summaries. Requires --save-json.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.iterations < 0:
        print("--iterations must be >= 0", file=sys.stderr)
        return 2
    if args.interval <= 0:
        print("--interval must be > 0", file=sys.stderr)
        return 2
    if args.quiet and not args.save_json:
        print("--quiet requires --save-json", file=sys.stderr)
        return 2

    client = RLIHttpClient(
        args.base_url,
        timeout=args.timeout,
        insecure=args.insecure,
        use_proxy=args.use_proxy,
    )

    iteration = 0
    while True:
        iteration += 1
        snapshot = _collect_snapshot(
            client,
            window=args.window,
            errors_recent_limit=args.errors_recent_limit,
            include_recent_errors=not args.no_recent_errors,
        )

        output_path = _write_snapshot(args.output_dir, snapshot) if args.save_json else None
        if not args.quiet:
            _print_summary(snapshot, output_path)

        if args.iterations != 0 and iteration >= args.iterations:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
