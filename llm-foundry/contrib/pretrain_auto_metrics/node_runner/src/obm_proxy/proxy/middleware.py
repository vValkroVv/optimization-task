from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

SKIP_PATHS = frozenset(("/ready", "/health", "/ping", "/metrics"))


@dataclass
class _RouteStats:
    count: int = 0
    latency_sum_ms: float = 0.0
    latency_max_ms: float = 0.0
    latency_min_ms: float = float("inf")
    status_codes: dict[int, int] = field(default_factory=lambda: defaultdict(int))


class ProxyMetrics:
    """HTTP-level metrics for every route passing through the proxy.

    Thread-safe; designed to be written from a middleware and read from
    a reporter task or a /metrics endpoint without holding the GIL.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._started_at = time.monotonic()
        self._active = 0
        self._peak = 0
        self._routes: dict[str, _RouteStats] = defaultdict(_RouteStats)
        self._error_types: dict[str, int] = defaultdict(int)

    def on_start(self) -> None:
        with self._lock:
            self._active += 1
            if self._active > self._peak:
                self._peak = self._active

    def on_finish(
        self,
        *,
        route: str,
        status: int,
        latency_ms: float,
        exc_name: Optional[str] = None,
    ) -> None:
        with self._lock:
            self._active = max(0, self._active - 1)
            rs = self._routes[route]
            rs.count += 1
            rs.status_codes[status] += 1
            rs.latency_sum_ms += latency_ms
            rs.latency_max_ms = max(rs.latency_max_ms, latency_ms)
            rs.latency_min_ms = min(rs.latency_min_ms, latency_ms)
            if exc_name:
                self._error_types[exc_name] += 1

    def snapshot(self) -> dict:
        with self._lock:
            uptime = time.monotonic() - self._started_at
            total = sum(rs.count for rs in self._routes.values())
            total_err = 0

            routes = {}
            for key, rs in sorted(self._routes.items()):
                avg = rs.latency_sum_ms / rs.count if rs.count else 0
                ok = sum(
                    count
                    for code, count in rs.status_codes.items()
                    if 200 <= code < 400
                )
                client_errors = sum(
                    count
                    for code, count in rs.status_codes.items()
                    if 400 <= code < 500
                )
                server_errors = rs.count - ok - client_errors
                total_err += client_errors + server_errors
                routes[key] = {
                    "total": rs.count,
                    "ok": ok,
                    "4xx": client_errors,
                    "5xx": server_errors,
                    "avg_ms": round(avg, 1),
                    "max_ms": round(rs.latency_max_ms, 1),
                    "min_ms": round(rs.latency_min_ms, 1) if rs.count else 0,
                    "codes": dict(rs.status_codes),
                }

            return {
                "uptime_sec": round(uptime, 1),
                "active": self._active,
                "peak": self._peak,
                "total": total,
                "errors": total_err,
                "error_rate_pct": (round(total_err / total * 100, 2) if total else 0),
                "rps": round(total / uptime, 2) if uptime > 0 else 0,
                "error_types": dict(self._error_types),
                "routes": routes,
            }


class MetricsMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, metrics: ProxyMetrics) -> None:
        super().__init__(app)
        self._m = metrics

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in SKIP_PATHS:
            return await call_next(request)

        route = f"{request.method} {request.url.path}"
        self._m.on_start()
        t0 = time.perf_counter()
        status = 500
        exc_name: Optional[str] = None
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        except Exception as exc:
            exc_name = type(exc).__name__
            raise
        finally:
            ms = (time.perf_counter() - t0) * 1000.0
            self._m.on_finish(
                route=route,
                status=status,
                latency_ms=ms,
                exc_name=exc_name,
            )


class MetricsReporter:
    def __init__(self, metrics: ProxyMetrics) -> None:
        self._m = metrics
        self._prev_total = 0
        self._prev_errors = 0
        self._prev_ts = time.monotonic()

    async def run(self, interval_sec: int) -> None:
        while True:
            await asyncio.sleep(interval_sec)
            snap = self._m.snapshot()
            now = time.monotonic()
            dt = max(now - self._prev_ts, 1e-9)
            interval_req = snap["total"] - self._prev_total
            interval_err = snap["errors"] - self._prev_errors

            report: dict = {
                "kind": "periodic",
                "uptime": snap["uptime_sec"],
                "active": snap["active"],
                "peak": snap["peak"],
                "total": snap["total"],
                "interval_req": interval_req,
                "interval_err": interval_err,
                "rps_avg": snap["rps"],
                "rps_now": round(interval_req / dt, 2),
                "error_rate": f"{snap['error_rate_pct']}%",
            }
            if snap["error_types"]:
                report["error_types"] = snap["error_types"]

            print(f"[obm_proxy][perf] {json.dumps(report, ensure_ascii=False)}")
            self._prev_total = snap["total"]
            self._prev_errors = snap["errors"]
            self._prev_ts = now

    def log_final(self) -> None:
        snap = self._m.snapshot()
        snap["kind"] = "final"
        print(f"[obm_proxy][perf] {json.dumps(snap, ensure_ascii=False)}")
