from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .application.error_handlers import register_error_handlers
from .application.service import ProxyService
from .config import configure_logging, load_settings, print_settings
from .infrastructure.sglang_gateway import SGLangGateway
from .infrastructure.tokenizer_gateway import TokenizerGateway
from .middleware import MetricsMiddleware, MetricsReporter, ProxyMetrics
from .presentation.router import build_router

settings = load_settings()
print_settings(settings)
configure_logging(settings)

metrics = ProxyMetrics()
metrics_reporter = MetricsReporter(metrics)
tokenizer_gateway = TokenizerGateway(settings)
sglang_gateway = SGLangGateway()
service = ProxyService(
    settings=settings,
    tokenizer_gateway=tokenizer_gateway,
    sglang_gateway=sglang_gateway,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    service.preload()

    app.state.http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(21600.0),
        limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
    )
    app.state.metrics_task = None
    if settings.stats_enabled:
        app.state.metrics_task = asyncio.create_task(
            metrics_reporter.run(settings.stats_interval_sec)
        )

    try:
        yield
    finally:
        task = getattr(app.state, "metrics_task", None)
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        if settings.stats_enabled:
            metrics_reporter.log_final()

        client: httpx.AsyncClient = app.state.http_client
        await client.aclose()


app = FastAPI(title="OBM proxy server", lifespan=lifespan)
app.add_middleware(MetricsMiddleware, metrics=metrics)
register_error_handlers(app, sglang_port=settings.sglang_port)
app.include_router(build_router(service, settings))
latency_service = ProxyService(
    settings=replace(settings, priority=1),
    tokenizer_gateway=tokenizer_gateway,
    sglang_gateway=sglang_gateway,
    dialog_affinity=True,
)
app.include_router(build_router(latency_service, settings), prefix="/latency")


@app.get("/metrics")
async def get_metrics() -> JSONResponse:
    return JSONResponse(content=metrics.snapshot())
