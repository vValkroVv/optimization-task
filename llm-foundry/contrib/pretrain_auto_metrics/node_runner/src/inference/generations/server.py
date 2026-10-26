from __future__ import annotations

import asyncio
import json
import time
from argparse import ArgumentParser
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, Response

from .store import DumpIndex, ShardWriter, request_key

UPSTREAM_TIMEOUT_SEC = 21600.0
BODY_PREVIEW_CHARS = 2000
MISS_MARKER = "No recorded generation"


readiness = APIRouter()


@readiness.get("/ready")
async def ready() -> JSONResponse:
    return JSONResponse(content={"ready": True})


def recording_app(dump_dir: Path, profile: str, upstream_url: str) -> FastAPI:
    writer = ShardWriter(dump_dir, profile)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(UPSTREAM_TIMEOUT_SEC),
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
        )
        try:
            yield
        finally:
            await app.state.http_client.aclose()
            writer.close()

    app = FastAPI(title="Generation recorder", lifespan=lifespan)
    app.include_router(readiness)

    @app.post("/{path:path}")
    async def record_and_forward(path: str, request: Request) -> Response:
        body = await request.json()
        routing_headers = {
            name: request.headers[name]
            for name in ("x-smg-routing-key",)
            if name in request.headers
        }
        upstream = await request.app.state.http_client.post(
            f"{upstream_url}/{path}", json=body, headers=routing_headers
        )
        writer.write(
            path=f"/{path}",
            body=body,
            status_code=upstream.status_code,
            response=upstream.text,
        )
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
        )

    return app


def miss_message(dump_index: DumpIndex, path: str, body: dict[str, Any]) -> str:
    preview = json.dumps(body, sort_keys=True, ensure_ascii=False)
    if len(preview) > BODY_PREVIEW_CHARS:
        preview = f"{preview[:BODY_PREVIEW_CHARS]}... (truncated)"
    key = request_key(profile=dump_index.profile, path=path, body=body)
    return (
        f"{MISS_MARKER} for profile={dump_index.profile} path={path} "
        f"key={key}. The dump holds {len(dump_index):,} keys for this profile, "
        "so the request itself is new: a shooter, a chat template or a tokenizer "
        f"changed and the generations have to be recorded again. Request was: {preview}"
    )


def replay_app(dump_dir: Path, profile: str) -> FastAPI:
    started = time.perf_counter()
    dump_index = DumpIndex(dump_dir, profile)
    print(
        f"Replaying {len(dump_index):,} recorded {profile} generations from "
        f"{dump_dir}, indexed in {time.perf_counter() - started:.1f}s",
        flush=True,
    )
    app = FastAPI(title="Generation replay")
    app.include_router(readiness)

    @app.post("/{path:path}")
    async def replay(path: str, request: Request) -> Response:
        body = await request.json()
        full_path = f"/{path}"
        record = await asyncio.to_thread(dump_index.lookup, path=full_path, body=body)
        if record is None:
            message = miss_message(dump_index, full_path, body)
            print(message, flush=True)
            return JSONResponse(status_code=404, content={"error": message})
        return Response(
            content=record["response"],
            status_code=record["status"],
            media_type="application/json",
        )

    return app


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--upstream-url")
    args = parser.parse_args()
    if args.upstream_url:
        app = recording_app(args.dump_dir, args.profile, args.upstream_url.rstrip("/"))
    else:
        app = replay_app(args.dump_dir, args.profile)
    uvicorn.run(app, host=args.host, port=args.port, log_level="error")


if __name__ == "__main__":
    main()
