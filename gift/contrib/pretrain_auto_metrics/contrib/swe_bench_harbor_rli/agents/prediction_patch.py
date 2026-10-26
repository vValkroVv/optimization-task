import asyncio
import logging
from pathlib import Path

from harbor.environments.base import BaseEnvironment


_PATCH_EXPORT_RETRIES = 2
_PATCH_EXPORT_RETRY_BACKOFF_SEC = 2.0
_PATCH_EXPORT_COMMAND = """
set -e
cd /testbed
index_file="$(mktemp)"
rm -f "$index_file"
trap 'rm -f "$index_file"' EXIT
GIT_INDEX_FILE="$index_file" git read-tree HEAD
GIT_INDEX_FILE="$index_file" git add -A
GIT_INDEX_FILE="$index_file" git diff --cached HEAD
""".strip()


async def export_prediction_patch(
    environment: BaseEnvironment,
    destination: Path,
    logger: logging.Logger,
    *,
    env: dict[str, str] | None = None,
) -> None:
    for attempt in range(_PATCH_EXPORT_RETRIES + 1):
        try:
            result = await environment.exec(
                _PATCH_EXPORT_COMMAND,
                env=env,
                timeout_sec=120,
            )
        except Exception as exc:
            if attempt < _PATCH_EXPORT_RETRIES:
                await asyncio.sleep(_PATCH_EXPORT_RETRY_BACKOFF_SEC)
                continue
            logger.debug("Failed to export prediction patch: %s", exc)
            return

        output = "\n".join(value for value in (result.stdout, result.stderr) if value)
        connection_lost = (
            result.return_code in (-1, 124)
            or "exec-proxy connection lost" in output.lower()
        )
        if connection_lost and attempt < _PATCH_EXPORT_RETRIES:
            await asyncio.sleep(_PATCH_EXPORT_RETRY_BACKOFF_SEC)
            continue
        if result.return_code != 0:
            logger.debug(
                "Failed to export prediction patch: return_code=%s, output=%r",
                result.return_code,
                output,
            )
            return

        patch = result.stdout or ""
        if patch.strip():
            destination.write_text(patch, encoding="utf-8")
        return
