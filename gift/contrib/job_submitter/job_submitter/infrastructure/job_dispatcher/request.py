from __future__ import annotations

import json
import shlex
import ssl
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

ATTEMPTS = 3


def to_urllib_request(
    method: str, url: str, data: Any = None, headers: dict[str, str] | None = None
) -> urllib.request.Request:
    return urllib.request.Request(
        url,
        data=None if data is None else json.dumps(data, allow_nan=False).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
        method=method,
    )


def send(
    method: str, url: str, data: Any = None, headers: dict[str, str] | None = None, timeout: float = 15
) -> bool:
    request = to_urllib_request(method, url, data, headers)
    # Retry every error, even 4xx: 3 attempts are cheap, per-error rules aren't worth it
    for attempt in range(ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=timeout, context=ssl._create_unverified_context()):
                pass
            return True
        except Exception as error:
            if attempt == ATTEMPTS - 1:
                print(f"{method} {url} failed: {type(error).__name__}: {error}", file=sys.stderr)
            else:
                time.sleep(4 * 2**attempt)
    return False


def build_send_command(method: str, url: str, data: Any = None, headers: dict[str, str] | None = None) -> str:
    payload = json.dumps({"method": method, "url": url, "data": data, "headers": headers or {}}, allow_nan=False)
    module = shlex.quote(str(Path(__file__).resolve()))
    return f"python3 {module} <<< {shlex.quote(payload)}"


if __name__ == "__main__":
    sys.exit(0 if send(**json.load(sys.stdin)) else 1)
