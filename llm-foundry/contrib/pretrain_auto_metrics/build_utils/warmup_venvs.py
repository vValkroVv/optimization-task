"""Pre-warm per-adapter uv-envs for GigaMetric at Docker build time.

Usage:
    /home/user/papayametric_venv/bin/python build_utils/warmup_venvs.py [adapter ...]

Без аргументов: читает `<papayametric_root>/pyproject.toml`,
из `project.optional-dependencies` забирает все ключи кроме `dev` и
prewarm-ит все соответствующие адаптеры.

Fail-soft: один сломанный адаптер не валит весь build — exit 0.
Каждый адаптер ставит свои deps в `<papayametric_root>/.uv_envs/<adapter>/`
через `BenchmarkEnvironmentManager`; envs затем COPY'атся в финальный образ.
"""

from __future__ import annotations

import asyncio
import sys
import traceback
from pathlib import Path

import tomli

PAPAYAMETRIC_ROOT = Path("/home/user/papayametric")


def _discover_adapters(root: Path) -> list[str]:
    with open(root / "pyproject.toml", "rb") as f:
        data = tomli.load(f)
    keys = data.get("project", {}).get("optional-dependencies", {}).keys()
    return [f"{k}-adapter" for k in keys if k != "dev"]


def _warmup(adapters: list[str]) -> int:
    sys.path.insert(0, str(PAPAYAMETRIC_ROOT))
    from src.env_manager.environment_manager import BenchmarkEnvironmentManager

    print(f"Prewarming {len(adapters)} adapter envs (fail-soft)", flush=True)
    mgr = BenchmarkEnvironmentManager()
    failed: list[str] = []
    ok_count = 0
    for name in adapters:
        print(f"--- {name} ---", flush=True)
        try:
            ok = asyncio.run(mgr.create_environment(name))
        except Exception as e:
            ok = False
            print(f"EXCEPTION: {e}", flush=True)
            traceback.print_exc()
        if ok:
            ok_count += 1
            print(f"OK: {name}", flush=True)
        else:
            failed.append(name)
            print(f"FAILED: {name}", flush=True)

    print(
        f"Prewarm summary: {ok_count}/{len(adapters)} ok, {len(failed)} failed",
        flush=True,
    )
    if failed:
        print("Failed adapters: " + ", ".join(failed), flush=True)
    return 0  # always 0 — failed adapters fall back to runtime install


def main() -> int:
    adapters = sys.argv[1:] or _discover_adapters(PAPAYAMETRIC_ROOT)
    return _warmup(adapters)


if __name__ == "__main__":
    sys.exit(main())
