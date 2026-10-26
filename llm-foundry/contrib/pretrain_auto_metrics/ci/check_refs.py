"""Проверяет, что каждый пин X_REF из refs.env стоит на HEAD ветки по умолчанию репозитория X_REPO_URL."""

from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REF_SUFFIX = "_REF"
REPO_URL_SUFFIX = "_REPO_URL"
LS_REMOTE_TIMEOUT_S = 120
CREDENTIALS_BY_HOST = {
    "gitlab.ai.cloud.ru": ("gitlab-ci-token", "CI_JOB_TOKEN"),
    "gitverse.ru": ("oauth2", "GITVERSE_READONLY_TOKEN"),
}


def read_env(path: Path) -> dict[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return dict(line.split("=", 1) for line in lines if line.strip())


def https_url(repo_url: str) -> str:
    host = repo_url.split("/", 1)[0]
    if host not in CREDENTIALS_BY_HOST:
        return f"https://{repo_url}"
    user, token_variable = CREDENTIALS_BY_HOST[host]
    return f"https://{user}:{os.environ[token_variable]}@{repo_url}"


def check(repo_url: str, ref: str) -> str | None:
    """Возвращает текст ошибки или None, если ref указывает на HEAD ветки по умолчанию.

    ref это sha коммита или имя тега/ветки; имя резолвится в коммит тем же ls-remote.
    """
    host = repo_url.split("/", 1)[0]
    if host in CREDENTIALS_BY_HOST and CREDENTIALS_BY_HOST[host][1] not in os.environ:
        return f"нет переменной {CREDENTIALS_BY_HOST[host][1]} для {host}"
    command = [
        "git",
        "ls-remote",
        "--symref",
        https_url(repo_url),
        "HEAD",
        ref,
        f"{ref}^{{}}",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=LS_REMOTE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return f"ls-remote {repo_url} не уложился в {LS_REMOTE_TIMEOUT_S} с"
    if result.returncode != 0:
        return f"ls-remote {repo_url}: {result.stderr.strip()}"
    default_branch = ""
    shas: dict[str, str] = {}
    for line in result.stdout.splitlines():
        sha, name = line.split("\t")
        if sha.startswith("ref: "):
            default_branch = sha.removeprefix("ref: refs/heads/")
        else:
            shas[name] = sha
    head = shas["HEAD"]
    pinned = (
        shas.get(f"refs/tags/{ref}^{{}}")
        or shas.get(f"refs/tags/{ref}")
        or shas.get(f"refs/heads/{ref}")
        or ref
    )
    if pinned != head:
        return f"HEAD ветки по умолчанию {default_branch} в {repo_url} это {head}"
    return None


def main(refs_path: Path) -> int:
    env = read_env(refs_path)
    names = sorted(
        key.removesuffix(REF_SUFFIX) for key in env if key.endswith(REF_SUFFIX)
    )
    missing_urls = [name for name in names if name + REPO_URL_SUFFIX not in env]
    for name in missing_urls:
        print(f"FAIL  {name}{REF_SUFFIX}: нет {name}{REPO_URL_SUFFIX} в {refs_path}")
    names = [name for name in names if name not in missing_urls]
    with ThreadPoolExecutor(max_workers=len(names)) as pool:
        errors = list(
            pool.map(
                lambda name: check(env[name + REPO_URL_SUFFIX], env[name + REF_SUFFIX]),
                names,
            )
        )
    for name, error in zip(names, errors):
        status = "ok  " if error is None else "FAIL"
        print(
            f"{status}  {name}{REF_SUFFIX}={env[name + REF_SUFFIX]}"
            + ("" if error is None else f": {error}")
        )
    return 1 if missing_urls or any(errors) else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
