from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit


TEST_RUNTIME_ARCHIVE = "/usr/local/share/.dev_logs/test_runtime.tar.zst"
TEST_RUNTIME_ROOT = "/var/lib/terminal-bench/test_runtime"
REMOTE_TESTS_DIR = "/tmp/terminal-bench/tests"
REMOTE_LOGS_DIR = "/tmp/terminal-bench/logs"
_FIX_OCAML_CLONE = (
    "git clone https://github.com/sadiqj/ocaml/ ocaml-original "
    "&& cd ocaml-original "
    "&& git checkout tag_purposefully_broken_sweeping_changes && cd .."
)
_URL_LITERAL_RE = re.compile(r"""["'](https?://[^"']+)["']""")
_GIT_CLONE_RE = re.compile(r"""["']git["']\s*,\s*["']clone["']""")
_RUNTIME_APT_RE = re.compile(r"\bapt(?:-get)?\s+(?:update|install)\b")
_LOCAL_PIP_INDEX_RE = re.compile(
    r"""--index-url["']?\s*,?\s*["']?"""
    r"""(https?://(?:localhost|127\.0\.0\.1)(?::\d+)?"""
    r"""(?:/[^\s"',\]]*)?)""",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CachedDownload:
    url: str
    output: str

    @property
    def archive_name(self) -> str:
        return PurePosixPath(self.output).name


@dataclass(frozen=True)
class CachedGitRepository:
    url: str

    @property
    def archive_name(self) -> str:
        return PurePosixPath(urlsplit(self.url).path).name


def _logical_lines(script: str) -> list[str]:
    return re.sub(r"\\\r?\n[ \t]*", " ", script).splitlines()


def _tokens(line: str) -> list[str]:
    try:
        return shlex.split(line, comments=True)
    except ValueError as exc:
        raise ValueError(f"cannot parse test.sh line: {line!r}") from exc


def _cached_download(tokens: list[str]) -> CachedDownload | None:
    if not tokens or tokens[0] != "curl":
        return None

    output = None
    url = None
    for index, token in enumerate(tokens):
        if token in {"-o", "--output"} and index + 1 < len(tokens):
            output = tokens[index + 1]
        if token.startswith(("http://", "https://")):
            url = token

    if url is None or "astral.sh/uv/" in url:
        return None
    if output is None:
        raise ValueError(f"remote curl command has no output path: {tokens!r}")
    return CachedDownload(url=url, output=output)


def find_cached_downloads(script: str) -> list[CachedDownload]:
    downloads = []
    for line in _logical_lines(script):
        download = _cached_download(_tokens(line))
        if download is not None:
            downloads.append(download)
    return downloads


def find_cached_python_downloads(tests_dir: Path) -> list[CachedDownload]:
    downloads: dict[str, CachedDownload] = {}
    for path in sorted(tests_dir.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "urlretrieve(" not in text:
            continue
        for url in _URL_LITERAL_RE.findall(text):
            path = urlsplit(url).path
            if path.endswith(".git"):
                continue
            archive_name = PurePosixPath(path).name
            if not archive_name:
                raise ValueError(f"download URL has no file name: {url}")
            downloads[url] = CachedDownload(url=url, output=archive_name)
    return list(downloads.values())


def find_cached_git_repositories(
    tests_dir: Path,
) -> list[CachedGitRepository]:
    repositories: dict[str, CachedGitRepository] = {}
    for path in sorted(tests_dir.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if not _GIT_CLONE_RE.search(text):
            continue
        for url in _URL_LITERAL_RE.findall(text):
            if urlsplit(url).path.endswith(".git"):
                repositories[url] = CachedGitRepository(url=url)
    return list(repositories.values())


def _runtime_setup(local_pip_index: str | None = None) -> list[str]:
    if local_pip_index is None:
        pip_setup = ['export PIP_NO_INDEX="${PIP_NO_INDEX:-1}"']
    else:
        pip_setup = [
            "export PIP_NO_INDEX=0",
            f"export PIP_INDEX_URL={shlex.quote(local_pip_index)}",
        ]

    return [
        f'TEST_RUNTIME_ROOT="${{TEST_RUNTIME_ROOT:-{TEST_RUNTIME_ROOT}}}"',
        f'TEST_RUNTIME_ARCHIVE="${{TEST_RUNTIME_ARCHIVE:-{TEST_RUNTIME_ARCHIVE}}}"',
        'if [ ! -d "$TEST_RUNTIME_ROOT" ]; then',
        '  mkdir -p "$(dirname "$TEST_RUNTIME_ROOT")"',
        '  tar --zstd -xf "$TEST_RUNTIME_ARCHIVE" -C "$(dirname "$TEST_RUNTIME_ROOT")"',
        "fi",
        'export PATH="$TEST_RUNTIME_ROOT/bin:$PATH"',
        'export UV_CACHE_DIR="$TEST_RUNTIME_ROOT/uv-cache"',
        'export UV_PYTHON_INSTALL_DIR="$TEST_RUNTIME_ROOT/uv-python"',
        'export UV_OFFLINE="${UV_OFFLINE:-1}"',
        'export HF_HOME="$TEST_RUNTIME_ROOT/huggingface"',
        'export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-$UV_OFFLINE}"',
        'export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-$UV_OFFLINE}"',
        *pip_setup,
        'export PIP_FIND_LINKS="$TEST_RUNTIME_ROOT/wheels"',
        'if compgen -G "$TEST_RUNTIME_ROOT/debs/*.deb" >/dev/null; then',
        '  dpkg -i "$TEST_RUNTIME_ROOT"/debs/*.deb',
        "fi",
    ]


def rewrite_runtime_paths(text: str) -> str:
    rewritten = text
    for source, target in (
        ("/logs", REMOTE_LOGS_DIR),
        ("/tests", REMOTE_TESTS_DIR),
    ):
        rewritten = re.sub(
            rf"(?:(?<=^)|(?<=[\s'\"=:(])){re.escape(source)}"
            rf"(?=($|/|\s|['\";|&()<>]))",
            target,
            rewritten,
        )
    return rewritten


def _rewrite_rli_isolation(text: str) -> str:
    return (
        text.replace(
            "chroot /jail /image",
            "proot -R /jail -w / /image",
        )
        .replace(
            'os.rename("mystery", "/tmp/mystery")',
            "pass",
        )
        .replace(
            "cd /tmp; mkdir -p /jail_orig; cp mystery /jail_orig/; "
            "chroot /jail_orig /mystery",
            "mkdir -p /jail_orig; cp /app/mystery /jail_orig/; "
            "proot -R /jail_orig -w / /mystery",
        )
        .replace(
            "chroot /jail_clean /reverse",
            "proot -R /jail_clean -w / /reverse",
        )
    )


def _find_local_pip_index(tests_dir: Path) -> str | None:
    for path in sorted(tests_dir.rglob("*")):
        if path.suffix not in {".py", ".sh"}:
            continue
        match = _LOCAL_PIP_INDEX_RE.search(path.read_text(encoding="utf-8"))
        if match:
            return match.group(1)
    return None


def _remove_runtime_apt(text: str) -> str:
    prepared = []
    for line in text.splitlines():
        if not _RUNTIME_APT_RE.search(line):
            prepared.append(line)
        elif line.lstrip().startswith(("os.system(", "subprocess.")):
            prepared.append(f"{line[: len(line) - len(line.lstrip())]}pass")
    return "\n".join(prepared) + ("\n" if text.endswith("\n") else "")


def prepare_test_assets(tests_dir: Path, test_path: Path) -> None:
    downloads = find_cached_python_downloads(tests_dir)
    repositories = find_cached_git_repositories(tests_dir)
    local_pip_index = _find_local_pip_index(tests_dir)
    for shell_test_path in tests_dir.rglob("*.sh"):
        shell_test_path.write_text(
            prepare_test_script(
                shell_test_path.read_text(encoding="utf-8"),
                local_pip_index=local_pip_index,
            ),
            encoding="utf-8",
        )
    for python_test_path in tests_dir.rglob("*.py"):
        text = rewrite_runtime_paths(python_test_path.read_text(encoding="utf-8"))
        text = _rewrite_rli_isolation(text)
        text = _remove_runtime_apt(text)
        for download in downloads:
            text = text.replace(
                download.url,
                f"file://{TEST_RUNTIME_ROOT}/downloads/{download.archive_name}",
            )
        for repository in repositories:
            text = text.replace(
                repository.url,
                f"{TEST_RUNTIME_ROOT}/git/{repository.archive_name}",
            )
        python_test_path.write_text(text, encoding="utf-8")


def prepare_test_script(
    script: str,
    *,
    local_pip_index: str | None = None,
) -> str:
    prepared: list[str] = []
    setup_added = False

    for line in _logical_lines(script):
        stripped = line.strip()
        if stripped.startswith("#!") and not setup_added:
            prepared.append(line)
            prepared.extend(_runtime_setup(local_pip_index))
            setup_added = True
            continue

        try:
            tokens = _tokens(line)
        except ValueError:
            prepared.append(line)
            continue
        if "apt-get" in tokens:
            continue
        if (
            tokens
            and tokens[0] == "curl"
            and any("astral.sh/uv/" in token for token in tokens)
        ):
            continue
        if tokens[:1] == ["source"] and any(
            ".local/bin/env" in token for token in tokens
        ):
            continue

        download = _cached_download(tokens)
        if download is not None:
            source = f'"$TEST_RUNTIME_ROOT/downloads/{download.archive_name}"'
            prepared.append(f"cp {source} {shlex.quote(download.output)}")
            continue

        if stripped == _FIX_OCAML_CLONE:
            prepared.extend(
                [
                    "rm -rf ocaml-original",
                    "mkdir -p ocaml-original",
                    (
                        'cp -a "$TEST_RUNTIME_ROOT/fix-ocaml-gc/testsuite" '
                        "ocaml-original/testsuite"
                    ),
                ]
            )
            continue

        prepared.append(line)

    if not setup_added:
        prepared = [*_runtime_setup(), *prepared]
    return rewrite_runtime_paths("\n".join(prepared) + "\n")
