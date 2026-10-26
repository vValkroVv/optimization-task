from __future__ import annotations

from terminal_bench.src.test_runtime import (
    CachedDownload,
    CachedGitRepository,
    find_cached_downloads,
    find_cached_git_repositories,
    find_cached_python_downloads,
    prepare_test_assets,
    prepare_test_script,
)


def test_prepare_test_script_uses_offline_uv_runtime() -> None:
    script = """#!/bin/bash
apt-get update
apt-get install -y curl imagemagick
curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh
source $HOME/.local/bin/env
uvx \\
  -p 3.13 \\
  -w pytest==8.4.1 \\
  pytest /tests/test_outputs.py
"""

    prepared = prepare_test_script(script)

    assert 'UV_OFFLINE="${UV_OFFLINE:-1}"' in prepared
    assert 'HF_HOME="$TEST_RUNTIME_ROOT/huggingface"' in prepared
    assert 'HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-$UV_OFFLINE}"' in prepared
    assert 'HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-$UV_OFFLINE}"' in prepared
    assert 'PIP_NO_INDEX="${PIP_NO_INDEX:-1}"' in prepared
    assert "apt-get" not in prepared
    assert "astral.sh" not in prepared
    assert ".local/bin/env" not in prepared
    assert "uvx  -p 3.13  -w pytest==8.4.1" in prepared
    assert "pytest /tmp/terminal-bench/tests/test_outputs.py" in prepared
    assert "/tests/test_outputs.py" not in prepared.replace(
        "/tmp/terminal-bench/tests/test_outputs.py", ""
    )


def test_prepare_test_script_keeps_direct_pip_install_offline() -> None:
    script = """#!/bin/bash
pip install pytest==8.4.1 requests==2.32.5
pytest /tests/test_outputs.py
"""

    prepared = prepare_test_script(script)

    assert 'PIP_FIND_LINKS="$TEST_RUNTIME_ROOT/wheels"' in prepared
    assert "pip install pytest==8.4.1 requests==2.32.5" in prepared


def test_prepare_test_script_rewrites_verifier_output_path() -> None:
    prepared = prepare_test_script("#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n")

    assert "echo 1 > /tmp/terminal-bench/logs/verifier/reward.txt" in prepared


def test_prepare_test_script_replaces_cached_download() -> None:
    script = """#!/bin/bash
curl -L -o mobile_sam.pt https://example.com/mobile_sam.pt
pytest /tests/test_outputs.py
"""

    prepared = prepare_test_script(script)

    assert "https://example.com" not in prepared
    assert 'cp "$TEST_RUNTIME_ROOT/downloads/mobile_sam.pt" mobile_sam.pt' in prepared


def test_find_cached_downloads_returns_remote_assets() -> None:
    script = """#!/bin/bash
curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh
curl -L -o mobile_sam.pt https://example.com/mobile_sam.pt
"""

    assert find_cached_downloads(script) == [
        CachedDownload(
            url="https://example.com/mobile_sam.pt",
            output="mobile_sam.pt",
        )
    ]


def test_prepare_test_script_restores_fix_ocaml_testsuite_offline() -> None:
    script = """#!/bin/bash
git clone https://github.com/sadiqj/ocaml/ ocaml-original && cd ocaml-original && git checkout tag_purposefully_broken_sweeping_changes && cd ..
rm -rf ocaml/testsuite && cp -r ocaml-original/testsuite ocaml/
"""

    prepared = prepare_test_script(script)

    assert "github.com/sadiqj/ocaml" not in prepared
    assert (
        'cp -a "$TEST_RUNTIME_ROOT/fix-ocaml-gc/testsuite" ocaml-original/testsuite'
    ) in prepared
    assert "cp -r ocaml-original/testsuite ocaml/" in prepared


def test_prepare_test_assets_caches_nested_network_dependencies(
    tmp_path,
) -> None:
    test_path = tmp_path / "test.sh"
    test_path.write_text("#!/bin/bash\npytest /tests/test_outputs.py\n")
    python_test_path = tmp_path / "test_outputs.py"
    python_test_path.write_text(
        "import os\n"
        "import urllib.request\n"
        "url = 'https://example.com/master.zip'\n"
        "urllib.request.urlretrieve(url, '/tmp/master.zip')\n"
        "git_cmd = ['git', 'clone', "
        "'https://example.com/project.git', '/tmp/project']\n"
        "apt-get update\n"
        "os.system('apt install -y --reinstall example-package')\n"
    )

    assert find_cached_python_downloads(tmp_path) == [
        CachedDownload(
            url="https://example.com/master.zip",
            output="master.zip",
        )
    ]
    assert find_cached_git_repositories(tmp_path) == [
        CachedGitRepository(url="https://example.com/project.git")
    ]

    prepare_test_assets(tmp_path, test_path)

    prepared = python_test_path.read_text()
    assert "https://example.com" not in prepared
    assert (
        "file:///var/lib/terminal-bench/test_runtime/downloads/master.zip" in prepared
    )
    assert "/var/lib/terminal-bench/test_runtime/git/project.git" in prepared
    assert "apt-get" not in prepared
    assert "apt install" not in prepared
    compile(prepared, str(python_test_path), "exec")


def test_prepare_test_assets_rewrites_rli_isolation_commands(
    tmp_path,
) -> None:
    test_path = tmp_path / "test.sh"
    test_path.write_text("#!/bin/bash\npytest /tests/test_outputs.py\n")
    python_test_path = tmp_path / "test_outputs.py"
    python_test_path.write_text(
        'os.rename("mystery", "/tmp/mystery")\n'
        'subprocess.run("chroot /jail /image", shell=True)\n'
        'subprocess.run("cd /tmp; mkdir -p /jail_orig; '
        'cp mystery /jail_orig/; chroot /jail_orig /mystery", '
        "shell=True)\n"
        'subprocess.run("chroot /jail_clean /reverse", shell=True)\n'
    )

    prepare_test_assets(tmp_path, test_path)

    prepared = python_test_path.read_text()
    assert "chroot" not in prepared
    assert "proot -R /jail -w / /image" in prepared
    assert "cp /app/mystery /jail_orig/" in prepared
    assert "proot -R /jail_orig -w / /mystery" in prepared
    assert "proot -R /jail_clean -w / /reverse" in prepared
