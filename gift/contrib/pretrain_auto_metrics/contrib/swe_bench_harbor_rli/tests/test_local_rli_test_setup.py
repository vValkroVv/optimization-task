from __future__ import annotations

import json

import pytest

from swe_bench_harbor_rli.src.local_rli_test_setup import (
    prepare_test_config,
    prepare_test_script,
)


_TEST_SCRIPT = """from swebench.harness.test_spec.test_spec import make_test_spec
test_spec   = make_test_spec(datum)
set +e
export PATH="/root/.local/bin:$PATH"
uv run parser.py | tee -a "$LOG_FILE"
exit_code=$?
set -e
# First we reset tracked files touched by the SWE-bench test patch.
"""


def test_prepare_test_script_rewrites_for_local_runtime() -> None:
    prepared = prepare_test_script(_TEST_SCRIPT)

    assert "make_test_spec" not in prepared
    assert "test_spec = TestSpec(" in prepared
    assert "/usr/local/share/.dev_logs/test_runtime.tar.zst" in prepared
    assert '"$runtime_dir/bin/python" parser.py' in prepared
    assert "uv run parser.py" not in prepared
    assert 'if [ "${SWEBENCH_INSTALL_ONLY:-0}" = 1 ]; then' in prepared


def test_prepare_test_config_corrects_astropy_7606_test_id() -> None:
    old = "astropy/units/tests/test_units.py::test_compose_roundtrip[]"
    new = "astropy/units/tests/test_units.py::test_compose_roundtrip[unit0]"
    config = {
        "instance_id": "astropy__astropy-7606",
        "PASS_TO_PASS": json.dumps(["other-test", old]),
    }

    prepared = prepare_test_config(config)

    assert json.loads(prepared["PASS_TO_PASS"]) == ["other-test", new]
    assert json.loads(config["PASS_TO_PASS"]) == ["other-test", old]


def test_prepare_test_config_keeps_other_tasks_unchanged() -> None:
    config = {
        "instance_id": "astropy__astropy-8707",
        "PASS_TO_PASS": '["test"]',
    }

    assert prepare_test_config(config) == config


def test_prepare_legacy_requests_starts_local_httpbin() -> None:
    script = (
        _TEST_SCRIPT
        + "\n#Then we run all the tests in the repository"
        + "\n# And we reset the test files back to the base commit.\n"
    )

    prepared = prepare_test_script(
        script,
        repo="psf/requests",
        version="2.4",
    )

    assert "HTTPBIN_URL=http://127.0.0.1:18080/" in prepared
    assert "HTTPSBIN_URL=https://127.0.0.1:18080/" in prepared
    assert 'listener.bind(("127.0.0.1", 18080))' in prepared
    assert "--bind 127.0.0.1:18081" in prepared
    assert "--bind 127.0.0.1:18443" in prepared
    assert "https://127.0.0.1:18080/redirect/5" in prepared
    assert "http://localhost:18080/" in prepared
    assert (
        "s#http://fooobarbangbazbing.httpbin.org#http://127.0.0.1:1#g"
        in prepared
    )
    assert "s#http://httpbin.org:1#http://127.0.0.1:1#g" in prepared
    assert (
        "s#https://httpbin.org/status/301#"
        "https://127.0.0.1:18080/status/301#g" in prepared
    )
    assert "requests.certs.where()" in prepared
    assert 'REQUESTS_CA_BUNDLE="$requests_ca_bundle"' in prepared
    assert 'cp -p "$requests_ca_backup" "$requests_ca_bundle"' in prepared
    assert '"$httpbin_runtime_dir/bin/gunicorn"' in prepared
    assert "cleanup_local_httpbin" in prepared


def test_prepare_legacy_requests_before_24_keeps_external_urls() -> None:
    script = (
        _TEST_SCRIPT
        + "\n#Then we run all the tests in the repository"
        + "\n# And we reset the test files back to the base commit.\n"
    )

    prepared = prepare_test_script(
        script,
        repo="psf/requests",
        version="2.3",
    )

    assert "httpbin.org/redirect/5" not in prepared
    assert "www.google.co.uk" not in prepared


def test_prepare_new_requests_does_not_start_local_httpbin() -> None:
    prepared = prepare_test_script(
        _TEST_SCRIPT,
        repo="psf/requests",
        version="2.9",
    )

    assert "HTTPBIN_URL=" not in prepared


def test_prepare_sklearn_limits_native_threads() -> None:
    script = _TEST_SCRIPT + "\n#Then we run all the tests in the repository\n"

    prepared = prepare_test_script(
        script,
        repo="scikit-learn/scikit-learn",
        version="0.21",
    )

    assert "export OMP_NUM_THREADS=1" in prepared
    assert "export OPENBLAS_NUM_THREADS=1" in prepared
    assert "export MKL_NUM_THREADS=1" in prepared


def test_prepare_django_uses_clean_tmpdir() -> None:
    script = _TEST_SCRIPT + "\n#Then we run all the tests in the repository\n"

    prepared = prepare_test_script(script, repo="django/django", version="4.2")

    assert "django_tmpdir=/tmp/t" in prepared
    assert 'chmod 1777 "$django_tmpdir"' in prepared
    assert 'export TMPDIR="$django_tmpdir"' in prepared


@pytest.mark.parametrize("version", ["3.2", "3.3", "3.4"])
def test_prepare_sphinx_starts_local_linkcheck(version: str) -> None:
    script = (
        _TEST_SCRIPT
        + "\n#Then we run all the tests in the repository"
        + "\n# And we reset the test files back to the base commit.\n"
    )

    prepared = prepare_test_script(
        script,
        repo="sphinx-doc/sphinx",
        version=version,
    )

    assert 'ThreadingHTTPServer(("127.0.0.1", 18082)' in prepared
    assert "http://127.0.0.1:18082/w3" in prepared
    assert "http://127.0.0.1:18082/google" in prepared
    assert "http://127.0.0.1:18082/sphinx/intro.html" in prepared
    assert "cleanup_local_linkcheck" in prepared


def test_prepare_test_script_skips_existing_locale() -> None:
    script = (
        _TEST_SCRIPT
        + "\nsed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen\n"
    )

    prepared = prepare_test_script(script)

    assert "if ! locale -a | grep -Fxiq en_US.utf8; then" in prepared
    assert "&& locale-gen" not in prepared


@pytest.mark.parametrize(
    ("repo", "package", "command"),
    [
        (
            "astropy/astropy",
            "astropy",
            "python -m pip install -e .[test] --verbose",
        ),
        (
            "matplotlib/matplotlib",
            "matplotlib",
            "python -m pip install -e .",
        ),
    ],
)
def test_prepare_test_script_guards_redundant_editable_install(
    repo: str,
    package: str,
    command: str,
) -> None:
    script = _TEST_SCRIPT.replace(
        "# First we reset tracked files touched by the SWE-bench test patch.",
        f"{command}\n# First we reset tracked files touched by the SWE-bench test patch.",
    )

    prepared = prepare_test_script(script, repo=repo)

    assert 'SWEBENCH_INSTALL_ONLY:-0' in prepared
    assert f"python -m pip show {package}" in prepared
    assert "git diff --name-only --no-renames HEAD" in prepared
    assert command in prepared


@pytest.mark.parametrize(
    "block",
    [
        "from swebench.harness.test_spec.test_spec import make_test_spec",
        "test_spec   = make_test_spec(datum)",
        "uv run parser.py | tee -a \"$LOG_FILE\"",
        "# First we reset tracked files touched by the SWE-bench test patch.",
    ],
)
def test_prepare_test_script_rejects_missing_block(block: str) -> None:
    with pytest.raises(ValueError, match="found 0"):
        prepare_test_script(_TEST_SCRIPT.replace(block, ""))
