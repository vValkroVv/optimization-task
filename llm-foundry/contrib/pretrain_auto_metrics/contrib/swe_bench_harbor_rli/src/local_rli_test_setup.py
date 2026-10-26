from __future__ import annotations

import json


_LEGACY_REQUESTS_VERSIONS = ("2.0", "2.3", "2.4")
_ASTROPY_7606_OLD_TEST = (
    "astropy/units/tests/test_units.py::test_compose_roundtrip[]"
)
_ASTROPY_7606_NEW_TEST = (
    "astropy/units/tests/test_units.py::test_compose_roundtrip[unit0]"
)
_TEST_RUN_BOUNDARY = "#Then we run all the tests in the repository"
_TEST_RESET_BOUNDARY = "# And we reset the test files back to the base commit."
_LOCALE_GEN_COMMAND = "sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen"
_LOCALE_GEN_GUARD = """if ! locale -a | grep -Fxiq en_US.utf8; then
  sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen
  locale-gen
fi"""

_EDITABLE_INSTALLS = {
    "astropy/astropy": (
        "astropy",
        "python -m pip install -e .[test] --verbose",
    ),
    "matplotlib/matplotlib": (
        "matplotlib",
        "python -m pip install -e .",
    ),
}

_BUILD_INPUT_PATTERN = (
    r"(^|/)(CMakeLists\.txt|MANIFEST\.in|meson\.build|meson_options\.txt|"
    r"pyproject\.toml|setup\.cfg|setup\.py|setup_package\.py|setupext\.py)$|"
    r"\.(c|cc|cpp|cxx|f|f03|f77|f90|f95|h|hh|hpp|hxx|pxd|pxi|pyx|rs)$"
)

_LOCAL_HTTPBIN_START = """httpbin_runtime_archive=/usr/local/share/.dev_logs/test_runtime.tar.zst
httpbin_runtime_dir=/tmp/test_runtime
httpbin_log=/tmp/httpbin.log
httpbin_proxy=/tmp/httpbin-protocol-proxy.py
rm -rf "$httpbin_runtime_dir"
tar --zstd -xf "$httpbin_runtime_archive" -C /tmp

export HTTPBIN_URL=http://127.0.0.1:18080/
export HTTPSBIN_URL=https://127.0.0.1:18080/
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"

requests_ca_bundle="$(python -c 'import requests.certs; print(requests.certs.where())')"
requests_ca_backup=/tmp/requests-ca-bundle.pem
cp -p "$requests_ca_bundle" "$requests_ca_backup"
cat "$httpbin_runtime_dir/httpbin-cert.pem" >>"$requests_ca_bundle"
export REQUESTS_CA_BUNDLE="$requests_ca_bundle"

httpbin_http_pid=
httpbin_https_pid=
httpbin_proxy_pid=
cleanup_local_httpbin() {
  if [ -n "$httpbin_proxy_pid" ]; then
    kill "$httpbin_proxy_pid" >/dev/null 2>&1 || true
    wait "$httpbin_proxy_pid" >/dev/null 2>&1 || true
  fi
  if [ -n "$httpbin_http_pid" ]; then
    kill "$httpbin_http_pid" >/dev/null 2>&1 || true
    wait "$httpbin_http_pid" >/dev/null 2>&1 || true
  fi
  if [ -n "$httpbin_https_pid" ]; then
    kill "$httpbin_https_pid" >/dev/null 2>&1 || true
    wait "$httpbin_https_pid" >/dev/null 2>&1 || true
  fi
  cp -p "$requests_ca_backup" "$requests_ca_bundle"
  rm -f "$requests_ca_backup"
  rm -f "$httpbin_proxy"
  rm -rf "$httpbin_runtime_dir"
}
trap cleanup_local_httpbin EXIT

"$httpbin_runtime_dir/bin/gunicorn" \\
  --bind 127.0.0.1:18081 \\
  --workers 1 \\
  httpbin:app >"$httpbin_log" 2>&1 &
httpbin_http_pid=$!

"$httpbin_runtime_dir/bin/gunicorn" \\
  --bind 127.0.0.1:18443 \\
  --workers 1 \\
  --certfile "$httpbin_runtime_dir/httpbin-cert.pem" \\
  --keyfile "$httpbin_runtime_dir/httpbin-key.pem" \\
  httpbin:app >>"$httpbin_log" 2>&1 &
httpbin_https_pid=$!

cat >"$httpbin_proxy" <<'PY'
import select
import socket
import threading


def proxy(client):
    try:
        first_byte = client.recv(1, socket.MSG_PEEK)
        port = 18443 if first_byte == b"\\x16" else 18081
        with client, socket.create_connection(("127.0.0.1", port)) as backend:
            peers = {client: backend, backend: client}
            while True:
                readable, _, _ = select.select(peers, [], [])
                for source in readable:
                    data = source.recv(65536)
                    if not data:
                        return
                    peers[source].sendall(data)
    except OSError:
        client.close()


with socket.socket() as listener:
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 18080))
    listener.listen()
    while True:
        connection, _ = listener.accept()
        threading.Thread(target=proxy, args=(connection,), daemon=True).start()
PY
"$httpbin_runtime_dir/bin/python" "$httpbin_proxy" >>"$httpbin_log" 2>&1 &
httpbin_proxy_pid=$!

httpbin_ready=0
for _ in $(seq 1 30); do
  if curl --noproxy '*' --fail --silent --max-time 2 \\
      "${HTTPBIN_URL}get" >/dev/null && \\
    curl --noproxy '*' --fail --silent --max-time 2 \\
      --cacert "$httpbin_runtime_dir/httpbin-cert.pem" \\
      "${HTTPSBIN_URL}get" >/dev/null; then
    httpbin_ready=1
    break
  fi
  sleep 1
done
if [ "$httpbin_ready" -ne 1 ]; then
  cat "$httpbin_log" >&2
  exit 1
fi

sed -i \
  -e 's#https://httpbin.org/status/301#https://127.0.0.1:18080/status/301#g' \
  -e 's#https://httpbin.org/delay/10#https://127.0.0.1:18080/delay/10#g' \
  test_requests.py

#Then we run all the tests in the repository"""

_REQUESTS_24_LOCAL_URLS = """sed -i \\
  -e 's#https://httpbin.org/redirect/5#https://127.0.0.1:18080/redirect/5#g' \\
  -e 's#http://www.google.co.uk#http://localhost:18080/#g' \\
  -e 's#http://fooobarbangbazbing.httpbin.org#http://127.0.0.1:1#g' \\
  -e 's#http://httpbin.org:1#http://127.0.0.1:1#g' \\
  test_requests.py

#Then we run all the tests in the repository"""

_SKLEARN_TEST_ENV = """export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

#Then we run all the tests in the repository"""

_MATPLOTLIB_TEST_ENV = """export GS_OPTIONS="${GS_OPTIONS:+$GS_OPTIONS }-dNOSAFER"

#Then we run all the tests in the repository"""

_DJANGO_TEST_ENV = """django_tmpdir=/tmp/t
rm -rf "$django_tmpdir"
mkdir -p "$django_tmpdir"
chmod 1777 "$django_tmpdir"
export TMPDIR="$django_tmpdir"

#Then we run all the tests in the repository"""

_DJANGO_16145_TEST_ENV = """django_tmpdir=/t
rm -rf "$django_tmpdir"
mkdir -p "$django_tmpdir"
chmod 1777 "$django_tmpdir"
export TMPDIR="$django_tmpdir"

#Then we run all the tests in the repository"""

_LOCAL_LINKCHECK_START = """linkcheck_dir=/tmp/swebench-linkcheck
linkcheck_server="$linkcheck_dir/server.py"
linkcheck_log="$linkcheck_dir/server.log"
rm -rf "$linkcheck_dir"
mkdir -p "$linkcheck_dir"
cat >"$linkcheck_server" <<'PY'
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    def respond(self, include_body):
        path = self.path.split("?", 1)[0]
        missing = path.endswith(".png")
        body = (
            b'<html><body><a id="defaulting"></a></body></html>'
            if path.startswith("/w3/")
            else b"<html><body>offline linkcheck target</body></html>"
        )
        self.send_response(404 if missing else 200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if include_body:
            self.wfile.write(body)

    def do_GET(self):
        self.respond(True)

    def do_HEAD(self):
        self.respond(False)

    def log_message(self, format, *args):
        pass


ThreadingHTTPServer(("127.0.0.1", 18082), Handler).serve_forever()
PY
python "$linkcheck_server" >"$linkcheck_log" 2>&1 &
linkcheck_pid=$!
cleanup_local_linkcheck() {
  kill "$linkcheck_pid" >/dev/null 2>&1 || true
  wait "$linkcheck_pid" >/dev/null 2>&1 || true
  rm -rf "$linkcheck_dir"
}
trap cleanup_local_linkcheck EXIT

for _ in $(seq 1 30); do
  if curl --noproxy '*' --fail --silent --max-time 2 \
      http://127.0.0.1:18082/ >/dev/null; then
    break
  fi
  sleep 1
done

sed -i \
  -e 's#https://www.w3.org#http://127.0.0.1:18082/w3#g' \
  -e 's#https://www.google.com#http://127.0.0.1:18082/google#g' \
  -e 's#http://www.sphinx-doc.org/en/1.7/intro.html#http://127.0.0.1:18082/sphinx/intro.html#g' \
  tests/roots/test-linkcheck/links.txt tests/test_build_linkcheck.py

#Then we run all the tests in the repository"""

_LOCAL_LINKCHECK_STOP = """cleanup_local_linkcheck
trap - EXIT

# And we reset the test files back to the base commit."""

_LOCAL_IMAGE_SERVER_START = """image_server_dir=/tmp/swebench-image-server
image_server_log="$image_server_dir/server.log"
rm -rf "$image_server_dir"
mkdir -p "$image_server_dir"
cp "$local_image_source" "$image_server_dir/$local_image_name"

export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"

python -m http.server 18083 \
  --bind 127.0.0.1 \
  --directory "$image_server_dir" >"$image_server_log" 2>&1 &
image_server_pid=$!
cleanup_local_image_server() {
  kill "$image_server_pid" >/dev/null 2>&1 || true
  wait "$image_server_pid" >/dev/null 2>&1 || true
  rm -rf "$image_server_dir"
}
trap cleanup_local_image_server EXIT

image_server_ready=0
for _ in $(seq 1 30); do
  if curl --noproxy '*' --fail --silent --max-time 2 \
      "http://127.0.0.1:18083/$local_image_name" >/dev/null; then
    image_server_ready=1
    break
  fi
  sleep 1
done
if [ "$image_server_ready" -ne 1 ]; then
  cat "$image_server_log" >&2
  exit 1
fi
"""

_MATPLOTLIB_34_LOCAL_IMAGE_START = (
    """local_image_source=lib/matplotlib/tests/baseline_images/pngsuite/basn3p04.png
local_image_name=logo2.png
"""
    + _LOCAL_IMAGE_SERVER_START
    + """
sed -i \
  's#https://matplotlib.org/1.5.0/_static/logo2.png#http://127.0.0.1:18083/logo2.png#g' \
  lib/matplotlib/tests/test_image.py

#Then we run all the tests in the repository"""
)

_SPHINX_50_LOCAL_IMAGE_START = (
    """local_image_source=tests/roots/test-images/img.png
local_image_name=python-logo.png
"""
    + _LOCAL_IMAGE_SERVER_START
    + """
sed -i \
  -e 's#https://www.python.org/static/img/python-logo.png#http://127.0.0.1:18083/python-logo.png#g' \
  -e 's#https://www.google.com/NOT_EXIST.PNG#http://127.0.0.1:18083/NOT_EXIST.PNG#g' \
  tests/roots/test-images/index.rst tests/test_build_latex.py

#Then we run all the tests in the repository"""
)

_LOCAL_IMAGE_SERVER_STOP = """cleanup_local_image_server
trap - EXIT

# And we reset the test files back to the base commit."""

_LOCAL_HTTPBIN_STOP = """cleanup_local_httpbin
trap - EXIT

# And we reset the test files back to the base commit."""

_INSTALL_ONLY_BOUNDARY = (
    "# First we reset tracked files touched by the SWE-bench test patch."
)

_INSTALL_ONLY_GUARD = """if [ "${SWEBENCH_INSTALL_ONLY:-0}" = 1 ]; then
  exit 0
fi

# First we reset tracked files touched by the SWE-bench test patch."""

_TEST_SPEC_FROM_CONFIG = """def _test_list(key):
    value = datum.get(key, [])
    return json.loads(value) if isinstance(value, str) else value

test_spec = TestSpec(
    instance_id=datum[KEY_INSTANCE_ID],
    repo=datum["repo"],
    version=datum["version"],
    repo_script_list=[],
    eval_script_list=[],
    env_script_list=[],
    arch="x86_64",
    FAIL_TO_PASS=_test_list(FAIL_TO_PASS),
    PASS_TO_PASS=_test_list(PASS_TO_PASS),
    language="",
    docker_specs={},
    namespace=None,
)"""

_RUNTIME_FROM_ARCHIVE = """runtime_archive=/usr/local/share/.dev_logs/test_runtime.tar.zst
runtime_dir=/tmp/test_runtime
rm -rf "$runtime_dir"
if [ ! -r "$runtime_archive" ]; then
  echo "test runtime archive not found: $runtime_archive" >&2
  exit 1
fi
if ! tar --zstd -xf "$runtime_archive" -C /tmp; then
  echo "failed to extract test runtime: $runtime_archive" >&2
  exit 1
fi

set +e
"$runtime_dir/bin/python" parser.py | tee -a "$LOG_FILE"
exit_code=$?
rm -rf "$runtime_dir"
set -e"""

_TEST_SCRIPT_REPLACEMENTS = (
    (
        "install-only boundary",
        _INSTALL_ONLY_BOUNDARY,
        _INSTALL_ONLY_GUARD,
    ),
    (
        "make_test_spec import",
        "from swebench.harness.test_spec.test_spec import make_test_spec",
        "from swebench.harness.test_spec.test_spec import TestSpec",
    ),
    (
        "make_test_spec call",
        "test_spec   = make_test_spec(datum)",
        _TEST_SPEC_FROM_CONFIG,
    ),
    (
        "uv parser execution",
        """set +e
export PATH="/root/.local/bin:$PATH"
uv run parser.py | tee -a "$LOG_FILE"
exit_code=$?
set -e""",
        _RUNTIME_FROM_ARCHIVE,
    ),
)


def _guard_editable_install(package: str, command: str) -> str:
    return f"""build_input_changes="$(
  {{
    git diff --name-only --no-renames HEAD
    git ls-files --others --exclude-standard
  }} | grep -E '{_BUILD_INPUT_PATTERN}' || true
)"
editable_location="$(
  python -m pip show {package} 2>/dev/null |
    sed -n 's/^Editable project location: //p' || true
)"
if [ "${{SWEBENCH_INSTALL_ONLY:-0}}" != 1 ] \\
    && [ "$editable_location" = /testbed ] \\
    && [ -z "$build_input_changes" ]; then
  echo "Skipping {package} editable reinstall: build inputs are unchanged"
else
  {command}
fi"""


def prepare_test_config(config: dict[str, object]) -> dict[str, object]:
    if config.get("instance_id") != "astropy__astropy-7606":
        return config

    value = config.get("PASS_TO_PASS")
    serialized = isinstance(value, str)
    tests = json.loads(value) if serialized else value
    if not isinstance(tests, list) or not all(
        isinstance(test, str) for test in tests
    ):
        raise ValueError("Expected astropy-7606 PASS_TO_PASS to be a test list")
    if tests.count(_ASTROPY_7606_OLD_TEST) != 1:
        raise ValueError("Expected stale astropy-7606 PASS_TO_PASS entry")

    corrected = [
        _ASTROPY_7606_NEW_TEST if test == _ASTROPY_7606_OLD_TEST else test
        for test in tests
    ]
    prepared = dict(config)
    prepared["PASS_TO_PASS"] = (
        json.dumps(corrected) if serialized else corrected
    )
    return prepared


def prepare_test_script(
    text: str,
    *,
    repo: str = "",
    version: str = "",
    instance_id: str = "",
) -> str:
    replacements = _TEST_SCRIPT_REPLACEMENTS
    if _LOCALE_GEN_COMMAND in text:
        replacements += (
            (
                "locale generation",
                _LOCALE_GEN_COMMAND,
                _LOCALE_GEN_GUARD,
            ),
        )
    editable_install = _EDITABLE_INSTALLS.get(repo)
    if editable_install:
        package, command = editable_install
        replacements += (
            (
                f"{repo} editable install",
                command,
                _guard_editable_install(package, command),
            ),
        )
    if repo == "psf/requests" and version in _LEGACY_REQUESTS_VERSIONS:
        httpbin_start = _LOCAL_HTTPBIN_START
        if version == "2.4":
            httpbin_start = httpbin_start.replace(
                _TEST_RUN_BOUNDARY,
                _REQUESTS_24_LOCAL_URLS,
            )
        replacements += (
            (
                "test execution boundary",
                _TEST_RUN_BOUNDARY,
                httpbin_start,
            ),
            (
                "test reset boundary",
                _TEST_RESET_BOUNDARY,
                _LOCAL_HTTPBIN_STOP,
            ),
        )
    if repo == "scikit-learn/scikit-learn":
        replacements += (
            (
                "test execution boundary",
                _TEST_RUN_BOUNDARY,
                _SKLEARN_TEST_ENV,
            ),
        )
    if repo == "matplotlib/matplotlib" and version == "3.1":
        replacements += (
            (
                "test execution boundary",
                _TEST_RUN_BOUNDARY,
                _MATPLOTLIB_TEST_ENV,
            ),
        )
    if repo == "matplotlib/matplotlib" and version == "3.4":
        replacements += (
            (
                "test execution boundary",
                _TEST_RUN_BOUNDARY,
                _MATPLOTLIB_34_LOCAL_IMAGE_START,
            ),
            (
                "test reset boundary",
                _TEST_RESET_BOUNDARY,
                _LOCAL_IMAGE_SERVER_STOP,
            ),
        )
    if repo == "django/django":
        if instance_id == "django__django-16145":
            replacements += (
                (
                    "test execution boundary",
                    _TEST_RUN_BOUNDARY,
                    _DJANGO_16145_TEST_ENV,
                ),
            )
        else:
            replacements += (
                (
                    "test execution boundary",
                    _TEST_RUN_BOUNDARY,
                    _DJANGO_TEST_ENV,
                ),
            )
    if repo == "sphinx-doc/sphinx" and version in ("3.2", "3.3", "3.4"):
        replacements += (
            (
                "test execution boundary",
                _TEST_RUN_BOUNDARY,
                _LOCAL_LINKCHECK_START,
            ),
            (
                "test reset boundary",
                _TEST_RESET_BOUNDARY,
                _LOCAL_LINKCHECK_STOP,
            ),
        )
    if repo == "sphinx-doc/sphinx" and version == "5.0":
        replacements += (
            (
                "test execution boundary",
                _TEST_RUN_BOUNDARY,
                _SPHINX_50_LOCAL_IMAGE_START,
            ),
            (
                "test reset boundary",
                _TEST_RESET_BOUNDARY,
                _LOCAL_IMAGE_SERVER_STOP,
            ),
        )

    for name, old, new in replacements:
        count = text.count(old)
        if count != 1:
            raise ValueError(
                f"Expected exactly one SWE-bench {name} block, found {count}"
            )
        text = text.replace(old, new)
    return text
