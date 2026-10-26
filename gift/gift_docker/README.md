# gift_docker

Build environment for the torchtitan training image: `Dockerfile` plus the
locked dependency set (`pyproject.toml`, `uv.lock`). CI builds and pushes the
image with the `build:docker_base` job.

## Python dependencies

The image dependency set supports CPython 3.12 on Linux x86_64. The private
nightly mirror currently provides only CPython 3.12 wheels for torch,
torchvision, torchcomms, and triton. Docker base images and unit CI runners
must provide `python3.12`; installation selects it explicitly and fails if it
is unavailable. This restriction applies to `gift_docker`, not to the root
package's Python support.

Docker pins uv 0.12.10. Unit/build CI jobs use the uv binary bundled in the
CI image and print its version before use. The bundled version must support
the index's `authenticate = "always"` setting.
Authentication uses `UV_INDEX_PYTORCH_NIGHTLY_CU130_USERNAME` and
`UV_INDEX_PYTORCH_NIGHTLY_CU130_PASSWORD`; no keyring provider is needed.
Docker receives the password through the `ci_job_token` BuildKit secret.

To refresh the lockfile with those credentials set:

```bash
uv lock --project gift_docker --python python3.12 --prerelease allow
```

Validate installation on a Linux x86_64 host with Python 3.12:

```bash
uv sync --project gift_docker --python python3.12 --frozen --all-extras --no-install-project --prerelease allow
```

## images.json

`images.json` is the single source of truth for the pinned torchtitan images.
It maps each cluster to the image reference to run there: registry references
for SR clusters, `.sqsh` paths for o-clusters.

Consumers:

- `contrib/job_submitter` GIFT jobs resolve `base_image` from this file for
  the detected cluster;
- `scripts/ci/submit_integration_job.py` (CI integration suites) defaults to
  the image pinned here;
- `gift_scripts/submit_infinite_job.py`.

After building and uploading a new image (and converting it to `.sqsh` on the
o-clusters), update every entry in `images.json`. Do not pin torchtitan image
references anywhere else.
