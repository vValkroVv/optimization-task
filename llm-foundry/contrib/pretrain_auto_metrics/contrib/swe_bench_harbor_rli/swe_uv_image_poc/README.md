# SWE-bench RLI images

Build and publish offline-capable RLI SWE environment images with `uv`
preinstalled. The image version is `1.0.1`; existing `1.0.0` environments are
not modified.

For a single task:

```text
task_id: django__django-13343
env_name: swe-bench-django-13343
new_env_version: 1.0.1
uv_version: 0.7.13
```

## 1. Build and Push Image

Login to the target registry before running this script.

```bash
cd /workspace-SR008.nfs2/zhmax/dev/swe_bench_rli_dev/pretrain-auto-metrics/contrib/swe_bench_harbor_rli
./swe_uv_image_poc/scripts/build_push_uv_image.sh django__django-13343
```

It builds:

```text
docker.sberdevices.ru/gigarli/environments/swe-bench_django_1776_django-13343:1.0.1
```

The image is a thin wrapper over:

```text
swebench/sweb.eval.x86_64.django_1776_django-13343:latest
```

## 2. Register RLI ENV

```bash
./swe_uv_image_poc/scripts/register_rli_env.py \
  django__django-13343 \
  docker.sberdevices.ru/gigarli/environments/swe-bench_django_1776_django-13343:1.0.1 \
  --digest sha256:REPLACE_WITH_IMAGE_DIGEST
```

This registers `swe-bench-django-13343` as version `1.0.1`.

## 3. Probe RLI Session

```bash
./swe_uv_image_poc/scripts/probe_rli_uv.py
```
