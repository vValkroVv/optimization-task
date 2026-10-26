from __future__ import annotations

from pathlib import Path

import pytest

from swe_bench_harbor_rli.swe_uv_image_poc.scripts import image_builder
from swe_bench_harbor_rli.swe_uv_image_poc.scripts import manifest


def test_source_image_and_target_name() -> None:
    source = "swebench/sweb.eval.x86_64.django_1776_django-13343:latest"

    assert image_builder.source_name(source) == "django_1776_django-13343"
    assert image_builder.target_image(source) == (
        "docker.sberdevices.ru/gigarli/environments/"
        "swe-bench_django_1776_django-13343:1.0.1"
    )


def test_render_dockerfile_keeps_oracle_assets_out_of_image_layers() -> None:
    dockerfile = image_builder.render_dockerfile(
        "source:latest",
        "django__django-13343",
        {"base_commit": "abc", "environment_setup_commit": "def"},
    )

    assert dockerfile.startswith("# syntax=docker/dockerfile:1\n")
    assert "RUN --mount=type=bind,source=task,target=/opt/swebench-task" in dockerfile
    assert "COPY task/" not in dockerfile
    assert "mkdir -p /tests /logs/verifier /tmp/t" in dockerfile
    assert "TMPDIR=/tmp/t bash /opt/swebench-task/tests/test.sh" in dockerfile
    assert (
        "ENV PIP_FIND_LINKS=/usr/local/share/.dev_logs/build_wheels" in dockerfile
    )
    assert "ENV ENVIRONMENT_SETUP_COMMIT=def" in dockerfile
    assert 'initial_commit="$(git -C /testbed rev-parse HEAD)"' in dockerfile
    assert "cache_build_requirements.py" in dockerfile
    assert 'git -C /testbed reset --hard "$initial_commit"' in dockerfile
    assert "git -C /testbed apply --binary /tmp/testbed-initial.patch" in dockerfile
    assert "cmp /tmp/testbed-initial.patch /tmp/testbed-restored.patch" in dockerfile
    assert "ENV NO_COLOR=1" in dockerfile
    assert "/opt/miniconda3/bin/python -m pip install" in dockerfile
    assert 'python_bin="$(command -v python3' not in dockerfile
    assert "(cd /tmp && /tmp/test_runtime/bin/python" in dockerfile
    assert "git -C /testbed ls-files --others --ignored" in dockerfile
    assert "tar -C /testbed -xf /tmp/testbed-ignored.tar" in dockerfile
    assert "git -C /testbed clean -fdx" in dockerfile
    assert "/tmp/testbed-ignored.files /tmp/testbed-ignored.tar" in dockerfile


def test_render_dockerfile_pins_setuptools_for_astropy_31() -> None:
    dockerfile = image_builder.render_dockerfile(
        "source:latest",
        "astropy__astropy-8872",
        {
            "repo": "astropy/astropy",
            "version": "3.1",
            "base_commit": "abc",
        },
    )

    assert (
        "ENV PIP_CONSTRAINT=/usr/local/share/.dev_logs/testbed-constraints.txt"
        in dockerfile
    )
    assert "printf '%s\\n' \"setuptools==59.8.0\"" in dockerfile
    assert (
        "/opt/miniconda3/envs/testbed/bin/python -m pip install \\\n"
        '      --no-cache-dir "setuptools==59.8.0"'
        in dockerfile
    )


def test_render_dockerfile_pins_pytest_for_astropy_8707() -> None:
    dockerfile = image_builder.render_dockerfile(
        "source:latest",
        "astropy__astropy-8707",
        {
            "instance_id": "astropy__astropy-8707",
            "repo": "astropy/astropy",
            "version": "3.1",
            "base_commit": "abc",
        },
    )

    assert '"setuptools==59.8.0" "pytest==7.1.3"' in dockerfile


def test_render_dockerfile_uses_sqlite_compat_library_for_django_10097() -> None:
    dockerfile = image_builder.render_dockerfile(
        "source:latest",
        "django__django-10097",
        {
            "repo": "django/django",
            "version": "2.2",
            "base_commit": "abc",
        },
    )

    assert "sqlite-3.25.3-h67949de_1000.tar.bz2" in dockerfile
    assert (
        "d140c8e08258e55e1774d8be4167650c5609d39f5ce2876447672cf3bc1c7729"
        in dockerfile
    )
    assert "lib/libsqlite3.so.0.8.6" in dockerfile
    assert "sqlite3.sqlite_version" in dockerfile
    assert "conda install" not in dockerfile

    other_django_22 = image_builder.render_dockerfile(
        "source:latest",
        "django__django-11001",
        {
            "repo": "django/django",
            "version": "2.2",
            "base_commit": "abc",
        },
    )
    assert "sqlite-3.25.3-h67949de_1000.tar.bz2" not in other_django_22


def test_render_dockerfile_adds_httpbin_for_legacy_requests() -> None:
    dockerfile = image_builder.render_dockerfile(
        "source:latest",
        "psf__requests-2317",
        {
            "repo": "psf/requests",
            "version": "2.4",
            "base_commit": "abc",
        },
    )

    assert '"httpbin==0.10.2"' in dockerfile
    assert '"gunicorn==23.0.0"' in dockerfile
    assert "import datasets, fastcore, swebench, gunicorn, httpbin" in dockerfile
    assert '-addext "subjectAltName=IP:127.0.0.1"' in dockerfile


def test_unknown_task_id_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="unknown task_id"):
        image_builder.find_task_assets("missing__task-1", tmp_path)


def _record(task_id: str) -> dict[str, str]:
    return {
        "task_id": task_id,
        "source_image": "image",
        "image": "target",
        "assets": "/tmp",
    }


def test_pilot_selection_allows_multiple_tasks_from_same_repository(
    tmp_path: Path,
) -> None:
    pilot = [
        "sphinx-doc__sphinx-8269",
        "sphinx-doc__sphinx-8475",
    ]
    records = [_record(task_id) for task_id in pilot]
    pilot_file = tmp_path / "pilot.txt"
    pilot_file.write_text("\n".join(pilot) + "\n", encoding="utf-8")

    selected = manifest.select_tasks(
        records,
        "pilot",
        pilot_tasks=manifest.read_pilot_tasks(pilot_file),
    )

    assert [record["task_id"] for record in selected] == pilot


def test_batch_selection_excludes_pilot_and_returns_five() -> None:
    pilot = ["repo0__repo-0"]
    records = [_record(f"repo{i}__repo-{i}") for i in range(7)]

    selected = manifest.select_tasks(records, "batch", batch_index=0, pilot_tasks=pilot)

    assert [record["task_id"] for record in selected] == [
        "repo1__repo-1",
        "repo2__repo-2",
        "repo3__repo-3",
        "repo4__repo-4",
        "repo5__repo-5",
    ]


def test_default_batch_selection_excludes_built_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [_record(f"repo{i}__repo-{i}") for i in range(7)]
    monkeypatch.setattr(manifest, "read_pilot_tasks", lambda: ["repo0__repo-0"])
    monkeypatch.setattr(manifest, "read_built_tasks", lambda: ["repo1__repo-1"])

    selected = manifest.select_tasks(records, "batch", batch_index=0)

    assert [record["task_id"] for record in selected] == [
        "repo2__repo-2",
        "repo3__repo-3",
        "repo4__repo-4",
        "repo5__repo-5",
        "repo6__repo-6",
    ]


def test_all_selection_excludes_pilot_and_built_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [_record(f"repo{i}__repo-{i}") for i in range(4)]
    monkeypatch.setattr(manifest, "read_pilot_tasks", lambda: ["repo0__repo-0"])
    monkeypatch.setattr(manifest, "read_built_tasks", lambda: ["repo1__repo-1"])

    selected = manifest.select_tasks(records, "all")

    assert [record["task_id"] for record in selected] == [
        "repo2__repo-2",
        "repo3__repo-3",
    ]


@pytest.mark.parametrize("value", [-1, None])
def test_invalid_batch_index_is_rejected(value: int | None) -> None:
    with pytest.raises(ValueError, match="SWE_BATCH_INDEX"):
        manifest.select_tasks([_record("repo__repo-1")], "batch", value, pilot_tasks=[])
