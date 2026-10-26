from __future__ import annotations

from swe_bench_harbor_rli.swe_uv_image_poc.scripts.pipeline import render_child_pipeline


def test_pipeline_contains_independent_jobs_and_summary() -> None:
    records = [
        {
            "task_id": "django__django-13343",
            "image": "image:django",
            "source_image": "source",
            "assets": "/tmp",
        },
        {
            "task_id": "sphinx-doc__sphinx-9258",
            "image": "image:sphinx",
            "source_image": "source",
            "assets": "/tmp",
        },
    ]

    pipeline = render_child_pipeline(
        records,
        "pilot",
        pilot_tasks=[record["task_id"] for record in records],
    )

    assert '"build-django-django-13343":' in pipeline
    assert '"build-sphinx-doc-sphinx-9258":' in pipeline
    assert "resource_group: swe-image-build-0" in pipeline
    assert "resource_group: swe-image-build-1" in pipeline
    assert "  allow_failure: true" in pipeline
    assert '"summary:batch":' in pipeline
    summary = pipeline.split('"summary:batch":', 1)[1]
    assert "allow_failure: true" not in summary
    assert "tags: [advanced]" in summary
    assert "needs:" not in summary
    assert '"register:batch":' not in pipeline
    assert "  - register" not in pipeline
    assert "docker login docker.sberdevices.ru" in pipeline
    assert "docker login docker.io" in pipeline
    assert "docker logout docker.io" in pipeline
    assert 'build_push_uv_image.sh "$TASK_ID"' in pipeline
    assert '"$TASK_ID" "$SWE_BENCH_DATASET_ROOT"' not in pipeline
    assert "swe_uv_image_poc/outputs/validation/django__django-13343/" in pipeline
    assert "batch_summary.py swe_uv_image_poc/outputs/validation" in pipeline
    assert "when: always" in pipeline
