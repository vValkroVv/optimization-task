from __future__ import annotations

import pytest

from metric_sampler.island.s3_target import S3Credentials, S3Target


@pytest.fixture
def target() -> S3Target:
    return S3Target(
        endpoint="https://s3.example.invalid",
        bucket="metrics-bucket",
        prefix="metric-sampler/v3",
        credentials=S3Credentials(access_key="test-key", secret_key="test-secret"),
    )
