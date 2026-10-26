import pytest

from job_submitter.jobs.transfers.rclone import (
    ISLAND_RCLONE_CONFIG,
    build_copy_script,
    resolve_rclone_config,
)

pytestmark = pytest.mark.unit

EXPECTED_SCRIPT_WITH_CONFIG = r"""SOURCE='/mnt/data/model dir/artifacts.tar'
DESTINATION=papayaeye-papaya:transfer2cloud/o1/artifacts.tar

rclone copyto -L "$SOURCE" "$DESTINATION" \
    --config /mnt/nfs/rclone.conf \
    --transfers 8 \
    --checkers 16 \
    --multi-thread-streams 8 \
    --s3-upload-concurrency 16 \
    --buffer-size 128Mi \
    --s3-chunk-size 128Mi \
    --s3-upload-cutoff 200M \
    --retries 3 \
    --low-level-retries 10 \
    --size-only \
    --stats 30s \
    --stats-one-line \
    --log-level INFO \
    --no-check-certificate"""


@pytest.mark.parametrize("region", ["o1", "o2", "o5"])
def test_resolve_rclone_config_uses_island_config(
    monkeypatch: pytest.MonkeyPatch, region: str
) -> None:
    monkeypatch.delenv("RCLONE_CONFIG", raising=False)

    assert resolve_rclone_config(region) == ISLAND_RCLONE_CONFIG


def test_resolve_rclone_config_is_empty_for_unknown_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RCLONE_CONFIG", raising=False)

    assert resolve_rclone_config("SR004") == ""


@pytest.mark.parametrize("region", ["o1", "SR004"])
def test_resolve_rclone_config_prefers_environment_override(
    monkeypatch: pytest.MonkeyPatch, region: str
) -> None:
    monkeypatch.setenv("RCLONE_CONFIG", "/custom/rclone.conf")

    assert resolve_rclone_config(region) == "/custom/rclone.conf"


def test_resolve_rclone_config_ignores_blank_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RCLONE_CONFIG", "   ")

    assert resolve_rclone_config("o1") == ISLAND_RCLONE_CONFIG


def test_build_copy_script_with_config() -> None:
    script = build_copy_script(
        "/mnt/data/model dir/artifacts.tar",
        "papayaeye-papaya:transfer2cloud/o1/artifacts.tar",
        "/mnt/nfs/rclone.conf",
    )

    assert script == EXPECTED_SCRIPT_WITH_CONFIG


def test_build_copy_script_without_config_omits_config_flag() -> None:
    script = build_copy_script(
        "/mnt/data/artifacts.tar",
        "remote:bucket/artifacts.tar",
    )

    assert "--config" not in script
    assert script.startswith("SOURCE=/mnt/data/artifacts.tar\n")
    assert "DESTINATION=remote:bucket/artifacts.tar" in script
    assert script.endswith("--no-check-certificate")
