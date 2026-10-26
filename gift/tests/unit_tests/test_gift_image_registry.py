import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GIFT_IMAGES_PATH = REPO_ROOT / "gift_docker" / "images.json"


def _pinned_images() -> dict[str, str]:
    return json.loads(GIFT_IMAGES_PATH.read_text(encoding="utf-8"))


def test_registry_pins_an_image_per_cluster() -> None:
    images = _pinned_images()

    assert images
    for cluster, image in images.items():
        assert re.fullmatch(r"o\d+|SR\d{3}", cluster), cluster
        assert isinstance(image, str) and image, cluster


def test_cluster_images_match_backend_format() -> None:
    for cluster, image in _pinned_images().items():
        if cluster.startswith("SR"):
            assert image.startswith("cr.ai.cloud.ru/"), cluster
        else:
            assert image.endswith(".sqsh"), cluster


def test_job_submitter_loads_this_registry(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(REPO_ROOT / "contrib" / "job_submitter"))
    from job_submitter.jobs.image_registry import load_gift_base_images

    assert load_gift_base_images(REPO_ROOT) == _pinned_images()
