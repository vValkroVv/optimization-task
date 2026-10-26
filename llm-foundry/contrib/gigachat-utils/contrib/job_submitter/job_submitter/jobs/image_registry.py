import json
from pathlib import Path

LLMF_BASE_IMAGES = {
    # Hopper regions run the DSA-enabled image (FlashMLA + tilelang + cutlass-dsl).
    "o1": "/mnt/nfs/07-share/transfer2papaya/images/llm-foundry_2026_10_08_commitdd3cd14f.sqsh",
    "o2": "/mnt/nfs/07-share/transfer2papaya/images/llm-foundry_2026_10_08_commitdd3cd14f.sqsh",
    "o5": "/mnt/nfs/07-share/transfer2papaya/images/llm-foundry_2026_10_08_commitdd3cd14f.sqsh",
    "SR008": "cr.ai.cloud.ru/19d74875-55e7-4f05-92d7-5bff1b43b8a0/llm-foundry:2026_10_08_commitdd3cd14f",
    # SR004 is A100 and SR006 is older still: both stay on the pre-DSA image, and the DSA
    # suites skip themselves there on the sm90 check rather than failing at kernel launch.
    "SR004": "cr.ai.cloud.ru/19d74875-55e7-4f05-92d7-5bff1b43b8a0/llm-foundry:2026_08_14_commitbcaad6f0",
    "SR006": "cr.ai.cloud.ru/19d74875-55e7-4f05-92d7-5bff1b43b8a0/llm-foundry:2026_04_17_commitb17dc8f8",
}

# Torchtitan base images are pinned in the torchtitan checkout as the single
# source of truth; GIFT jobs pull them from there via load_gift_base_images().
GIFT_IMAGES_PATH = Path("gift_docker") / "images.json"


def load_gift_base_images(gift_root: str | Path) -> dict[str, str]:
    """Load the region -> torchtitan image mapping pinned in a GIFT checkout."""
    path = Path(gift_root) / GIFT_IMAGES_PATH
    try:
        images = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(
            f"GIFT base image registry not found: {path}. Point gift_root at a "
            "torchtitan checkout containing it or pass base_image explicitly."
        ) from exc
    if not isinstance(images, dict) or not all(
        isinstance(region, str) and isinstance(image, str) and image
        for region, image in images.items()
    ):
        raise ValueError(
            f"Malformed GIFT base image registry {path}: expected a JSON object "
            "mapping region names to image references."
        )
    return images


PAM_BASE_IMAGES = {
    "opensource": {
        "o1": "/mnt/nfs/01-home/infidelis/transfer2papaya/images/pretrain-auto-metrics_opensource_2026_10_08_commitf91eb911.sqsh",
        "o2": "/mnt/nfs/07-share/transfer2papaya/images/pretrain-auto-metrics_opensource_2026_10_08_commitf91eb911.sqsh",
        "o5": "/mnt/nfs/01-home/infidelis/transfer2papaya/images/pretrain-auto-metrics_opensource_2026_10_08_commitf91eb911.sqsh",
        "SR004": "cr.ai.cloud.ru/19d74875-55e7-4f05-92d7-5bff1b43b8a0/pretrain-auto-metrics_opensource:2026_10_08_commitf91eb911",
        "SR006": "cr.ai.cloud.ru/19d74875-55e7-4f05-92d7-5bff1b43b8a0/pretrain-auto-metrics_opensource:2026_10_08_commitf91eb911",
        "SR008": "cr.ai.cloud.ru/19d74875-55e7-4f05-92d7-5bff1b43b8a0/pretrain-auto-metrics_opensource:2026_10_08_commitf91eb911",
    },
    "custom": {
        "o1": "/mnt/nfs/01-home/infidelis/transfer2papaya/images/pretrain-auto-metrics_custom_2026_10_08_commitf91eb911.sqsh",
        "o2": "/mnt/nfs/07-share/transfer2papaya/images/pretrain-auto-metrics_custom_2026_10_08_commitf91eb911.sqsh",
        "o5": "/mnt/nfs/01-home/infidelis/transfer2papaya/images/pretrain-auto-metrics_custom_2026_10_08_commitf91eb911.sqsh",
        "SR004": "cr.ai.cloud.ru/19d74875-55e7-4f05-92d7-5bff1b43b8a0/pretrain-auto-metrics_custom:2026_10_17_commita2fbe156",
        "SR006": "cr.ai.cloud.ru/19d74875-55e7-4f05-92d7-5bff1b43b8a0/pretrain-auto-metrics_custom:2026_10_17_commita2fbe156",
        "SR008": "cr.ai.cloud.ru/19d74875-55e7-4f05-92d7-5bff1b43b8a0/pretrain-auto-metrics_custom:2026_10_17_commita2fbe156",
    },
    "custom_vision": {
        "SR004": "cr.ai.cloud.ru/19d74875-55e7-4f05-92d7-5bff1b43b8a0/pretrain-auto-metrics_custom_vision:2026_07_01_commitbef24cbf",
        "SR006": "cr.ai.cloud.ru/19d74875-55e7-4f05-92d7-5bff1b43b8a0/pretrain-auto-metrics_custom_vision:2026_07_01_commitbef24cbf",
        "SR008": "cr.ai.cloud.ru/19d74875-55e7-4f05-92d7-5bff1b43b8a0/pretrain-auto-metrics_custom_vision:2026_07_01_commitbef24cbf",
    },
}

PAM_IMAGES_PATH = Path("images.json")


def load_pam_base_images(pam_root: str | Path) -> dict[str, dict[str, str]]:
    path = Path(pam_root) / PAM_IMAGES_PATH
    if not path.is_file():
        return PAM_BASE_IMAGES
    return json.loads(path.read_text(encoding="utf-8"))


VERL_BASE_IMAGES = {
    "sglang": {
        "o1": "/mnt/nfs/01-home/infidelis/transfer2papaya/images/verl_0.7.1_app.sglang0.5.14.mcore.2026_09_29_commitc61c0b12.sqsh",
        "o2": "/mnt/nfs/07-share/transfer2papaya/images/verl_0.7.1_app.sglang0.5.14.mcore.2026_09_29_commitc61c0b12.sqsh",
        "o5": "/mnt/nfs/01-home/infidelis/transfer2papaya/images/verl_0.7.1_app.sglang0.5.14.mcore.2026_09_29_commitc61c0b12.sqsh",
        "SR004": "cr.ai.cloud.ru/19d74875-55e7-4f05-92d7-5bff1b43b8a0/verl_0.7.1:app.sglang0.5.14.mcore.2026_09_29_commitc61c0b12",
        "SR006": "cr.ai.cloud.ru/19d74875-55e7-4f05-92d7-5bff1b43b8a0/verl_0.7.1:app.sglang0.5.14.mcore.2026_09_29_commitc61c0b12",
        "SR008": "cr.ai.cloud.ru/19d74875-55e7-4f05-92d7-5bff1b43b8a0/verl_0.7.1:app.sglang0.5.14.mcore.2026_09_29_commitc61c0b12",
    },
    "sglang_custom": {
        "o1": "/mnt/nfs/01-home/infidelis/transfer2papaya/images/verl_0.7.1_app.sglang_custom0.5.14.mcore.2026_09_29_commitc61c0b12.sqsh",
        "o2": "/mnt/nfs/07-share/transfer2papaya/images/verl_0.7.1_app.sglang_custom0.5.14.mcore.2026_09_29_commitc61c0b12.sqsh",
        "o5": "/mnt/nfs/01-home/infidelis/transfer2papaya/images/verl_0.7.1_app.sglang_custom0.5.14.mcore.2026_09_29_commitc61c0b12.sqsh",
        "SR004": "cr.ai.cloud.ru/19d74875-55e7-4f05-92d7-5bff1b43b8a0/verl_0.7.1:app.sglang_custom0.5.14.mcore.2026_10_12_commit0d602eb4",
        "SR006": "cr.ai.cloud.ru/19d74875-55e7-4f05-92d7-5bff1b43b8a0/verl_0.7.1:app.sglang_custom0.5.14.mcore.2026_10_12_commit0d602eb4",
        "SR008": "cr.ai.cloud.ru/19d74875-55e7-4f05-92d7-5bff1b43b8a0/verl_0.7.1:app.sglang_custom0.5.14.mcore.2026_10_12_commit0d602eb4",
    },
    "vllm": {
        "o1": "/mnt/nfs/01-home/infidelis/transfer2papaya/images/verl_0.7.1_app.vllm0.17.0.mcore.2026_03_27_commitbb79ea8b.sqsh",
        "o5": "/mnt/nfs/01-home/infidelis/transfer2papaya/images/verl_0.7.1_app.vllm0.17.0.mcore.2026_03_27_commitbb79ea8b.sqsh",
        "SR004": "cr.ai.cloud.ru/19d74875-55e7-4f05-92d7-5bff1b43b8a0/verl_0.7.1:app.vllm0.17.0.mcore.2026_03_27_commitbb79ea8b",
        "SR006": "cr.ai.cloud.ru/19d74875-55e7-4f05-92d7-5bff1b43b8a0/verl_0.7.1:app.vllm0.17.0.mcore.2026_03_27_commitbb79ea8b",
        "SR008": "cr.ai.cloud.ru/19d74875-55e7-4f05-92d7-5bff1b43b8a0/verl_0.7.1:app.vllm0.17.0.mcore.2026_03_27_commitbb79ea8b",
    },
}
