import json
import re
import sys
from pathlib import Path

IMAGES_PATH = Path(__file__).parent.parent / "images.json"
CI_PATH = Path(__file__).parent.parent / ".gitlab-ci.yml"


def main(image_tag: str) -> None:
    images = json.loads(IMAGES_PATH.read_text(encoding="utf-8"))
    for sglang_tag in ("custom", "opensource"):
        for region, image in images[sglang_tag].items():
            if image.endswith(".sqsh"):
                images_dir = image.rsplit("/", 1)[0]
                images[sglang_tag][region] = (
                    f"{images_dir}/pretrain-auto-metrics_{sglang_tag}_{image_tag}.sqsh"
                )
            else:
                registry_image = image.rsplit(":", 1)[0]
                images[sglang_tag][region] = f"{registry_image}:{image_tag}"
    IMAGES_PATH.write_text(json.dumps(images, indent=4) + "\n", encoding="utf-8")
    ci = CI_PATH.read_text(encoding="utf-8")
    ci = re.sub(
        r'^(  PAM_IMAGE: )".*"$',
        rf'\1"{images["custom"]["SR008"]}"',
        ci,
        count=1,
        flags=re.M,
    )
    CI_PATH.write_text(ci, encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
