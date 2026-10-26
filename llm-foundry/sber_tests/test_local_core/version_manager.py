import os
import re

from .constants import ARTIFACT_DIR

VERSION_PATTERN = r'^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$'


def validate_and_update_version(version: str, version_name: str, allow_update: bool = False) -> str:
    if allow_update:
        prompt_message = f"Enter new {version_name}: "
        if version_name == "GOLDEN_VERSION":
            prompt_message = "Are you sure that you want to generate new golden checkpoints? Enter new golden version: "

        new_version = input(prompt_message)

        if os.path.exists(os.path.join(ARTIFACT_DIR, new_version)):
            raise ValueError(f"{version_name}={new_version} already exists! Please update the version.")

        if not re.match(VERSION_PATTERN, new_version):
            raise ValueError(f"{version_name}={new_version} does not match the pattern {VERSION_PATTERN}.")

        return new_version
    else:
        if not re.match(VERSION_PATTERN, version):
            raise ValueError(f"{version_name}={version} does not match the pattern {VERSION_PATTERN}.")
        return version
