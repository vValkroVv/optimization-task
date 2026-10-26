from __future__ import annotations

import os
import shlex

from job_submitter.core.job_templates.region_configs import SLURM_REGIONS

# Slurm REST jobs start without HOME, so rclone cannot resolve the default
# ~/.config/rclone/rclone.conf; island jobs must pass this config explicitly.
ISLAND_RCLONE_CONFIG = "/mnt/nfs/01-home/optimusprime/.config/rclone/rclone.conf"

ISLAND_REGIONS = {SLURM_REGIONS.o1, SLURM_REGIONS.o2, SLURM_REGIONS.o5}


def resolve_rclone_config(region: str) -> str:
    override = os.getenv("RCLONE_CONFIG", "").strip()
    if override:
        return override
    if region in ISLAND_REGIONS:
        return ISLAND_RCLONE_CONFIG
    return ""


def build_copy_script(
    source_path: str,
    destination_path: str,
    rclone_config: str = "",
) -> str:
    config_line = ""
    if rclone_config:
        config_line = f"    --config {shlex.quote(rclone_config)} \\\n"
    return f"""
SOURCE={shlex.quote(source_path)}
DESTINATION={shlex.quote(destination_path)}

rclone copyto -L "$SOURCE" "$DESTINATION" \\
{config_line}    --transfers 8 \\
    --checkers 16 \\
    --multi-thread-streams 8 \\
    --s3-upload-concurrency 16 \\
    --buffer-size 128Mi \\
    --s3-chunk-size 128Mi \\
    --s3-upload-cutoff 200M \\
    --retries 3 \\
    --low-level-retries 10 \\
    --size-only \\
    --stats 30s \\
    --stats-one-line \\
    --log-level INFO \\
    --no-check-certificate
""".strip()
