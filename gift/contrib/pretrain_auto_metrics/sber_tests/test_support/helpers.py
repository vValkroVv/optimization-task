import os
from contextlib import contextmanager
from pathlib import Path

from node_runner.src.config_model import JudgeEndpoint


def setup_tmp_checkpoint(tmp_path: Path):
    checkpoint_path = tmp_path / "checkpoint"
    artifacts_path = checkpoint_path / "artifacts"
    config_path = checkpoint_path / "artifacts" / "pam_config.yaml"
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(artifacts_path, exist_ok=True)
    return checkpoint_path, artifacts_path, config_path


@contextmanager
def set_env(**environ):
    old_values = {key: os.environ.get(key) for key in environ}
    os.environ.update(environ)
    try:
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def judge_endpoints(
    base_urls: dict[str, str], instances: dict[str, int] | None = None
) -> dict[str, JudgeEndpoint]:
    endpoints = {}
    for name, base_url in base_urls.items():
        endpoints[name] = JudgeEndpoint(base_url, (instances or {}).get(name, 1))
    return endpoints
