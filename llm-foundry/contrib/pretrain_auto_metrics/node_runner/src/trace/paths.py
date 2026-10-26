from pathlib import Path


def node_trace_dir(artifacts_path: str | Path, node_name: str) -> Path:
    return Path(artifacts_path, "traces", node_name)
