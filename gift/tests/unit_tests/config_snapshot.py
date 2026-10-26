"""Readable YAML snapshots with aliases for exactly equal JSON subtrees."""

import hashlib
import json
import re
from typing import Any

import yaml


def dump_snapshot(value: Any) -> str:
    """Write ordinary YAML; ``yaml.safe_load`` restores the complete JSON value.

    Intern only substantial containers, preserving explicit nulls, type identities
    and list order. Content-based names keep unrelated aliases stable after edits.
    """
    interned = {}
    metadata = {}
    node_metadata = {}
    anchors = {}

    def intern(item):
        signature = json.dumps(
            item, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        if not isinstance(item, (dict, list)):
            return item
        share = len(signature) >= 128
        if share and signature in interned:
            return interned[signature]
        if isinstance(item, dict):
            result = {key: intern(item[key]) for key in sorted(item)}
            label = item.get("__type__", "mapping")
        else:
            result = [intern(child) for child in item]
            label = "sequence"
        label = label.rsplit(".", 1)[-1] if isinstance(label, str) else "mapping"
        label = re.sub(r"[^A-Za-z0-9_-]+", "_", label).strip("_") or "mapping"
        metadata[id(result)] = (label, signature)
        if share:
            interned[signature] = result
        return result

    class SnapshotDumper(yaml.SafeDumper):
        def represent_data(self, data):
            node = super().represent_data(data)
            if isinstance(data, (dict, list)):
                node_metadata[node] = metadata[id(data)]
            return node

        def generate_anchor(self, node):
            label, signature = node_metadata[node]
            digest = hashlib.sha256(signature.encode()).hexdigest()[:16]
            name = f"{label}_{digest}"
            previous = anchors.setdefault(name, node)
            if previous is not node:
                raise ValueError(f"snapshot anchor collision: {name}")
            return name

    return yaml.dump(
        intern(value),
        Dumper=SnapshotDumper,
        sort_keys=False,
        allow_unicode=True,
        width=100,
    )
