"""Readable snapshots of decoded configs and materialized model trees.

Each experiment and model preset has a YAML snapshot under ``golden_configs/``.
Snapshots retain unset fields so additions and removals appear in ordinary diffs.
Registered YAML types distinguish components; Python
module locations of registered components and nonsemantic provenance do not
affect the snapshot. Unregistered classes retain qualified identities to avoid
collisions between unrelated schemas with the same class name.
To see what changed, diff the dump on both
checkouts::

    python -m tests.unit_tests.config_golden experiments/llama3_debug.yaml > before.json

``--semantic`` prints the exact representation used by the snapshots. The default
dump retains qualified class names for diagnostics; ``--no-class`` removes them.
After reviewing an intended change, update snapshots::

    TORCHTITAN_UPDATE_GOLDEN=1 python -m pytest tests/unit_tests/test_experiment_parity.py \\
        tests/unit_tests/test_model_arch_presets.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from tests.unit_tests.config_parity import structural_dump
from tests.unit_tests.config_snapshot import dump_snapshot

GOLDEN_DIR = Path(__file__).with_name("golden_configs")
UPDATE_ENV = "TORCHTITAN_UPDATE_GOLDEN"


def canonical_json(
    obj, *, with_class: bool = True, semantic_types: bool = False
) -> str:
    """The structural dump as deterministic, human-readable JSON."""
    dump = structural_dump(obj, semantic_types=semantic_types)
    return json.dumps(
        dump if with_class else strip_class(dump), indent=1, sort_keys=True
    )


def strip_class(dump):
    """*dump* without its ``__class__`` entries."""
    if isinstance(dump, dict):
        return {k: strip_class(v) for k, v in dump.items() if k != "__class__"}
    if isinstance(dump, list):
        return [strip_class(v) for v in dump]
    return dump


def load_golden() -> dict[str, Any]:
    return {
        str(path.relative_to(GOLDEN_DIR).with_suffix(".yaml")): yaml.safe_load(
            path.read_text()
        )
        for path in sorted(GOLDEN_DIR.rglob("*.yaml"))
    }


def check_golden(key: str, obj) -> None:
    """Assert *obj* matches the readable snapshot for *key*.

    With ``TORCHTITAN_UPDATE_GOLDEN=1`` the pin is (re)written instead.
    """
    actual = canonical_json(obj, semantic_types=True) + "\n"
    path = GOLDEN_DIR / key
    if os.environ.get(UPDATE_ENV) == "1":
        compact = dump_snapshot(json.loads(actual))
        if not path.exists() or path.read_text() != compact:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(compact)
        return
    assert path.exists(), (
        f"{key}: no golden snapshot in {GOLDEN_DIR.name}; create it with "
        f"{UPDATE_ENV}=1 (see tests/unit_tests/config_golden.py)"
    )
    expected = (
        json.dumps(yaml.safe_load(path.read_text()), indent=1, sort_keys=True) + "\n"
    )
    if expected != actual:
        from difflib import unified_diff

        difference = "\n".join(
            unified_diff(
                expected.splitlines(),
                actual.splitlines(),
                fromfile=str(path.relative_to(GOLDEN_DIR)),
                tofile="current config",
            )
        )
        raise AssertionError(f"{key}: decoded config changed:\n{difference}")


def resolve(key: str):
    """The decoded object a golden *key* stands for."""
    from tests.unit_tests.config_helpers import EXPERIMENT_ROOT, PRESET_ROOT
    from torchtitan.config import load_document, load_raw_document
    from torchtitan.config.build import materialize_model_config, prepare_document
    from torchtitan.config.decode import decode
    from torchtitan.config.sections import ModelSection

    kind, _, name = key.partition("/")
    if kind == "experiments":
        document, _ = load_document(str(EXPERIMENT_ROOT / name))
        prepared = prepare_document(document)
        return {
            "trainer": prepared.trainer_config,
            "model": prepared.model_section,
            "materialized_model": prepared.model_config,
        }
    elif kind == "presets":
        document = load_raw_document(str(PRESET_ROOT / name))
    else:
        raise SystemExit(
            f"unknown golden key {key!r}: expected experiments/... or presets/..."
        )
    section = decode(ModelSection, document.data["model"], path="model")
    return {
        "model": section,
        "materialized_model": materialize_model_config(section.config),
    }


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    with_class = "--no-class" not in args
    semantic_types = "--semantic" in args
    for key in args:
        if key not in ("--no-class", "--semantic"):
            print(
                canonical_json(
                    resolve(key), with_class=with_class, semantic_types=semantic_types
                )
            )


if __name__ == "__main__":
    main()
