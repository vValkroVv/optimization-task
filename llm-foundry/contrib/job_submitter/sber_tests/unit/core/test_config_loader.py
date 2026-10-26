import pytest
from omegaconf import OmegaConf as om

from job_submitter.core.config_loader import (
    apply_overrides,
    register_relative_config_loader,
)


@pytest.mark.unit
class TestApplyOverrides:
    def test_merges_override_block(self) -> None:
        cfg = om.create(
            {
                "a": 1,
                "nested": {"x": 1, "y": 2},
                "overrides": {"nested": {"y": 20}, "b": 2},
            }
        )

        merged = apply_overrides(cfg)
        assert merged.a == 1
        assert merged.b == 2
        assert merged.nested.x == 1
        assert merged.nested.y == 20
        assert "overrides" not in merged

    def test_returns_same_config_without_override(self) -> None:
        cfg = om.create({"a": 1})
        result = apply_overrides(cfg)
        assert result.a == 1


@pytest.mark.unit
class TestRelativeConfigLoader:
    def test_supports_rp_load_and_rp_merge(self, tmp_yaml) -> None:
        tmp_yaml("base.yaml", "a: 1\nnested:\n  x: 10\n")
        tmp_yaml("patch.yaml", "nested:\n  y: 20\n")
        root = tmp_yaml(
            "root.yaml",
            "loaded: ${rp.load:base.yaml}\nmerged: ${rp.merge:base.yaml,patch.yaml}\n",
        )

        register_relative_config_loader(root.parent)
        cfg = om.load(root)
        om.resolve(cfg)
        loaded = om.to_container(cfg, resolve=True)

        assert loaded["loaded"]["a"] == 1
        assert loaded["merged"]["nested"]["x"] == 10
        assert loaded["merged"]["nested"]["y"] == 20

    def test_rp_merge_without_paths_returns_empty_dict(self, tmp_yaml) -> None:
        root = tmp_yaml("root.yaml", "merged: ${rp.merge:}\n")
        register_relative_config_loader(root.parent)
        cfg = om.load(root)
        om.resolve(cfg)
        loaded = om.to_container(cfg, resolve=True)
        assert loaded["merged"] == {}
