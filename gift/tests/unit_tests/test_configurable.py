# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import unittest
from dataclasses import field

from typing import TypeAlias

from torchtitan.config.configurable import config, Configurable
from torchtitan.config.validation import Issue, validate_config_tree, ValidationContext


class SpecOwner(Configurable):
    """Bound to its config by ``_owner_spec`` only, never by a ``Config`` alias."""

    __slots__ = ("config",)

    def __init__(self, config):
        self.config = config


@config
class WiredParentConfig(Configurable.Config):
    x: int = 1


class WiredParent(Configurable):
    """Its module is imported, so ``WiredParentConfig._owner`` is set."""

    Config: TypeAlias = WiredParentConfig

    def __init__(self, config):
        self.config = config


@config
class SpecConfig(WiredParentConfig):
    """Inherits the parent's wired ``_owner``; its own owner is reached by spec only."""

    _owner_spec = "tests.unit_tests.test_configurable:SpecOwner"


class TestConfigurable(unittest.TestCase):
    class OldStyleComponent(Configurable):
        """__init__ takes extra runtime kwargs (not config fields)."""

        @config
        class Config(Configurable.Config):
            x: int = 5

        def __init__(self, config: Config, *, dim: int):
            self.config = config
            self.dim = dim

    class NoKwargsComponent(Configurable):
        """Takes only config, no extra kwargs."""

        @config
        class Config(Configurable.Config):
            x: int = 5

        def __init__(self, config: Config):
            self.config = config

    def test_valid_config(self):
        """Config with kw_only=True and slots=True should work."""

        class MyComponent(Configurable):
            @config
            class Config(Configurable.Config):
                x: int = 5

            def __init__(self, config: Config):
                self.config = config

        cfg = MyComponent.Config(x=10)
        obj = cfg.build()
        self.assertIsInstance(obj, MyComponent)
        self.assertEqual(obj.config.x, 10)

    def test_config_decorator_makes_kw_only_slots(self):
        @config
        class Cfg(Configurable.Config):
            x: int = 5

        self.assertIn("__slots__", Cfg.__dict__)
        with self.assertRaises(TypeError):
            Cfg(7)  # pyrefly: ignore [bad-argument-count]

    def test_config_decorator_rejects_non_config(self):
        with self.assertRaises(TypeError):

            @config
            class NotAConfig:
                x: int = 5

    def test_build_without_owner_raises(self):
        """Calling build() on the base Config should raise NotImplementedError."""
        cfg = Configurable.Config()
        with self.assertRaises(NotImplementedError):
            cfg.build()
        with self.assertRaises(NotImplementedError):
            Configurable.Config.owner()

    def test_owner_resolves_spec_lazily(self):
        self.assertIs(WiredParentConfig.owner(), WiredParent)
        self.assertNotIn("_owner", SpecConfig.__dict__)
        obj = SpecConfig(x=3).build()
        self.assertIsInstance(obj, SpecOwner)
        self.assertIs(SpecConfig.owner(), SpecOwner)

    def test_owner_wired_through_alias(self):
        @config
        class AliasedConfig(Configurable.Config):
            x: int = 5

        class Aliased(Configurable):
            Config = AliasedConfig

            def __init__(self, config):
                self.config = config

        self.assertIs(AliasedConfig.owner(), Aliased)
        self.assertIsInstance(AliasedConfig().build(), Aliased)

    def test_validate_config_tree_prefixes_paths(self):
        class Inner(Configurable):
            @config
            class Config(Configurable.Config):
                x: int = 0

                def validate(self, ctx):
                    if self.x < 0:
                        yield Issue(path="x", message="negative")
                    if ctx.stage == "in_container" and self.x == 99:
                        yield Issue(path="", message="node-level", severity="warning")

            def __init__(self, config):
                self.config = config

        class Outer(Configurable):
            @config
            class Config(Configurable.Config):
                inner: Inner.Config = field(default_factory=Inner.Config)
                items: list[Inner.Config] = field(default_factory=list)
                named: dict[str, Inner.Config] = field(default_factory=dict)

            def __init__(self, config):
                self.config = config

        cfg = Outer.Config(
            inner=Inner.Config(x=-1),
            items=[Inner.Config(), Inner.Config(x=-2)],
            named={"k": Inner.Config(x=99)},
        )
        report = validate_config_tree(cfg, ValidationContext(stage="in_container"))
        self.assertEqual(
            {str(i) for i in report.issues},
            {"inner.x: negative", "items[1].x: negative", "named.k: node-level"},
        )
        self.assertFalse(report.ok)
        self.assertTrue(
            validate_config_tree(
                Outer.Config(), ValidationContext(stage="pre_submit")
            ).ok
        )

    def test_old_style_forwarding(self):
        """kwargs not in config fields are forwarded to __init__."""
        cfg = self.OldStyleComponent.Config(x=10)
        obj = cfg.build(dim=64)
        self.assertIsInstance(obj, self.OldStyleComponent)
        self.assertEqual(obj.config.x, 10)
        self.assertEqual(obj.dim, 64)

    def test_clone_isolation_old_style(self):
        """Original config is not mutated in old-style path."""
        cfg = self.OldStyleComponent.Config(x=10)
        obj = cfg.build(dim=64)
        obj.config.x = 999
        self.assertEqual(cfg.x, 10)

    def test_no_kwargs(self):
        """build() with no kwargs clones config and constructs."""
        cfg = self.NoKwargsComponent.Config(x=42)
        obj = cfg.build()
        self.assertIsInstance(obj, self.NoKwargsComponent)
        self.assertEqual(obj.config.x, 42)

    def test_no_kwargs_clone_isolation(self):
        """build() with no kwargs still clones the config."""
        cfg = self.NoKwargsComponent.Config(x=42)
        obj = cfg.build()
        obj.config.x = 999
        self.assertEqual(cfg.x, 42)

    def test_build_carries_init_false_fields(self):
        """``dataclasses.replace`` re-runs ``__init__`` and so resets a field
        declared ``init=False`` (``TrainerConfig.model_spec`` is one); the
        clone ``build()`` hands to the owner must keep the assigned value."""

        class Derived(Configurable):
            @config
            class Config(Configurable.Config):
                x: int = 1
                derived: object = field(init=False, default=None)

            def __init__(self, config: Config, *, dim: int = 0):
                self.config = config

        cfg = Derived.Config(x=2)
        cfg.derived = marker = object()
        self.assertIs(cfg.build().config.derived, marker)
        self.assertIs(cfg.build(dim=1).config.derived, marker)

    def test_to_dict_two_layer(self):
        """to_dict serializes nested configs (two layers deep)."""

        class Inner(Configurable):
            @config
            class Config(Configurable.Config):
                a: int = 1
                b: int = 2

            def __init__(self, config: Config):
                self.config = config

        class Outer(Configurable):
            @config
            class Config(Configurable.Config):
                x: int = 10
                inner: Inner.Config = field(default_factory=Inner.Config)

            def __init__(self, config: Config):
                self.config = config

        cfg = Outer.Config(x=42)
        d = cfg.to_dict()
        self.assertEqual(d["x"], 42)
        # Inner config is serialised via its own to_dict
        self.assertIn("inner", d)
        self.assertEqual(d["inner"]["a"], 1)
        self.assertEqual(d["inner"]["b"], 2)

        # After build: all fields present
        obj = cfg.build()
        d2 = obj.config.to_dict()
        self.assertEqual(d2["x"], 42)
        self.assertEqual(d2["inner"]["a"], 1)
        self.assertEqual(d2["inner"]["b"], 2)

    def test_to_dict_serializes_dicts_of_configs(self):
        """dict[str, Config] fields (e.g. the trainer's callbacks) become plain data."""

        class Callback(Configurable):
            @config
            class Config(Configurable.Config):
                interval: int = 10

            def __init__(self, config: Config):
                self.config = config

        class Manager(Configurable):
            @config
            class Config(Configurable.Config):
                callbacks: dict[str, Callback.Config] = field(default_factory=dict)

            def __init__(self, config: Config):
                self.config = config

        cfg = Manager.Config(callbacks={"ema": Callback.Config(interval=5)})
        d = cfg.to_dict()
        self.assertEqual(d["callbacks"], {"ema": {"interval": 5}})
        json.dumps(d)  # the debug.print_config path

    def test_repr(self):
        """repr() works for configs."""
        cfg = self.NoKwargsComponent.Config(x=42)
        r = repr(cfg)
        self.assertIn("x=42", r)


if __name__ == "__main__":
    unittest.main()
