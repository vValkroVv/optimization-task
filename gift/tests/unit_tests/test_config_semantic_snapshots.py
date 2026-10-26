"""Semantic snapshots survive Python module moves without erasing component types."""

from dataclasses import field, make_dataclass

from tests.unit_tests.config_golden import canonical_json
from torchtitan.config.schema.protocols.module import Ones, Zeros


def test_unregistered_same_named_classes_remain_distinct():
    before = make_dataclass("ProjectionShape", [("width", int)])
    after = make_dataclass("ProjectionShape", [("width", int)])
    before.__module__ = "old_model.schema"
    after.__module__ = "new_model.schema"
    assert canonical_json(before(4)) != canonical_json(after(4))
    assert canonical_json(before(4), semantic_types=True) != canonical_json(
        after(4), semantic_types=True
    )


def test_registered_component_module_move_keeps_semantic_snapshot(monkeypatch):
    before = canonical_json(Zeros(), semantic_types=True)
    diagnostic = canonical_json(Zeros())
    monkeypatch.setattr(Zeros, "__module__", "relocated.schema")
    assert canonical_json(Zeros()) != diagnostic
    assert canonical_json(Zeros(), semantic_types=True) == before


def test_parameter_initializer_types_remain_distinct():
    assert canonical_json(Zeros(), semantic_types=True) != canonical_json(
        Ones(), semantic_types=True
    )
    assert "param_init:zeros" in canonical_json(Zeros(), semantic_types=True)


def test_geometry_and_scalar_options_still_change_snapshot():
    shape = make_dataclass("ProjectionShape", [("width", int), ("bias", bool)])
    assert canonical_json(shape(4, False), semantic_types=True) != canonical_json(
        shape(8, False), semantic_types=True
    )
    assert canonical_json(shape(4, False), semantic_types=True) != canonical_json(
        shape(4, True), semantic_types=True
    )


def test_new_unset_options_are_visible_but_provenance_remains_nonsemantic():
    before = make_dataclass("ProjectionShape", [("width", int)])
    after = make_dataclass(
        "ProjectionShape",
        [
            ("width", int),
            ("optional_template", object, field(default=None)),
            ("source", str, field(default="first.yaml", compare=False)),
        ],
    )
    assert canonical_json(before(4), semantic_types=True) != canonical_json(
        after(4), semantic_types=True
    )
    assert '"optional_template": null' in canonical_json(after(4), semantic_types=True)
    assert canonical_json(
        after(4, source="first.yaml"), semantic_types=True
    ) == canonical_json(after(4, source="second.yaml"), semantic_types=True)
