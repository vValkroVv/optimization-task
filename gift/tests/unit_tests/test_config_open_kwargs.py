"""Open external kwargs still obey the declarative config value contract."""

import pytest

from torchtitan.config.decode import decode
from torchtitan.config.errors import ConfigSchemaError
from torchtitan.config.schema.models.flux.model.hf_embedder import FluxEmbedderConfig


def test_nested_external_kwargs_preserve_values_and_types():
    kwargs = {
        "revision": "main",
        "enabled": True,
        "options": [1, 2.5, None, {"x": "y"}],
    }
    cfg = decode(FluxEmbedderConfig, {"version": "model", "hf_kwargs": kwargs})
    assert cfg.hf_kwargs == kwargs
    assert type(cfg.hf_kwargs["enabled"]) is bool
    assert cfg.hf_kwargs is not kwargs


@pytest.mark.parametrize("value", [object(), {1: "invalid"}, {"nested": object()}])
def test_external_kwargs_reject_nonportable_values(value):
    with pytest.raises(ConfigSchemaError, match="hf_kwargs"):
        decode(FluxEmbedderConfig, {"version": "model", "hf_kwargs": {"option": value}})


@pytest.mark.parametrize("kind", ["list", "mapping"])
def test_external_kwargs_reject_cycles_with_a_configuration_error(kind):
    cyclic = [] if kind == "list" else {}
    if isinstance(cyclic, list):
        cyclic.append(cyclic)
    else:
        cyclic["again"] = cyclic
    with pytest.raises(ConfigSchemaError, match="hf_kwargs.option.*cyclic"):
        decode(
            FluxEmbedderConfig, {"version": "model", "hf_kwargs": {"option": cyclic}}
        )


def test_shared_external_kwargs_containers_are_independent_copies():
    shared = {"value": [None, 3]}
    cfg = decode(
        FluxEmbedderConfig,
        {"version": "model", "hf_kwargs": {"first": shared, "second": shared}},
    )
    assert cfg.hf_kwargs == {"first": shared, "second": shared}
    assert cfg.hf_kwargs["first"] is not cfg.hf_kwargs["second"]
