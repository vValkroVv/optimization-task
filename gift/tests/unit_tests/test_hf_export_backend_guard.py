"""HF conversion and backends with their own checkpoint adapter are incompatible.

The in-repo HF export and load read the plain model state dict; a backend that
checkpoints through a ``CheckpointAdapter`` (gfsdp2) does not provide one, and
the export would die inside ``save()`` after the DCP was already written.
"""

import pytest

from torchtitan.components.checkpoint import _reject_hf_with_checkpoint_adapter
from torchtitan.config.configs import ParallelismConfig
from torchtitan.config.schema.protocols.model import BaseModelConfig
from torchtitan.distributed.backends import build_sharding_backend

_FLAGS = ("save_in_hf", "last_save_in_hf", "initial_load_in_hf")


def _flags(**enabled: bool) -> dict[str, bool]:
    return {flag: enabled.get(flag, False) for flag in _FLAGS}


@pytest.mark.parametrize("flag", _FLAGS)
def test_hf_conversion_is_refused_with_an_adapter(flag):
    with pytest.raises(ValueError, match=f"checkpoint.{flag}") as excinfo:
        _reject_hf_with_checkpoint_adapter(
            fsdp_backend="gfsdp2",
            has_checkpoint_adapter=True,
            **_flags(**{flag: True}),
        )
    assert "gfsdp2" in str(excinfo.value)


@pytest.mark.parametrize("flag", _FLAGS)
def test_hf_conversion_is_kept_without_an_adapter(flag):
    _reject_hf_with_checkpoint_adapter(
        fsdp_backend="fsdp2",
        has_checkpoint_adapter=False,
        **_flags(**{flag: True}),
    )


def test_an_adapter_without_hf_conversion_is_the_ordinary_case():
    _reject_hf_with_checkpoint_adapter(
        fsdp_backend="gfsdp2", has_checkpoint_adapter=True, **_flags()
    )


def test_fsdp2_has_no_checkpoint_adapter():
    backend = build_sharding_backend(ParallelismConfig(), BaseModelConfig())
    assert backend.checkpoint_adapter([]) is None


def test_fsdp2_is_the_backend_that_exports_hf():
    backend = build_sharding_backend(ParallelismConfig(), BaseModelConfig())
    assert backend.exports_hf_checkpoint is True


def test_an_unknown_backend_is_refused():
    with pytest.raises(ValueError, match="unknown parallelism.fsdp_backend"):
        build_sharding_backend(ParallelismConfig(fsdp_backend="ddp"), BaseModelConfig())
