# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the activation CPU offload role registry in ``sac_utils``."""

import pytest
import torch

from composer.trainer.sac_utils import (
    _OFFLOAD_INPUTS_MODULES_ALLOWLIST,
    _OFFLOAD_ROLE_CLASS_REGISTRY,
    count_activation_offload_slots,
    validate_cpu_offload_roles,
)


def _make_norm_class(name: str):
    """Create a throwaway ``nn.Module`` subclass whose ``__name__`` is ``name``.

    ``validate_cpu_offload_roles`` matches modules by class name only, so the
    real llm-foundry norm implementations (which need GPU kernels) are not
    required here.
    """
    return type(name, (torch.nn.Module,), {})


class _Layer(torch.nn.Module):

    def __init__(self, norm: torch.nn.Module):
        super().__init__()
        self.post_feedforward_layernorm = norm
        self.post_feedforward_layernorm._cpu_offload_role = 'post_feedforward_layernorm'


def test_registry_roles_match_allowlist():
    assert set(_OFFLOAD_ROLE_CLASS_REGISTRY) == set(_OFFLOAD_INPUTS_MODULES_ALLOWLIST)


@pytest.mark.parametrize('norm_cls_name', ['ZeroCenteredGatedNorm', 'GatedNorm'])
def test_post_feedforward_layernorm_accepts_gated_norms(norm_cls_name: str):
    layer = _Layer(_make_norm_class(norm_cls_name)())
    # Must not raise.
    validate_cpu_offload_roles([layer], ['post_feedforward_layernorm'])


def test_post_feedforward_layernorm_rejects_unknown_class():
    layer = _Layer(_make_norm_class('SomeOtherNorm')())
    with pytest.raises(ValueError, match="role 'post_feedforward_layernorm'"):
        validate_cpu_offload_roles([layer], ['post_feedforward_layernorm'])


def test_role_not_requested_is_not_validated():
    """Only roles listed in ``offload_inputs_modules`` are checked."""
    layer = _Layer(_make_norm_class('SomeOtherNorm')())
    validate_cpu_offload_roles([layer], ['self_attn'])


def test_root_module_is_used_for_class_check():
    """Wrapped modules (e.g. checkpoint wrappers) expose the real class via ``root_module``."""
    inner = _make_norm_class('GatedNorm')()
    wrapper = torch.nn.Module()
    wrapper.root_module = inner
    layer = _Layer(wrapper)
    validate_cpu_offload_roles([layer], ['post_feedforward_layernorm'])


def test_count_activation_offload_slots_counts_marked_leaves():
    layer = _Layer(_make_norm_class('GatedNorm')())
    assert count_activation_offload_slots(layer) == 1
    layer._cpu_offload_role = 'decoder_layer'  # parent marked too: still one leaf slot
    assert count_activation_offload_slots(layer) == 1
