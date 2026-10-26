from typing import Callable

import pytest

from .data_dpo import DPOModules
from .data_mla import MLAModules
from .data_moe import MoeModules


@pytest.fixture(scope="session")
def unified_module() -> Callable:
    """Can be used for single module import."""
    from importlib import import_module

    def _load(module_path: str, module_name: str):
        try:
            module = import_module(module_path)
        except ImportError as e:
            raise ImportError(f"Failed to import module {module_path}: {e}") from e
        try:
            return getattr(module, module_name)
        except AttributeError as e:
            raise AttributeError(
                f"Module {module_path} has no attribute {module_name}: {e}"
            ) from e

    return _load


@pytest.fixture(scope="session")
def dpo_modules() -> DPOModules:
    try:
        from flash_attn.losses.cross_entropy import (
            CrossEntropyLoss as FusedCrossEntropyLoss,
        )
    except ImportError as e:
        raise ImportError(f"flash_attn.losses.cross_entropy import failed: {e}")

    from llmfoundry.models.dpo import (
        ComposerDPOModel,
        calculate_chosen_and_rejected_logprobs,
        calculate_nll,
        calculate_po_loss,
        calculate_spectoken_nll,
        filter_output,
        get_masked_labels,
        reshape_batch,
        reshape_images,
    )

    return DPOModules(
        FusedCrossEntropyLoss=FusedCrossEntropyLoss,
        ComposerDPOModel=ComposerDPOModel,
        calculate_chosen_and_rejected_logprobs=calculate_chosen_and_rejected_logprobs,
        calculate_nll=calculate_nll,
        calculate_po_loss=calculate_po_loss,
        calculate_spectoken_nll=calculate_spectoken_nll,
        get_masked_labels=get_masked_labels,
        reshape_batch=reshape_batch,
        reshape_images=reshape_images,
        filter_output=filter_output,
    )


@pytest.fixture(scope="session")
def mla_modules() -> MLAModules:
    from llmfoundry.models.layers.custom_embedding import (
        LlamaRotaryEmbedding,
        apply_rotary_emb,
    )
    from llmfoundry.models.layers.triton_rotary_embeddings import apply_rotary_emb_mla

    return MLAModules(
        LlamaRotaryEmbedding=LlamaRotaryEmbedding,
        apply_rotary_emb=apply_rotary_emb,
        apply_rotary_emb_mla=apply_rotary_emb_mla,
    )


@pytest.fixture(scope="session")
def moe_modules() -> MoeModules:
    from llmfoundry.models.layers.moe.moe_layers import (
        DeepseekGMMMoeBlock,
        SonicGMMMoeBlock,
    )

    return MoeModules(
        DeepseekGMMMoeBlock=DeepseekGMMMoeBlock,
        SonicGMMMoeBlock=SonicGMMMoeBlock,
    )
