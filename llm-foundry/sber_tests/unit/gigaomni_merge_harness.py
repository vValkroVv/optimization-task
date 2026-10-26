"""Shared inputs for the CPU and GPU GigaOmni modality-merge tests."""

from types import SimpleNamespace

import torch

from llmfoundry.models.gigaomni.modelling_gigaomni import (
    GigaOmniDataState,
    GigaOmniForCausalLM,
)


def build_merge_inputs(
    device: str, *, modality_rows: int = 2
) -> tuple[object, GigaOmniDataState]:
    """Build the minimum GigaOmni owner and data state for modality merging."""
    model = object.__new__(GigaOmniForCausalLM)
    model.modalities = [
        SimpleNamespace(
            name="vision",
            config=SimpleNamespace(
                data_flow=SimpleNamespace(
                    encoders=[
                        SimpleNamespace(
                            inputs=[
                                SimpleNamespace(
                                    type="modality_inputs",
                                    name="vision",
                                    keys=["pixel_values"],
                                )
                            ]
                        )
                    ]
                ),
                modality_output={
                    "type": "adapter_outputs",
                    "name": "vision",
                    "key": "embeddings",
                    "placeholder_token_id": 99,
                },
            ),
        ),
    ]
    state = GigaOmniDataState()
    state.update(
        "encoder_outputs",
        "text",
        {
            "inputs_embeds": torch.tensor(
                [[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]],
                device=device,
            ),
        },
    )
    state.update(
        "modality_inputs",
        "text",
        {"input_ids": torch.tensor([[10, 99, 20, 99]], device=device)},
    )
    state.update(
        "adapter_outputs",
        "vision",
        {
            "embeddings": torch.arange(
                modality_rows * 2,
                dtype=torch.float32,
                device=device,
            ).reshape(modality_rows, 2),
        },
    )
    return model, state
