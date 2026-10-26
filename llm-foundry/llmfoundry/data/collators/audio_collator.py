import torch

from typing import Dict, List, Mapping, Optional

from llmfoundry.data.collators.base_collator import BaseCollator
from llmfoundry.data.processors.base_processor import DataModality


def collate_spectrograms(
    examples: List[Optional[Dict[str, torch.Tensor]]],
) -> Dict[str, torch.Tensor]:
    """Concatenate the clips of `examples` into one conformer bundle.

    Returns:
        `spectrograms` [sum(T), n_mels], `spectrogram_lengths` [n_clips] and
        `num_spectrograms_per_sample` [len(examples)]; {} when no example
        carries a spectrogram.
    """
    with_spectrograms = [
        example for example in examples if example and "spectrograms" in example
    ]

    if not with_spectrograms:
        return {}

    return {
        "spectrograms": torch.cat(
            [example["spectrograms"] for example in with_spectrograms]
        ),
        "spectrogram_lengths": torch.cat(
            [example["spectrogram_lengths"] for example in with_spectrograms]
        ),
        "num_spectrograms_per_sample": torch.tensor(
            [
                len(example["spectrogram_lengths"])
                if example and "spectrograms" in example
                else 0
                for example in examples
            ],
            dtype=torch.long,
        ),
    }


def split_spectrograms(
    modality_batch: Mapping[str, torch.Tensor], microbatch_size: int
) -> List[Dict[str, torch.Tensor]]:
    """Slice a `collate_spectrograms` bundle into one bundle per microbatch.

    A microbatch without a clip gets {}. `num_spectrograms_per_sample` stays in
    the slices so that a microbatch can be split again (dynamic SP splits the
    device batch twice).
    """
    num_spectrograms_per_sample_split = modality_batch[
        "num_spectrograms_per_sample"
    ].split(microbatch_size)
    spectrogram_lengths_split = modality_batch["spectrogram_lengths"].split(
        [n.sum().item() for n in num_spectrograms_per_sample_split]
    )
    spectrograms_split = modality_batch["spectrograms"].split(
        [lengths.sum().item() for lengths in spectrogram_lengths_split]
    )

    return [
        {
            "spectrograms": spectrograms,
            "spectrogram_lengths": spectrogram_lengths,
            "num_spectrograms_per_sample": num_spectrograms_per_sample,
        }
        if len(spectrogram_lengths)
        else {}
        for spectrograms, spectrogram_lengths, num_spectrograms_per_sample in zip(
            spectrograms_split,
            spectrogram_lengths_split,
            num_spectrograms_per_sample_split,
        )
    ]


class AudioCollator(BaseCollator):
    data_modality = DataModality.AUDIO

    def __call__(
        self, examples: List[Optional[Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        batch = collate_spectrograms(examples)

        if batch:
            batch["processor_fallback_count"] = sum(
                example["processor_fallback_count"] if example else 0
                for example in examples
            )

        return batch

    @classmethod
    def split_batch(cls, batch: Mapping, microbatch_size: int):
        assert batch["input_ids"].shape[0] % microbatch_size == 0, (
            f"Batch size {batch['input_ids'].shape[0]} is not divisible by "
            f"microbatch size {microbatch_size}!"
        )

        num_microbatches = batch["input_ids"].shape[0] // microbatch_size

        data_modality_str = cls.data_modality.value
        if data_modality_str not in batch["modality_inputs"]:
            return [{} for _ in range(num_microbatches)]

        return [
            {data_modality_str: microbatch} if microbatch else {}
            for microbatch in split_spectrograms(
                batch["modality_inputs"][data_modality_str], microbatch_size
            )
        ]
