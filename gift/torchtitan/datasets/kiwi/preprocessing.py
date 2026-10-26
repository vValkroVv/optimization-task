import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Self, cast

import numpy as np
import torch
from einops import rearrange
from torch import nn

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.datasets.streaming.text.transforms import FIMTransform
from .configs import PreprocessorConfig
from .constants import (
    AUXILIARY_TOKENS,
    BOS_TOKEN,
    DIFFUSION_GENERATION,
    ENCODER_MAPPING,
    ENCODER_ORDER,
    EOS_TOKEN,
    FLASH_ATTN_BLOCK_SIZE,
    PAD_TOKEN,
    TIMESTEP_TOKEN,
    DataModality,
)
from .position_projector import PositionProjector3d


class TargetMask:
    @classmethod
    def empty(cls, data: torch.Tensor) -> torch.Tensor:
        return torch.zeros(data.shape[0], dtype=torch.bool)

    @classmethod
    def causal(cls, data: torch.Tensor) -> torch.Tensor:
        target_mask = torch.ones(data.shape[0], dtype=torch.bool)
        target_mask[0] = 0
        return target_mask

    @classmethod
    def full(cls, data: torch.Tensor) -> torch.Tensor:
        return torch.ones(data.shape[0], dtype=torch.bool)


@dataclass
class DataClass:
    tokens: torch.Tensor
    index: torch.Tensor
    mask: Literal[0, 1, 2]
    position: torch.Tensor
    target_mask: torch.Tensor
    q_len: torch.Tensor | int | None = None
    is_padding: bool = False
    span_ids: torch.Tensor = field(init=False)

    def offset(self, offset: int | torch.Tensor) -> None:
        """Offset indexes."""
        self.index = self.index + offset

    def offset_position(self, offset: int | torch.Tensor) -> None:
        """Offset position for PE."""
        if not self.is_padding:
            self.position = PositionProjector3d.offset(self.position, offset)

    def offset_span_ids(self, offset: int | torch.Tensor) -> None:
        self.span_ids = self.span_ids + offset

    def __len__(
        self,
    ) -> int:
        return self.index.shape[0]

    def position_length(
        self,
    ) -> int:
        return int((self.position[..., 0][-1] - self.position[..., 0][0]).item()) + 1

    def add(self, data: Self) -> None:
        assert self.mask == data.mask
        assert self.is_padding == data.is_padding
        assert (self.q_len is None) == (data.q_len is None)
        assert self.tokens.shape[1:] == data.tokens.shape[1:]

        self.tokens = torch.concat([self.tokens, data.tokens], dim=0)
        offset_index = self.index[-1] + 1 - data.index[0]
        offset_pe = self.position[..., 0][-1] + 1 - data.position[..., 0][0]
        data.offset(offset_index)
        data.offset_position(offset_pe)

        self.index = torch.concat([self.index, data.index], dim=0)
        self.position = torch.concat([self.position, data.position], dim=0)
        self.target_mask = torch.concat([self.target_mask, data.target_mask])
        self.span_ids = torch.concat([self.span_ids, data.span_ids], dim=0)

        if isinstance(self.q_len, torch.Tensor) and isinstance(
            data.q_len, torch.Tensor
        ):
            self.q_len = torch.concat([self.q_len, data.q_len], dim=0)


@dataclass
class SampleRef:
    """Checkpoint cursor for one consumed stream index."""

    idx: int | None = None
    source_dataset: str = ""
    source_stream: str = ""


class SampleClass:
    def __init__(
        self,
        type_of_content: str = "",
    ):
        self.type_of_content = type_of_content

        self.text: list[DataClass] = []
        self.semantic_visual: list[DataClass] = []
        self.semantic_audio: list[DataClass] = []
        self.pixel_visual: list[DataClass] = []
        self.pixel_audio: list[DataClass] = []
        self.gen_visual: list[DataClass] = []
        self.gen_audio: list[DataClass] = []

        self.length: int = 0
        self.n_items: int = 0

        self.last_position: int = 0

        self.mapping = ENCODER_MAPPING
        self.last_encoder_type: str | None = None

        self.mask_type: list[list[int]] = [
            [] for _ in ENCODER_ORDER
        ]  # for mask creatio
        self.item_length: list[list[int]] = [
            [] for _ in ENCODER_ORDER
        ]  # for mask creation
        self.ref_edges: list[torch.Tensor] = []  # for mask creation
        self.ref_offset: torch.Tensor = torch.full((len(ENCODER_ORDER),), float(0))
        self.dst: torch.Tensor = torch.full((len(ENCODER_ORDER), 0), float("nan"))
        self.n_spans = 0
        self.span_properties: list[set[str]] = []

    def _add_item_for_mask(self, data: DataClass, encoder_type: str) -> None:
        idx = ENCODER_ORDER.index(encoder_type)

        src = torch.full((len(ENCODER_ORDER), 1), float("nan"))
        src[idx] = self.ref_offset[idx]

        edge = torch.concat(
            [src.expand_as(self.dst).unsqueeze(0), self.dst.unsqueeze(0)], dim=0
        )
        self.ref_edges.append(edge)
        self.dst = torch.concat([self.dst, src], dim=-1)

        self.ref_offset[idx] += 1
        self.mask_type[idx].append(data.mask)
        self.item_length[idx].append(len(data))

    def get_ref_edges(
        self,
    ) -> torch.Tensor:
        return torch.concat(self.ref_edges, dim=-1)

    def get_ref_offset(
        self,
    ) -> torch.Tensor:
        return self.ref_offset

    def add(
        self,
        data: DataClass,
        data_type: DataModality,
        properties: set[str] | None = None,
    ) -> None:
        if self.item_length is None:
            raise ValueError("Item_length not initialized")

        if len(data) == 0:
            return

        data.span_ids = torch.full((len(data),), self.n_spans, dtype=torch.long)
        self.span_properties.append(properties or set())
        self.n_spans += 1

        encoder_type: str = self.mapping[data_type]
        if encoder_type == self.last_encoder_type == "text":
            encoder_block_type = getattr(self, encoder_type)
            encoder_block_type[-1].add(data)
            self.item_length[ENCODER_ORDER.index(encoder_type)][-1] += len(data)
        else:
            self._add_item_for_mask(data, encoder_type)  # [num_edges, 2]

            self.n_items += 1
            data.offset(self.length)
            data.offset_position(self.last_position)
            encoder_block_type = getattr(self, encoder_type)
            encoder_block_type.append(data)

        self.length += len(data)
        if not data.is_padding:
            self.last_position = int(data.position.max()) + 1

        self.last_encoder_type = encoder_type

    def __len__(
        self,
    ) -> int:
        return self.length


class ModalityPreprocessor(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> DataClass: ...

    def get_padding(self, size: int) -> DataClass: ...


class TextPreprocessor:
    SPECIAL_TOKENS = [
        x for stend in AUXILIARY_TOKENS.values() for x in stend if len(x) > 0
    ] + [
        PAD_TOKEN,
        TIMESTEP_TOKEN,
    ]

    @staticmethod
    def _resolve_pad_token_id(tokenizer: HuggingFaceTokenizer) -> int:
        hf_config = getattr(tokenizer, "_hf_config", None) or {}
        pad_token = hf_config.get("pad_token")
        if isinstance(pad_token, dict):
            pad_token = pad_token.get("content")
        candidates = (
            [pad_token, PAD_TOKEN] if isinstance(pad_token, str) else [PAD_TOKEN]
        )
        for token in candidates:
            pad_id = tokenizer.token_to_id(token)
            if pad_id is not None:
                return pad_id
        raise ValueError(
            "Pad token not found in tokenizer vocabulary "
            f"(tried tokenizer_config pad_token and {PAD_TOKEN!r})"
        )

    def __init__(
        self,
        tokenizer_path: str,
        max_seq_len: int | None = None,
        *,
        enable_syntax_aware_fim: bool = False,
    ) -> None:
        self.tokenizer = HuggingFaceTokenizer(tokenizer_path=tokenizer_path)
        self._max_seq_len = max_seq_len
        self._enable_syntax_aware_fim = enable_syntax_aware_fim
        self._fim_transform: FIMTransform | None = None
        self._pad_token_id = self._resolve_pad_token_id(self.tokenizer)

    def _resolve_fim_rates(self, fim_rates: dict[str, float]) -> dict[str, float]:
        rates = dict(fim_rates)
        if not self._enable_syntax_aware_fim:
            rates["syntax_aware_fim"] = 0.0
        return rates

    def _ensure_fim_transform(self) -> FIMTransform:
        if self._fim_transform is None:
            if self._max_seq_len is None or self._max_seq_len <= 0:
                raise ValueError(
                    "preprocessor.max_seq_len must be positive to enable FIM"
                )
            init_rates = (
                {"syntax_aware_fim": 1.0} if self._enable_syntax_aware_fim else {}
            )
            self._fim_transform = FIMTransform(
                tokenizer=self.tokenizer,
                max_seq_len=self._max_seq_len,
                seed=0,
                default_fim_rates=init_rates,
            )
        return self._fim_transform

    def __call__(
        self,
        text: str | torch.Tensor,
        fim_rates: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> DataClass:
        if isinstance(text, str):
            if len(text) == 0:
                tokens = torch.tensor([])
            else:
                token_ids = self.tokenizer.encode(
                    text,
                    add_bos=False,
                    add_eos=False,
                    add_special_tokens=False,
                )
                tokens = torch.as_tensor(token_ids, dtype=torch.int64)
        else:
            tokens = text

        if fim_rates is not None and sum(fim_rates.values()) > 0:
            rates = self._resolve_fim_rates(fim_rates)
            rng = np.random.default_rng(int(np.random.randint(0, 2**63)))
            tokens = self._ensure_fim_transform().transform(
                tokens,
                fim_rates=rates,
                rng=rng,
            )

        sample = DataClass(
            tokens=tokens,
            index=torch.arange(len(tokens)),
            mask=1,
            q_len=None,
            position=PositionProjector3d.process(tokens),
            is_padding=False,
            target_mask=TargetMask.full(tokens),
        )
        return sample

    def get_padding(self, size: int) -> DataClass:
        tokens = torch.ones(size, dtype=torch.int64) * self._pad_token_id
        return DataClass(
            tokens=tokens,
            index=torch.arange(size),
            mask=0,
            q_len=None,
            position=PositionProjector3d.process(tokens),
            is_padding=True,
            target_mask=TargetMask.empty(tokens),
        )


class VisionPreprocessor:
    def __init__(
        self,
        patch_size: int,
        max_len: None | int = None,
        num_channels: int = 3,
    ):
        self.patch_size = patch_size
        self.max_len = max_len
        self.num_channels = num_channels
        self.padding_size = FLASH_ATTN_BLOCK_SIZE

    def __call__(self, visual: torch.Tensor, **kwargs: Any) -> DataClass:
        """
        input: t, h, w, c
        output: t, h//p, w//p, c, p*p
        """

        if self.max_len:
            visual = visual[: self.max_len]

        visual = rearrange(
            visual,
            "t (h p1) (w p2) c -> t h w (p1 p2) c",
            p1=self.patch_size,
            p2=self.patch_size,
        )
        t, h, w, c, _ = visual.shape

        visual = visual.flatten(-2, -1)
        position = PositionProjector3d.process(visual)

        visual = visual.flatten(0, 2)
        position = position.flatten(0, 2)

        block_size = visual.shape[0] // t

        sample = DataClass(
            tokens=visual,
            index=torch.arange(visual.shape[0]),
            mask=2,
            q_len=torch.arange(0, visual.shape[0], block_size),
            position=position,
            is_padding=False,
            target_mask=TargetMask.full(visual),
        )
        return sample

    def get_padding(self, size: int = 1) -> DataClass:
        tokens = torch.zeros(
            (size, self.num_channels * self.patch_size * self.patch_size)
        )
        return DataClass(
            tokens=tokens,
            index=torch.arange(size),
            mask=0,
            q_len=None,
            position=PositionProjector3d.process(tokens),
            is_padding=True,
            target_mask=TargetMask.empty(tokens),
        )


class Siglip2ImagePreprocessor:
    def __init__(self, patch_size: int, max_size: int, num_channels: int = 3):
        self.patch_size = patch_size
        self.max_size = max_size
        self.num_channels = num_channels
        self.image_mean = (
            torch.tensor([0.5, 0.5, 0.5]).to(dtype=torch.float32, device="cpu") * 255.0
        )
        self.image_std = (
            torch.tensor([0.5, 0.5, 0.5]).to(dtype=torch.float32, device="cpu") * 255.0
        )
        self.resample = "bilinear"

    def convert_image_to_patches(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_channels, image_height, image_width = image.shape
        num_patches_height = image_height // self.patch_size
        num_patches_width = image_width // self.patch_size
        patched_image = image.reshape(
            num_channels,
            num_patches_height,
            self.patch_size,
            num_patches_width,
            self.patch_size,
        )
        patched_image = patched_image.permute(1, 3, 2, 4, 0)

        position = PositionProjector3d.process(
            patched_image.flatten(-3, -1).unsqueeze(0)
        ).flatten(0, 2)

        patched_image = patched_image.reshape(
            num_patches_height * num_patches_width, -1
        )
        return patched_image, position

    def get_image_size_for_max_size(
        self, image_height: int, image_width: int
    ) -> tuple[int, int]:
        h, w = image_height, image_width

        orig_area = h * w
        max_area = self.max_size**2

        if orig_area <= max_area:
            new_h = (h // self.patch_size) * self.patch_size
            new_w = (w // self.patch_size) * self.patch_size
            return max(new_h, self.patch_size), max(new_w, self.patch_size)

        r = w / h
        t = self.max_size / self.patch_size

        k = int(t / (r**0.5))
        k = max(k, 1)

        m = int(round(k * r))

        while k * m > t * t:
            m -= 1

        while m <= 0:
            k -= 1
            m = int(round(k * r))

        new_h = k * self.patch_size
        new_w = m * self.patch_size

        return new_h, new_w

    def __call__(
        self, visual_input: dict[str, torch.Tensor], **kwargs: Any
    ) -> DataClass:
        visual = visual_input["image"][0]
        height, width = self.get_image_size_for_max_size(
            image_height=visual.shape[1], image_width=visual.shape[2]
        )

        visual = nn.functional.interpolate(
            visual.unsqueeze(0), (height, width), mode=self.resample, antialias=True
        ).squeeze(0)

        visual = (
            visual.to(dtype=torch.float32) - self.image_mean[:, None, None]
        ) / self.image_std[:, None, None]
        visual, position = self.convert_image_to_patches(visual)

        block_size = visual.shape[0]

        sample = DataClass(
            tokens=visual,
            index=torch.arange(visual.shape[0]),
            mask=2,
            q_len=torch.arange(0, visual.shape[0], block_size),
            position=position,
            is_padding=False,
            target_mask=TargetMask.empty(visual),
        )

        return sample

    def get_padding(self, size: int = 1) -> DataClass:
        tokens = torch.zeros(
            (size, self.num_channels * self.patch_size * self.patch_size)
        )
        return DataClass(
            tokens=tokens,
            index=torch.arange(size),
            mask=0,
            q_len=None,
            position=PositionProjector3d.process(tokens),
            is_padding=True,
            target_mask=TargetMask.empty(tokens),
        )


class Preprocessor:
    aux_tokens = AUXILIARY_TOKENS

    def __init__(self, cfg: PreprocessorConfig):
        self.cfg = cfg

        self.prpr_mapper: dict[str, Any] = {}
        self.prpr_mapper["text"] = TextPreprocessor(
            self.cfg.tokenizer_path,
            self.cfg.max_seq_len,
            enable_syntax_aware_fim=self.cfg.enable_syntax_aware_fim,
        )

        if "semantic_image" in self.cfg.template:
            assert self.cfg.patch_size is not None
            self.prpr_mapper["semantic_image"] = Siglip2ImagePreprocessor(
                patch_size=self.cfg.patch_size, max_size=self.cfg.max_size
            )

        if "semantic_video" in self.cfg.template:
            assert self.cfg.patch_size is not None
            assert self.cfg.max_len is not None

            self.prpr_mapper["semantic_video"] = VisionPreprocessor(
                patch_size=self.cfg.patch_size, max_len=self.cfg.max_len
            )

        if "pixel_image" in self.cfg.template:
            assert self.cfg.patch_size_latent is not None

            self.prpr_mapper["pixel_image"] = VisionPreprocessor(
                patch_size=self.cfg.patch_size_latent, max_len=None
            )

        if "pixel_video" in self.cfg.template:
            assert self.cfg.max_len_latent is not None
            assert self.cfg.patch_size_latent is not None

            self.prpr_mapper["pixel_video"] = VisionPreprocessor(
                patch_size=self.cfg.patch_size_latent, max_len=self.cfg.max_len_latent
            )

        if "gen_image" in self.cfg.template:
            assert self.cfg.patch_size_latent is not None

            self.prpr_mapper["gen_image"] = VisionPreprocessor(
                patch_size=self.cfg.patch_size_latent, max_len=None
            )

        if "gen_video" in self.cfg.template:
            assert self.cfg.max_len_latent is not None
            assert self.cfg.patch_size_latent is not None

            self.prpr_mapper["gen_video"] = VisionPreprocessor(
                patch_size=self.cfg.patch_size_latent, max_len=self.cfg.max_len_latent
            )

    def _split_sample_by_text_template(
        self,
        layout: str,
        sample: dict[str, Any],
    ) -> tuple[list[str | None], dict[str, deque[Any]]]:
        """Docstring for _split_sample_by_text_template.

        :param self: Description
        :type sample: dict[str, Any]
        :return: split data in deque by type
        :rtype: dict[str, deque | list]
        """

        data: dict[str, deque[Any]] = {}
        pattern_raw = [re.escape(word) for word in self.aux_tokens]
        pattern_raw.sort(key=len, reverse=True)
        pattern: list[str] = pattern_raw
        joined_pattern: str = "|".join(pattern)
        parts = [
            p.strip(" ")
            for p in re.split(f"({joined_pattern})", layout)
            if len(p.strip()) > 0 and p
        ]

        clean_text = list(filter(lambda x: re.match(joined_pattern, x) is None, parts))

        parts = [
            p[1:-1] for p in parts
        ]  # [p[1:-1] if re.match("|".join(pattern), p) else None for p in parts]

        for k, v in sample.items():
            if isinstance(v, list):
                data[k] = deque(v)
            else:
                data[k] = deque([v])

        if len(clean_text) > 0:
            data["text"] = deque(clean_text)

        del sample

        return parts, data

    def __call__(
        self,
        layout: str,
        text: list[str | torch.Tensor] | None = None,
        semantic_image: list[torch.Tensor | dict[str, Any]] | None = None,
        semantic_video: list[torch.Tensor | dict[str, Any]] | None = None,
        semantic_audio: list[torch.Tensor | dict[str, Any]] | None = None,
        pixel_image: list[torch.Tensor | dict[str, Any]] | None = None,
        pixel_video: list[torch.Tensor | dict[str, Any]] | None = None,
        pixel_audio: list[torch.Tensor | dict[str, Any]] | None = None,
        gen_image: list[torch.Tensor | dict[str, Any]] | None = None,
        gen_video: list[torch.Tensor | dict[str, Any]] | None = None,
        gen_audio: list[torch.Tensor | dict[str, Any]] | None = None,
        properties: dict[str, list[set[str]]] | None = None,
        add_special_tokens: bool = True,
        **preprocessor_kwargs: Any,
    ) -> SampleClass:
        """
            layout contains text and tokens from constants/AUXILIARY_TOKENS.keys()
            e.g. "text": "pretty text <gen_image>".
            And all keys which are mentioned in text and included in constants/AUXILIARY_TOKENS.keys().

            Expl:
            layout =  'pretty text <gen_image>'
            gen_image = [image]

            layout = 'pretty text <semantic_image> <semantic_image> <gen_image>'
            gen_image = [image]
            semantic_image = [image_1, image_2]
        }

        """

        out = SampleClass()
        sample = {
            "text": text if text is not None else [],
            "semantic_image": semantic_image if semantic_image is not None else [],
            "semantic_video": semantic_video if semantic_video is not None else [],
            "semantic_audio": semantic_audio if semantic_audio is not None else [],
            "pixel_image": pixel_image if pixel_image is not None else [],
            "pixel_video": pixel_video if pixel_video is not None else [],
            "pixel_audio": pixel_audio if pixel_audio is not None else [],
            "gen_image": gen_image if gen_image is not None else [],
            "gen_video": gen_video if gen_video is not None else [],
            "gen_audio": gen_audio if gen_audio is not None else [],
        }

        parts, sample_splited = self._split_sample_by_text_template(
            layout,
            sample,
        )
        property_deques = (
            {k: deque(v) for k, v in properties.items()} if properties else {}
        )

        bos, eos = self._get_bos_eos()
        if add_special_tokens:
            bos.target_mask = TargetMask.empty(bos.tokens)
            out.add(bos, "text")

        for data_type in parts:
            assert data_type in sample_splited, f"Get bad data type {data_type}."
            try:
                data = sample_splited[data_type].popleft()
            except IndexError as e:
                raise ValueError(
                    f"layout asks for more {data_type} items than the sample provides"
                ) from e
            data_properties = (
                property_deques[data_type].popleft()
                if data_type in property_deques
                else set()
            )
            start_token, end_token = self._get_aux_tokens(data_type)
            start_token.target_mask = TargetMask.full(start_token.tokens)
            out.add(start_token, "text")

            if f"<{data_type}>" in DIFFUSION_GENERATION:
                dtimestep = self._get_dtimestep()
                dtimestep.target_mask = TargetMask.empty(dtimestep.tokens)
                out.add(dtimestep, "text")

            data = self.prpr_mapper[data_type](data, **preprocessor_kwargs)
            if data_type == "text" and not preprocessor_kwargs.get("text_loss"):
                data.target_mask = TargetMask.empty(data.tokens)
            out.add(data, cast(DataModality, data_type), properties=data_properties)

            end_token.target_mask = TargetMask.empty(end_token.tokens)
            out.add(end_token, "text")

        if add_special_tokens:
            eos.target_mask = TargetMask.full(eos.tokens)
            out.add(eos, "text")

        return out

    def _get_aux_tokens(self, data_type: str) -> tuple[DataClass, DataClass]:
        data_type = f"<{data_type}>"
        text_preprocessor: TextPreprocessor = self.prpr_mapper["text"]
        return (
            text_preprocessor(self.aux_tokens[data_type][0]),
            text_preprocessor(self.aux_tokens[data_type][1]),
        )

    def _get_bos_eos(self) -> tuple[DataClass, DataClass]:
        text_preprocessor: TextPreprocessor = self.prpr_mapper["text"]
        return (text_preprocessor(BOS_TOKEN), text_preprocessor(EOS_TOKEN))

    def _get_dtimestep(self) -> DataClass:
        text_preprocessor: TextPreprocessor = self.prpr_mapper["text"]
        return text_preprocessor(TIMESTEP_TOKEN)
