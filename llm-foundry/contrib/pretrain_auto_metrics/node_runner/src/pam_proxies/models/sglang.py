from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SGLBaseModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class SamplingParams(SGLBaseModel):
    # Core generation params
    max_new_tokens: int = Field(default=128, ge=0)
    stop: Optional[Union[str, List[str]]] = None
    stop_token_ids: Optional[List[int]] = None
    stop_regex: Optional[Union[str, List[str]]] = None

    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None

    # Penalties
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    repetition_penalty: float = Field(default=1.0, ge=0.0, le=2.0)
    min_new_tokens: int = Field(default=0, ge=0)

    # Structured / constrained decoding
    json_schema: Optional[str] = None
    regex: Optional[str] = None
    ebnf: Optional[str] = None
    structural_tag: Optional[str] = None

    # Other decoding options
    n: int = Field(default=1, ge=1)
    ignore_eos: bool = False
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    no_stop_trim: bool = False

    # Docs show batch-capable shape for custom params; allow both single and batch form
    custom_params: Optional[Union[Dict[str, Any], List[Optional[Dict[str, Any]]]]] = (
        None
    )

    @model_validator(mode="after")
    def validate_constraints(self) -> "SamplingParams":
        constrained = [
            self.json_schema is not None,
            self.regex is not None,
            self.ebnf is not None,
        ]
        if sum(constrained) > 1:
            raise ValueError("Only one of json_schema, regex, ebnf may be set.")
        if self.min_new_tokens > self.max_new_tokens:
            raise ValueError("min_new_tokens cannot exceed max_new_tokens.")
        return self


ImageDataItem = Union[str, Dict[str, Any]]
AudioDataItem = Union[str, Dict[str, Any]]


class SGLangGenerateRequest(SGLBaseModel):
    # Exactly one of these three must be provided
    text: Optional[Union[str, List[str]]] = None
    input_ids: Optional[Union[List[int], List[List[int]]]] = None
    input_embeds: Optional[Union[List[List[float]], List[List[List[float]]]]] = None

    # Multimodal inputs
    image_data: Optional[
        Union[
            ImageDataItem,
            List[ImageDataItem],
            List[List[ImageDataItem]],
        ]
    ] = None
    audio_data: Optional[Union[AudioDataItem, List[AudioDataItem]]] = None

    # Generation config
    sampling_params: Optional[Union[SamplingParams, List[SamplingParams]]] = None

    # Request id
    rid: Optional[Union[str, List[str]]] = None

    # Logprobs
    return_logprob: Optional[Union[bool, List[bool]]] = None
    logprob_start_len: Optional[Union[int, List[int]]] = None
    top_logprobs_num: Optional[Union[int, List[int]]] = None
    token_ids_logprob: Optional[Union[List[int], List[List[int]]]] = None
    return_text_in_logprobs: bool = False

    # Streaming
    stream: bool = False

    # LoRA / custom processor
    lora_path: Optional[Union[str, List[Optional[str]]]] = None
    custom_logit_processor: Optional[Union[str, List[Optional[str]]]] = None

    # Extra return payloads
    return_hidden_states: Union[bool, List[bool]] = False
    return_routed_experts: bool = False

    @model_validator(mode="after")
    def validate_exactly_one_input(self) -> "SGLangGenerateRequest":
        provided = sum(
            value is not None
            for value in (self.text, self.input_ids, self.input_embeds)
        )
        if provided != 1:
            raise ValueError(
                "Exactly one of text, input_ids, input_embeds must be provided."
            )
        return self


class SGLangGenerateFinishReason(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    matched: Optional[str | int] = None
    length: Optional[int] = None


class SGLangGenerateMetaInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    finish_reason: SGLangGenerateFinishReason
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    e2e_latency: Optional[float] = None


class SGLangGenerateResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    text: str
    output_ids: Optional[list[int]] = None
    meta_info: SGLangGenerateMetaInfo


class SGLangGenerateStreamChunk(BaseModel):
    model_config = ConfigDict(extra="allow")

    text: str
    meta_info: SGLangGenerateMetaInfo
