from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict


class _V2Model(BaseModel):
    # Permissive by design: the devices-v2 wire contract may carry extra
    # keys we don't model; we never want to reject those.
    model_config = ConfigDict(extra="allow")


# =========================
# Content parts
# =========================
#
# devices-v2 represents a turn as a list of heterogeneous "parts". We do NOT
# use a discriminated union here: the parts are distinguished by which key is
# present ({"text"}, {"function_call"}, {"function_result"}) rather than a
# "type" tag, and keeping them as plain permissive dicts avoids colliding with
# the OpenAI ContentPart discriminator union in openai.py. Messages therefore
# keep `content` as list[dict] and the adapter inspects the keys directly.


class V2FunctionCall(_V2Model):
    name: str
    # devices-v2 carries arguments already as a dict (not a JSON string).
    arguments: Dict[str, Any] = {}


class V2FunctionResult(_V2Model):
    name: str
    result: Any = None


class V2TextPart(_V2Model):
    text: str


class V2FunctionCallPart(_V2Model):
    function_call: V2FunctionCall


class V2FunctionResultPart(_V2Model):
    function_result: V2FunctionResult


# =========================
# Messages
# =========================


class V2Message(_V2Model):
    role: str
    # List of part dicts ({text}/{function_call}/{function_result}); kept as
    # raw dicts so the adapter can branch on the present key without fighting a
    # discriminator. A user/system turn is typically [{"text": "..."}].
    content: List[Dict[str, Any]] = []
    # Per-message subset of available functions; ignored by the v5 serving
    # template (it re-emits a global namespace from `functions`), preserved for
    # round-trip fidelity only.
    function_names: Optional[List[str]] = None


# =========================
# Functions (tool descriptions)
# =========================


class V2Function(_V2Model):
    # build_gigachat_v2 emits each function as a flat dict carrying role/name
    # plus the schema fields the v5 jinja consumes.
    role: str = "assistant"
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    return_parameters: Optional[Dict[str, Any]] = None
    few_shot_examples: Optional[List[Any]] = None


# =========================
# Request
# =========================


class DevicesV2ChatCompletionRequest(_V2Model):
    messages: List[V2Message]
    functions: Optional[List[V2Function]] = None

    model: Optional[str] = None
    routing_key: Optional[str] = None

    # Sampling / generation fields mirrored from the OpenAI/SGLang request so
    # that ProxyService._build_sampling_params / _inject_* helpers work against
    # the converted body unchanged.
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    priority: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stop_token_ids: Optional[List[int]] = None

    chat_template_kwargs: Optional[Dict[str, Any]] = None
