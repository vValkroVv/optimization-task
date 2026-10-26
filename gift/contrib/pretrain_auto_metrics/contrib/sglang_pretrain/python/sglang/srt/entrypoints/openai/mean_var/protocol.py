"""Checkpoint protocol validation and exact marker-prefix preparation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from transformers import PreTrainedTokenizerBase

from . import SIGMA_FLOOR
from .formatting import render_reward_content, validate_reward_template

SCHEMA = "gemma-mean-var-v1"
LEGACY_TOKENIZATION = {
    "chat_template_kwargs": {"bos_token": "<bos>", "preserve_thinking": False},
    "generation_suffix": "<|channel>thought\n<channel|>",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol(checkpoint: Path) -> dict:
    manifest = json.loads((checkpoint / "export_manifest.json").read_text())
    if not manifest["complete"] or sha256(checkpoint / "mean_var_protocol.json") != manifest["protocol_sha256"]:
        raise ValueError("incomplete export or modified protocol")
    protocol = json.loads((checkpoint / "mean_var_protocol.json").read_text())
    if protocol["schema"] != SCHEMA or protocol["cdf"] != "logistic" or protocol["rho"] != 0:
        raise ValueError("unsupported Mean-Var inference protocol")
    if not math.isfinite(protocol["tie_margin"]) or protocol["tie_margin"] <= 0:
        raise ValueError("invalid learned D")
    if protocol["sigma_floor"] != SIGMA_FLOOR or protocol["probability_floor"] != 0.05:
        raise ValueError("unsupported Mean-Var floors")
    if not {"tokenizer.json", "chat_template.jinja"} <= protocol["runtime_sha256"].keys():
        raise ValueError("Mean-Var requires hashed tokenizer.json and chat_template.jinja")
    for name, digest in protocol["runtime_sha256"].items():
        if Path(name).name != name or sha256(checkpoint / name) != digest:
            raise ValueError(f"checkpoint protocol hash mismatch: {name}")
    validate_reward_template(protocol["user_template"])
    try:
        protocol["user_template"].format(DOMAIN="", QUESTION="", RESPONSE="")
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(f"invalid reward template: {exc}") from exc
    if "tokenization" not in protocol:
        if "config.json" not in protocol["runtime_sha256"]:
            raise ValueError("Legacy Mean-Var tokenization requires hashed config.json")
        model_type = json.loads((checkpoint / "config.json").read_text()).get("model_type")
        if model_type in {"gemma4", "gemma4_text"}:
            protocol["tokenization"] = LEGACY_TOKENIZATION
        elif model_type == "qwen3_5":
            protocol["tokenization"] = {
                "chat_template_kwargs": {},
                "generation_suffix": "<|im_start|>assistant\n<think>\n\n</think>\n\n",
            }
        else:
            raise ValueError(f"Unsupported legacy Mean-Var model_type: {model_type!r}")
    tokenization = protocol["tokenization"]
    if not isinstance(tokenization, dict) or set(tokenization) != {"chat_template_kwargs", "generation_suffix"}:
        raise ValueError("invalid Mean-Var tokenization profile")
    kwargs = tokenization["chat_template_kwargs"]
    if (
        not isinstance(kwargs, dict)
        or set(kwargs) - {"bos_token", "preserve_thinking"}
        or ("bos_token" in kwargs and not isinstance(kwargs["bos_token"], str))
        or kwargs.get("preserve_thinking", False) is not False
        or not isinstance(tokenization["generation_suffix"], str)
        or not tokenization["generation_suffix"]
    ):
        raise ValueError("unsupported Mean-Var tokenization profile")
    return protocol


def marker_prefix(
    tokenizer: PreTrainedTokenizerBase,
    protocol: dict,
    question: str,
    response: str,
    domain: str = "",
) -> list[int]:
    content = render_reward_content(
        protocol["user_template"],
        domain=domain,
        question=question,
        response=response,
        marker=protocol["marker"],
    )
    marker_ids = tokenizer.encode(protocol["marker"], add_special_tokens=False)
    if not marker_ids:
        raise ValueError("verdict marker must tokenize to at least one token")
    tokenization = protocol["tokenization"]
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": protocol["system_prompt"]},
            {"role": "user", "content": content},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
        tools=None,
        **tokenization["chat_template_kwargs"],
    )
    if not rendered.endswith(tokenization["generation_suffix"]):
        raise ValueError("Chat template emitted an unexpected generation prefix")
    input_ids = tokenizer.encode(rendered, add_special_tokens=False)
    marker_positions = [
        index
        for index in range(len(input_ids) - len(marker_ids) + 1)
        if input_ids[index : index + len(marker_ids)] == marker_ids
    ]
    if not marker_positions:
        raise ValueError("input lacks the reward marker")
    # Enforce the original length policy BEFORE slicing the chat suffix.
    if len(input_ids) > protocol["max_length"]:
        raise ValueError(
            f"input exceeds frozen max_length={protocol['max_length']}; truncation is forbidden"
        )
    return input_ids[: marker_positions[-1] + len(marker_ids)]
