import pytest

pytestmark = pytest.mark.unit

from sglang.srt.entrypoints.openai.mean_var.protocol import load_protocol, marker_prefix


def test_explicit_gemma_profile_preserves_legacy_tokens(tokenizer, protocol):
    legacy = marker_prefix(tokenizer, protocol, "question <REWARD>", "answer")
    protocol["tokenization"] = {
        "chat_template_kwargs": {"bos_token": "<bos>", "preserve_thinking": False},
        "generation_suffix": "<|channel>thought\n<channel|>",
    }
    assert marker_prefix(tokenizer, protocol, "question <REWARD>", "answer") == legacy


def test_qwen_profile_uses_its_template_without_gemma_bos(tokenizer, protocol):
    tokenizer.chat_template = (
        "{{ messages[0].content }} {{ messages[1].content }} "
        "{% if enable_thinking %}BAD{% else %}<|im_start|>assistant\n<think>\n\n</think>\n\n{% endif %}"
    )
    protocol["tokenization"] = {
        "chat_template_kwargs": {},
        "generation_suffix": "<|im_start|>assistant\n<think>\n\n</think>\n\n",
    }
    assert marker_prefix(tokenizer, protocol, "question", "answer") == [2, 3, 4, 5]


def test_reasoning_disabled(tokenizer, protocol):
    assert marker_prefix(tokenizer, protocol, "question", "answer") == [1, 2, 3, 4, 5]


def test_missing_bundle_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_protocol(tmp_path)


def test_missing_marker_rejected(tokenizer, protocol):
    tokenizer.chat_template = (
        "{{ bos_token }} judge answer <|channel>thought\n<channel|>"
    )
    with pytest.raises(ValueError, match="lacks the reward marker"):
        marker_prefix(tokenizer, protocol, "question", "answer")


def test_empty_marker_rejected(tokenizer, protocol):
    protocol["marker"] = ""
    with pytest.raises(ValueError, match="at least one token"):
        marker_prefix(tokenizer, protocol, "question", "answer")


def test_unexpected_template_suffix_rejected(tokenizer, protocol):
    tokenizer.chat_template = "bad"
    with pytest.raises(ValueError, match="unexpected generation prefix"):
        marker_prefix(tokenizer, protocol, "question", "answer")
