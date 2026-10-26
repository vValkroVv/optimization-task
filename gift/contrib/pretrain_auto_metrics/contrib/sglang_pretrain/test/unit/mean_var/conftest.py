import json

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from sglang.srt.entrypoints.openai.mean_var.protocol import sha256


@pytest.fixture(autouse=True)
def deterministic_environment(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.delenv("GEPA_MEAN_VAR_BACKBONE", raising=False)


@pytest.fixture
def protocol():
    return {
        "schema": "gemma-mean-var-v1",
        "cdf": "logistic",
        "rho": 0,
        "tie_margin": 0.4820893704891205,
        "sigma_floor": 0.1,
        "probability_floor": 0.05,
        "runtime_sha256": {},
        "max_length": 1000,
        "system_prompt": "judge",
        "user_template": "{QUESTION} {RESPONSE}",
        "marker": "<REWARD>",
        "tokenization": {
            "chat_template_kwargs": {"bos_token": "<bos>", "preserve_thinking": False},
            "generation_suffix": "<|channel>thought\n<channel|>",
        },
    }


@pytest.fixture
def bundle(tmp_path, protocol):
    tokenizer = Tokenizer(
        models.WordLevel(
            {
                word: i
                for i, word in enumerate(
                    [
                        "[UNK]",
                        "<bos>",
                        "judge",
                        "question",
                        "answer",
                        "<REWARD>",
                        "<|channel>thought",
                        "<channel|>",
                    ]
                )
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, bos_token="<bos>")
    tokenizer.chat_template = (
        "{{ bos_token }} {{ messages[0].content }} {{ messages[1].content }} "
        "{% if enable_thinking %}BAD{% else %}<|channel>thought\n<channel|>{% endif %}"
    )
    tokenizer.save_pretrained(tmp_path)
    protocol["runtime_sha256"] = {
        name: sha256(tmp_path / name)
        for name in (
            "tokenizer.json",
            "chat_template.jinja",
        )
    }
    (tmp_path / "mean_var_protocol.json").write_text(json.dumps(protocol))
    (tmp_path / "export_manifest.json").write_text(
        json.dumps(
            {
                "complete": True,
                "protocol_sha256": sha256(tmp_path / "mean_var_protocol.json"),
            }
        )
    )
    return tmp_path


@pytest.fixture
def tokenizer(bundle):
    return AutoTokenizer.from_pretrained(bundle, local_files_only=True)
