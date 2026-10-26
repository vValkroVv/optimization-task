import json

import pytest

pytestmark = pytest.mark.unit

from sglang.srt.entrypoints.openai.mean_var.protocol import load_protocol, marker_prefix, sha256


def test_valid_bundle_loads(bundle, protocol):
    assert load_protocol(bundle) == protocol


@pytest.mark.parametrize(
    "model_type", ["gemma4", "gemma4_text", "qwen3_5", "unknown", None]
)
def test_legacy_profile_uses_hashed_model_family(bundle, protocol, model_type):
    protocol.pop("tokenization")
    if model_type is not None:
        (bundle / "config.json").write_text(json.dumps({"model_type": model_type}))
        protocol["runtime_sha256"]["config.json"] = sha256(bundle / "config.json")
    (bundle / "mean_var_protocol.json").write_text(json.dumps(protocol))
    protocol_hash = sha256(bundle / "mean_var_protocol.json")
    manifest_path = bundle / "export_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["protocol_sha256"] = protocol_hash
    manifest_path.write_text(json.dumps(manifest))

    if model_type in (None, "unknown"):
        with pytest.raises(
            ValueError, match="Legacy.*hashed config|Unsupported legacy"
        ):
            load_protocol(bundle)
        return

    loaded = load_protocol(bundle)
    expected = {
        "chat_template_kwargs": {"bos_token": "<bos>", "preserve_thinking": False},
        "generation_suffix": "<|channel>thought\n<channel|>",
    }
    if model_type == "qwen3_5":
        expected = {
            "chat_template_kwargs": {},
            "generation_suffix": "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        }
    assert loaded["tokenization"] == expected
    assert sha256(bundle / "mean_var_protocol.json") == protocol_hash


@pytest.mark.parametrize(
    "field,value",
    [
        ("user_template", "{QUESTION}"),
        ("user_template", "{QUESTION} {RESPONSE} {role_a}"),
        ("user_template", "{QUESTION} {RESPONSE} {UNKNOWN}"),
        ("user_template", "{QUESTION} {RESPONSE} {}"),
        ("user_template", "{QUESTION} {RESPONSE} {"),
        ("cdf", "normal"),
        ("rho", 0.1),
        ("schema", "unknown"),
        ("tie_margin", 0),
        ("tie_margin", float("nan")),
        ("sigma_floor", 0.2),
        ("probability_floor", 0.1),
        ("runtime_sha256", {"../escape": "bad"}),
        ("tokenization", {}),
        ("tokenization", {"chat_template_kwargs": {"enable_thinking": True}, "generation_suffix": "suffix"}),
        ("tokenization", {"chat_template_kwargs": {"preserve_thinking": True}, "generation_suffix": "suffix"}),
        ("tokenization", {"chat_template_kwargs": {}, "generation_suffix": ""}),
    ],
)
def test_unsupported_protocol_rejected(bundle, protocol, field, value):
    protocol[field] = value
    (bundle / "mean_var_protocol.json").write_text(json.dumps(protocol))
    (bundle / "export_manifest.json").write_text(
        json.dumps(
            {
                "complete": True,
                "protocol_sha256": sha256(bundle / "mean_var_protocol.json"),
            }
        )
    )
    with pytest.raises(ValueError):
        load_protocol(bundle)


def test_modified_protocol_rejected(bundle):
    with (bundle / "mean_var_protocol.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="modified protocol"):
        load_protocol(bundle)


def test_modified_tokenizer_rejected(bundle):
    with (bundle / "tokenizer.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_protocol(bundle)


@pytest.mark.parametrize("filename", ["tokenizer.json", "chat_template.jinja"])
def test_missing_tokenizer_hash_rejected(bundle, protocol, filename):
    protocol["runtime_sha256"].pop(filename)
    (bundle / "mean_var_protocol.json").write_text(json.dumps(protocol))
    manifest_path = bundle / "export_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["protocol_sha256"] = sha256(bundle / "mean_var_protocol.json")
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="requires hashed"):
        load_protocol(bundle)


def test_incomplete_bundle_rejected(bundle):
    (bundle / "export_manifest.json").write_text('{"complete": false}')
    with pytest.raises(ValueError, match="incomplete export"):
        load_protocol(bundle)


def test_prefix_ends_at_last_marker(tokenizer, protocol):
    prefix = marker_prefix(tokenizer, protocol, "question <REWARD>", "answer")
    assert prefix == [1, 2, 3, 5, 4, 5]


def test_full_length_checked_before_slicing(tokenizer, protocol):
    protocol["max_length"] = 5
    with pytest.raises(ValueError, match="truncation is forbidden"):
        marker_prefix(tokenizer, protocol, "question", "answer")


def test_no_domain_template_ignores_domain(tokenizer, protocol):
    assert marker_prefix(tokenizer, protocol, "question", "answer", "math") == marker_prefix(
        tokenizer,
        protocol,
        "question",
        "answer",
        "biology",
    )
