import pytest

from node_runner.src.pam_proxies.models.sglang import (
    SGLangGenerateRequest,
    SamplingParams,
)

pytestmark = pytest.mark.unit


class TestSamplingParamsValidation:
    def test_only_one_structured_constraint_is_allowed(self):
        with pytest.raises(
            ValueError, match="Only one of json_schema, regex, ebnf may be set."
        ):
            SamplingParams(json_schema="{}", regex="x")

    def test_min_new_tokens_cannot_exceed_max_new_tokens(self):
        with pytest.raises(
            ValueError, match="min_new_tokens cannot exceed max_new_tokens."
        ):
            SamplingParams(max_new_tokens=2, min_new_tokens=3)


class TestSGLangGenerateRequestValidation:
    def test_exactly_one_input_must_be_provided_none_case(self):
        with pytest.raises(
            ValueError,
            match="Exactly one of text, input_ids, input_embeds must be provided.",
        ):
            SGLangGenerateRequest()

    def test_exactly_one_input_must_be_provided_multiple_case(self):
        with pytest.raises(
            ValueError,
            match="Exactly one of text, input_ids, input_embeds must be provided.",
        ):
            SGLangGenerateRequest(text="hello", input_ids=[1, 2, 3])

    def test_exactly_one_input_accepts_text(self):
        req = SGLangGenerateRequest(text="hello")
        assert req.text == "hello"
        assert req.input_ids is None
        assert req.input_embeds is None
