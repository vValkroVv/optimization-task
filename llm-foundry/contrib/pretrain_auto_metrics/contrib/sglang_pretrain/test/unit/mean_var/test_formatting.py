import pytest

pytestmark = pytest.mark.unit

from sglang.srt.entrypoints.openai.mean_var.formatting import render_reward_content, validate_reward_template


def test_with_domain_renders_exactly():
    assert (
        render_reward_content(
            "Domain: {DOMAIN}\n{QUESTION}\n{RESPONSE}  ", domain="math", question="q", response="a", marker="R"
        )
        == "Domain: math\nq\na\n\nR"
    )


@pytest.mark.parametrize("template", ["{QUESTION}", "{RESPONSE}"])
def test_invalid_template_rejected(template):
    with pytest.raises(ValueError):
        validate_reward_template(template)
