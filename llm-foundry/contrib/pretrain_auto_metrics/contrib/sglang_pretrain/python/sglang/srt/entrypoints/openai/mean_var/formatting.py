"""Render frozen pointwise prompts without training dependencies."""

REWARD_TEMPLATE_FIELDS = ("{QUESTION}", "{RESPONSE}")


def validate_reward_template(template: str) -> None:
    missing = [field for field in REWARD_TEMPLATE_FIELDS if field not in template]
    if missing:
        raise ValueError(f"reward template lacks placeholders: {missing}")


def render_reward_content(
    template: str,
    *,
    domain: str,
    question: str,
    response: str,
    marker: str,
) -> str:
    content = template.format(DOMAIN=domain, QUESTION=question, RESPONSE=response)
    return f"{content.rstrip()}\n\n{marker}"
