from typing import Any

REASONING_KEYS = frozenset(
    ("enable_thinking", "thinking", "reasoning", "reasoning_effort")
)


def is_reasoning_active(chat_template_kwargs: dict[str, Any]) -> bool:
    for key in REASONING_KEYS:
        value = chat_template_kwargs.get(key)
        if key == "reasoning_effort":
            if value is None or value == "none":
                continue
        elif not value:
            continue
        return True
    return False
