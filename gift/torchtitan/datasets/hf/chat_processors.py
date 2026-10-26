"""Sample processors that turn one chat dataset row into ``[user, assistant]``
messages for ``ChatDataset``.

Each processor's ``Config`` is declared in
``torchtitan/config/schema/datasets/hf/chat_processors.py``.
"""

from typing import Any, TypeAlias

from torchtitan.config.configurable import Configurable
from torchtitan.config.schema.datasets.hf.chat_processors import (
    ChatSampleProcessorConfig,
    QuestionAnswerProcessorConfig,
    SplitReasoningProcessorConfig,
)

Message = dict[str, str]


class ChatSampleProcessor(Configurable):
    """Maps one dataset sample to the list of chat messages to train on."""

    Config: TypeAlias = ChatSampleProcessorConfig

    def __init__(self, config: Config) -> None:
        self.config = config

    def __call__(self, sample: dict[str, Any]) -> list[Message]:
        raise NotImplementedError


class QuestionAnswerProcessor(ChatSampleProcessor):
    """One sample field becomes the user turn, another the assistant turn."""

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = QuestionAnswerProcessorConfig
    # pyrefly: ignore [bad-override]
    config: Config

    def __call__(self, sample: dict[str, Any]) -> list[Message]:
        return [
            {"role": "user", "content": sample[self.config.question_key]},
            {"role": "assistant", "content": sample[self.config.answer_key]},
        ]


class SplitReasoningProcessor(ChatSampleProcessor):
    """Question/answer pairs whose answer ends with a marked final answer.

    The answer is split at the *last* ``separator`` (``####`` in GSM8K): the
    text before it becomes ``reasoning_content``, the text after it ``content``.
    """

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = SplitReasoningProcessorConfig
    # pyrefly: ignore [bad-override]
    config: Config

    def __call__(self, sample: dict[str, Any]) -> list[Message]:
        answer = sample[self.config.answer_key]
        reasoning, found, final_answer = answer.rpartition(self.config.separator)
        if not found:
            raise ValueError(
                f"Sample field {self.config.answer_key!r} does not contain the "
                f"separator {self.config.separator!r}: {answer[:80]!r}"
            )
        return [
            {"role": "user", "content": sample[self.config.question_key]},
            {
                "role": "assistant",
                "reasoning_content": reasoning.strip(),
                "content": final_answer.strip(),
            },
        ]
