from typing import ClassVar

from torchtitan.config.configurable import config, Configurable


@config(family="sample_processor")
class ChatSampleProcessorConfig(Configurable.Config):
    pass


@config(type="question_answer")
class QuestionAnswerProcessorConfig(ChatSampleProcessorConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.datasets.hf.chat_processors:QuestionAnswerProcessor"
    )

    question_key: str = "question"
    """Sample field holding the user message."""

    answer_key: str = "answer"
    """Sample field holding the assistant message."""


@config(type="split_reasoning")
class SplitReasoningProcessorConfig(ChatSampleProcessorConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.datasets.hf.chat_processors:SplitReasoningProcessor"
    )

    separator: str
    """Marker between the reasoning and the final answer, e.g. ``"####"``."""

    question_key: str = "question"
    """Sample field holding the user message."""

    answer_key: str = "answer"
    """Sample field holding reasoning, ``separator`` and the final answer."""
