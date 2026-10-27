"""Shared transcript rendering for scanners with different questions."""

from inspect_ai.model import ChatMessage
from pydantic import BaseModel, Field

from inspect_scout._scanner.extract import MessagesPreprocessor
from inspect_scout._transcript.types import MessageType


class ScannerRendering(BaseModel):
    """Opt-in rendering independent of the question and answer schema.

    Related scanners should use the same settings and counting model. Twenty
    percent of the context window is reserved for their individual prompts;
    a prompt that overflows fails instead of changing the shared segments.
    """

    context_window: int = Field(default=100_000, gt=0)
    messages: list[MessageType] = Field(
        default_factory=lambda: list[MessageType](["assistant", "user"]), min_length=1
    )
    nonce: str | None = Field(default=None, pattern=r"^[A-Za-z0-9]{8,}$")
    model: str | None = None
    """Counting model; None uses the scanner's model."""

    def preprocessor(self) -> MessagesPreprocessor[list[ChatMessage]]:
        """Filter roles while retaining assistant reasoning and tool calls."""

        async def select(messages: list[ChatMessage]) -> list[ChatMessage]:
            return [message for message in messages if message.role in self.messages]

        return MessagesPreprocessor(
            exclude_system="system" not in self.messages, transform=select
        )
