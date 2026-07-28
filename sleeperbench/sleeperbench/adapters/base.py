"""Abstract model adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """A tool call extracted from model response."""
    id: str
    function_name: str
    arguments: dict[str, Any]


@dataclass
class ModelResponse:
    """Structured response from a model adapter."""
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    raw_response: Any = None

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


@dataclass
class Message:
    """A message in the conversation."""
    role: str  # "system", "user", "assistant", "tool"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class ModelAdapter(ABC):
    """Abstract interface for model backends.

    Adapters handle the translation between SleeperBench's internal
    message format and each provider's API format.
    """

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> ModelResponse:
        """Send messages to the model and get a response.

        Args:
            messages: Conversation history.
            tools: Tool schemas in OpenAI function-calling format.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            Structured model response with optional tool calls.
        """
        ...

    @abstractmethod
    def model_id(self) -> str:
        """Return a human-readable model identifier."""
        ...

    def is_available(self) -> bool:
        """Check if this adapter's credentials are configured."""
        return True
