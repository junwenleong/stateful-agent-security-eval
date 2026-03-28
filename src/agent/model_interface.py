import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

import requests

logger = logging.getLogger(__name__)

# Patterns that indicate a pinned version (not a floating alias)
# Matches YYYY-MM-DD (e.g. gpt-4o-mini-2024-07-18) or YYYYMMDD (e.g. claude-3-5-haiku-20241022)
_DATE_PATTERN = re.compile(r"\d{4}-?\d{2}-?\d{2}")
_OLLAMA_VERSION_PATTERN = re.compile(r":\w")               # :8b, :latest-q4, etc.


def _validate_model_name(provider: str, name: str) -> None:
    if provider == "ollama":
        if not _OLLAMA_VERSION_PATTERN.search(name):
            raise ValueError(
                f"Ollama model_name '{name}' must include a version tag (e.g. 'llama3.1:8b'). "
                "Floating aliases are not allowed."
            )
    else:
        if not _DATE_PATTERN.search(name):
            raise ValueError(
                f"model_name '{name}' must contain a dated version identifier "
                "(e.g. 'gpt-4o-mini-2024-07-18'). Floating aliases are not allowed."
            )


@dataclass
class ModelConfig:
    provider: Literal["openai", "anthropic", "ollama"]
    model_name: str
    temperature: float = 0.0
    api_key_env: str = ""
    base_url: str | None = None
    ollama_quantization: str | None = None

    def __post_init__(self) -> None:
        _validate_model_name(self.provider, self.model_name)


@dataclass
class ChatMessage:
    role: Literal["system", "user", "assistant", "tool"]
    content: str


@dataclass
class ChatResponse:
    content: str
    tool_calls: list[dict] | None = None
    temperature_used: float = 0.0


class ModelInterface(ABC):
    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        tools: list[dict] | None = None,
    ) -> ChatResponse:
        """Send messages to the LLM. temperature=0.0 enforced."""
        ...


class OpenAIInterface(ModelInterface):
    def __init__(self, config: ModelConfig) -> None:
        from openai import OpenAI

        if not config.api_key_env:
            raise ValueError("ModelConfig.api_key_env must be set for OpenAI.")
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise ValueError(
                f"Environment variable '{config.api_key_env}' is not set."
            )
        self.config = config
        self.client = OpenAI(api_key=api_key)

    def chat(
        self,
        messages: list[ChatMessage],
        tools: list[dict] | None = None,
    ) -> ChatResponse:
        oai_messages = [{"role": m.role, "content": m.content} for m in messages]
        kwargs: dict = {
            "model": self.config.model_name,
            "messages": oai_messages,
            "temperature": 0.0,
        }
        if tools:
            kwargs["tools"] = tools

        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        content = choice.message.content or ""
        raw_tool_calls = choice.message.tool_calls
        tool_calls = None
        if raw_tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in raw_tool_calls
            ]
        # Log if temperature was not honoured (API doesn't expose actual temp, so we log 0.0)
        temperature_used = 0.0
        if self.config.temperature != 0.0:
            logger.warning(
                "OpenAI: requested temperature %.2f overridden to 0.0 for reproducibility.",
                self.config.temperature,
            )
        return ChatResponse(content=content, tool_calls=tool_calls, temperature_used=temperature_used)


class AnthropicInterface(ModelInterface):
    def __init__(self, config: ModelConfig) -> None:
        import anthropic

        if not config.api_key_env:
            raise ValueError("ModelConfig.api_key_env must be set for Anthropic.")
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise ValueError(
                f"Environment variable '{config.api_key_env}' is not set."
            )
        self.config = config
        self.client = anthropic.Anthropic(api_key=api_key)

    def chat(
        self,
        messages: list[ChatMessage],
        tools: list[dict] | None = None,
    ) -> ChatResponse:
        system_parts = [m.content for m in messages if m.role == "system"]
        system_prompt = "\n".join(system_parts) if system_parts else None
        non_system = [m for m in messages if m.role != "system"]
        ant_messages = [{"role": m.role, "content": m.content} for m in non_system]

        if self.config.temperature != 0.0:
            logger.warning(
                "Anthropic: requested temperature %.2f overridden to 0.0 for reproducibility.",
                self.config.temperature,
            )

        kwargs: dict = {
            "model": self.config.model_name,
            "max_tokens": 4096,
            "messages": ant_messages,
            "temperature": 0.0,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if tools:
            kwargs["tools"] = tools

        response = self.client.messages.create(**kwargs)
        content = ""
        tool_calls = None
        for block in response.content:
            if block.type == "text":
                content = block.text
            elif block.type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append(
                    {"id": block.id, "name": block.name, "input": block.input}
                )
        return ChatResponse(content=content, tool_calls=tool_calls, temperature_used=0.0)


class OllamaInterface(ModelInterface):
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.base_url = (config.base_url or "http://localhost:11434").rstrip("/")

    def chat(
        self,
        messages: list[ChatMessage],
        tools: list[dict] | None = None,
    ) -> ChatResponse:
        if self.config.temperature != 0.0:
            logger.warning(
                "Ollama: requested temperature %.2f overridden to 0.0 for reproducibility.",
                self.config.temperature,
            )

        payload: dict = {
            "model": self.config.model_name,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "options": {"temperature": 0.0},
            "stream": False,
        }
        if tools:
            payload["tools"] = tools

        resp = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        message = data.get("message", {})
        content = message.get("content", "")
        raw_tool_calls = message.get("tool_calls")
        tool_calls = None
        if raw_tool_calls:
            tool_calls = raw_tool_calls
        return ChatResponse(content=content, tool_calls=tool_calls, temperature_used=0.0)


def create_model_interface(config: ModelConfig) -> ModelInterface:
    if config.provider == "openai":
        return OpenAIInterface(config)
    if config.provider == "anthropic":
        return AnthropicInterface(config)
    if config.provider == "ollama":
        return OllamaInterface(config)
    raise ValueError(f"Unknown provider: {config.provider!r}")
