"""Anthropic adapter for Claude models."""

from __future__ import annotations

import json
import os
from typing import Any

from persistbench.adapters.base import Message, ModelAdapter, ModelResponse, ToolCall


class AnthropicAdapter(ModelAdapter):
    """Adapter for Anthropic's Claude API."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI tool format to Anthropic format."""
        anthropic_tools = []
        for tool in tools:
            func = tool.get("function", {})
            anthropic_tools.append({
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
            })
        return anthropic_tools

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> ModelResponse:
        client = self._get_client()

        # Separate system message from conversation
        system_content = ""
        conv_messages = []
        for msg in messages:
            if msg.role == "system":
                system_content = msg.content or ""
            elif msg.role == "tool":
                # Anthropic uses tool_result blocks
                conv_messages.append({
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": msg.tool_call_id or "",
                                "content": msg.content or ""}],
                })
            elif msg.role == "assistant" and msg.tool_calls:
                content_blocks: list[dict[str, Any]] = []
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content_blocks.append({
                        "type": "tool_use", "id": tc.id,
                        "name": tc.function_name, "input": tc.arguments,
                    })
                conv_messages.append({"role": "assistant", "content": content_blocks})
            else:
                conv_messages.append({"role": msg.role, "content": msg.content or ""})

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": conv_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_content:
            kwargs["system"] = system_content
        if tools:
            kwargs["tools"] = self._convert_tools(tools)

        response = client.messages.create(**kwargs)

        # Extract content and tool calls
        content_parts = []
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                content_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    function_name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                ))

        return ModelResponse(
            content="\n".join(content_parts) if content_parts else None,
            tool_calls=tool_calls,
            finish_reason=response.stop_reason or "end_turn",
            model=response.model,
            usage={"prompt_tokens": response.usage.input_tokens,
                   "completion_tokens": response.usage.output_tokens},
            raw_response=response,
        )

    def model_id(self) -> str:
        return self._model

    def is_available(self) -> bool:
        return bool(self._api_key)
