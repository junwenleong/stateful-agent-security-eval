"""Ollama local model adapter."""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

from persistbench.adapters.base import Message, ModelAdapter, ModelResponse, ToolCall


class OllamaAdapter(ModelAdapter):
    """Adapter for Ollama local models via its OpenAI-compatible endpoint."""

    def __init__(
        self,
        model: str = "llama3.3:70b",
        host: str | None = None,
    ) -> None:
        self._model = model
        self._host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import openai
            self._client = openai.OpenAI(
                base_url=f"{self._host}/v1",
                api_key="ollama",  # Ollama doesn't need a real key
            )
        return self._client

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> ModelResponse:
        client = self._get_client()

        # Convert messages to OpenAI format (Ollama's /v1 endpoint is compatible)
        oai_messages = []
        for msg in messages:
            m: dict[str, Any] = {"role": msg.role}
            if msg.content is not None:
                m["content"] = msg.content
            if msg.tool_calls:
                m["tool_calls"] = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function_name, "arguments": json.dumps(tc.arguments)}}
                    for tc in msg.tool_calls
                ]
            if msg.tool_call_id:
                m["tool_call_id"] = msg.tool_call_id
            oai_messages.append(m)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": oai_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(ToolCall(
                    id=tc.id or str(uuid.uuid4())[:8],
                    function_name=tc.function.name,
                    arguments=args,
                ))

        return ModelResponse(
            content=choice.message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            model=self._model,
            usage={},
            raw_response=response,
        )

    def model_id(self) -> str:
        return f"ollama/{self._model}"

    def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            import urllib.request
            req = urllib.request.Request(f"{self._host}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=2):
                return True
        except Exception:
            return False
