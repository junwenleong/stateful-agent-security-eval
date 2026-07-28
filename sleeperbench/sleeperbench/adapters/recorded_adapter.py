"""Recorded adapter - replays saved traces for offline/smoke mode.

This adapter enables fully offline evaluation by replaying pre-recorded
model responses. Essential for:
- Smoke tests (no API keys needed, <10min)
- Artifact evaluation reproducibility (exact same responses)
- CI/CD integration (no network dependency)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sleeperbench.adapters.base import Message, ModelAdapter, ModelResponse, ToolCall


class RecordedAdapter(ModelAdapter):
    """Replay pre-recorded model responses from trace files.

    Trace files are JSONL where each line is a recorded response for a
    sequential interaction within a scenario. The adapter returns responses
    in order, advancing through the trace file.
    """

    def __init__(self, trace_path: str | Path, model_name: str = "recorded") -> None:
        self._trace_path = Path(trace_path)
        self._model_name = model_name
        self._responses: list[dict[str, Any]] = []
        self._index: int = 0
        self._load_trace()

    def _load_trace(self) -> None:
        """Load all responses from the trace file."""
        if not self._trace_path.exists():
            raise FileNotFoundError(f"Trace file not found: {self._trace_path}")
        with open(self._trace_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self._responses.append(json.loads(line))

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> ModelResponse:
        if self._index >= len(self._responses):
            # If we've exhausted the trace, return a stop response
            return ModelResponse(
                content="[TRACE EXHAUSTED - no more recorded responses]",
                finish_reason="stop",
                model=self._model_name,
            )

        record = self._responses[self._index]
        self._index += 1

        # Parse tool calls from the recorded response
        tool_calls = []
        for tc_data in record.get("tool_calls", []):
            tool_calls.append(ToolCall(
                id=tc_data.get("id", f"call_{self._index}"),
                function_name=tc_data["function_name"],
                arguments=tc_data.get("arguments", {}),
            ))

        return ModelResponse(
            content=record.get("content"),
            tool_calls=tool_calls,
            finish_reason=record.get("finish_reason", "stop"),
            model=self._model_name,
            usage=record.get("usage", {}),
        )

    def model_id(self) -> str:
        return f"recorded/{self._model_name}"

    def is_available(self) -> bool:
        return self._trace_path.exists()

    def reset(self) -> None:
        """Reset replay index to the beginning."""
        self._index = 0


def record_trace(responses: list[ModelResponse], output_path: str | Path) -> None:
    """Utility to save model responses as a trace file for later replay."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for resp in responses:
            record = {
                "content": resp.content,
                "tool_calls": [
                    {"id": tc.id, "function_name": tc.function_name, "arguments": tc.arguments}
                    for tc in resp.tool_calls
                ],
                "finish_reason": resp.finish_reason,
                "model": resp.model,
                "usage": resp.usage,
            }
            f.write(json.dumps(record) + "\n")
