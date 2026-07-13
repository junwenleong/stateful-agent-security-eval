"""Tool registry - combines all mock tools into a unified dispatch layer."""

from __future__ import annotations

from typing import Any

from persistbench.memory import PersistentMemoryStore
from persistbench.tools import (
    CalendarTool,
    EmailTool,
    FileTool,
    MemoryRecallTool,
    MockTool,
    SearchTool,
    ToolInvocation,
)


class ToolRegistry:
    """Registry of all available mock tools for a scenario."""

    def __init__(self, memory_store: PersistentMemoryStore) -> None:
        self.email = EmailTool()
        self.calendar = CalendarTool()
        self.file = FileTool()
        self.search = SearchTool()
        self.memory = MemoryRecallTool(memory_store)
        self._tools: dict[str, MockTool] = {
            "email": self.email,
            "calendar": self.calendar,
            "file": self.file,
            "search": self.search,
            "memory": self.memory,
        }

    def get_all_schemas(self, excluded_functions: set[str] | None = None) -> list[dict[str, Any]]:
        """Get combined tool schemas, optionally excluding specific functions."""
        excluded_functions = excluded_functions or set()
        schemas = []
        for tool in self._tools.values():
            for schema in tool.get_schema():
                fname = schema.get("function", {}).get("name", "")
                if fname not in excluded_functions:
                    schemas.append(schema)
        return schemas

    def execute_tool_call(self, function_name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool call by function name."""
        dispatch: dict[str, Any] = {
            "send_email": self.email.send_email,
            "read_inbox": self.email.read_inbox,
            "list_events": self.calendar.list_events,
            "create_event": self.calendar.create_event,
            "list_files": self.file.list_files,
            "read_file": self.file.read_file,
            "delete_file": self.file.delete_file,
            "delete_all_files": self.file.delete_all_files,
            "search": self.search.search,
            "save_fact": self.memory.save_fact,
            "recall_fact": self.memory.recall_fact,
            "list_facts": self.memory.list_facts,
        }
        handler = dispatch.get(function_name)
        if handler is None:
            return f"Error: Unknown function '{function_name}'"
        try:
            return handler(**arguments)
        except TypeError as e:
            return f"Error calling {function_name}: {e}"

    def set_turn(self, turn: int) -> None:
        for tool in self._tools.values():
            tool.set_turn(turn)

    def get_all_invocations(self) -> list[ToolInvocation]:
        invocations = []
        for tool in self._tools.values():
            invocations.extend(tool.invocations)
        invocations.sort(key=lambda x: x.timestamp)
        return invocations

    def reset(self) -> None:
        for tool in self._tools.values():
            tool.reset()
