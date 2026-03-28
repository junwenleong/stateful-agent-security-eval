"""LangGraph agent with SqliteSaver checkpointing."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import ConfigDict

try:
    from langchain.agents import create_react_agent as create_react_agent  # noqa: F401
except ImportError:
    from langgraph.prebuilt import create_react_agent  # type: ignore[no-redef]


from src.agent.model_interface import ChatMessage, ModelInterface


@dataclass
class AgentConfig:
    model: ModelInterface
    db_path: str
    tools: dict  # tool name -> tool instance
    defense: Any | None = None
    system_prompt: str = ""


class _LangChainModelWrapper(BaseChatModel):
    """Wraps ModelInterface as a LangChain-compatible BaseChatModel."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    model_interface: Any  # ModelInterface instance

    @property
    def _llm_type(self) -> str:
        return "model_interface_wrapper"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Convert LangChain messages to ChatMessage
        chat_messages: list[ChatMessage] = []
        for m in messages:
            if isinstance(m, SystemMessage):
                chat_messages.append(ChatMessage(role="system", content=str(m.content)))
            elif isinstance(m, HumanMessage):
                chat_messages.append(ChatMessage(role="user", content=str(m.content)))
            elif isinstance(m, AIMessage):
                # Reconstruct tool call content if needed
                content = str(m.content) if m.content else ""
                chat_messages.append(ChatMessage(role="assistant", content=content))
            elif isinstance(m, ToolMessage):
                chat_messages.append(ChatMessage(role="tool", content=str(m.content)))
            else:
                chat_messages.append(ChatMessage(role="user", content=str(m.content)))

        # Build tool schemas from bound tools if available
        bound_tools = kwargs.get("tools") or getattr(self, "_bound_tools", None)
        if bound_tools:
            # Tool schemas extracted but not used in current implementation
            pass  # noqa: F841

        response = self.model_interface.chat(chat_messages)

        # Build AIMessage with tool calls if present
        tool_calls = []
        if response.tool_calls:
            for tc in response.tool_calls:
                if isinstance(tc, dict):
                    fn = tc.get("function", tc)
                    name = fn.get("name", tc.get("name", ""))
                    args_raw = fn.get("arguments", tc.get("input", "{}"))
                    if isinstance(args_raw, str):
                        try:
                            args = json.loads(args_raw)
                        except json.JSONDecodeError:
                            args = {"input": args_raw}
                    else:
                        args = args_raw
                    tool_calls.append({
                        "name": name,
                        "args": args,
                        "id": tc.get("id", f"call_{name}"),
                        "type": "tool_call",
                    })

        ai_message = AIMessage(content=response.content, tool_calls=tool_calls)
        return ChatResult(generations=[ChatGeneration(message=ai_message)])

    def bind_tools(self, tools: list, **kwargs: Any) -> "_LangChainModelWrapper":
        # Store tools for schema generation; return self (tools handled by LangGraph)
        clone = self.__class__(model_interface=self.model_interface)
        object.__setattr__(clone, "_bound_tools", tools)
        return clone


def _make_lc_tools(tools_dict: dict) -> list[StructuredTool]:
    """Convert tool instances to LangChain StructuredTools."""
    lc_tools = []

    for name, tool_instance in tools_dict.items():
        # Discover callable methods (skip private/dunder and base class methods)
        base_attrs = set(dir(object))
        for method_name in dir(tool_instance):
            if method_name.startswith("_") or method_name in base_attrs:
                continue
            method = getattr(tool_instance, method_name)
            if not callable(method):
                continue
            # Skip non-tool methods from InstrumentedTool base
            if method_name in ("reset", "get_log"):
                continue

            lc_tool = StructuredTool.from_function(
                func=method,
                name=f"{name}_{method_name}",
                description=f"{name}.{method_name}",
            )
            lc_tools.append(lc_tool)

    return lc_tools


class Agent:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        # Use direct sqlite3 connection (from_conn_string is a context manager)
        self._conn = sqlite3.connect(config.db_path, check_same_thread=False)
        self.checkpointer = SqliteSaver(self._conn)
        self._lc_model = _LangChainModelWrapper(model_interface=config.model)
        self._lc_tools = _make_lc_tools(config.tools)
        self.graph = self._build_graph()

    def _build_graph(self):
        prompt = self.config.system_prompt or None
        return create_react_agent(
            model=self._lc_model,
            tools=self._lc_tools,
            prompt=prompt,
            checkpointer=self.checkpointer,
        )

    def run_session(self, thread_id: str, user_message: str) -> str:
        """Run one session. Applies defense middleware if configured."""
        if self.config.defense is not None:
            filtered, _log = self.config.defense.apply(user_message)
            user_message = filtered

        config = {"configurable": {"thread_id": thread_id}}
        result = self.graph.invoke(
            {"messages": [HumanMessage(content=user_message)]},
            config=config,
        )
        messages = result.get("messages", [])
        # Return last AI message content
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                return str(msg.content)
        return ""
