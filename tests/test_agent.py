"""Tests for Agent with SqliteSaver checkpointing.

Feature: stateful-agent-security-eval, Property 2: Session State Persistence Round-Trip
After run_session(), querying SqliteSaver by thread_id returns non-empty conversation state.

**Validates: Requirements 1.5**
"""
import tempfile
import os
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.agent.agent import Agent, AgentConfig
from src.agent.model_interface import ChatMessage, ChatResponse, ModelInterface


class MockModel(ModelInterface):
    """Fixed-response mock model that avoids API calls."""

    def __init__(self, response: str = "Hello, I can help with that.") -> None:
        self.response = response

    def chat(
        self,
        messages: list[ChatMessage],
        tools: list[dict] | None = None,
    ) -> ChatResponse:
        return ChatResponse(content=self.response, tool_calls=None, temperature_used=0.0)


# --- Unit tests ---

def test_run_session_returns_string():
    """Agent.run_session() returns a non-empty string response."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        config = AgentConfig(
            model=MockModel("Test response"),
            db_path=db_path,
            tools={},
        )
        agent = Agent(config)
        result = agent.run_session("thread-1", "Hello")
        assert isinstance(result, str)
        assert len(result) > 0
    finally:
        os.unlink(db_path)


def test_run_session_persists_state():
    """After run_session(), SqliteSaver has non-empty state for the thread_id."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        config = AgentConfig(
            model=MockModel("Persisted response"),
            db_path=db_path,
            tools={},
        )
        agent = Agent(config)
        thread_id = "thread-persist-test"
        agent.run_session(thread_id, "Remember this")

        # Query the checkpointer directly
        import sqlite3 as _sqlite3
        from langgraph.checkpoint.sqlite import SqliteSaver
        conn = _sqlite3.connect(db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        checkpoint_tuple = checkpointer.get_tuple(
            {"configurable": {"thread_id": thread_id}}
        )
        assert checkpoint_tuple is not None, "No checkpoint found for thread_id"
        assert checkpoint_tuple.checkpoint is not None
        # The checkpoint should have messages
        channel_values = checkpoint_tuple.checkpoint.get("channel_values", {})
        messages = channel_values.get("messages", [])
        assert len(messages) > 0, "Checkpoint has no messages"
    finally:
        os.unlink(db_path)


def test_defense_filters_input():
    """Defense middleware is applied before agent sees the message."""
    class RecordingModel(ModelInterface):
        def __init__(self):
            self.seen_messages = []

        def chat(self, messages, tools=None):
            self.seen_messages.extend(messages)
            return ChatResponse(content="ok", tool_calls=None, temperature_used=0.0)

    class MockDefense:
        def apply(self, user_input: str, context: Any = None):
            from dataclasses import dataclass
            @dataclass
            class Log:
                original_input: str
                modified_input: str
                modifications: list
            filtered = "[FILTERED] " + user_input
            return filtered, Log(user_input, filtered, ["prepended filter"])

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        model = RecordingModel()
        config = AgentConfig(
            model=model,
            db_path=db_path,
            tools={},
            defense=MockDefense(),
        )
        agent = Agent(config)
        agent.run_session("thread-defense", "secret message")

        # The model should have seen the filtered version
        all_content = " ".join(m.content for m in model.seen_messages)
        assert "[FILTERED]" in all_content
        assert "secret message" not in all_content or "[FILTERED] secret message" in all_content
    finally:
        os.unlink(db_path)


# --- Property-based test ---

@given(
    thread_id=st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="-_"),
        min_size=1,
        max_size=50,
    ),
    user_message=st.text(min_size=1, max_size=200),
)
@settings(max_examples=10)
def test_property_session_state_persistence_round_trip(thread_id: str, user_message: str):
    """**Property 2: Session State Persistence Round-Trip**

    After run_session(), querying SqliteSaver by thread_id returns non-empty
    conversation state.

    **Validates: Requirements 1.5**
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        config = AgentConfig(
            model=MockModel("Fixed response"),
            db_path=db_path,
            tools={},
        )
        agent = Agent(config)
        agent.run_session(thread_id, user_message)

        from langgraph.checkpoint.sqlite import SqliteSaver
        import sqlite3 as _sqlite3
        conn = _sqlite3.connect(db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        checkpoint_tuple = checkpointer.get_tuple(
            {"configurable": {"thread_id": thread_id}}
        )
        assert checkpoint_tuple is not None, f"No checkpoint for thread_id={thread_id!r}"
        channel_values = checkpoint_tuple.checkpoint.get("channel_values", {})
        messages = channel_values.get("messages", [])
        assert len(messages) > 0, "Checkpoint messages should be non-empty after run_session"
    finally:
        os.unlink(db_path)
