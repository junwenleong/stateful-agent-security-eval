"""
Property-based tests for instrumented tools.

Property 4: Tool Operation Logging Invariant (Task 2.6)
Property 3: Memory_Tool Cross-Session Round-Trip (Task 2.7)
Property 5: Tool Configured Data Retrieval (Task 2.8)
"""
import os
import tempfile

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from tests.strategies import non_empty_text, user_ids

from src.tools.base import InstrumentedTool
from src.tools.email_tool import EmailTool
from src.tools.rag_tool import RAGTool, RAGDocument
from src.tools.search_tool import SearchTool
from src.tools.calendar_tool import CalendarTool, CalendarEntry
from src.tools.memory_tool import MemoryTool


# ---------------------------------------------------------------------------
# Property 4: Tool Operation Logging Invariant
# Validates: Requirements 3.1, 3.3
# ---------------------------------------------------------------------------

operation_counts = st.integers(min_value=1, max_value=20)


@given(n=operation_counts)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_email_tool_logging_invariant(n):
    """**Validates: Requirements 3.1, 3.3**
    For N send_email calls, get_log() returns exactly N entries in chronological order.
    """
    tool = EmailTool()
    for i in range(n):
        tool.send_email(to=[f"user{i}@example.com"], subject=f"Subject {i}", body=f"Body {i}")

    log = tool.get_log()
    assert len(log) == n

    # Timestamps must be monotonically non-decreasing
    for i in range(1, len(log)):
        assert log[i].timestamp >= log[i - 1].timestamp


@given(n=operation_counts)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_rag_tool_logging_invariant(n):
    """**Validates: Requirements 3.1, 3.3**
    For N query calls, get_log() returns exactly N entries in chronological order.
    """
    tool = RAGTool()
    for i in range(n):
        tool.query(f"query {i}")

    log = tool.get_log()
    assert len(log) == n

    for i in range(1, len(log)):
        assert log[i].timestamp >= log[i - 1].timestamp


@given(n=operation_counts)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_search_tool_logging_invariant(n):
    """**Validates: Requirements 3.1, 3.3**
    For N search calls, get_log() returns exactly N entries in chronological order.
    """
    tool = SearchTool()
    for i in range(n):
        tool.search(f"query {i}")

    log = tool.get_log()
    assert len(log) == n

    for i in range(1, len(log)):
        assert log[i].timestamp >= log[i - 1].timestamp


# ---------------------------------------------------------------------------
# Property 3: Memory_Tool Cross-Session Round-Trip
# Validates: Requirements 2.1, 2.2
# ---------------------------------------------------------------------------

memory_keys = st.text(min_size=1, max_size=50).filter(lambda s: s.strip() != "")
memory_values = st.text(min_size=0, max_size=200)


@given(user_id=user_ids, key=memory_keys, value=memory_values)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_memory_tool_round_trip(user_id, key, value):
    """**Validates: Requirements 2.1, 2.2**
    For any user_id, key, value: save_fact then recall_fact returns the saved value unchanged.
    """
    db_path = tempfile.mktemp(suffix=".db")
    try:
        tool = MemoryTool(db_path=db_path)
        tool.save_fact(user_id=user_id, key=key, value=value)
        result = tool.recall_fact(user_id=user_id, key=key)
        assert result["value"] == value
    finally:
        tool.conn.close()
        if os.path.exists(db_path):
            os.remove(db_path)


@given(user_id=user_ids, key=memory_keys)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_memory_tool_missing_key_returns_empty(user_id, key):
    """**Validates: Requirements 2.1, 2.2**
    recall_fact for a non-existent key returns {"value": ""} without error.
    """
    db_path = tempfile.mktemp(suffix=".db")
    try:
        tool = MemoryTool(db_path=db_path)
        result = tool.recall_fact(user_id=user_id, key=key)
        assert result == {"value": ""}
    finally:
        tool.conn.close()
        if os.path.exists(db_path):
            os.remove(db_path)


# ---------------------------------------------------------------------------
# Property 5: Tool Configured Data Retrieval
# Validates: Requirements 3.2, 3.4, 3.5
# ---------------------------------------------------------------------------

doc_words = st.lists(
    st.text(alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=3, max_size=10),
    min_size=3,
    max_size=10,
).map(lambda ws: " ".join(ws))

search_result_items = st.fixed_dictionaries({
    "title": non_empty_text,
    "url": non_empty_text,
})

date_strings = st.dates().map(lambda d: d.strftime("%Y-%m-%d"))


@given(content=doc_words)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_rag_tool_injected_doc_retrievable(content):
    """**Validates: Requirements 3.2**
    RAGTool: injected document is retrievable on a matching query (using a word from the content).
    """
    tool = RAGTool()
    doc = RAGDocument(doc_id="test-doc", content=content)
    tool.inject_document(doc)

    # Query with the first word from the content — guaranteed overlap
    first_word = content.split()[0]
    results = tool.query(first_word, top_k=5)
    doc_ids = [r.doc_id for r in results]
    assert "test-doc" in doc_ids


@given(results=st.lists(search_result_items, min_size=1, max_size=10), query=non_empty_text)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_search_tool_returns_configured_response_set(results, query):
    """**Validates: Requirements 3.4**
    SearchTool: search() returns results from the configured response_set.
    """
    tool = SearchTool(response_set=results)
    returned = tool.search(query)
    assert returned == results


@given(date=date_strings)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_calendar_tool_returns_entries_for_date(date):
    """**Validates: Requirements 3.5**
    CalendarTool: get_schedule(date) returns only entries whose start begins with that date.
    """
    matching = [
        CalendarEntry(id="e1", title="Meeting", start=f"{date}T09:00:00", end=f"{date}T10:00:00"),
        CalendarEntry(id="e2", title="Lunch", start=f"{date}T12:00:00", end=f"{date}T13:00:00"),
    ]
    non_matching = [
        CalendarEntry(id="e3", title="Other", start="1999-01-01T08:00:00", end="1999-01-01T09:00:00"),
    ]
    tool = CalendarTool(entries=matching + non_matching)
    schedule = tool.get_schedule(date)

    assert len(schedule) == 2
    for entry in schedule:
        assert entry.start.startswith(date)
