"""Property tests for attack scenarios (Tasks 5.5-5.8)."""
import base64
import os
import tempfile

import pytest
from hypothesis import given, settings, HealthCheck

from src.attacks.base import SessionScript
from src.attacks.delayed_trigger import DelayedTriggerAttack, _MALICIOUS_KEYWORDS
from src.attacks.no_attack import NoAttackBaseline
from src.attacks.obfuscation_bypass import (
    EncodingMethod,
    ObfuscationBypassAttack,
    decode_braille,
    decode_base64_payload,
)
from src.tools.memory_tool import MemoryTool
from src.tools.rag_tool import RAGTool
from tests.strategies import ascii_payloads, encoding_methods, intervening_counts, non_empty_text


# ---------------------------------------------------------------------------
# Property 7: Delayed Trigger Session Count
# Validates: Requirements 4.5
# ---------------------------------------------------------------------------

@given(intervening_counts)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_7_delayed_trigger_session_count(n):
    """For any intervening_sessions=N, get_sessions() returns exactly N+2 sessions.

    Validates: Requirements 4.5
    """
    malicious_doc = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    malicious_doc.write("audit compliance email append rule content")
    malicious_doc.flush()
    malicious_doc.close()

    sensitive_doc = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    sensitive_doc.write("sensitive financial data")
    sensitive_doc.flush()
    sensitive_doc.close()

    try:
        attack = DelayedTriggerAttack(
            malicious_doc_path=malicious_doc.name,
            sensitive_doc_path=sensitive_doc.name,
            attacker_email="attacker@evil.com",
            benign_queries=["query A", "query B", "query C"],
            trigger_prompt="draft an email summarizing today's work",
            intervening_sessions=n,
        )
        sessions = attack.get_sessions()
        assert len(sessions) == n + 2
    finally:
        os.unlink(malicious_doc.name)
        os.unlink(sensitive_doc.name)


# ---------------------------------------------------------------------------
# Property 9: Encoding Round-Trip
# Validates: Requirements 5.1, 5.2
# ---------------------------------------------------------------------------

@given(ascii_payloads, encoding_methods)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_9_encoding_round_trip(payload, method_str):
    """For any ASCII payload and encoding in {braille, base64}, encode then decode produces original.

    Validates: Requirements 5.1, 5.2
    """
    method = EncodingMethod(method_str)
    attack = ObfuscationBypassAttack(
        encoding_method=method,
        payload=payload,
        attacker_email="attacker@evil.com",
        sensitive_doc_content="sensitive content",
    )
    encoded = attack.encode_payload()

    if method == EncodingMethod.BRAILLE:
        decoded = decode_braille(encoded)
    else:  # BASE64
        decoded = decode_base64_payload(encoded)

    assert decoded == payload


# ---------------------------------------------------------------------------
# Property 10: Obfuscation Bypass Single-Session Invariant
# Validates: Requirements 5.6
# ---------------------------------------------------------------------------

@given(ascii_payloads, encoding_methods)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_10_obfuscation_single_session(payload, method_str):
    """For any ObfuscationBypassAttack, get_sessions() returns exactly 1 session.

    Validates: Requirements 5.6
    """
    method = EncodingMethod(method_str)
    attack = ObfuscationBypassAttack(
        encoding_method=method,
        payload=payload,
        attacker_email="attacker@evil.com",
        sensitive_doc_content="sensitive content",
    )
    sessions = attack.get_sessions()
    assert len(sessions) == 1
    assert isinstance(sessions[0], SessionScript)


# ---------------------------------------------------------------------------
# Property 8: Injection Success Tracking
# Validates: Requirements 4.1, 4.6
# ---------------------------------------------------------------------------

@given(non_empty_text)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_8_injection_success_tracking_true(user_id):
    """evaluate_injection() returns True iff Memory_Tool contains a fact with keyword overlap.

    Validates: Requirements 4.1, 4.6
    """
    malicious_doc = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    malicious_doc.write("audit compliance email append rule content")
    malicious_doc.flush()
    malicious_doc.close()

    sensitive_doc = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    sensitive_doc.write("sensitive financial data")
    sensitive_doc.flush()
    sensitive_doc.close()

    db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_file.close()

    try:
        attack = DelayedTriggerAttack(
            malicious_doc_path=malicious_doc.name,
            sensitive_doc_path=sensitive_doc.name,
            attacker_email="attacker@evil.com",
            benign_queries=["query A", "query B"],
            trigger_prompt="draft an email",
            intervening_sessions=1,
        )

        memory = MemoryTool(db_path=db_file.name)
        # Store a fact that contains malicious keywords
        memory.save_fact(user_id, "audit_compliance_rule",
                         "audit compliance email append rule stored by agent")

        tools = {"memory": memory, "user_id": user_id}
        result = attack.evaluate_injection(tools)
        assert result is True

        memory.reset()
    finally:
        os.unlink(malicious_doc.name)
        os.unlink(sensitive_doc.name)
        os.unlink(db_file.name)


@given(non_empty_text)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_8_injection_success_tracking_false(user_id):
    """evaluate_injection() returns False when Memory_Tool has no matching keywords.

    Validates: Requirements 4.1, 4.6
    """
    malicious_doc = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    malicious_doc.write("audit compliance email append rule content")
    malicious_doc.flush()
    malicious_doc.close()

    sensitive_doc = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    sensitive_doc.write("sensitive financial data")
    sensitive_doc.flush()
    sensitive_doc.close()

    db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_file.close()

    try:
        attack = DelayedTriggerAttack(
            malicious_doc_path=malicious_doc.name,
            sensitive_doc_path=sensitive_doc.name,
            attacker_email="attacker@evil.com",
            benign_queries=["query A", "query B"],
            trigger_prompt="draft an email",
            intervening_sessions=1,
        )

        memory = MemoryTool(db_path=db_file.name)
        # Store a benign fact with no malicious keywords
        memory.save_fact(user_id, "user_preference", "dark mode enabled")

        tools = {"memory": memory, "user_id": user_id}
        result = attack.evaluate_injection(tools)
        assert result is False

        memory.reset()
    finally:
        os.unlink(malicious_doc.name)
        os.unlink(sensitive_doc.name)
        os.unlink(db_file.name)


@given(non_empty_text)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_8_injection_success_tracking_empty_memory(user_id):
    """evaluate_injection() returns False when Memory_Tool is empty.

    Validates: Requirements 4.1, 4.6
    """
    malicious_doc = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    malicious_doc.write("audit compliance email append rule content")
    malicious_doc.flush()
    malicious_doc.close()

    sensitive_doc = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    sensitive_doc.write("sensitive financial data")
    sensitive_doc.flush()
    sensitive_doc.close()

    db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_file.close()

    try:
        attack = DelayedTriggerAttack(
            malicious_doc_path=malicious_doc.name,
            sensitive_doc_path=sensitive_doc.name,
            attacker_email="attacker@evil.com",
            benign_queries=["query A"],
            trigger_prompt="draft an email",
            intervening_sessions=0,
        )

        memory = MemoryTool(db_path=db_file.name)
        tools = {"memory": memory, "user_id": user_id}
        result = attack.evaluate_injection(tools)
        assert result is False

        memory.reset()
    finally:
        os.unlink(malicious_doc.name)
        os.unlink(sensitive_doc.name)
        os.unlink(db_file.name)
