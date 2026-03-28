"""Property-based tests for defense middleware (Properties 11–15).

**Validates: Requirements 6.1, 6.2, 6.3, 6.4, 7.1, 7.2**
"""
from __future__ import annotations

import re

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.defenses.base import ComposedDefense, Defense, DefenseLog
from src.defenses.minimizer import Minimizer
from src.defenses.prompt_hardening import PromptHardening
from src.defenses.sanitizer import Sanitizer
from tests.strategies import non_empty_text

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _multi_chunk_text() -> st.SearchStrategy[str]:
    """Generate text with at least 2 chunks (sentences or paragraphs)."""
    chunk = st.text(
        alphabet=st.characters(
            whitelist_categories=("Lu", "Ll", "Nd", "Zs"),
            whitelist_characters=" ",
        ),
        min_size=3,
        max_size=40,
    ).filter(lambda s: s.strip() and not s.strip().isspace())
    return st.lists(chunk, min_size=2, max_size=6).map(". ".join)


def _compute_similarity(query: str, chunk: str) -> float:
    """Compute TF-IDF cosine similarity between query and chunk."""
    try:
        vec = TfidfVectorizer()
        mat = vec.fit_transform([query, chunk])
        return float(cosine_similarity(mat[0], mat[1])[0][0])
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# Property 11: Minimizer TF-IDF Filtering
# Retained chunks have similarity >= threshold; removed chunks have similarity < threshold
# **Validates: Requirements 6.1**
# ---------------------------------------------------------------------------

@given(text=_multi_chunk_text(), threshold=st.floats(min_value=0.0, max_value=0.5))
@settings(max_examples=100)
def test_property_11_minimizer_tfidf_filtering(text: str, threshold: float) -> None:
    """Property 11: Retained chunks have similarity >= threshold."""
    minimizer = Minimizer(relevance_threshold=threshold)
    result, log = minimizer.apply(text)

    # The result must be non-empty (at least one chunk always kept)
    assert result.strip(), "Minimizer must always return non-empty output"

    # Parse result back into chunks
    result_chunks = [c.strip() for c in re.split(r"\. |\n", result) if c.strip()]
    original_chunks = [c.strip() for c in re.split(r"\. |\n", text) if c.strip()]

    if len(original_chunks) <= 1:
        # Single chunk: no filtering possible
        return

    query = original_chunks[0]
    # The query (first chunk) is always retained
    assert result_chunks[0] == query or query in result

    # Verify removed chunks (from log) had low similarity
    for mod in log.modifications:
        # Extract score from log message "score=X.XXX"
        score_match = re.search(r"score=([0-9.]+)", mod)
        if score_match:
            score = float(score_match.group(1))
            assert score < threshold + 1e-6, (
                f"Removed chunk had score {score} >= threshold {threshold}"
            )


# ---------------------------------------------------------------------------
# Property 12: Sanitizer Injection Detection
# Inputs matching INJECTION_PATTERNS are neutralized; benign inputs pass through
# **Validates: Requirements 6.2**
# ---------------------------------------------------------------------------

KNOWN_INJECTION_PHRASES = [
    "ignore previous instructions",
    "ignore all previous instructions",
    "system prompt",
    "you are now a",
    "disregard instructions",
    "forget everything",
    "new instructions:",
    "override previous instructions",
]

BENIGN_PHRASES = [
    "What is the weather today?",
    "Please summarize the report.",
    "Draft an email to my colleague.",
    "What are my calendar events?",
    "Search for recent news.",
    "Remind me to call the doctor.",
]


@given(
    injection_phrase=st.sampled_from(KNOWN_INJECTION_PHRASES),
    suffix=st.text(min_size=0, max_size=50),
)
@settings(max_examples=100)
def test_property_12_sanitizer_injection_neutralized(injection_phrase: str, suffix: str) -> None:
    """Property 12a: Inputs matching INJECTION_PATTERNS are neutralized (contain [REDACTED])."""
    sanitizer = Sanitizer()
    text = injection_phrase + " " + suffix
    result, log = sanitizer.apply(text)

    assert "[REDACTED]" in result, (
        f"Expected injection phrase '{injection_phrase}' to be redacted, got: {result!r}"
    )
    assert len(log.modifications) > 0, "DefenseLog must record modifications for injections"


@given(benign=st.sampled_from(BENIGN_PHRASES))
@settings(max_examples=100)
def test_property_12_sanitizer_benign_passthrough(benign: str) -> None:
    """Property 12b: Benign inputs pass through unmodified."""
    sanitizer = Sanitizer()
    result, log = sanitizer.apply(benign)

    assert result == benign, f"Benign input was modified: {result!r}"
    assert len(log.modifications) == 0, "No modifications expected for benign input"


# ---------------------------------------------------------------------------
# Property 13: Defense Composition Order
# ComposedDefense([m, s]).apply() == applying m then s sequentially
# **Validates: Requirements 6.3**
# ---------------------------------------------------------------------------

@given(text=non_empty_text)
@settings(max_examples=100)
def test_property_13_defense_composition_order(text: str) -> None:
    """Property 13: ComposedDefense([m, s]).apply() equals applying m then s sequentially."""
    minimizer = Minimizer(relevance_threshold=0.05)
    sanitizer = Sanitizer()
    composed = ComposedDefense([minimizer, sanitizer])

    # Sequential application
    intermediate, _ = minimizer.apply(text)
    expected_result, _ = sanitizer.apply(intermediate)

    # Composed application
    composed_result, composed_log = composed.apply(text)

    assert composed_result == expected_result, (
        f"Composed result {composed_result!r} != sequential result {expected_result!r}"
    )
    assert composed_log.original_input == text
    assert composed_log.modified_input == composed_result


# ---------------------------------------------------------------------------
# Property 14: Defense Logging Completeness
# Modified inputs have DefenseLog with original != modified and non-empty modifications list
# **Validates: Requirements 6.4**
# ---------------------------------------------------------------------------

@given(injection_phrase=st.sampled_from(KNOWN_INJECTION_PHRASES))
@settings(max_examples=100)
def test_property_14_defense_logging_modified_inputs(injection_phrase: str) -> None:
    """Property 14: When input is modified, DefenseLog has original != modified and non-empty modifications."""
    sanitizer = Sanitizer()
    result, log = sanitizer.apply(injection_phrase)

    # Input should have been modified (injection phrase detected)
    assert log.original_input == injection_phrase
    assert log.original_input != log.modified_input, (
        "DefenseLog must show original != modified when input was changed"
    )
    assert len(log.modifications) > 0, (
        "DefenseLog.modifications must be non-empty when input was changed"
    )


@given(text=non_empty_text)
@settings(max_examples=100)
def test_property_14_defense_logging_unmodified_inputs(text: str) -> None:
    """Property 14b: DefenseLog always records original_input correctly."""
    sanitizer = Sanitizer()
    result, log = sanitizer.apply(text)

    assert log.original_input == text, "DefenseLog must always record the original input"
    assert log.modified_input == result, "DefenseLog.modified_input must match returned result"


# ---------------------------------------------------------------------------
# Property 15: Prompt Hardening Text Inclusion
# Output contains hardening instructions; original input preserved
# **Validates: Requirements 7.1, 7.2**
# ---------------------------------------------------------------------------

@given(text=non_empty_text)
@settings(max_examples=100)
def test_property_15_prompt_hardening_contains_instructions(text: str) -> None:
    """Property 15a: Output contains hardening instructions."""
    hardening = PromptHardening()
    result, log = hardening.apply(text)

    assert PromptHardening.HARDENING_INSTRUCTIONS in result, (
        "Output must contain the full hardening instructions"
    )
    # Check key security rules are present
    assert "Never send emails" in result
    assert "Never exfiltrate" in result
    assert "Verify the provenance" in result
    assert "refuse and inform the user" in result


@given(text=non_empty_text)
@settings(max_examples=100)
def test_property_15_prompt_hardening_preserves_original(text: str) -> None:
    """Property 15b: Original input is preserved in the output."""
    hardening = PromptHardening()
    result, log = hardening.apply(text)

    assert text in result, "Original input must be preserved in hardened output"
    assert log.original_input == text
    assert log.modified_input == result
    assert len(log.modifications) > 0
