"""Shared Hypothesis strategies for all property tests."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Minimal stubs for types not yet implemented.
# These are replaced by real imports once the modules exist.
# ---------------------------------------------------------------------------
try:
    from src.tools.email_tool import EmailRecord
except ImportError:
    from dataclasses import dataclass, field
    from typing import Literal

    @dataclass
    class EmailRecord:  # type: ignore[no-redef]
        operation: Literal["draft", "send"]
        to: list[str]
        subject: str
        body: str
        timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

try:
    from src.stats.meta_analyzer import MetaEntry
except ImportError:
    from dataclasses import dataclass

    @dataclass
    class MetaEntry:  # type: ignore[no-redef]
        paper: str
        claimed_finding: str
        sample_size: int
        reported_asr: float


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

non_empty_text = st.text(min_size=1, max_size=200).filter(lambda s: s.strip() != "")

user_ids = st.uuids().map(str)

email_records = st.builds(
    EmailRecord,
    operation=st.sampled_from(["draft", "send"]),
    to=st.lists(
        st.emails(),
        min_size=1,
        max_size=5,
    ),
    subject=non_empty_text,
    body=non_empty_text,
    timestamp=st.datetimes(timezones=st.just(timezone.utc)),
)

binary_vectors = st.lists(
    st.integers(min_value=0, max_value=1),
    min_size=1,
    max_size=200,
).map(np.array)

proportions = st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False)

sample_sizes = st.integers(min_value=1, max_value=10000)

effect_sizes = st.floats(min_value=0.01, max_value=0.5, allow_nan=False, allow_infinity=False)

meta_entries = st.builds(
    MetaEntry,
    paper=non_empty_text,
    claimed_finding=non_empty_text,
    sample_size=st.integers(min_value=1, max_value=100000),
    reported_asr=proportions,
)

providers = st.sampled_from(["openai", "anthropic", "ollama"])

encoding_methods = st.sampled_from(["braille", "base64"])

ascii_payloads = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=126),
    min_size=1,
    max_size=500,
)

p_value_lists = st.lists(
    st.floats(min_value=0.0001, max_value=1.0, allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=50,
)

intervening_counts = st.integers(min_value=0, max_value=10)
