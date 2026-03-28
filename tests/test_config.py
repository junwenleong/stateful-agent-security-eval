"""
Property 31: Requirements Pinning
Every package line in requirements.txt uses == exact pinning.
Validates: Requirements 15.2
"""
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st


def get_requirement_lines() -> list[str]:
    req_path = Path(__file__).parent.parent / "requirements.txt"
    lines = req_path.read_text().splitlines()
    return [
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    ]


# **Validates: Requirements 15.2**
def test_property_31_requirements_pinning():
    """Property 31: Every non-comment, non-empty line in requirements.txt uses == exact pinning."""
    lines = get_requirement_lines()
    assert lines, "requirements.txt has no package lines"

    unpinned = [line for line in lines if "==" not in line]
    assert unpinned == [], (
        f"The following lines are not exactly pinned with ==:\n"
        + "\n".join(f"  {line}" for line in unpinned)
    )


@given(st.data())
@settings(max_examples=1)
def test_property_31_requirements_pinning_pbt(data):
    """Property 31 (PBT): For each sampled requirement line, it must contain ==."""
    lines = get_requirement_lines()
    assert lines, "requirements.txt has no package lines"

    line = data.draw(st.sampled_from(lines))
    assert "==" in line, f"Line is not exactly pinned: {line!r}"
