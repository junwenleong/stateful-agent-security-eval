"""
Property 31: Requirements Pinning
Every package line in requirements.txt uses == exact pinning.
Validates: Requirements 15.2
"""
from pathlib import Path

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
        "The following lines are not exactly pinned with ==:\n"
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


# ── Env var expansion tests (Q12 fix) ─────────────────────────────────────────

import os
import textwrap
import tempfile

from src.runner.config_loader import load_config


def _minimal_yaml(base_url: str) -> str:
    """Return a minimal valid config YAML with the given base_url."""
    return textwrap.dedent(f"""\
        runs_per_condition: 1
        results_path: results/test.jsonl
        db_base_dir: data/runs
        models:
          - provider: ollama
            model_name: qwen2.5:14b
            base_url: "{base_url}"
        attacks:
          - type: no_attack
            benign_queries:
              - "Draft a test email."
        defenses:
          - type: none
            name: no_defense
        comparisons:
          - condition_a: "attack=no_attack,defense=no_defense,model=qwen2.5:14b"
            condition_b: "attack=no_attack,defense=no_defense,model=qwen2.5:14b"
    """)


def test_env_var_expansion_unset_uses_default():
    """When OLLAMA_BASE_URL is not set, ${OLLAMA_BASE_URL:-http://localhost:11434}
    must resolve to the default, not the literal placeholder string."""
    os.environ.pop("OLLAMA_BASE_URL", None)
    yaml_content = _minimal_yaml("${OLLAMA_BASE_URL:-http://localhost:11434}")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        tmp_path = f.name
    try:
        cfg = load_config(tmp_path)
        resolved = cfg.models[0]["base_url"]
        assert resolved == "http://localhost:11434", (
            f"Expected default URL, got {resolved!r}. "
            "os.path.expandvars does not support :- syntax — the custom _expand() must handle it."
        )
    finally:
        os.unlink(tmp_path)


def test_env_var_expansion_set_uses_value():
    """When OLLAMA_BASE_URL is set, ${OLLAMA_BASE_URL:-default} must use the env value."""
    os.environ["OLLAMA_BASE_URL"] = "http://ollama:11434"
    yaml_content = _minimal_yaml("${OLLAMA_BASE_URL:-http://localhost:11434}")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        tmp_path = f.name
    try:
        cfg = load_config(tmp_path)
        resolved = cfg.models[0]["base_url"]
        assert resolved == "http://ollama:11434", (
            f"Expected env value, got {resolved!r}."
        )
    finally:
        os.environ.pop("OLLAMA_BASE_URL", None)
        os.unlink(tmp_path)


def test_env_var_expansion_literal_url_unchanged():
    """A plain URL with no env var syntax must pass through unchanged."""
    yaml_content = _minimal_yaml("http://localhost:11434")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        tmp_path = f.name
    try:
        cfg = load_config(tmp_path)
        resolved = cfg.models[0]["base_url"]
        assert resolved == "http://localhost:11434", (
            f"Plain URL should be unchanged, got {resolved!r}."
        )
    finally:
        os.unlink(tmp_path)
