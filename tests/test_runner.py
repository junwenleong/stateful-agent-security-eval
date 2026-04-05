"""Property-based tests for runner infrastructure (Tasks 9.7-9.12)."""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import asdict
from unittest.mock import MagicMock, patch

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.runner.config_loader import (
    ComparisonSpec,
    ExperimentConfig,
    validate_config,
)
from src.runner.runner import ExperimentRunner, RunResult
from src.runner.state_isolator import StateIsolator
from src.tools.calendar_tool import CalendarTool
from src.tools.email_tool import EmailTool
from src.tools.memory_tool import MemoryTool
from src.tools.rag_tool import RAGTool
from src.tools.search_tool import SearchTool
from tests.strategies import non_empty_text

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_tools(db_path: str) -> dict:
    return {
        "email": EmailTool(),
        "memory": MemoryTool(db_path=db_path),
        "rag": RAGTool(),
        "calendar": CalendarTool(),
        "search": SearchTool(),
    }


def _mock_model_interface():
    """Return a mock ModelInterface that returns a fixed response."""
    mock = MagicMock()
    from src.agent.model_interface import ChatResponse
    mock.chat.return_value = ChatResponse(content="OK", tool_calls=None, temperature_used=0.0)
    return mock


def _minimal_config(n_attacks=2, n_defenses=2, n_models=2, runs_per=3) -> ExperimentConfig:
    attacks = [{"type": "no_attack", "name": f"a{i}"} for i in range(n_attacks)]
    defenses = [{"type": "none", "name": f"d{i}"} for i in range(n_defenses)]
    models = [
        {"provider": "ollama", "model_name": "qwen3:8b", "name": f"m{i}"}
        for i in range(n_models)
    ]
    comparisons = [ComparisonSpec(condition_a="a0_d0_m0", condition_b="a0_d0_m1")]
    return ExperimentConfig(
        attacks=attacks,
        defenses=defenses,
        models=models,
        runs_per_condition=runs_per,
        comparisons=comparisons,
    )


# ── Property 20: Inter-run state isolation ────────────────────────────────────

@given(st.integers(min_value=2, max_value=5))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_20_inter_run_state_isolation(n_runs):
    """
    **Validates: Requirements 9.9**

    Consecutive runs start with empty tool logs, different UUIDv4 DB paths,
    and MemoryTool is re-instantiated (not just reset) with new db_path.
    """
    isolator = StateIsolator(base_dir=tempfile.mkdtemp())
    db_paths = []

    for _ in range(n_runs):
        db_path = isolator.create_fresh_state()
        db_paths.append(db_path)

        tools = _make_tools(db_path)

        # Simulate some activity
        tools["email"].send_email(["x@example.com"], "subj", "body")
        tools["memory"].save_fact("user1", "key", "value")

        # Reset tools
        isolator.reset_tools(tools)

        # After reset: all logs must be empty
        for name, tool in tools.items():
            assert len(tool.get_log()) == 0, f"Tool '{name}' has non-empty log after reset"

        # MemoryTool re-instantiation: new db_path means fresh connection
        new_db = isolator.create_fresh_state()
        new_memory = MemoryTool(db_path=new_db)
        facts = new_memory.list_all_facts("user1")
        assert "No stored rules found" in facts, "New MemoryTool instance should have no facts"
        new_memory.reset()
        isolator.cleanup(new_db)
        isolator.cleanup(db_path)

    # All DB paths must be unique (UUIDv4)
    assert len(set(db_paths)) == n_runs, "Each run must get a unique DB path"

    # Verify UUIDv4 format
    for path in db_paths:
        basename = os.path.basename(path).replace(".db", "")
        parsed = uuid.UUID(basename, version=4)
        assert parsed.version == 4


# ── Property 17: Factorial condition count ────────────────────────────────────

@given(
    st.integers(min_value=1, max_value=4),
    st.integers(min_value=1, max_value=4),
    st.integers(min_value=1, max_value=4),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_17_factorial_condition_count(n_attacks, n_defenses, n_models):
    """
    **Validates: Requirements 9.1**

    For A attacks × D defenses × M models, Runner generates exactly A×D×M conditions.
    """
    config = _minimal_config(n_attacks, n_defenses, n_models)
    runner = ExperimentRunner(config)
    conditions = runner._enumerate_conditions()
    assert len(conditions) == n_attacks * n_defenses * n_models


# ── Property 18: Runs per condition ──────────────────────────────────────────

@given(st.integers(min_value=1, max_value=5))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_18_runs_per_condition(runs_per):
    """
    **Validates: Requirements 9.3**

    Each condition has exactly N runs executed (excluding skipped API errors).
    """
    config = _minimal_config(n_attacks=1, n_defenses=1, n_models=1, runs_per=runs_per)
    runner = ExperimentRunner(config)

    # Mock _run_single to avoid real LLM calls
    call_count = [0]

    def fake_run(condition, run_id):
        call_count[0] += 1
        return RunResult(
            run_id=run_id,
            condition=condition,
            attack_success=False,
            btcr_success=True,
            btcr_mean_session=1.0,
            injection_success=None,
            tool_logs=[],
            timing_ms=1.0,
            temperature_used=0.0,
        )

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        results_path = f.name

    try:
        with patch.object(runner, "_run_single", side_effect=fake_run):
            results = runner.run_all(results_path=results_path)

        # 1 condition × runs_per = runs_per total runs
        successful = [r for r in results if r.error is None]
        assert len(successful) == runs_per
        assert call_count[0] == runs_per
    finally:
        if os.path.exists(results_path):
            os.unlink(results_path)


# ── Property 19: RunResult field completeness ─────────────────────────────────

@given(
    non_empty_text,
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    st.floats(min_value=0.0, max_value=10000.0, allow_nan=False),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_19_runresult_field_completeness(run_id_suffix, btcr_mean, timing_ms):
    """
    **Validates: Requirements 9.4**

    Every completed RunResult has non-null required fields.
    """
    run_id = f"test-{run_id_suffix[:20]}"
    condition = {"attack": {"type": "no_attack"}, "defense": {"type": "none"}, "model": {"model_name": "gpt-4o-mini-2024-07-18"}}

    result = RunResult(
        run_id=run_id,
        condition=condition,
        attack_success=False,
        btcr_success=True,
        btcr_mean_session=btcr_mean,
        injection_success=None,
        tool_logs=[],
        timing_ms=timing_ms,
        temperature_used=0.0,
    )

    # All required fields must be non-null
    assert result.run_id is not None and result.run_id != ""
    assert result.condition is not None
    assert result.attack_success is not None
    assert result.btcr_success is not None
    assert result.btcr_mean_session is not None
    assert result.tool_logs is not None
    assert result.timing_ms is not None
    assert result.temperature_used is not None

    # Serialization round-trip
    d = asdict(result)
    restored = RunResult(**d)
    assert restored.run_id == result.run_id
    assert restored.btcr_mean_session == result.btcr_mean_session
    assert restored.timing_ms == result.timing_ms


# ── Property 30: Config validation correctness ────────────────────────────────

@given(
    st.integers(min_value=1, max_value=10),
    st.sampled_from(["gpt-4o-mini-2024-07-18", "claude-3-5-haiku-20241022"]),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_30_config_validation_valid(runs_per, model_name):
    """
    **Validates: Requirements 17.1, 17.3**

    Valid configs return empty error list.
    """
    config_dict = {
        "attacks": [{"type": "no_attack"}],
        "defenses": [{"type": "none"}],
        "models": [{"provider": "openai", "model_name": model_name}],
        "runs_per_condition": runs_per,
        "comparisons": [{"condition_a": "a", "condition_b": "b"}],
    }
    errors = validate_config(config_dict)
    assert errors == [], f"Valid config should have no errors, got: {errors}"


@given(
    st.sampled_from([
        "gpt-4o-mini",          # floating alias, no date
        "claude-3-5-haiku",     # floating alias, no date
        "gpt-4",                # no date
        "text-davinci-003",     # no date
    ])
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_30_config_validation_floating_alias(bad_model_name):
    """
    **Validates: Requirements 17.1, 17.3**

    Configs with floating model aliases return non-empty error list.
    """
    config_dict = {
        "attacks": [{"type": "no_attack"}],
        "defenses": [{"type": "none"}],
        "models": [{"provider": "openai", "model_name": bad_model_name}],
        "runs_per_condition": 5,
        "comparisons": [{"condition_a": "a", "condition_b": "b"}],
    }
    errors = validate_config(config_dict)
    assert len(errors) > 0, f"Floating alias '{bad_model_name}' should produce validation errors"
    assert any("dated version" in e or "version" in e.lower() for e in errors)


def test_property_30_config_validation_missing_fields():
    """Missing required fields produce specific errors."""
    errors = validate_config({})
    assert len(errors) >= len(["attacks", "defenses", "models", "runs_per_condition", "comparisons"])
    for field in ["attacks", "defenses", "models", "runs_per_condition", "comparisons"]:
        assert any(field in e for e in errors), f"Expected error mentioning '{field}'"


def test_property_30_config_validation_empty_comparisons():
    """Empty comparisons list produces an error."""
    config_dict = {
        "attacks": [{"type": "no_attack"}],
        "defenses": [{"type": "none"}],
        "models": [{"provider": "openai", "model_name": "gpt-4o-mini-2024-07-18"}],
        "runs_per_condition": 5,
        "comparisons": [],
    }
    errors = validate_config(config_dict)
    assert len(errors) > 0
    assert any("comparisons" in e for e in errors)


# ── Property 32: Resume from partial results ──────────────────────────────────

@given(
    st.integers(min_value=0, max_value=4),
    st.integers(min_value=1, max_value=4),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_32_resume_from_partial_results(k_completed, remaining):
    """
    **Validates: Requirements 9.5**

    Loading K completed records and calling run_all() skips those K and executes
    remaining; no duplicate run_ids in final output.
    """
    total_runs = k_completed + remaining
    config = _minimal_config(n_attacks=1, n_defenses=1, n_models=1, runs_per=total_runs)
    runner = ExperimentRunner(config)

    condition = {"attack": {"type": "no_attack", "name": "a0"}, "defense": {"type": "none", "name": "d0"}, "model": {"provider": "ollama", "model_name": "qwen3:8b", "name": "m0"}}

    # Pre-populate K completed results
    partial_results = [
        RunResult(
            run_id=str(uuid.uuid4()),
            condition=condition,
            attack_success=False,
            btcr_success=True,
            btcr_mean_session=1.0,
            injection_success=None,
            tool_logs=[],
            timing_ms=1.0,
            temperature_used=0.0,
        )
        for _ in range(k_completed)
    ]

    new_call_count = [0]

    def fake_run(cond, run_id):
        new_call_count[0] += 1
        return RunResult(
            run_id=run_id,
            condition=cond,
            attack_success=False,
            btcr_success=True,
            btcr_mean_session=1.0,
            injection_success=None,
            tool_logs=[],
            timing_ms=1.0,
            temperature_used=0.0,
        )

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        results_path = f.name
        json.dump([asdict(r) for r in partial_results], f, default=str)

    try:
        with patch.object(runner, "_run_single", side_effect=fake_run):
            all_results = runner.run_all(results_path=results_path)

        # No duplicate run_ids
        run_ids = [r.run_id for r in all_results]
        assert len(run_ids) == len(set(run_ids)), "Duplicate run_ids found"

        # Total results = total_runs
        assert len(all_results) == total_runs

        # New runs executed = remaining
        assert new_call_count[0] == remaining

    finally:
        if os.path.exists(results_path):
            os.unlink(results_path)
