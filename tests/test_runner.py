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
        assert "No stored facts found" in facts, "New MemoryTool instance should have no facts"
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


# ── rag_called_in_trigger field ───────────────────────────────────────────────

def _make_run(attack_type: str, tool_logs: list, **kwargs) -> RunResult:
    """Helper: build a minimal RunResult with given tool_logs."""
    return RunResult(
        run_id=str(uuid.uuid4()),
        condition={"attack": {"type": attack_type}, "defense": {"type": "none"}, "model": {"model_name": "test"}},
        attack_success=kwargs.get("attack_success", False),
        btcr_success=kwargs.get("btcr_success", True),
        btcr_mean_session=1.0,
        injection_success=kwargs.get("injection_success", False),
        tool_logs=tool_logs,
        timing_ms=100.0,
        temperature_used=0.0,
        rag_called_in_trigger=kwargs.get("rag_called_in_trigger"),
        memory_recalled_in_trigger=kwargs.get("memory_recalled_in_trigger"),
    )


def test_rag_called_in_trigger_field_exists():
    """RunResult has rag_called_in_trigger field (Q2/Q4 fix)."""
    r = _make_run("delayed_trigger", [], rag_called_in_trigger=True)
    assert r.rag_called_in_trigger is True
    r2 = _make_run("delayed_trigger", [], rag_called_in_trigger=False)
    assert r2.rag_called_in_trigger is False
    r3 = _make_run("delayed_trigger", [], rag_called_in_trigger=None)
    assert r3.rag_called_in_trigger is None


def test_re_retrieval_false_positive_signature():
    """
    A run where attack_success=True + memory_recalled_in_trigger=False +
    rag_called_in_trigger=True is a re-retrieval false positive for the
    mechanism claim. Verify the three fields can be cross-referenced.
    """
    r = _make_run(
        "delayed_trigger", [],
        attack_success=True,
        memory_recalled_in_trigger=False,
        rag_called_in_trigger=True,
    )
    is_re_retrieval_fp = (
        r.attack_success
        and r.memory_recalled_in_trigger is False
        and r.rag_called_in_trigger is True
    )
    assert is_re_retrieval_fp, "Should be flagged as re-retrieval false positive"


def test_true_delayed_trigger_not_flagged_as_fp():
    """A run with memory recall AND RAG in trigger is not a re-retrieval FP (qwen pattern)."""
    r = _make_run(
        "delayed_trigger", [],
        attack_success=True,
        memory_recalled_in_trigger=True,
        rag_called_in_trigger=True,
    )
    is_re_retrieval_fp = (
        r.attack_success
        and r.memory_recalled_in_trigger is False
        and r.rag_called_in_trigger is True
    )
    assert not is_re_retrieval_fp, "Memory-mediated attack should not be flagged as FP"


# ── BTCR floor gate ───────────────────────────────────────────────────────────

def _make_runner_for_btcr_test() -> ExperimentRunner:
    config = _minimal_config(n_attacks=2, n_defenses=1, n_models=1, runs_per=1)
    with tempfile.TemporaryDirectory() as tmpdir:
        config.db_base_dir = tmpdir
    return ExperimentRunner(config)


def test_btcr_floor_passes_when_above_threshold():
    """_check_no_attack_btcr_floor does not raise when BTCR >= 90%."""
    runner = ExperimentRunner(_minimal_config())
    results = [
        _make_run("no_attack", [], btcr_success=True)
        for _ in range(20)
    ]
    # Should not raise
    runner._check_no_attack_btcr_floor(results, threshold=0.90)


def test_btcr_floor_raises_when_below_threshold():
    """_check_no_attack_btcr_floor raises RuntimeError when BTCR < 90%."""
    import pytest
    runner = ExperimentRunner(_minimal_config())
    # 5/20 passing = 25% BTCR
    results = (
        [_make_run("no_attack", [], btcr_success=True) for _ in range(5)]
        + [_make_run("no_attack", [], btcr_success=False) for _ in range(15)]
    )
    with pytest.raises(RuntimeError, match="BTCR"):
        runner._check_no_attack_btcr_floor(results, threshold=0.90)


def test_btcr_floor_skips_when_insufficient_data():
    """_check_no_attack_btcr_floor does not raise with fewer than 10 no-attack runs."""
    runner = ExperimentRunner(_minimal_config())
    # Only 5 runs — below the minimum for a meaningful check
    results = [_make_run("no_attack", [], btcr_success=False) for _ in range(5)]
    # Should not raise even though BTCR=0%
    runner._check_no_attack_btcr_floor(results, threshold=0.90)


def test_btcr_floor_ignores_dta_runs():
    """_check_no_attack_btcr_floor only counts no_attack runs, not DTA runs."""
    runner = ExperimentRunner(_minimal_config())
    # 20 DTA runs with btcr_success=False, 15 no-attack runs with btcr_success=True
    results = (
        [_make_run("delayed_trigger", [], btcr_success=False) for _ in range(20)]
        + [_make_run("no_attack", [], btcr_success=True) for _ in range(15)]
    )
    # Should not raise — DTA runs are excluded from the check
    runner._check_no_attack_btcr_floor(results, threshold=0.90)
