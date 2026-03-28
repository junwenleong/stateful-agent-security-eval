"""Property-based tests for statistical analysis modules (Properties 23-29)."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.stats.bootstrap_engine import BootstrapEngine, CIResult, ComparisonResult
from src.stats.meta_analyzer import MetaAnalyzer, MetaResult
from tests.strategies import (
    binary_vectors,
    effect_sizes,
    meta_entries,
    p_value_lists,
    proportions,
    sample_sizes,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _engine() -> BootstrapEngine:
    return BootstrapEngine(n_resamples=500, alpha=0.05, seed=42)


# ---------------------------------------------------------------------------
# Property 23: Bootstrap CI Mathematical Properties
# Validates: Requirements 10.1, 10.3
# ---------------------------------------------------------------------------

@given(binary_vectors)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_23_bootstrap_ci_bounds(outcomes):
    """**Validates: Requirements 10.1, 10.3**
    lower ≤ point_estimate ≤ upper, bounds in [0,1], warning if n<20.
    """
    engine = _engine()
    result = engine.compute_ci(outcomes)

    assert 0.0 <= result.lower <= result.point_estimate + 1e-9, (
        f"lower={result.lower} > point_estimate={result.point_estimate}"
    )
    assert result.point_estimate - 1e-9 <= result.upper + 1e-9, (
        f"point_estimate={result.point_estimate} > upper={result.upper}"
    )
    assert 0.0 <= result.lower <= 1.0, f"lower={result.lower} out of [0,1]"
    assert 0.0 <= result.upper <= 1.0, f"upper={result.upper} out of [0,1]"
    assert result.n_observations == len(outcomes)

    if len(outcomes) < 20:
        assert result.warning is not None, "Expected warning for n < 20"


def test_property_23_edge_case_all_zeros():
    """Edge case: all-0 vector falls back to Wilson Score CI."""
    engine = _engine()
    outcomes = np.zeros(10)
    result = engine.compute_ci(outcomes)
    assert result.point_estimate == 0.0
    assert result.lower == 0.0
    assert result.upper >= 0.0
    assert result.warning is not None


def test_property_23_edge_case_all_ones():
    """Edge case: all-1 vector falls back to Wilson Score CI."""
    engine = _engine()
    outcomes = np.ones(10)
    result = engine.compute_ci(outcomes)
    assert result.point_estimate == 1.0
    assert result.upper == pytest.approx(1.0, abs=1e-9)
    assert result.lower <= 1.0
    assert result.warning is not None


def test_property_23_edge_case_single_element():
    """Edge case: single element falls back to Wilson Score CI."""
    engine = _engine()
    for val in [0, 1]:
        outcomes = np.array([val])
        result = engine.compute_ci(outcomes)
        assert 0.0 <= result.lower <= result.upper <= 1.0
        assert result.warning is not None


# ---------------------------------------------------------------------------
# Property 24: Bootstrap Difference CI Properties
# Validates: Requirements 10.2
# ---------------------------------------------------------------------------

@given(binary_vectors, binary_vectors)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_24_diff_ci_bounds(outcomes_a, outcomes_b):
    """**Validates: Requirements 10.2**
    lower ≤ point_estimate ≤ upper, bounds in [-1,1].
    """
    engine = _engine()
    result = engine.compute_diff_ci(outcomes_a, outcomes_b)

    assert -1.0 <= result.lower <= 1.0, f"lower={result.lower} out of [-1,1]"
    assert -1.0 <= result.upper <= 1.0, f"upper={result.upper} out of [-1,1]"
    assert result.lower <= result.point_estimate + 1e-9, (
        f"lower={result.lower} > point_estimate={result.point_estimate}"
    )
    assert result.point_estimate <= result.upper + 1e-9, (
        f"point_estimate={result.point_estimate} > upper={result.upper}"
    )


# ---------------------------------------------------------------------------
# Property 25: Power Analysis Sample Size
# Validates: Requirements 11.1, 13.3
# ---------------------------------------------------------------------------

@given(effect_sizes)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_property_25_power_analysis_min_n(effect_size):
    """**Validates: Requirements 11.1, 13.3**
    min_n achieves specified power; min_n ≥ 1.
    """
    engine = _engine()
    result = engine.compute_power(effect_size=effect_size, alpha=0.05, power=0.80)
    assert result.min_n >= 1
    assert result.effect_size == effect_size
    assert result.alpha == 0.05
    assert result.power == 0.80


def test_property_25_extreme_rates():
    """Test power analysis with extreme baseline rates."""
    engine = _engine()
    for effect in [0.05, 0.10, 0.20]:
        result = engine.compute_power(effect_size=effect, alpha=0.05, power=0.80)
        assert result.min_n >= 1


# ---------------------------------------------------------------------------
# Property 26: Holm-Bonferroni Correction Properties
# Validates: Requirements 12.1
# ---------------------------------------------------------------------------

def _make_comparison(p: float, idx: int = 0) -> ComparisonResult:
    ci = CIResult(point_estimate=0.0, lower=-0.1, upper=0.1, n_resamples=500, n_observations=40)
    return ComparisonResult(
        condition_a=f"a{idx}",
        condition_b=f"b{idx}",
        diff_ci=ci,
        p_value=p,
    )


@given(p_value_lists)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_26_holm_bonferroni(p_values):
    """**Validates: Requirements 12.1**
    corrected_p_value ≥ p_value, corrected_p_value ≤ 1.0, step-down procedure.
    """
    engine = _engine()
    comparisons = [_make_comparison(p, i) for i, p in enumerate(p_values)]
    corrected = engine.holm_bonferroni(comparisons)

    assert len(corrected) == len(comparisons)
    for orig, corr in zip(comparisons, corrected):
        assert corr.corrected_p_value is not None
        assert corr.corrected_p_value >= orig.p_value - 1e-9, (
            f"corrected={corr.corrected_p_value} < original={orig.p_value}"
        )
        assert corr.corrected_p_value <= 1.0 + 1e-9, (
            f"corrected={corr.corrected_p_value} > 1.0"
        )


def test_property_26_holm_bonferroni_empty():
    """Empty list returns empty list."""
    engine = _engine()
    assert engine.holm_bonferroni([]) == []


def test_property_26_holm_bonferroni_single():
    """Single comparison: corrected == min(p * 1, 1.0)."""
    engine = _engine()
    comp = _make_comparison(0.03)
    result = engine.holm_bonferroni([comp])
    assert result[0].corrected_p_value == pytest.approx(min(0.03, 1.0), abs=1e-9)


# ---------------------------------------------------------------------------
# Property 27: Wilson Score CI Mathematical Properties
# Validates: Requirements 13.2
# ---------------------------------------------------------------------------

@given(sample_sizes, proportions)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_27_wilson_score_ci(n, p):
    """**Validates: Requirements 13.2**
    lower ≤ p ≤ upper, bounds in [0,1].
    """
    analyzer = MetaAnalyzer(alpha=0.05)
    lower, upper = analyzer.wilson_score_ci(n, p)
    assert 0.0 <= lower <= 1.0, f"lower={lower} out of [0,1]"
    assert 0.0 <= upper <= 1.0, f"upper={upper} out of [0,1]"
    assert lower <= p + 1e-9, f"lower={lower} > p={p}"
    assert p <= upper + 1e-9, f"p={p} > upper={upper}"


@given(proportions)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_27_wilson_width_decreases_with_n(p):
    """Interval width decreases as n increases."""
    analyzer = MetaAnalyzer(alpha=0.05)
    ns = [10, 50, 200, 1000]
    widths = [
        analyzer.wilson_score_ci(n, p)[1] - analyzer.wilson_score_ci(n, p)[0]
        for n in ns
    ]
    for i in range(len(widths) - 1):
        assert widths[i] >= widths[i + 1] - 1e-9, (
            f"Width did not decrease: n={ns[i]} width={widths[i]}, "
            f"n={ns[i+1]} width={widths[i+1]}"
        )


# ---------------------------------------------------------------------------
# Property 28: MetaResult Completeness
# Validates: Requirements 13.4, 13.6
# ---------------------------------------------------------------------------

@given(st.lists(meta_entries, min_size=1, max_size=20))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_28_meta_result_completeness(entries):
    """**Validates: Requirements 13.4, 13.6**
    Every MetaResult has required fields, ci_method='Wilson Score', valid verdict.
    """
    analyzer = MetaAnalyzer(alpha=0.05, power=0.80)
    results = analyzer.analyze(entries)

    assert len(results) == len(entries)
    for r in results:
        assert r.paper, "paper must be non-empty"
        assert r.claimed_finding, "claimed_finding must be non-empty"
        assert r.sample_size > 0, "sample_size must be > 0"
        assert r.ci_method == "Wilson Score", f"ci_method={r.ci_method!r}"
        assert r.verdict in {"supported", "underpowered", "inconclusive"}, (
            f"verdict={r.verdict!r} not in allowed set"
        )
        assert 0.0 <= r.wilson_ci_lower <= 1.0
        assert 0.0 <= r.wilson_ci_upper <= 1.0
        assert r.min_n_needed >= 1


# ---------------------------------------------------------------------------
# Property 29: Serialization Round-Trip
# Validates: Requirements 14.1, 14.2, 14.3
# ---------------------------------------------------------------------------

@given(st.lists(meta_entries, min_size=1, max_size=10))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_29_meta_result_json_roundtrip(entries):
    """**Validates: Requirements 14.1, 14.2, 14.3**
    JSON serialize/deserialize produces equivalent MetaResult objects.
    """

    analyzer = MetaAnalyzer(alpha=0.05, power=0.80)
    results = analyzer.analyze(entries)
    json_str = analyzer.to_json(results)

    # Deserialize
    raw = json.loads(json_str)
    assert len(raw) == len(results)

    for orig, d in zip(results, raw):
        reconstructed = MetaResult(**d)
        assert reconstructed.paper == orig.paper
        assert reconstructed.claimed_finding == orig.claimed_finding
        assert reconstructed.sample_size == orig.sample_size
        assert reconstructed.ci_method == orig.ci_method
        assert reconstructed.verdict == orig.verdict
        assert math.isclose(reconstructed.wilson_ci_lower, orig.wilson_ci_lower, abs_tol=1e-9)
        assert math.isclose(reconstructed.wilson_ci_upper, orig.wilson_ci_upper, abs_tol=1e-9)


@given(binary_vectors)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_property_29_ci_result_json_roundtrip(outcomes):
    """CIResult fields survive JSON round-trip."""
    import dataclasses

    engine = _engine()
    ci = engine.compute_ci(outcomes)
    d = dataclasses.asdict(ci)
    json_str = json.dumps(d)
    restored = json.loads(json_str)

    assert math.isclose(restored["point_estimate"], ci.point_estimate, abs_tol=1e-12)
    assert math.isclose(restored["lower"], ci.lower, abs_tol=1e-12)
    assert math.isclose(restored["upper"], ci.upper, abs_tol=1e-12)
    assert restored["n_observations"] == ci.n_observations
