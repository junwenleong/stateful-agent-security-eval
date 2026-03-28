"""Bootstrap CI engine with BCa CIs, power analysis, and Holm-Bonferroni correction."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


@dataclass
class CIResult:
    point_estimate: float
    lower: float
    upper: float
    n_resamples: int
    n_observations: int
    warning: Optional[str] = None


@dataclass
class ComparisonResult:
    condition_a: str
    condition_b: str
    diff_ci: CIResult
    p_value: float
    corrected_p_value: Optional[float] = None
    is_underpowered: bool = False


@dataclass
class PowerResult:
    min_n: int
    effect_size: float
    alpha: float
    power: float
    actual_n: Optional[int] = None
    is_underpowered: bool = False


def _wilson_score_ci(n: int, p: float, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson Score CI for a proportion."""
    z = scipy_stats.norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


class BootstrapEngine:
    def __init__(self, n_resamples: int = 10000, alpha: float = 0.05, seed: int = 42):
        self.n_resamples = n_resamples
        self.alpha = alpha
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def compute_ci(self, outcomes: np.ndarray) -> CIResult:
        """BCa CI for a proportion. Falls back to Wilson Score for degenerate cases."""
        n = len(outcomes)
        point = float(np.mean(outcomes))
        warning: Optional[str] = None

        if n < 20:
            warning = f"n={n} < 20; CI may be unreliable"

        # Degenerate: all-0 or all-1
        if np.all(outcomes == 0) or np.all(outcomes == 1):
            logger.info("Degenerate vector (all-%d); falling back to Wilson Score CI", int(point))
            lower, upper = _wilson_score_ci(max(n, 1), point, self.alpha)
            w = "Degenerate vector; Wilson Score CI used" + (f"; {warning}" if warning else "")
            return CIResult(
                point_estimate=point,
                lower=lower,
                upper=upper,
                n_resamples=0,
                n_observations=n,
                warning=w,
            )

        # Single element — can't bootstrap
        if n == 1:
            lower, upper = _wilson_score_ci(1, point, self.alpha)
            w = "n=1; Wilson Score CI used" + (f"; {warning}" if warning else "")
            return CIResult(
                point_estimate=point,
                lower=lower,
                upper=upper,
                n_resamples=0,
                n_observations=n,
                warning=w,
            )

        try:
            rng = np.random.default_rng(self.seed)
            result = scipy_stats.bootstrap(
                (outcomes,),
                statistic=np.mean,
                n_resamples=self.n_resamples,
                confidence_level=1 - self.alpha,
                method="BCa",
                random_state=rng,
            )
            lower = float(result.confidence_interval.low)
            upper = float(result.confidence_interval.high)
            # Clamp to [0, 1]
            lower = max(0.0, lower)
            upper = min(1.0, upper)
        except Exception as exc:
            logger.warning("BCa failed (%s); falling back to Wilson Score CI", exc)
            lower, upper = _wilson_score_ci(n, point, self.alpha)
            w = "BCa failed; Wilson Score CI used" + (f"; {warning}" if warning else "")
            warning = w

        return CIResult(
            point_estimate=point,
            lower=lower,
            upper=upper,
            n_resamples=self.n_resamples,
            n_observations=n,
            warning=warning,
        )

    def compute_diff_ci(self, outcomes_a: np.ndarray, outcomes_b: np.ndarray) -> CIResult:
        """BCa CI for difference in proportions (a - b)."""
        point = float(np.mean(outcomes_a)) - float(np.mean(outcomes_b))
        n_a, n_b = len(outcomes_a), len(outcomes_b)
        warning: Optional[str] = None

        if n_a < 20 or n_b < 20:
            warning = f"n_a={n_a}, n_b={n_b}; one or both < 20"

        try:
            rng = np.random.default_rng(self.seed)
            result = scipy_stats.bootstrap(
                (outcomes_a, outcomes_b),
                statistic=lambda a, b: np.mean(a) - np.mean(b),
                n_resamples=self.n_resamples,
                confidence_level=1 - self.alpha,
                method="BCa",
                random_state=rng,
                paired=False,
            )
            lower = float(result.confidence_interval.low)
            upper = float(result.confidence_interval.high)
            lower = max(-1.0, lower)
            upper = min(1.0, upper)
        except Exception as exc:
            logger.warning("BCa diff CI failed (%s); using percentile fallback", exc)
            # Percentile bootstrap fallback
            rng2 = np.random.default_rng(self.seed)
            diffs = [
                np.mean(rng2.choice(outcomes_a, size=n_a, replace=True))
                - np.mean(rng2.choice(outcomes_b, size=n_b, replace=True))
                for _ in range(self.n_resamples)
            ]
            lower = float(np.percentile(diffs, 100 * self.alpha / 2))
            upper = float(np.percentile(diffs, 100 * (1 - self.alpha / 2)))
            lower = max(-1.0, lower)
            upper = min(1.0, upper)
            warning = (warning or "") + "; BCa failed, percentile used"

        return CIResult(
            point_estimate=point,
            lower=lower,
            upper=upper,
            n_resamples=self.n_resamples,
            n_observations=n_a + n_b,
            warning=warning,
        )

    def compute_power(
        self,
        effect_size: float = 0.10,
        alpha: float = 0.05,
        power: float = 0.80,
        baseline_rate: float = 0.5,
    ) -> PowerResult:
        """Minimum N for two-proportion z-test using statsmodels NormalIndPower."""
        from statsmodels.stats.power import NormalIndPower

        analysis = NormalIndPower()
        min_n = analysis.solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            alternative="two-sided",
        )
        min_n = max(1, int(np.ceil(min_n)))
        return PowerResult(
            min_n=min_n,
            effect_size=effect_size,
            alpha=alpha,
            power=power,
        )

    def holm_bonferroni(self, comparisons: list[ComparisonResult]) -> list[ComparisonResult]:
        """Step-down Holm-Bonferroni correction on the provided comparison set."""
        if not comparisons:
            return comparisons

        k = len(comparisons)
        # Sort ascending by p_value
        indexed = sorted(enumerate(comparisons), key=lambda x: x[1].p_value)

        corrected = [None] * k
        running_max = 0.0
        for rank, (orig_idx, comp) in enumerate(indexed):
            multiplier = k - rank  # (k - i + 1) where i is 1-based rank
            adjusted = comp.p_value * multiplier
            running_max = max(running_max, adjusted)
            capped = min(1.0, running_max)
            import dataclasses
            corrected[orig_idx] = dataclasses.replace(comp, corrected_p_value=capped)

        return corrected

    def analyze_experiment(self, results: list[dict], comparisons: list[dict]) -> dict:
        """Full pipeline: per-condition CIs, diff CIs, power, Holm-Bonferroni."""

        # Per-condition CIs
        condition_stats: dict[str, dict] = {}
        for condition, outcomes in results:
            arr = np.array(outcomes, dtype=float)
            ci = self.compute_ci(arr)
            condition_stats[condition] = {
                "point_estimate": ci.point_estimate,
                "lower": ci.lower,
                "upper": ci.upper,
                "n_observations": ci.n_observations,
                "warning": ci.warning,
            }

        # Diff CIs and comparison results
        comparison_results: list[ComparisonResult] = []
        for comp in comparisons:
            a_name = comp["condition_a"]
            b_name = comp["condition_b"]
            outcomes_a = np.array(next(o for c, o in results if c == a_name), dtype=float)
            outcomes_b = np.array(next(o for c, o in results if c == b_name), dtype=float)
            diff_ci = self.compute_diff_ci(outcomes_a, outcomes_b)
            # Simple z-test p-value
            n_a, n_b = len(outcomes_a), len(outcomes_b)
            p_a = np.mean(outcomes_a)
            p_b = np.mean(outcomes_b)
            p_pool = (np.sum(outcomes_a) + np.sum(outcomes_b)) / (n_a + n_b)
            se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
            z = (p_a - p_b) / se if se > 0 else 0.0
            p_value = float(2 * (1 - scipy_stats.norm.cdf(abs(z))))
            comparison_results.append(ComparisonResult(
                condition_a=a_name,
                condition_b=b_name,
                diff_ci=diff_ci,
                p_value=p_value,
            ))

        corrected = self.holm_bonferroni(comparison_results)

        return {
            "condition_stats": condition_stats,
            "comparisons": [
                {
                    "condition_a": c.condition_a,
                    "condition_b": c.condition_b,
                    "diff_lower": c.diff_ci.lower,
                    "diff_upper": c.diff_ci.upper,
                    "diff_point": c.diff_ci.point_estimate,
                    "p_value": c.p_value,
                    "corrected_p_value": c.corrected_p_value,
                }
                for c in corrected
            ],
        }
