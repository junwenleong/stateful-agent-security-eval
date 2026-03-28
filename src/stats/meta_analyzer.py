"""Meta-analyzer with Wilson Score CIs for published benchmark statistics."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Literal

from scipy import stats as scipy_stats


@dataclass
class MetaEntry:
    paper: str
    claimed_finding: str
    sample_size: int
    reported_asr: float


@dataclass
class MetaResult:
    paper: str
    claimed_finding: str
    sample_size: int
    reported_asr: float
    wilson_ci_lower: float
    wilson_ci_upper: float
    min_n_needed: int
    verdict: Literal["supported", "underpowered", "inconclusive"]
    ci_method: str = "Wilson Score"
    iid_assumption_note: str = (
        "Wilson Score CI assumes i.i.d. Bernoulli trials. "
        "Published benchmarks may violate i.i.d. due to correlated prompts, "
        "shared model checkpoints, or non-random sampling."
    )


class MetaAnalyzer:
    def __init__(self, alpha: float = 0.05, power: float = 0.80):
        self.alpha = alpha
        self.power = power

    def wilson_score_ci(self, n: int, p: float) -> tuple[float, float]:
        """Wilson Score CI for a proportion."""
        if n <= 0:
            return 0.0, 1.0
        z = scipy_stats.norm.ppf(1 - self.alpha / 2)
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        margin = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
        return max(0.0, centre - margin), min(1.0, centre + margin)

    def min_sample_size(self, p1: float, p2: float) -> int:
        """Minimum N per group for two-proportion z-test at configured power/alpha."""
        from statsmodels.stats.power import NormalIndPower
        import numpy as np

        # Cohen's h effect size
        h = abs(2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2)))
        if h < 1e-9:
            return 10**6  # infinite — no detectable difference

        analysis = NormalIndPower()
        n = analysis.solve_power(
            effect_size=h,
            alpha=self.alpha,
            power=self.power,
            alternative="two-sided",
        )
        return max(1, int(math.ceil(n)))

    def analyze(self, entries: list[MetaEntry]) -> list[MetaResult]:
        results = []
        for entry in entries:
            lower, upper = self.wilson_score_ci(entry.sample_size, entry.reported_asr)
            # Baseline: compare reported ASR vs 0.5 (chance)
            baseline = 0.5
            min_n = self.min_sample_size(entry.reported_asr, baseline)

            if entry.sample_size >= min_n:
                verdict: Literal["supported", "underpowered", "inconclusive"] = "supported"
            elif entry.sample_size < min_n // 2:
                verdict = "underpowered"
            else:
                verdict = "inconclusive"

            results.append(MetaResult(
                paper=entry.paper,
                claimed_finding=entry.claimed_finding,
                sample_size=entry.sample_size,
                reported_asr=entry.reported_asr,
                wilson_ci_lower=lower,
                wilson_ci_upper=upper,
                min_n_needed=min_n,
                verdict=verdict,
            ))
        return results

    def to_latex(self, results: list[MetaResult]) -> str:
        lines = [
            r"\begin{tabular}{lllrrrrll}",
            r"\hline",
            r"Paper & Finding & N & ASR & CI Lower & CI Upper & Min N & Verdict \\",
            r"\hline",
        ]
        for r in results:
            paper = r.paper.replace("_", r"\_").replace("&", r"\&")
            finding = r.claimed_finding[:40].replace("_", r"\_").replace("&", r"\&")
            lines.append(
                f"{paper} & {finding} & {r.sample_size} & "
                f"{r.reported_asr:.3f} & {r.wilson_ci_lower:.3f} & "
                f"{r.wilson_ci_upper:.3f} & {r.min_n_needed} & {r.verdict} \\\\"
            )
        lines += [r"\hline", r"\end{tabular}"]
        return "\n".join(lines)

    def to_json(self, results: list[MetaResult]) -> str:
        import dataclasses
        return json.dumps([dataclasses.asdict(r) for r in results], indent=2)
