"""Results aggregation, ASR calculation, CI computation, JSONL output."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from persistbench.verifiers.tool_call import VerificationResult


@dataclass
class ASREstimate:
    """Attack Success Rate with Wilson score confidence interval."""
    successes: int
    total: int
    asr: float
    ci_lower: float
    ci_upper: float
    z: float = 1.96  # 95% CI

    def to_dict(self) -> dict[str, Any]:
        return {
            "successes": self.successes,
            "total": self.total,
            "asr": round(self.asr, 4),
            "ci_lower": round(self.ci_lower, 4),
            "ci_upper": round(self.ci_upper, 4),
            "ci_level": 0.95,
        }


def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Compute Wilson score confidence interval for a proportion."""
    if total == 0:
        return (0.0, 0.0)
    p_hat = successes / total
    denominator = 1 + z * z / total
    center = (p_hat + z * z / (2 * total)) / denominator
    spread = (z / denominator) * math.sqrt(p_hat * (1 - p_hat) / total + z * z / (4 * total * total))
    return (max(0.0, center - spread), min(1.0, center + spread))


def compute_asr(results: list[VerificationResult]) -> ASREstimate:
    """Compute ASR with Wilson CI from verification results."""
    total = len(results)
    successes = sum(1 for r in results if r.attack_success)
    asr = successes / total if total > 0 else 0.0
    ci_lower, ci_upper = wilson_ci(successes, total)
    return ASREstimate(
        successes=successes, total=total, asr=asr,
        ci_lower=ci_lower, ci_upper=ci_upper,
    )


@dataclass
class BenchmarkReport:
    """Aggregated benchmark results."""
    total_runs: int = 0
    overall_asr: ASREstimate | None = None
    by_attack: dict[str, ASREstimate] = field(default_factory=dict)
    by_defense: dict[str, ASREstimate] = field(default_factory=dict)
    by_model: dict[str, ASREstimate] = field(default_factory=dict)
    by_attack_defense: dict[str, ASREstimate] = field(default_factory=dict)
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_runs": self.total_runs,
            "overall_asr": self.overall_asr.to_dict() if self.overall_asr else None,
            "by_attack": {k: v.to_dict() for k, v in self.by_attack.items()},
            "by_defense": {k: v.to_dict() for k, v in self.by_defense.items()},
            "by_model": {k: v.to_dict() for k, v in self.by_model.items()},
            "by_attack_defense": {k: v.to_dict() for k, v in self.by_attack_defense.items()},
            "duration_seconds": round(self.duration_seconds, 2),
        }

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"PersistBench Results Summary",
            f"{'=' * 40}",
            f"Total runs: {self.total_runs}",
            f"Overall ASR: {self.overall_asr.asr:.1%} [{self.overall_asr.ci_lower:.1%}, {self.overall_asr.ci_upper:.1%}]"
            if self.overall_asr else "No results",
            "",
            "By Attack:",
        ]
        for name, asr in self.by_attack.items():
            lines.append(f"  {name}: {asr.asr:.1%} [{asr.ci_lower:.1%}, {asr.ci_upper:.1%}] (N={asr.total})")
        lines.append("")
        lines.append("By Defense:")
        for name, asr in self.by_defense.items():
            lines.append(f"  {name}: {asr.asr:.1%} [{asr.ci_lower:.1%}, {asr.ci_upper:.1%}] (N={asr.total})")
        if self.by_model:
            lines.append("")
            lines.append("By Model:")
            for name, asr in self.by_model.items():
                lines.append(f"  {name}: {asr.asr:.1%} [{asr.ci_lower:.1%}, {asr.ci_upper:.1%}] (N={asr.total})")
        lines.append(f"\nDuration: {self.duration_seconds:.1f}s")
        return "\n".join(lines)


def generate_report(results: list[VerificationResult], duration: float = 0.0) -> BenchmarkReport:
    """Generate a full benchmark report from verification results."""
    report = BenchmarkReport(total_runs=len(results), duration_seconds=duration)

    if not results:
        return report

    report.overall_asr = compute_asr(results)

    # Group by attack
    by_attack: dict[str, list[VerificationResult]] = {}
    by_defense: dict[str, list[VerificationResult]] = {}
    by_model: dict[str, list[VerificationResult]] = {}
    by_attack_defense: dict[str, list[VerificationResult]] = {}

    for r in results:
        by_attack.setdefault(r.attack_name, []).append(r)
        by_defense.setdefault(r.defense_name, []).append(r)
        by_model.setdefault(r.model_id, []).append(r)
        key = f"{r.attack_name}+{r.defense_name}"
        by_attack_defense.setdefault(key, []).append(r)

    report.by_attack = {k: compute_asr(v) for k, v in by_attack.items()}
    report.by_defense = {k: compute_asr(v) for k, v in by_defense.items()}
    report.by_model = {k: compute_asr(v) for k, v in by_model.items()}
    report.by_attack_defense = {k: compute_asr(v) for k, v in by_attack_defense.items()}

    return report


def save_report(report: BenchmarkReport, output_dir: str | Path) -> None:
    """Save report as JSON and human-readable text."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "report.json", "w") as f:
        json.dump(report.to_dict(), f, indent=2)

    with open(output_dir / "report.txt", "w") as f:
        f.write(report.summary())
