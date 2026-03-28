"""Plot generation functions for the security evaluation framework."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _use_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")


def plot_asr_by_condition(stats: dict, output_path: str) -> None:
    """Bar chart: ASR per condition with BCa CI error bars.

    stats format: {"condition_name": {"asr": {"point_estimate": float, "lower": float, "upper": float}}}
    """
    if not stats:
        fig, ax = plt.subplots()
        ax.set_title("ASR by Condition (no data)")
        plt.savefig(output_path)
        plt.close()
        return

    _use_style()
    conditions = list(stats.keys())
    points = [stats[c]["asr"]["point_estimate"] for c in conditions]
    lowers = [stats[c]["asr"]["point_estimate"] - stats[c]["asr"]["lower"] for c in conditions]
    uppers = [stats[c]["asr"]["upper"] - stats[c]["asr"]["point_estimate"] for c in conditions]

    x = np.arange(len(conditions))
    fig, ax = plt.subplots(figsize=(max(6, len(conditions) * 0.8), 5))
    ax.bar(x, points, yerr=[lowers, uppers], capsize=4, color="steelblue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha="right")
    ax.set_ylabel("ASR")
    ax.set_ylim(0, 1)
    ax.set_title("Attack Success Rate by Condition")
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_btcr_by_condition(stats: dict, output_path: str) -> None:
    """Bar chart: BTCR per condition with BCa CI error bars.

    stats format: {"condition_name": {"btcr": {"point_estimate": float, "lower": float, "upper": float}}}
    """
    if not stats:
        fig, ax = plt.subplots()
        ax.set_title("BTCR by Condition (no data)")
        plt.savefig(output_path)
        plt.close()
        return

    _use_style()
    conditions = list(stats.keys())
    points = [stats[c]["btcr"]["point_estimate"] for c in conditions]
    lowers = [stats[c]["btcr"]["point_estimate"] - stats[c]["btcr"]["lower"] for c in conditions]
    uppers = [stats[c]["btcr"]["upper"] - stats[c]["btcr"]["point_estimate"] for c in conditions]

    x = np.arange(len(conditions))
    fig, ax = plt.subplots(figsize=(max(6, len(conditions) * 0.8), 5))
    ax.bar(x, points, yerr=[lowers, uppers], capsize=4, color="seagreen", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha="right")
    ax.set_ylabel("BTCR")
    ax.set_ylim(0, 1)
    ax.set_title("Benign Task Completion Rate by Condition")
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_asr_vs_btcr(stats: dict, output_path: str) -> None:
    """Scatter plot: ASR (x) vs BTCR (y) per condition, labeled with condition name."""
    if not stats:
        fig, ax = plt.subplots()
        ax.set_title("ASR vs BTCR (no data)")
        plt.savefig(output_path)
        plt.close()
        return

    _use_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    for condition, data in stats.items():
        asr = data["asr"]["point_estimate"]
        btcr = data["btcr"]["point_estimate"]
        ax.scatter(asr, btcr, s=80, zorder=3)
        ax.annotate(condition, (asr, btcr), textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_xlabel("ASR (Attack Success Rate)")
    ax.set_ylabel("BTCR (Benign Task Completion Rate)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Security–Utility Tradeoff")
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_factorial_grid(stats: dict, output_path: str) -> None:
    """3×3 grid plot (rows=attacks, cols=models, hue=defense) for ASR with CI error bars.

    stats format: {"attack_name": {"model_name": {"defense_name": {"asr": {...}}}}}
    """
    if not stats:
        fig, ax = plt.subplots()
        ax.set_title("Factorial Grid (no data)")
        plt.savefig(output_path)
        plt.close()
        return

    _use_style()
    attacks = list(stats.keys())
    models = list({m for a in stats.values() for m in a.keys()})
    defenses = list({d for a in stats.values() for m in a.values() for d in m.keys()})

    n_rows, n_cols = len(attacks), len(models)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5), squeeze=False)

    colors = plt.cm.tab10.colors
    x = np.arange(len(defenses))
    width = 0.6

    for r, attack in enumerate(attacks):
        for c, model in enumerate(models):
            ax = axes[r][c]
            cell = stats.get(attack, {}).get(model, {})
            points, lowers, uppers = [], [], []
            for defense in defenses:
                d = cell.get(defense, {}).get("asr", {"point_estimate": 0, "lower": 0, "upper": 0})
                points.append(d["point_estimate"])
                lowers.append(d["point_estimate"] - d["lower"])
                uppers.append(d["upper"] - d["point_estimate"])

            bars = ax.bar(x, points, width, yerr=[lowers, uppers], capsize=3,
                          color=[colors[i % len(colors)] for i in range(len(defenses))], alpha=0.8)
            ax.set_ylim(0, 1)
            ax.set_xticks(x)
            ax.set_xticklabels(defenses, rotation=30, ha="right", fontsize=7)
            if r == 0:
                ax.set_title(model, fontsize=9)
            if c == 0:
                ax.set_ylabel(attack, fontsize=8)

    fig.suptitle("ASR: Factorial Grid (rows=attacks, cols=models, bars=defenses)", fontsize=11)
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_meta_analysis(meta_results: list, output_path: str) -> None:
    """Forest plot of Wilson Score CIs for published benchmarks.

    Each row is a paper; x-axis is ASR; horizontal bars show CI.
    meta_results: list of MetaResult (or dicts with paper, reported_asr, wilson_ci_lower, wilson_ci_upper, verdict)
    """
    if not meta_results:
        fig, ax = plt.subplots()
        ax.set_title("Meta-Analysis (no data)")
        plt.savefig(output_path)
        plt.close()
        return

    _use_style()
    n = len(meta_results)
    fig, ax = plt.subplots(figsize=(8, max(3, n * 0.6 + 1)))

    verdict_colors = {"supported": "steelblue", "underpowered": "tomato", "inconclusive": "goldenrod"}

    for i, r in enumerate(meta_results):
        # Support both dataclass and dict
        paper = r.paper if hasattr(r, "paper") else r["paper"]
        asr = r.reported_asr if hasattr(r, "reported_asr") else r["reported_asr"]
        lo = r.wilson_ci_lower if hasattr(r, "wilson_ci_lower") else r["wilson_ci_lower"]
        hi = r.wilson_ci_upper if hasattr(r, "wilson_ci_upper") else r["wilson_ci_upper"]
        verdict = r.verdict if hasattr(r, "verdict") else r.get("verdict", "inconclusive")

        color = verdict_colors.get(verdict, "gray")
        y = n - 1 - i
        ax.plot([lo, hi], [y, y], color=color, linewidth=2)
        ax.scatter([asr], [y], color=color, s=50, zorder=3)

    ax.set_yticks(range(n))
    labels = [
        (r.paper if hasattr(r, "paper") else r["paper"])
        for r in reversed(meta_results)
    ]
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("ASR")
    ax.set_xlim(0, 1)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_title("Meta-Analysis: Wilson Score CIs for Published Benchmarks")

    # Legend
    for verdict, color in verdict_colors.items():
        ax.scatter([], [], color=color, label=verdict, s=40)
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    plt.savefig(output_path)
    plt.close()
