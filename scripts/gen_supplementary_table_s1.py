#!/usr/bin/env python3
"""Generate Supplementary Table S1 from analysis.json.

Each comparison is rendered with the point estimate and CI for the metric that
comparison is actually scored on: BTCR for no-attack (btcr_success) rows, ASR for
attack (attack_success) rows. This avoids the earlier defect where the no-attack
rows displayed a column headed "ASR" while the significance flag was computed on
BTCR -- the displayed quantity did not match the tested quantity.

Reads : results/defense_factorial/analysis.json
Writes: results/defense_factorial/supplementary_table_s1.md
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "results" / "defense_factorial" / "analysis.json"
OUT = ROOT / "results" / "defense_factorial" / "supplementary_table_s1.md"

ATTACK_ABBR = {"delayed_trigger": "DTA", "no_attack": "NA"}
METRIC_LABEL = {"attack_success": "ASR", "btcr_success": "BTCR"}


def parse_condition(key: str) -> dict[str, str]:
    """'attack=delayed_trigger,defense=memory_sandbox,model=qwq:32b' -> dict."""
    out = {}
    for part in key.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
    return out


def short(key: str) -> str:
    c = parse_condition(key)
    attack = ATTACK_ABBR.get(c.get("attack", ""), c.get("attack", "?"))
    return f"{attack}/{c.get('defense', '?')}/{c.get('model', '?')}"


def rate_cell(stats: dict, cond_key: str, metric: str) -> str:
    """Point estimate + 95% CI for the metric this comparison is scored on."""
    field = "btcr" if metric == "btcr_success" else "asr"
    s = stats[cond_key][field]
    return f"{s['point_estimate']:.3f} [{s['lower']:.3f}, {s['upper']:.3f}]"


def existing_na_reasons() -> dict[tuple[str, str], str]:
    """Preserve the richer N/A reason text already committed in the table, if present."""
    if not OUT.exists():
        return {}
    text = OUT.read_text()
    if "## N/A Comparisons" not in text:
        return {}
    na_sec = text.split("## N/A Comparisons", 1)[1]
    out: dict[tuple[str, str], str] = {}
    for line in na_sec.splitlines():
        if not line.startswith("| "):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) >= 3 and cells[0] not in ("Condition A", "------------"):
            out[(cells[0], cells[1])] = cells[2]
    return out


def main() -> None:
    data = json.loads(ANALYSIS.read_text())
    stats = data["stats"]
    comparisons = data["comparisons"]
    na_text = existing_na_reasons()

    active = [c for c in comparisons if not c.get("na_reason")]
    na = [c for c in comparisons if c.get("na_reason")]
    n_sig = sum(1 for c in active if str(c.get("significant_holm")) == "True")

    # Reader-friendly ordering: Holm-significant comparisons first (stable),
    # then the remaining comparisons in their original analysis.json order.
    active = sorted(
        active,
        key=lambda c: 0 if str(c.get("significant_holm")) == "True" else 1,
    )

    lines: list[str] = []
    lines.append("# Supplementary Table S1: Full Holm-Bonferroni Comparison Table")
    lines.append("")
    lines.append(
        f"**{len(active)} active pre-specified comparisons** "
        f"({len(comparisons)} total; {len(na)} annotated N/A for qwq:32b)."
    )
    lines.append(
        "Significance criterion: Holm-Bonferroni step-down correction (alpha=0.05) applied to the"
    )
    lines.append(
        "per-comparison Fisher exact p-value on each 2x2 (successes, failures) contingency table; a"
    )
    lines.append(
        "comparison is significant when its Holm-corrected Fisher p is at most alpha. As a concordant"
    )
    lines.append(
        "sign check we also record whether the 95% difference interval excludes zero (BCa when both arms"
    )
    lines.append(
        "are non-degenerate, else the conservative Wilson-difference bound); the two criteria agree on all"
    )
    lines.append(
        f"{len(active)} comparisons. This matches `scripts/analyze_results.py` and `analysis.json`, which score the"
    )
    lines.append(
        "no-attack comparisons on `btcr_success` and the attack comparisons on `attack_success`."
    )
    lines.append("")
    lines.append(
        "The **Metric** column states the quantity scored for each row: ASR (attack success rate,"
    )
    lines.append(
        "attack arm) or BTCR (benign task completion rate, no-attack arm). The Rate_A / Rate_B columns"
    )
    lines.append(
        "report the point estimate and 95% CI of that metric for each condition, so the displayed value"
    )
    lines.append(
        "is the same quantity the significance flag tests. A trailing `*` on the Holm flag marks a"
    )
    lines.append("Holm-significant comparison.")
    lines.append("")
    lines.append(f"**Summary**: {n_sig} significant / {len(active)} active comparisons.")
    lines.append("")
    lines.append(
        "| # | Metric | Condition A | Rate_A [95% CI] | Condition B | Rate_B [95% CI] | Sig (raw) | Sig (Holm) |"
    )
    lines.append(
        "|---|--------|------------|-----------------|------------|-----------------|-----------|------------|"
    )

    for i, c in enumerate(active, start=1):
        metric = c["metric"]
        mlabel = METRIC_LABEL.get(metric, metric)
        a = short(c["condition_a"])
        b = short(c["condition_b"])
        ra = rate_cell(stats, c["condition_a"], metric)
        rb = rate_cell(stats, c["condition_b"], metric)
        raw = "\u2713" if str(c.get("significant_pre_correction")) == "True" else "\u2717"
        holm_sig = str(c.get("significant_holm")) == "True"
        holm = ("\u2713*" if holm_sig else "\u2717")
        lines.append(f"| {i} | {mlabel} | {a} | {ra} | {b} | {rb} | {raw} | {holm} |")

    lines.append("")
    lines.append(f"## N/A Comparisons (qwq:32b, {len(na)} total)")
    lines.append("")
    lines.append("| Condition A | Condition B | Reason |")
    lines.append("|------------|------------|--------|")
    for c in na:
        key = (short(c["condition_a"]), short(c["condition_b"]))
        reason = na_text.get(key, c.get("na_reason", ""))
        lines.append(f"| {short(c['condition_a'])} | {short(c['condition_b'])} | {reason} |")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT} ({len(active)} active rows, {n_sig} significant, {len(na)} N/A)")


if __name__ == "__main__":
    main()
