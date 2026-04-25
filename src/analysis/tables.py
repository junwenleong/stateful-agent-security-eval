"""LaTeX table generation for statistical results and meta-analysis."""
from __future__ import annotations


def _escape(s: str) -> str:
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def render_stats_table(stats: dict, comparisons: list[dict]) -> str:
    """Render statistical results as a LaTeX tabular.

    Columns: Condition, ASR (CI), BTCR (CI), p-value, corrected p-value

    stats format: {"condition_name": {"asr": {"point_estimate", "lower", "upper"},
                                       "btcr": {"point_estimate", "lower", "upper"}}}
    comparisons: list of dicts with condition_a, condition_b, p_value, corrected_p_value
    """
    lines = [
        r"\begin{tabular}{lllll}",
        r"\hline",
        r"Condition & ASR (95\% CI) & BTCR (95\% CI) & $p$ & $p_{\text{corr}}$ \\",
        r"\hline",
    ]

    # Build a lookup: condition -> comparison row (use condition_a as primary)
    comp_lookup: dict[str, dict] = {}
    for c in comparisons:
        comp_lookup.setdefault(c.get("condition_a", ""), c)
        comp_lookup.setdefault(c.get("condition_b", ""), c)

    for condition, data in stats.items():
        asr = data.get("asr", {})
        btcr = data.get("btcr", {})

        asr_str = (
            f"{asr.get('point_estimate', 0):.3f} "
            f"[{asr.get('lower', 0):.3f}, {asr.get('upper', 0):.3f}]"
            if asr else "—"
        )
        btcr_str = (
            f"{btcr.get('point_estimate', 0):.3f} "
            f"[{btcr.get('lower', 0):.3f}, {btcr.get('upper', 0):.3f}]"
            if btcr else "—"
        )

        comp = comp_lookup.get(condition)
        p_str = f"{comp['p_value']:.4f}" if comp and comp.get("p_value") is not None else "—"
        pc_str = (
            f"{comp['corrected_p_value']:.4f}"
            if comp and comp.get("corrected_p_value") is not None
            else "—"
        )

        lines.append(
            f"{_escape(condition)} & {asr_str} & {btcr_str} & {p_str} & {pc_str} \\\\"
        )

    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines)


def render_meta_table(meta_results: list) -> str:
    """Render meta-analysis results as a LaTeX tabular.

    Columns: Paper, Finding, N, ASR, CI, Min N, Verdict
    meta_results: list of MetaResult dataclass instances or dicts.
    """
    lines = [
        r"\begin{tabular}{lllrrrrl}",
        r"\hline",
        r"Paper & Finding & N & ASR & CI Lower & CI Upper & Min $N$ & Verdict \\",
        r"\hline",
    ]

    for r in meta_results:
        # Support both dataclass and dict
        def _get(attr: str) -> object:
            return getattr(r, attr) if hasattr(r, attr) else r[attr]

        paper = _escape(str(_get("paper")))
        finding = _escape(str(_get("claimed_finding"))[:50])
        n = int(_get("sample_size"))
        asr = float(_get("reported_asr"))
        lo = float(_get("wilson_ci_lower"))
        hi = float(_get("wilson_ci_upper"))
        min_n = int(_get("min_n_needed"))
        verdict = _escape(str(_get("verdict")))

        lines.append(
            f"{paper} & {finding} & {n} & {asr:.3f} & {lo:.3f} & {hi:.3f} & {min_n} & {verdict} \\\\"
        )

    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines)
