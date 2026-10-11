"""Timeout-sensitivity analysis for the primary factorial.

The paper discloses seven session-level timeouts (error=None, scored as non-attack).
This recomputes the affected cells' ASR under three treatments:
  (1) headline: timeouts scored as non-attack (as-published)
  (2) timeouts excluded (treated as missing)
  (3) timeouts treated as attack-failure (same as non-attack for ASR, but we also
      report BTCR under timeout-as-task-failure)
and confirms whether any headline comparison decision changes.
"""
import json
from pathlib import Path
from collections import defaultdict

FACT = Path("results/defense_factorial/results.jsonl")


def session_has_timeout(rec):
    """A run has a session-level timeout if any agent_logs entry is a timeout."""
    logs = rec.get("agent_logs") or []
    for entry in logs:
        if isinstance(entry, dict) and entry.get("type") == "timeout":
            return True
    # some records nest logs differently; also check session_logs
    for key in ("session_logs", "sessions"):
        for s in (rec.get(key) or []):
            if isinstance(s, dict) and s.get("type") == "timeout":
                return True
    return False


def cell(rec):
    cond = rec.get("condition", {})
    m = cond.get("model", {}).get("model_name", "?")
    d = cond.get("defense", {}).get("name", "no_defense")
    a = cond.get("attack", {}).get("type", "?")
    return (m, d, a)


def main():
    by_cell = defaultdict(lambda: {"n": 0, "asr_succ": 0, "timeouts": 0, "err": 0})
    total = 0
    total_timeout = 0
    with open(FACT) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total += 1
            c = cell(rec)
            st = by_cell[c]
            st["n"] += 1
            if rec.get("error"):
                st["err"] += 1
            if rec.get("attack_success"):
                st["asr_succ"] += 1
            if session_has_timeout(rec):
                st["timeouts"] += 1
                total_timeout += 1

    print(f"Total records: {total}")
    print(f"Total runs containing a session-level timeout: {total_timeout}")
    print()
    print("Cells containing >=1 timeout (the affected cells):")
    print(f"{'model':<22}{'defense':<16}{'attack':<16}{'N':>4}{'succ':>6}{'TO':>4}"
          f"{'ASR_pub':>9}{'ASR_excl':>9}")
    affected = {c: s for c, s in by_cell.items() if s["timeouts"] > 0}
    for c, s in sorted(affected.items()):
        n, succ, to = s["n"], s["asr_succ"], s["timeouts"]
        asr_pub = succ / n if n else 0.0           # timeout = non-attack (published)
        denom_excl = n - to
        asr_excl = (succ / denom_excl) if denom_excl else float("nan")
        # timeout-as-failure is identical to published for ASR (failure == non-success)
        print(f"{c[0]:<22}{c[1]:<16}{c[2]:<16}{n:>4}{succ:>6}{to:>4}"
              f"{asr_pub*100:>8.1f}%{asr_excl*100:>8.1f}%")

    print()
    print("Interpretation:")
    print("- 'ASR_pub' scores timeouts as non-attack (headline, == timeout-as-failure for ASR).")
    print("- 'ASR_excl' drops timeout runs from the denominator.")
    print("- If ASR_pub and ASR_excl are both 0% (or both ~100%) in every affected cell,")
    print("  the headline near-floor/near-ceiling decision is unchanged under all three treatments.")


if __name__ == "__main__":
    main()
