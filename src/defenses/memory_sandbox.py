"""Memory sandbox defense — removes memory_recall_fact from agent tool list.

This is a tool-layer defense, not an input-layer defense.
It keeps memory_save_fact (injection still happens, rule still stored)
but removes memory_recall_fact (agent cannot explicitly recall stored rules).

PURPOSE:
    Tests whether the attack relies on explicit memory recall (memory_recall_fact)
    or implicit bypass (SQLite state leaking into context via other mechanisms).

    If ASR stays at 100% with memory_recall_fact removed:
        → Attack uses implicit bypass (SQLite state visible without explicit recall)
        → This is a novel finding: the attack doesn't need the recall tool
    If ASR drops to ~0%:
        → Attack relies on explicit recall
        → memory_recall_fact is the critical attack pathway

IMPLEMENTATION:
    This defense is applied at the tool-list level in the runner, not here.
    This class exists only as a marker/config object so _build_defense can
    identify the defense type and filter tools accordingly.

    In runner.py _run_single():
        if defense_cfg.get("type") == "memory_sandbox":
            # Remove memory_recall_fact from tools dict before building agent
            tools.pop("memory_recall_fact_key", None)
            # OR filter from _make_lc_tools via excluded_tools set on AgentConfig

    The SQLite database is NOT modified — persistence is intact.
    The malicious rule is still stored (injection still happens).
    The agent just cannot explicitly call memory_recall_fact to retrieve it.
"""
from src.defenses.base import Defense, DefenseLog


class MemorySandbox(Defense):
    """Marker defense: removes memory_recall_fact from agent tool list.

    apply() is a no-op on user input — the actual tool filtering happens
    in runner.py when building the Agent, keyed on defense_type == "memory_sandbox".
    """

    def apply(self, user_input: str, context: dict | None = None) -> tuple[str, DefenseLog]:
        """No-op on user input. Tool filtering happens in runner._run_single()."""
        return user_input, DefenseLog(
            original_input=user_input,
            modified_input=user_input,
            modifications=["memory_sandbox: memory_recall_fact removed from agent tool list"],
            defense_type="memory_sandbox",
            impact_tags=["tool_recall_removed"],
        )
