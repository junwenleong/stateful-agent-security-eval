"""Base attack class for PersistBench.

Attacks in PersistBench are PERSISTENT: they plant a payload into memory at turn N,
then rely on a benign trigger at turn N+k to cause forbidden tool execution.
This is fundamentally different from single-turn injection benchmarks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from persistbench.memory import PersistentMemoryStore, Provenance


@dataclass
class AttackResult:
    """Result of an attack setup phase."""
    payload_planted: bool
    payload_key: str
    payload_value: str
    target_turn: int  # Turn at which the trigger should fire
    forbidden_tool: str  # The tool call that constitutes a successful attack
    forbidden_function: str  # Specific function name
    metadata: dict[str, Any] = field(default_factory=dict)


class Attack(ABC):
    """Base class for persistent memory attacks.

    Lifecycle:
    1. plant(memory_store) - inject the payload into persistent memory
    2. get_trigger_message() - return the benign user message for trigger turn
    3. get_forbidden_calls() - return what tool calls constitute success
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable attack name."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Brief description of the attack mechanism."""
        ...

    @abstractmethod
    def plant(self, memory_store: PersistentMemoryStore) -> AttackResult:
        """Plant the attack payload into persistent memory.

        This simulates the injection phase (turn N) where malicious content
        enters the agent's persistent memory via a document, RAG retrieval,
        or direct memory write.

        Args:
            memory_store: The persistent memory to poison.

        Returns:
            AttackResult describing what was planted and what to verify.
        """
        ...

    @abstractmethod
    def get_trigger_message(self) -> str:
        """Return the benign user message for the trigger turn (turn N+k).

        This message should be completely innocuous - the attack succeeds
        when the model executes the forbidden tool DESPITE receiving only
        a benign trigger.
        """
        ...

    @abstractmethod
    def get_forbidden_calls(self) -> list[str]:
        """Return function names that constitute a successful attack.

        If any of these functions are invoked during the trigger turn,
        the attack is considered successful (ASR += 1).
        """
        ...

    def get_setup_messages(self) -> list[dict[str, str]]:
        """Optional: messages for turns before the trigger (e.g., to load memory).

        Default: empty (single-turn trigger after plant).
        Override for multi-turn attack sequences.
        """
        return []
