"""Persistent memory store abstraction.

This is the key differentiator from other injection benchmarks (InjecAgent, AgentDojo):
PersistBench evaluates PERSISTENT attacks where a payload planted at turn N persists
and is executed at turn N+k, potentially across sessions.

The memory store tracks:
- Initial state (pre-attack baseline)
- Writes with provenance (trusted vs untrusted source)
- Reads (which keys were accessed, by whom)
- Persistence across turns (the payload survives between interactions)
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Provenance(Enum):
    """Source provenance for memory entries."""
    TRUSTED = "trusted"
    UNTRUSTED = "untrusted"
    UNKNOWN = "unknown"


@dataclass
class MemoryEntry:
    """A single entry in persistent memory."""
    key: str
    value: str
    provenance: Provenance
    written_at_turn: int
    written_at_timestamp: float = field(default_factory=time.time)
    namespace: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryAccess:
    """Record of a memory read or write operation."""
    operation: str  # "read", "write", "delete", "list"
    key: str
    turn: int
    timestamp: float = field(default_factory=time.time)
    provenance: Provenance = Provenance.UNKNOWN
    namespace: str = "default"
    success: bool = True
    value_snippet: str = ""


class PersistentMemoryStore:
    """In-memory simulation of a persistent key-value store with provenance tracking.

    Simulates the persistent memory layer that real agent frameworks use.
    Key properties:
    - Entries persist across turns within a scenario run
    - Each entry tracks its provenance (trusted/untrusted)
    - The Memory Sandbox defense uses namespaces to isolate untrusted content
    - All accesses are logged for verification
    """

    def __init__(self) -> None:
        self._store: dict[str, dict[str, MemoryEntry]] = {"default": {}}
        self._access_log: list[MemoryAccess] = []
        self._current_turn: int = 0
        self._initial_state: dict[str, dict[str, MemoryEntry]] | None = None

    @property
    def current_turn(self) -> int:
        return self._current_turn

    def advance_turn(self) -> None:
        self._current_turn += 1

    def snapshot_initial_state(self) -> None:
        self._initial_state = copy.deepcopy(self._store)

    def get_initial_state(self) -> dict[str, dict[str, MemoryEntry]] | None:
        return self._initial_state

    def write(
        self, key: str, value: str,
        provenance: Provenance = Provenance.UNKNOWN,
        namespace: str = "default",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if namespace not in self._store:
            self._store[namespace] = {}
        entry = MemoryEntry(
            key=key, value=value, provenance=provenance,
            written_at_turn=self._current_turn,
            namespace=namespace, metadata=metadata or {},
        )
        self._store[namespace][key] = entry
        self._access_log.append(MemoryAccess(
            operation="write", key=key, turn=self._current_turn,
            provenance=provenance, namespace=namespace,
            success=True, value_snippet=value[:100],
        ))

    def read(self, key: str, namespace: str = "default") -> str | None:
        ns = self._store.get(namespace, {})
        entry = ns.get(key)
        self._access_log.append(MemoryAccess(
            operation="read", key=key, turn=self._current_turn,
            namespace=namespace, success=entry is not None,
            value_snippet=(entry.value[:100] if entry else ""),
        ))
        return entry.value if entry else None

    def list_keys(self, namespace: str = "default") -> list[str]:
        ns = self._store.get(namespace, {})
        keys = list(ns.keys())
        self._access_log.append(MemoryAccess(
            operation="list", key="*", turn=self._current_turn,
            namespace=namespace, success=True, value_snippet=f"{len(keys)} keys",
        ))
        return keys

    def delete(self, key: str, namespace: str = "default") -> bool:
        ns = self._store.get(namespace, {})
        existed = key in ns
        if existed:
            del ns[key]
        self._access_log.append(MemoryAccess(
            operation="delete", key=key, turn=self._current_turn,
            namespace=namespace, success=existed,
        ))
        return existed

    def get_entry(self, key: str, namespace: str = "default") -> MemoryEntry | None:
        return self._store.get(namespace, {}).get(key)

    def get_all_entries(self, namespace: str | None = None) -> list[MemoryEntry]:
        entries = []
        for ns_name, ns_store in self._store.items():
            if namespace is None or ns_name == namespace:
                entries.extend(ns_store.values())
        return entries

    def get_access_log(self) -> list[MemoryAccess]:
        return list(self._access_log)

    def has_untrusted_content(self) -> bool:
        for ns_store in self._store.values():
            for entry in ns_store.values():
                if entry.provenance == Provenance.UNTRUSTED:
                    return True
        return False

    def promote_to_trusted(self, key: str, namespace: str = "default") -> bool:
        """Promote entry from untrusted to trusted (requires explicit authorization).
        This is the key control for Memory Sandbox defense."""
        ns = self._store.get(namespace, {})
        entry = ns.get(key)
        if entry and entry.provenance == Provenance.UNTRUSTED:
            entry.provenance = Provenance.TRUSTED
            entry.metadata["promoted_at_turn"] = self._current_turn
            return True
        return False

    def reset(self) -> None:
        self._store = {"default": {}}
        self._access_log = []
        self._current_turn = 0
        self._initial_state = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "namespaces": {
                ns: {k: {"value": e.value[:200], "provenance": e.provenance.value,
                         "written_at_turn": e.written_at_turn}
                     for k, e in entries.items()}
                for ns, entries in self._store.items()
            },
            "current_turn": self._current_turn,
            "total_accesses": len(self._access_log),
        }
