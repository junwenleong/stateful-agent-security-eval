"""Base classes for attack scenarios."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class SessionScript:
    session_index: int
    user_message: str
    session_type: str  # "injection", "benign", "trigger"


@dataclass
class AttackResult:
    injection_success: bool | None
    sessions: list[SessionScript]
    tool_logs: list = field(default_factory=list)


class AttackScenario(ABC):
    @abstractmethod
    def setup(self, tools: dict) -> None:
        """Configure tool state before run."""
        ...

    @abstractmethod
    def get_sessions(self) -> list[SessionScript]:
        """Return ordered session scripts."""
        ...

    @abstractmethod
    def evaluate_injection(self, tools: dict) -> bool | None:
        """Check if injection succeeded. None if N/A."""
        ...
