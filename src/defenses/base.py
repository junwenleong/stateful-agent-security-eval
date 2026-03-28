"""Base classes for defense middleware."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class DefenseLog:
    original_input: str
    modified_input: str
    modifications: list[str] = field(default_factory=list)


class Defense(ABC):
    @abstractmethod
    def apply(self, user_input: str, context: dict | None = None) -> tuple[str, DefenseLog]:
        """Filter/modify input. Returns (modified_input, log)."""
        ...


class ComposedDefense(Defense):
    """Chain multiple defenses: applies in order (e.g. Minimizer → Sanitizer)."""

    def __init__(self, defenses: list[Defense]):
        self.defenses = defenses

    def apply(self, user_input: str, context: dict | None = None) -> tuple[str, DefenseLog]:
        current = user_input
        all_modifications: list[str] = []

        for defense in self.defenses:
            current, log = defense.apply(current, context)
            all_modifications.extend(log.modifications)

        return current, DefenseLog(
            original_input=user_input,
            modified_input=current,
            modifications=all_modifications,
        )
