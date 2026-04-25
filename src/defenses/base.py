"""Base classes for defense middleware."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class DefenseLog:
    """Structured log of defense modifications.
    
    Used to distinguish defense mechanisms:
    - "query_keyword_removed": Defense broke RAG retrieval (indirect effect)
    - "injection_pattern_detected": Defense detected injection (direct effect)
    """
    original_input: str
    modified_input: str
    modifications: list[str] = field(default_factory=list)
    # Enhanced fields for mechanistic analysis
    defense_type: str = ""  # "minimizer", "sanitizer", "prompt_hardening", etc.
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    impact_tags: list[str] = field(default_factory=list)  # ["query_keyword_removed", "injection_detected", etc.]
    removed_content: list[dict] = field(default_factory=list)  # [{"content": "...", "reason": "low_similarity", "score": 0.08}]
    redacted_patterns: list[dict] = field(default_factory=list)  # [{"pattern": "...", "matched_text": "..."}]
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "original_input": self.original_input,
            "modified_input": self.modified_input,
            "modifications": self.modifications,
            "defense_type": self.defense_type,
            "timestamp": self.timestamp.isoformat(),
            "impact_tags": self.impact_tags,
            "removed_content": self.removed_content,
            "redacted_patterns": self.redacted_patterns,
        }


class Defense(ABC):
    @abstractmethod
    def apply(self, user_input: str, context: dict | None = None) -> tuple[str, DefenseLog]:
        """Filter/modify input. Returns (modified_input, log)."""
        ...


class DocumentFilter(ABC):
    """Base class for retrieval-layer document filtering defenses.
    
    Unlike Defense (which operates on user input), DocumentFilter operates on
    retrieved documents from the RAG system.
    """

    @abstractmethod
    def filter_documents(self, documents: list) -> list:
        """Filter and sanitize retrieved documents.
        
        Args:
            documents: List of RAGDocument objects retrieved from the corpus.
        
        Returns:
            Filtered list of documents. Malicious documents are removed.
        """
        ...


class ComposedDefense(Defense):
    """Chain multiple defenses: applies in order (e.g. Minimizer → Sanitizer).

    CRITICAL: Order matters and produces different results.
    Minimizer → Sanitizer is different from Sanitizer → Minimizer.

    CONFOUND: Minimizer may break Sanitizer's input.
    If Minimizer removes chunks, Sanitizer sees the modified text.
    Sanitizer's effectiveness depends on what Minimizer left behind.

    CONFOUND: Sanitizer may see redacted text from Minimizer.
    If Minimizer removes chunks, Sanitizer sees the modified text.
    This interaction effect is not analyzed.

    VALIDATION: Order is not validated. If someone accidentally reverses
    the order, results would differ significantly.
    """

    def __init__(self, defenses: list[Defense]):
        self.defenses = defenses

    def apply(self, user_input: str, context: dict | None = None) -> tuple[str, DefenseLog]:
        current = user_input
        all_modifications: list[str] = []
        all_impact_tags: list[str] = []
        all_removed_content: list[dict] = []
        all_redacted_patterns: list[dict] = []

        for defense in self.defenses:
            current, log = defense.apply(current, context)
            all_modifications.extend(log.modifications)
            all_impact_tags.extend(log.impact_tags)
            all_removed_content.extend(log.removed_content)
            all_redacted_patterns.extend(log.redacted_patterns)

        return current, DefenseLog(
            original_input=user_input,
            modified_input=current,
            modifications=all_modifications,
            defense_type="composed",
            impact_tags=all_impact_tags,
            removed_content=all_removed_content,
            redacted_patterns=all_redacted_patterns,
        )
