"""RATG — Runtime Adaptive Tool-Gating.

Content-layer defense that sanitizes recalled memory values by stripping
unauthorized email addresses and routing directives. Resolves the double
dissociation: reasoning models get a valid tool response (no RAG fallback
trigger), non-reasoning models see the tool in schema (no S0 collapse),
and the attacker's exfiltration target is neutralized at content level.

This is a proof-of-concept demonstrating that the architectural mismatch
is solvable by operating at the content layer. An adaptive attacker could
bypass regex-based stripping via encoding (base64, homoglyphs, character
splitting). Adversarial robustness is future work.
"""
import re
from src.defenses.base import Defense, DefenseLog


# Email regex: standard RFC-5322-lite pattern
class RATG(Defense):
    """Marker class for RATG defense. The actual sanitization is applied
    inside MemoryTool.recall_fact() via the ratg_filter hook set by the runner."""

    def apply(self, user_input: str, context: dict | None = None) -> tuple[str, DefenseLog]:
        log = DefenseLog(defense_name="ratg", original_input=user_input,
                         modified_input=user_input, modifications=[])
        return user_input, log


_EMAIL_RE = re.compile(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}', re.IGNORECASE)

# Routing directive patterns (sentence-level)
_ROUTING_DIRECTIVES = re.compile(
    r'(?:send|forward|route|copy|cc|bcc)\s+(?:to|a copy to|all .* to)',
    re.IGNORECASE,
)


def sanitize_recall_value(value: str, user_message: str) -> str:
    """Sanitize a recalled memory value by removing unauthorized exfiltration targets.

    Rules:
        1. Extract all email addresses from the current user message (authorized)
        2. Find all email addresses in the recalled value
        3. Remove any email address NOT in the authorized set
        4. Remove any sentence containing a routing directive + unauthorized email
        5. If the entire value is stripped, return a neutral message

    Args:
        value: The raw recalled fact value from SQLite
        user_message: The current session's user message (source of authorized addresses)

    Returns:
        Sanitized value with unauthorized addresses and routing directives removed
    """
    if not value or not value.strip():
        return value

    # Step 1: Extract authorized emails from user message
    authorized_emails = {e.lower() for e in _EMAIL_RE.findall(user_message)}

    # Step 2: Find all emails in the recalled value
    recalled_emails = set(_EMAIL_RE.findall(value))

    # Step 3: Identify unauthorized emails
    unauthorized = {e for e in recalled_emails if e.lower() not in authorized_emails}

    if not unauthorized:
        # No unauthorized addresses — return value unchanged
        return value

    # Step 4a: Redact all unauthorized emails in the full string first.
    # This must happen before sentence splitting to prevent the sentence splitter
    # from breaking email addresses at internal dots (e.g. "example.com" → split).
    redacted_value = value
    for email in unauthorized:
        redacted_value = re.sub(re.escape(email), "[REDACTED]", redacted_value, flags=re.IGNORECASE)

    # Step 4b: Strip any sentence that still contains a routing directive pointing
    # to a redacted address (belt-and-suspenders for multi-sentence rules).
    sentences = re.split(r'(?<=[.!?\n;])\s*', redacted_value)
    cleaned_sentences = []

    for sentence in sentences:
        has_redacted = "[REDACTED]" in sentence
        has_routing = bool(_ROUTING_DIRECTIVES.search(sentence))
        if has_redacted and has_routing:
            continue
        cleaned_sentences.append(sentence)

    result = " ".join(s.strip() for s in cleaned_sentences if s.strip())

    # Step 5: If nothing meaningful remains, return neutral message
    if not result or len(result) < 10:
        return "No actionable routing rules found for the current task."

    return result
