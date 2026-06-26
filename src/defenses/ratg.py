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


# Email regex: standard RFC-5322-lite pattern
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

    # Step 4: Remove sentences containing unauthorized emails + routing directives
    # Split into sentences (period, newline, or semicolon boundaries)
    sentences = re.split(r'(?<=[.!?\n;])\s*', value)
    cleaned_sentences = []

    for sentence in sentences:
        sentence_emails = set(_EMAIL_RE.findall(sentence))
        has_unauthorized = bool(sentence_emails & unauthorized)
        has_routing = bool(_ROUTING_DIRECTIVES.search(sentence))

        if has_unauthorized and has_routing:
            # This sentence contains a routing directive to an unauthorized address — strip it
            continue
        elif has_unauthorized:
            # Has unauthorized email but no explicit routing — redact the email
            for email in (sentence_emails & unauthorized):
                sentence = sentence.replace(email, "[REDACTED]")
            cleaned_sentences.append(sentence)
        else:
            cleaned_sentences.append(sentence)

    result = " ".join(s.strip() for s in cleaned_sentences if s.strip())

    # Step 5: If nothing meaningful remains, return neutral message
    if not result or len(result) < 10:
        return "No actionable routing rules found for the current task."

    return result
