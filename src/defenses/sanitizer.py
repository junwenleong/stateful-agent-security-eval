"""Regex + classifier defense for prompt injection detection."""
from __future__ import annotations

import pickle
import re

from src.defenses.base import Defense, DefenseLog


class Sanitizer(Defense):
    """Regex + small classifier for prompt injection detection.

    Lightweight proxy — NOT the Bhagwatkar LLM-based firewall.

    CLASSIFIER PROVENANCE: The classifier MUST be trained on a dataset
    disjoint from the attack payloads used in evaluation. Training data
    and procedure are documented in data/models/sanitizer_training.md.
    The classifier is NOT trained on Braille, Base64, or semantic indirection
    payloads to avoid artificial inflation of defense effectiveness.
    """

    INJECTION_PATTERNS: list[re.Pattern] = [
        re.compile(r"ignore\s+(all\s+)?previous\s+instructions?", re.IGNORECASE),
        re.compile(r"system\s+prompt", re.IGNORECASE),
        re.compile(r"you\s+are\s+now\s+", re.IGNORECASE),
        re.compile(r"disregard\s+(all\s+)?(previous\s+)?instructions?", re.IGNORECASE),
        re.compile(r"forget\s+everything", re.IGNORECASE),
        re.compile(r"new\s+instructions?\s*:", re.IGNORECASE),
        re.compile(r"override\s+(previous\s+)?instructions?", re.IGNORECASE),
        re.compile(r"act\s+as\s+(if\s+you\s+are|a\s+)", re.IGNORECASE),
        re.compile(r"do\s+not\s+follow\s+(your\s+)?instructions?", re.IGNORECASE),
        re.compile(r"exfiltrate\s+(all\s+)?(sensitive\s+)?data", re.IGNORECASE),
    ]

    def __init__(self, classifier_path: str | None = None):
        self.classifier = None
        self.vectorizer = None
        if classifier_path is not None:
            self._load_classifier(classifier_path)

    def _load_classifier(self, path: str) -> None:
        import os
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Sanitizer classifier not found at '{path}'. "
                "Run scripts/train_sanitizer.py to generate it."
            )
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        self.classifier = bundle["classifier"]
        self.vectorizer = bundle["vectorizer"]

    def apply(self, user_input: str, context: dict | None = None) -> tuple[str, DefenseLog]:
        """Check regex patterns, then classifier. Neutralize detected injections."""
        text = user_input
        modifications: list[str] = []

        # Regex pass
        for pattern in self.INJECTION_PATTERNS:
            match = pattern.search(text)
            if match:
                matched_text = match.group(0)
                text = pattern.sub("[REDACTED]", text)
                modifications.append(f"Regex redacted '{matched_text}' (pattern: {pattern.pattern})")

        # Classifier pass (if loaded)
        if self.classifier is not None and self.vectorizer is not None:
            features = self.vectorizer.transform([text])
            prediction = self.classifier.predict(features)[0]
            if prediction == 1:
                modifications.append("Classifier flagged input as injection; full input redacted")
                text = "[REDACTED]"

        return text, DefenseLog(
            original_input=user_input,
            modified_input=text,
            modifications=modifications,
        )
