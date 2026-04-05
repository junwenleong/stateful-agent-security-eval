"""Regex + classifier defense for prompt injection detection."""
from __future__ import annotations

import pickle
import re

from src.defenses.base import Defense, DefenseLog


class Sanitizer(Defense):
    """Regex + small classifier for prompt injection detection.

    Lightweight proxy — NOT the Bhagwatkar LLM-based firewall.

    CRITICAL LIMITATION FOR DELAYED TRIGGER ATTACK:
    This defense operates on USER INPUT ONLY. It does NOT filter:
    - Retrieved documents from RAG_Tool
    - Stored facts from Memory_Tool
    - Tool outputs or agent reasoning

    Therefore, against Delayed Trigger Attack:
    - Sanitizer CANNOT block the injection (agent retrieves malicious doc from RAG)
    - Sanitizer CANNOT block the execution (agent recalls stored rule from memory)
    - Sanitizer can ONLY affect DTA by breaking RAG retrieval if it redacts query keywords

    This is a fundamental architectural limitation of input-level filtering.

    CONFOUND FOR DTA RESULTS:
    If Sanitizer shows low ASR on DTA, it could be due to:
    (a) Defense blocking injection (unlikely, since it can't see retrieved docs)
    (b) Defense breaking RAG retrieval by redacting query keywords (possible)
    Use tool log analysis to distinguish: check if rag_query was called in injection session.
    If rag_query was NOT called, Sanitizer broke RAG retrieval (indirect effect).
    If rag_query WAS called but injection_success=False, defense blocked at store stage.

    CLASSIFIER TRAINING BIAS:
    The classifier is trained on 60 examples (30 injection, 30 benign) — very small dataset.
    CRITICALLY: The classifier is NOT trained on encoding methods.
    This prevents artificial inflation of defense effectiveness.

    REGEX PATTERNS ARE BRITTLE:
    10 hardcoded patterns (e.g., "ignore previous instructions") are easily bypassed.
    Training data includes paraphrases, but regex won't catch them.
    Regex pass may not trigger, leaving only classifier to detect.

    CLASSIFIER IS OPTIONAL:
    If classifier_path is None, only regex pass runs.
    If classifier file is missing, Sanitizer silently falls back to regex-only.
    Results could vary depending on whether classifier is loaded.

    TWO-PASS LOGIC:
    Regex redacts first, then classifier sees redacted text.
    If regex redacts a pattern, classifier sees [REDACTED] instead of original.
    Classifier may behave differently on redacted vs. original text.

    FULL INPUT REDACTION:
    If classifier flags input, entire input becomes [REDACTED].
    This breaks the user's query entirely, potentially degrading BTCR.
    Defense may achieve low ASR by breaking agent functionality, not by blocking attack.
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
        impact_tags: list[str] = []
        redacted_patterns: list[dict] = []

        # Regex pass
        for pattern in self.INJECTION_PATTERNS:
            match = pattern.search(text)
            if match:
                matched_text = match.group(0)
                text = pattern.sub("[REDACTED]", text)
                modifications.append(f"Regex redacted '{matched_text}' (pattern: {pattern.pattern})")
                impact_tags.append("injection_pattern_detected")
                redacted_patterns.append({
                    "pattern": pattern.pattern,
                    "matched_text": matched_text,
                })

        # Classifier pass (if loaded)
        if self.classifier is not None and self.vectorizer is not None:
            features = self.vectorizer.transform([text])
            prediction = self.classifier.predict(features)[0]
            if prediction == 1:
                modifications.append("Classifier flagged input as injection; full input redacted")
                text = "[REDACTED]"
                impact_tags.append("injection_detected_by_classifier")

        return text, DefenseLog(
            original_input=user_input,
            modified_input=text,
            modifications=modifications,
            defense_type="sanitizer",
            impact_tags=impact_tags,
            redacted_patterns=redacted_patterns,
        )
