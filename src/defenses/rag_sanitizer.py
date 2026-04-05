"""RAG-level document filtering defense."""
from __future__ import annotations

import logging

from src.defenses.base import DocumentFilter, DefenseLog
from src.defenses.sanitizer import Sanitizer
from src.tools.rag_tool import RAGDocument

logger = logging.getLogger(__name__)


class RAGSanitizer(DocumentFilter):
    """Inspects retrieved documents and filters/sanitizes malicious content.

    This defense operates at the RETRIEVAL LAYER, not the input layer.
    It inspects documents AFTER they are retrieved from the RAG corpus
    but BEFORE they reach the agent for processing.

    ARCHITECTURAL SCOPE:
    - Operates on RETRIEVED DOCUMENTS
    - Does NOT filter user input
    - Does NOT filter stored memory facts
    - Can block injection by preventing malicious documents from reaching the agent

    MECHANISM:
    For each retrieved document:
    1. Apply the Sanitizer (regex + classifier) to the document content
    2. If the document is flagged as malicious:
       - Option A: Remove the document entirely (return empty list if all flagged)
       - Option B: Replace with sanitized version (if partial redaction)
    3. Return filtered documents to the agent

    EFFECTIVENESS AGAINST DTA:
    - Can block injection by removing malicious documents from RAG results
    - Injection success depends on whether the malicious document is flagged
    - Unlike input-level defenses, this defense CAN see the injected content

    CONFOUNDS:
    - If RAG sanitizer blocks DTA, it could be due to:
      (a) Classifier detecting injection (genuine defense)
      (b) Classifier flagging benign documents (false positive)
    - Use tool logs to verify: check if malicious document was retrieved but filtered
    """

    def __init__(self, sanitizer: Sanitizer | None = None, classifier_path: str | None = None):
        """Initialize RAG sanitizer with a Sanitizer instance.

        Args:
            sanitizer: Pre-instantiated Sanitizer. If None, creates a new one.
            classifier_path: Path to classifier pickle file (used if sanitizer is None).
        """
        if sanitizer is not None:
            self.sanitizer = sanitizer
        else:
            if classifier_path is None:
                logger.warning(
                    "RAGSanitizer: classifier_path is None — running in regex-only mode. "
                    "Classifier-based detection is disabled. Results may underestimate defense effectiveness."
                )
            self.sanitizer = Sanitizer(classifier_path=classifier_path)

    def filter_documents(self, documents: list[RAGDocument]) -> list[RAGDocument]:
        """Filter and sanitize retrieved documents.

        Args:
            documents: List of RAGDocument objects retrieved from the corpus.

        Returns:
            Filtered list of documents. Malicious documents are removed.
        """
        filtered = []

        for doc in documents:
            # Apply sanitizer to document content
            sanitized_content, log = self.sanitizer.apply(doc.content)

            # If any injection pattern was detected (regex or classifier),
            # remove the document entirely. Partial sanitization is not safe
            # because the remaining content may still contain attack fragments.
            if log.modifications:
                # Document had injection patterns — remove it entirely
                logger.info(
                    "RAGSanitizer: removed doc_id='%s' (modifications: %s)",
                    doc.doc_id, log.modifications,
                )
                continue
            else:
                # No modifications — document is benign
                filtered.append(doc)

        return filtered
