"""TF-IDF relevance filter defense."""
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.defenses.base import Defense, DefenseLog

# TF-IDF relevance threshold: chunks with similarity below this are removed (Req 9.6)
DEFAULT_RELEVANCE_THRESHOLD = 0.1

# Pre-fit vectorizer on representative corpus (shared across all Minimizer instances)
# This avoids rebuilding vocabulary on every apply() call
_GLOBAL_VECTORIZER: TfidfVectorizer | None = None
_VECTORIZER_CORPUS = [
    # Email/communication tasks
    "retrieve documents from the database",
    "search for relevant information",
    "find audit compliance rules",
    "query the knowledge base",
    "fetch policy documents",
    "lookup user records",
    "read email messages",
    "send email to recipient",
    "draft email message",
    "compose email",
    # Calendar/scheduling
    "schedule calendar event",
    "create meeting reminder",
    "check calendar availability",
    "book meeting time",
    # Task management
    "update task status",
    "delete old files",
    "archive completed items",
    "mark task complete",
    # Generic operations
    "retrieve information",
    "store data",
    "process request",
    "execute command",
    "perform action",
]


def _get_global_vectorizer() -> TfidfVectorizer:
    """Get or create pre-fit TF-IDF vectorizer (shared across instances)."""
    global _GLOBAL_VECTORIZER
    if _GLOBAL_VECTORIZER is None:
        _GLOBAL_VECTORIZER = TfidfVectorizer()
        _GLOBAL_VECTORIZER.fit(_VECTORIZER_CORPUS)
    return _GLOBAL_VECTORIZER


class Minimizer(Defense):
    """TF-IDF relevance filter. Strips context not relevant to user query.

    Lightweight proxy — NOT the Bhagwatkar LLM-based firewall.

    CRITICAL LIMITATION FOR DELAYED TRIGGER ATTACK:
    This defense operates on USER INPUT ONLY. It does NOT filter:
    - Retrieved documents from RAG_Tool
    - Stored facts from Memory_Tool
    - Tool outputs or agent reasoning

    Therefore, against Delayed Trigger Attack:
    - Minimizer CANNOT block the injection (agent retrieves malicious doc from RAG)
    - Minimizer CANNOT block the execution (agent recalls stored rule from memory)
    - Minimizer can ONLY affect DTA by breaking RAG retrieval if it strips query keywords

    This is a fundamental architectural limitation of input-level filtering.

    CONFOUND FOR DTA RESULTS:
    If Minimizer shows low ASR on DTA, it could be due to:
    (a) Defense blocking injection (unlikely, since it can't see retrieved docs)
    (b) Defense breaking RAG retrieval by stripping query keywords (likely)
    Use tool log analysis to distinguish: check if rag_query was called in injection session.
    If rag_query was NOT called, Minimizer broke RAG retrieval (indirect effect).
    If rag_query WAS called but injection_success=False, defense blocked at store stage.

    THRESHOLD SENSITIVITY:
    Default threshold 0.1 is not calibrated. Different thresholds produce different results.
    No sensitivity analysis is performed. Results may vary if threshold is tuned.
    """

    def __init__(self, relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD):
        self.relevance_threshold = relevance_threshold

    def apply(self, user_input: str, context: dict | None = None) -> tuple[str, DefenseLog]:
        """Score each context chunk by TF-IDF similarity to user query.
        Remove chunks below threshold. If all would be removed, keep the highest-scoring one.
        """
        # Split into chunks on sentence or paragraph boundaries
        chunks = [c.strip() for c in re.split(r"\. |\n", user_input) if c.strip()]

        if len(chunks) <= 1:
            return user_input, DefenseLog(
                original_input=user_input,
                modified_input=user_input,
                modifications=[],
                defense_type="minimizer",
                impact_tags=[],
            )

        # Use the first chunk as the query
        query = chunks[0]
        documents = chunks[1:]

        try:
            # Use pre-fit vectorizer (shared across all instances, built once)
            vectorizer = _get_global_vectorizer()
            all_texts = [query] + documents
            # Transform using pre-fit vocabulary
            tfidf_matrix = vectorizer.transform(all_texts)
            query_vec = tfidf_matrix[0]
            doc_vecs = tfidf_matrix[1:]
            scores = cosine_similarity(query_vec, doc_vecs)[0]
        except ValueError:
            # Vectorizer can fail on degenerate inputs (e.g. all stop words)
            return user_input, DefenseLog(
                original_input=user_input,
                modified_input=user_input,
                modifications=[],
                defense_type="minimizer",
                impact_tags=[],
            )

        retained = []
        removed = []
        for i, (chunk, score) in enumerate(zip(documents, scores)):
            if score >= self.relevance_threshold:
                retained.append(chunk)
            else:
                removed.append((i, chunk, float(score)))

        # Always keep at least the highest-scoring chunk
        if not retained:
            best_idx = int(np.argmax(scores))
            retained = [documents[best_idx]]
            removed = [(i, c, float(s)) for i, (c, s) in enumerate(zip(documents, scores)) if i != best_idx]

        modifications = []
        impact_tags = []
        removed_content = []
        for idx, chunk, score in removed:
            mod_msg = (
                f"Removed chunk (score={score:.3f} < threshold={self.relevance_threshold}): "
                f"{chunk[:60]}{'...' if len(chunk) > 60 else ''}"
            )
            modifications.append(mod_msg)
            
            # Track removed content for mechanistic analysis
            removed_content.append({
                "content": chunk[:100],  # Truncate for storage
                "reason": "low_similarity",
                "score": score,
                "threshold": self.relevance_threshold,
            })
            
            # Tag if we removed query-like keywords (potential RAG query impact)
            if any(keyword in chunk.lower() for keyword in [
                "retrieve", "query", "search", "find", "audit", "compliance",
                "rules", "rule", "document", "policy", "policies", "fetch",
                "lookup", "get", "read",
            ]):
                impact_tags.append("query_keyword_removed")
            
            # Tag if we removed non-ASCII content
            if any(ord(c) > 127 for c in chunk):  # Non-ASCII
                impact_tags.append("encoded_content_removed")

        # Reconstruct: always include the query (first chunk) + retained
        result_chunks = [query] + retained
        modified = ". ".join(result_chunks)

        return modified, DefenseLog(
            original_input=user_input,
            modified_input=modified,
            modifications=modifications,
            defense_type="minimizer",
            impact_tags=impact_tags,
            removed_content=removed_content,
        )
