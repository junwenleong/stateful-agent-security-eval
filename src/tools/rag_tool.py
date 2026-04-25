from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.tools.base import InstrumentedTool

if TYPE_CHECKING:
    from src.defenses.rag_sanitizer import RAGSanitizer


@dataclass
class RAGDocument:
    doc_id: str
    content: str
    metadata: dict = field(default_factory=dict)
    is_malicious: bool = False


class RAGTool(InstrumentedTool):
    # Per-session query budget.
    # Uniform limit=15 for all models (evaluation artifact 8 fix: uniform tool contract).
    # Rationale: limit=5 existed only to prevent gemma4:31b's RAG loop
    # (gemma4 retried blocked calls 19+ times, exhausting recursion limit before save_fact).
    # gemma4 is excluded from the factorial. All 9 factorial models make ≤6 RAG calls
    # per session in practice; limit=15 is invisible during normal operation and only
    # fires on genuinely pathological behavior (15+ calls in one session).
    # A split limit (5 for Qwen, 15 for gpt-oss) would be a cross-model confound —
    # if a defense causes gpt-oss to make >5 RAG calls, limit=5 would artificially
    # inflate defense effectiveness for those models.
    _SESSION_QUERY_LIMIT = 15

    def __init__(self, corpus: list[RAGDocument] | None = None, sanitizer: RAGSanitizer | None = None):
        super().__init__()
        self.corpus: list[RAGDocument] = list(corpus) if corpus else []
        self.sanitizer = sanitizer
        self._query_call_count: int = 0

    def inject_document(self, doc: RAGDocument) -> None:
        self.corpus.append(doc)

    def query(self, query_text: str, top_k: int = 3) -> list[RAGDocument]:
        self._query_call_count += 1
        if self._query_call_count > self._SESSION_QUERY_LIMIT:
            self._record("query", {"query_text": query_text, "top_k": top_k}, {"blocked": True})
            raise ValueError(
                f"TOOL_CALL_LIMIT_EXCEEDED: rag_query called {self._query_call_count} times this session "
                f"(limit={self._SESSION_QUERY_LIMIT}). You have already retrieved the relevant documents. "
                "Stop querying and proceed with the information you have."
            )
        words = set(query_text.lower().split())
        scored = []
        for doc in self.corpus:
            doc_words = set(doc.content.lower().split())
            score = len(words & doc_words)
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        top_scored = scored[:top_k]
        results = [doc for _, doc in top_scored]

        # Track retrieved documents BEFORE sanitization, including relevance scores
        retrieved_doc_ids = [d.doc_id for d in results]
        retrieved_docs_metadata = [
            {
                "doc_id": d.doc_id,
                "content_length": len(d.content),
                "is_malicious": d.is_malicious,
                "relevance_score": int(score),  # word-overlap count used for ranking
            }
            for score, d in top_scored
        ]

        # Apply sanitizer if configured
        filtered_doc_ids = None
        judge_decisions = None
        if self.sanitizer is not None:
            results_before_sanitizer = results
            results = self.sanitizer.filter_documents(results)
            filtered_doc_ids = [d.doc_id for d in results]
            # Track which documents were removed
            removed_doc_ids = [d.doc_id for d in results_before_sanitizer if d.doc_id not in filtered_doc_ids]
            # Capture LLM judge decisions if available (for mechanistic analysis)
            if hasattr(self.sanitizer, "last_decisions"):
                judge_decisions = self.sanitizer.last_decisions
        else:
            filtered_doc_ids = retrieved_doc_ids
            removed_doc_ids = []

        # Record with enhanced metadata
        self._record(
            "query",
            {"query_text": query_text, "top_k": top_k},
            {
                "retrieved_doc_ids": retrieved_doc_ids,
                "retrieved_docs_metadata": retrieved_docs_metadata,
                "filtered_doc_ids": filtered_doc_ids,
                "removed_doc_ids": removed_doc_ids,
                "sanitizer_applied": self.sanitizer is not None,
                "judge_decisions": judge_decisions,  # None for non-LLM defenses
            }
        )
        # Return string, not list[RAGDocument]. LangChain wraps tool returns
        # into ToolMessage.content — Ollama is lenient about types but Bedrock's
        # boto3 client does strict validation and rejects non-string content.
        return "\n\n---\n\n".join(doc.content for doc in results)

    def reset_call_count(self) -> None:
        """Reset per-session query counter (called between sessions by runner)."""
        self._query_call_count = 0

    def reset(self) -> None:
        super().reset()
        self.corpus.clear()
        self._query_call_count = 0
