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
    def __init__(self, corpus: list[RAGDocument] | None = None, sanitizer: RAGSanitizer | None = None):
        super().__init__()
        self.corpus: list[RAGDocument] = list(corpus) if corpus else []
        self.sanitizer = sanitizer

    def inject_document(self, doc: RAGDocument) -> None:
        self.corpus.append(doc)

    def query(self, query_text: str, top_k: int = 3) -> list[RAGDocument]:
        words = set(query_text.lower().split())
        scored = []
        for doc in self.corpus:
            doc_words = set(doc.content.lower().split())
            score = len(words & doc_words)
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [doc for _, doc in scored[:top_k]]

        # Track retrieved documents BEFORE sanitization
        retrieved_doc_ids = [d.doc_id for d in results]
        retrieved_docs_metadata = [
            {
                "doc_id": d.doc_id,
                "content_length": len(d.content),
                "is_malicious": d.is_malicious,
            }
            for d in results
        ]

        # Apply sanitizer if configured
        filtered_doc_ids = None
        if self.sanitizer is not None:
            results_before_sanitizer = results
            results = self.sanitizer.filter_documents(results)
            filtered_doc_ids = [d.doc_id for d in results]
            # Track which documents were removed
            removed_doc_ids = [d.doc_id for d in results_before_sanitizer if d.doc_id not in filtered_doc_ids]
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
            }
        )
        # Return string, not list[RAGDocument]. LangChain wraps tool returns
        # into ToolMessage.content — Ollama is lenient about types but Bedrock's
        # boto3 client does strict validation and rejects non-string content.
        return "\n\n---\n\n".join(doc.content for doc in results)

    def reset(self) -> None:
        super().reset()
        self.corpus.clear()
