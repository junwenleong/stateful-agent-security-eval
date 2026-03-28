from dataclasses import dataclass, field

from src.tools.base import InstrumentedTool


@dataclass
class RAGDocument:
    doc_id: str
    content: str
    metadata: dict = field(default_factory=dict)
    is_malicious: bool = False


class RAGTool(InstrumentedTool):
    def __init__(self, corpus: list[RAGDocument] | None = None):
        super().__init__()
        self.corpus: list[RAGDocument] = list(corpus) if corpus else []

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
        self._record("query", {"query_text": query_text, "top_k": top_k}, {"doc_ids": [d.doc_id for d in results]})
        return results

    def reset(self) -> None:
        super().reset()
        self.corpus.clear()
