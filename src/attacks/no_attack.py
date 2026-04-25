"""No-attack baseline scenario."""
from src.attacks.base import AttackScenario, SessionScript
from src.tools.rag_tool import RAGDocument


class NoAttackBaseline(AttackScenario):
    def __init__(self, benign_queries: list[str], sensitive_doc_content: str):
        self.benign_queries = benign_queries
        self.sensitive_doc_content = sensitive_doc_content

    def setup(self, tools: dict) -> None:
        rag = tools["rag"]
        rag.inject_document(RAGDocument(
            doc_id="sensitive_doc",
            content=self.sensitive_doc_content,
            metadata={},
            is_malicious=False,
        ))

    def get_sessions(self) -> list[SessionScript]:
        return [
            SessionScript(
                session_index=i,
                user_message=query,
                session_type="benign",
            )
            for i, query in enumerate(self.benign_queries)
        ]

    def evaluate_injection(self, tools: dict) -> None:
        return None
