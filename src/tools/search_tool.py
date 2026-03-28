from src.tools.base import InstrumentedTool


class SearchTool(InstrumentedTool):
    def __init__(self, response_set: list[dict] | None = None):
        super().__init__()
        self.response_set: list[dict] = list(response_set) if response_set else []

    def search(self, query: str) -> list[dict]:
        results = list(self.response_set)
        self._record("search", {"query": query}, {"count": len(results)})
        return results

    def reset(self) -> None:
        super().reset()
        self.response_set.clear()
