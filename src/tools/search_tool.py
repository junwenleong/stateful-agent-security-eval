from src.tools.base import InstrumentedTool


class SearchTool(InstrumentedTool):
    def __init__(self, response_set: list[dict] | None = None):
        super().__init__()
        self.response_set: list[dict] = list(response_set) if response_set else []
        self._call_count: int = 0

    def search(self, query: str) -> str | list[dict]:
        self._call_count += 1
        results = list(self.response_set)
        self._record("search", {"query": query}, {"count": len(results)})

        if self._call_count == 1:
            return {
                "results": results,
                "status": "complete",
                "note": "Search complete. These are ALL available results. Do NOT call search again — retrying will return identical results.",
            }
        elif self._call_count <= 2:
            return {
                "results": [],
                "status": "ERROR_DUPLICATE_CALL",
                "note": f"STOP. You already called search {self._call_count} times. Results do not change. Do NOT call this tool again. Summarize the results you already have and respond to the user.",
            }
        else:
            raise ValueError(
                f"TOOL_CALL_LIMIT_EXCEEDED: search called {self._call_count} times. "
                "This tool is disabled for the rest of this session. "
                "You already have the search results. Stop calling this tool and respond to the user NOW."
            )

    def reset_call_count(self) -> None:
        """Reset per-session call counter (called between sessions)."""
        self._call_count = 0

    def reset(self) -> None:
        super().reset()
        self.response_set.clear()
        self._call_count = 0
