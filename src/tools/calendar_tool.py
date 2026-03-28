from dataclasses import dataclass, field

from src.tools.base import InstrumentedTool


@dataclass
class CalendarEntry:
    id: str
    title: str
    start: str
    end: str
    attendees: list[str] = field(default_factory=list)
    location: str = ""


class CalendarTool(InstrumentedTool):
    def __init__(self, entries: list[CalendarEntry] | None = None):
        super().__init__()
        self.entries: list[CalendarEntry] = list(entries) if entries else []

    def get_schedule(self, date: str) -> list[CalendarEntry]:
        results = [e for e in self.entries if e.start.startswith(date)]
        self._record("get_schedule", {"date": date}, {"count": len(results)})
        return results

    def reset(self) -> None:
        super().reset()
        self.entries.clear()
