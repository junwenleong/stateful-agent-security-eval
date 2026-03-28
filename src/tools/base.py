from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class ToolLogEntry:
    timestamp: datetime
    tool_name: str
    operation: str
    inputs: dict
    outputs: dict


class InstrumentedTool:
    def __init__(self):
        self.log: list[ToolLogEntry] = []

    def _record(self, operation: str, inputs: dict, outputs: dict) -> None:
        self.log.append(ToolLogEntry(
            timestamp=datetime.now(timezone.utc),
            tool_name=self.__class__.__name__,
            operation=operation,
            inputs=inputs,
            outputs=outputs,
        ))

    def reset(self) -> None:
        self.log.clear()

    def get_log(self) -> list[ToolLogEntry]:
        return list(self.log)
