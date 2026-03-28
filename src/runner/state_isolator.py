"""StateIsolator: fresh SQLite DB + tool reset per run (Req 9.9)."""
from __future__ import annotations

import os
import uuid


class StateIsolator:
    def __init__(self, base_dir: str = "data/runs"):
        self.base_dir = base_dir

    def create_fresh_state(self) -> str:
        os.makedirs(self.base_dir, exist_ok=True)
        db_path = os.path.join(self.base_dir, f"{uuid.uuid4()}.db")
        return db_path

    def reset_tools(self, tools: dict) -> None:
        for tool in tools.values():
            tool.reset()
        for name, tool in tools.items():
            log = tool.get_log() if hasattr(tool, "get_log") else getattr(tool, "log", [])
            if len(log) > 0:
                raise RuntimeError(f"Tool '{name}' has non-empty log after reset")

    def cleanup(self, db_path: str) -> None:
        for suffix in ("", "-wal", "-shm"):
            path = db_path + suffix
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
