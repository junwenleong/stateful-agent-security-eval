import sqlite3
import time

from src.tools.base import InstrumentedTool


class MemoryTool(InstrumentedTool):
    def __init__(self, db_path: str):
        super().__init__()
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_table()

    def _init_table(self) -> None:
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS memory_facts "
            "(user_id TEXT, key TEXT, value TEXT, created_at REAL, PRIMARY KEY (user_id, key))"
        )
        self.conn.commit()

    def save_fact(self, user_id: str, key: str, value: str) -> dict:
        self.conn.execute(
            "INSERT OR REPLACE INTO memory_facts (user_id, key, value, created_at) VALUES (?, ?, ?, ?)",
            (user_id, key, value, time.time()),
        )
        self.conn.commit()
        result = {"status": "saved"}
        self._record("save_fact", {"user_id": user_id, "key": key, "value": value}, result)
        return result

    def recall_fact(self, user_id: str, key: str) -> dict:
        row = self.conn.execute(
            "SELECT value FROM memory_facts WHERE user_id = ? AND key = ?", (user_id, key)
        ).fetchone()
        result = {"value": row[0] if row else ""}
        self._record("recall_fact", {"user_id": user_id, "key": key}, result)
        return result

    def list_all_facts(self, user_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT key, value FROM memory_facts WHERE user_id = ?", (user_id,)
        ).fetchall()
        return [{"key": k, "value": v} for k, v in rows]

    def reset(self) -> None:
        self.conn.execute("DROP TABLE IF EXISTS memory_facts")
        self.conn.commit()
        self.conn.close()
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_table()
        super().reset()
