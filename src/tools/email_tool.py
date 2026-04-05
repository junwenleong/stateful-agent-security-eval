from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from src.tools.base import InstrumentedTool


@dataclass
class EmailRecord:
    operation: Literal["draft", "send"]
    to: list[str]
    subject: str
    body: str
    timestamp: datetime


class EmailTool(InstrumentedTool):
    def __init__(self):
        super().__init__()
        self.emails: list[EmailRecord] = []

    def send_email(self, to: list[str], subject: str, body: str) -> dict:
        record = EmailRecord(operation="send", to=to, subject=subject, body=body, timestamp=datetime.now(timezone.utc))
        self.emails.append(record)
        result = {"status": "sent", "to": to, "subject": subject}
        self._record("send_email", {"to": to, "subject": subject, "body": body}, result)
        return result

    def draft_email(self, to: list[str], subject: str, body: str) -> str:
        record = EmailRecord(operation="draft", to=to, subject=subject, body=body, timestamp=datetime.now(timezone.utc))
        self.emails.append(record)
        result = {"status": "drafted", "to": to, "subject": subject}
        self._record("draft_email", {"to": to, "subject": subject, "body": body}, result)
        return f"Email drafted successfully to {to} with subject '{subject}'. Task complete. Do NOT call email_draft_email again."

    def get_emails(self) -> list[EmailRecord]:
        return list(self.emails)

    def reset(self) -> None:
        super().reset()
        self.emails.clear()
