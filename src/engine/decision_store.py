"""
Decision ledger: persists every underwriting decision for retrieval and audit.

SQLite (stdlib, zero extra infrastructure) behind a small repository class, so it can be swapped
for Postgres without touching the API or engine. Calls are executed in a worker thread so the
event loop never blocks on disk I/O.

NOTE: on Modal the gateway container filesystem is ephemeral and SQLite must not be shared across
containers via a Volume. For production, point this repository at a managed database (see
PRODUCTION_READINESS_REPORT.md, "Remaining Manual Decisions"). The structured audit log line
emitted for every decision is the durable record in the meantime.
"""

import asyncio
import sqlite3
import threading
from pathlib import Path

from src.common.schemas import DecisionPage, DecisionRecord, UnderwritingDecision

_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    decision_id    TEXT PRIMARY KEY,
    application_id TEXT NOT NULL,
    decision       TEXT NOT NULL,
    decided_at     TEXT NOT NULL,
    decided_by     TEXT NOT NULL,
    request_id     TEXT,
    record_json    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_decisions_application ON decisions (application_id, decided_at);
CREATE INDEX IF NOT EXISTS ix_decisions_decided_at ON decisions (decided_at);
"""


class DecisionStore:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None

    # -- lifecycle -----------------------------------------------------------------------
    def open(self) -> None:
        if self._conn is not None:
            return
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(_SCHEMA)
        self._conn = conn

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.open()
        assert self._conn is not None
        return self._conn

    # -- sync operations -----------------------------------------------------------------
    def _save(self, record: DecisionRecord) -> None:
        with self._lock, self.conn:  # one transaction per decision
            self.conn.execute(
                "INSERT INTO decisions VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    record.decision_id,
                    record.memo.application_id,
                    record.memo.decision.value,
                    record.decided_at.isoformat(),
                    record.decided_by,
                    record.request_id,
                    record.model_dump_json(),
                ),
            )

    def _get(self, decision_id: str) -> DecisionRecord | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT record_json FROM decisions WHERE decision_id = ?", (decision_id,)
            ).fetchone()
        return DecisionRecord.model_validate_json(row[0]) if row else None

    def _list(
        self,
        limit: int,
        offset: int,
        application_id: str | None,
        decision: UnderwritingDecision | None,
    ) -> DecisionPage:
        where, params = [], []
        if application_id:
            where.append("application_id = ?")
            params.append(application_id)
        if decision:
            where.append("decision = ?")
            params.append(decision.value)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        with self._lock:
            total = self.conn.execute(
                f"SELECT COUNT(*) FROM decisions {clause}", params
            ).fetchone()[0]
            rows = self.conn.execute(
                f"SELECT record_json FROM decisions {clause} "
                "ORDER BY decided_at DESC LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
        return DecisionPage(
            items=[DecisionRecord.model_validate_json(r[0]) for r in rows],
            total=total,
            limit=limit,
            offset=offset,
        )

    def ping(self) -> bool:
        with self._lock:
            self.conn.execute("SELECT 1").fetchone()
        return True

    # -- async facade --------------------------------------------------------------------
    async def save(self, record: DecisionRecord) -> None:
        await asyncio.to_thread(self._save, record)

    async def get(self, decision_id: str) -> DecisionRecord | None:
        return await asyncio.to_thread(self._get, decision_id)

    async def list(
        self,
        limit: int = 50,
        offset: int = 0,
        application_id: str | None = None,
        decision: UnderwritingDecision | None = None,
    ) -> DecisionPage:
        return await asyncio.to_thread(self._list, limit, offset, application_id, decision)

    async def aping(self) -> bool:
        return await asyncio.to_thread(self.ping)
