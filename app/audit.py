"""Audit store: runs + an append-only, SHA-256 hash-chained event log (SQLAlchemy).
Any edit or deletion of a past event breaks the chain and is reported by verify_chain()."""
import hashlib
import json
import threading
import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.config import settings

engine = create_engine(settings.database_url, connect_args={"check_same_thread": False}
                       if settings.database_url.startswith("sqlite") else {})
SessionLocal = sessionmaker(engine, expire_on_commit=False)
_chain_lock = threading.Lock()
GENESIS = "0" * 64


class Base(DeclarativeBase):
    pass


def _now():
    return datetime.now(timezone.utc)


class Run(Base):
    __tablename__ = "runs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: uuid.uuid4().hex[:12])
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    status: Mapped[str] = mapped_column(String(32), default="running")
    # running | pending_approval | completed | blocked | failed
    input_sha256: Mapped[str] = mapped_column(String(64))
    input_sanitized: Mapped[str] = mapped_column(Text)
    output: Mapped[str | None] = mapped_column(Text, nullable=True)
    messages: Mapped[list] = mapped_column(JSON, default=list)
    pending_tool_calls: Mapped[list | None] = mapped_column(JSON, nullable=True)
    judge: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    otel_trace_ids: Mapped[list] = mapped_column(JSON, default=list)
    total_input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_output_tokens: Mapped[int] = mapped_column(Integer, default=0)


class AuditEvent(Base):
    __tablename__ = "audit_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    event: Mapped[str] = mapped_column(String(64))
    actor: Mapped[str] = mapped_column(String(64), default="agent")
    data: Mapped[dict] = mapped_column(JSON, default=dict)
    prev_hash: Mapped[str] = mapped_column(String(64))
    hash: Mapped[str] = mapped_column(String(64))


class Setting(Base):
    """Choices made in the dashboard that must survive a restart. Not part of the audit chain."""
    __tablename__ = "settings"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


Base.metadata.create_all(engine)


def _digest(run_id, ts, event, actor, data, prev_hash) -> str:
    payload = json.dumps([run_id, ts.isoformat(), event, actor, data, prev_hash], sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def log_event(db: Session, run_id: str, event: str, actor: str = "agent", **data) -> AuditEvent:
    with _chain_lock:
        last = db.scalar(select(AuditEvent).order_by(AuditEvent.id.desc()).limit(1))
        prev = last.hash if last else GENESIS
        ts = _now().replace(microsecond=0)
        data = json.loads(json.dumps(data, default=str))
        ev = AuditEvent(run_id=run_id, ts=ts, event=event, actor=actor, data=data, prev_hash=prev,
                        hash=_digest(run_id, ts, event, actor, data, prev))
        db.add(ev)
        db.commit()
        return ev


def llm_calls(db: Session, run_id: str | None = None) -> list[AuditEvent]:
    query = select(AuditEvent).where(AuditEvent.event == "llm_call").order_by(AuditEvent.id)
    return db.scalars(query.where(AuditEvent.run_id == run_id) if run_id else query).all()


def summarize_llm_calls(calls: list[dict]) -> dict:
    """Totals across llm_call event data. Events from before this was recorded count as 0."""
    def total(key):
        return sum(c.get(key) or 0 for c in calls)
    return {"calls": len(calls), "errors": sum(c.get("status") == "error" for c in calls),
            "duration_ms": round(total("duration_ms"), 1), "input_tokens": total("input_tokens"),
            "output_tokens": total("output_tokens"),
            "total_tokens": sum(c.get("total_tokens") or (c.get("input_tokens") or 0) + (c.get("output_tokens") or 0)
                                for c in calls),
            "request_bytes": total("request_bytes"), "response_bytes": total("response_bytes")}


def verify_chain(db: Session) -> dict:
    prev = GENESIS
    events = db.scalars(select(AuditEvent).order_by(AuditEvent.id)).all()
    for e in events:
        ts = e.ts if e.ts.tzinfo else e.ts.replace(tzinfo=timezone.utc)
        if e.prev_hash != prev or e.hash != _digest(e.run_id, ts, e.event, e.actor, e.data, e.prev_hash):
            return {"valid": False, "events_checked": len(events), "broken_at_event_id": e.id}
        prev = e.hash
    return {"valid": True, "events_checked": len(events), "head": prev}
