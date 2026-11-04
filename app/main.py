"""FastAPI service. Interactive OpenAPI docs at /docs."""
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from app import agent
from app.audit import AuditEvent, Run, SessionLocal, verify_chain
from app.config import settings

app = FastAPI(title="HelloBot - Responsible AI Agent",
              description="Guardrails · Observability & Audit · Evals · Human-in-the-loop · LLM-as-judge",
              version="1.0.0")


class ChatIn(BaseModel):
    message: str = Field(..., examples=["Please email a hello to Alice at alice@example.com"])


class DecisionIn(BaseModel):
    approved: bool
    approver: str = "reviewer"
    note: str = ""


class RunOut(BaseModel):
    model_config = {"from_attributes": True}
    id: str
    created_at: datetime
    status: str
    input_sanitized: str
    output: str | None
    pending_tool_calls: list | None
    judge: dict | None
    otel_trace_ids: list
    total_input_tokens: int
    total_output_tokens: int


class EventOut(BaseModel):
    model_config = {"from_attributes": True}
    id: int
    ts: datetime
    event: str
    actor: str
    data: dict
    prev_hash: str
    hash: str


@app.get("/health")
def health():
    return {"status": "ok", "mode": "offline-stub" if settings.offline else "live",
            "agent_model": settings.agent_model, "judge_model": settings.judge_model}


@app.post("/api/chat", response_model=RunOut, tags=["agent"])
def chat(body: ChatIn):
    return agent.start(body.message)


@app.post("/api/runs/{run_id}/decision", response_model=RunOut, tags=["human-in-the-loop"])
def decide(run_id: str, body: DecisionIn):
    try:
        return agent.decide(run_id, body.approved, body.approver, body.note)
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.get("/api/approvals", response_model=list[RunOut], tags=["human-in-the-loop"])
def pending_approvals():
    with SessionLocal() as db:
        return db.scalars(select(Run).where(Run.status == "pending_approval")
                          .order_by(Run.created_at.desc())).all()


@app.get("/api/runs", response_model=list[RunOut], tags=["audit"])
def list_runs(limit: int = 50):
    with SessionLocal() as db:
        return db.scalars(select(Run).order_by(Run.created_at.desc()).limit(limit)).all()


@app.get("/api/runs/{run_id}/events", response_model=list[EventOut], tags=["audit"])
def run_events(run_id: str):
    with SessionLocal() as db:
        return db.scalars(select(AuditEvent).where(AuditEvent.run_id == run_id).order_by(AuditEvent.id)).all()


@app.get("/api/audit/verify", tags=["audit"])
def audit_verify():
    with SessionLocal() as db:
        return verify_chain(db)


@app.get("/api/metrics", tags=["observability"])
def metrics():
    with SessionLocal() as db:
        by_status = dict(db.execute(select(Run.status, func.count()).group_by(Run.status)).all())
        runs = db.scalars(select(Run).where(Run.judge.is_not(None))).all()
        guard = db.scalars(select(AuditEvent).where(AuditEvent.event.in_(["input_guardrail", "output_guardrail"]))).all()
        tokens = db.execute(select(func.sum(Run.total_input_tokens), func.sum(Run.total_output_tokens))).one()
    scores = [r.judge["score"] for r in runs]
    policies: dict = {}
    for e in guard:
        for v in e.data.get("violations", []):
            policies[v["policy"]] = policies.get(v["policy"], 0) + 1
    return {"runs_by_status": by_status, "avg_judge_score": round(sum(scores) / len(scores), 2) if scores else None,
            "judge_pass_rate": round(sum(r.judge["passed"] for r in runs) / len(runs), 2) if runs else None,
            "guardrail_hits": policies, "input_tokens": tokens[0] or 0, "output_tokens": tokens[1] or 0}


@app.get("/api/graph", tags=["observability"])
def agent_graph():
    """The LangGraph state machine as a Mermaid diagram."""
    return {"mermaid": agent.mermaid()}


@app.post("/api/evals", tags=["evals"])
def run_evals():
    from evals.runner import run_suite
    return run_suite()
