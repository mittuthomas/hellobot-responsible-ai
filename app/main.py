"""FastAPI service. Interactive OpenAPI docs at /docs."""
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from app import agent, model_choice
from app.audit import AuditEvent, Run, SessionLocal, llm_calls, summarize_llm_calls, verify_chain
from app.config import settings


@asynccontextmanager
async def lifespan(_: FastAPI):
    agent.fail_orphaned_runs()
    yield


app = FastAPI(title="HelloBot - Responsible AI Agent", lifespan=lifespan,
              description="Guardrails · Observability & Audit · Evals · Human-in-the-loop · LLM-as-judge",
              version="1.0.0")


class ChatIn(BaseModel):
    message: str = Field(..., examples=["Please email a hello to Alice at alice@example.com"])
    context_run_ids: list[str] = Field(default=[], description="Earlier runs in this conversation, oldest first")
    background: bool = Field(default=False, description="Return at once and poll the run for progress")


class ModelIn(BaseModel):
    model: str = Field(..., examples=[model_choice.GEMINI])


class DecisionIn(BaseModel):
    approved: bool
    approver: str = "reviewer"
    note: str = ""
    simulate_lie: bool = Field(default=False, description="Demo: swap in a false reply after a denial, "
                                                          "to show the honesty check catching it")
    background: bool = Field(default=False, description="Return at once and poll the run for progress")


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
            "agent_model": settings.agent_model, "judge_model": settings.judge}


@app.post("/api/chat", response_model=RunOut, tags=["agent"])
def chat(body: ChatIn, tasks: BackgroundTasks):
    context = agent.context_from_runs(body.context_run_ids)
    if not body.background:
        return agent.start(body.message, context)
    run = agent.create(body.message)
    tasks.add_task(agent.in_background, agent.execute, run.id, body.message, context)
    return run


@app.post("/api/runs/{run_id}/decision", response_model=RunOut, tags=["human-in-the-loop"])
def decide(run_id: str, body: DecisionIn, tasks: BackgroundTasks):
    try:
        agent.claim_decision(run_id)
    except ValueError as e:
        raise HTTPException(409, str(e))
    args = (run_id, body.approved, body.approver, body.note, body.simulate_lie)
    if not body.background:
        return agent.resume(*args)
    tasks.add_task(agent.in_background, agent.resume, *args)
    return get_run(run_id)


@app.get("/api/approvals", response_model=list[RunOut], tags=["human-in-the-loop"])
def pending_approvals():
    with SessionLocal() as db:
        return db.scalars(select(Run).where(Run.status == "pending_approval")
                          .order_by(Run.created_at.desc())).all()


@app.get("/api/runs", response_model=list[RunOut], tags=["audit"])
def list_runs(limit: int = 50):
    with SessionLocal() as db:
        return db.scalars(select(Run).order_by(Run.created_at.desc()).limit(limit)).all()


@app.get("/api/runs/{run_id}", response_model=RunOut, tags=["audit"])
def get_run(run_id: str):
    with SessionLocal() as db:
        run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    return run


@app.get("/api/runs/{run_id}/events", response_model=list[EventOut], tags=["audit"])
def run_events(run_id: str):
    with SessionLocal() as db:
        return db.scalars(select(AuditEvent).where(AuditEvent.run_id == run_id).order_by(AuditEvent.id)).all()


@app.get("/api/runs/{run_id}/llm-calls", tags=["observability"])
def run_llm_calls(run_id: str):
    """Every model call in a run: timing, tokens, sizes, and the full request and response."""
    with SessionLocal() as db:
        calls = [e.data for e in llm_calls(db, run_id)]
    return {"summary": summarize_llm_calls(calls), "calls": calls}


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
        calls = [e.data for e in llm_calls(db)]
        decisions = db.scalars(select(AuditEvent).where(AuditEvent.event == "human_decision")).all()
    scores = [r.judge["score"] for r in runs]
    policies: dict = {}
    for e in guard:
        for v in e.data.get("violations", []):
            policies[v["policy"]] = policies.get(v["policy"], 0) + 1
    timed = [c["duration_ms"] for c in calls if c.get("duration_ms") is not None]
    return {"llm": {**summarize_llm_calls(calls), "avg_duration_ms": round(sum(timed) / len(timed), 1) if timed else None},
            "runs_by_status": by_status, "avg_judge_score": round(sum(scores) / len(scores), 2) if scores else None,
            "judge_pass_rate": round(sum(r.judge["passed"] for r in runs) / len(runs), 2) if runs else None,
            "human_decisions": {"approved": sum(bool(e.data.get("approved")) for e in decisions),
                                "denied": sum(not e.data.get("approved") for e in decisions)},
            "guardrail_hits": policies, "input_tokens": tokens[0] or 0, "output_tokens": tokens[1] or 0}


@app.get("/api/settings/model", tags=["settings"])
def get_model():
    return model_choice.describe()


@app.put("/api/settings/model", tags=["settings"])
def set_model(body: ModelIn):
    """Save the model choice. New model calls use it at once, and it survives restarts."""
    try:
        return model_choice.select(body.model)
    except KeyError:
        raise HTTPException(400, f"unknown model: {body.model}")
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.get("/api/graph", tags=["observability"])
def agent_graph():
    """The LangGraph state machine as a Mermaid diagram."""
    return {"mermaid": agent.mermaid()}


@app.post("/api/evals", tags=["evals"])
def run_evals():
    from evals.runner import run_suite
    return run_suite()


@app.post("/api/evals/jobs", tags=["evals"])
def start_eval_job(tasks: BackgroundTasks):
    """Run the suite in the background; poll the job for per-case progress."""
    from evals.runner import create_job, run_job
    job, is_new = create_job()
    if is_new:
        tasks.add_task(run_job, job)
    return job


@app.get("/api/evals/jobs/{job_id}", tags=["evals"])
def eval_job(job_id: str):
    from evals.runner import JOBS
    if job_id not in JOBS:
        raise HTTPException(404, "job not found")
    return JOBS[job_id]


@app.post("/api/evals/jobs/{job_id}/stop", tags=["evals"])
def stop_eval_job(job_id: str):
    """Stops after the case in progress; the report covers the cases that finished."""
    from evals.runner import JOBS
    if job_id not in JOBS:
        raise HTTPException(404, "job not found")
    JOBS[job_id]["stop_requested"] = True
    return JOBS[job_id]
