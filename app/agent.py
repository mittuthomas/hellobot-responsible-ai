"""Public entry points used by the API and evals. The pipeline itself is the
LangGraph state machine in app.graph; each run is one LangGraph thread."""
import hashlib

from langgraph.types import Command

from app.audit import Run, SessionLocal
from app.graph import graph, stash_raw_input
from app.telemetry import tracer


def _config(run_id: str) -> dict:
    return {"configurable": {"thread_id": run_id}}


def _load(run_id: str) -> Run:
    with SessionLocal() as db:
        return db.get(Run, run_id)


def start(user_input: str) -> Run:
    with tracer.start_as_current_span("agent.run") as span, SessionLocal() as db:
        run = Run(input_sha256=hashlib.sha256(user_input.encode()).hexdigest(), input_sanitized="",
                  status="running", messages=[], otel_trace_ids=[], total_input_tokens=0, total_output_tokens=0)
        db.add(run)
        db.commit()
        span.set_attribute("app.run_id", run.id)
        stash_raw_input(run.id, user_input)
        graph.invoke({"run_id": run.id}, _config(run.id))   # returns when finished or interrupted
        return _load(run.id)


def decide(run_id: str, approved: bool, approver: str, note: str = "") -> Run:
    """Human-in-the-loop decision: resumes the interrupted LangGraph thread."""
    run = _load(run_id)
    if run is None or run.status != "pending_approval":
        raise ValueError("run is not awaiting approval")
    with tracer.start_as_current_span("agent.resume") as span:
        span.set_attribute("app.run_id", run_id)
        span.set_attribute("hitl.approved", approved)
        graph.invoke(Command(resume={"approved": approved, "approver": approver, "note": note}), _config(run_id))
        return _load(run_id)


def mermaid() -> str:
    return graph.get_graph().draw_mermaid()
