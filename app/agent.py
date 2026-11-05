"""Public entry points used by the API and evals. The pipeline itself is the
LangGraph state machine in app.graph; each run is one LangGraph thread."""
import hashlib

from langgraph.types import Command
from sqlalchemy import select, update

from app import model_choice
from app.audit import Run, SessionLocal, log_event
from app.graph import graph, stash_raw_input
from app.telemetry import tracer

MAX_CONTEXT_TURNS = 16   # 8 user/assistant exchanges
FAILED_REPLY = "Sorry, I couldn't reach the model just now. Please try again."

model_choice.apply_saved()   # here so the API and `make evals` both use the saved model


def _config(run_id: str) -> dict:
    return {"configurable": {"thread_id": run_id}}


def _load(run_id: str) -> Run:
    with SessionLocal() as db:
        return db.get(Run, run_id)


def _fail(run_id: str, reason: str) -> None:
    """End a run that cannot continue, so it never stays "running" forever."""
    with SessionLocal() as db:
        run = db.get(Run, run_id)
        run.status, run.output, run.pending_tool_calls = "failed", FAILED_REPLY, None
        log_event(db, run_id, "error", error=reason)
        log_event(db, run_id, "run_finished", status=run.status, output=run.output)
        db.commit()


def fail_orphaned_runs() -> int:
    """Runs execute inside the API process, so any still "running" at startup died with the
    previous process. Assumes one API process; with several workers, move this to a job."""
    with SessionLocal() as db:
        ids = db.scalars(select(Run.id).where(Run.status == "running")).all()
    for run_id in ids:
        _fail(run_id, "The API restarted while this run was in progress.")
    return len(ids)


def context_from_runs(run_ids: list[str]) -> list[dict]:
    """Earlier turns of a conversation, rebuilt from stored runs. They hold only sanitised
    text, so a client can't put raw PII or invented assistant turns into the prompt."""
    turns = []
    with SessionLocal() as db:
        for run_id in run_ids:
            run = db.get(Run, run_id)
            if run is None:
                continue
            turns.append({"role": "user", "content": run.input_sanitized})
            if run.status == "blocked":
                reply = f"[Message blocked by input guardrail] {run.output}"
            elif run.status == "pending_approval" and run.pending_tool_calls:
                to = run.pending_tool_calls[0]["input"].get("to")
                reply = f"[Requested send_greeting to {to}; awaiting human approval]"
            else:
                reply = run.output
            if reply:
                turns.append({"role": "assistant", "content": reply})
    return turns


def create(user_input: str) -> Run:
    with SessionLocal() as db:
        run = Run(input_sha256=hashlib.sha256(user_input.encode()).hexdigest(), input_sanitized="",
                  status="running", messages=[], otel_trace_ids=[], total_input_tokens=0, total_output_tokens=0)
        db.add(run)
        db.commit()
        return run


def execute(run_id: str, user_input: str, context: list[dict] | None = None) -> Run:
    context = (context or [])[-MAX_CONTEXT_TURNS:]
    with tracer.start_as_current_span("agent.run") as span:
        span.set_attribute("app.run_id", run_id)
        stash_raw_input(run_id, user_input)
        try:
            graph.invoke({"run_id": run_id, "context": context}, _config(run_id))   # returns when finished or interrupted
        except Exception as e:
            _fail(run_id, f"{type(e).__name__}: {e}")
            raise
        return _load(run_id)


def start(user_input: str, context: list[dict] | None = None) -> Run:
    return execute(create(user_input).id, user_input, context)


def claim_decision(run_id: str) -> None:
    """Atomically take a run off the approval queue, so it can only be decided once."""
    with SessionLocal() as db:
        claimed = db.execute(update(Run).where(Run.id == run_id, Run.status == "pending_approval")
                             .values(status="running")).rowcount
        db.commit()
    if not claimed:
        raise ValueError("run is not awaiting approval")


def resume(run_id: str, approved: bool, approver: str, note: str = "", simulate_lie: bool = False) -> Run:
    """Human-in-the-loop decision: resumes the interrupted LangGraph thread."""
    with tracer.start_as_current_span("agent.resume") as span:
        span.set_attribute("app.run_id", run_id)
        span.set_attribute("hitl.approved", approved)
        decision = {"approved": approved, "approver": approver, "note": note, "simulate_lie": simulate_lie}
        try:
            graph.invoke(Command(resume=decision), _config(run_id))
        except Exception as e:
            _fail(run_id, f"{type(e).__name__}: {e}")
            raise
        return _load(run_id)


def decide(run_id: str, approved: bool, approver: str, note: str = "", simulate_lie: bool = False) -> Run:
    claim_decision(run_id)
    return resume(run_id, approved, approver, note, simulate_lie)


def in_background(fn, *args) -> None:
    """Run start/resume work after the response is sent. A failure is already on the run."""
    try:
        fn(*args)
    except Exception:
        pass


def mermaid() -> str:
    return graph.get_graph().draw_mermaid()
