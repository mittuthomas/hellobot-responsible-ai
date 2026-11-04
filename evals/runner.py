"""Eval harness: runs the golden dataset end-to-end through the real agent (including
guardrails and the HITL pause/resume path), grades each case two ways, and reports
per-category pass rates. Usable from the CLI, the API, the dashboard and pytest/CI."""
import json
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path

import yaml

from app import agent
from app.config import settings
from app.judge import judge

DATASET = Path(__file__).parent / "dataset.yaml"


def assertions(case: dict, run) -> list[str]:
    out, fails = (run.output or "").lower(), []
    if run.status != case["expected_status"]:
        fails.append(f"status={run.status}, expected {case['expected_status']}")
    if "contains_any" in case and not any(s in out for s in case["contains_any"]):
        fails.append(f"missing any of {case['contains_any']}")
    fails += [f"contains forbidden '{s}'" for s in case.get("not_contains", []) if s.lower() in out]
    return fails


def run_case(case: dict) -> dict:
    t0 = time.time()
    context = case.get("context") or []
    run = agent.start(case["input"], context)
    if run.status == "pending_approval":
        run = agent.decide(run.id, approved=case.get("approval") == "approve",
                           approver="eval-bot", note="simulated reviewer", simulate_lie=bool(case.get("lie")))
    fails = assertions(case, run)
    earlier = "".join(f"{t['role']}: {t['content']}\n" for t in context)
    verdict = judge(f"{earlier}user: {case['input']}" if earlier else case["input"],
                    run.output or "", case.get("behaviour", ""))
    return {"id": case["id"], "category": case["category"], "run_id": run.id,
            "passed": not fails and verdict.passed, "assertion_failures": fails,
            "judge_score": verdict.score, "judge_reasoning": verdict.reasoning,
            "output": run.output, "latency_s": round(time.time() - t0, 2),
            "tokens": run.total_input_tokens + run.total_output_tokens}


def _load_cases() -> list[dict]:
    return yaml.safe_load(DATASET.read_text())


def _report(results: list[dict]) -> dict:
    cats: dict = {}
    for r in results:
        c = cats.setdefault(r["category"], {"passed": 0, "total": 0})
        c["total"] += 1
        c["passed"] += r["passed"]
    passed = sum(r["passed"] for r in results)
    scores = [r["judge_score"] for r in results if r["judge_score"] is not None]
    return {"model": settings.agent_model if not settings.offline else "offline-stub",
            "pass_rate": passed / len(results) if results else 0, "passed": passed, "total": len(results),
            "avg_judge_score": round(sum(scores) / len(scores), 2) if scores else None,
            "threshold": settings.eval_threshold, "by_category": cats, "results": results}


def run_suite(on_result: Callable[[dict], None] | None = None,
              should_stop: Callable[[], bool] | None = None) -> dict:
    results = []
    for case in _load_cases():
        if should_stop and should_stop():
            break
        try:
            result = run_case(case)
        except Exception as e:   # one broken case (a model outage, say) must not lose the others
            result = {"id": case["id"], "category": case["category"], "run_id": None, "passed": False,
                      "assertion_failures": [f"{type(e).__name__}: {e}"], "judge_score": None,
                      "judge_reasoning": "", "output": None, "latency_s": 0, "tokens": 0}
        results.append(result)
        if on_result:
            on_result(result)
    return _report(results)


# ------------------------------------------------------------------ background jobs
# In memory on purpose: a job is only progress for whoever is watching. Results also reach the audit log.
JOBS: dict[str, dict] = {}


def create_job() -> tuple[dict, bool]:
    """Returns (job, is_new). Only one suite runs at a time; a second request gets the running one."""
    for job in JOBS.values():
        if job["state"] == "running":
            return job, False
    job = {"id": uuid.uuid4().hex[:12], "state": "running", "stop_requested": False,
           "cases": [{k: c.get(k) for k in ("id", "category", "input")} for c in _load_cases()],
           "results": [], "report": None}
    JOBS[job["id"]] = job
    return job, True


def run_job(job: dict) -> None:
    try:
        job["report"] = run_suite(on_result=job["results"].append, should_stop=lambda: job["stop_requested"])
        job["state"] = "stopped" if job["stop_requested"] else "done"
    except Exception as e:
        job["state"], job["error"] = "error", f"{type(e).__name__}: {e}"


if __name__ == "__main__":
    rep = run_suite()
    for r in rep["results"]:
        print(f"{'PASS' if r['passed'] else 'FAIL'}  {r['id']:<18} {r['category']:<14} "
              f"judge={r['judge_score']}/5  run={r['run_id']}  {'; '.join(r['assertion_failures'])}")
    print(f"\nPass rate {rep['passed']}/{rep['total']} = {rep['pass_rate']:.0%} "
          f"(threshold {rep['threshold']:.0%}), avg judge {rep['avg_judge_score']}/5")
    Path("eval_report.json").write_text(json.dumps(rep, indent=2))
    sys.exit(0 if rep["pass_rate"] >= rep["threshold"] else 1)
