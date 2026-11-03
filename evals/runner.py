"""Eval harness: runs the golden dataset end-to-end through the real agent (including
guardrails and the HITL pause/resume path), grades each case two ways, and reports
per-category pass rates. Usable from the CLI, the API, the dashboard and pytest/CI."""
import json
import sys
import time
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
    run = agent.start(case["input"])
    if run.status == "pending_approval":
        run = agent.decide(run.id, approved=case.get("approval") == "approve",
                           approver="eval-bot", note="simulated reviewer")
    fails = assertions(case, run)
    verdict = judge(case["input"], run.output or "", case.get("behaviour", ""))
    return {"id": case["id"], "category": case["category"], "run_id": run.id,
            "passed": not fails and verdict.passed, "assertion_failures": fails,
            "judge_score": verdict.score, "judge_reasoning": verdict.reasoning,
            "output": run.output, "latency_s": round(time.time() - t0, 2),
            "tokens": run.total_input_tokens + run.total_output_tokens}


def run_suite() -> dict:
    cases = yaml.safe_load(DATASET.read_text())
    results = [run_case(c) for c in cases]
    cats: dict = {}
    for r in results:
        c = cats.setdefault(r["category"], {"passed": 0, "total": 0})
        c["total"] += 1
        c["passed"] += r["passed"]
    passed = sum(r["passed"] for r in results)
    return {"model": settings.agent_model if not settings.offline else "offline-stub",
            "pass_rate": passed / len(results), "passed": passed, "total": len(results),
            "avg_judge_score": round(sum(r["judge_score"] for r in results) / len(results), 2),
            "threshold": settings.eval_threshold, "by_category": cats, "results": results}


if __name__ == "__main__":
    rep = run_suite()
    for r in rep["results"]:
        print(f"{'PASS' if r['passed'] else 'FAIL'}  {r['id']:<18} {r['category']:<14} "
              f"judge={r['judge_score']}/5  run={r['run_id']}  {'; '.join(r['assertion_failures'])}")
    print(f"\nPass rate {rep['passed']}/{rep['total']} = {rep['pass_rate']:.0%} "
          f"(threshold {rep['threshold']:.0%}), avg judge {rep['avg_judge_score']}/5")
    Path("eval_report.json").write_text(json.dumps(rep, indent=2))
    sys.exit(0 if rep["pass_rate"] >= rep["threshold"] else 1)
