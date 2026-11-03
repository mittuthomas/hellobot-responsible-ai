"""CI quality gate: the golden-dataset pass rate must meet the threshold."""
from evals.runner import run_suite


def test_eval_suite_meets_threshold():
    rep = run_suite()
    failed = [(r["id"], r["assertion_failures"], r["judge_reasoning"]) for r in rep["results"] if not r["passed"]]
    assert rep["pass_rate"] >= rep["threshold"], failed
