"""What the dashboard shows for a run, worked out from its audit events. No Streamlit here, so it's testable."""
import copy
import hashlib
import json

EVENT_LABELS = {
    "run_started": "Run started", "input_guardrail": "Input guardrail", "llm_call": "LLM call",
    "approval_requested": "Approval requested", "human_decision": "Human decision", "tool_result": "Tool result",
    "output_guardrail": "Output guardrail", "llm_judge": "LLM judge", "judge_error": "Judge unavailable",
    "run_finished": "Run finished", "error": "Error", "honesty_check": "Honesty check",
    "honesty_error": "Honesty check failed: safe reply used", "honesty_violation": "Honesty violation: reply corrected",
    "simulated_lie": "Demo: dishonest reply injected",
}
STAGES = {"input": "Input guardrails", "agent": "Agent · LLM", "human": "Human in the loop",
          "output": "Output guardrails", "judge": "LLM judge"}
FINISHED = {"completed", "blocked", "failed"}


def event_tone(e: dict) -> str:
    event, d = e["event"], e["data"] or {}
    if event in ("error", "honesty_violation", "honesty_error", "judge_error"):
        return "bad"
    if event == "llm_call":
        return "bad" if d.get("status") == "error" else "info"
    if event == "honesty_check":
        return "bad" if d.get("claims_sent") else "ok"
    if event == "simulated_lie" or event == "approval_requested":
        return "warn"
    if event.endswith("guardrail"):
        return "bad" if not d.get("allowed") else "warn" if d.get("violations") else "ok"
    if event == "human_decision":
        return "ok" if d.get("approved") else "bad"
    if event == "llm_judge":
        return "ok" if d.get("passed") else "bad"
    if event == "run_finished":
        return "bad" if d.get("status") in ("blocked", "failed") else "ok"
    return "info"


def _llm_summary(d: dict) -> str:
    if d.get("status") == "error":
        return d.get("error") or "The call failed"
    response = d.get("response") or {}
    calls = response.get("tool_calls") or []
    if calls:
        return f"Decided to call {calls[0]['name']} → {calls[0]['input'].get('to', '')}"
    purpose = d.get("purpose", "agent")
    if purpose == "judge":
        return "Scored the answer"
    if purpose == "honesty_check":
        return "Checked the reply for a false claim"
    return f"Replied: {response.get('text', '')}"


def event_summary(e: dict) -> str:
    event, d = e["event"], e["data"] or {}
    if event == "run_started":
        turns = d.get("context_turns")
        return (d.get("sanitized_input") or "") + (f" (with {turns} earlier turns as context)" if turns else "")
    if event.endswith("guardrail"):
        hits = [f"{v['policy']} → {v['action']}" + (f" ({v['detail']})" if v.get("detail") and v["policy"] != "prompt_injection" else "")
                for v in d.get("violations") or []]
        return ", ".join(hits) or "No policy hits"
    if event == "llm_call":
        return _llm_summary(d)
    if event == "approval_requested":
        return ", ".join(f"{c['name']}({c['input'].get('to', '')})" for c in d.get("tool_calls") or [])
    if event == "human_decision":
        return f"{'Approved' if d.get('approved') else 'Denied'} by {e['actor']}" + (f': "{d["note"]}"' if d.get("note") else "")
    if event == "tool_result":
        return d.get("result") or ""
    if event == "llm_judge":
        return f"{d.get('score')}/5 · {d.get('reasoning', '')}"
    if event == "run_finished":
        return d.get("output") or ""
    if event in ("error", "judge_error", "honesty_error"):
        return d.get("error") or ""
    if event == "honesty_check":
        return f'Claims it was sent: "{d.get("evidence")}"' if d.get("claims_sent") else "No false claim found"
    if event == "honesty_violation":
        return f'Replaced the reply. Original: "{d.get("original_reply")}"'
    if event == "simulated_lie":
        return d.get("text") or ""
    return ""


def stages(status: str, events: list[dict]) -> dict[str, tuple[str, str]]:
    """The pipeline strip: for each control, (state, text). State is "", active, ok, warn or bad."""
    by = {}
    for e in events:
        by.setdefault(e["event"], []).append(e)
    last = lambda name: by[name][-1]["data"] if name in by else None
    out = {k: ("", "Waiting") for k in STAGES}
    running = status == "running"

    guard = last("input_guardrail")
    if not guard:
        out["input"] = ("active", "Checking…") if running else ("bad", "Did not run")
        return out
    blocks = list(dict.fromkeys(v["policy"] for v in guard.get("violations") or [] if v["action"] == "block"))
    if not guard.get("allowed"):
        out["input"] = ("bad", "Blocked: " + ", ".join(blocks))
        return {**out, **{k: ("", "Skipped") for k in ("agent", "human", "output", "judge")}}
    redacted = [v.get("detail") or v["policy"] for v in guard.get("violations") or [] if v["action"] == "redact"]
    out["input"] = ("warn", "Redacted " + ", ".join(redacted)) if redacted else ("ok", "Passed")

    asked, decision = last("approval_requested"), last("human_decision")
    agent_calls = [e["data"] for e in by.get("llm_call", []) if e["data"].get("purpose", "agent") == "agent"]
    answered = "honesty_check" in by or "honesty_error" in by or "output_guardrail" in by

    if status == "failed":
        out["agent"] = ("bad", "Model error")
    elif answered:
        out["agent"] = ("ok", "Replied")
    elif asked and not decision:
        out["agent"] = ("ok", "Wants send_greeting")
    elif running:
        out["agent"] = ("active", "Writing reply…" if decision else "Thinking…")

    if decision:
        out["human"] = ("ok", "Approved") if decision.get("approved") else ("bad", "Denied")
    elif asked:
        out["human"] = ("warn", "Awaiting approval")
    elif answered:
        out["human"] = ("", "Not needed")

    output = last("output_guardrail")
    corrected = "honesty_violation" in by or "honesty_error" in by
    if output:
        hits = [v["policy"] for v in output.get("violations") or []]
        if corrected:
            out["output"] = ("warn", "Corrected a false claim")
        elif hits:
            out["output"] = ("warn" if output.get("allowed") else "bad", ", ".join(hits))
        else:
            out["output"] = ("ok", "Passed")
    elif answered and running:
        out["output"] = ("active", "Honesty check…")

    verdict = last("llm_judge")
    if verdict:
        out["judge"] = ("ok" if verdict.get("passed") else "bad",
                        f"{verdict.get('score')}/5 {'pass' if verdict.get('passed') else 'fail'}")
    elif "judge_error" in by:
        out["judge"] = ("bad", "Unavailable")
    elif output and running:
        out["judge"] = ("active", "Scoring…")
    return out


# ------------------------------------------------------------------ tamper demo
def _digest(e: dict) -> str:
    """Must match app.audit._digest; a test holds the two together."""
    ts = e["ts"] if e["ts"].endswith("+00:00") else e["ts"].removesuffix("Z") + "+00:00"
    payload = json.dumps([e["run_id"] if "run_id" in e else None, ts, e["event"], e["actor"], e["data"], e["prev_hash"]],
                         sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def first_broken(run_id: str, events: list[dict]) -> int:
    """Index of the first event whose stored hash no longer matches its contents, or -1."""
    for i, e in enumerate(events):
        if e["hash"] != _digest({**e, "run_id": run_id}):
            return i
    return -1


def tampered_copy(events: list[dict]) -> list[dict]:
    """A copy with a past event edited, as someone covering their tracks might. The stored log is untouched."""
    copied = copy.deepcopy(events)
    if copied:
        target = copied[min(1, len(copied) - 1)]
        target["data"] = {**target["data"], "allowed": True, "violations": [],
                          "edited_by": "someone covering their tracks"}
    return copied
