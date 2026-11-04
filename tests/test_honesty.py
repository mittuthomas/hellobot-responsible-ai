"""The honesty check: after a human denial, a reply that claims the action happened is corrected."""
from fastapi.testclient import TestClient

from app import llm
from app.judge import SAFE_DENIAL_REPLY
from app.main import app

client = TestClient(app)


def _events(run_id):
    return [e["event"] for e in client.get(f"/api/runs/{run_id}/events").json()]


def _ask_and_decide(approved):
    r = client.post("/api/chat", json={"message": "Please email a hello to Alice at alice@example.com"}).json()
    return client.post(f"/api/runs/{r['id']}/decision", json={"approved": approved, "approver": "qa"}).json()


def test_dishonest_reply_after_denial_is_corrected(monkeypatch):
    from langchain_core.messages import AIMessage, ToolMessage
    real = llm.chat

    def lying_agent(**kw):
        if kw.get("tools") and isinstance(kw["messages"][-1], ToolMessage):   # the reply after the tool result
            text = "Done! I've sent the greeting to Alice."
            return llm.LLMResponse(text=text, model="lying-stub", message=AIMessage(text))
        return real(**kw)

    monkeypatch.setattr(llm, "chat", lying_agent)
    done = _ask_and_decide(approved=False)
    assert done["output"] == SAFE_DENIAL_REPLY
    assert "honesty_violation" in _events(done["id"])


def test_honest_denial_passes_unchanged():
    done = _ask_and_decide(approved=False)
    ev = _events(done["id"])
    assert "honesty_check" in ev and "honesty_violation" not in ev
    assert done["output"] != SAFE_DENIAL_REPLY


def test_approved_run_skips_honesty_check():
    done = _ask_and_decide(approved=True)
    assert "honesty_check" not in _events(done["id"])


def test_judge_outage_still_completes_run(monkeypatch):
    import app.graph as g

    def boom(*a, **k):
        raise RuntimeError("judge API down")

    monkeypatch.setattr(g, "judge", boom)
    r = client.post("/api/chat", json={"message": "Hi there!"}).json()
    assert r["status"] == "completed" and r["output"] and r["judge"] is None
    assert "judge_error" in _events(r["id"])
