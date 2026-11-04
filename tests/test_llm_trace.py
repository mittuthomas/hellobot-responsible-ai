"""Every LLM call is audited and traced: time, tokens, sizes, full request and response."""
import json
from datetime import timezone

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from app import llm
from app.audit import AuditEvent, SessionLocal, _digest
from app.main import app

client = TestClient(app, raise_server_exceptions=False)


def _calls(run_id):
    return client.get(f"/api/runs/{run_id}/llm-calls").json()


def test_agent_and_judge_calls_are_all_recorded():
    run = client.post("/api/chat", json={"message": "Hi!"}).json()
    data = _calls(run["id"])
    assert [c["purpose"] for c in data["calls"]] == ["agent", "judge"]
    assert data["summary"]["calls"] == 2 and data["summary"]["errors"] == 0
    for c in data["calls"]:
        assert c["status"] == "ok" and c["duration_ms"] >= 0 and c["started_at"]
        assert c["request_bytes"] > 0 and c["response_bytes"] > 0
        assert len(c["otel_trace_id"]) == 32 and len(c["otel_span_id"]) == 16
    agent, judge = data["calls"]
    assert agent["request"]["system"].startswith("You are HelloBot")
    assert agent["request"]["messages"] == [{"role": "human", "content": "Hi!"}]
    assert agent["request"]["tools"][0]["function"]["name"] == "send_greeting"
    assert agent["response"]["text"] == run["output"]
    assert "AGENT ANSWER" in judge["request"]["messages"][0]["content"]


def test_run_totals_include_judge_tokens(monkeypatch):
    real = llm._stub_chat

    def counted(messages, tools):
        r = real(messages, tools)
        r.input_tokens, r.output_tokens = 10, 5
        return r

    monkeypatch.setattr(llm, "_stub_chat", counted)
    run = client.post("/api/chat", json={"message": "Hi!"}).json()
    assert (run["total_input_tokens"], run["total_output_tokens"]) == (20, 10)
    events = client.get(f"/api/runs/{run['id']}/events").json()
    finished = next(e for e in events if e["event"] == "run_finished")
    assert finished["data"]["llm"]["total_tokens"] == 30 and finished["data"]["llm"]["calls"] == 2


def test_failed_call_is_recorded_with_its_error(monkeypatch):
    def down(messages, tools):
        raise RuntimeError("503 model overloaded")

    monkeypatch.setattr(llm, "_stub_chat", down)
    assert client.post("/api/chat", json={"message": "Hi!"}).status_code == 500
    latest = client.get("/api/runs", params={"limit": 1}).json()[0]
    data = _calls(latest["id"])
    call = data["calls"][0]
    assert call["status"] == "error" and "503 model overloaded" in call["error"]
    assert call["response"] is None and call["response_bytes"] == 0 and call["request_bytes"] > 0
    assert data["summary"]["errors"] == 1


def test_raw_pii_never_reaches_the_recorded_request():
    run = client.post("/api/chat", json={"message": "Hello, my SSN is 123-45-6789"}).json()
    assert "123-45-6789" not in json.dumps(client.get(f"/api/runs/{run['id']}/events").json())
    assert "<US_SSN>" in _calls(run["id"])["calls"][0]["request"]["messages"][0]["content"]


def test_large_llm_events_keep_a_valid_hash():
    run = client.post("/api/chat", json={"message": "Hi!"}).json()
    with SessionLocal() as db:
        events = db.query(AuditEvent).filter_by(run_id=run["id"], event="llm_call").all()
        assert events
        for e in events:
            ts = e.ts.replace(tzinfo=timezone.utc)
            assert e.hash == _digest(e.run_id, ts, e.event, e.actor, e.data, e.prev_hash)


def test_provider_usage_is_captured(monkeypatch):
    class FakeModel:
        def invoke(self, messages):
            return AIMessage("hi", response_metadata={"finish_reason": "STOP", "model_name": "fake-1"},
                             usage_metadata={"input_tokens": 7, "output_tokens": 9, "total_tokens": 16,
                                             "output_token_details": {"reasoning": 4},
                                             "input_token_details": {"cache_read": 2}})

    monkeypatch.setattr(llm, "_model", lambda name: FakeModel())
    monkeypatch.setattr(llm.settings, "force_offline", False)
    monkeypatch.setattr(llm.settings, "agent_model", "fake:fake-1")
    seen = []
    with llm.record_calls(seen.append):
        llm.chat(model="fake:fake-1", system="s", messages=[HumanMessage("x")])
    c = seen[0]
    assert (c["provider"], c["model_requested"], c["model"]) == ("fake", "fake:fake-1", "fake-1")
    assert (c["input_tokens"], c["output_tokens"], c["total_tokens"]) == (7, 9, 16)
    assert (c["reasoning_tokens"], c["cached_input_tokens"]) == (4, 2)
    assert c["response"]["response_metadata"]["finish_reason"] == "STOP"


def test_metrics_report_llm_totals():
    client.post("/api/chat", json={"message": "Hi!"})
    llm_metrics = client.get("/api/metrics").json()["llm"]
    assert llm_metrics["calls"] >= 2 and llm_metrics["avg_duration_ms"] is not None
