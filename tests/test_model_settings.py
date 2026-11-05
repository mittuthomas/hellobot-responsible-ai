"""The model chosen in the Settings tab: validated, saved in SQLite, and applied at once."""
import pytest
from fastapi.testclient import TestClient

from app import model_choice as mc
from app.audit import SessionLocal, Setting
from app.config import settings
from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.setattr(settings, "agent_model", None)
    monkeypatch.setattr(settings, "judge_model", None)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    yield
    with SessionLocal() as db:
        db.query(Setting).delete()
        db.commit()


def _put(model):
    return client.put("/api/settings/model", json={"model": model})


def test_choice_applies_at_once_and_judge_follows():
    assert _put(mc.GEMINI).status_code == 200
    assert settings.agent_model == mc.GEMINI
    health = client.get("/health").json()
    assert health["agent_model"] == mc.GEMINI and health["judge_model"] == mc.GEMINI


def test_choice_survives_a_restart(monkeypatch):
    _put(mc.GEMINI)
    monkeypatch.setattr(settings, "agent_model", "from-dot-env")   # as a fresh process would start
    mc.apply_saved()
    assert settings.agent_model == mc.GEMINI


def test_dot_env_applies_when_nothing_is_saved(monkeypatch):
    monkeypatch.setattr(settings, "agent_model", "from-dot-env")
    mc.apply_saved()
    assert settings.agent_model == "from-dot-env"
    data = client.get("/api/settings/model").json()
    assert data["selected"] == "from-dot-env" and data["saved_at"] is None
    assert "from-dot-env" in [o["id"] for o in data["options"]]


def test_offline_choice_runs_on_the_stub_and_persists(monkeypatch):
    monkeypatch.setattr(settings, "force_offline", False)
    _put(mc.GEMINI)
    _put(mc.OFFLINE)
    assert settings.agent_model is None and settings.offline
    monkeypatch.setattr(settings, "agent_model", "from-dot-env")
    mc.apply_saved()
    assert settings.offline


def test_unknown_model_is_rejected():
    assert _put("evil:model").status_code == 400
    assert settings.agent_model is None


def test_unusable_model_is_rejected_with_the_reason_and_not_saved(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    r = _put(mc.GEMINI)
    assert r.status_code == 409 and "GEMINI_API_KEY" in r.json()["detail"]
    assert client.get("/api/settings/model").json()["saved_at"] is None


def test_unreachable_ollama_is_reported(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")
    options = {o["id"]: o for o in client.get("/api/settings/model").json()["options"]}
    assert options[mc.OLLAMA]["problem"] == "Ollama isn't running"
