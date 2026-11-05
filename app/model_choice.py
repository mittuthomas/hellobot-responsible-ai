"""Which model the agent uses: chosen in the dashboard and saved in SQLite, so it survives restarts.
With nothing saved yet, AGENT_MODEL from .env applies. FORCE_OFFLINE still wins over both."""
import os

import httpx

from app.audit import SessionLocal, Setting, _now
from app.config import settings

KEY = "agent_model"
OFFLINE = "offline"
GEMINI = "google_genai:gemini-3.5-flash-lite"
OLLAMA = "ollama:gemma4:12b"
LABELS = {GEMINI: "Gemini 3.5 Flash Lite", OLLAMA: "Ollama · gemma4:12b (local)", OFFLINE: "Offline stub (scripted replies)"}


def _ollama_problem(model: str) -> str | None:
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    host = host if "://" in host else f"http://{host}"
    try:
        names = [m["name"] for m in httpx.get(f"{host}/api/tags", timeout=1).raise_for_status().json()["models"]]
    except (httpx.HTTPError, KeyError, ValueError):
        return "Ollama isn't running"
    return None if model in names else f"Not pulled yet: ollama pull {model}"


def problem_with(model: str) -> str | None:
    """Why this model can't be used right now, or None if it can."""
    if model.startswith("google_genai:") and not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        return "GEMINI_API_KEY isn't set"
    if model.startswith("ollama:"):
        return _ollama_problem(model.split(":", 1)[1])
    return None


def current() -> str:
    return settings.agent_model or OFFLINE


def _apply(model: str) -> None:
    settings.agent_model = None if model == OFFLINE else model


def apply_saved() -> None:
    with SessionLocal() as db:
        saved = db.get(Setting, KEY)
    if saved:
        _apply(saved.value)


def describe() -> dict:
    selected = current()
    ids = [*LABELS, *([selected] if selected not in LABELS else [])]   # a model set in .env stays selectable
    with SessionLocal() as db:
        saved = db.get(Setting, KEY)
    return {"selected": selected, "judge": settings.judge or OFFLINE, "judge_is_separate": bool(settings.judge_model),
            "force_offline": settings.force_offline, "saved_at": saved.updated_at if saved else None,
            "options": [{"id": m, "label": LABELS.get(m, m), "problem": problem_with(m)} for m in ids]}


def select(model: str) -> dict:
    """Save and apply a choice. New model calls use it at once, including runs already in progress."""
    if model not in LABELS and model != current():
        raise KeyError(model)
    if problem := problem_with(model):
        raise ValueError(problem)
    with SessionLocal() as db:
        db.merge(Setting(key=KEY, value=model, updated_at=_now()))
        db.commit()
    _apply(model)
    return describe()
