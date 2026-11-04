"""Provider-neutral model wiring: the same code builds a tool-calling chat model for
each provider. No network calls; fake keys only satisfy client construction."""
import pytest

from app import llm
from app.graph import TOOLS


@pytest.mark.parametrize("name,cls", [
    ("openai:gpt-4.1-mini", "ChatOpenAI"),
    ("mistralai:mistral-large-latest", "ChatMistralAI"),
    ("anthropic:claude-opus-5-5", "ChatAnthropic"),
])
def test_any_provider_builds_and_binds_tools(monkeypatch, name, cls):
    for key in ("OPENAI_API_KEY", "MISTRAL_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(key, "test-key")
    model = llm._model(name)
    assert type(model).__name__ == cls
    bound = model.bind_tools([llm._tool_spec(t) for t in TOOLS])
    assert bound is not None


def test_structured_output_retries_once_then_validates(monkeypatch):
    from app.judge import Verdict
    replies = iter(["not json at all", '{"reasoning": "ok", "score": 5, "passed": true}'])
    monkeypatch.setattr(llm, "chat", lambda **kw: llm.LLMResponse(text=next(replies)))
    v = llm.structured(model="any:model", system="s", prompt="p", schema=Verdict, purpose="test")
    assert v.score == 5 and v.passed
