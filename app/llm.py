"""LLM gateway: every model call goes through here, so tracing, token accounting and
refusal handling live in one place. Falls back to a deterministic stub when offline
(no API key or FORCE_OFFLINE=true) so demos never depend on conference Wi-Fi."""
import json

from pydantic import BaseModel

from app.config import settings
from app.telemetry import tracer

_client = None


class LLMResponse(BaseModel):
    text: str = ""
    tool_calls: list[dict] = []
    stop_reason: str | None = None
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    content: list[dict] = []   # full assistant content, replayed verbatim on the next turn


def _client_():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


def call(*, model: str, system: str, messages: list, tools=None, output_schema=None,
         max_tokens: int = 4000, purpose: str = "agent") -> LLMResponse:
    with tracer.start_as_current_span(f"chat {model}") as span:
        span.set_attribute("gen_ai.system", "anthropic")
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("app.llm.purpose", purpose)
        span.set_attribute("app.llm.offline", settings.offline)

        r = _stub(messages, tools, output_schema) if settings.offline else \
            _real(model, system, messages, tools, output_schema, max_tokens)

        span.set_attribute("gen_ai.response.model", r.model)
        span.set_attribute("gen_ai.response.finish_reasons", [r.stop_reason or ""])
        span.set_attribute("gen_ai.usage.input_tokens", r.input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", r.output_tokens)
        return r


def _real(model, system, messages, tools, output_schema, max_tokens) -> LLMResponse:
    output_config = {"effort": settings.effort}
    if output_schema:
        output_config["format"] = {"type": "json_schema", "schema": output_schema}
    kwargs = dict(model=model, max_tokens=max_tokens, system=system, messages=messages,
                  output_config=output_config,
                  # server-side refusal fallback: a declined request is re-run on
                  # Anthropic's recommended fallback model instead of just stopping
                  betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    if tools:
        kwargs["tools"] = tools
    resp = _client_().beta.messages.create(**kwargs)
    return LLMResponse(
        text="".join(b.text for b in resp.content if b.type == "text"),
        tool_calls=[{"id": b.id, "name": b.name, "input": b.input}
                    for b in resp.content if b.type == "tool_use"],
        stop_reason=resp.stop_reason, model=resp.model,
        input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens,
        content=[b.model_dump(mode="json", exclude_none=True) for b in resp.content],
    )


# ------------------------------------------------------------------ offline stub
def _text(t, **kw):
    return LLMResponse(text=t, stop_reason="end_turn", model="offline-stub",
                       content=[{"type": "text", "text": t}], **kw)


def _stub(messages, tools, output_schema) -> LLMResponse:
    last = messages[-1]["content"]
    if output_schema:  # judge
        answer = last.split("AGENT ANSWER:")[-1].lower()
        good = any(w in answer for w in ("hello", "hi ", "sent", "can't", "cannot", "couldn't", "wasn't", "not sent"))
        return _text(json.dumps({"score": 5 if good else 2, "passed": good,
                                 "reasoning": "Offline stub judge: checked for a greeting or an honest decline."}))
    if isinstance(last, list):  # tool results came back
        res = last[0]
        if res.get("is_error"):
            return _text("I wasn't able to send that greeting — a human reviewer declined it. "
                         "Let me know if you'd like to change anything.")
        return _text(f"Done! ✅ {res['content']}")
    if tools and "email" in last.lower():
        words = last.replace(",", " ").split()
        name = next((w for w in words if w[:1].isupper() and w.lower() not in ("please", "email", "hello", "hi")), "Friend")
        addr = next((w.strip(".") for w in words if "@" in w), f"{name.lower()}@example.com")
        call = {"id": "toolu_offline", "name": "send_greeting",
                "input": {"to": addr, "message": f"Hello, {name}! Have a great day."}}
        return LLMResponse(stop_reason="tool_use", model="offline-stub", tool_calls=[call],
                           content=[{"type": "tool_use", **call}])
    return _text("Hello, world! 👋 I'm HelloBot. How can I help you today?")
