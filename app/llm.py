"""LLM gateway: every model call goes through here, so tracing, token accounting and
refusal handling live in one place, whatever the provider.

Models are chosen by a "provider:model" string, for example
    openai:gpt-4.1   mistralai:mistral-large-latest   anthropic:<model>
    google_genai:gemini-3.5-flash-lite   ollama:gemma4:12b
and created with LangChain's standard chat-model interface (init_chat_model), so the
same code drives any provider that supports tool calling. Each provider reads its own
API key from the environment (OPENAI_API_KEY, MISTRAL_API_KEY, ...).

With no model configured (or FORCE_OFFLINE=true) a deterministic stub answers instead,
so demos and CI never depend on network access.

chat() also measures every call (time, tokens, sizes, full request and response). The
OTel span always gets this; the audit log gets it through a recorder set by record_calls().
"""
import json
import re
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, TypeVar

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, ValidationError

from app.config import settings
from app.telemetry import tracer

T = TypeVar("T", bound=BaseModel)

# Finish reasons meaning the provider withheld the answer (lowercase; Gemini reports uppercase).
_REFUSAL_FINISH_REASONS = {"refusal", "content_filter", "safety", "recitation", "blocklist",
                           "prohibited_content", "spii", "image_safety"}


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)   # [{id, name, input}]
    stop_reason: str = "end_turn"                          # end_turn | tool_use | refusal
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    message: Any = None   # the provider's AIMessage, appended to the conversation as-is
    usage: dict = field(default_factory=dict)              # the provider's full token breakdown


@lru_cache(maxsize=8)
def _model(name: str):
    from langchain.chat_models import init_chat_model
    if _provider(name) == "ollama":
        # ChatOllama silently ignores max_tokens. Thinking is off because it made local
        # models about 5x slower here, enough to outlast the dashboard's request timeout.
        return init_chat_model(name, num_predict=settings.max_output_tokens, reasoning=False)
    return init_chat_model(name, max_tokens=settings.max_output_tokens)


def _provider(name: str | None) -> str:
    return (name or "offline").split(":", 1)[0]


def _tool_spec(t: dict) -> dict:
    """Our neutral tool definition -> the function-calling format every provider accepts."""
    return {"type": "function",
            "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}


# ------------------------------------------------------------------ per-call record
_recorder: ContextVar[Callable[[dict], None] | None] = ContextVar("llm_recorder", default=None)


@contextmanager
def record_calls(recorder: Callable[[dict], None]):
    """Send a record of every chat() call in this block, failed ones included, to `recorder`."""
    token = _recorder.set(recorder)
    try:
        yield
    finally:
        _recorder.reset(token)


def _compact(payload: Any) -> str:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False, default=str)


def _message_dump(m: BaseMessage) -> dict:
    d = {"role": m.type, "content": m.content}
    for key in ("tool_calls", "tool_call_id", "status"):
        if value := getattr(m, key, None):
            d[key] = value
    return d


def _request_payload(system: str, messages: list[BaseMessage], tools) -> dict:
    """What we hand to the model: provider-neutral, and never includes credentials."""
    return {"system": system, "messages": [_message_dump(m) for m in messages],
            "tools": [_tool_spec(t) for t in tools or []],
            "params": {"max_tokens": settings.max_output_tokens}}


def _response_payload(r: LLMResponse) -> dict:
    msg = r.message
    return {"text": r.text, "tool_calls": r.tool_calls, "stop_reason": r.stop_reason,
            "content": getattr(msg, "content", None),
            "response_metadata": getattr(msg, "response_metadata", None) or {},
            "usage": r.usage}


def _record_call(span, *, model, purpose, started_at, started, request, r: LLMResponse | None = None,
                 error: Exception | None = None) -> None:
    duration_ms = round((time.perf_counter() - started) * 1000, 1)
    response = _response_payload(r) if r else None
    request_json, response_json = _compact(request), _compact(response) if r else ""
    usage = r.usage if r else {}
    ctx = span.get_span_context()

    span.set_attribute("app.llm.duration_ms", duration_ms)
    span.set_attribute("app.llm.request_bytes", len(request_json.encode()))
    span.set_attribute("app.llm.response_bytes", len(response_json.encode()))
    if settings.otel_capture_llm_content:
        span.set_attribute("gen_ai.input.messages", request_json)
        if r:
            span.set_attribute("gen_ai.output.messages", response_json)

    if recorder := _recorder.get():
        input_tokens, output_tokens = (r.input_tokens, r.output_tokens) if r else (0, 0)
        recorder({
            "call_id": uuid.uuid4().hex[:12], "purpose": purpose, "provider": _provider(model),
            "model_requested": model or "offline-stub", "model": r.model if r else model or "offline-stub",
            "status": "error" if error else "ok", "error": f"{type(error).__name__}: {error}" if error else None,
            "stop_reason": r.stop_reason if r else None,
            "started_at": started_at.isoformat(timespec="milliseconds"), "duration_ms": duration_ms,
            "input_tokens": input_tokens, "output_tokens": output_tokens,
            "total_tokens": usage.get("total_tokens", input_tokens + output_tokens),
            "reasoning_tokens": usage.get("output_token_details", {}).get("reasoning", 0),
            "cached_input_tokens": usage.get("input_token_details", {}).get("cache_read", 0),
            "request_bytes": len(request_json.encode()), "response_bytes": len(response_json.encode()),
            "message_count": len(request["messages"]), "tool_call_count": len(r.tool_calls) if r else 0,
            "otel_trace_id": format(ctx.trace_id, "032x"), "otel_span_id": format(ctx.span_id, "016x"),
            "request": request, "response": response,
        })


def chat(*, model: str | None, system: str, messages: list[BaseMessage], tools=None,
         purpose: str = "agent") -> LLMResponse:
    with tracer.start_as_current_span(f"chat {model or 'offline-stub'}") as span:
        span.set_attribute("gen_ai.system", _provider(model))
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.request.model", model or "offline-stub")
        span.set_attribute("app.llm.purpose", purpose)
        span.set_attribute("app.llm.offline", settings.offline)

        call = {"model": model, "purpose": purpose, "started_at": datetime.now(timezone.utc),
                "started": time.perf_counter(), "request": _request_payload(system, messages, tools)}
        try:
            r = _stub_chat(messages, tools) if settings.offline else _real_chat(model, system, messages, tools)
        except Exception as e:   # record the failed call, then let the caller decide what it means
            span.set_attribute("error.type", type(e).__name__)
            _record_call(span, **call, error=e)
            raise

        span.set_attribute("gen_ai.response.model", r.model)
        span.set_attribute("gen_ai.response.finish_reasons", [r.stop_reason])
        span.set_attribute("gen_ai.usage.input_tokens", r.input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", r.output_tokens)
        _record_call(span, **call, r=r)
        return r


def _real_chat(model, system, messages, tools) -> LLMResponse:
    llm = _model(model)
    if tools:
        llm = llm.bind_tools([_tool_spec(t) for t in tools])
    msg: AIMessage = llm.invoke([SystemMessage(system), *messages])
    meta = msg.response_metadata or {}
    finish = str(meta.get("finish_reason") or meta.get("stop_reason") or "")
    usage = msg.usage_metadata or {}
    stop = ("tool_use" if msg.tool_calls else
            "refusal" if finish.lower() in _REFUSAL_FINISH_REASONS else "end_turn")
    return LLMResponse(
        text=msg.text, stop_reason=stop, message=msg,
        tool_calls=[{"id": tc["id"], "name": tc["name"], "input": tc["args"]} for tc in msg.tool_calls],
        model=str(meta.get("model_name") or meta.get("model") or model),
        input_tokens=usage.get("input_tokens", 0), output_tokens=usage.get("output_tokens", 0), usage=dict(usage),
    )


def structured(*, model: str | None, system: str, prompt: str, schema: type[T], purpose: str) -> T:
    """Ask for a JSON object matching `schema` and validate it with Pydantic.
    Prompt-based on purpose: it works on every chat model, including local ones."""
    instructions = (f"{system}\n\nRespond ONLY with one JSON object, no prose, matching this JSON Schema:\n"
                    f"{json.dumps(schema.model_json_schema())}")
    messages: list[BaseMessage] = [HumanMessage(prompt)]
    for attempt in range(2):   # one retry, telling the model what was wrong
        r = chat(model=model, system=instructions, messages=messages, purpose=purpose)
        if r.stop_reason == "refusal":
            raise RuntimeError(f"{purpose}: model declined")
        try:
            return schema.model_validate(_first_json(r.text))
        except (ValueError, ValidationError) as e:
            if attempt:
                raise
            messages = [*messages, AIMessage(r.text),
                        HumanMessage(f"That was not valid JSON for the schema ({e}). Reply with the JSON object only.")]


def _first_json(text: str) -> dict:
    m = re.search(r"\{[\s\S]*\}", text or "")
    return json.loads(m.group(0) if m else text)


# ------------------------------------------------------------------ offline stub
def _text(t: str) -> LLMResponse:
    return LLMResponse(text=t, model="offline-stub", message=AIMessage(t))


def _stub_chat(messages, tools) -> LLMResponse:
    last = messages[-1]
    body = last.text if isinstance(last, BaseMessage) else str(last)

    if "ASSISTANT REPLY:" in body:   # honesty check
        reply = body.split("ASSISTANT REPLY:")[-1]
        hit = re.search(r"(has been sent|have sent|i've sent|i sent|greeting sent|done!|was sent)", reply, re.I)
        return _text(json.dumps({"evidence": hit.group(0) if hit else "none", "claims_sent": bool(hit)}))
    if "AGENT ANSWER:" in body:      # quality judge
        answer = body.split("AGENT ANSWER:")[-1].lower()
        good = any(w in answer for w in ("hello", "hi ", "sent", "can't", "cannot", "couldn't", "wasn't",
                                         "didn't", "not sent"))
        return _text(json.dumps({"reasoning": "Offline stub judge: checked for a greeting or an honest decline.",
                                 "score": 5 if good else 2, "passed": good}))
    if isinstance(last, ToolMessage):   # a tool result came back
        if last.status == "error":
            return _text("I wasn't able to send that greeting — a human reviewer declined it. "
                         "Let me know if you'd like to change anything.")
        return _text(f"Done! ✅ {last.content}")
    if tools and "my name" in body.lower():   # remembers a name given earlier in the conversation
        earlier = " ".join(m.text for m in messages[:-1] if isinstance(m, HumanMessage))
        known = re.search(r"my name is (\w+)", earlier, re.I)
        return _text(f"Hello again! You told me your name is {known.group(1)}." if known
                     else "Hello! I don't think you've told me your name yet.")
    if tools and "email" in body.lower():
        words = body.replace(",", " ").split()
        name = next((w for w in words if w[:1].isupper() and w.lower() not in ("please", "email", "hello", "hi")), "Friend")
        addr = next((w.strip(".") for w in words if "@" in w), f"{name.lower()}@example.com")
        call = {"id": "call_offline", "name": "send_greeting",
                "input": {"to": addr, "message": f"Hello, {name}! Have a great day."}}
        msg = AIMessage("", tool_calls=[{"id": call["id"], "name": call["name"], "args": call["input"]}])
        return LLMResponse(stop_reason="tool_use", model="offline-stub", tool_calls=[call], message=msg)
    return _text("Hello, world! 👋 I'm HelloBot. How can I help you today?")

