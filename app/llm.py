"""LLM gateway: every model call goes through here, so tracing, token accounting and
refusal handling live in one place, whatever the provider.

Models are chosen by a "provider:model" string, for example
    openai:gpt-4.1   mistralai:mistral-large-latest   anthropic:<model>   ollama:llama3.1
and created with LangChain's standard chat-model interface (init_chat_model), so the
same code drives any provider that supports tool calling. Each provider reads its own
API key from the environment (OPENAI_API_KEY, MISTRAL_API_KEY, ...).

With no model configured (or FORCE_OFFLINE=true) a deterministic stub answers instead,
so demos and CI never depend on network access.
"""
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, TypeVar

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, ValidationError

from app.config import settings
from app.telemetry import tracer

T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)   # [{id, name, input}]
    stop_reason: str = "end_turn"                          # end_turn | tool_use | refusal
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    message: Any = None   # the provider's AIMessage, appended to the conversation as-is


@lru_cache(maxsize=8)
def _model(name: str):
    from langchain.chat_models import init_chat_model
    return init_chat_model(name, max_tokens=settings.max_output_tokens)


def _provider(name: str | None) -> str:
    return (name or "offline").split(":", 1)[0]


def _tool_spec(t: dict) -> dict:
    """Our neutral tool definition -> the function-calling format every provider accepts."""
    return {"type": "function",
            "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}


def chat(*, model: str | None, system: str, messages: list[BaseMessage], tools=None,
         purpose: str = "agent") -> LLMResponse:
    with tracer.start_as_current_span(f"chat {model or 'offline-stub'}") as span:
        span.set_attribute("gen_ai.system", _provider(model))
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.request.model", model or "offline-stub")
        span.set_attribute("app.llm.purpose", purpose)
        span.set_attribute("app.llm.offline", settings.offline)

        r = _stub_chat(messages, tools) if settings.offline else _real_chat(model, system, messages, tools)

        span.set_attribute("gen_ai.response.model", r.model)
        span.set_attribute("gen_ai.response.finish_reasons", [r.stop_reason])
        span.set_attribute("gen_ai.usage.input_tokens", r.input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", r.output_tokens)
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
            "refusal" if finish in ("refusal", "content_filter") else "end_turn")
    return LLMResponse(
        text=msg.text, stop_reason=stop, message=msg,
        tool_calls=[{"id": tc["id"], "name": tc["name"], "input": tc["args"]} for tc in msg.tool_calls],
        model=str(meta.get("model_name") or meta.get("model") or model),
        input_tokens=usage.get("input_tokens", 0), output_tokens=usage.get("output_tokens", 0),
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
    if tools and "email" in body.lower():
        words = body.replace(",", " ").split()
        name = next((w for w in words if w[:1].isupper() and w.lower() not in ("please", "email", "hello", "hi")), "Friend")
        addr = next((w.strip(".") for w in words if "@" in w), f"{name.lower()}@example.com")
        call = {"id": "call_offline", "name": "send_greeting",
                "input": {"to": addr, "message": f"Hello, {name}! Have a great day."}}
        msg = AIMessage("", tool_calls=[{"id": call["id"], "name": call["name"], "args": call["input"]}])
        return LLMResponse(stop_reason="tool_use", model="offline-stub", tool_calls=[call], message=msg)
    return _text("Hello, world! 👋 I'm HelloBot. How can I help you today?")

