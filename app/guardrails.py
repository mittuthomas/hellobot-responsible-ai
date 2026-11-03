"""Input & output guardrails: deterministic policy checks that run before and after the model.
Every decision carries a policy id, so the audit trail explains why something was blocked or redacted.
The PIIDetector interface lets you swap the regex detector for Microsoft Presidio or a cloud DLP API."""
import re
from typing import Protocol

from pydantic import BaseModel

from app.config import settings
from app.telemetry import tracer


class Violation(BaseModel):
    policy: str
    action: str          # block | redact | flag
    detail: str = ""


class GuardResult(BaseModel):
    allowed: bool
    text: str
    violations: list[Violation] = []


class PIIDetector(Protocol):
    def redact(self, text: str, skip: set[str]) -> tuple[str, list[str]]: ...


class RegexPIIDetector:
    PATTERNS = {
        "EMAIL": r"[\w.+-]+@[\w-]+\.[\w.]+",
        "US_SSN": r"\b\d{3}-\d{2}-\d{4}\b",
        "CREDIT_CARD": r"\b(?:\d[ -]?){13,16}\b",
        "IN_AADHAAR": r"\b\d{4}\s\d{4}\s\d{4}\b",
        "IN_PAN": r"\b[A-Z]{5}\d{4}[A-Z]\b",
        "PHONE": r"(?<!\d)(?:\+?\d{1,3}[ -]?)?\d{10}(?!\d)",
    }

    def redact(self, text, skip):
        found = []
        for name, p in self.PATTERNS.items():
            if name not in skip and re.search(p, text):
                text = re.sub(p, f"<{name}>", text)
                found.append(name)
        return text, found


pii = RegexPIIDetector()

INJECTION = [
    r"ignore (all |any )?(previous|prior|above) (instructions|rules)",
    r"(reveal|print|show) (me )?(your|the) (system )?(prompt|instructions)",
    r"you are now (dan|in developer mode|jailbroken)",
    r"disregard (your|all) (guidelines|rules)",
]
DENIED_TOPICS = {"weapons": r"\b(build|make) (a )?(bomb|weapon|explosive)\b",
                 "malware": r"\b(write|create) (a )?(virus|ransomware|keylogger)\b"}
LEAK = r"(my|the) system prompt|my instructions (are|say)"


def check_input(text: str) -> GuardResult:
    with tracer.start_as_current_span("guardrail.input") as span:
        v: list[Violation] = []
        if not text.strip():
            v.append(Violation(policy="non_empty", action="block"))
        if len(text) > settings.max_input_chars:
            v.append(Violation(policy="max_length", action="block", detail=f"{len(text)} chars"))
        for p in INJECTION:
            if re.search(p, text, re.I):
                v.append(Violation(policy="prompt_injection", action="block", detail=p))
        for topic, p in DENIED_TOPICS.items():
            if re.search(p, text, re.I):
                v.append(Violation(policy="denied_topic", action="block", detail=topic))
        # Emails are allowed through: the send_greeting tool needs them
        clean, found = pii.redact(text, skip={"EMAIL"})
        v += [Violation(policy="pii", action="redact", detail=f) for f in found]
        allowed = not any(x.action == "block" for x in v)
        span.set_attribute("guardrail.allowed", allowed)
        span.set_attribute("guardrail.violations", [x.policy for x in v])
        return GuardResult(allowed=allowed, text=clean, violations=v)


def check_output(text: str) -> GuardResult:
    with tracer.start_as_current_span("guardrail.output") as span:
        v: list[Violation] = []
        clean, found = pii.redact(text, skip={"EMAIL"})
        v += [Violation(policy="pii", action="redact", detail=f) for f in found]
        if re.search(LEAK, clean, re.I):
            v.append(Violation(policy="prompt_leak", action="block"))
            clean = "Sorry, I can't share that."
        if not clean.strip():
            v.append(Violation(policy="non_empty", action="block"))
            clean = "Sorry, I couldn't produce an answer."
        allowed = not any(x.action == "block" for x in v)
        span.set_attribute("guardrail.allowed", allowed)
        return GuardResult(allowed=allowed, text=clean, violations=v)
