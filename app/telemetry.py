"""OpenTelemetry tracing using the GenAI semantic conventions (gen_ai.*).
Spans export over OTLP to Jaeger / Grafana Tempo / Datadog / any OTel backend."""
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from app.config import settings

_provider = TracerProvider(resource=Resource.create({"service.name": settings.service_name}))
if settings.otel_exporter_otlp_endpoint:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    _provider.add_span_processor(BatchSpanProcessor(
        OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint.rstrip("/") + "/v1/traces")))
trace.set_tracer_provider(_provider)

tracer = trace.get_tracer("hello_agent")


def current_trace_id() -> str:
    ctx = trace.get_current_span().get_span_context()
    return format(ctx.trace_id, "032x") if ctx.trace_id else ""
