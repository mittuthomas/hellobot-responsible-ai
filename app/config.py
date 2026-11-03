"""Typed, env-driven configuration (12-factor). Override any field with an env var or .env."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str | None = None
    agent_model: str = "claude-opus-5-5"
    judge_model: str = "claude-opus-5-5"
    effort: str = "low"
    force_offline: bool = False            # demo safety net: run on the stub LLM

    database_url: str = "sqlite:///./hello_agent.db"
    otel_exporter_otlp_endpoint: str | None = None   # e.g. http://localhost:4318 (Jaeger)
    service_name: str = "hello-agent"

    max_input_chars: int = 2000
    max_agent_steps: int = 4
    eval_threshold: float = 0.8

    @property
    def offline(self) -> bool:
        return self.force_offline or not self.anthropic_api_key


settings = Settings()
