"""Typed, env-driven configuration (12-factor). Override any field with an env var or .env."""
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()   # puts provider API keys from .env (OPENAI_API_KEY, MISTRAL_API_KEY, ...) into the environment


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # "provider:model", e.g. openai:gpt-4.1, mistralai:mistral-large-latest, anthropic:<model>,
    # google_genai:gemini-3.6-flash, ollama:llama3.1
    agent_model: str | None = None
    judge_model: str | None = None        # defaults to agent_model; a stronger model is better
    max_output_tokens: int = 4000
    force_offline: bool = False           # demo safety net: run on the stub LLM

    database_url: str = "sqlite:///./hello_agent.db"
    checkpoint_db: str = "./checkpoints.db"           # LangGraph checkpointer (paused runs)
    otel_exporter_otlp_endpoint: str | None = None   # e.g. http://localhost:4318 (Jaeger)
    service_name: str = "hello-agent"
    otel_capture_llm_content: bool = True            # put full LLM requests/responses on trace spans

    max_input_chars: int = 2000
    max_agent_steps: int = 4
    eval_threshold: float = 0.8

    @property
    def offline(self) -> bool:
        return self.force_offline or not self.agent_model

    @property
    def judge(self) -> str | None:
        return self.judge_model or self.agent_model


settings = Settings()
