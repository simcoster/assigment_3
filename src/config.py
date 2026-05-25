"""Application configuration loaded from environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime settings for the Bitext data analyst agent."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    nebius_api_key: str = ""
    nebius_base_url: str = "https://api.tokenfactory.nebius.com/v1/"
    router_model: str = "Qwen/Qwen3-32B"
    agent_model: str = "meta-llama/Llama-3.3-70B-Instruct"
    max_iterations: int = 12
    dataset_path: Path = PROJECT_ROOT / "data" / "bitext.parquet"
    checkpoint_path: Path = PROJECT_ROOT / "data" / "checkpoints.sqlite"
    profile_dir: Path = PROJECT_ROOT / "data" / "profiles"

    @property
    def dataset_hf_id(self) -> str:
        return "bitext/Bitext-customer-support-llm-chatbot-training-dataset"


def get_settings() -> Settings:
    """Return application settings, loading from `.env` when present."""
    return Settings()
