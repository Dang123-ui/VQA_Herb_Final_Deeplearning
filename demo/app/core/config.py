from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Vietnamese Medicinal Herb VQA"
    model_name: str = "Qwen2.5-VL-3B-Instruct + QLoRA"
    gradio_api_url: str = ""
    gradio_api_name: str = "/predict"
    request_timeout_seconds: int = Field(default=180, ge=10)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()
