from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Data Bus API"
    app_host: str = "localhost"
    app_port: int = 8001
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/webhook_db"
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    project_root: Path = Path(__file__).parent.parent.resolve()
    model_config = SettingsConfigDict(env_file=".env")


