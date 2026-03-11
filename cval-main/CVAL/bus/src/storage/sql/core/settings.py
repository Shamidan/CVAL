from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Data Bus API"
    app_host: str = "localhost"
    app_port: int = 8001

    database_url: str = "sqlite+aiosqlite:///./webhook_db.sqlite"
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    class Config:
        env_file = Path(__file__).parent.parent.parent.parent.parent / ".env"
        extra = 'ignore'
