from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "saloniq"
    POSTGRES_USER: str = "saloniq"
    POSTGRES_PASSWORD: str = "saloniq"

    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379

    MODEL_PATH: str = "/app/models"

    LOG_LEVEL: str = "INFO"

    class Config:
        env_prefix = ""
        case_sensitive = True


settings = Settings()
