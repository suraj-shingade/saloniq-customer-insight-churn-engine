from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Environment-driven configuration for the feature-store service."""

    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "saloniq"
    POSTGRES_USER: str = "saloniq"
    POSTGRES_PASSWORD: str = "changeme"

    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379

    LOG_LEVEL: str = "INFO"

    model_config = {"env_prefix": ""}


settings = Settings()
